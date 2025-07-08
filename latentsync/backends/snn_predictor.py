import yaml
import os
import json
import numpy as np
import cv2
import time
import torch
import tempfile
import subprocess
import soundfile as sf
import tqdm
import pickle
import gzip

from typing import Dict, List, Union, Optional, Callable
from oneservice import MonitorTimer, load_config
from loguru import logger
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import BaseOutput
from dataclasses import dataclass

from .snn_torch_backend import SNNTorchBackend
from .snn_fast_backend import SNNFastBackend
from .snn_video_backend import SNNVideoBackend
from .snn_preprocess import SNNPreprocessor
from .snn_postprocess import SNNPostprocessor
from latentsync import NVTXContext, PROFILER_AVAILABLE
from latentsync.src.utils.status_code import PI_STATUS_CODE

# 配置torch.compile缓存设置
torch._dynamo.config.cache_size_limit = 64
torch.set_float32_matmul_precision('high')

# UNet3DConditionOutput已移至SNNTorchBackend中定义

class SNNPredictor(object):
    def __init__(self, config_file: str):
        config = yaml.safe_load(open(config_file).read())["predictors"]
        dir_path = os.path.dirname(os.path.realpath(config_file))

        # 1. monitor init process time
        with NVTXContext("SNNPredictor.init"):
            with MonitorTimer("init"):
                if config["predictor"] == "snn_torch_predictor":
                    self._predictor_type = "snn_torch_predictor"
                    config = load_config(
                        config["snn_torch_predictor"], "snn_torch_predictor"
                    )

                    config.model_path = os.path.join(dir_path, config.model_path)

                    # 检查是否有视频生成配置
                    video_config_path = os.path.join(dir_path, "..", "models", "V1.0_latentsync_video_predictor.json")
                    video_config_path = os.path.abspath(video_config_path)
                    self._has_video_config = os.path.exists(video_config_path)
                    logger.info(f"Video config path: {video_config_path}")
                    logger.info(f"Video config exists: {self._has_video_config}")
                    if self._has_video_config:
                        logger.info("Found video config, video generation capability will be available")
                    
                    # 根据配置文件中的backend字段选择后端
                    backend_type = getattr(config, 'backend', 'TorchBackend')
                    backend_str = str(backend_type).split('.')[-1] if hasattr(backend_type, '__class__') else str(backend_type)
                    
                    if 'TorchBackend' in backend_str:
                        self._engine = SNNTorchBackend(config)
                    elif 'TensorRTBackend' in backend_str:
                        self._engine = SNNFastBackend(config)
                    else:
                        raise ValueError(f"Unsupported backend type: {backend_type}")
                    
                    # 初始化前处理器
                    self._preprocessor = SNNPreprocessor()
                    
                    # 初始化后处理器
                    upload_config, debug_enabled, debug_output_path = self._load_postprocess_config(config)
                    self._postprocessor = SNNPostprocessor(
                        upload_config=upload_config,
                        debug_enabled=debug_enabled,
                        debug_output_path=debug_output_path
                    )

                    self._max_batch_size = config.max_batch_size

                elif config["predictor"] == "snn_fast_predictor":
                    # NOTE: for AI TryOn TensorRT inference
                    logger.info("Run on snn_fast_predictor")
                    self._predictor_type = "snn_fast_predictor"
                    config = load_config(config["snn_fast_predictor"], "snn_fast_predictor")
                    config.model_path = os.path.join(dir_path, config.model_path)
                    self._engine = SNNFastBackend(config)
                    
                    # 初始化前处理器
                    self._preprocessor = SNNPreprocessor()
                    
                    # 初始化后处理器
                    upload_config, debug_enabled, debug_output_path = self._load_postprocess_config(config)
                    self._postprocessor = SNNPostprocessor(
                        upload_config=upload_config,
                        debug_enabled=debug_enabled,
                        debug_output_path=debug_output_path
                    )
                    
                    self._max_batch_size = config.max_batch_size
                else:
                    raise NotImplementedError  # TODO
    
    # 移除重复的视频模型初始化方法，统一使用SNNTorchBackend
    
    # DiyCache相关方法已移至SNNTorchBackend中实现

    def _load_postprocess_config(self, config):
        """
        加载后处理配置
        
        Args:
            config: 模型配置
            
        Returns:
            tuple: (upload_config, debug_enabled, debug_output_path)
        """
        upload_config = None
        debug_enabled = False
        debug_output_path = './output/'
        
        try:
            # 加载上传配置
            model_config_path = config.model_path + "/V2.6_hunyuan-brushnet_online.json"
            if os.path.exists(model_config_path):
                e2e_config = json.load(open(model_config_path))
                upload_config = e2e_config
            else:
                logger.warning(f"Config file not found: {model_config_path}")
            
            # 加载debug配置
            debug_enabled = getattr(config, 'save_debug_images', False)
            if isinstance(debug_enabled, str):
                debug_enabled = debug_enabled.lower() in ('true', '1', 'yes', 'on')
            
            debug_output_path = getattr(config, 'debug_output_path', './output/')
            
        except Exception as e:
            logger.warning(f"Failed to load postprocess config: {e}")
        
        return upload_config, debug_enabled, debug_output_path

    def predict(
        self,
        aigc_segmentation_image_id: Union[List[str], List[np.ndarray]],
        generated_config: List[Dict],
    ):
        """
        预测方法，支持三种输入数据类型，返回格式与输入格式保持一致
        
        Args:
            aigc_segmentation_image_id: 图片数据，支持三种格式：
                - List[str] (image_id): 图片ID列表，需要下载，返回image_id
                - List[str] (base64): base64编码图片列表，需要解码，返回base64  
                - List[np.ndarray]: 图片二进制数据列表，直接使用，返回ndarray
            generated_config: 生成配置列表
            
        Returns:
            tuple: 根据输入类型返回不同格式的结果
            - 输入为 image_id 时：(generated_image_ids: List[str], mask_image_ids: List[str], status_codes, has_nsfw)
            - 输入为 base64 时：(generated_base64s: List[str], mask_base64s: List[str], status_codes, has_nsfw)
            - 输入为 ndarray 时：(generated_image_arrays: List[np.ndarray], mask_image_arrays: List[np.ndarray], status_codes, has_nsfw)
        """
        with NVTXContext("SNNPredictor.predict"):
            # 检测输入数据类型和格式
            input_type, need_preprocessing = self._detect_input_type(aigc_segmentation_image_id)
            
            # 2. forword
            with MonitorTimer("predict"):
                # 注意：虽然变量名是generated_image_ids和mask_image_ids，
                # 但根据输入类型，实际存储的可能是image_ids(str)、base64(str)或image_arrays(np.ndarray)
                generated_image_ids = []
                mask_image_ids = []
                status_codes = []
                has_nsfw = []
                
                if (
                    self._predictor_type == "snn_torch_predictor"
                    or self._predictor_type == "snn_fast_predictor"
                ):
                    num_images = len(aigc_segmentation_image_id)
                    
                    # 现在使用真正的batch处理，无需手动分批
                    logger.info(f"Processing {num_images} images with true batch processing")
                    
                    # 根据输入类型处理整个批次
                    if need_preprocessing:
                        batch_results = self._process_batch_from_strings(aigc_segmentation_image_id, generated_config, input_type)
                    else:
                        batch_results = self._process_batch_from_arrays(aigc_segmentation_image_id, generated_config)
                    
                    # 收集结果
                    batch_gen_ids, batch_mask_ids, batch_status, batch_nsfw = batch_results
                    generated_image_ids.extend(batch_gen_ids)
                    mask_image_ids.extend(batch_mask_ids)
                    status_codes.extend(batch_status)
                    has_nsfw.extend(batch_nsfw)
                    
                    logger.info(f"Batch processing completed for {num_images} images")
                else:
                    raise NotImplementedError  # TODO

        return generated_image_ids, mask_image_ids, status_codes, has_nsfw

    def predict_video(
        self,
        audio_paths: List[str],
        output_paths: List[str],
        mask_output_paths: Optional[List[str]] = None,
    ):
        """
        视频生成预测方法
        
        Args:
            audio_paths: 音频文件路径列表
            output_paths: 输出视频路径列表
            mask_output_paths: 输出mask视频路径列表（可选）
            
        Returns:
            tuple: (video_paths: List[str], mask_paths: List[str], status_codes, error_messages: List[str])
        """
        with NVTXContext("SNNPredictor.predict_video"):
            if not hasattr(self, '_has_video_config') or not self._has_video_config:
                raise ValueError(f"Video prediction not supported for this predictor type: {self._predictor_type}")
            
            if not hasattr(self._engine, 'pipeline'):
                raise ValueError(f"Video prediction not supported for this backend type")
            
            with MonitorTimer("predict_video"):
                # 使用engine的forward方法进行视频生成
                results = self._engine.forward(audio_paths, output_paths, mask_output_paths)
                
                # 转换结果格式
                video_paths = []
                mask_paths = []
                status_codes = []
                error_messages = []
                
                for result in results:
                    video_paths.append(result.get('video_path'))
                    mask_paths.append(result.get('mask_path'))
                    status_codes.append(result.get('status_code', 'GENERATION_ERROR'))
                    error_messages.append(result.get('error_message'))
                
                return video_paths, mask_paths, status_codes, error_messages
    
    def _detect_input_type(self, input_data: Union[List[str], List[np.ndarray]]) -> tuple:
        """
        检测输入数据类型和格式
        
        Args:
            input_data: 输入数据
            
        Returns:
            tuple: (input_type: str, need_preprocessing: bool)
            - input_type: 'image_id', 'base64', 或 'ndarray'
            - need_preprocessing: 是否需要预处理
        """
        if not input_data:
            raise ValueError("Input data is empty")
        
        first_item = input_data[0]
        
        if isinstance(first_item, str):
            # 检测字符串是base64还是image_id
            if self._preprocessor._is_base64_data(first_item):
                logger.info(f"Detected input type: base64 list (need preprocessing), length: {len(input_data)}")
                return 'base64', True
            else:
                logger.info(f"Detected input type: image_id list (need preprocessing), length: {len(input_data)}")
                return 'image_id', True
        elif isinstance(first_item, np.ndarray):
            logger.info(f"Detected input type: numpy array list (ready to use), length: {len(input_data)}")
            return 'ndarray', False
        else:
            raise ValueError(f"Unsupported input data type: {type(first_item)}")
    
    def _process_batch_from_strings(self, string_data: List[str], configs: List[Dict], input_type: str) -> tuple:
        """
        从字符串数据处理一个批次的图片（可能是image_id或base64数据）
        直接进行batch推理，无需配置分组
        
        Args:
            string_data: 字符串数据列表（image_id或base64编码的图片）
            configs: 配置列表
            input_type: 输入类型 ('image_id' 或 'base64')
            
        Returns:
            tuple: 根据input_type返回相应格式的结果
        """
        with NVTXContext("SNNPredictor.process_batch_from_strings"):
            total_count = len(string_data)
            
            # 1. 预处理：让snn_preprocess处理所有字符串数据，统一转换为ndarray
            images, masks, valid_data_ids, preprocess_status_codes = self._preprocessor.batch_process_input_data(string_data)
            
            # 准备结果列表，初始化为失败状态
            generated_results = [None] * total_count
            mask_results = [None] * total_count
            status_codes = preprocess_status_codes.copy()
            has_nsfw = [False] * total_count
            
            # 2. 如果有成功处理的图片，直接进行batch推理（无需分组）
            if images:
                # 构建有效图片的索引映射
                valid_indices = []
                for j, status in enumerate(preprocess_status_codes):
                    if status == PI_STATUS_CODE.SUCCESS:
                        valid_indices.append(j)
                
                # 获取成功处理图片对应的配置
                valid_configs = [configs[i] for i in valid_indices]
                
                logger.info(f"Processing {len(images)} valid images in one batch (no config grouping needed)")
                
                # 直接进行batch推理
                batch_start = time.time()
                batch_gen_results = self._engine.forward(
                    images=images,
                    masks=masks,
                    generated_config=valid_configs,
                )
                batch_end = time.time()
                
                logger.info(f"Batch inference time: {(batch_end - batch_start)*1000:.2f} ms for {len(images)} images")
                
                # 后处理结果
                batch_processed_results = self._postprocessor.process_batch_results(
                    batch_gen_results, valid_data_ids, input_type
                )
                
                # 将结果映射回原始位置
                for j, (result, gen_result) in enumerate(zip(batch_processed_results, batch_gen_results)):
                    original_idx = valid_indices[j]
                    
                    if input_type == 'image_id':
                        generated_results[original_idx] = result.get('aigc_generated_image_id')
                        mask_results[original_idx] = result.get('aigc_generated_mask_id')
                    elif input_type == 'base64':
                        generated_results[original_idx] = result.get('aigc_generated_base64')
                        mask_results[original_idx] = result.get('aigc_generated_mask_base64')
                    
                    status_codes[original_idx] = result.get('status_code', PI_STATUS_CODE.SUCCESS)
                    has_nsfw[original_idx] = gen_result.get('has_nsfw_concept', False)
                
                # 统计总体结果
                total_nsfw = sum(has_nsfw)
                total_success = sum(1 for code in status_codes if code == PI_STATUS_CODE.SUCCESS)
                logger.info(f"Batch processing completed: {total_success}/{total_count} successful, {total_nsfw} NSFW detected")
            
            return generated_results, mask_results, status_codes, has_nsfw
    
    def _process_batch_from_arrays(self, image_arrays: List[np.ndarray], configs: List[Dict]) -> tuple:
        """
        从numpy数组处理一个批次的图片（直接使用现成的数组）
        直接进行batch推理，无需配置分组
        
        Args:
            image_arrays: 图片数组列表，每个数组应该是BGRA格式
            configs: 配置列表
            
        Returns:
            tuple: (generated_image_arrays: List[np.ndarray], mask_image_arrays: List[np.ndarray], status_codes, has_nsfw)
            注意：与_process_batch_from_strings不同，这里返回的是numpy arrays而不是image IDs
        """
        with NVTXContext("SNNPredictor.process_batch_from_arrays"):
            total_count = len(image_arrays)
            
            # 1. 预处理：验证和转换BGRA格式的数组为RGB图片和mask
            images = []
            masks = []
            valid_indices = []
            status_codes = [PI_STATUS_CODE.DOWNLOAD_FG_ERROR] * total_count  # 初始化为失败状态
            
            for i, image_array in enumerate(image_arrays):
                try:
                    # 验证输入格式
                    if self._validate_input_array(image_array):
                        # 转换格式：BGRA -> RGB和mask，复用预处理器的方法
                        image, mask = self._preprocessor._process_image_format(image_array)
                        images.append(image)
                        masks.append(mask)
                        valid_indices.append(i)
                        status_codes[i] = PI_STATUS_CODE.SUCCESS
                    else:
                        logger.warning(f"Invalid input array format at index {i}")
                except Exception as e:
                    logger.error(f"Error processing array at index {i}: {e}")
            
            # 准备结果列表
            generated_image_arrays = [None] * total_count
            mask_image_arrays = [None] * total_count
            has_nsfw = [False] * total_count
            
            # 2. 如果有有效的图片，直接进行batch推理（无需分组）
            if images:
                # 获取有效图片对应的配置
                valid_configs = [configs[i] for i in valid_indices]
                
                logger.info(f"Processing {len(images)} valid images in one batch (no config grouping needed)")
                
                # 生成虚拟的图片ID用于后处理
                valid_virtual_ids = [f"array_{i}" for i in valid_indices]
                
                # 直接进行batch推理
                batch_start = time.time() 
                batch_gen_results = self._engine.forward(
                    images=images,
                    masks=masks,
                    generated_config=valid_configs,
                )
                batch_end = time.time()
                
                logger.info(f"Batch inference time: {(batch_end - batch_start)*1000:.2f} ms for {len(images)} images")
                
                # 后处理结果
                batch_processed_results = self._postprocessor.process_batch_results(
                    batch_gen_results, valid_virtual_ids, 'ndarray'
                )
                
                # 将结果映射回原始位置
                for j, (result, gen_result) in enumerate(zip(batch_processed_results, batch_gen_results)):
                    original_idx = valid_indices[j]
                    generated_image_arrays[original_idx] = result.get('aigc_generated_image_array')
                    mask_image_arrays[original_idx] = result.get('aigc_generated_mask_array')
                    status_codes[original_idx] = result.get('status_code', PI_STATUS_CODE.SUCCESS)
                    has_nsfw[original_idx] = gen_result.get('has_nsfw_concept', False)
                
                # 统计总体结果
                total_nsfw = sum(has_nsfw)
                total_success = sum(1 for code in status_codes if code == PI_STATUS_CODE.SUCCESS)
                logger.info(f"Batch processing completed: {total_success}/{total_count} successful, {total_nsfw} NSFW detected")
            
            return generated_image_arrays, mask_image_arrays, status_codes, has_nsfw
    
    def _validate_input_array(self, image_array: np.ndarray) -> bool:
        """
        验证输入数组格式
        
        Args:
            image_array: 输入图片数组
            
        Returns:
            bool: 是否为有效格式
        """
        if image_array is None:
            return False
        
        if not isinstance(image_array, np.ndarray):
            return False
        
        if len(image_array.shape) != 3:
            return False
        
        # 应该是BGRA格式（4通道）
        if image_array.shape[2] != 4:
            logger.warning(f"Expected BGRA format (4 channels), got {image_array.shape[2]} channels")
            return False
        
        return True