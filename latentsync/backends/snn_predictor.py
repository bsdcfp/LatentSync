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
                    
                    # 视频生成不需要前后处理器
                    self._max_batch_size = config.max_batch_size

                elif config["predictor"] == "snn_fast_predictor":
                    # NOTE: for AI TryOn TensorRT inference
                    logger.info("Run on snn_fast_predictor")
                    self._predictor_type = "snn_fast_predictor"
                    config = load_config(config["snn_fast_predictor"], "snn_fast_predictor")
                    config.model_path = os.path.join(dir_path, config.model_path)
                    self._engine = SNNFastBackend(config)
                    
                    # AI TryOn 不需要前后处理器（已废弃）
                    self._max_batch_size = config.max_batch_size
                else:
                    raise NotImplementedError  # TODO
    
    # 移除重复的视频模型初始化方法，统一使用SNNTorchBackend
    
    # DiyCache相关方法已移至SNNTorchBackend中实现





    def _preprocess_video_data(self, audio_path: str) -> Dict:
        """
        视频数据预处理
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            dict: 预处理后的数据
        """
        # 获取pipeline和配置
        pipeline = self._engine.get_pipeline()
        video_config = self._engine.get_video_config()
        
        # 统一路径处理：区分模型库路径和代码路径
        code_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_base = self._engine.config.model_path
        def resolve_path(path):
            if not path:
                return path
            if os.path.isabs(path):
                return path
            # 以weights/开头的，拼到模型库
            if path.startswith("weights/"):
                return os.path.join(model_base, path)
            # 以latentsync/开头的，拼到代码根目录
            if path.startswith("latentsync/"):
                return os.path.join(code_base, path)
            # 以stabilityai/开头的（VAE模型），直接用
            if path.startswith("stabilityai/"):
                return path
            # 其他路径（如0612_online_templete_female_32fps_resize_720_1560/），拼到模型库
            return os.path.join(model_base, path)
        
        # 解析路径
        video_cache_dir = resolve_path(video_config.get("video_cache_dir"))
        mask_base_dir = resolve_path(video_config.get("mask_base_dir")) if video_config.get("mask_base_dir") else None
        
        debug = video_config.get("debug_pipeline", False)
        if debug:
            print(f"🔧 路径解析:")
            print(f"  - video_cache_dir: {video_config.get('video_cache_dir')} -> {video_cache_dir}")
            print(f"  - mask_base_dir: {video_config.get('mask_base_dir')} -> {mask_base_dir}")
        
        return pipeline.preprocess_data_0701(
            video_cache_dir=video_cache_dir,
            audio_path=audio_path,
            mask_base_dir=mask_base_dir,
            num_frames=16,
            video_fps=video_config.get("target_fps", 25),
            need_padding_to_16x=True,
            debug=debug,
        )

    def _inference_video_chunks(
        self,
        preprocess_data: Dict,
        num_frames: int = 16,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        weight_dtype: Optional[torch.dtype] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        height: int = 256,
        width: int = 256,
        debug: bool = False,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        视频chunk推理
        
        Args:
            preprocess_data: 预处理数据
            num_frames: 每个chunk的帧数
            num_inference_steps: 推理步数
            guidance_scale: 引导尺度
            weight_dtype: 权重数据类型
            eta: eta参数
            generator: 随机数生成器
            callback: 回调函数
            callback_steps: 回调步数
            height: 高度
            width: 宽度
            debug: 调试模式
            **kwargs: 其他参数
            
        Returns:
            List[torch.Tensor]: 推理结果列表
        """
        # 获取pipeline和配置
        pipeline = self._engine.get_pipeline()
        video_config = self._engine.get_video_config()
        
        # 如果没有指定weight_dtype，使用默认值
        if weight_dtype is None:
            weight_dtype = self._engine.get_dtype()
        
        # 如果没有指定debug，使用配置中的值
        if debug is None:
            debug = video_config.get("debug_pipeline", False)
        
        synced_video_frames = []
        faces_by_chunk = preprocess_data['faces_by_chunk']
        whisper_chunks_by_chunk = preprocess_data['whisper_chunks_by_chunk']
        
        for chunk_idx in range(preprocess_data['num_inferences']):
            chunk_start_time = time.time()
            
            decoded_latents = self._engine.model_inference_by_chunk(
                chunk_faces=faces_by_chunk[chunk_idx],
                chunk_whisper_features=whisper_chunks_by_chunk[chunk_idx],
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                weight_dtype=weight_dtype,
                eta=eta,
                generator=generator,
                callback=callback,
                callback_steps=callback_steps,
                height=height,
                width=width,
                debug=debug,
                **kwargs
            )
            
            synced_video_frames.append(decoded_latents)
            
            chunk_time = time.time() - chunk_start_time
            logger.info(f"Chunk {chunk_idx+1}/{preprocess_data['num_inferences']} completed in {chunk_time:.3f}s")
        
        return synced_video_frames

    def _postprocess_video_data(self, preprocess_data: Dict, synced_video_frames: List[torch.Tensor]) -> np.ndarray:
        """
        视频数据后处理
        
        Args:
            preprocess_data: 预处理数据
            synced_video_frames: 推理结果列表
            
        Returns:
            np.ndarray: 最终的视频帧数组
        """
        # 获取pipeline和配置
        pipeline = self._engine.get_pipeline()
        video_config = self._engine.get_video_config()
        
        return pipeline.postprocess_video(
            synced_video_frames=synced_video_frames,
            original_video_frames=preprocess_data['original_video_frames'],
            boxes=preprocess_data['boxes'],
            affine_matrices=preprocess_data['affine_matrices'],
            original_faces_len=preprocess_data['original_faces_len'],
            num_frames=16,
            debug=video_config.get("debug_pipeline", False),
        )

    def _save_video_output(self, preprocess_data: Dict, final_video_frames: np.ndarray, video_out_path: str, mask_output_path: Optional[str] = None):
        """
        保存视频输出
        
        Args:
            preprocess_data: 预处理数据
            final_video_frames: 最终视频帧数组
            video_out_path: 视频输出路径
            mask_output_path: mask视频输出路径（可选）
        """
        # 获取pipeline和配置
        pipeline = self._engine.get_pipeline()
        video_config = self._engine.get_video_config()
        
        return pipeline.save_final_output(
            final_video_frames=final_video_frames,
            audio_samples=preprocess_data['audio_samples'],
            video_out_path=video_out_path,
            mask_frames=preprocess_data['mask_frames'],
            mask_output_path=mask_output_path,
            video_fps=video_config.get("target_fps", 25),
            audio_sample_rate=16000,
            convert_audio_to_48k=True,
            debug=video_config.get("debug_pipeline", False),
        )
    
