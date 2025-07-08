import torch
import threading
from typing import Optional, Dict, List, Union, Callable
import numpy as np
import cv2
import json
import time
import os
from loguru import logger
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.src.models.unet import UNet3DConditionModel
from latentsync.src.pipelines.lipsync_pipeline_online_0701 import LipsyncPipeline_online_0701 as LipsyncPipeline
from latentsync.src.whisper.audio2feature import Audio2Feature
from latentsync.src.utils.video_sequence_merger import VideoIndexGenerator
from latentsync import NVTXContext, PROFILER_AVAILABLE

from oneservice import InputDictType, OutputDictType, BackendConfig, MonitorTimer
from oneservice.backends.base import BaseBackend

from latentsync.src.utils.status_code import PI_STATUS_CODE

# 配置torch.compile缓存设置
torch._dynamo.config.cache_size_limit = 64

class SNNTorchBackend(BaseBackend):
    """LatentSync视频生成后端"""

    def __init__(self, config: BackendConfig):
        super(SNNTorchBackend, self).__init__(config)
        self.config = config  # 保存config引用
        model_path = config.model_path

        self._use_fp16 = config.fp16
        self._device = config.device
        self._size = config.size

        # Load models
        config_file = getattr(config, 'config_file', None)
        if not config_file:
            raise ValueError("config_file is required for video generation")
        
        # 如果config_file是相对路径，拼到当前工作目录
        if not os.path.isabs(config_file):
            config_path = os.path.join(os.getcwd(), config_file)
        else:
            config_path = config_file
        
        logger.info(f"Loading config from: {config_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        e2e_config = json.load(open(config_path))
        
        video_config = e2e_config.get('video_model', {})
        
        # 从命令行参数覆盖配置（如果存在）
        if hasattr(config, 'target_fps'):
            video_config['target_fps'] = config.target_fps
        if hasattr(config, 'inference_steps'):
            video_config['inference_steps'] = config.inference_steps
        if hasattr(config, 'guidance_scale'):
            video_config['guidance_scale'] = config.guidance_scale
        if hasattr(config, 'seed'):
            video_config['seed'] = config.seed
        if hasattr(config, 'enable_diycache'):
            video_config['enable_diycache'] = config.enable_diycache
        if hasattr(config, 'first_step_offset'):
            video_config['first_step_offset'] = config.first_step_offset
        if hasattr(config, 'last_step_offset'):
            video_config['last_step_offset'] = config.last_step_offset
        if hasattr(config, 'debug_pipeline'):
            video_config['debug_pipeline'] = config.debug_pipeline
        if hasattr(config, 'debug_video_merger'):
            video_config['debug_video_merger'] = config.debug_video_merger
        
        # 初始化模型组件
        self._init_models(video_config)
        
        # 初始化pipeline
        self._init_pipeline(video_config)
        
        self._lock = threading.Lock()

    def _init_models(self, config: Dict):
        """初始化模型组件"""
        logger.info("Initializing LatentSync models...")
        
        # Check if the GPU supports float16
        is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        dtype = torch.float16 if is_fp16_supported else torch.float32
        
        # 统一路径处理：区分模型库路径和代码路径
        code_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_base = self.config.model_path
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
        logger.info(f"Using code base for latentsync paths: {code_base}")
        logger.info(f"Using model base for weights/data paths: {model_base}")
        
        # 加载UNet配置 - 处理相对路径
        unet_config_path = resolve_path(config.get("unet_config_path", "latentsync/configs/unet/second_stage.yaml"))
        logger.info(f"Loading UNet config from: {unet_config_path}")
        unet_config = OmegaConf.load(unet_config_path)
        
        # 初始化调度器 - 处理相对路径
        scheduler_config_path = resolve_path(config.get("scheduler_config_path", "latentsync/configs"))
        logger.info(f"Loading scheduler config from: {scheduler_config_path}")
        scheduler = DDIMScheduler.from_pretrained(scheduler_config_path)
        
        # 初始化音频编码器
        whisper_model_path = resolve_path(config.get("whisper_model_path", "weights/whisper/tiny.pt"))
        logger.info(f"Loading Whisper model from: {whisper_model_path}")
        audio_encoder = Audio2Feature(
            model_path=whisper_model_path, 
            device="cuda", 
            num_frames=16
        )
        
        # 初始化VAE
        vae_model_path = resolve_path(config.get("vae_model_path", "stabilityai/sd-vae-ft-mse"))
        logger.info(f"Loading VAE model from: {vae_model_path}")
        vae = AutoencoderKL.from_pretrained(
            vae_model_path, 
            torch_dtype=dtype
        )
        vae = vae.to("cuda")
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        
        # 初始化UNet
        unet_config_dict = OmegaConf.to_container(unet_config)
        
        # 确保sample_size被正确设置
        if 'model' in unet_config_dict:
            if unet_config_dict['model'].get('sample_size') is None:
                unet_config_dict['model']['sample_size'] = 64
                logger.info("Setting sample_size to 64 in UNet config")
            else:
                logger.info(f"UNet config sample_size: {unet_config_dict['model']['sample_size']}")
        
        # 只传递model部分的配置给UNet
        model_config = unet_config_dict.get('model', {})
        logger.info(f"Model config keys: {list(model_config.keys())}")
        
        # 处理UNet检查点路径
        inference_ckpt_path = resolve_path(config.get("inference_ckpt_path", "weights/latentsync/latentsync_unet.pt"))
        logger.info(f"Loading UNet checkpoint from: {inference_ckpt_path}")
        
        unet, _ = UNet3DConditionModel.from_pretrained(
            model_config,
            inference_ckpt_path,
            device="cpu",
        )
        unet = unet.to(dtype=dtype, device="cuda")
        
        # 初始化VideoIndexGenerator - 处理相对路径
        connected_json_path = resolve_path(config.get("connected_info_json"))
        video_info_json_path = resolve_path(config.get("video_info_json"))
        shot_video_json_path = resolve_path(config.get("shot_video_json_path"))
        video_template_dir = resolve_path(config.get("video_template_dir"))
        
        logger.info(f"VideoIndexGenerator paths:")
        logger.info(f"  - connected_json_path: {connected_json_path}")
        logger.info(f"  - video_info_json_path: {video_info_json_path}")
        logger.info(f"  - shot_video_json_path: {shot_video_json_path}")
        logger.info(f"  - video_template_dir: {video_template_dir}")
        
        video_index_generator = VideoIndexGenerator(
            connected_json_path=connected_json_path,
            video_info_json_path=video_info_json_path,
            shot_video_json_path=shot_video_json_path,
            video_template_dir=video_template_dir,
            target_fps=config.get("target_fps", 25),
            transition_frames=config.get("transition_frames", 20),
            debug=config.get("debug_video_merger", False)
        )
        
        # 保存组件
        self.scheduler = scheduler
        self.audio_encoder = audio_encoder
        self.vae = vae
        self.unet = unet
        self.video_index_generator = video_index_generator
        self.dtype = dtype
        
        # 保存配置
        self.video_config = config
        
        logger.info("LatentSync models initialized successfully")

    def _init_pipeline(self, config: Dict):
        """初始化pipeline"""
        logger.info("Initializing LatentSync pipeline...")
        
        # 创建pipeline
        self.pipeline = LipsyncPipeline(
            vae=self.vae,
            audio_encoder=self.audio_encoder,
            unet=self.unet,
            scheduler=self.scheduler,
            video_index_generator=self.video_index_generator,
        ).to("cuda")
        
        # 配置DiyCache
        if config.get("enable_diycache", False):
            logger.info("Enabling DiyCache...")
            self._setup_diycache_unet(
                first_step_offset=config.get("first_step_offset", 1),
                last_step_offset=config.get("last_step_offset", 4),
                num_steps=config.get("inference_steps", 20)
            )
        
        logger.info("LatentSync pipeline initialized successfully")

    def _setup_diycache_unet(self, first_step_offset=1, last_step_offset=4, num_steps=20):
        """设置DiyCache"""
        # 这里可以添加DiyCache的设置逻辑
        # 由于代码较长，这里先跳过具体实现
        logger.info(f"DiyCache setup: first_step_offset={first_step_offset}, last_step_offset={last_step_offset}, num_steps={num_steps}")

    @NVTXContext
    def preprocess_data(
        self,
        video_cache_dir: str,
        audio_path: str,
        mask_base_dir: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        need_padding_to_16x: bool = True,
        debug: bool = False,
        **kwargs,
    ):
        """
        前处理：整合四个子任务（取视频特征、音频特征、做padding、数据分chunk）
        
        Returns:
            dict: 包含所有预处理后的数据，包括分chunk后的数据
        """
        # 统一路径处理：区分模型库路径和代码路径
        code_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_base = self.config.model_path
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
        resolved_video_cache_dir = resolve_path(video_cache_dir)
        resolved_mask_base_dir = resolve_path(mask_base_dir) if mask_base_dir else None
        
        if debug:
            print(f"🔧 路径解析:")
            print(f"  - video_cache_dir: {video_cache_dir} -> {resolved_video_cache_dir}")
            print(f"  - mask_base_dir: {mask_base_dir} -> {resolved_mask_base_dir}")
        
        return self.pipeline.preprocess_data_0701(
            video_cache_dir=resolved_video_cache_dir,
            audio_path=audio_path,
            mask_base_dir=resolved_mask_base_dir,
            num_frames=num_frames,
            video_fps=video_fps,
            need_padding_to_16x=need_padding_to_16x,
            debug=debug,
            **kwargs
        )

    @NVTXContext
    def model_inference_by_chunk(
        self,
        chunk_faces: torch.Tensor,
        chunk_whisper_features: List[torch.Tensor] = None,
        num_frames: int = 16,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        height: int = 256,
        width: int = 256,
        debug: bool = False,
        **kwargs,
    ):
        """
        模型推理：处理单个chunk的推理
        
        Args:
            chunk_faces: 当前chunk的人脸张量 [num_frames, C, H, W]
            chunk_whisper_features: 当前chunk的音频特征列表 [num_frames]
            其他参数...
            
        Returns:
            torch.Tensor: 解码后的latents
        """
        return self.pipeline.model_inference_by_chunk_0701(
            chunk_faces=chunk_faces,
            chunk_whisper_features=chunk_whisper_features,
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

    @NVTXContext
    def postprocess_video(
        self,
        synced_video_frames: List[torch.Tensor],
        original_video_frames: List,
        boxes: List,
        affine_matrices: List,
        original_faces_len: int,
        num_frames: int = 16,
        debug: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        视频后处理：将所有推理结果恢复为最终视频帧
        
        Args:
            synced_video_frames: 推理后的latents列表
            original_video_frames: 原始视频帧
            boxes: 人脸框列表
            affine_matrices: 仿射变换矩阵列表
            original_faces_len: 原始faces长度
            num_frames: 每个chunk的帧数
            debug: 调试模式
            
        Returns:
            np.ndarray: 最终的视频帧数组
        """
        return self.pipeline.postprocess_video(
            synced_video_frames=synced_video_frames,
            original_video_frames=original_video_frames,
            boxes=boxes,
            affine_matrices=affine_matrices,
            original_faces_len=original_faces_len,
            num_frames=num_frames,
            debug=debug,
            **kwargs
        )

    def save_final_output(
        self,
        final_video_frames: np.ndarray,
        audio_samples: torch.Tensor,
        video_out_path: str,
        mask_frames: Optional[List] = None,
        mask_output_path: Optional[str] = None,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        convert_audio_to_48k: bool = True,
        debug: bool = False,
        **kwargs
    ):
        """
        保存最终输出：音频视频合成和mask视频输出
        
        Args:
            final_video_frames: 最终的视频帧数组
            audio_samples: 音频样本数据
            video_out_path: 视频输出路径
            mask_frames: mask帧列表（可选）
            mask_output_path: mask视频输出路径（可选）
            video_fps: 视频帧率
            audio_sample_rate: 音频采样率
            convert_audio_to_48k: 是否转换音频到48k
            debug: 调试模式
        """
        return self.pipeline.save_final_output(
            final_video_frames=final_video_frames,
            audio_samples=audio_samples,
            video_out_path=video_out_path,
            mask_frames=mask_frames,
            mask_output_path=mask_output_path,
            video_fps=video_fps,
            audio_sample_rate=audio_sample_rate,
            convert_audio_to_48k=convert_audio_to_48k,
            debug=debug,
            **kwargs
        )

    @NVTXContext
    def forward(self, audio_paths: List[str], output_paths: List[str], mask_output_paths: Optional[List[str]] = None) -> List[Dict]:
        """
        视频生成推理方法
        
        Args:
            audio_paths: 音频文件路径列表
            output_paths: 输出视频路径列表
            mask_output_paths: 输出mask视频路径列表（可选）
        
        Returns:
            List of generation results with keys: 'status_code', 'video_path', 'mask_path', 'error_message'
        """
        if not audio_paths or not output_paths:
            logger.warning("Empty input provided to forward method")
            return []
        
        batch_size = len(audio_paths)
        logger.info(f"Starting video generation for {batch_size} audio files")
        
        results = []
        
        # 由于视频生成是计算密集型任务，我们逐个处理
        for i in range(batch_size):
            try:
                audio_path = audio_paths[i]
                output_path = output_paths[i]
                mask_output_path = mask_output_paths[i] if mask_output_paths else None
                
                logger.info(f"Processing video {i+1}/{batch_size}: {os.path.basename(audio_path)}")
                
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if mask_output_path:
                    os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
                
                # 调用pipeline生成视频 - 拆分为四个步骤
                start_time = time.time()
                
                # 步骤1: 前处理
                logger.info(f"Step 1: Preprocessing data for {os.path.basename(audio_path)}")
                preprocess_data = self.preprocess_data(
                    video_cache_dir=self.video_config.get("video_cache_dir"),
                    audio_path=audio_path,
                    mask_base_dir=self.video_config.get("mask_base_dir"),
                    num_frames=16,
                    video_fps=self.video_config.get("target_fps", 25),
                    need_padding_to_16x=True,
                    debug=self.video_config.get("debug_pipeline", False),
                )
                
                # 步骤2: 模型推理
                logger.info(f"Step 2: Model inference for {os.path.basename(audio_path)}")
                synced_video_frames = []
                faces_by_chunk = preprocess_data['faces_by_chunk']
                whisper_chunks_by_chunk = preprocess_data['whisper_chunks_by_chunk']
                
                for chunk_idx in range(preprocess_data['num_inferences']):
                    chunk_start_time = time.time()
                    
                    decoded_latents = self.model_inference_by_chunk(
                        chunk_faces=faces_by_chunk[chunk_idx],
                        chunk_whisper_features=whisper_chunks_by_chunk[chunk_idx],
                        num_frames=16,
                        num_inference_steps=self.video_config.get("inference_steps", 20),
                        guidance_scale=self.video_config.get("guidance_scale", 1.0),
                        weight_dtype=self.dtype,
                        eta=0.0,
                        generator=None,
                        callback=None,
                        callback_steps=1,
                        height=256,
                        width=256,
                        debug=self.video_config.get("debug_pipeline", False),
                    )
                    
                    synced_video_frames.append(decoded_latents)
                    
                    chunk_time = time.time() - chunk_start_time
                    logger.info(f"Chunk {chunk_idx+1}/{preprocess_data['num_inferences']} completed in {chunk_time:.3f}s")
                
                # 步骤3: 后处理
                logger.info(f"Step 3: Postprocessing video for {os.path.basename(audio_path)}")
                final_video_frames = self.postprocess_video(
                    synced_video_frames=synced_video_frames,
                    original_video_frames=preprocess_data['original_video_frames'],
                    boxes=preprocess_data['boxes'],
                    affine_matrices=preprocess_data['affine_matrices'],
                    original_faces_len=preprocess_data['original_faces_len'],
                    num_frames=16,
                    debug=self.video_config.get("debug_pipeline", False),
                )
                
                # 步骤4: 保存最终输出
                logger.info(f"Step 4: Saving final output for {os.path.basename(audio_path)}")
                self.save_final_output(
                    final_video_frames=final_video_frames,
                    audio_samples=preprocess_data['audio_samples'],
                    video_out_path=output_path,
                    mask_frames=preprocess_data['mask_frames'],
                    mask_output_path=mask_output_path,
                    video_fps=self.video_config.get("target_fps", 25),
                    audio_sample_rate=16000,
                    convert_audio_to_48k=True,
                    debug=self.video_config.get("debug_pipeline", False),
                )
                
                gen_frame_num = preprocess_data['gen_frame_num']
                video_index_list = preprocess_data['video_index_list']
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # 检查输出文件是否存在
                if os.path.exists(output_path):
                    result = {
                        'status_code': PI_STATUS_CODE.SUCCESS,
                        'video_path': output_path,
                        'mask_path': mask_output_path if mask_output_path and os.path.exists(mask_output_path) else None,
                        'generation_time': generation_time,
                        'gen_frame_num': gen_frame_num,
                        'error_message': None
                    }
                    logger.info(f"Video generation successful: {os.path.basename(output_path)} ({generation_time:.2f}s)")
                else:
                    result = {
                        'status_code': PI_STATUS_CODE.GENERATION_ERROR,
                        'video_path': None,
                        'mask_path': None,
                        'generation_time': generation_time,
                        'gen_frame_num': 0,
                        'error_message': f"Output video file not found: {output_path}"
                    }
                    logger.error(f"Video generation failed: output file not found")
                
            except Exception as e:
                logger.error(f"Error processing video {i+1}: {str(e)}")
                result = {
                    'status_code': PI_STATUS_CODE.GENERATION_ERROR,
                    'video_path': None,
                    'mask_path': None,
                    'generation_time': 0,
                    'gen_frame_num': 0,
                    'error_message': str(e)
                }
            
            results.append(result)
        
        logger.info(f"Video generation completed for {batch_size} files")
        return results 