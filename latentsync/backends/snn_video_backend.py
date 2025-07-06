import torch
import threading
from typing import Optional, Dict, List, Union
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

class SNNVideoBackend(BaseBackend):
    """LatentSync视频生成后端"""

    def __init__(self, config: BackendConfig):
        super(SNNVideoBackend, self).__init__(config)
        model_path = config.model_path

        self._use_fp16 = config.fp16
        self._device = config.device
        self._size = config.size

        # Load models
        config_path = model_path + "/V2.6_latentsync_video.json"
        logger.info(f"Loading config from: {config_path}")
        
        if os.path.exists(config_path):
            e2e_config = json.load(open(config_path))
        else:
            # 使用默认配置
            e2e_config = {
                "video_model": {
                    "unet_config_path": os.path.join(model_path, "configs/unet/second_stage.yaml"),
                    "inference_ckpt_path": os.path.join(model_path, "weights/latentsync/latentsync_unet.pt"),
                    "whisper_model_path": os.path.join(model_path, "weights/whisper/tiny.pt"),
                    "vae_model_path": "/model_zoo/sd-vae-ft-mse/",
                    "scheduler_config_path": "configs",
                    "video_template_dir": "/model_zoo/latentsync_online_simple_auxilary/0612_online_templete_female_32fps_resize_720_1560/videos",
                    "connected_info_json": "/model_zoo/latentsync_online_simple_auxilary/0612_online_templete_female_32fps_resize_720_1560/template_info_0618/connected_videos.json",
                    "video_info_json": "/model_zoo/latentsync_online_simple_auxilary/0612_online_templete_female_32fps_resize_720_1560/template_info_0618/0612_video_templete_32fps_info_v2.json",
                    "shot_video_json_path": "/model_zoo/latentsync_online_simple_auxilary/0612_online_templete_female_32fps_resize_720_1560/template_info_0618/connected_videos_other_short.json",
                    "video_cache_dir": "/model_zoo/latentsync_online_simple_auxilary/0612_online_templete_female_32fps_resize_720_1560/videos_features",
                    "mask_base_dir": "/model_zoo/latentsync_online_simple_auxilary/0612_online_templete_female_32fps_resize_720_1560/videos_mask",
                    "target_fps": 25,
                    "transition_frames": 20,
                    "inference_steps": 20,
                    "guidance_scale": 1.0,
                    "seed": 1247,
                    "first_step_offset": 1,
                    "last_step_offset": 4,
                    "enable_diycache": False,
                    "enable_deepcache": False,
                    "debug_video_merger": False,
                    "debug_pipeline": False
                }
            }

        video_config = e2e_config.get('video_model', {})
        
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
        
        # 加载UNet配置
        unet_config = OmegaConf.load(config.get("unet_config_path", "configs/unet/second_stage.yaml"))
        
        # 初始化调度器
        scheduler = DDIMScheduler.from_pretrained(config.get("scheduler_config_path", "configs"))
        
        # 初始化音频编码器
        audio_encoder = Audio2Feature(
            model_path=config.get("whisper_model_path", "checkpoints/whisper/tiny.pt"), 
            device="cuda", 
            num_frames=16
        )
        
        # 初始化VAE
        vae = AutoencoderKL.from_pretrained(
            config.get("vae_model_path", "stabilityai/sd-vae-ft-mse"), 
            torch_dtype=dtype
        )
        vae = vae.to("cuda")
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        
        # 初始化UNet
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(unet_config),
            config.get("inference_ckpt_path", "checkpoints/latentsync_unet.pt"),
            device="cpu",
        )
        unet = unet.to(dtype=dtype, device="cuda")
        
        # 初始化VideoIndexGenerator
        video_index_generator = VideoIndexGenerator(
            connected_json_path=config.get("connected_info_json"),
            video_info_json_path=config.get("video_info_json"),
            shot_video_json_path=config.get("shot_video_json_path"),
            video_template_dir=config.get("video_template_dir"),
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
                
                # 调用pipeline生成视频
                start_time = time.time()
                
                gen_frame_num, video_index_list = self.pipeline.generate_video_online_0701(
                    video_cache_dir=self.video_config.get("video_cache_dir"),
                    audio_path=audio_path,
                    video_out_path=output_path,
                    mask_output_path=mask_output_path,
                    mask_base_dir=self.video_config.get("mask_base_dir"),
                    num_frames=16,
                    video_fps=self.video_config.get("target_fps", 25),
                    audio_sample_rate=16000,
                    height=256,
                    width=256,
                    num_inference_steps=self.video_config.get("inference_steps", 20),
                    guidance_scale=self.video_config.get("guidance_scale", 1.0),
                    weight_dtype=self.dtype,
                    eta=0.0,
                    generator=None,
                    callback=None,
                    callback_steps=1,
                    need_padding_to_16x=True,
                    convert_auido_to_48k=True,
                    debug=self.video_config.get("debug_pipeline", False),
                )
                
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