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
from latentsync.backends.utils import setup_diycache_unet

from latentsync.backends.utils import (
    load_trt_unet,
    load_trt_vae,
    calculate_max_device_memory,
    activate_engine,
    reshape,
    decode_latents,
    prepare_mask_latents,
    prepare_image_latents,
)

from cuda import cudart
# 配置torch.compile缓存设置
torch._dynamo.config.cache_size_limit = 64

class SNNTorchBackend(BaseBackend):
    """LatentSync视频生成后端 - 仅负责模型推理"""

    def __init__(self, config: BackendConfig, lightweight_mode: bool = False):
        super(SNNTorchBackend, self).__init__(config)
        self.config = config  # 保存config引用
        self.lightweight_mode = lightweight_mode  # 轻量级模式标志
        model_path = config.model_path

        self._use_fp16 = config.fp16
        self._device = config.device
        self._size = config.size

        # Load models
        config_file = getattr(config, 'config_file', None)
        if not config_file:
            raise ValueError("config_file is required for video generation")

        # 如果config_file是相对路径，拼到models目录
        if not os.path.isabs(config_file):
            # 获取models目录路径
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
            config_path = os.path.join(models_dir, config_file)
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
        if self.lightweight_mode:
            logger.info("🔧 Lightweight mode: loading audio models only")
            self._init_models(video_config, lightweight_mode=True)
            # 轻量级模式也需要创建一个临时的pipeline用于预处理和后处理
            self._init_lightweight_pipeline(video_config)
        else:
            logger.info("🔧 Full mode: loading all models")
            self._init_models(video_config, lightweight_mode=False)
            # 初始化pipeline
            self._init_pipeline(video_config)

        self._lock = threading.Lock()

    def _init_models(self, config: Dict, lightweight_mode: bool = False):
        """初始化模型组件"""
        if lightweight_mode:
            logger.info("Initializing lightweight LatentSync models (audio only)...")
        else:
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

        # 初始化调度器 - 处理相对路径（仅在完整模式下）
        if not lightweight_mode:
            scheduler_config_path = resolve_path(config.get("scheduler_config_path", "latentsync/configs"))
            logger.info(f"Loading scheduler config from: {scheduler_config_path}")
            scheduler = DDIMScheduler.from_pretrained(scheduler_config_path)
        else:
            scheduler = None
            logger.info("Skipping scheduler loading in lightweight mode")

        # 初始化音频编码器
        whisper_model_path = resolve_path(config.get("whisper_model_path", "weights/whisper/tiny.pt"))
        logger.info(f"Loading Whisper model from: {whisper_model_path}")
        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device="cuda",
            num_frames=16
        )

        # 初始化VAE（仅在完整模式下）
        if not lightweight_mode:
            vae_model_path = resolve_path(config.get("vae_model_path", "stabilityai/sd-vae-ft-mse"))
            logger.info(f"Loading VAE model from: {vae_model_path}")
            vae = AutoencoderKL.from_pretrained(
                vae_model_path,
                torch_dtype=dtype
            )
            vae = vae.to("cuda")
            vae.config.scaling_factor = 0.18215
            vae.config.shift_factor = 0
        else:
            vae = None
            logger.info("Skipping VAE loading in lightweight mode")

        # 初始化UNet（仅在完整模式下）
        if not lightweight_mode:
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
        else:
            unet = None
            logger.info("Skipping UNet loading in lightweight mode")

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

        # load TensorRT engine（仅在完整模式下）
        if not lightweight_mode:
            if self.config.fast_engine:
                logger.info("Loading TensorRT engine... ")
                device_path = torch.cuda.get_device_name(torch.cuda.current_device()).split()[-1]
                trt_root_path = os.path.join(self.config.model_path, "tensorrt", device_path)

                self.unet.__class__.load_trt = load_trt_unet
                self.unet.__class__.calculate_max_device_memory = calculate_max_device_memory
                self.unet.__class__.activate_engine = activate_engine
                self.unet.__class__.reshape = reshape
                self.unet.__class__.verbose = getattr(self.config, 'verbose', False)
                self.unet.__class__.backend = 'torch'

                self.vae.__class__.load_trt = load_trt_vae
                self.vae.__class__.calculate_max_device_memory = calculate_max_device_memory
                self.vae.__class__.activate_engine = activate_engine
                self.vae.__class__.reshape = reshape
                self.vae.__class__.verbose = getattr(self.config, 'verbose', False)
                self.vae.__class__.backend = 'torch'

                self.vae.__class__.decode_latents = decode_latents
                self.vae.__class__.prepare_mask_latents = prepare_mask_latents
                self.vae.__class__.prepare_image_latents = prepare_image_latents

                # Only support batch size 1 for now
                batch_size = 1

                trt_model_list = [
                    self.unet,
                    self.vae
                ]

                for item in trt_model_list:
                    item.load_trt(trt_root_path, batch_size, use_cuda_graph=True, verbose=getattr(self.config, 'verbose', False))

                max_device_memory = 0
                for item in trt_model_list:
                    max_device_memory = max(
                        max_device_memory, item.calculate_max_device_memory()
                    )

                _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
                for item in trt_model_list:
                    item.activate_engine(shared_device_memory, batch_size)

                logger.info("TensorRT models loaded successfully")
            else:
                self.unet.__class__.verbose = getattr(self.config, 'verbose', False)
                self.unet.__class__.backend = 'torch'
                self.vae.__class__.verbose = getattr(self.config, 'verbose', False)
                self.vae.__class__.backend = 'torch'
        else:
            logger.info("Skipping TensorRT setup in lightweight mode")

        if lightweight_mode:
            logger.info("Lightweight LatentSync models initialized successfully")
        else:
            logger.info("LatentSync models initialized successfully")

    def _init_lightweight_pipeline(self, config: Dict):
        """初始化轻量级pipeline（仅用于预处理和后处理）"""
        logger.info("Initializing lightweight LatentSync pipeline...")
        
        # 调用修改后的_init_pipeline方法，传入lightweight_mode=True
        self._init_pipeline(config, lightweight_mode=True)
        
        logger.info("Lightweight LatentSync pipeline initialized successfully")

    def _init_pipeline(self, config: Dict, lightweight_mode: bool = False):
        """初始化pipeline"""
        if lightweight_mode:
            logger.info("Initializing lightweight LatentSync pipeline...")
        else:
            logger.info("Initializing LatentSync pipeline...")

        # 创建pipeline
        self.pipeline = LipsyncPipeline(
            vae=self.vae,
            audio_encoder=self.audio_encoder,
            unet=self.unet,
            scheduler=self.scheduler,
            video_index_generator=self.video_index_generator,
            lightweight_mode=lightweight_mode,
        ).to("cuda")

        # 配置DiyCache（仅在完整模式下）
        if not lightweight_mode and config.get("enable_diycache", False):
            logger.info("Enabling DiyCache...")
            setup_diycache_unet(
                unet=self.unet,
                first_step_offset=config.get("first_step_offset", 1),
                last_step_offset=config.get("last_step_offset", 4),
                num_steps=config.get("inference_steps", 20)
            )

        if lightweight_mode:
            logger.info("Lightweight LatentSync pipeline initialized successfully")
        else:
            logger.info("LatentSync pipeline initialized successfully")

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

    def get_pipeline(self):
        """获取pipeline实例，供外部使用"""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            return self.pipeline
        
        # 如果没有pipeline，说明初始化有问题
        if self.lightweight_mode:
            raise RuntimeError("Lightweight pipeline not initialized. This should not happen.")
        else:
            raise RuntimeError("Full pipeline not initialized. This should not happen.")

    def get_video_config(self):
        """获取视频配置"""
        return self.video_config

    def get_dtype(self):
        """获取数据类型"""
        return self.dtype

    def get_device(self):
        """获取设备"""
        return self._device

