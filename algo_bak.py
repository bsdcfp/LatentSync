# coding: utf-8
from mmu_service_grammar.runtime import *
import mmu_service_grammar.triton.logger as logger
from mmu_service_grammar.triton.mock.context import MockContext


import io
import json
import asyncio
import sys
import gzip
import pickle
import subprocess
import soundfile as sf
import requests
import base64
import orjson

import os
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
import tempfile
import librosa
import time
import traceback
from typing import Union, Tuple, Optional, List, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

from einops import rearrange
from latentsync.models.unet import UNet3DConditionModel
# 临时使用
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import BaseOutput
import inspect

# 导入必要的模块
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.utils.util import write_video, read_video
from latentsync.utils.video_sequence_merger import VideoIndexGenerator
from latentsync.utils.video_template_cache import VideoTemplateCache
# 上传相关导入
from core.env import env
from core.config.client import init_default_config_mgr
from uploadSdk import sdk
from uploadSdk.sdk import new_upload_sdk
from uploadSdk.model import FullUploadInfo, MediaInfo
from latentsync.utils.image_processor_v01_0516 import ImageProcessor

# 尝试导入UNet3DConditionOutput
try:
    from latentsync.models.unet import UNet3DConditionOutput
except ImportError:
    @dataclass
    class UNet3DConditionOutput(BaseOutput):
        sample: torch.FloatTensor

SUCCESS_CODE = 0
CHUNK_INFERENCE_FAILED_CODE = 1
UPLOAD_ERROR_CODE = 3       

class ModelConfig:
    """模型配置类，整合所有配置参数"""
    def __init__(self):
        # 基础路径配置 - 动态获取当前文件的父目录
        auto_script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
        # 如果自动获取的路径不存在，使用默认路径
        if os.path.exists(auto_script_dir):
            self.script_dir = auto_script_dir
        else:
            self.script_dir = "/home/work/houyi/mmuplt-ailive1"  # fallback到默认路径
            if not os.path.exists(self.script_dir):
                self.script_dir = "/home/work/mmuplt-ailive1"
            logger.warning(f"⚠️  自动路径不存在，使用默认路径: {self.script_dir}")
            
        #self.auxiliary_dir = "/home/work/latentsync_online_simple_auxilary"
        self.auxiliary_dir = "/home/work/houyi/latentsync_online_simple_auxilary"
        if not os.path.exists(self.auxiliary_dir):
            self.auxiliary_dir = "/home/work/houyi/latentsync_online_simple_auxilary"
            logger.warning(f"⚠️  auxiliary_dir路径不存在，使用默认路径: {self.auxiliary_dir}")
            
        self.example_audio_path = os.path.join(self.auxiliary_dir, "input_example/beauty_v2_20s.MP3")
        
        # 模型配置文件
        self.unet_config_path = os.path.join(self.script_dir, "configs/unet/second_stage.yaml")
        
        # 权重路径
        self.weights_dir = os.path.join(self.auxiliary_dir, "weights/latentsync")
        self.inference_ckpt_path = os.path.join(self.weights_dir, "latentsync_unet.pt")
        self.whisper_model_path = os.path.join(self.weights_dir, "whisper/tiny.pt")
        
        # 模板配置
        #self.template_dir = os.path.join(self.auxiliary_dir, "0612_online_templete_female_32fps_resize_720_1280")
        self.template_dir = os.path.join(self.auxiliary_dir, "templates/0627_online_template_male_32fps_resize_720_1280")
        self.video_template_dir = os.path.join(self.template_dir, "videos")
        self.cache_dir = os.path.join(self.template_dir, "videos_features")
        self.mask_base_dir = os.path.join(self.template_dir, "videos_mask")
        
        # 模板信息文件
        self.connected_video_info = os.path.join(self.template_dir, "meta_files/connected_videos.json")
        self.shot_video_json = os.path.join(self.template_dir, "meta_files/connected_videos_other_short.json")
        self.video_info_json = os.path.join(self.template_dir, "meta_files/video_template_32fps_info.json")
        self.start_video_list = os.path.join(self.template_dir, "meta_files/start_video_list.json")

        
        # 推理参数
        self.target_fps = 25
        self.transition_frames = 20
        self.inference_steps = 20
        self.guidance_scale = 1.0
        self.seed = 1247
        self.num_frames = 16
        
        # GPU配置
        self.device = "cuda"
        self.dtype = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7 else torch.float32
        
        # DiyCache配置
        self.enable_diycache = True  # 默认启用diycache
        self.first_step_offset = 1   # 第一步偏移
        self.last_step_offset = 3    # 最后一步偏移
        
        # 返回配置
        # self.return_mode = "base64"  # "base64" 或 "url"
        self.return_mode = "url"  # "base64" 或 "url"
        self.upload_environment = "live"  # "test" 或 "live"
        
        # 打印路径信息用于调试
        logger.debug(f"🔧 路径配置信息:")
        logger.debug(f"  - 自动获取路径: {auto_script_dir}")
        logger.debug(f"  - 实际使用路径: {self.script_dir}")
        logger.debug(f"  - 使用方式: {'自动获取' if self.script_dir == auto_script_dir else '默认fallback'}")
        logger.debug(f"  - Auxiliary目录: {self.auxiliary_dir}")
        
        # 检查关键路径是否存在
        logger.debug(f"🔍 路径存在性检查:")
        logger.debug(f"  - Script目录存在: {os.path.exists(self.script_dir)}")
        logger.debug(f"  - UNet配置文件存在: {os.path.exists(self.unet_config_path)}")
        logger.debug(f"  - Auxiliary目录存在: {os.path.exists(self.auxiliary_dir)}")
        logger.debug(f"  - 权重目录存在: {os.path.exists(self.weights_dir)}")
        logger.debug(f"  - 模板目录存在: {os.path.exists(self.template_dir)}")
        
        if not os.path.exists(self.script_dir):
            logger.warning(f"⚠️  警告: Script目录不存在: {self.script_dir}")
        if not os.path.exists(self.auxiliary_dir):
            logger.warning(f"⚠️  警告: Auxiliary目录不存在: {self.auxiliary_dir}")

"""
视频特征算子
"""
@deployment(num_replicas=5)
class VideoFeature(object):
    def __init__(self):
        self.config = ModelConfig()
        # 一次性load所有视频数据
        logger.info(f"VideoFeature init: video_cache_dir: {self.config.cache_dir}, mask_base_dir: {self.config.mask_base_dir}")
        self.video_cache_dir = self.config.cache_dir
        self.mask_base_dir = self.config.mask_base_dir
        self.device = torch.device("cuda:0") # TODO 待实现
        self.video_index_generator = VideoIndexGenerator(
            connected_json_path=self.config.connected_video_info,
            video_info_json_path=self.config.video_info_json,
            shot_video_json_path=self.config.shot_video_json,
            start_video_list_path=self.config.start_video_list,
            video_template_dir=self.config.video_template_dir,
            target_fps=self.config.target_fps,
            transition_frames=self.config.transition_frames,
            debug=False
        )
        self.video_template_cache = VideoTemplateCache(
            video_info_config_path=self.config.video_info_json,
            video_template_dir=self.config.video_template_dir,
            video_feature_dir=self.config.cache_dir,
            mask_base_dir=self.config.mask_base_dir
        )
        self.video_template_cache.load_all_data(debug=False)
    
    async def __call__(self, audio_duration):
        # 获取本次请求需要的数据 # TODO 待实现
        faces = None
        original_video_frames = None
        boxes = None
        affine_matrices = None
        mask_frames = None
        video_index_list = None
        original_faces_len = None
        # 获取视频索引列表（只需要音频时长）
        video_index_list = self.video_index_generator.get_video_index(audio_path=None, audio_duration=audio_duration)
        # 加载视频帧和特征数据
        faces, original_video_frames, boxes, affine_matrices, mask_frames = self._load_video_and_mask_data_from_index_list_v2(
            video_index_list, self.device, self.video_template_cache)

        original_faces_len = len(faces)
        video_data =  {
            'faces': faces,
            'original_video_frames': original_video_frames,
            'boxes': boxes,
            'affine_matrices': affine_matrices,
            'mask_frames': mask_frames,
            'video_index_list': video_index_list,
            'original_faces_len': original_faces_len,
        }
        return video_data
    
    def _load_video_and_mask_data_from_index_list_v2(self, video_index_list: List[dict], device, video_template_cache: VideoTemplateCache):
        all_faces = []
        all_original_frames = []
        all_boxes = []
        all_affine_matrices = []
        all_mask_frames = []
        
        # 统计总的预期帧数
        total_expected_frames = sum(len(item['frame_indices']) for item in video_index_list)
        
        for video_info in video_index_list:
            video_name = video_info["video_name"]
            frame_indices = video_info["frame_indices"]
            video_path = video_info["video_path"]

            video_template_data = self.video_template_cache.get_cache_data(video_name)

            #1 加载缓存特征
            video_feature = video_template_data['video_feature']
            if video_feature is None:
                if self.debug:
                    print('[DEBUG]>>>>> video_feature for video[{video_name}] is None')
                continue
            cached_faces = video_feature['faces']
            if torch.is_tensor(cached_faces):
                cached_faces = cached_faces.to(device).float()
            cached_boxes = video_feature['boxes']
            cached_affine_matrices = video_feature['affine_matrices']

            #2 加载原始视频帧
            original_video_frames = video_template_data['video_frames']
            if original_video_frames is None:
                if self.debug:
                    print('[DEBUG]>>>>> original_video_frames for video[{video_name}] is None')
                continue

            #3 加载mask视频帧
            mask_video_frames = video_template_data['mask_video_frames']

            # 4. 根据frame_indices提取对应的数据
            selected_faces = []
            selected_original_frames = []
            selected_boxes = []
            selected_affine_matrices = []
            selected_mask_frames = []
            
            for idx in frame_indices:
                if 0 <= idx < len(cached_faces) and 0 <= idx < len(original_video_frames):
                    selected_faces.append(cached_faces[idx])
                    selected_original_frames.append(original_video_frames[idx])
                    selected_boxes.append(cached_boxes[idx])
                    selected_affine_matrices.append(cached_affine_matrices[idx])
                    
                    # 添加mask帧（如果存在）
                    if mask_video_frames is not None and 0 <= idx < len(mask_video_frames):
                        selected_mask_frames.append(mask_video_frames[idx])
                    elif self.video_template_cache.mask_base_dir is not None:
                        # 如果需要mask但该帧不存在，记录警告
                        print(f"⚠️  警告: mask帧索引 {idx} 超出范围, len(mask_video_frames): {len(mask_video_frames) if mask_video_frames else 0}")
                else:
                    print(f"⚠️  警告: 帧索引 {idx} 超出范围, len(cached_faces): {len(cached_faces)}, original_video_frames: {len(original_video_frames)}, cache_file: {cache_file}, video_path: {video_path}")
            
            # 5. 添加到总列表
            all_faces.extend(selected_faces)
            all_original_frames.extend(selected_original_frames)
            all_boxes.extend(selected_boxes)
            all_affine_matrices.extend(selected_affine_matrices)
            if self.video_template_cache.mask_base_dir is not None:
                all_mask_frames.extend(selected_mask_frames)
        
        if not all_faces:
            raise ValueError("没有成功加载任何视频数据")
        
        # 5. 转换为张量格式
        faces = torch.stack(all_faces) if torch.is_tensor(all_faces[0]) else torch.stack([torch.tensor(f) for f in all_faces])
        
        return faces, all_original_frames, all_boxes, all_affine_matrices, all_mask_frames
    
    # 这个是原始的加载逻辑，后面等标哥替换掉
    def _load_video_and_mask_data_from_index_list(self, video_index_list: List[dict], device, video_cache_dir, mask_base_dir):
        """
        根据视频索引列表加载视频帧和特征数据
        
        参数:
            video_index_list: 视频索引列表，每个元素包含video_name、frame_indices和video_path
            device: 目标设备
            video_cache_dir: 缓存目录
            mask_base_dir: mask视频基础目录
            debug: 是否启用调试模式
            
        返回:
            faces: 人脸特征张量
            original_video_frames: 原始视频帧列表
            boxes: 人脸框列表
            affine_matrices: 仿射变换矩阵列表
            mask_frames: mask视频帧列表
        """

        all_faces = []
        all_original_frames = []
        all_boxes = []
        all_affine_matrices = []
        all_mask_frames = []
        
        # 统计总的预期帧数
        total_expected_frames = sum(len(item['frame_indices']) for item in video_index_list)
        
        for video_info in video_index_list:
            video_name = video_info["video_name"]
            frame_indices = video_info["frame_indices"]
            video_path = video_info["video_path"]
            
            # 1. 加载缓存的特征数据

            cache_file = os.path.join(video_cache_dir, f"{video_name}_features.pkl")
            
            if not os.path.exists(cache_file):
                if debug:
                    print(f"⚠️  警告: 特征文件不存在: {video_path}")
                continue
                
            # 加载缓存的特征
            with gzip.open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cached_faces = cached_data['faces']
                if torch.is_tensor(cached_faces):
                    cached_faces = cached_faces.to(device).float()
                cached_boxes = cached_data['boxes']
                cached_affine_matrices = cached_data['affine_matrices']
            
            # 2. 读取原始视频帧
            if not os.path.exists(video_path):
                continue
            
            original_video_frames = read_video(video_path, use_decord=False, change_fps=False)
            # print(f"video_path: {video_path}: len original_video_frames: {len(original_video_frames)}, original_video_frames[0].shape: {original_video_frames[0].shape}, original_video_frames[0].dtype: {original_video_frames[0].dtype}")

            # 3. 如果需要加载mask视频
            mask_video_frames = None
            if mask_base_dir is not None:
                mask_video_name = f"{video_name}_mask.mp4"
                mask_video_path = os.path.join(mask_base_dir, mask_video_name)
                
                if os.path.exists(mask_video_path):
                    mask_video_frames = read_video(mask_video_path, use_decord=False, change_fps=False)
                    
                else:
                    print(f"⚠️  警告: mask视频文件不存在: {mask_video_path}")
            
            # 4. 根据frame_indices提取对应的数据
            selected_faces = []
            selected_original_frames = []
            selected_boxes = []
            selected_affine_matrices = []
            selected_mask_frames = []
            
            for idx in frame_indices:
                if 0 <= idx < len(cached_faces) and 0 <= idx < len(original_video_frames):
                    selected_faces.append(cached_faces[idx])
                    selected_original_frames.append(original_video_frames[idx])
                    selected_boxes.append(cached_boxes[idx])
                    selected_affine_matrices.append(cached_affine_matrices[idx])
                    
                    # 添加mask帧（如果存在）
                    if mask_video_frames is not None and 0 <= idx < len(mask_video_frames):
                        selected_mask_frames.append(mask_video_frames[idx])
                    elif mask_base_dir is not None:
                        # 如果需要mask但该帧不存在，记录警告
                        print(f"⚠️  警告: mask帧索引 {idx} 超出范围, len(mask_video_frames): {len(mask_video_frames) if mask_video_frames else 0}")
                else:
                    print(f"⚠️  警告: 帧索引 {idx} 超出范围, len(cached_faces): {len(cached_faces)}, original_video_frames: {len(original_video_frames)}, cache_file: {cache_file}, video_path: {video_path}")
            
            # 5. 添加到总列表
            all_faces.extend(selected_faces)
            all_original_frames.extend(selected_original_frames)
            all_boxes.extend(selected_boxes)
            all_affine_matrices.extend(selected_affine_matrices)
            if mask_base_dir is not None:
                all_mask_frames.extend(selected_mask_frames)
        
        if not all_faces:
            raise ValueError("没有成功加载任何视频数据")
        
        # 5. 转换为张量格式
        faces = torch.stack(all_faces) if torch.is_tensor(all_faces[0]) else torch.stack([torch.tensor(f) for f in all_faces])
        
        return faces, all_original_frames, all_boxes, all_affine_matrices, all_mask_frames


"""
audio特征算子
"""
@deployment(num_replicas=5)
class AudioFeature(object):
    def __init__(self):
        self.config = ModelConfig()
        # 加载UNet配置
        self.unet_config = OmegaConf.load(self.config.unet_config_path)
        logger.info(f"AudioFeature init: model_path: {self.config.whisper_model_path}, device: {self.config.device}, num_frames: {self.unet_config.data.num_frames}")
        self.audio_encoder = Audio2Feature(
            model_path=self.config.whisper_model_path, 
            device=self.config.device, 
            num_frames=self.unet_config.data.num_frames
        )
    
    async def __call__(self, audio_bytes, fps):
        audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        audio_samples = torch.from_numpy(audio_array).float()
        audio_duration = len(audio_array) / sr
        preloaded_audio = (audio_array, sr)  # 保存预加载的音频数据
        whisper_feature = self.audio_encoder.audio2feat(audio_bytes, preloaded_audio=preloaded_audio)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=fps)
        audio_features = {
            'whisper_chunks': whisper_chunks,
            'audio_samples': audio_samples,
            'audio_duration': audio_duration
        }
        return audio_features

def apply_padding(
        faces: torch.Tensor,
        whisper_chunks: Optional[List[torch.Tensor]],
        num_frames: int = 16,
        need_padding_to_16x: bool = True
    ) -> dict:
        """
        对faces和whisper特征一起做padding，确保长度匹配
        
        Args:
            faces: 视频faces张量
            whisper_chunks: whisper特征列表
            num_frames: 每个chunk的帧数
            need_padding_to_16x: 是否需要padding到16的倍数
            debug: 调试模式
            
        Returns:
            dict: 包含padding后的数据和计算信息
        """
        padding_frames = 0
        num_inferences = 0
        
        if whisper_chunks is not None:
            if need_padding_to_16x:
                original_audio_len = len(whisper_chunks)
                remainder = original_audio_len % num_frames
                padding_frames = (num_frames - remainder) % num_frames
                
                # 对音频特征进行padding
                if padding_frames != 0:
                    padding_method = "last_frame_padding"
                    if padding_method == "zero_padding":
                        padding_audio_tensor = [torch.zeros_like(whisper_chunks[0]) for _ in range(padding_frames)]
                    elif padding_method == "last_frame_padding":
                        padding_audio_tensor = [whisper_chunks[-1]] * padding_frames
                    else:
                        raise NotImplementedError(f"padding_method {padding_method} not implemented!")
                    whisper_chunks += padding_audio_tensor

                num_inferences = len(whisper_chunks) // num_frames
                faces_len = len(faces)
                required_frames = num_inferences * num_frames
                
                # 对faces进行padding或截取
                if faces_len >= required_frames:
                    faces = faces[:required_frames]
                else:
                    faces_padding_needed = required_frames - faces_len
                    face_last_frame = faces[-1:] if faces_len > 0 else faces[:1]
                    faces_padding_frames = face_last_frame.repeat(faces_padding_needed, 1, 1, 1)
                    faces = torch.cat([faces, faces_padding_frames], dim=0)
            else:
                num_inferences = min(len(faces), len(whisper_chunks)) // num_frames
                faces = faces[:num_inferences * num_frames]
                whisper_chunks = whisper_chunks[:num_inferences * num_frames]
        else:
            # 没有音频特征的情况
            num_inferences = len(faces) // num_frames
            faces = faces[:num_inferences * num_frames]

        gen_frame_num = num_inferences * num_frames
        return {
            'faces': faces,  # padding后的faces
            'whisper_chunks': whisper_chunks,  # padding后的whisper特征
            'num_inferences': num_inferences,
            'gen_frame_num': gen_frame_num,
            'padding_frames': padding_frames
        }

def split_data_by_chunk(video_data, audio_features, padding_data, num_frames):
    faces_by_chunk = []
    whisper_chunks_by_chunk = []
    for i in range(padding_data['num_inferences']):
        start_idx = i * num_frames
        end_idx = (i + 1) * num_frames
        # 分割faces
        face_chunk = padding_data['faces'][start_idx:end_idx]
        faces_by_chunk.append(face_chunk)
        # 分割whisper特征
        if padding_data['whisper_chunks'] is not None:
            whisper_chunk = padding_data['whisper_chunks'][start_idx:end_idx]
            whisper_chunks_by_chunk.append(whisper_chunk)
        else:
            whisper_chunks_by_chunk.append(None)
    
    return {
        # 视频数据相关
        'original_video_frames': video_data['original_video_frames'],
        'boxes': video_data['boxes'],
        'affine_matrices': video_data['affine_matrices'],
        'mask_frames': video_data['mask_frames'],
        'audio_samples': audio_features['audio_samples'],
        'video_index_list': video_data['video_index_list'],
        'original_faces_len': video_data['original_faces_len'],
            
        # padding处理结果
        'num_inferences': padding_data['num_inferences'],
        'gen_frame_num': padding_data['gen_frame_num'],
        'padding_frames': padding_data['padding_frames'],
            
        # 分chunk后的数据（新增）
        'faces_by_chunk': faces_by_chunk,  # List[torch.Tensor], 每个元素形状为 [num_frames, C, H, W]
        'whisper_chunks_by_chunk': whisper_chunks_by_chunk,  # List[List[torch.Tensor]], 每个chunk的音频特征
    }
    
def gen_mask_video(mask_output_path, mask_frames, video_fps, final_video_frame_num):
    logger.info(f"🔍 生成mask视频, 开始时间: {time.time()}")
    mask_frames_output = mask_frames[:final_video_frame_num]
    os.makedirs(os.path.dirname(os.path.abspath(mask_output_path)), exist_ok=True)
    write_video(mask_output_path, mask_frames_output, fps=video_fps)
    logger.info(f"🔍 生成mask视频, 结束时间: {time.time()}")


def diycache_forward(
    self,
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    return_dict: bool = True,
) -> Union[BaseOutput, Tuple]:
    """DiyCache版本的UNet3D forward方法"""
    # 基本设置
    default_overall_up_factor = 2**self.num_upsamplers
    forward_upsample_size = False

    if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
        forward_upsample_size = True

    # 准备attention_mask
    if attention_mask is not None:
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # 中心化输入
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 时间嵌入处理
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)

    timesteps = timesteps.expand(sample.shape[0])
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=self.dtype)
    emb = self.time_embedding(t_emb)

    # 类别嵌入
    if self.class_embedding is not None:
        if class_labels is None:
            raise ValueError("class_labels should be provided when num_class_embeds > 0")
        
        if self.config.class_embed_type == "timestep":
            class_labels = self.time_proj(class_labels)
        
        class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
        emb = emb + class_emb

    # 预处理
    sample = self.conv_in(sample)

    # 记录总步数统计
    if hasattr(self, 'total_steps'):
        self.total_steps += 1

    # 主要计算逻辑 - DiyCache核心逻辑
    if self.enable_diy_tcache:
        if self.cnt < self.first_step_offset or self.cnt >= self.num_steps - self.last_step_offset:
            # 重新计算
            ori_sample = sample.clone()
            final_sample = self._compute_unet_blocks(
                sample, emb, encoder_hidden_states, attention_mask,
                down_block_additional_residuals, mid_block_additional_residual,
                forward_upsample_size
            )
            current_residual = final_sample - ori_sample
            
            if hasattr(self, 'previous_residual') and self.previous_residual is not None:
                self.previous_residual = current_residual
            else:
                self.previous_residual = current_residual
        else:
            # 使用缓存的残差
            final_sample = sample + self.previous_residual
    else:
        # 标准计算路径
        final_sample = self._compute_unet_blocks(
            sample, emb, encoder_hidden_states, attention_mask,
            down_block_additional_residuals, mid_block_additional_residual,
            forward_upsample_size
        )

    # 后处理
    final_sample = self.conv_norm_out(final_sample)
    final_sample = self.conv_act(final_sample)
    final_sample = self.conv_out(final_sample)

    # 更新计数器
    self.cnt += 1
    if self.cnt >= self.num_steps:
        # 重置状态
        self.cnt = 0
        self.previous_residual = None

    if not return_dict:
        return (final_sample,)

    return UNet3DConditionOutput(sample=final_sample)

def _compute_unet_blocks(
    self,
    sample,
    emb,
    encoder_hidden_states,
    attention_mask,
    down_block_additional_residuals,
    mid_block_additional_residual,
    forward_upsample_size
):
    """执行UNet的主要计算：down blocks + mid block + up blocks"""
    # Down blocks
    down_block_res_samples = (sample,)
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )
        else:
            sample, res_samples = downsample_block(
                hidden_states=sample, 
                temb=emb, 
                encoder_hidden_states=encoder_hidden_states
            )
        down_block_res_samples += res_samples

    # 支持ControlNet
    down_block_res_samples = list(down_block_res_samples)
    if down_block_additional_residuals is not None:
        for i, down_block_additional_residual in enumerate(down_block_additional_residuals):
            if down_block_additional_residual.dim() == 4:
                down_block_additional_residual = down_block_additional_residual.unsqueeze(2)
            down_block_res_samples[i] = down_block_res_samples[i] + down_block_additional_residual

    # Mid block
    sample = self.mid_block(
        sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
    )

    # 支持ControlNet
    if mid_block_additional_residual is not None:
        if mid_block_additional_residual.dim() == 4:
            mid_block_additional_residual = mid_block_additional_residual.unsqueeze(2)
        sample = sample + mid_block_additional_residual

    # Up blocks
    for i, upsample_block in enumerate(self.up_blocks):
        is_final_block = i == len(self.up_blocks) - 1

        res_samples = down_block_res_samples[-len(upsample_block.resnets):]
        down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]
        else:
            upsample_size = None

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                upsample_size=upsample_size,
                attention_mask=attention_mask,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
                encoder_hidden_states=encoder_hidden_states,
            )

    return sample

def setup_diycache_unet(unet, first_step_offset=1, last_step_offset=3, num_steps=20):
    """
    为UNet3DConditionModel设置DiyCache功能
    
    Args:
        unet: UNet3DConditionModel实例
        first_step_offset: 第一步的偏移系数，此前不能跳步
        last_step_offset: 最后一步的偏移系数，此后不能跳步  
        num_steps: 总推理步数
    """
    # 添加DiyCache相关属性
    unet.__class__.enable_diy_tcache = True
    unet.__class__.forward = diycache_forward
    unet.__class__._compute_unet_blocks = _compute_unet_blocks
    
    # 初始化DiyCache参数
    unet.__class__.cnt = 0
    unet.__class__.num_steps = num_steps
    unet.__class__.first_step_offset = first_step_offset
    unet.__class__.last_step_offset = last_step_offset
    
    # 初始化缓存
    unet.__class__.previous_residual = None
    
    logger.info(f"✅ DiyCache已启用 - 第一步偏移: {first_step_offset}, 最后一步偏移: {last_step_offset}, 总步数: {num_steps}")
    return unet

@deployment(num_replicas=2)
class ModelInferenceByRpc(object):
    def __init__(self, ):
        self.config = ModelConfig()
        # 加载UNet配置
        self.unet_config = OmegaConf.load(self.config.unet_config_path)
        # 初始化线程池
        self.pool = ThreadPoolExecutor(max_workers=16)
        self.remote_url = "http://sg12.aip.mlp.shopee.io/services/555883/api/process"

    async def __call__(self,
        num_inferences: int,
        chunk_faces: torch.Tensor,
        chunk_whisper_features: List[torch.Tensor] = None,
        num_frames: int = 16,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        weight_dtype: Optional[torch.dtype] = torch.float16, # 远程调用时，不使用这个参数，服务器端自己设置
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, # 远程调用时，不使用这个参数，为了和本地推理的兼容性，暂时保留
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None, # 远程调用时，不使用这个参数，为了和本地推理的兼容性，暂时保留
        callback_steps: Optional[int] = 1, # 远程调用时，不使用这个参数，为了和本地推理的兼容性，暂时保留
        height: int = 256,
        width: int = 256):
        synced_video_frames = []
        args_list = []
        futures = []
        model_inference_by_rpc_start_time = time.time()
        for i in range(num_inferences):
            chunk_id = i
            chunk_faces_chunk=chunk_faces[i] # 当前chunk的faces
            chunk_whisper_features_chunk=chunk_whisper_features[i]  # 当前chunk的音频特征
            args_list.append((chunk_id, chunk_faces_chunk, chunk_whisper_features_chunk, num_frames, num_inference_steps, guidance_scale, eta, height, width))
            
        for args in args_list:
            future = self.pool.submit(self.model_inference_by_rpc, *args) 
            futures.append(future)
        for future in futures:
            synced_video_frames.append(future.result())
        model_inference_by_rpc_end_time = time.time()
        model_inference_by_rpc_time = model_inference_by_rpc_end_time - model_inference_by_rpc_start_time
        logger.info(f"SUCCESS - Model inference by rpc time: {model_inference_by_rpc_time:.2f} s")
        return synced_video_frames
    
    def model_inference_by_rpc(self,
        chunk_id: str,
        chunk_faces: torch.Tensor,  # 只传入当前chunk的faces, 形状为 [num_frames, C, H, W]
        chunk_whisper_features: List[torch.Tensor] = None,  # 只传入当前chunk的音频特征
        num_frames: int = 16,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
        height: int = 256,
        width: int = 256):
        single_chunk_inference_start_time = time.time()
        request_data = self.prepare_request_payload(
            chunk_id, 
            chunk_faces, 
            chunk_whisper_features, 
            num_frames, 
            num_inference_steps, 
            guidance_scale, 
            eta, 
            height, 
            width 
        )
        single_chunk_prepare_request_payload_end_time = time.time()
        single_chunk_prepare_request_payload_time = single_chunk_prepare_request_payload_end_time - single_chunk_inference_start_time
        logger.info(f"[TEST {chunk_id}] SUCCESS - Single chunk prepare request payload time: {single_chunk_prepare_request_payload_time:.2f} s")   
        request_start_time = time.time()
        chunk_result = self.request(request_data, chunk_id)
        request_end_time = time.time()
        request_time = request_end_time - request_start_time
        if chunk_result['status'] != 'SUCCESS':
            logger.error(f"[TEST {chunk_id}] FAILED - {chunk_result['error_message']}")
            return None
        response_data = chunk_result['response_data']
        if response_data is None:
            logger.error(f"[TEST {chunk_id}] FAILED - No response data")
            return None
        decoded_latents_raw = response_data.get('decoded_latents')
        if decoded_latents_raw is None:
            logger.error(f"[TEST {chunk_id}] FAILED - No decoded latents")
            return None
        logger.info(f"[TEST {chunk_id}] SUCCESS - Request time: {request_time:.2f} s")
        # 转换回tensor格式
        single_chunk_from_numpy_start_time = time.time()
        decoded_latents = torch.from_numpy(decoded_latents_raw)
        single_chunk_from_numpy_end_time = time.time()
        single_chunk_from_numpy_time = single_chunk_from_numpy_end_time - single_chunk_from_numpy_start_time
        logger.info(f"[TEST {chunk_id}] SUCCESS - Single chunk decompress time: {single_chunk_from_numpy_time:.2f} s")
        single_chunk_to_device_start_time = time.time()
        decoded_latents = decoded_latents.to(torch.device("cuda:0"))
        single_chunk_to_device_end_time = time.time()
        single_chunk_to_device_time = single_chunk_to_device_end_time - single_chunk_to_device_start_time
        logger.info(f"[TEST {chunk_id}] SUCCESS - Single chunk to device time: {single_chunk_to_device_time:.2f} s")
        single_chunk_inference_end_time = time.time()
        single_chunk_inference_time = single_chunk_inference_end_time - single_chunk_inference_start_time
        logger.info(f"[TEST {chunk_id}] SUCCESS - Single chunk inference time: {single_chunk_inference_time:.2f} s")
        return decoded_latents
    
    def prepare_request_payload(self, 
        chunk_id, 
        chunk_faces, 
        chunk_whisper_features, 
        num_frames, 
        num_inference_steps, 
        guidance_scale, 
        eta, 
        height, 
        width):
        chunk_faces_numpy_start_time = time.time()
        faces_np = chunk_faces.cpu().numpy()
        chunk_faces_end_end_time = time.time()
        chunk_faces_numpy_time = chunk_faces_end_end_time - chunk_faces_numpy_start_time
        logger.info(f"SUCCESS - Chunk faces numpy time: {chunk_faces_numpy_time:.2f} s")
        if isinstance(chunk_whisper_features, list) and len(chunk_whisper_features) > 0:
            chunk_whisper_features_numpy_start_time = time.time()
            whisper_np = [w.cpu().numpy() for w in chunk_whisper_features]
            chunk_whisper_features_numpy_end_time = time.time()
            chunk_whisper_features_numpy_time = chunk_whisper_features_numpy_end_time - chunk_whisper_features_numpy_start_time
            logger.info(f"SUCCESS - Chunk whisper features numpy time: {chunk_whisper_features_numpy_time:.2f} s")
        else:
            whisper_np = chunk_whisper_features.cpu().numpy()
        chunk_request_payload_start_time = time.time()
        chunk_request_payload = orjson.dumps({
                   "logid": chunk_id * 1000,
                   "data": {
                       "chunk_faces": faces_np,
                       "chunk_whisper_features": whisper_np,
                       "num_frames": num_frames,
                       "num_inference_steps": num_inference_steps,
                       "guidance_scale": guidance_scale,
                       "eta": eta,
                       "height": height,
                       "width": width
                   }
        }, option=orjson.OPT_SERIALIZE_NUMPY)
        chunk_request_payload_end_time = time.time()
        chunk_request_payload_time = chunk_request_payload_end_time - chunk_request_payload_start_time
        logger.info(f"SUCCESS - Chunk request payload time: {chunk_request_payload_time:.2f} s")
        return chunk_request_payload
    
    def tensor_to_json_compressed(self, tensor, compression='gzip', flag=''):
        """将tensor压缩后转为JSON友好格式"""
        if isinstance(tensor, torch.Tensor):
           tensor_bytes_start_time = time.time()
           tensor = tensor.cpu().numpy()
           tensor_bytes = tensor.tobytes()
           tensor_bytes_end_time = time.time()
           tensor_bytes_time = tensor_bytes_end_time - tensor_bytes_start_time
           logger.info(f"SUCCESS - Tensor bytes time: {flag} {tensor_bytes_time:.2f} s")
        if compression == 'gzip':
           compressed_start_time = time.time()
           compressed = gzip.compress(tensor_bytes)
           compressed_end_time = time.time()
           compressed_time = compressed_end_time - compressed_start_time
           logger.info(f"SUCCESS - Compressed time: {flag} {compressed_time:.2f} s")
        else:
           compressed = tensor_bytes
        encoded_start_time = time.time()
        encoded = base64.b64encode(compressed).decode('utf-8')
        encoded_end_time = time.time()
        encoded_time = encoded_end_time - encoded_start_time
        logger.info(f"SUCCESS - Encoded time: {flag} {encoded_time:.2f} s")
        return {
            'data': encoded,
            'shape': tensor.shape,
            'dtype': str(tensor.dtype),
            'compression': compression
        }
    
    def json_to_tensor_decompressed(self, json_data):
        """从JSON恢复tensor"""
        # 解码
        compressed = base64.b64decode(json_data['data'])
        # 解压
        if json_data.get('compression') == 'gzip':
            tensor_bytes = gzip.decompress(compressed)
        else:
            tensor_bytes = compressed
        # 重建tensor
        np_dtype = np.dtype(json_data['dtype'])
        tensor = np.frombuffer(tensor_bytes, dtype=np_dtype)
        return tensor.reshape(json_data['shape'])


    
    def request(self, request_data, test_id):
        try:
            result = requests.post(url=self.remote_url, data=request_data, timeout=10)  # 添加超时设置
            if result.status_code == 200:
                response_json = result.json()
                if "result" in response_json and response_json["result"] is not None:
                    results = response_json["result"]
                    return {
                       'test_id': test_id,
                       'status': 'SUCCESS',
                       'http_status': result.status_code,
                       'response_data': results
                    }
                else:
                    logger.error(f"[TEST {test_id}] FAILED - No result in response: {response_json}")
                    return {
                        'test_id': test_id,
                        'status': 'FAILED',
                        'http_status': result.status_code,
                        'error_message': f"No result in response: {response_json}"
                    }
            else:
                logger.error(f"[TEST {test_id}] FAILED - HTTP {result.status_code}: {result.text}")
                return {
                    'test_id': test_id,
                    'status': 'FAILED',
                    'http_status': result.status_code,
                    'error_message': result.text
                }
          
        except Exception as e:
            logger.error(f"[TEST {test_id}] FAILED - Exception: {str(e)}")
            return {
                'test_id': test_id,
                'status': 'FAILED',
                'request_time_ms': 0,
                'http_status': 0,
                'error_message': str(e)
            }

        

@deployment(num_replicas=1)
class ModelInferenceLocal(object):
    def __init__(self, ):
        self.config = ModelConfig()
        # 加载UNet配置
        self.unet_config = OmegaConf.load(self.config.unet_config_path)
        # 初始化线程池
        self.pool = ThreadPoolExecutor(max_workers=16)
        # 临时需要
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.config.dtype)
        self.vae.config.scaling_factor = 0.18215
        self.vae.config.shift_factor = 0
        self.vae.to(self.config.device)
        self.image_processor = ImageProcessor(resolution=256, mask="fix_mask", device="cuda", mask_path="latentsync/utils/mask.png")
        self.scheduler = DDIMScheduler.from_pretrained("configs")
        self.unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(self.unet_config.model),
            self.config.inference_ckpt_path,
            device="cpu",
        )
        self.unet = self.unet.to(dtype=self.config.dtype)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.unet.to(self.config.device)
        self._setup_diycache()
        
        # 设置xformers
        if is_xformers_available():
            logger.info("✅ 启用xformers内存高效注意力")
            self.unet.enable_xformers_memory_efficient_attention()
        else:
            logger.warning("⚠️  xformers不可用，跳过内存高效注意力")
    
    def _setup_diycache(self):
        """设置DiyCache加速"""
        logger.info("⚡ 启用DiyCache加速...")
        
        # DiyCache配置参数
        diycache_config = {
            "num_steps": self.config.inference_steps,
            "first_step_offset": self.config.first_step_offset,
            "last_step_offset": self.config.last_step_offset,
        }
        
        # 为UNet启用DiyCache
        self.unet = setup_diycache_unet(self.unet, **diycache_config)
        
        logger.debug(f"  - 第一步偏移: {diycache_config['first_step_offset']}")
        logger.debug(f"  - 最后一步偏移: {diycache_config['last_step_offset']}")
        logger.debug(f"  - 推理步数: {diycache_config['num_steps']}")
    


    
    async def __call__(self,
        num_inferences: int,
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
    ):
        synced_video_frames = []

        futures = []
        
        for i in range(num_inferences):
            # 单chunk推理 - 只传输当前chunk需要的数据
            chunk_faces_chunk = chunk_faces[i]
            chunk_whisper_features_chunk = chunk_whisper_features[i]
            #args = (chunk_faces_chunk, chunk_whisper_features_chunk, num_frames, num_inference_steps, guidance_scale, weight_dtype, eta, generator, callback, callback_steps, height, width, True)
            #future = self.pool.submit(self.model_inference_by_chunk_0701, *args) 
            #futures.append(future)
            decoded_latents = self.model_inference_by_chunk_0701(
                chunk_faces=chunk_faces[i],  # 只传入当前chunk的faces
                chunk_whisper_features=chunk_whisper_features[i],  # 只传入当前chunk的音频特征
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
                debug=True
            )          
            synced_video_frames.append(decoded_latents)
        #for future in futures:
        #    synced_video_frames.append(future.result())
        return synced_video_frames
    
    @torch.no_grad()
    def model_inference_by_chunk_0701(
        self,
        chunk_faces: torch.Tensor,  # 只传入当前chunk的faces, 形状为 [num_frames, C, H, W]
        chunk_whisper_features: List[torch.Tensor] = None,  # 只传入当前chunk的音频特征
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
        debug: bool = False
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
        device = self._execution_device
        batch_size = 1
        
        # 验证输入数据大小
        if chunk_faces.shape[0] != num_frames:
            raise ValueError(f"chunk_faces应该包含{num_frames}帧，但实际包含{chunk_faces.shape[0]}帧")
        if chunk_whisper_features is not None and len(chunk_whisper_features) != num_frames:
            raise ValueError(f"chunk_whisper_features应该包含{num_frames}个特征，但实际包含{len(chunk_whisper_features)}个")
        
        # 设置timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # 准备额外参数
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # 判断是否使用classifier free guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 准备音频嵌入
        audio_embeds = torch.stack(chunk_whisper_features)
        audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
        if do_classifier_free_guidance:
            null_audio_embeds = torch.zeros_like(audio_embeds)
            audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
        
        # 使用传入的chunk faces
        inference_faces = chunk_faces
        
        # 准备latents
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size,
            num_frames,
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )
        
        # 准备masks和masked images
        pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
            inference_faces, affine_transform=False
        )

        # 准备mask latents
        mask_latents, masked_image_latents = self.prepare_mask_latents(
            masks,
            masked_pixel_values,
            height,
            width,
            weight_dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 准备image latents
        image_latents = self.prepare_image_latents(
            pixel_values,
            device,
            weight_dtype,
            generator,
            do_classifier_free_guidance,
        )

        # 去噪循环
        denoising_start_time = time.time()
        total_unet_time = 0.0
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        for j, t in enumerate(timesteps):
            # 扩展latents用于classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # 连接latents, mask, masked_image_latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = torch.cat(
                [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
            )
                
            # 预测噪声
            if debug:
                unet_start_time = time.time()
                    
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample
                
            if debug:
                unet_time = time.time() - unet_start_time
                total_unet_time += unet_time

            # 执行guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

            # 计算前一个noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
            # 调用callback
            if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                if callback is not None and j % callback_steps == 0:
                    callback(j, t, latents)
        
        if debug:
            denoising_total_time = time.time() - denoising_start_time
            avg_unet_time = total_unet_time / num_inference_steps
            print(f"Chunk: 去噪耗时={denoising_total_time:.3f}s, 平均UNet推理耗时={avg_unet_time:.3f}s")

        # 解码latents
        decoded_latents = self.decode_latents(latents)
        decoded_latents = self.paste_surrounding_pixels_back(
            decoded_latents, pixel_values, 1 - masks, device, weight_dtype
        )
        
        return decoded_latents
    
    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents
    
    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values
    
    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents
    
    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    @property
    def _execution_device(self):
        self.device = torch.device("cuda:0")
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device


def restore_video_torch_simple(image_processor, faces, video_frames, boxes, affine_matrices): # original_version
    video_frames = video_frames[: faces.shape[0]]
    out_frames = []
    for index, face in enumerate(faces):
        x1, y1, x2, y2 = boxes[index]
        height = int(y2 - y1)
        width = int(x2 - x1)
        face = torchvision.transforms.functional.resize(face, size=(height, width), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
        out_frame = image_processor.restorer.restore_img_torch_simple(video_frames[index], face, affine_matrices[index])

        out_frames.append(out_frame)
    return np.stack(out_frames, axis=0)

@deployment(num_replicas=2)
class PostprocessVideoBak(object):
    def __init__(self):
        height=256
        mask="fix_mask"
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda", mask_path="latentsync/utils/mask.png")
    
    async def __call__(
        self,
        synced_video_frames: List[torch.Tensor],
        original_video_frames: List,
        boxes: List,
        affine_matrices: List,
        original_faces_len: int,
        num_frames: int = 16
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
        # 合并所有解码后的latents
        merged_frames = torch.cat(synced_video_frames)
        merged_frames = merged_frames[:original_faces_len]
        
        # 按num_frames为chunk大小进行后处理
        all_restored_frames = []
        
        # for i in tqdm.tqdm(range(0, len(merged_frames), num_frames), desc="后处理中..."):
        for i in range(0, len(merged_frames), num_frames):
            chunk_start = i
            chunk_end = min(i + num_frames, len(merged_frames))
            
            # 获取当前chunk的数据
            chunk_decoded_latents = merged_frames[chunk_start:chunk_end]
            chunk_original_frames = original_video_frames[chunk_start:chunk_end]
            chunk_boxes = boxes[chunk_start:chunk_end]
            chunk_affine_matrices = affine_matrices[chunk_start:chunk_end]
            
            # 恢复当前chunk的视频帧
            chunk_restored_frames = restore_video_torch_simple(
                self.image_processor,
                chunk_decoded_latents, chunk_original_frames, chunk_boxes, chunk_affine_matrices
            )
            
            all_restored_frames.append(chunk_restored_frames)
            
        # 拼接所有恢复的帧
        final_video_frames = np.concatenate(all_restored_frames, axis=0)
        
        return final_video_frames

#def restore_video_torch_simple_multi_stream(stream, image_processor, chunk_decoded_latents, chunk_original_frames, chunk_boxes, chunk_affine_matrices):
#    with torch.cuda.stream(stream):
#        chunk_restored_frames = restore_video_torch_simple(image_processor, chunk_decoded_latents, chunk_original_frames, chunk_boxes, chunk_affine_matrices)
#        stream.synchronize()``
        return chunk_restored_frames

@deployment(num_replicas=1)
class PostprocessVideo(object):
    def __init__(self):
        height=256
        mask="fix_mask"
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda", mask_path="latentsync/utils/mask.png")
        self.pool = ThreadPoolExecutor(max_workers=4)
    
    async def __call__(
        self,
        synced_video_frames: List[torch.Tensor],
        original_video_frames: List,
        boxes: List,
        affine_matrices: List,
        original_faces_len: int,
        num_frames: int = 16
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
        # 合并所有解码后的latents
        print('synced_video_frames device: ', synced_video_frames[0].device)
        merged_frames = torch.cat(synced_video_frames)
        merged_frames = merged_frames[:original_faces_len]
        
        # 按num_frames为chunk大小进行后处理
        args_list = []
        futures = []
        all_restored_frames = []
        
        # for i in tqdm.tqdm(range(0, len(merged_frames), num_frames), desc="后处理中..."):
        for i in range(0, len(merged_frames), num_frames):
            chunk_start = i
            chunk_end = min(i + num_frames, len(merged_frames))
            
            # 获取当前chunk的数据
            chunk_decoded_latents = merged_frames[chunk_start:chunk_end]
            chunk_original_frames = original_video_frames[chunk_start:chunk_end]
            chunk_boxes = boxes[chunk_start:chunk_end]
            chunk_affine_matrices = affine_matrices[chunk_start:chunk_end]
            args_list.append((self.image_processor, chunk_decoded_latents, chunk_original_frames, chunk_boxes, chunk_affine_matrices))
            
        for args in args_list:
            future = self.pool.submit(restore_video_torch_simple, *args) 
            futures.append(future)

        for f in futures:
            result = f.result()
            all_restored_frames.append(result)
        # 拼接所有恢复的帧
        final_video_frames = np.concatenate(all_restored_frames, axis=0)
        
        return final_video_frames

@deployment(num_replicas=2)
class MergeVideoAndAudio(object):
    def __init__(self):
        pass # TODO 待实现
    
    async def __call__(
        self, 
        final_video_frames: np.ndarray,
        audio_samples: torch.Tensor,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        convert_audio_to_48k: bool = True):

        # TODO 待重新实现
        temp_dir = tempfile.mkdtemp()
        video_out_path = os.path.join(temp_dir, "generated_video.mp4")
        self.save_final_output(final_video_frames,
                               audio_samples,
                               video_out_path,
                               video_fps,
                               audio_sample_rate,
                               convert_audio_to_48k)
        return video_out_path
    
    def save_final_output(
        self,
        final_video_frames: np.ndarray,
        audio_samples: torch.Tensor,
        video_out_path: str,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        convert_audio_to_48k: bool = True
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
        # 1. 调整音频长度以匹配视频
        audio_samples_remain_length = int(final_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()            
        
        # 2. 使用临时目录处理文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存视频帧和音频到临时文件
            temp_video_path = os.path.join(temp_dir, "video.mp4")
            temp_audio_path = os.path.join(temp_dir, "audio.wav")
            
            # 保存到临时文件
            write_video(temp_video_path, final_video_frames, fps=video_fps) # TODO GPU编码
            sf.write(temp_audio_path, audio_samples, audio_sample_rate) # TODO 2 提前触发
            
            # 3. 音频格式转换（如果需要）
            if convert_audio_to_48k:
                temp_converted_audio_path = os.path.join(temp_dir, "audio_converted.wav")
                cmd = [
                    'ffmpeg', '-i', temp_audio_path,
                    '-acodec', 'pcm_s16le',  # 16比特PCM编码
                    '-ar', '48000',  # 48kHz采样率
                    '-ac', '2',  # 双声道
                    '-y',  # 覆盖输出文件
                    temp_converted_audio_path
                ]
                
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                temp_audio_path = temp_converted_audio_path
            
            # 4. 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(video_out_path)), exist_ok=True)
            
            # 5. 使用ffmpeg命令合并音视频
            if convert_audio_to_48k:
                output_ext = os.path.splitext(video_out_path)[1].lower()
                if output_ext in ['.mkv', '.avi', '.mov']:
                    command = f"ffmpeg -y -loglevel error -nostdin -i {temp_video_path} -i {temp_audio_path} -c:v libx264 -c:a pcm_s16le -ar 48000 -ac 2 -q:v 0 {video_out_path}"
                else:
                    command = f"ffmpeg -y -loglevel error -nostdin -i {temp_video_path} -i {temp_audio_path} -c:v libx264 -c:a aac -ar 48000 -ac 2 -b:a 256k -q:v 0 {video_out_path}"
            else:
                command = f"ffmpeg -y -loglevel error -nostdin -i {temp_video_path} -i {temp_audio_path} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
            
            subprocess.run(command, shell=True)
        return video_out_path            
@deployment(num_replicas=3)
class Uploader(object):
    def __init__(self, environment="live"):
        env.set_cid("sg")
        if environment == "test":
            env.set_environment("test")
            biz = 212
            secret = "1389bc284e60bcc806c2a315c16d7df2"
        else:
            env.set_environment("live")
            biz = 212
            secret = "773a89bd688ec8d108b028fa2b73f41c"
        self.biz = biz
        self.secret = secret
        init_default_config_mgr(biz, secret, skip_scc_init=True)
        self.nus = new_upload_sdk(int(self.biz), self.secret)
        self.pool = ThreadPoolExecutor(max_workers=2)

    async def __call__(self, mask_video_path, video_path):
        results = list(self.pool.map(self.upload_video, [mask_video_path, video_path]))
        mask_vid, mask_url, mask_upload_code = results[0]
        video_vid, video_url, video_upload_code = results[1]
        return (mask_vid, mask_url, mask_upload_code), (video_vid, video_url, video_upload_code)
    
    def upload_video(self, video_path):
        logger.info(f"🔍 上传视频, 开始时间: {time.time()}, path: {video_path}")
        try:
            mediaInfo = MediaInfo(sdk.MEDIA_TYPE_VIDEO, video_path)
            uploadInfo = FullUploadInfo()
            resp = self.nus.full_upload(media_info=mediaInfo, upload_info=uploadInfo)
            logger.info(f"🔍 上传视频, 结束时间: {time.time()}, vid: {resp.vid}, url: {resp.media_url}")
            return resp.vid, resp.media_url, SUCCESS_CODE        
        except Exception as e:
            error_traceback = traceback.format_exc()
            error_lineno = traceback.extract_tb(e.__traceback__)[-1].lineno
            logger.error(f"上传过程中出错: {e}")
            logger.debug(f"📍 上传错误行号: {error_lineno}")
            logger.debug(f"🔍 上传堆栈跟踪:\n{error_traceback}")
            return None, None, UPLOAD_ERROR_CODE

"""
Algorithm API implementation
"""
@deployment(num_replicas=2)
@ingress
class AlgoModel(object):

    def __init__(self):
        """模型初始化"""
        logger.info("🚀 开始初始化 AlgoModel...")
         # 加载配置
        self.config = ModelConfig()
        # 加载UNet配置
        self.unet_config = OmegaConf.load(self.config.unet_config_path)
        # 初始化编排需要的各种算子
        self.audio_feature_actor = AudioFeature.bind()
        self.video_feature_actor = VideoFeature.bind()
        self.model_inference_by_actor = ModelInferenceByRpc.bind()
        self.model_inference_by_actor = ModelInferenceLocal.bind()
        self.postprocess_video_actor = PostprocessVideo.bind()
        self.merge_video_and_audio_actor = MergeVideoAndAudio.bind()
        self.uploader_actor = Uploader.bind()
        
    
    async def __call__(self, ctx):
        start_time = time.time()
        """
        编排逻辑
        """
        # 1. 解析请求数据
        get_request_data_start_time = time.time()
        audio_info = ctx.get_audio_list()[0]
        audio_bytes = audio_info['audio_data']
        get_request_data_end_time = time.time()
        logger.info(f"🔍 获取请求数据时间: {get_request_data_end_time - get_request_data_start_time:.2f}s")
        # 2. 预处理本次请求数据（音频，视频特征，mask，合并mask视频并上传）
        # 2.1 音频特征
        get_audio_feature_start_time = time.time()
        video_fps = 25
        audio_features = await self.audio_feature_actor.remote(audio_bytes, video_fps)
        get_audio_feature_end_time = time.time()
        logger.info(f"🔍 获取音频特征时间: {get_audio_feature_end_time - get_audio_feature_start_time:.2f}s")
        # 2.2 视频特征
        get_video_feature_start_time = time.time()
        video_data = await self.video_feature_actor.remote(audio_features['audio_duration'])
        get_video_feature_end_time = time.time()
        logger.info(f"🔍 获取视频特征时间: {get_video_feature_end_time - get_video_feature_start_time:.2f}s")
        # 2.3 padding
        get_padding_start_time = time.time()
        num_frames = self.config.num_frames
        need_padding_to_16x = True
        padding_data = apply_padding(
            faces = video_data['faces'],
            whisper_chunks = audio_features['whisper_chunks'],
            num_frames = num_frames,
            need_padding_to_16x = need_padding_to_16x
        )
        get_padding_end_time = time.time()
        logger.info(f"🔍 获取padding数据时间: {get_padding_end_time - get_padding_start_time:.2f}s")
        # 2.4 分chunk（chunk_size=16）
        split_data_by_chunk_start_time = time.time()
        preprocess_data = split_data_by_chunk(video_data, audio_features, padding_data, num_frames)
        split_data_by_chunk_end_time = time.time()
        logger.info(f"🔍 分chunk数据时间: {split_data_by_chunk_end_time - split_data_by_chunk_start_time:.2f}s")
         # 异步生成mask视频
        temp_dir = tempfile.mkdtemp()
        mask_output_path = os.path.join(temp_dir, "generated_mask.mp4")
        t1 = threading.Thread(target=gen_mask_video, args=(mask_output_path, preprocess_data['mask_frames'], video_fps, len(preprocess_data['original_video_frames'])))
        t1.start()
        # 3. 推理（并行调用rpc服务，并行推理）
        num_inference_steps=self.config.inference_steps
        guidance_scale=self.config.guidance_scale
        weight_dtype=self.config.dtype
        width=self.unet_config.data.resolution
        height=self.unet_config.data.resolution 
        async_model_inference_start_time = time.time()
        model_inference_task = self.model_inference_by_actor.remote(
            num_inferences=preprocess_data['num_inferences'],
            chunk_faces=preprocess_data['faces_by_chunk'],
            chunk_whisper_features=preprocess_data['whisper_chunks_by_chunk'],
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=weight_dtype,
            width=width,
            height=height
        )
        async_model_inference_end_time = time.time()
        logger.info(f"🔍 异步模型推理时间: {async_model_inference_end_time - async_model_inference_start_time:.2f}s")
        # 等待mask视频生成、等待模型推理完成
        get_mask_video_and_model_inference_start_time = time.time()
        synced_video_frames = await model_inference_task
        missing_chunks = [i for i, frame in enumerate(synced_video_frames) if frame is None]
        if len(missing_chunks) > 0:
            logger.error(f"🔍 模型推理失败, 缺失的chunk: {missing_chunks}")
            ctx.set_result_dict({
                "code": CHUNK_INFERENCE_FAILED_CODE,
                "extra_info": json.dumps({})
            })
            return
        t1.join()
        get_mask_video_and_model_inference_end_time = time.time()
        logger.info(f"🔍 等待mask视频生成、等待模型推理完成时间: {get_mask_video_and_model_inference_end_time - get_mask_video_and_model_inference_start_time:.2f}s")
        # 4. 后处理（贴脸，尽量每16帧batch处理；合并视频音频、上传）
        # 4.1 贴脸
        postprocess_video_start_time = time.time()
        final_video_frames = await self.postprocess_video_actor.remote(
            synced_video_frames=synced_video_frames,
            original_video_frames=preprocess_data['original_video_frames'],
            boxes=preprocess_data['boxes'],
            affine_matrices=preprocess_data['affine_matrices'],
            original_faces_len=preprocess_data['original_faces_len'],
            num_frames=num_frames
        )
        postprocess_video_end_time = time.time()
        logger.info(f"🔍 后处理时间: {postprocess_video_end_time - postprocess_video_start_time:.2f}s")
        # 4.2 合并视频音频
        merge_video_and_audio_start_time = time.time()
        convert_audio_to_48k = True
        audio_sample_rate = 16000
        video_out_path = await self.merge_video_and_audio_actor.remote(
            final_video_frames=final_video_frames,
            audio_samples=preprocess_data['audio_samples'],
            video_fps=video_fps,
            audio_sample_rate=audio_sample_rate,
            convert_audio_to_48k=convert_audio_to_48k
        )
        merge_video_and_audio_end_time = time.time()
        logger.info(f"🔍 合并视频音频时间: {merge_video_and_audio_end_time - merge_video_and_audio_start_time:.2f}s")
        # 4.3 上传视频
        upload_start_time = time.time()
        (mask_vid, mask_url, mask_upload_code), (video_vid, video_url, video_upload_code) = await self.uploader_actor.remote(mask_output_path, video_out_path)
        upload_end_time = time.time()
        logger.info(f"🔍 上传视频时间: {upload_end_time - upload_start_time:.2f}s")
        processing_time = time.time() - start_time
        # 5. 返回结果
        if video_url is not None and video_vid is not None and mask_vid is not None and mask_url is not None:
            generation_info = {
                                "video_url": video_url,
                                "mask_url": mask_url,
                                "video_vid": video_vid,
                                "mask_vid": mask_vid,
                                "generated_frames": preprocess_data['original_faces_len'],
                                "processing_time": round(processing_time, 2),
                                "content_type": "url"
                            }
            result = {
                        "code": SUCCESS_CODE,
                        "extra_info": json.dumps(generation_info)
                    }
            # 获取文件大小用于统计
            video_size = os.path.getsize(video_out_path)
            mask_size = os.path.getsize(mask_output_path)
            logger.info(f"📊 生成统计: {preprocess_data['original_faces_len']}帧, 视频{video_size}字节, 遮罩{mask_size}字节, 生成时间：{processing_time:.2f}s")

        else:
            generation_info = {}
            result = {
                        "code": max(video_upload_code, mask_upload_code),
                        "extra_info": json.dumps(generation_info)
                    }
        ctx.set_result_dict(result)  

def unit_test():
    algo_module_actor = AlgoModel.bind()
    ctx = MockContext()
    # 音频文件路径
    audio_file_path = "/home/work/houyi/latentsync_online_simple/test_case/beauty_v2_13s.wav"
    audio_file_path = "/home/work/houyi/latentsync_online_simple/test_case/beauty_v2_20s.MP3"
    audio_file_path = "/home/work/houyi/latentsync_online_simple_auxilary/input_example/beauty_v2_20s.MP3"
    with open(audio_file_path, 'rb') as f:
        audio_bytes = f.read()

    ctx.set_request({
        'audio_list': [{
            'audio_data': audio_bytes
        }],
    })
    for i in range(10):
        asyncio.run(algo_module_actor.remote(ctx))
        result_dict = ctx.get_result_dict()
        print(result_dict)
    time.sleep(1)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "unit_test":
        unit_test()
