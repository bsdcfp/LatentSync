# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import os
from typing import Callable, List, Optional, Union
import subprocess
import time
import pickle
import gzip

import numpy as np
import torch
import torchvision

from diffusers.utils import is_accelerate_available
from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..utils.image_processor_v01_0516 import ImageProcessor
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

from ..utils.video_sequence_merger import VideoIndexGenerator

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

import tempfile
# import sys
# import types
from latentsync import NVTXContext
from line_profiler import profile


class LipsyncPipeline_online_0701(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        video_index_generator: VideoIndexGenerator,
        lightweight_mode: bool = False,
    ):
        super().__init__()

        # 在轻量级模式下，跳过这些检查以避免None值问题
        if scheduler is not None:
            if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
                deprecation_message = (
                    f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                    f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                    "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                    " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                    " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                    " file"
                )
                deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(scheduler.config)
                new_config["steps_offset"] = 1
                scheduler._internal_dict = FrozenDict(new_config)

            if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
                deprecation_message = (
                    f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                    " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                    " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                    " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                    " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
                )
                deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(scheduler.config)
                new_config["clip_sample"] = False
                scheduler._internal_dict = FrozenDict(new_config)

        if unet is not None:
            is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
                version.parse(unet.config._diffusers_version).base_version
            ) < version.parse("0.9.0.dev0")
            is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
            if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
                deprecation_message = (
                    "The configuration file of the unet has set the default `sample_size` to smaller than"
                    " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                    " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                    " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                    " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                    " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                    " in the config might lead to incorrect results in future versions. If you have downloaded this"
                    " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                    " the `unet/config.json` file"
                )
                deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
                new_config = dict(unet.config)
                new_config["sample_size"] = 64
                unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
            video_index_generator=video_index_generator,
        )

        # 保存轻量级模式标志
        self.lightweight_mode = lightweight_mode

        # 在轻量级模式下，vae可能为None，设置一个默认值
        if self.vae is not None:
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = 8  # 默认值，用于轻量级模式

        self.set_progress_bar_config(desc="Steps")

        height=256
        mask="fix_mask"
        mask_path = os.path.join(os.path.dirname(__file__), "..", "utils", "mask.png")
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda", mask_path=mask_path)
        self.video_index_generator = video_index_generator
        # video_index_generator = VideoIndexGenerator(
        #     connected_json_path=args.connected_info_json,
        #     video_info_json_path=args.video_info_json,
        #     video_template_dir=args.video_template_dir,
        #     target_fps=args.target_fps,
        #     transition_frames=args.transition_frames,
        #     debug=args.debug_video_merger
        # )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
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

    @NVTXContext
    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

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

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    @NVTXContext
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

    @NVTXContext
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

    @NVTXContext
    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    @NVTXContext
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, use_decord=False, change_fps=False)
        # debugging
        # video_frames = video_frames[:128]
        print(f"video_frames: {len(video_frames)}")
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        return faces, video_frames, boxes, affine_matrices

    # def restore_video(self, faces, video_frames, boxes, affine_matrices): # original_version
    #     video_frames = video_frames[: faces.shape[0]]
    #     out_frames = []
    #     print(f"Restoring {len(faces)} faces...")
    #     for index, face in enumerate(tqdm.tqdm(faces)):
    #         x1, y1, x2, y2 = boxes[index]
    #         height = int(y2 - y1)
    #         width = int(x2 - x1)
    #         face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
    #         # 保持face为torch.Tensor格式，不转换为numpy数组
    #         face = rearrange(face, "c h w -> h w c")
    #         face = (face / 2 + 0.5).clamp(0, 1)
    #         face = (face * 255).to(torch.uint8).cpu().numpy()
            
    #         # face = cv2.resize(face, (width, height), interpolation=cv2.INTER_LANCZOS4)
    #         # out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
    #         out_frame = self.image_processor.restorer.restore_img_torch(video_frames[index], face, affine_matrices[index])
    #         # out_frame = self.image_processor.restorer.restore_img_torch_v2_hangge(video_frames[index], face, affine_matrices[index])
    #         out_frames.append(out_frame)
    #     return np.stack(out_frames, axis=0)

    @NVTXContext
    def restore_video_torch_simple(self, faces, video_frames, boxes, affine_matrices): # original_version
        video_frames = video_frames[: faces.shape[0]]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(face, size=(height, width), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
            # out_frame = self.image_processor.restorer.restore_img_torch_v01_0516_fixed(video_frames[index], face, affine_matrices[index])
            out_frame = self.image_processor.restorer.restore_img_torch_simple(video_frames[index], face, affine_matrices[index])

            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    # new version - 拆分成视频和音频前处理
    @NVTXContext
    def preprocess_video_data(
        self,
        video_cache_dir: str,
        audio_path: str,
        mask_base_dir: str = None,
        debug: bool = False,
        **kwargs,
    ) -> dict:
        """
        视频数据前处理：加载视频特征、原始帧等（可在CPU上运行）
        
        Returns:
            dict: 包含视频相关的预处理数据
        """
        device = self._execution_device
        
        # 获取音频时长
        audio_samples, audio_duration = read_audio(audio_path, return_duration=True)
        if debug:
            print(f"音频时长: {audio_duration} 秒")

        # 获取视频索引列表
        video_index_list = self.video_index_generator.get_video_index(audio_path, audio_duration=audio_duration)

        if debug:
            print(f"video_index_list: {video_index_list}")
            print(f"📋 获取的视频索引列表:")
            total_frames = 0
            for item in video_index_list:
                frame_count = len(item['frame_indices'])
                total_frames += frame_count
                print(f"  - {item['video_name']}: {frame_count} 帧")
            print(f"  📊 总计: {total_frames} 帧")

        # 加载视频帧和特征数据
        faces, original_video_frames, boxes, affine_matrices, mask_frames = self._load_video_and_mask_data_from_index_list(
            video_index_list, device, video_cache_dir, mask_base_dir, debug=debug
        )
        original_faces_len = len(faces)
        
        if debug:
            print(f"✅ 成功加载视频数据，faces形状: {faces.shape}")
            print(f"✅ 成功加载视频数据，original_video_frames形状: {len(original_video_frames)}")
            if mask_frames is not None:
                print(f"✅ 成功加载mask数据，mask_frames形状: {len(mask_frames)}")

        return {
            'faces': faces,
            'original_video_frames': original_video_frames,
            'boxes': boxes,
            'affine_matrices': affine_matrices,
            'mask_frames': mask_frames,
            'audio_samples': audio_samples,
            'video_index_list': video_index_list,
            'original_faces_len': original_faces_len,
        }

    @NVTXContext
    def preprocess_audio_features(
        self,
        audio_path: str,
        video_fps: int = 25,
        debug: bool = False,
        **kwargs,
    ) -> dict:
        """
        音频特征提取：只处理音频特征，与faces无关（可能需要GPU）
        
        Args:
            audio_path: 音频路径
            video_fps: 视频帧率
            debug: 调试模式
            
        Returns:
            dict: 包含音频特征数据
        """
        whisper_chunks = None
        
        if self.lightweight_mode or (self.unet is not None and self.unet.add_audio_layer):
            # 音频特征提取（可能需要GPU）
            if debug:
                print(f"🎵 开始音频特征提取...")
            whisper_feature = self.audio_encoder.audio2feat(audio_path)
            whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
            
            if debug:
                print(f"✅ 音频特征提取完成，共{len(whisper_chunks)}个特征")
        else:
            if debug:
                print(f"🎵 跳过音频特征提取（模型不使用音频层）")

        return {
            'whisper_chunks': whisper_chunks,
        }

    @NVTXContext
    def apply_padding(
        self,
        faces: torch.Tensor,
        whisper_chunks: Optional[List[torch.Tensor]],
        num_frames: int = 16,
        need_padding_to_16x: bool = True,
        debug: bool = False,
        **kwargs,
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
        
        if (self.lightweight_mode or (self.unet is not None and self.unet.add_audio_layer)) and whisper_chunks is not None:
            if need_padding_to_16x:
                original_audio_len = len(whisper_chunks)
                remainder = original_audio_len % num_frames
                padding_frames = (num_frames - remainder) % num_frames
                
                if debug:
                    print(f"🔧 音频padding: original_audio_len={original_audio_len}, remainder={remainder}, padding_frames={padding_frames}")
                
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
                
                if debug:
                    print(f"🔧 faces padding: faces_len={faces_len}, required_frames={required_frames}")
                
                # 对faces进行padding或截取
                if faces_len >= required_frames:
                    faces = faces[:required_frames]
                    if debug:
                        print(f"faces足够长，截取到{required_frames}帧")
                else:
                    faces_padding_needed = required_frames - faces_len
                    if debug:
                        print(f"faces不够长，需要padding {faces_padding_needed}帧")
                    
                    face_last_frame = faces[-1:] if faces_len > 0 else faces[:1]
                    faces_padding_frames = face_last_frame.repeat(faces_padding_needed, 1, 1, 1)
                    faces = torch.cat([faces, faces_padding_frames], dim=0)
                    
                    if debug:
                        print(f"完成faces padding，最终形状: {faces.shape}")
            else:
                num_inferences = min(len(faces), len(whisper_chunks)) // num_frames
                faces = faces[:num_inferences * num_frames]
                whisper_chunks = whisper_chunks[:num_inferences * num_frames]
        else:
            # 没有音频特征的情况
            num_inferences = len(faces) // num_frames
            faces = faces[:num_inferences * num_frames]

        gen_frame_num = num_inferences * num_frames
        if debug:
            print(f"✅ Padding完成: num_inferences={num_inferences}, gen_frame_num={gen_frame_num}, padding_frames={padding_frames}")

        return {
            'faces': faces,  # padding后的faces
            'whisper_chunks': whisper_chunks,  # padding后的whisper特征
            'num_inferences': num_inferences,
            'gen_frame_num': gen_frame_num,
            'padding_frames': padding_frames
        }



    @torch.no_grad()
    @NVTXContext
    def preprocess_data_0701(
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
        if debug:
            print("🔄 开始前处理（四步骤整合版本）...")
            
        # 子任务1: 取视频特征（可在CPU上）
        if debug:
            print("📹 步骤1: 取视频特征...")
        video_data = self.preprocess_video_data(
            video_cache_dir=video_cache_dir,
            audio_path=audio_path,
            mask_base_dir=mask_base_dir,
            debug=debug,
            **kwargs
        )
        
        # 子任务2: 音频特征提取（可能需要GPU）
        if debug:
            print("🎵 步骤2: 音频特征提取...")
        audio_features = self.preprocess_audio_features(
            audio_path=audio_path,
            video_fps=video_fps,
            debug=debug,
            **kwargs
        )
        
        # 子任务3: 做padding（faces和whisper特征一起处理）
        if debug:
            print("🔧 步骤3: 应用padding...")
        padding_data = self.apply_padding(
            faces=video_data['faces'],
            whisper_chunks=audio_features['whisper_chunks'],
            num_frames=num_frames,
            need_padding_to_16x=need_padding_to_16x,
            debug=debug,
            **kwargs
        )
        
        # 子任务4: 数据分chunk（整合到前处理中）
        if debug:
            print("🔪 步骤4: 数据分chunk...")
            print(f"   开始数据分chunk，num_inferences: {padding_data['num_inferences']}, num_frames: {num_frames}")
        
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
        
        if debug:
            print(f"   ✅ 数据分chunk完成，共{len(faces_by_chunk)}个face chunks，{len(whisper_chunks_by_chunk)}个audio chunks")
        
        # 调试信息
        if debug:
            print("✅ 前处理（四步骤整合版本）完成")
            print(f"   - 视频帧数: {len(video_data['original_video_frames'])}")
            print(f"   - Faces形状: {padding_data['faces'].shape}")
            print(f"   - 推理chunks数: {padding_data['num_inferences']}")
            print(f"   - 生成帧数: {padding_data['gen_frame_num']}")
            print(f"   - Chunk数量: {len(faces_by_chunk)}")
        
        # 返回具体的键值对，便于阅读和理解
        return {
            # 视频数据相关
            # 'faces': padding_data['faces'],  # padding处理后的faces（用于兼容性）
            'original_video_frames': video_data['original_video_frames'],
            'boxes': video_data['boxes'],
            'affine_matrices': video_data['affine_matrices'],
            'mask_frames': video_data['mask_frames'],
            'audio_samples': video_data['audio_samples'],
            'video_index_list': video_data['video_index_list'],
            'original_faces_len': video_data['original_faces_len'],
            
            # 音频特征相关
            # 'whisper_chunks': padding_data['whisper_chunks'],  # padding处理后的whisper特征（用于兼容性）
            
            # padding处理结果
            'num_inferences': padding_data['num_inferences'],
            'gen_frame_num': padding_data['gen_frame_num'],
            'padding_frames': padding_data['padding_frames'],
            
            # 分chunk后的数据（新增）
            'faces_by_chunk': faces_by_chunk,  # List[torch.Tensor], 每个元素形状为 [num_frames, C, H, W]
            'whisper_chunks_by_chunk': whisper_chunks_by_chunk,  # List[List[torch.Tensor]], 每个chunk的音频特征
        }

    # @profile
    @torch.no_grad()
    @NVTXContext
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
        audio_embeds = None
        if (self.lightweight_mode or (self.unet is not None and self.unet.add_audio_layer)) and chunk_whisper_features is not None:
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
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
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
                
                # 在轻量级模式下，UNet为None，不能进行推理
                if self.lightweight_mode:
                    raise RuntimeError("Cannot perform inference in lightweight mode: UNet is None. This pipeline is only for preprocessing and postprocessing.")
                    
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
                    progress_bar.update()
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

    @torch.no_grad()
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
        if debug:
            print("🎬 开始后处理...")
        
        # 合并所有解码后的latents
        merged_frames = torch.cat(synced_video_frames)
        merged_frames = merged_frames[:original_faces_len]
        
        if debug:
            print(f"   合并后帧数: {len(merged_frames)}")
        
        # 按num_frames为chunk大小进行后处理
        all_restored_frames = []
        
        for i in tqdm.tqdm(range(0, len(merged_frames), num_frames), desc="后处理中..."):
            chunk_start = i
            chunk_end = min(i + num_frames, len(merged_frames))
            
            # 获取当前chunk的数据
            chunk_decoded_latents = merged_frames[chunk_start:chunk_end]
            chunk_original_frames = original_video_frames[chunk_start:chunk_end]
            chunk_boxes = boxes[chunk_start:chunk_end]
            chunk_affine_matrices = affine_matrices[chunk_start:chunk_end]
            
            # 恢复当前chunk的视频帧
            chunk_restored_frames = self.restore_video_torch_simple(
                chunk_decoded_latents, chunk_original_frames, chunk_boxes, chunk_affine_matrices
            )
            
            all_restored_frames.append(chunk_restored_frames)
            
            if debug:
                print(f"   完成chunk {i//num_frames + 1}, 帧数: {len(chunk_restored_frames)}")
        
        # 拼接所有恢复的帧
        final_video_frames = np.concatenate(all_restored_frames, axis=0)
        
        if debug:
            print(f"✅ 后处理完成，最终视频帧数: {len(final_video_frames)}")
        
        return final_video_frames

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
        if debug:
            print("🎵 开始处理音频和输出...")
        
        # 1. 调整音频长度以匹配视频
        audio_samples_remain_length = int(final_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()
        
        # 2. 使用临时目录处理文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存视频帧和音频到临时文件
            temp_video_path = os.path.join(temp_dir, "video.mp4")
            temp_audio_path = os.path.join(temp_dir, "audio.wav")
            
            # 保存到临时文件
            write_video(temp_video_path, final_video_frames, fps=video_fps)
            sf.write(temp_audio_path, audio_samples, audio_sample_rate)
            
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
                if debug:
                    print(f"✅ 48khz音频转换完成")
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
                    if debug:
                        print(f"⚠️  注意: {output_ext}容器不支持16比特PCM格式，使用高质量AAC (48kHz双声道256kbps)")
            else:
                command = f"ffmpeg -y -loglevel error -nostdin -i {temp_video_path} -i {temp_audio_path} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
            
            subprocess.run(command, shell=True)
            
            if debug:
                print(f"✅ 主视频已保存到: {video_out_path}")
        
        # 6. 处理mask视频输出（如果需要）
        if mask_output_path is not None and mask_frames is not None:
            mask_frames_output = mask_frames[:len(final_video_frames)]
            os.makedirs(os.path.dirname(os.path.abspath(mask_output_path)), exist_ok=True)
            write_video(mask_output_path, mask_frames_output, fps=video_fps)
            
            if debug:
                print(f"✅ Mask视频已保存到: {mask_output_path}")

    @torch.no_grad()
    @NVTXContext
    def generate_video_online_0701(
        self,
        video_cache_dir: str ,
        audio_path: str,
        video_out_path: str,
        mask_output_path: str = None,  # 新增：mask输出路径
        mask_base_dir: str = None,  # 新增：mask视频基础目录
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        need_padding_to_16x=True,
        convert_auido_to_48k=True,
        debug=False,  # 新增debug参数
        **kwargs,
    ):
        time_begin = time.time()
        is_train = self.unet.training
        self.unet.eval()

        # 设置默认高度和宽度（从主函数确定）
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        device = self._execution_device

        # 1. 前处理：调用完整的四步骤前处理函数
        print("🔄 开始完整前处理...")
        preprocess_data = self.preprocess_data_0701(
            video_cache_dir=video_cache_dir,
            audio_path=audio_path,
            mask_base_dir=mask_base_dir,
            num_frames=num_frames,
            video_fps=video_fps,
            need_padding_to_16x=need_padding_to_16x,
            debug=debug,
            **kwargs
        )
        
        # 提取预处理数据（直接从前处理结果获取）
        faces_by_chunk = preprocess_data['faces_by_chunk']  # List[torch.Tensor]
        whisper_chunks_by_chunk = preprocess_data['whisper_chunks_by_chunk']  # List[List[torch.Tensor]]
        print(f"✅ 完整前处理完成，准备进行 {preprocess_data['num_inferences']} 个chunks的推理")

        # 2. 模型推理：逐chunk处理
        print("🤖 开始模型推理...")
        synced_video_frames = []

        for i in tqdm.tqdm(range(preprocess_data['num_inferences']), desc="模型推理中..."):
            chunk_start_time = time.time()
            
            # 单chunk推理 - 只传输当前chunk需要的数据
            decoded_latents = self.model_inference_by_chunk_0701(
                chunk_faces=faces_by_chunk[i],  # 只传入当前chunk的faces
                chunk_whisper_features=whisper_chunks_by_chunk[i],  # 只传入当前chunk的音频特征
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
            
            if debug:
                chunk_time = time.time() - chunk_start_time
                print(f"chunks {i}/{preprocess_data['num_inferences']} 完成，耗时: {chunk_time:.3f}s")

        print("✅ 模型推理完成")

        # 3. 后处理：整合的视频恢复
        final_video_frames = self.postprocess_video(
            synced_video_frames=synced_video_frames,
            original_video_frames=preprocess_data['original_video_frames'],
            boxes=preprocess_data['boxes'],
            affine_matrices=preprocess_data['affine_matrices'],
            original_faces_len=preprocess_data['original_faces_len'],
            num_frames=num_frames,
            debug=debug,
            **kwargs
        )

        # 4. 保存最终输出
        self.save_final_output(
            final_video_frames=final_video_frames,
            audio_samples=preprocess_data['audio_samples'],
            video_out_path=video_out_path,
            mask_frames=preprocess_data['mask_frames'],
            mask_output_path=mask_output_path,
            video_fps=video_fps,
            audio_sample_rate=audio_sample_rate,
            convert_audio_to_48k=convert_auido_to_48k,
            debug=debug,
            **kwargs
        )

        # 恢复训练模式
        if is_train:
            self.unet.train()

        time_end = time.time()
        print(f"🎉 视频生成完成，耗时: {time_end - time_begin:.2f}s, 音频模板：{audio_path}")
        return preprocess_data['gen_frame_num'], preprocess_data['video_index_list']

    @NVTXContext
    def _load_video_and_mask_data_from_index_list(self, video_index_list: List[dict], device, video_cache_dir = "/home/work/houyi/templates/0612_online_templete_female_32fps_features", mask_base_dir: str = None, debug=False):
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
        # video_cache_dir = "/home/work/houyi/templates/0612_online_templete_female_32fps_features"
        # video_cache_dir = video_cache_dir
        
        all_faces = []
        all_original_frames = []
        all_boxes = []
        all_affine_matrices = []
        all_mask_frames = []
        
        if debug:
            print(f"🎬 开始从索引列表加载视频数据...")
            # 统计总的预期帧数
        total_expected_frames = sum(len(item['frame_indices']) for item in video_index_list)
        if debug:
            print(f"📊 预期加载总帧数: {total_expected_frames}")
        
        for video_info in video_index_list:
            video_name = video_info["video_name"]
            frame_indices = video_info["frame_indices"]
            video_path = video_info["video_path"]
            
            if debug:
                print(f"📹 处理视频: {video_name}")
                print(f"   需要帧数: {len(frame_indices)}")
                print(f"   帧索引范围: {min(frame_indices)} - {max(frame_indices)}")
            
            # 1. 加载缓存的特征数据

            cache_file = os.path.join(video_cache_dir, f"{video_name}_features.pkl")
            
            if not os.path.exists(cache_file):
                if debug:
                    print(f"⚠️  警告: 特征缓存文件不存在: {cache_file}")
                    print(f"   请先运行特征提取生成缓存文件")
                continue
                
            # 加载缓存的特征
            if debug:
                print(f"🔄 加载缓存特征: {cache_file}")
            with gzip.open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cached_faces = cached_data['faces']
                if torch.is_tensor(cached_faces):
                    cached_faces = cached_faces.to(device).float()
                cached_boxes = cached_data['boxes']
                cached_affine_matrices = cached_data['affine_matrices']
            
            # 2. 读取原始视频帧
            if debug:
                print(f"🎬 读取原始视频帧: {video_path}")
            if not os.path.exists(video_path):
                if debug:
                    print(f"⚠️  警告: 视频文件不存在: {video_path}")
                continue
            
            original_video_frames = read_video(video_path, use_decord=False, change_fps=False)
            # print(f"video_path: {video_path}: len original_video_frames: {len(original_video_frames)}, original_video_frames[0].shape: {original_video_frames[0].shape}, original_video_frames[0].dtype: {original_video_frames[0].dtype}")

            # 3. 如果需要加载mask视频
            mask_video_frames = None
            if mask_base_dir is not None:
                mask_video_name = f"{video_name}_mask.mp4"
                mask_video_path = os.path.join(mask_base_dir, mask_video_name)
                
                if os.path.exists(mask_video_path):
                    if debug:
                        print(f"🎭 读取mask视频帧: {mask_video_path}")
                    mask_video_frames = read_video(mask_video_path, use_decord=False, change_fps=False)
                    # print(f"mask_video_path: {mask_video_path}: len mask_video_frames: {len(mask_video_frames)}, mask_video_frames[0].shape: {mask_video_frames[0].shape}, mask_video_frames[0].dtype: {mask_video_frames[0].dtype}")

                else:
                    # if debug:
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
                        if debug:
                            print(f"⚠️  警告: mask帧索引 {idx} 超出范围, len(mask_video_frames): {len(mask_video_frames) if mask_video_frames else 0}")
                else:
                    # if debug:
                    print(f"⚠️  警告: 帧索引 {idx} 超出范围, len(cached_faces): {len(cached_faces)}, original_video_frames: {len(original_video_frames)}, cache_file: {cache_file}, video_path: {video_path}")
            
            if debug:
                print(f"   ✅ 成功提取 {len(selected_faces)} 帧数据")
                if mask_base_dir is not None:
                    print(f"   ✅ 成功提取 {len(selected_mask_frames)} mask帧数据")
            
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
        
        if debug:
            print(f"✅ 总共加载了 {len(all_faces)} 帧数据")
            print(f"   - faces形状: {faces.shape}")
            print(f"   - 原始帧数: {len(all_original_frames)}")
            print(f"   - boxes数量: {len(all_boxes)}")
            print(f"   - 仿射矩阵数量: {len(all_affine_matrices)}")
            if mask_base_dir is not None:
                print(f"   - mask帧数量: {len(all_mask_frames)}")
        
        return faces, all_original_frames, all_boxes, all_affine_matrices, all_mask_frames