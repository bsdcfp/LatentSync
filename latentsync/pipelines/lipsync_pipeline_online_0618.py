# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess
import time
import pickle
import hashlib
import gzip

import numpy as np
import torch
import torchvision

from diffusers.utils import is_accelerate_available
from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
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
import cv2

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
from simpleprofiler.profiler import NVTXContext

class LipsyncPipeline_online_0618(DiffusionPipeline):
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
    ):
        super().__init__()

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
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

        height=256
        mask="fix_mask"
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda", mask_path="latentsync/utils/mask.png")
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

    # @torch.no_grad()
    @torch.no_grad()
    #@profile
    @NVTXContext
    def generate_video_online_0618(
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

        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")
        
        # 获取音频时长
        with NVTXContext(f"{self.__class__.__qualname__}.read_audio"):
                audio_samples, audio_duration = read_audio(audio_path, return_duration=True)
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
        else:
            pass

        # 通过video_index_list加载视频帧和特征
        faces, original_video_frames, boxes, affine_matrices, mask_frames = self._load_video_and_mask_data_from_index_list(
            video_index_list, device, video_cache_dir, mask_base_dir if mask_output_path is not None else None, debug=debug
        )
        original_faces_len = len(faces)
        # print(f"original_faces_len: {original_faces_len}")
        
        if debug:
            print(f"✅ 成功加载视频数据，faces形状: {faces.shape}")
            print(f"✅ 成功加载视频数据，original_video_frames形状: {len(original_video_frames)}, {original_video_frames[0].shape}")
            print(f"✅ 成功加载视频数据，boxes形状: {len(boxes)}")
            print(f"✅ 成功加载视频数据，affine_matrices形状: {len(affine_matrices)}")
            if mask_frames is not None:
                print(f"✅ 成功加载mask数据，mask_frames形状: {len(mask_frames)}")
        
                
        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        # self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self.video_fps = video_fps

        with NVTXContext(f"{self.__class__.__qualname__}.prepare_audio_chunks"):
            padding_frames = 0
            if self.unet.add_audio_layer:
                whisper_feature = self.audio_encoder.audio2feat(audio_path)
                whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
                if need_padding_to_16x:
                    # 计算需要补齐的帧数
                    original_audio_len = len(whisper_chunks)
                    remainder = original_audio_len % num_frames
                    padding_frames = (num_frames - remainder) % num_frames # 0-15
                    if debug:
                        print(f"original_audio_len: {original_audio_len}, remainder: {remainder}, padding_frames: {padding_frames}")
                    if padding_frames != 0:
                        # 创建补零的tensor
                        # padding_audio_tensor = torch.zeros_like(whisper_chunks[0]).unsqueeze(0).repeat(padding_frames, 1)
                        padding_method = "last_frame_padding"
                        if padding_method == "zero_padding":
                        # method1
                            padding_audio_tensor = [torch.zeros_like(whisper_chunks[0]) for _ in range(padding_frames)]
                        # method2
                        elif padding_method == "last_frame_padding":
                            padding_audio_tensor = [whisper_chunks[-1]] * padding_frames
                        else:
                            raise NotImplementedError(f"padding_method {padding_method} not implemented!")
                        # 将补零的tensor添加到原始chunks中
                        # whisper_chunks = torch.cat([whisper_chunks, padding_audio_tensor], dim=0)
                        whisper_chunks += padding_audio_tensor

                    # 确保无论是否需要padding都定义num_inferences
                    num_inferences = len(whisper_chunks) // num_frames
                    faces_len = len(faces)
                    required_frames = num_inferences * num_frames
                    
                    if faces_len >= required_frames:
                        # 如果faces够长，直接截取
                        faces = faces[:required_frames]
                        if debug:
                            print(f"faces足够长，截取到{required_frames}帧")
                    else:
                        # 如果faces不够长，需要做padding
                        faces_padding_needed = required_frames - faces_len
                        if debug:
                            print(f"faces不够长: faces_len={faces_len}, required_frames={required_frames}, 需要padding {faces_padding_needed}帧")
                        
                        # 使用最后一帧重复填充的方式进行padding
                        face_last_frame = faces[-1:] if faces_len > 0 else faces[:1]  # 获取最后一帧，保持维度
                        faces_padding_frames = face_last_frame.repeat(faces_padding_needed, 1, 1, 1)  # 重复最后一帧
                        faces = torch.cat([faces, faces_padding_frames], dim=0)
                        
                        if debug:
                            print(f"完成faces padding，最终形状: {faces.shape}")
                else:
                    num_inferences = min(len(faces), len(whisper_chunks)) // num_frames
                    faces = faces[:num_inferences * num_frames]
                
            else:
                num_inferences = len(faces) // num_frames
                faces = faces[:num_inferences * num_frames]
        
        # faces = faces[:num_inferences * num_frames]
        gen_frame_num = num_inferences * num_frames
        if debug:
            print(f"num_inferences: {num_inferences}, gen_frame_num: {gen_frame_num}, padding_frames: {padding_frames}")

        synced_video_frames = []
        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        all_latents = self.prepare_latents(
            batch_size,
            num_frames * num_inferences,
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        # 初始化总体统计变量
        if debug:
            total_batch_denoising_time = 0.0
            total_batch_unet_time = 0.0
        
        inference_start_time = time.time()
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            with NVTXContext(f"{self.__class__.__qualname__}.doing_inference_{i}"):
                if self.unet.add_audio_layer:
                    audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                    audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                    if do_classifier_free_guidance:
                        null_audio_embeds = torch.zeros_like(audio_embeds)
                        audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
                else:
                    audio_embeds = None
                inference_faces = faces[i * num_frames : (i + 1) * num_frames]
                latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
                pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                    inference_faces, affine_transform=False
                )

                # 7. Prepare mask latent variables
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

                # 8. Prepare image latents
                image_latents = self.prepare_image_latents(
                    pixel_values,
                    device,
                    weight_dtype,
                    generator,
                    do_classifier_free_guidance,
                )

                # 9. Denoising loop
                denoising_start_time = time.time()
                total_unet_time = 0.0
                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    with NVTXContext(f"{self.__class__.__qualname__}.denoising_loop_{i}"):
                        for j, t in enumerate(timesteps):
                            
                            # expand the latents if we are doing classifier free guidance
                            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                            # concat latents, mask, masked_image_latents in the channel dimension
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                            latent_model_input = torch.cat(
                                [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
                            )
                            # predict the noise residual
                            if debug:
                                unet_start_time = time.time()
                            # print(f"num_inference_steps: {num_inference_steps}, latent_model_input: {latent_model_input.shape}, latent_model_input.dtype: {latent_model_input.dtype}, t: {t}, audio_embeds: {audio_embeds.shape}, audio_embeds.dtype: {audio_embeds.dtype}")
                            with NVTXContext(f"{self.__class__.__qualname__}.unet_forward_{i}_{j}"):
                                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample
                            # print(f"noise_pred: {noise_pred.shape}, noise_pred.dtype: {noise_pred.dtype}")
                            if debug:
                                unet_time = time.time() - unet_start_time
                                total_unet_time += unet_time

                            # perform guidance
                            if do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                            # compute the previous noisy sample x_t -> x_t-1
                            with NVTXContext(f"{self.__class__.__qualname__}.scheduler_step_{i}_{j}"):
                                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                            
                            # call the callback, if provided
                            if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                                progress_bar.update()
                                if callback is not None and j % callback_steps == 0:
                                    callback(j, t, latents)
                if debug:
                    denoising_total_time = time.time() - denoising_start_time

                    # 累积到总体统计
                    total_batch_denoising_time += denoising_total_time
                    total_batch_unet_time += total_unet_time

                # Recover the pixel values
                decoded_latents = self.decode_latents(latents)
                decoded_latents = self.paste_surrounding_pixels_back(
                    decoded_latents, pixel_values, 1 - masks, device, weight_dtype
                )
                synced_video_frames.append(decoded_latents)
        inference_end_time = time.time()
        print(f"🎯 推理耗时: {inference_end_time - inference_start_time:.3f}s")

        # 打印所有批次的总体统计
        if num_inferences > 0 and debug:
            overall_avg_denoising_time = total_batch_denoising_time / num_inferences
            overall_avg_unet_time = total_batch_unet_time / (num_inferences * num_inference_steps)
            print(f"🎯 所有批次统计: 共{num_inferences}个批次, 总去噪耗时={total_batch_denoising_time:.3f}s, 总UNet推理耗时={total_batch_unet_time:.3f}s, 平均去噪耗时={overall_avg_denoising_time:.3f}s, 平均UNet推理耗时={overall_avg_unet_time:.3f}s")

        synced_video_frames = torch.cat(synced_video_frames)
        # if debug:
        # print(f"synced_video_frames: {len(synced_video_frames)}, original_video_frames: {len(original_video_frames)}, boxes: {len(boxes)}, affine_matrices: {len(affine_matrices)}, original_faces_len: {original_faces_len}")
        synced_video_frames = synced_video_frames[:original_faces_len]
        # print(f"synced_video_frames: {synced_video_frames[0].shape} {synced_video_frames[0].dtype}, original_video_frames: {original_video_frames[0].shape} {original_video_frames[0].dtype}")
        synced_video_frames = self.restore_video_torch_simple(
            synced_video_frames, original_video_frames, boxes, affine_matrices
        )
        # print(f"synced_video_frames: {synced_video_frames[0].shape} {synced_video_frames[0].dtype}")

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存视频帧和音频到临时文件
            temp_video_path = os.path.join(temp_dir, "video.mp4")
            temp_audio_path = os.path.join(temp_dir, "audio.wav")
            
            # 保存到临时文件
            write_video(temp_video_path, synced_video_frames, fps=video_fps) # 保存视频帧 video_fps=25
            sf.write(temp_audio_path, audio_samples, audio_sample_rate)

            if convert_auido_to_48k: 
                temp_converted_audio_path = os.path.join(temp_dir, "audio_converted.wav")
                cmd = [
                    'ffmpeg', '-i', temp_audio_path,
                    '-acodec', 'pcm_s16le',  # 16比特PCM编码
                    '-ar', '48000',  # 48kHz采样率
                    '-ac', '2',  # 双声道
                    '-y',  # 覆盖输出文件
                    temp_converted_audio_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"✅ 48khz音频转换完成: {temp_converted_audio_path}")
                temp_audio_path = temp_converted_audio_path


            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(video_out_path)), exist_ok=True)
            
            # 使用ffmpeg命令合并（最可靠的方法）
            if convert_auido_to_48k:
                # 根据输出文件扩展名选择合适的音频编码
                output_ext = os.path.splitext(video_out_path)[1].lower()
                if output_ext in ['.mkv', '.avi', '.mov']:
                    # 这些容器支持PCM，保持16比特PCM音频格式
                    command = f"ffmpeg -y -loglevel error -nostdin -i {temp_video_path} -i {temp_audio_path} -c:v libx264 -c:a pcm_s16le -ar 48000 -ac 2 -q:v 0 {video_out_path}"
                else:
                    # MP4等容器不支持PCM，使用高质量AAC但保持48kHz双声道
                    command = f"ffmpeg -y -loglevel error -nostdin -i {temp_video_path} -i {temp_audio_path} -c:v libx264 -c:a aac -ar 48000 -ac 2 -b:a 256k -q:v 0 {video_out_path}"
                    print(f"⚠️  注意: {output_ext}容器不支持16比特PCM格式，使用高质量AAC (48kHz双声道256kbps)")
            else:
                # 使用AAC编码
                command = f"ffmpeg -y -loglevel error -nostdin -i {temp_video_path} -i {temp_audio_path} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
            subprocess.run(command, shell=True)
            
        # 如果需要输出mask视频，直接保存到目标位置
        if mask_output_path is not None and mask_frames is not None:
            # 截取与synced_video_frames相同长度的mask帧
            mask_frames_output = mask_frames[:len(synced_video_frames)]
            
            # 确保mask输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(mask_output_path)), exist_ok=True)
            
            # 直接保存mask视频到目标位置（不需要音频）
            write_video(mask_output_path, mask_frames_output, fps=video_fps)
            
            if debug:
                print(f"✅ Mask视频已保存到: {mask_output_path}")

        time_end = time.time()
        print(f"🎉 视频生成完成，耗时: {time_end - time_begin:.2f}s, 音频模板：{audio_path}")
        return gen_frame_num, video_index_list

    @NVTXContext
    def _load_video_and_mask_data_from_index_list(self, video_index_list: List[dict], device, video_cache_dir = "/home/work/hy_01/templates/0612_online_templete_female_32fps_features", mask_base_dir: str = None, debug=False):
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
        # video_cache_dir = "/home/work/hy_01/templates/0612_online_templete_female_32fps_features"
        video_cache_dir = video_cache_dir
        
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
            # video_hash = hashlib.md5(video_path.encode()).hexdigest()

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

