from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import torch
import numpy as np

from diffusers.utils import BaseOutput
from latentsync.src.models.unet import UNet3DConditionOutput

from latentsync import NVTXContext
from latentsync.trt_backend import TrtExecutor
from cuda import cudart

from einops import rearrange
import os
from typing import Optional

# 在文件开头添加这些函数定义
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
    """
    DiyCache版本的UNet3D forward方法
    
    Args:
        sample: 输入的噪声样本 [batch, channel, frames, height, width]
        timestep: 时间步
        encoder_hidden_states: 编码器隐藏状态（音频特征）
        class_labels: 类别标签（可选）
        attention_mask: 注意力掩码（可选）
        down_block_additional_residuals: 下采样块额外残差（用于ControlNet）
        mid_block_additional_residual: 中间块额外残差（用于ControlNet）
        return_dict: 是否返回字典格式
    
    Returns:
        UNet3DConditionOutput或tuple: 去噪后的样本
    """
    # print(f"DiyCache forward called")
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
    sample = self.conv_in(sample) # torch.Size([2, 320, 16, 48, 48]) for 384

    # 记录总步数统计
    if hasattr(self, 'total_steps'):
        self.total_steps += 1

    # 主要计算逻辑
    if self.enable_diy_tcache: # 自定义直接跳步
        if self.cnt < self.first_step_offset or self.cnt > self.num_steps - self.last_step_offset:
            should_calc = True
            # 重新计算
            ori_sample = sample.clone()
            ############################### BACKEND #################################
            if self.verbose:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            if self.backend == 'trt' and 'unet' in self.trt_executor.stages:
                feed_dict = {
                    "sample": sample,
                    "emb": emb,
                    "encoder_hidden_states": encoder_hidden_states,
                }

                final_sample =  self.trt_executor.runEngine('unet', feed_dict)['out_sample']
                real_backend = 'trt'
            elif self.backend == 'torch':
                final_sample = self._compute_unet_blocks(
                sample, emb, encoder_hidden_states, attention_mask,
                down_block_additional_residuals, mid_block_additional_residual,
                forward_upsample_size
            )
                real_backend = 'torch'
            else:
                final_sample = self._compute_unet_blocks(
                sample, emb, encoder_hidden_states, attention_mask,
                down_block_additional_residuals, mid_block_additional_residual,
                forward_upsample_size
            )
                real_backend = 'torch'
            if self.verbose:
                end.record()
                torch.cuda.synchronize()
                print("[INFO] unet_cache", real_backend, "execution time {:.2f}".format(start.elapsed_time(end)), "ms")
            ############################### BACKEND #################################

            current_residual = final_sample - ori_sample
            
            # 根据配置的更新模式更新previous_residual（在计算一致性指标之后）
            
            if hasattr(self, 'previous_residual') and self.previous_residual is not None:
                self.previous_residual = current_residual
            else:
                # 第一次计算，所有模式都直接使用当前残差
                self.previous_residual = current_residual

        else:
            should_calc = False
            final_sample = sample + self.previous_residual
    else:
        # 标准计算路径
        ############################### BACKEND #################################
        if self.verbose:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        if self.backend == 'trt' and 'unet' in self.trt_executor.stages:
            feed_dict = {
                "sample": sample,
                "emb": emb,
                "encoder_hidden_states": encoder_hidden_states,
            }

            final_sample =  self.trt_executor.runEngine('unet', feed_dict)['out_sample']
            real_backend = 'trt'
        elif self.backend == 'torch':
            final_sample = self._compute_unet_blocks(
            sample, emb, encoder_hidden_states, attention_mask,
            down_block_additional_residuals, mid_block_additional_residual,
            forward_upsample_size
        )
            real_backend = 'torch'
        else:
            final_sample = self._compute_unet_blocks(
            sample, emb, encoder_hidden_states, attention_mask,
            down_block_additional_residuals, mid_block_additional_residual,
            forward_upsample_size
        )
            real_backend = 'torch'
        if self.verbose:
            end.record()
            torch.cuda.synchronize()
            print("[INFO] unet_cache", real_backend, "execution time {:.2f}".format(start.elapsed_time(end)), "ms")
        ############################### BACKEND #################################
    # print(f"final_sample 1 shape: {final_sample.shape}") # torch.Size([1, 320, 16, 32, 32])

    # 后处理
    final_sample = self.conv_norm_out(final_sample)
    final_sample = self.conv_act(final_sample)
    # print(f"final_sample 2 shape: {final_sample.shape}") # torch.Size([1, 320, 16, 32, 32])
    final_sample = self.conv_out(final_sample)
    # print(f"final_sample 3 shape: {final_sample.shape}") # torch.Size([1, 4, 16, 32, 32])

    # 更新计数器
    self.cnt += 1
    if self.cnt >= self.num_steps:
        
        # 重置会影响下一个推理任务的状态
        self.cnt = 0
        self.previous_residual = None
        
        # 注意：统计信息（skipped_steps, total_steps等）不重置
        # 这些信息需要跨多个推理任务累积，最后统一输出

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
    """
    执行UNet的主要计算：down blocks + mid block + up blocks
    """
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
            if down_block_additional_residual.dim() == 4:  # 广播
                down_block_additional_residual = down_block_additional_residual.unsqueeze(2)
            down_block_res_samples[i] = down_block_res_samples[i] + down_block_additional_residual

    # Mid block
    sample = self.mid_block(
        sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
    )

    # 支持ControlNet
    if mid_block_additional_residual is not None:
        if mid_block_additional_residual.dim() == 4:  # 广播
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


def setup_diycache_unet(unet, first_step_offset=1, last_step_offset=4, num_steps=20, show_stats=False, ema_alpha=0.7, residual_update_mode='ema', momentum_beta=0.9):
    """
    为UNet3DConditionModel设置TeaCache功能
    
    Args:
        unet: UNet3DConditionModel实例
        first_step_offset: 第一步的偏移系数，此前不能跳步，默认为1，可配置为其他值
        last_step_offset: 最后一步的偏移系数，此后不能跳步，默认为5，可配置为其他值
        num_steps: 总推理步数
        show_stats: 是否启用跳步统计
        ema_alpha: EMA更新的alpha参数（当residual_update_mode='ema'时使用）
        residual_update_mode: residual更新模式，支持 'ema', 'native', 'momentum'
        momentum_beta: Momentum更新的beta参数（当residual_update_mode='momentum'时使用）
    """
    # 添加TeaCache相关属性
    unet.__class__.enable_diy_tcache = True
    unet.__class__.forward = diycache_forward
    unet.__class__._compute_unet_blocks = _compute_unet_blocks
    
    # 初始化TeaCache参数
    unet.__class__.cnt = 0
    unet.__class__.num_steps = num_steps  # 不需要乘以2，因为CFG是batch concat的
    unet.__class__.first_step_offset = first_step_offset
    unet.__class__.last_step_offset = last_step_offset  # 添加最后一步偏移系数
    
    # # 初始化累积距离和缓存（简化为单一路径）
    unet.__class__.previous_residual = None
    
    
    print(f"DiyCache已启用 - 第一步偏移: {first_step_offset}, 最后一步偏移: {last_step_offset}")
    return unet

def load_trt_unet(self, engine_root_path, batch_size=1, use_cuda_graph=False, verbose=False):
    print("[INFO] Start loading trt engine for Unet")
    self.backend = 'trt'
    self.verbose = verbose
    self.trt_executor = TrtExecutor(stages=['unet'], use_cuda_graph=use_cuda_graph, verbose=verbose)
    self.trt_executor.loadEngines(engine_root_path)

def load_trt_vae(self, engine_root_path, batch_size=1, use_cuda_graph=False, verbose=False):
    print("[INFO] Start loading trt engine for vae encode and decode")
    self.backend = 'trt'
    self.verbose = verbose
    self.trt_executor = TrtExecutor(stages=['vae_decode', 'vae_encode'], use_cuda_graph=use_cuda_graph, verbose=verbose)
    self.trt_executor.loadEngines(engine_root_path)

def calculate_max_device_memory(self):
    return self.trt_executor.calculateMaxDeviceMemory()

def activate_engine(self, shared_device_memory, batch_size=1):
    self.trt_executor.activateEngines(shared_device_memory)
    self.trt_executor.loadResources()
    self.trt_executor.reshape(batch_size)

def reshape(self, batch_size=1, remove_cfg=False):
    self.trt_executor.reshape(batch_size, remove_cfg)


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
        ############################### BACKEND #################################
        if self.vae.verbose:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        if self.vae.backend == 'trt' and 'vae_decode' in self.vae.trt_executor.stages:
            feed_dict = {
                "latents": masked_image,
            }
            masked_image_latents =  self.vae.trt_executor.runEngine('vae_encode', feed_dict)['output'].to(torch.float16)
            real_backend = 'trt'
        elif self.vae.backend == 'torch':
            masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
            real_backend = 'torch'
        else:
            masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
            real_backend = 'torch'
        if self.vae.verbose:
            end.record()
            torch.cuda.synchronize()
            print("[INFO] vae encode mask", real_backend, "execution time {:.2f}".format(start.elapsed_time(end)), "ms")
        ############################### BACKEND #################################

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

    ############################### BACKEND #################################
    if self.vae.verbose:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    if self.vae.backend == 'trt' and 'vae_decode' in self.vae.trt_executor.stages:
        feed_dict = {
            "latents": images,
        }
        image_latents =  self.vae.trt_executor.runEngine('vae_encode', feed_dict)['output'].to(torch.float16)
        real_backend = 'trt'
    elif self.vae.backend == 'torch':
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        real_backend = 'torch'
    else:
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        real_backend = 'torch'
    if self.vae.verbose:
        end.record()
        torch.cuda.synchronize()

        print("[INFO] vae encode image", real_backend, "execution time {:.2f}".format(start.elapsed_time(end)), "ms")
    ############################### BACKEND #################################

    image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
    image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
    image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

    return image_latents

@NVTXContext
def decode_latents(self, latents):
    latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    ############################### BACKEND #################################
    if self.vae.verbose:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    if self.vae.backend == 'trt' and 'vae_decode' in self.vae.trt_executor.stages:
        feed_dict = {
            "latents": latents,
        }

        decoded_latents =  self.vae.trt_executor.runEngine('vae_decode', feed_dict)['output']
        real_backend = 'trt'
    elif self.vae.backend == 'torch':
        decoded_latents = self.vae.decode(latents).sample
        real_backend = 'torch'
    else:
        decoded_latents = self.vae.decode(latents).sample
        real_backend = 'torch'
    if self.vae.verbose:
        end.record()
        torch.cuda.synchronize()
        print("[INFO] vae decode", real_backend, "execution time {:.2f}".format(start.elapsed_time(end)), "ms")
    ############################### BACKEND #################################
    return decoded_latents

def resolve_path(path: Optional[str], model_base: str, code_base: str) -> Optional[str]:
    """
    统一路径解析函数
    
    Args:
        path: 要解析的路径
        model_base: 模型库基础路径
        code_base: 代码基础路径
        
    Returns:
        解析后的绝对路径
    """
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
        return os.path.join(model_base, path)
    # 其他路径（如0612_online_templete_female_32fps_resize_720_1560/），拼到模型库
    return os.path.join(model_base, path)
