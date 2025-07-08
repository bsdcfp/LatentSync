from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import torch
import numpy as np

from diffusers.utils import BaseOutput
from latentsync.src.models.unet import UNet3DConditionOutput

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
            final_sample = self._compute_unet_blocks(
                sample, emb, encoder_hidden_states, attention_mask,
                down_block_additional_residuals, mid_block_additional_residual,
                forward_upsample_size
            )
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
        final_sample = self._compute_unet_blocks(
            sample, emb, encoder_hidden_states, attention_mask,
            down_block_additional_residuals, mid_block_additional_residual,
            forward_upsample_size
        )
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
