# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline_online_0618 import LipsyncPipeline_online_0618 as LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
import os
import sys  # 添加sys模块用于重定向输出
from typing import Union, Tuple, Optional
import numpy as np
from diffusers.utils import BaseOutput
import time  # 添加时间模块
import matplotlib.pyplot as plt

# 尝试导入UNet3DConditionOutput
try:
    # 首先尝试从当前目录的unet模块导入
    from unet import UNet3DConditionOutput
except ImportError:
    try:
        # 如果失败，尝试从latentsync.models.unet导入
        from latentsync.models.unet import UNet3DConditionOutput
    except ImportError:
        from dataclasses import dataclass
        # 如果都失败了，创建一个简单的输出类
        @dataclass
        class UNet3DConditionOutput(BaseOutput):
            sample: torch.FloatTensor

from DeepCache import DeepCacheSDHelper

# 导入VideoIndexGenerator用于在线模式
from latentsync.utils.video_sequence_merger import VideoIndexGenerator

class Tee:
    """用于同时将输出写入控制台和文件的类"""
    def __init__(self, log_file_path):
        self.file = open(log_file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        
    def write(self, data):
        self.file.write(data)
        self.file.flush()  # 立即刷新到文件
        self.stdout.write(data)
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def close(self):
        self.file.close()


def main(config, args):
    # 设置日志输出重定向
    tee = None
    if args.log_file:
        # 确保日志文件的目录存在
        log_dir = os.path.dirname(args.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # 创建Tee对象并重定向stdout
        tee = Tee(args.log_file)
        sys.stdout = tee
        print(f"📝 日志文件已启用: {args.log_file}")
        print("=" * 60)

    # 验证mask相关参数
    if args.mask_output_path is not None and args.mask_base_dir is None:
        print("❌ 错误: 当指定 --mask_output_path 时，必须同时指定 --mask_base_dir")
        if tee:
            tee.close()
        return
    
    if args.mask_output_path is not None:
        print(f"🎭 Mask视频输出已启用:")
        print(f"  - Mask基础目录: {args.mask_base_dir}")
        print(f"  - Mask输出路径: {args.mask_output_path}")

    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print("🚀 使用在线模式 (generate_video_online)")
    print(f"  - 视频模板目录: {args.video_template_dir}")
    print(f"  - 连接视频信息: {args.connected_info_json}")
    print(f"  - 视频信息文件: {args.video_info_json}")
    print(f"  - 输入音频路径: {args.audio_path}")
    print(f"  - 加载检查点路径: {args.inference_ckpt_path}")

    scheduler = DDIMScheduler.from_pretrained(args.scheduler_config_path)

    # 直接使用命令行参数中的whisper模型路径
    whisper_model_path = args.whisper_model_path

    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

    vae = AutoencoderKL.from_pretrained(args.vae_model_path, torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,  # load checkpoint
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    # set xformers
    if is_xformers_available():
        print(f"Xformers is available, enabling xformers memory efficient attention")
        unet.enable_xformers_memory_efficient_attention()
    else:
        print(f"Xformers is not available, skipping xformers memory efficient attention")


    # 在线模式：使用VideoIndexGenerator + generate_video_online
    print("🔄 初始化VideoIndexGenerator...")
    

    video_index_generator = VideoIndexGenerator(
        connected_json_path=args.connected_info_json,
        video_info_json_path=args.video_info_json,
        shot_video_json_path=args.shot_video_json_path,
        video_template_dir=args.video_template_dir,
        target_fps=args.target_fps,
        transition_frames=args.transition_frames,
        debug=args.debug_video_merger
    )

    # 创建在线模式pipeline
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
        video_index_generator=video_index_generator,
    ).to("cuda")

    if args.enable_deepcache:
        # 初始化DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(
            cache_interval=3,
            cache_branch_id=0,
        )
        helper.enable()

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    # 根据命令行参数决定是否启用DiyCache
    if args.enable_diycache:
        print("正在启用DiyCache...")
        
        # diyCache配置参数
        diycache_config = {
            "num_steps": args.inference_steps,  # 使用inference_steps作为num_steps
            "first_step_offset": args.first_step_offset,  # 添加第一步偏移
            "last_step_offset": args.last_step_offset,  # 添加最后一步偏移
        }
        
        # 为UNet启用diyCache
        setup_diycache_unet(pipeline.unet, **diycache_config)
        
        print(f"DiyCache已启用:")
        print(f"  - 第一步偏移: {diycache_config['first_step_offset']}")
        print(f"  - 最后一步偏移: {diycache_config['last_step_offset']}")
        print(f"  - 推理步数: {diycache_config['num_steps']}")

    else:
        print("DiyCache未启用，使用标准推理模式")

    # 改进后的目录创建方法，自动处理已存在的情况
    os.makedirs(os.path.dirname(args.video_out_path), exist_ok=True)
    
    # 记录pipeline开始时间
    print("=" * 60)
    print("🚀 开始在线视频生成...")
    start_time = time.time()

    # exit(0)
    pipeline.generate_video_online_0618(
        video_cache_dir=args.video_cache_dir,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        mask_output_path=args.mask_output_path,
        mask_base_dir=args.mask_base_dir,
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,

        debug=args.debug_pipeline,
    )

    # 记录pipeline结束时间并计算总耗时
    end_time = time.time()
    total_time = end_time - start_time
    
    print("=" * 60)
    print(f"🎉 视频生成完成！")
    print(f"📊 性能统计:")
    print(f"  - 总执行时间: {total_time:.2f} 秒")
    print(f"  - 推理步数: {args.inference_steps}")
    # print(f"  - 每步平均时间: {total_time/args.inference_steps:.3f} 秒")
    if args.enable_deepcache:
        print(f"  - DeepCache状态: 已启用")
    else:
        print(f"  - DeepCache状态: 未启用")
    if args.enable_diycache:
        print(f"  - DiyCache状态: 已启用 ")
        # 显示DiyCache统计信息
    else:
        print(f"  - DiyCache状态: 未启用")
    print(f"  - 输出视频: {args.video_out_path}")
    print("=" * 60)

    # 清理日志文件重定向
    if tee:
        sys.stdout = tee.stdout  # 恢复原来的stdout
        tee.close()  # 关闭日志文件
        print(f"📝 日志已保存到: {args.log_file}")


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
    unet.__class__.enable_diy_tcache = False
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在线模式视频生成脚本")
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--whisper_model_path", type=str, default="checkpoints/whisper/tiny.pt")
    
    # SD模型路径参数
    parser.add_argument("--vae_model_path", type=str, default="stabilityai/sd-vae-ft-mse",
                       help="VAE模型路径，可以是HuggingFace模型ID或本地路径")
    parser.add_argument("--scheduler_config_path", type=str, default="configs",
                       help="调度器配置路径")
    
    # 输入输出参数
    parser.add_argument("--audio_path", type=str, required=True, help="输入音频路径")

    # 在线模式必需参数
    parser.add_argument("--video_cache_dir", type=str, 
                       default="/home/work/hy_01/templates/0612_online_templete_female_32fps_features",
                       help="视频特征缓存目录")
    parser.add_argument("--video_out_path", type=str, required=True, help="输出视频路径")
    parser.add_argument("--mask_output_path", type=str, default=None, help="mask视频输出路径（可选）")
    parser.add_argument("--mask_base_dir", type=str, default=None, 
                       help="mask视频基础目录（当指定mask_output_path时必需）")
    parser.add_argument("--video_template_dir", type=str, required=True,
                       help="视频模板目录")
    parser.add_argument("--connected_info_json", type=str, required=True,
                       help="连接视频信息JSON文件路径")
    parser.add_argument("--video_info_json", type=str, required=True,
                       help="视频信息JSON文件路径")
    parser.add_argument("--shot_video_json_path", type=str, required=True,
                       help="镜头视频JSON文件路径")
    
    # 推理参数
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--log_file", type=str, default=None, 
                       help="指定日志文件路径，所有print输出将同时保存到此文件（可选）")


    # 加速参数
    parser.add_argument("--enable_deepcache", action="store_true", default=False, 
                       help="启用DeepCache加速推理（默认不启用）")
    
    # DiyCache相关参数
    parser.add_argument("--enable_diycache", action="store_true", default=False, 
                       help="启用自定义DiyCache加速推理（默认不启用）")
    parser.add_argument("--first_step_offset", type=int, default=1,
                       help="第一步的偏移系数，此前不能跳步，默认为1，可配置为其他值")
    parser.add_argument("--last_step_offset", type=int, default=4,
                       help="最后一步的偏移系数，此后不能跳步，默认为4，可配置为其他值")
    
    # 在线模式配置参数
    parser.add_argument("--target_fps", type=int, default=25,
                       help="目标FPS")
    parser.add_argument("--transition_frames", type=int, default=20,
                       help="过渡帧数")

    parser.add_argument("--debug_video_merger", action="store_true", default=False,
                       help="启用VideoIndexGenerator调试模式")
    parser.add_argument("--debug_pipeline", action="store_true", default=False,
                       help="启用Pipeline调试模式，显示详细处理信息")
    
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)

    main(config, args)
