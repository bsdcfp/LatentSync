# coding: utf-8
from mmu_service_grammar.runtime import *
import mmu_service_grammar.triton.logger as logger


import base64
import io
import json

import os
import cv2
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import BaseOutput
import tempfile
import time
from typing import Union, Tuple, Optional
from dataclasses import dataclass

# 导入必要的模块
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline_online_0701 import LipsyncPipeline_online_0701 as LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.utils.video_sequence_merger import VideoIndexGenerator

# 尝试导入UNet3DConditionOutput
try:
    from latentsync.models.unet import UNet3DConditionOutput
except ImportError:
    @dataclass
    class UNet3DConditionOutput(BaseOutput):
        sample: torch.FloatTensor



#@deployment(num_replicas=2)
#class ModelA:
#    def __init__(self):
#        self._model = Model("xxx.torch")
#
#    def inference(params):
#        return self._model.inference(params)
        




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
            print(f"⚠️  自动路径不存在，使用默认路径: {self.script_dir}")
            
        self.auxiliary_dir = "/home/work/latentsync_online_simple_auxilary"
        if not os.path.exists(self.auxiliary_dir):
            self.auxiliary_dir = "/home/work/houyi/latentsync_online_simple_auxilary"
            print(f"⚠️  auxiliary_dir路径不存在，使用默认路径: {self.auxiliary_dir}")
        
        # 模型配置文件
        self.unet_config_path = os.path.join(self.script_dir, "configs/unet/second_stage.yaml")
        
        # 权重路径
        self.weights_dir = os.path.join(self.auxiliary_dir, "weights/latentsync")
        self.inference_ckpt_path = os.path.join(self.weights_dir, "latentsync_unet.pt")
        self.whisper_model_path = os.path.join(self.weights_dir, "whisper/tiny.pt")
        
        # 模板配置
        self.template_dir = os.path.join(self.auxiliary_dir, "0612_online_templete_female_32fps_resize_720_1280")
        self.video_template_dir = os.path.join(self.template_dir, "videos")
        self.cache_dir = os.path.join(self.template_dir, "videos_features")
        self.mask_base_dir = os.path.join(self.template_dir, "videos_mask")
        
        # 模板信息文件
        self.connected_video_info = os.path.join(self.template_dir, "template_info_0618/connected_videos.json")
        self.shot_video_json = os.path.join(self.template_dir, "template_info_0618/connected_videos_other_short.json")
        self.video_info_json = os.path.join(self.template_dir, "template_info_0618/0612_video_templete_32fps_info_v2.json")
        
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
        
        # 打印路径信息用于调试
        print(f"🔧 路径配置信息:")
        print(f"  - 自动获取路径: {auto_script_dir}")
        print(f"  - 实际使用路径: {self.script_dir}")
        print(f"  - 使用方式: {'自动获取' if self.script_dir == auto_script_dir else '默认fallback'}")
        print(f"  - Auxiliary目录: {self.auxiliary_dir}")
        
        # 检查关键路径是否存在
        print(f"🔍 路径存在性检查:")
        print(f"  - Script目录存在: {os.path.exists(self.script_dir)}")
        print(f"  - UNet配置文件存在: {os.path.exists(self.unet_config_path)}")
        print(f"  - Auxiliary目录存在: {os.path.exists(self.auxiliary_dir)}")
        print(f"  - 权重目录存在: {os.path.exists(self.weights_dir)}")
        print(f"  - 模板目录存在: {os.path.exists(self.template_dir)}")
        
        if not os.path.exists(self.script_dir):
            print(f"⚠️  警告: Script目录不存在: {self.script_dir}")
        if not os.path.exists(self.auxiliary_dir):
            print(f"⚠️  警告: Auxiliary目录不存在: {self.auxiliary_dir}")

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
    
    print(f"✅ DiyCache已启用 - 第一步偏移: {first_step_offset}, 最后一步偏移: {last_step_offset}, 总步数: {num_steps}")
    return unet

"""
Algorithm API implementation
"""
@deployment(num_replicas=2)
@ingress
class AlgoModel(object):
    def __init__(self):
        """模型初始化"""
        print("🚀 开始初始化 AlgoModel...")
        
        # 加载配置
        self.config = ModelConfig()
        
        # 设置工作目录和Python路径
        os.chdir(self.config.script_dir)
        if self.config.script_dir not in os.environ.get('PYTHONPATH', ''):
            os.environ['PYTHONPATH'] = f"{self.config.script_dir}:{os.environ.get('PYTHONPATH', '')}"
        
        # 加载UNet配置
        self.unet_config = OmegaConf.load(self.config.unet_config_path)
        
        # 初始化各个组件
        self._init_scheduler()
        self._init_audio_encoder()
        self._init_vae()
        self._init_unet()
        self._init_video_index_generator()
        self._init_pipeline()
        
        # 启用DiyCache（如果配置启用）
        if self.config.enable_diycache:
            self._setup_diycache()
        
        print("✅ AlgoModel 初始化完成!")

    def _init_scheduler(self):
        """初始化调度器"""
        print("📋 初始化调度器...")
        self.scheduler = DDIMScheduler.from_pretrained("configs")

    def _init_audio_encoder(self):
        """初始化音频编码器"""
        print("🎵 初始化音频编码器...")
        if self.unet_config.model.cross_attention_dim == 768:
            whisper_model_path = self.config.whisper_model_path
        elif self.unet_config.model.cross_attention_dim == 384:
            whisper_model_path = self.config.whisper_model_path
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")
            
        self.audio_encoder = Audio2Feature(
            model_path=whisper_model_path, 
            device=self.config.device, 
            num_frames=self.unet_config.data.num_frames
        )

    def _init_vae(self):
        """初始化VAE"""
        print("🎨 初始化VAE...")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.config.dtype)
        self.vae.config.scaling_factor = 0.18215
        self.vae.config.shift_factor = 0

    def _init_unet(self):
        """初始化UNet"""
        print("🧠 初始化UNet...")
        self.unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(self.unet_config.model),
            self.config.inference_ckpt_path,
            device="cpu",
        )
        self.unet = self.unet.to(dtype=self.config.dtype)
        
        # 设置xformers
        if is_xformers_available():
            print("✅ 启用xformers内存高效注意力")
            self.unet.enable_xformers_memory_efficient_attention()
        else:
            print("⚠️  xformers不可用，跳过内存高效注意力")

    def _init_video_index_generator(self):
        """初始化视频索引生成器"""
        print("🎬 初始化视频索引生成器...")
        self.video_index_generator = VideoIndexGenerator(
            connected_json_path=self.config.connected_video_info,
            video_info_json_path=self.config.video_info_json,
            shot_video_json_path=self.config.shot_video_json,
            video_template_dir=self.config.video_template_dir,
            target_fps=self.config.target_fps,
            transition_frames=self.config.transition_frames,
            debug=False
        )

    def _init_pipeline(self):
        """初始化推理管道"""
        print("🔧 初始化推理管道...")
        self.pipeline = LipsyncPipeline(
            vae=self.vae,
            audio_encoder=self.audio_encoder,
            unet=self.unet,
            scheduler=self.scheduler,
            video_index_generator=self.video_index_generator,
        ).to(self.config.device)

    def _setup_diycache(self):
        """设置DiyCache加速"""
        print("⚡ 启用DiyCache加速...")
        
        # DiyCache配置参数
        diycache_config = {
            "num_steps": self.config.inference_steps,
            "first_step_offset": self.config.first_step_offset,
            "last_step_offset": self.config.last_step_offset,
        }
        
        # 为UNet启用DiyCache
        setup_diycache_unet(self.pipeline.unet, **diycache_config)
        
        print(f"  - 第一步偏移: {diycache_config['first_step_offset']}")
        print(f"  - 最后一步偏移: {diycache_config['last_step_offset']}")
        print(f"  - 推理步数: {diycache_config['num_steps']}")

    def __call__(self, ctx):
        """
        主要推理逻辑
        
        期望输入格式:
        ctx.get_auio_list()[0] = {
            'audio_data': bytes,  # 音频文件的二进制数据
            'extra_info': str     # JSON字符串，包含额外参数（可为空）
        }
        """
        try:
            # 获取输入数据
            audio_info = ctx.get_auio_list()[0]
            audio_info = json.loads(audio_info)
            audio_data = io.BytesIO(audio_info['audio_data'])
            # audio_url = audio_info['audio_url']
            
            # 创建临时文件保存音频
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
                temp_audio_file.write(audio_data.getvalue())
                temp_audio_path = temp_audio_file.name
            
            # 创建临时输出目录
            with tempfile.TemporaryDirectory() as temp_output_dir:
                video_out_path = os.path.join(temp_output_dir, "generated_video.mp4")
                mask_output_path = os.path.join(temp_output_dir, "generated_mask.mp4")
                try:
                    # 执行视频生成
                    print(f"🎬 开始视频生成推理...")
                    # print(f"⚡ DiyCache状态: {'已启用' if self.config.enable_diycache else '未启用'}")
                    start_time = time.time()
                    
                    gen_frame_num, video_index_list = self.pipeline.generate_video_online_0701(
                        video_cache_dir=self.config.cache_dir,
                        audio_path=temp_audio_path,
                        video_out_path=video_out_path,
                        mask_output_path=mask_output_path,
                        mask_base_dir=self.config.mask_base_dir,
                        num_frames=self.config.num_frames,
                        num_inference_steps=self.config.inference_steps,
                        guidance_scale=self.config.guidance_scale,
                        weight_dtype=self.config.dtype,
                        width=self.unet_config.data.resolution,
                        height=self.unet_config.data.resolution,
                        debug=False,
                    )
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    print(f"✅ 视频生成完成，耗时: {processing_time:.2f}s")
                    
                    # 读取生成的视频文件并转为base64
                    with open(video_out_path, 'rb') as f:
                        video_data = f.read()
                    
                    video_base64 = base64.b64encode(video_data).decode('utf-8')
                    
                    # 返回结果
                    result = {
                        "status": "success",
                        "video_base64": video_base64,
                        "generated_frames": gen_frame_num,
                        "processing_time": processing_time,
                        "video_size_bytes": len(video_data),
                        "fps": self.config.target_fps,
                        "diycache_enabled": self.config.enable_diycache,
                        "inference_steps": self.config.inference_steps
                    }
                    
                    print(f"📊 生成统计: {gen_frame_num}帧, {len(video_data)}字节, {processing_time:.2f}s")
                    
                except Exception as inference_error:
                    print(f"❌ 推理过程出错: {str(inference_error)}")
                    result = {
                        "status": "error",
                        "error_message": f"推理失败: {str(inference_error)}",
                        "error_type": "inference_error"
                    }
                    
                finally:
                    # 清理临时音频文件
                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
            
            # 设置返回结果
            ctx.set_result_dict({
                "extra_info": json.dumps(result),
            })
            
        except Exception as e:
            print(f"❌ 处理请求时出错: {str(e)}")
            # 错误处理
            error_result = {
                "status": "error", 
                "error_message": str(e),
                "error_type": "general_error"
            }
            ctx.set_result_dict({
                "extra_info": json.dumps(error_result),
            })


# if __name__ == '__main__':
#     # 测试代码
#     import requests
#     from unittest.mock import Mock
#     
#     # 创建模拟的ctx对象
#     ctx = Mock()
#     
#     # 使用本地测试文件
#     with open("/home/work/houyi/latentsync_online_simple_auxilary/input_example/beauty_v2_20s.MP3", "rb") as f:
#         audio_data = f.read()
#     
#     ctx.get_auio_list.return_value = [
#         {
#             "audio_data": audio_data,
#             "extra_info": json.dumps({})  # 现在只需要空的JSON即可
#         }
#     ]
#     
#     # 模拟set_result_dict方法
#     result_dict = {}
#     def mock_set_result_dict(data):
#         result_dict.update(data)
#     ctx.set_result_dict = mock_set_result_dict
#     
#     # 测试
#     print("🧪 开始测试...")
#     algomodel = AlgoModel()
#     algomodel(ctx)
#     
#     # 检查结果
#     result = json.loads(result_dict['extra_info'])
#     print("📋 测试结果:", result)
#     
#     if result['status'] == 'success':
#         # 保存测试视频
#         video_data = base64.b64decode(result['video_base64'])
#         with open('test_output.mp4', 'wb') as f:
#             f.write(video_data)
#         print("✅ 测试成功，视频已保存为 test_output.mp4")
#         print(f"🚀 DiyCache状态: {result.get('diycache_enabled', 'Unknown')}")
#         print(f"⚡ 加速效果: {result.get('processing_time', 0):.2f}s")
#     else:
#         print("❌ 测试失败:", result)
