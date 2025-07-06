
from latentsync.utils.video_sequence_merger import VideoIndexGenerator
import torch
import os
from omegaconf import OmegaConf


class Config:
    def __init__(self):
        # 基础配置
        self.checkpoint_path = "/home/work/hy_01/weights/latentsync/latentsync_unet.pt"
        # self.config_yaml_path = os.path.join(WORK_DIR, "latent_sync/configs/unet/second_stage.yaml")
        self.config_yaml_path = "/home/work/hy_01/mmuplt-avatarv3/latent_sync/configs/unet/second_stage.yaml"
        self.config = OmegaConf.load(self.config_yaml_path)
        
        # 数据类型配置
        is_fp16_supported = torch.cuda.is_available()
        self.dtype = torch.float16 if is_fp16_supported else torch.float32

        # 在线模式配置
        self.video_template_dir = "/home/work/hy_01/templates/0612_online_templete_female_32fps_resize_720_1560/videos"
        self.connected_info_json = "/home/work/hy_01/mmuplt-avatarv3/video_scripts_online/0618/connected_videos.json"
        self.video_info_json = "/home/work/hy_01/mmuplt-avatarv3/video_scripts_online/0618/0612_video_templete_32fps_info_v2.json"
        self.shot_video_json_path = "/home/work/hy_01/mmuplt-avatarv3/video_scripts_online/0618/connected_videos_other_short.json"
        self.video_cache_dir = "/home/work/hy_01/templates/0612_online_templete_female_32fps_resize_720_1560/videos_features"
        self.mask_base_dir = "/home/work/hy_01/templates/0612_online_templete_female_32fps_resize_720_1560/videos_mask"
        
        # 推理参数
        self.target_fps = 25
        self.transition_frames = 20
        self.inference_steps = 20
        self.guidance_scale = 1.0
        self.seed = 1247
        
        # DiyCache配置
        self.enable_diycache = True
        self.first_step_offset = 1
        self.last_step_offset = 4
        self.show_stats = False
        
        # 日志配置 - 可一键修改的日志路径
        self.log_base_dir = os.getenv('CUSTOM_LOG_DIR', '/home/work/hy_01/logs/avatarv8_0618_prod')
        print(f"📁 日志输出目录: {self.log_base_dir}")
        
        # 跳过已处理文件的配置
        self.enable_skip_processed = os.getenv('ENABLE_SKIP_PROCESSED', 'True').lower() == 'true'
        print(f"⏭️  跳过已处理文件: {'启用' if self.enable_skip_processed else '禁用'}")
        self.residual_update_mode = 'native'
        self.ema_alpha = 0.7
        self.momentum_beta = 0.9
        
        # # 输入数据配置
        # self.tts_meta_json_path = "/home/work/hy_01/tts_gen/tts_0616_meta.json"
        # self.tts_audio_base_dir = "/home/work/hy_01/tts_gen/tts_0616"  # 音频文件基础目录
        # 输入数据配置
        # self.tts_meta_json_path = "/home/work/hy_01/tts_gen/demo_tts_0620_meta.json"
        self.tts_meta_json_path = "/home/work/hy_01/tts_gen/reproduced_datalist/tts_0616_meta.json"
        self.tts_audio_base_dir = "/home/work/hy_01/tts_gen/tts_0616"
        # self.tts_audio_base_dir = "/home/work/hy_01/tts_gen/demo_tts_0620"  # 音频文件基础目录

cfg = Config()

video_index_generator = VideoIndexGenerator(
            connected_json_path=cfg.connected_info_json,
            video_info_json_path=cfg.video_info_json,
            shot_video_json_path=cfg.shot_video_json_path,
            video_template_dir=cfg.video_template_dir,
            target_fps=cfg.target_fps,
            transition_frames=cfg.transition_frames,
            debug=False  # 在batch模式下关闭调试
        )

audio_path = "/home/work/hy_01/tts_gen/tts_0616/91EBB1E77DEA7951E324CBDB65B2E01C_yesthisispearl.wav"
video_index_list = video_index_generator.get_video_index(audio_path)
print(video_index_list)