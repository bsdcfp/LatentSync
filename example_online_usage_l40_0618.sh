#!/bin/bash

# 需要自定义的几个参数
# SCRIPT_DIR：工作目录
# AUXILIARY_DIR：辅助目录，包括权重、模板、输入音频等
# AUDIO_PATH：输入音频文件
# OUTPUT_DIR：输出目录


# 在线模式使用示例脚本
# 设置基本路径
SCRIPT_DIR="/home/work/hy_01/latentsync_online_simple"
# 设置辅助目录，包括权重、模板、输入音频等
AUXILIARY_DIR="/home/work/hy_01/latentsync_online_simple_auxilary"

# 切换到工作目录（必须！）
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 模型和配置路径
INFERENCE_SCRIPT="$SCRIPT_DIR/scripts/inference_online_0618.py"
UNET_CONFIG="$SCRIPT_DIR/configs/unet/second_stage.yaml"

WEIGHTS_DIR="$AUXILIARY_DIR/weights/latentsync"
INFERENCE_CKPT="$WEIGHTS_DIR/latentsync_unet.pt"
WHISPER_MODEL_PATH="$WEIGHTS_DIR/whisper/tiny.pt"

# 配置模版缓存参数
TEMPLATE_DIR="$AUXILIARY_DIR/0612_online_templete_female_32fps_resize_720_1560"
VIDEO_TEMPLATE_DIR="$TEMPLATE_DIR/videos"
CACHE_DIR="$TEMPLATE_DIR/videos_features"
MASK_BASE_DIR="$TEMPLATE_DIR/videos_mask"

CONNECTED_VIDEO_INFO="$TEMPLATE_DIR/template_info_0618/connected_videos.json"
SHOT_VIDEO_JSON="$TEMPLATE_DIR/template_info_0618/connected_videos_other_short.json"
VIDEO_INFO_JSON="$TEMPLATE_DIR/template_info_0618/0612_video_templete_32fps_info_v2.json"

# 唯一输出：输入音频文件
AUDIO_PATH="$AUXILIARY_DIR/input_example/beauty_v2_20s.MP3"

# 输出两个：视频输出路径 + mask输出路径
# OUTPUT_DIR="/home/work/hy_01/outputs/online_mode_test_0624_20s_final"
OUTPUT_DIR="$SCRIPT_DIR/outputs/online_mode_test_0624_20s_final"

mkdir -p "$OUTPUT_DIR"
VIDEO_OUT_PATH="$OUTPUT_DIR/generated_video_$(date +%Y%m%d_%H%M%S).mp4"
MASK_OUTPUT_PATH="$OUTPUT_DIR/generated_mask_video_$(date +%Y%m%d_%H%M%S).mp4"

# 日志文件
LOG_FILE="$OUTPUT_DIR/inference_log_$(date +%Y%m%d_%H%M%S).txt"

echo "🚀 开始在线模式视频生成..."
echo "📝 日志将保存到: $LOG_FILE"
echo "🎬 输出视频: $VIDEO_OUT_PATH"
echo "💾 缓存目录: $CACHE_DIR"
echo "============================================================"


python "$INFERENCE_SCRIPT" \
    --unet_config_path "$UNET_CONFIG" \
    --inference_ckpt_path "$INFERENCE_CKPT" \
    --whisper_model_path "$WHISPER_MODEL_PATH" \
    --audio_path "$AUDIO_PATH" \
    --video_out_path "$VIDEO_OUT_PATH" \
    --video_template_dir "$VIDEO_TEMPLATE_DIR" \
    --connected_info_json "$CONNECTED_VIDEO_INFO" \
    --video_info_json "$VIDEO_INFO_JSON" \
    --shot_video_json_path "$SHOT_VIDEO_JSON" \
    --video_cache_dir "$CACHE_DIR" \
    --mask_base_dir "$MASK_BASE_DIR" \
    --mask_output_path "$MASK_OUTPUT_PATH" \
    --target_fps 25 \
    --transition_frames 20 \
    --inference_steps 20 \
    --guidance_scale 1.0 \
    --seed 1247 \
    --log_file "$LOG_FILE" \
    --enable_diycache \
    --first_step_offset 1 \
    --last_step_offset 4 \

echo "============================================================"
echo "✅ 在线模式视频生成完成！"
echo "📁 输出文件:"
echo "  - 视频: $VIDEO_OUT_PATH"
echo "  - 日志: $LOG_FILE"

# 检查输出文件是否存在
if [ -f "$VIDEO_OUT_PATH" ]; then
    echo "🎉 视频生成成功！"
    echo "📊 文件大小: $(du -h "$VIDEO_OUT_PATH" | cut -f1)"
else
    echo "❌ 视频生成失败，请检查日志文件"
fi 
