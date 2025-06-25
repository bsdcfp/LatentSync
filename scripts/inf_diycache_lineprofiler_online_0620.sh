
# 设置基本路径
SCRIPT_DIR="/home/work/hy_01/mmuplt-avatarv3/latent_sync"
INFERENCE_SCRIPT="$SCRIPT_DIR/scripts/inference_online_0618.py"

# 模型和配置路径
UNET_CONFIG="configs/unet/second_stage.yaml"
INFERENCE_CKPT="/home/work/hy_01/weights/latentsync/latentsync_unet.pt"

# 输入文件
# AUDIO_PATH="/home/work/hy_01/mmuplt-avatarv3/latent_sync/test_case/beauty_v2_13s.wav"
# AUDIO_PATH="/home/work/hy_01/mmuplt-avatarv3/latent_sync/test_case/beauty_v2_50s.MP3"
AUDIO_PATH="/home/work/hy_01/mmuplt-avatarv3/latent_sync/test_case/beauty_v2_20s.MP3"

# 在线模式配置文件
# VIDEO_TEMPLATE_DIR="/home/work/hy_01/templates/0612_online_templete_female_32fps"
VIDEO_TEMPLATE_DIR="/home/work/hy_01/templates/0612_online_templete_female_32fps_resize_720_1560/videos"
CONNECTED_VIDEO_INFO="/home/work/hy_01/mmuplt-avatarv3/video_scripts_online/0618/connected_videos.json"
VIDEO_INFO_JSON="/home/work/hy_01/mmuplt-avatarv3/video_scripts_online/0618/0612_video_templete_32fps_info_v2.json"
SHOT_VIDEO_JSON="/home/work/hy_01/mmuplt-avatarv3/video_scripts_online/0618/connected_videos_other_short.json"
# CACHE_DIR="/home/work/hy_01/templates/0612_online_templete_female_32fps_features"
CACHE_DIR="/home/work/hy_01/templates/0612_online_templete_female_32fps_resize_720_1560/videos_features"
MASK_BASE_DIR="/home/work/hy_01/templates/0612_online_templete_female_32fps_resize_720_1560/videos_mask"


# 输出路径
OUTPUT_DIR="/home/work/hy_01/outputs/online_mode_test_0620_20s_final"
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

# 切换到工作目录
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"


PROFILE_REPORT_PATH="$OUTPUT_DIR/profiling_report_l40_cache_20s_online_0620.txt"
# python "$INFERENCE_SCRIPT" \
kernprof -l -v "$INFERENCE_SCRIPT" \
    --unet_config_path "$UNET_CONFIG" \
    --inference_ckpt_path "$INFERENCE_CKPT" \
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
    --last_step_offset 4 2>&1 | tee "$PROFILE_REPORT_PATH"
    # --show_stats \
    # --residual_update_mode native

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


# python -m scripts.inference_dicache_0604 \
#     --unet_config_path "configs/unet/second_stage.yaml" \
#     --inference_ckpt_path "$OFFICIAL_CKPT" \
#     --inference_steps 20 \
#     --guidance_scale "$GUIDANCE_SCALE" \
#     --video_path "$VIDEO_PATH" \
#     --audio_path "$AUDIO_PATH" \
#     --video_out_path "$OUTPUT_PATH" \
#     --log_file "$LOG_PATH" \
#     --enable_diycache \
#     --first_step_offset 1 \
#     --last_step_offset 4 \

# python -m scripts.inference_dicache_0604 \
#     --unet_config_path "configs/unet/second_stage.yaml" \
#     --inference_ckpt_path "$OFFICIAL_CKPT" \
#     --inference_steps 20 \
#     --guidance_scale "$GUIDANCE_SCALE" \
#     --video_path "$VIDEO_PATH" \
#     --audio_path "$AUDIO_PATH" \
#     --video_out_path "$OUTPUT_PATH" \
#     --log_file "$LOG_PATH" \
#     --enable_diycache \
#     --first_step_offset 1 \
#     --last_step_offset 4 \
#     # --show_stats \
#     # --residual_update_mode native \
#     # --ema_alpha 0.7 \
#     # --output_orig_face_video \
#     # --output_small_face_video \