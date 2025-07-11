#!/bin/bash

# 进入项目根目录
cd /home/work/hy_01/mmuplt-avatarv3/latent_sync

# 设置PYTHONPATH，确保latentsync包可被正确import
export PYTHONPATH=/home/work/hy_01/mmuplt-avatarv3/latent_sync:$PYTHONPATH

# 配置参数
UNET_CONFIG_PATH="configs/unet/second_stage.yaml"
INFERENCE_CKPT_PATH="/home/work/hy_01/weights/latentsync/latentsync_unet.pt"
WHISPER_MODEL_PATH="/home/work/hy_01/weights/latentsync/whisper/tiny.pt"
# INPUT_DIR="/home/work/hy_01/templates/0612_online_templete_female_32fps"
# OUTPUT_DIR="/home/work/hy_01/templates/0612_online_templete_female_32fps_features"
INPUT_DIR="/home/work/hy_01/templates/0612_online_templete_female_32fps_resize_720_1560/videos"
OUTPUT_DIR="/home/work/hy_01/templates/0612_online_templete_female_32fps_resize_720_1560/videos_features"
MAX_VIDEOS=-1  # -1表示全部，调试时可设为小于实际视频数

# 执行批量特征提取
python /home/work/hy_01/mmuplt-avatarv3/latent_sync/latentsync/preprocess_cache/preprocess_face_features_0613.py \
  --unet_config_path "$UNET_CONFIG_PATH" \
  --inference_ckpt_path "$INFERENCE_CKPT_PATH" \
  --whisper_model_path "$WHISPER_MODEL_PATH" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --max_videos $MAX_VIDEOS 