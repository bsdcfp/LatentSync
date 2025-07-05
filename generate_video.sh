#!/bin/bash

# 在线视频生成脚本 - 修正版本
# 支持位置参数 + 环境变量的混合方式

set -e

# =========================
# 简单帮助信息
# =========================
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    cat << EOF
🎬 在线视频生成工具

用法: $0 [音频文件] [输出目录] [帧率] [推理步数] [随机种子]

位置参数:
  1. 音频文件      输入音频路径 (必需)
  2. 输出目录      输出目录路径 (可选，默认自动生成)
  3. 帧率         目标帧率 (可选，默认25)
  4. 推理步数      推理步数 (可选，默认20)
  5. 随机种子      随机种子 (可选，默认1247)

环境变量 (高级选项):
  SCRIPT_DIR       工作目录
  AUXILIARY_DIR    辅助文件目录
  ENABLE_CACHE     启用缓存 (默认: true)

示例:
  $0 audio.mp3                          # 最简用法
  $0 audio.mp3 ./output                 # 指定输出目录
  $0 audio.mp3 ./output 30              # 30fps
  $0 audio.mp3 ./output 30 25           # 30fps, 25步推理
  $0 audio.mp3 ./output 30 25 42        # 30fps, 25步, 种子42
  
  # 环境变量方式
  ENABLE_CACHE=false $0 audio.mp3
EOF
    exit 0
fi

# =========================
# 参数解析 - 位置参数优先
# =========================

# 基础路径
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
AUXILIARY_DIR="${AUXILIARY_DIR:-/model_zoo/latentsync_online_simple_auxilary/}"

# 必需参数
AUDIO_PATH="$1"
if [[ -z "$AUDIO_PATH" ]]; then
    echo "❌ 错误: 必须指定音频文件"
    echo "💡 用法: $0 <音频文件> [输出目录] [帧率] [推理步数] [种子]"
    echo "💡 帮助: $0 --help"
    exit 1
fi

# 可选参数 - 位置参数优先，然后是环境变量，最后是默认值
OUTPUT_DIR="${2:-${OUTPUT_DIR:-$SCRIPT_DIR/outputs/$(date +%Y%m%d)}}"
TARGET_FPS="${3:-${TARGET_FPS:-25}}"
INFERENCE_STEPS="${4:-${INFERENCE_STEPS:-20}}"
SEED="${5:-${SEED:-1247}}"

# 高级选项 - 仅环境变量
ENABLE_CACHE="${ENABLE_CACHE:-true}"

# =========================
# 基本验证
# =========================
if [[ ! -f "$AUDIO_PATH" ]]; then
    echo "❌ 音频文件不存在: $AUDIO_PATH"
    exit 1
fi

if [[ ! -d "$AUXILIARY_DIR" ]]; then
    echo "❌ 辅助目录不存在: $AUXILIARY_DIR"
    echo "💡 请设置环境变量: export AUXILIARY_DIR=/path/to/aux"
    exit 1
fi

# 数值验证
if ! [[ "$TARGET_FPS" =~ ^[0-9]+$ ]] || [ "$TARGET_FPS" -lt 1 ] || [ "$TARGET_FPS" -gt 60 ]; then
    echo "❌ 无效的帧率: $TARGET_FPS (应为1-60之间的整数)"
    exit 1
fi

if ! [[ "$INFERENCE_STEPS" =~ ^[0-9]+$ ]] || [ "$INFERENCE_STEPS" -lt 1 ] || [ "$INFERENCE_STEPS" -gt 100 ]; then
    echo "❌ 无效的推理步数: $INFERENCE_STEPS (应为1-100之间的整数)"
    exit 1
fi

# =========================
# 自动构建路径
# =========================
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 模型路径
WEIGHTS_DIR="$AUXILIARY_DIR/weights/latentsync"
VAE_WEIGHTS_DIR="/model_zoo/sd-vae-ft-mse/"
TEMPLATE_DIR="$AUXILIARY_DIR/0612_online_templete_female_32fps_resize_720_1560"

# 检查关键文件
required_files=(
    "$SCRIPT_DIR/scripts/inference_online_0618.py"
    "$WEIGHTS_DIR/latentsync_unet.pt"
    "$WEIGHTS_DIR/whisper/tiny.pt"
    "$TEMPLATE_DIR/template_info_0618/connected_videos.json"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ 缺少必需文件: $file"
        exit 1
    fi
done

# 输出路径
mkdir -p "$OUTPUT_DIR"
VIDEO_OUT_PATH="$OUTPUT_DIR/video.mp4"
MASK_OUTPUT_PATH="$OUTPUT_DIR/mask.mp4"
LOG_FILE="$OUTPUT_DIR/inference.log"

# =========================
# 显示配置并执行
# =========================
echo "🚀 开始生成视频..."
echo "🎵 音频: $(basename "$AUDIO_PATH")"
echo "📁 输出: $OUTPUT_DIR"
echo "⚙️  配置: ${TARGET_FPS}fps | ${INFERENCE_STEPS}步 | 种子${SEED} | 缓存$([ "$ENABLE_CACHE" = "true" ] && echo "启用" || echo "禁用")"
echo "============================================================"

# 构建完整的Python命令
PYTHON_CMD="python $SCRIPT_DIR/scripts/inference_online_0618.py \\
    --unet_config_path $SCRIPT_DIR/configs/unet/second_stage.yaml \\
    --inference_ckpt_path $WEIGHTS_DIR/latentsync_unet.pt \\
    --whisper_model_path $WEIGHTS_DIR/whisper/tiny.pt \\
    --vae_model_path $VAE_WEIGHTS_DIR \\
    --audio_path $AUDIO_PATH \\
    --video_out_path $VIDEO_OUT_PATH \\
    --video_template_dir $TEMPLATE_DIR/videos \\
    --connected_info_json $TEMPLATE_DIR/template_info_0618/connected_videos.json \\
    --video_info_json $TEMPLATE_DIR/template_info_0618/0612_video_templete_32fps_info_v2.json \\
    --shot_video_json_path $TEMPLATE_DIR/template_info_0618/connected_videos_other_short.json \\
    --video_cache_dir $TEMPLATE_DIR/videos_features \\
    --mask_base_dir $TEMPLATE_DIR/videos_mask \\
    --mask_output_path $MASK_OUTPUT_PATH \\
    --target_fps $TARGET_FPS \\
    --transition_frames 20 \\
    --inference_steps $INFERENCE_STEPS \\
    --guidance_scale 1.0 \\
    --seed $SEED \\
    --log_file $LOG_FILE \\
    --first_step_offset 1 \\
    --last_step_offset 4" 
    #--debug_pipeline"

# 如果启用缓存，添加--enable_diycache参数
if [ "$ENABLE_CACHE" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --enable_diycache"
fi

echo "🔧 执行的Python命令："
echo "$PYTHON_CMD"
echo "============================================================"

# 执行Python命令，确保PYTHONPATH环境变量被传递
PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" eval "$PYTHON_CMD"

echo "============================================================"
if [[ -f "$VIDEO_OUT_PATH" ]]; then
    echo "✅ 生成成功！"
    echo "📹 视频: $VIDEO_OUT_PATH ($(du -h "$VIDEO_OUT_PATH" | cut -f1))"
    echo "🎭 遮罩: $MASK_OUTPUT_PATH"
    echo "📋 日志: $LOG_FILE"
else
    echo "❌ 生成失败，查看日志: $LOG_FILE"
    exit 1
fi
