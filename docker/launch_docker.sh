#!/bin/bash  

# ---- 帮助信息 ----
show_help() {
  echo "用法: $0 [rm] [--name 容器名] [--image 镜像名] [--workspace 路径] [-h|--help]"
  echo "参数说明:"
  echo "  rm                先删除同名容器（可选）"
  echo "  --name NAME       指定容器名（默认: 自动生成）"
  echo "  --image IMAGE     指定镜像名（默认: 官方 pytorch2.7-cu12.6）"
  echo "  --workspace PATH  指定工作区路径（默认: \$HOME）"
  echo "  -h, --help        显示本帮助信息"
  exit 0
}

# ---- 默认前缀与时间戳自动C_NAME ----  
C_PREFIX="fuping-dev"  
TIMESTAMP=$(date +"%Y%m%d-%H%M")  
DEFAULT_C_NAME="${C_PREFIX}-${TIMESTAMP}"
DEFAULT_IMAGE_URL="harbor.shopeemobile.com/aip/aip-image-hub/aip-prod/projects/85/pytorch2.7-cu12.6-py3.10-trt10.3:vscode"

# ---- 解析参数 ----  
RM_FLAG=""  
USER_C_NAME=""  
USER_IMAGE_URL=""  

# 检查 -h/--help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  show_help
fi

while [[ $# -gt 0 ]]; do  
  case "$1" in  
    rm)  
      RM_FLAG="1"  
      shift  
      ;;  
    --name)  
      USER_C_NAME="$2"  
      shift 2  
      ;;  
    --image)  
      USER_IMAGE_URL="$2"  
      shift 2  
      ;;  
    --workspace)
      USER_WORKSPACE_PATH="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      ;;
    *)  
      shift  
      ;;  
  esac  
done  

# ---- 容器名和镜像名：优先用参数，否则用默认 ----  
C_NAME="${USER_C_NAME:-$DEFAULT_C_NAME}"  
IMAGE_URL="${USER_IMAGE_URL:-$DEFAULT_IMAGE_URL}"  
WORKSPACE_PATH="${USER_WORKSPACE_PATH:-$HOME}"
DATASETS="$WORKSPACE_PATH/datasets"  
MODEL_ZOO="$WORKSPACE_PATH/model_zoo"  
VSCODE_SERVER="$HOME/.vscode-server/"
CURSOR_SERVER="$HOME/.cursor-server/"

if [[ -n "$RM_FLAG" ]]; then  
  docker rm -f "${C_NAME}"  
fi  

FLAGS="-itd --privileged"  

# 构建条件挂载参数
VSCODE_MOUNT=""
CURSOR_MOUNT=""

if [[ -d "${VSCODE_SERVER}" ]]; then
  VSCODE_MOUNT="-v ${VSCODE_SERVER}:/root/.vscode-server"
fi

if [[ -d "${CURSOR_SERVER}" ]]; then
  CURSOR_MOUNT="-v ${CURSOR_SERVER}:/root/.cursor-server"
fi

cmd="docker run -u root $FLAGS --name ${C_NAME} \
  --gpus all \
  --shm-size=16g \
  --net=host \
  -v ${WORKSPACE_PATH}:/home/fuping.chu \
  -v ${DATASETS}:/datasets \
  -v ${MODEL_ZOO}:/model_zoo \
  -v /etc/localtime:/etc/localtime:ro \
  -v /etc/timezone:/etc/timezone:ro \
  ${VSCODE_MOUNT} \
  ${CURSOR_MOUNT} \
  ${IMAGE_URL} bash"  

echo "$cmd"  
eval "$cmd"  
