#!/bin/bash
# Run the docker container with the specified parameters

C_NAME="fuping-swap-face-test-docker-image"



FLAGS="--rm -it --privileged "

WORKSPACE_PATH="$HOME"

IMAGE_URL=$1
if [ -z "$IMAGE_URL" ]; then  
    echo "ERROR: No image URL provided."  
    echo "Usage: $0 <IMAGE_URL>"  
    exit 1  # 以非零状态退出，表示错误  
fi  
ENTRY_POINT=${2:-bash}

# 确保 ENTRY_POINT 以 'bash' 开始  
if [[ $ENTRY_POINT != bash* ]]; then  
    ENTRY_POINT="bash -c $ENTRY_POINT"  
fi  

RM=$3
if [ "${RM}" = "rm" ];then
	  rm_cmd="docker rm -f ${C_NAME}"
    $rm_cmd

fi

# 输出参数以确认  
echo "Image URL: $IMAGE_URL"  
echo "Entry Point: $ENTRY_POINT"  

DATASETS="$WORKSPACE_PATH/datasets"

# MODEL_ZOO="/data1/fuping.chu/model_zoo"
MODEL_ZOO="$WORKSPACE_PATH/model_zoo"

cmd="docker run -u root $FLAGS --name ${C_NAME} \
  --gpus all \
  --shm-size=16g \
  --net=host \
  -v ${WORKSPACE_PATH}:/home/run \
  -v ${DATASETS}:/datasets \
  -v ${MODEL_ZOO}:/model_zoo \
  ${IMAGE_URL} ${ENTRY_POINT} "
echo $cmd
$cmd

# echo "========================================="  
# echo "         Docker Exec Command             "  
# echo "========================================="  
# echo "                                         "  
# echo "docker exec -it ${C_NAME} bash           "  
# echo "docker rm -f ${C_NAME}                   "  
# echo "                                         "  
# echo "========================================="  