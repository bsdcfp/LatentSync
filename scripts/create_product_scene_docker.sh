#!/bin/bash

# 容器名称
local_name=aip_product_scene
version=$1

container_name=$local_name-$1

# 检查容器是否存在
if docker inspect "$container_name" >/dev/null 2>&1; then
    # 容器存在，直接打开
    echo "$container_name"
    docker start $container_name
    docker exec -it $container_name /bin/bash
else
    # 容器不存在，进行其他操作
    echo "Container does not exist."
    docker run \
       --name $container_name \
       -it \
       --gpus all \
       -v `pwd`:/home/workcode \
       --workdir /home/workcode \
       harbor.shopeemobile.com/aip/aip_docker_test/aip_aigc_product_scene_docker:$1 /bin/bash
fi