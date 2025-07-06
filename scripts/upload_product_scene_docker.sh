#!/bin/bash

arg_dockerfile=Dockerfile
arg_imagename=aip_aigc_product_scene_docker

docker_args="-f $arg_dockerfile --build-arg uid=$(id -u) --build-arg gid=$(id -g) --tag=$arg_imagename ."

echo "Building container:"
echo "> docker build $docker_args"
docker build --no-cache $docker_args

# docker tag SOURCE_IMAGE[:TAG] harbor.shopeemobile.com/aip/REPOSITORY[:TAG]

docker tag $arg_imagename harbor.shopeemobile.com/aip/aip_docker_test/$arg_imagename:$1

# docker push harbor.shopeemobile.com/aip/REPOSITORY[:TAG]

docker push harbor.shopeemobile.com/aip/aip_docker_test/$arg_imagename:$1