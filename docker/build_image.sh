#!/bin/bash  
# set -x
# 提取输入参数  
dockerfile=$1  
# 设置工作目录，如果没有提供则默认为当前目录  
working_directory=${2:-.}  

# 检查是否提供 Dockerfile 参数  
if [ -z "$dockerfile" ]; then  
    echo "Error: A Dockerfile must be provided as the first argument."  
    echo "Usage: $0 <Dockerfile> [working_directory]"  # 提供用法说明  
    exit 1  
fi  

# 检查文件是否存在且可读取  
if [ ! -f "$dockerfile" ]; then  
    echo "Error: The specified Dockerfile '$dockerfile' does not exist or is not a file."  
    exit 1  
fi  

# 输出确认信息  
echo -e "\n================================================="  
echo -e "                 Docker Build Script               "  
echo -e "=================================================\n"  

echo "Using Dockerfile: $dockerfile"  
echo "Working directory: $working_directory"  

# 提取新的版本号和仓库名称  
new_version=${dockerfile##*-}  
repo_name=$(awk 'NR==1 {sub(/^FROM /, ""); sub(/:[^ ]+$/, ""); print}' "${dockerfile}")  
old_version=$(awk 'NR==1 {match($0, /:[^ ]+/); print substr($0, RSTART+1, RLENGTH-1)}' "${dockerfile}")  

# 输出版本变更信息  
echo -e "\n================================================="  
echo -e "                Image Version Change              "  
echo -e "=================================================\n"  

echo "FROM : ${repo_name}:${old_version}"  
echo "TO   : ${repo_name}:${new_version}"  

# 构建 Docker 镜像的参数  
docker_args="-f $dockerfile --build-arg uid=$(id -u) --build-arg gid=$(id -g) --tag=$repo_name:$new_version $working_directory"  

# 打印 Docker 构建信息  
echo -e "\n================================================="  
echo -e "                    Docker Build                   "  
echo -e "=================================================\n"  

# 显示构建命令  
echo "Building Docker image:"  
echo -e "> docker build ${docker_args}\n"  

# 运行构建命令（如果需要的话，可以取消注释）  
# docker build $docker_args  
# 检查是否存在具有相同仓库和标签的镜像  
if docker images | grep -q "${repo_name} *${new_version}"; then  
    echo "Warning: An image with the name ${repo_name}:${new_version} already exists."  
    echo "You may want to manually remove it or modify the TAG name."  
    echo "To remove the existing image, use:"  
    echo "    docker rmi ${repo_name}:${new_version}"  
    echo "To modify the TAG name, change the 'new_version' variable in your script."  
    exit 1  # 退出脚本，防止继续构建 
else  
    # 运行构建命令  
    docker build $docker_args 
    if [ $? -ne 0 ]; then  
        echo "Error: Docker build failed."  
        exit 1  
    fi  
fi  

# 提供后续操作的提示  
echo -e "\n================================================="  
echo -e "              Image Tagging and Publishing        "  
echo -e "=================================================\n"  

echo "If you want to change the image tag, use the following command:"  
echo "    docker tag <CURRENT_NAME> <NEW_NAME>\n"  

echo "If you want to publish this image, execute the command:"  
echo "    docker push ${repo_name}:${new_version}"  