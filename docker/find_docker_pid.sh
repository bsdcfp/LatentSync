#!/bin/bash  

# 检查是否提供了PID参数  
if [ $# -eq 0 ]; then  
    echo "错误: 请提供PID作为参数"  
    echo "用法: $0 <PID>"  
    exit 1  
fi  

PID="$1"  
found=false  

echo "正在查找PID ${PID}所属的容器..."  

# 遍历所有运行中的容器  
for container_id in $(docker ps -q); do  
    container_name=$(docker inspect --format="{{.Name}}" "$container_id" | sed 's/^\///')  
    
    # 检查PID是否在此容器中  
    if docker top "$container_id" | grep -q "$PID"; then  
        found=true  
        echo "============================================="  
        echo "找到PID ${PID}在容器中:"  
        echo "容器ID: $container_id"  
        echo "容器名称: $container_name"  
        echo "============================================="  
        echo "进程上下文(前后5行):"  
        docker top "$container_id" | grep -C 5 "$PID"  
        echo "============================================="  
    fi  
done  

# 如果没有找到PID  
if [ "$found" = false ]; then  
    echo "没有找到PID ${PID}所属的Docker容器"  
    exit 2  
fi  

exit 0  