#!/bin/bash
set -exo pipefail

model_dir=${AIP_MODEL_PATH:-models}
PROCESS_NUM=${PROCESS_NUM:-2}
config_file=$(find -L ${model_dir} -name "config.yaml")
echo "Running service using config version $config_file...\n"

# mass log storage
if [[ -z "${POD_NAME}" ]]; then
  echo "Environment variable POD_NAME not set"
else
  # 确保目标日志目录存在
  if [[ ! -d "/root/log/$POD_NAME" ]]; then
    echo "Creating log directory: /root/log/$POD_NAME"
    mkdir -p /root/log/$POD_NAME
  else
    echo "Log directory already exists: /root/log/$POD_NAME"
  fi
  
  # 如果log目录已存在，先删除它
  if [[ -L "log" ]]; then
    rm log
  elif [[ -d "log" ]]; then
    rm -rf log
  fi
  ln -s /root/log/$POD_NAME log
fi

if [[ $AIP_USER_PROTOCOLS == http* ]]; then
  echo "launching http service\n"
  if [[ -z "${AIP_DEBUG_BIND}" ]]; then
    oneservice -c $config_file server_http:app
  else
    oneservice -c $config_file -b ${AIP_DEBUG_BIND} server_http:app
  fi
# elif [[ $AIP_USER_PROTOCOLS == grpc* ]]; then
#   echo "launching grpc service\n"
#   if [[ -z "${AIP_DEBUG_BIND}" ]]; then
#     python3 server_grpc.py -c $config_file # --workers=$PROCESS_NUM
#   else
#     python3 server_grpc.py -c $config_file -b ${AIP_DEBUG_BIND}
#   fi
elif [[ $AIP_USER_PROTOCOLS == spex ]]; then
  echo "launching http service\n"
  oneservice -c $config_file server_http:app &
  pid=$!
  echo "launching spex sidecar service\n"
  chmod +x /workspace/sidecar/aip-spex-sidecar
  /workspace/sidecar/aip-spex-sidecar &
  sidecar_pid=$!
  trap 'echo "Stopping services..."; kill -9 $sidecar_pid $pid; exit' SIGINT SIGTERM
  wait $sidecar_pid
  wait $pid
  echo "service stopped"
else
  echo "not supported protocal: $AIP_USER_PROTOCOLS"
fi