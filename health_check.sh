#!/bin/bash
set -exo pipefail

if [[ $AIP_USER_PROTOCOLS == http* ]]; then
  echo "health check http service...\n"
  curl --fail http://0.0.0.0:80/api/ping
# elif [[ $AIP_USER_PROTOCOLS == grpc* ]]; then
#   echo "health check  grpc service...\n"
#   grpcurl -d '{"service": "maascheck"}' -plaintext 0.0.0.0:50051 AIPService/HealthCheck
elif [[ $AIP_USER_PROTOCOLS == spex ]]; then
  echo "health check spex service...\n"
  curl --fail http://0.0.0.0:80/api/ping && curl --fail http://0.0.0.0:15090/health
else
  echo "not supported protocal: $AIP_USER_PROTOCOLS"
  exit 1
fi