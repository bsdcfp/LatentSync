#!/bin/bash
set -exo pipefail

# For gRPC healthcheck
wget https://github.com/fullstorydev/grpcurl/releases/download/v1.8.7/grpcurl_1.8.7_linux_x86_64.tar.gz \
    && tar -xvf grpcurl_1.8.7_linux_x86_64.tar.gz -C $AIP_BIN \
    && chmod +x $AIP_BIN/grpcurl && rm -rf grpcurl_1.8.7_linux_x86_64.tar.gz