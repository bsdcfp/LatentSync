FROM harbor.shopeemobile.com/aip/platform/aip-spex-sidecar:v03cda51 as builder
FROM harbor.shopeemobile.com/aip/aip-image-hub/aip-prod/projects/100208/latentsync-service:ray-v5

USER root

RUN mkdir -p /workspace/LatentSync
COPY latentsync /workspace/LatentSync/latentsync
COPY *.py /workspace/LatentSync/
COPY *.sh /workspace/LatentSync/
COPY pyproject.toml /workspace/LatentSync/

# COPY requirements.txt /workspace/
COPY --from=builder /src/aip-spex-sidecar /workspace/sidecar/aip-spex-sidecar

WORKDIR /workspace/LatentSync

# 先安装系统依赖
RUN pip install --no-cache-dir --trusted-host pypi.shopee.io \
    -i http://pypi.shopee.io/simple/ aip-oneservice==0.4.9 \
    -i http://pypi.shopee.io/simple/ mms-sdk==1.0.30

# 再安装项目依赖
RUN pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install --upgrade pip
RUN pip uninstall -y opencv-python opencv-contrib-python \
    && pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple flash-attn==2.7.3 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple onnx_graphsurgeon==0.3.23 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple onnxsim==0.4.3 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime==1.19.0

ENV AIP_BIN=/usr/local/bin/

# NOTE: PROMETHEUS_MULTIPROC_DIR https://github.com/prometheus/client_python，用于监控性能
ENV prometheus_multiproc_dir /workspace/metrics
RUN mkdir -p /workspace/metrics

CMD ["bash", "start_server.sh"]
