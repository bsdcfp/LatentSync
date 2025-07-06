FROM harbor.shopeemobile.com/aip/platform/aip-spex-sidecar:v03cda51 as builder
FROM nvcr.io/nvidia/tensorrt:23.04-py3

USER root

COPY build_env /workspace/build_env
COPY productscene /workspace/productscene
COPY *.py /workspace/
COPY *.sh /workspace/
# COPY requirements.txt /workspace/
COPY --from=builder /src/aip-spex-sidecar /workspace/sidecar/aip-spex-sidecar

WORKDIR /workspace

# RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --trusted-host pypi.shopee.io -i http://pypi.shopee.io/simple/ aip-oneservice==0.4.9
RUN pip install --no-cache-dir --trusted-host pypi.shopee.io -i http://pypi.shopee.io/simple/ mms-sdk==1.0.30

ENV AIP_BIN=/usr/local/bin/
RUN chmod +x setup.sh
RUN ./setup.sh

RUN apt-get update && apt-get install -y libgl1

# NOTE: PROMETHEUS_MULTIPROC_DIR https://github.com/prometheus/client_python，用于监控性能
ENV prometheus_multiproc_dir /workspace/metrics
RUN mkdir -p /workspace/metrics

RUN bash build_env/init_hunyuan_env.sh /workspace

# CMD ["bash", "build_env/init_ais_run.sh /workspace"]

CMD ["bash", "start_server.sh"]
