# 安装
```sh
cd tensorrt
pip install -e . 
```
> 注意：模型库只有 L40s 的 TensorRT Engine

# 模型下载
url: https://ais.mlp.shopee.io/projects/model-repo/model-repo/detail/versionDetail?current=1&modelId=37793&pageSize=10&projectId=100209&tab=basic&versionId=52128 
```sh
ais model download \
        --model_id=37793 \
        --version_id=52128 \
        --output_path="./latentsync_merge_all-v2/" \
        --project=100209
```

# 本地测试
```sh
python test/local_test.py \
    --model-path /model_zoo/latentsync_merge_all-v2 \
    --audio-file /model_zoo/latentsync_merge_all-v2/input_example/beauty_v2_20s.MP3 \
    --debug-pipeline 
```
> 注意：这里的`--model-path`和/model_zoo/latentsync_merge_all-v2/config.yaml 里的配置一致

打开耗时打印：修改`/model_zoo/latentsync_merge_all-v2/config.yaml`里的`verbose: True`