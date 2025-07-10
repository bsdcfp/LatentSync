######################################################################
#
# Copyright (c) 2025 Shopee Inc. All Rights Reserved.
#
######################################################################

"""
file: server_http.py
"""

import cv2
import glob
import os
import time
import torch
import numpy as np
import base64
import gzip
import orjson
from aipinfer import logger, exceptions
from typing import Optional, Dict, List, Any

from latentsync.backends.snn_predictor import SNNPredictor

from oneservice import BaseProcessor, ResParams, MonitorTimer, ReqParams, server

import threading
_predictor_lock = threading.Lock()


# def json_to_tensor_decompressed(json_data):
#     """从JSON恢复tensor"""
#     # 解码
#     compressed = base64.b64decode(json_data['data'])
#     # 解压
#     if json_data.get('compression') == 'gzip':
#         tensor_bytes = gzip.decompress(compressed)
#     else:
#         tensor_bytes = compressed
#     # 重建tensor
#     np_dtype = np.dtype(json_data['dtype'])
#     tensor = np.frombuffer(tensor_bytes, dtype=np_dtype)
#     return tensor.reshape(json_data['shape'])

# def tensor_to_json_compressed(tensor, compression='gzip'):
#     """将tensor压缩后转为JSON友好格式"""
#     if isinstance(tensor, torch.Tensor):
#         tensor = tensor.cpu().numpy()
#     tensor_bytes = tensor.tobytes()
#     if compression == 'gzip':
#         compressed = gzip.compress(tensor_bytes)
#     else:
#         compressed = tensor_bytes
#     encoded = base64.b64encode(compressed).decode('utf-8')
#     return {
#         'data': encoded,
#         'shape': tensor.shape,
#         'dtype': str(tensor.dtype),
#         'compression': compression
#     }

class Processor(BaseProcessor):
    def __init__(self, config_file: str):
        # NOTE: 创建视频生成Predictor - 使用完整模式进行推理
        self._video_predictor = SNNPredictor(config_file=config_file, lightweight_mode=False)
        
        # 简单的初始化，不需要复杂的内存池
        
        # 执行预热推理，避免第一次请求的延迟
        logger.info("Performing warmup inference to initialize models...")
        self._warmup_inference()
        logger.info("Warmup inference completed")

    def __call__(self, req_params: ReqParams, res_params: ResParams):
        with _predictor_lock:
            # NOTE: 解析请求参数获取推理所需参数
            with MonitorTimer("decode"):
                # 获取实际的请求数据 - 客户端发送的是嵌套结构
                # try:
                #     raw_request_bytes = req_params.data
                #     payload_dict = orjson.loads(raw_request_bytes)
                # except Exception as e:
                #     res_params.err_num = exceptions.NUM_ILLEGAL_ARGS
                #     res_params.err_msg = f"Invalid request data: {str(e)}"
                #     return
                # # data = request_data.get("data", request_data)  # 兼容两种格式
                data = req_params.data
                if not isinstance(data, dict):
                    res_params.err_num = exceptions.NUM_ILLEGAL_ARGS
                    res_params.err_msg = f"Request field 'data' must be a dictionary. Found: {type(data).__name__}"
                    return
                # 检查必需参数
                if data.get("chunk_faces") is None or data.get("chunk_whisper_features") is None:
                    res_params.err_num = exceptions.NUM_ILLEGAL_ARGS
                    res_params.err_msg = "chunk_faces/chunk_whisper_features cann't find in request"
                    return

                chunk_faces = data["chunk_faces"]
                chunk_whisper_features = data["chunk_whisper_features"]
                
                # # 自动解压 chunk_faces
                # if isinstance(chunk_faces, dict) and 'data' in chunk_faces:
                #     chunk_faces = json_to_tensor_decompressed(chunk_faces)
                # # 自动解压 chunk_whisper_features
                # if isinstance(chunk_whisper_features, list) and len(chunk_whisper_features) > 0 and isinstance(chunk_whisper_features[0], dict) and 'data' in chunk_whisper_features[0]:
                #     chunk_whisper_features = [json_to_tensor_decompressed(w) for w in chunk_whisper_features]

                # 获取推理参数（可选，有默认值）
                num_frames = data.get("num_frames", 16)
                num_inference_steps = data.get("num_inference_steps", 20)
                guidance_scale = data.get("guidance_scale", 1.0)
                weight_dtype = data.get("weight_dtype", "float16")
                eta = data.get("eta", 0.0)
                height = data.get("height", 256)
                width = data.get("width", 256)
                debug = data.get("debug", False)

                # 验证参数
                # if not isinstance(chunk_faces, list) or not isinstance(chunk_whisper_features, list):
                #     res_params.err_num = exceptions.NUM_ILLEGAL_ARGS
                #     res_params.err_msg = "chunk_faces/chunk_whisper_features must be lists"
                #     return

                # 简单的数据转换
                convert_start = time.time()
                
                # 直接创建tensor
                if isinstance(chunk_faces, list):
                    chunk_faces_tensor = torch.tensor(chunk_faces, dtype=torch.float32, device="cuda").contiguous()
                else:
                    chunk_faces_tensor = torch.from_numpy(chunk_faces).to(torch.float32).to("cuda").contiguous() if isinstance(chunk_faces, np.ndarray) else chunk_faces
                    
                # 处理whisper特征
                if isinstance(chunk_whisper_features, list) and len(chunk_whisper_features) > 0:
                    if isinstance(chunk_whisper_features[0], list):
                        chunk_whisper_tensor = [torch.tensor(w, dtype=torch.float32, device="cuda").contiguous() for w in chunk_whisper_features]
                    elif isinstance(chunk_whisper_features[0], np.ndarray):
                        chunk_whisper_tensor = [torch.from_numpy(w).to(torch.float32).to("cuda").contiguous() for w in chunk_whisper_features]
                    else:
                        chunk_whisper_tensor = chunk_whisper_features
                else:
                    chunk_whisper_tensor = chunk_whisper_features

                convert_time = time.time() - convert_start
                logger.info(f"Data conversion completed in {convert_time:.3f}s")

                # 转换weight_dtype字符串为torch.dtype
                if weight_dtype == "float16":
                    weight_dtype = torch.float16
                elif weight_dtype == "float32":
                    weight_dtype = torch.float32
                else:
                    weight_dtype = torch.float16  # 默认值

            with MonitorTimer("predictor"):
                try:
                    # NOTE: 使用与local_test.py相同的推理方式
                    logger.info(f"Starting chunk inference...")
                    
                    # 添加推理时间监控
                    inference_start = time.time()
                    
                    # 执行推理
                    decoded_latents = self._video_predictor._engine.model_inference_by_chunk(
                        chunk_faces=chunk_faces_tensor,
                        chunk_whisper_features=chunk_whisper_tensor,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        weight_dtype=weight_dtype,
                        eta=eta,
                        generator=None,
                        callback=None,
                        callback_steps=1,
                        height=height,
                        width=width
                    )
                except Exception as e:
                    logger.error(f"Video inference failed: {str(e)}")
                    res_params.err_num = exceptions.NUM_BACKEND_ERROR
                    res_params.err_msg = f"Video inference error: {str(e)}"
                    return
                inference_time = time.time() - inference_start
                logger.info(f"Model inference completed in {inference_time:.3f}s")                

            with MonitorTimer("encoder"):
                # 将结果转换为可序列化的格式
                # 注意：decoded_latents是torch.Tensor，需要转换为numpy数组
                serialize_start = time.time()
                frame_np = decoded_latents.cpu().numpy()
    
                # 设置返回结果
                result_data = {
                    "decoded_latents": frame_np,
                    "num_frames": num_frames,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "height": height,
                    "width": width,
                    "debug": debug
                }
                final_payload_bytes = orjson.dumps(result_data, option=orjson.OPT_SERIALIZE_NUMPY)
                serialize_time = time.time() - serialize_start
                logger.info(f"Result serialization with orjson completed in {serialize_time:.3f}s")
                
                # 4. 调试和日志记录
                if debug:
                    uncompressed_bytes = frame_np.nbytes
                    compressed_bytes = len(final_payload_bytes) # 整个 payload 的大小
                    logger.info(f"[DEBUG] Original numpy array size: {uncompressed_bytes/1024/1024:.2f} MB")
                    logger.info(f"[DEBUG] Total orjson payload size: {compressed_bytes/1024/1024:.2f} MB")

            res_params.result = final_payload_bytes

            return

    def _warmup_inference(self):
        """执行预热推理，初始化模型和缓存"""
        try:
            # 创建小的测试数据
            batch_size = 1
            num_frames = 16
            height = 256
            width = 256
            
            # 创建随机测试数据 - 使用正确的维度
            test_faces = torch.randn(num_frames, 3, height, width, dtype=torch.float32, device="cuda")
            # whisper特征应该是3D的，形状为 [seq_len, hidden_dim]
            test_whisper = [torch.randn(1500, 384, dtype=torch.float32, device="cuda") for _ in range(num_frames)]
            
            logger.info("Starting warmup inference with test data...")
            warmup_start = time.time()
            
            # 执行预热推理
            _ = self._video_predictor._engine.model_inference_by_chunk(
                chunk_faces=test_faces,
                chunk_whisper_features=test_whisper,
                num_frames=num_frames,
                num_inference_steps=5,  # 使用较少的步数进行预热
                guidance_scale=1.0,
                weight_dtype=torch.float16,
                eta=0.0,
                generator=None,
                callback=None,
                callback_steps=1,
                height=height,
                width=width,
                debug=False,
            )
            
            warmup_time = time.time() - warmup_start
            logger.info(f"Warmup inference completed in {warmup_time:.3f}s")
            
            # 清理测试数据，但不清理GPU缓存以保持模型状态
            del test_faces, test_whisper
            
        except Exception as e:
            logger.warn(f"Warmup inference failed: {str(e)} - this is not critical")


# NOTE: 从MaaS标准模型路径获取config文件
def find_config_yaml_files(dir_path: str):
    return glob.glob(f"{dir_path}/**/config.yaml", recursive=True)


aip_model_dir = os.environ.get("AIP_MODEL_PATH", "./models")
logger.info(f"[CONFIG] Model directory: {aip_model_dir}")

config_files = find_config_yaml_files(aip_model_dir)
logger.info(f"[CONFIG] Found {len(config_files)} config file(s): {config_files}")

config_file = config_files[0]
logger.info(f"[CONFIG] Using config file: {config_file}")

# # NOTE: Flask的应用程序入口方式
logger.info(f"[SERVER] Initializing Video Generation Processor with config: {config_file}")
server.register(Processor(config_file=config_file), ReqParams, ResParams)
app = server.app

if __name__ == "__main__":
    server.app.run(host="127.0.0.1", port=80, threaded=True)