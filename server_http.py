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

from aipinfer import logger, exceptions
from typing import Optional, Dict, List, Any

from latentsync.backends.snn_predictor import SNNPredictor

from oneservice import BaseProcessor, ResParams, MonitorTimer, ReqParams, server


class Processor(BaseProcessor):
    def __init__(self, config_file: str):
        # NOTE: 创建视频生成Predictor
        self._video_predictor = SNNPredictor(config_file=config_file)

    def __call__(self, req_params: ReqParams, res_params: ResParams):
        # NOTE: 解析请求参数获取推理所需参数
        with MonitorTimer("decode"):
            data = req_params.data

            # 检查必需参数
            if data.get("chunk_faces") is None or data.get("chunk_whisper_features") is None:
                res_params.err_num = exceptions.NUM_ILLEGAL_ARGS
                res_params.err_msg = "chunk_faces/chunk_whisper_features cann't find in request"
                return

            chunk_faces = data["chunk_faces"]
            chunk_whisper_features = data["chunk_whisper_features"]
            
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
            if not isinstance(chunk_faces, list) or not isinstance(chunk_whisper_features, list):
                res_params.err_num = exceptions.NUM_ILLEGAL_ARGS
                res_params.err_msg = "chunk_faces/chunk_whisper_features must be lists"
                return

            # 转换序列化数据为tensor格式
            convert_start = time.time()
            
            if isinstance(chunk_faces[0], list):
                chunk_faces_tensor = torch.tensor(chunk_faces, dtype=torch.float32)
            else:
                chunk_faces_tensor = chunk_faces
                
            # 处理whisper特征 - 这是一个列表的列表，每个元素需要单独转换
            if isinstance(chunk_whisper_features, list) and len(chunk_whisper_features) > 0:
                if isinstance(chunk_whisper_features[0], list):
                    # 每个元素是一个tensor的序列化数据
                    chunk_whisper_tensor = [torch.tensor(w, dtype=torch.float32) for w in chunk_whisper_features]
                else:
                    chunk_whisper_tensor = chunk_whisper_features
            else:
                chunk_whisper_tensor = chunk_whisper_features

            convert_time = time.time() - convert_start
            logger.info(f"Data conversion completed in {convert_time:.3f}s")

            # 获取设备信息并移动tensor到正确的设备
            device = self._video_predictor._engine.get_device()
            logger.info(f"Moving tensors to device: {device}")
            
            move_start = time.time()
            
            # 移动faces tensor到设备
            if isinstance(chunk_faces_tensor, torch.Tensor):
                chunk_faces_tensor = chunk_faces_tensor.to(device)
            
            # 移动whisper特征到设备
            if isinstance(chunk_whisper_tensor, list):
                chunk_whisper_tensor = [w.to(device) if isinstance(w, torch.Tensor) else w for w in chunk_whisper_tensor]
            
            move_time = time.time() - move_start
            logger.info(f"Tensor device movement completed in {move_time:.3f}s")

            # 转换weight_dtype字符串为torch.dtype
            if weight_dtype == "float16":
                weight_dtype = torch.float16
            elif weight_dtype == "float32":
                weight_dtype = torch.float32
            else:
                weight_dtype = torch.float16  # 默认值

        with MonitorTimer("predictor"):
            try:
                # NOTE: 执行单个chunk的推理
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
                    width=width,
                    debug=debug,
                )
                
                inference_time = time.time() - inference_start
                logger.info(f"Model inference completed in {inference_time:.3f}s")
                
                # 将结果转换为可序列化的格式
                # 注意：decoded_latents是torch.Tensor，需要转换为numpy数组
                serialize_start = time.time()
                frame_np = decoded_latents.cpu().numpy()
                serializable_frame = frame_np.tolist()  # 转换为Python列表以便JSON序列化
                serialize_time = time.time() - serialize_start
                logger.info(f"Result serialization completed in {serialize_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Video inference failed: {str(e)}")
                res_params.err_num = exceptions.NUM_BACKEND_ERROR
                res_params.err_msg = f"Video inference error: {str(e)}"
                return

        # 设置返回结果
        res_params.result = {
            "decoded_latents": serializable_frame,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "debug": debug
        }

        return


# NOTE: 从MaaS标准模型路径获取config文件
def find_config_yaml_files(dir_path: str):
    return glob.glob(f"{dir_path}/**/config.yaml", recursive=True)


aip_model_dir = os.environ.get("AIP_MODEL_PATH", "./models")
logger.info(f"[CONFIG] Model directory: {aip_model_dir}")

config_files = find_config_yaml_files(aip_model_dir)
logger.info(f"[CONFIG] Found {len(config_files)} config file(s): {config_files}")

config_file = config_files[0]
logger.info(f"[CONFIG] Using config file: {config_file}")

# NOTE: Flask的应用程序入口方式
logger.info(f"[SERVER] Initializing Video Generation Processor with config: {config_file}")
server.register(Processor(config_file=config_file), ReqParams, ResParams)
app = server.app

if __name__ == "__main__":
    server.app.run(host="127.0.0.1", port=80, threaded=True)