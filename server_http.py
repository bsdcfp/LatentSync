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
from queue import Queue, Empty
from threading import Thread, Lock
from concurrent.futures import Future
import orjson
from aipinfer import logger, exceptions
from typing import Optional, Dict, List, Any
import traceback
from latentsync.backends.snn_predictor import SNNPredictor

from oneservice import BaseProcessor, ResParams, MonitorTimer, ReqParams, server

import threading
_predictor_lock = threading.Lock()

class Processor(BaseProcessor):
    def __init__(self, config_file: str, num_encoder_threads: int = 2):
        # 1. 初始化模型
        self._video_predictor = SNNPredictor(config_file=config_file, lightweight_mode=False)
        logger.info("Performing warmup inference...")
        self._warmup_inference()
        logger.info("Warmup inference completed")

        # 2. 初始化生产者-消费者队列
        self._encoding_queue = Queue()

        # 3. 创建并启动独立的编码器线程
        self._encoder_threads = []
        for i in range(num_encoder_threads):
            thread = Thread(
                target=self._encoder_worker,
                name=f"encoder-worker-{i}",
                daemon=True # 设置为守护线程，主程序退出时它会自动退出
            )
            thread.start()
            self._encoder_threads.append(thread)
        logger.info(f"Started {num_encoder_threads} encoder worker threads.")

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


    def _encoder_worker(self):
        """
        消费者：在后台线程中运行，处理编码任务。
        """
        logger.info("Encoder worker started.")
        while True:
            try:
                result_future = None
                # 从队列中获取任务，如果队列为空，会阻塞等待
                task = self._encoding_queue.get()
                if task is None: # 优雅退出的信号
                    break

                # 解包任务
                decoded_latents_gpu, result_future, metadata = task
                
                # --- 这是我们之前优化的编码逻辑 ---
                encoder_start_time = time.time()
                pinned_buffer = torch.empty(
                    decoded_latents_gpu.shape, 
                    dtype=decoded_latents_gpu.dtype, 
                    device='cpu'
                ).pin_memory()
                
                pinned_buffer.copy_(decoded_latents_gpu, non_blocking=True)
                torch.cuda.synchronize() # 等待复制到pinned_buffer完成
                
                # 复制完成后，可以安全地释放GPU Tensor的内存了
                del decoded_latents_gpu

                frame_np = pinned_buffer.numpy()

                result_data = {
                    "decoded_latents": frame_np,
                    **metadata # 合并元数据
                }
                final_payload_bytes = orjson.dumps(result_data, option=orjson.OPT_SERIALIZE_NUMPY)
                
                # 将最终结果设置到Future对象上，通知等待的请求
                result_future.set_result(final_payload_bytes)
                
                logger.info(f"Encoding task completed in {time.time() - encoder_start_time:.3f}s")
                # --- 编码逻辑结束 ---

            except Exception as e:
                # 如果编码出错，将异常设置到Future上，让客户端知道出错了
                logger.error(f"Error in encoder worker: {e}\n{traceback.format_exc()}")
                if 'result_future' in locals() and not result_future.done():
                    result_future.set_exception(e)
        logger.info("Encoder worker stopping.")

    def __call__(self, req_params: ReqParams, res_params: ResParams):
        with _predictor_lock:
            # ========================================================================
            #  1. PREDICTOR / DECODE BLOCK - (此部分保持不变)
            # ========================================================================
            with MonitorTimer("predictor"):
                data = req_params.data
                if not isinstance(data, dict):
                    res_params.err_num = exceptions.NUM_ILLEGAL_ARGS
                    res_params.err_msg = f"Request field 'data' must be a dictionary. Found: {type(data).__name__}"
                    return
                
                if data.get("chunk_faces") is None or data.get("chunk_whisper_features") is None:
                    res_params.err_num = exceptions.NUM_ILLEGAL_ARGS
                    res_params.err_msg = "chunk_faces/chunk_whisper_features cannot be found in request"
                    return

                chunk_faces = data["chunk_faces"]
                chunk_whisper_features = data["chunk_whisper_features"]
                
                num_frames = data.get("num_frames", 16)
                num_inference_steps = data.get("num_inference_steps", 20)
                guidance_scale = data.get("guidance_scale", 1.0)
                weight_dtype_str = data.get("weight_dtype", "float16")
                eta = data.get("eta", 0.0)
                height = data.get("height", 256)
                width = data.get("width", 256)
                debug = data.get("debug", False)

                convert_start = time.time()
                
                if isinstance(chunk_faces, list):
                    chunk_faces_tensor = torch.tensor(chunk_faces, dtype=torch.float32, device="cuda").contiguous()
                else:
                    chunk_faces_tensor = torch.from_numpy(chunk_faces).to(torch.float32, device="cuda").contiguous() if isinstance(chunk_faces, np.ndarray) else chunk_faces
                    
                if isinstance(chunk_whisper_features, list) and len(chunk_whisper_features) > 0:
                    if isinstance(chunk_whisper_features[0], list):
                        chunk_whisper_tensor = [torch.tensor(w, dtype=torch.float32, device="cuda").contiguous() for w in chunk_whisper_features]
                    elif isinstance(chunk_whisper_features[0], np.ndarray):
                        chunk_whisper_tensor = [torch.from_numpy(w).to(torch.float32, device="cuda").contiguous() for w in chunk_whisper_features]
                    else:
                        chunk_whisper_tensor = chunk_whisper_features
                else:
                    chunk_whisper_tensor = chunk_whisper_features

                logger.info(f"Data conversion completed in {time.time() - convert_start:.3f}s")

                weight_dtype = torch.float16 if weight_dtype_str == "float16" else torch.float32

                try:
                    logger.info("Starting chunk inference...")
                    inference_start = time.time()
                    
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
                    inference_time = time.time() - inference_start
                    logger.info(f"Model inference completed in {inference_time:.3f}s")

                except Exception as e:
                    logger.error(f"Video inference failed: {e}", exc_info=True)
                    res_params.err_num = exceptions.NUM_BACKEND_ERROR
                    res_params.err_msg = f"Video inference error: {str(e)}"
                    return

        # ========================================================================
        #  2. ENCODER BLOCK - (全新优化的部分)
        # ========================================================================
        try:
            with MonitorTimer("encoder"):
                # a. 创建一个Future对象，用于稍后接收编码结果
                result_future = Future()
                
                # b. 收集所有需要在编码器中使用的元数据
                metadata = {
                    "num_frames": num_frames,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "height": height,
                    "width": width,
                    "debug": debug
                }

                # c. 将 (GPU张量, Future, 元数据) 打包成一个任务，放入队列
                self._encoding_queue.put((decoded_latents, result_future, metadata))

                # d. 阻塞等待，直到编码器线程通过 future.set_result() 完成任务
                #    这里的超时是一个安全措施
                final_payload_bytes = result_future.result(timeout=10.0) 
                
                # e. 将最终的二进制结果设置到响应中
                res_params.result = final_payload_bytes

        except Exception as e:
            logger.error(f"Failed to get result from encoder thread: {e}", exc_info=True)
            res_params.err_num = exceptions.NUM_BACKEND_ERROR
            res_params.err_msg = f"Encoding failed or timed out: {str(e)}"

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

# # NOTE: Flask的应用程序入口方式
logger.info(f"[SERVER] Initializing Video Generation Processor with config: {config_file}")
server.register(Processor(config_file=config_file), ReqParams, ResParams)
app = server.app

if __name__ == "__main__":
    server.app.run(host="127.0.0.1", port=80, threaded=True)