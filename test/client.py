######################################################################
#
# Copyright (c) 2025 Shopee Inc. All Rights Reserved.
#
######################################################################

"""
file: client.py

brief: HTTP Client for Video Generation Testing
"""

import os
import sys

# 设置环境变量来避免tokenizers的并行处理警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置 torch.compile 相关环境变量
# os.environ["TORCH_LOGS"] = "recompiles"  # 可选：记录重编译原因
# os.environ["TORCHDYNAMO_DISABLE"] = "1"  # 可选：完全禁用dynamo

# 将当前文件所在目录的父目录添加到 sys.path 中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

import base64
import orjson
import argparse
import glob
import random
import shutil
import time
import cv2
import numpy as np
import torch
from statistics import mean, stdev

import requests
import pandas as pd
import tqdm
from scipy.spatial import distance
import gzip
from typing import Union, List

# 导入视频生成相关模块
from latentsync.backends.snn_predictor import SNNPredictor

local_url = "http://127.0.0.1:80/api/process"
remote_url = "http://sg12.aip.mlp.shopee.io/services/555883/api/process"

# 全局时间统计列表
time_list = []

# 全局predictor实例
_global_predictor = None


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

def get_predictor(config_file: str, lightweight_mode: bool = True) -> SNNPredictor:
    """
    获取全局predictor实例，如果不存在则初始化
    
    Args:
        config_file: 配置文件路径
        lightweight_mode: 是否使用轻量级模式（客户端默认True）
        
    Returns:
        SNNPredictor: predictor实例
    """
    global _global_predictor
    if _global_predictor is None:
        print(f"Initializing global SNNPredictor (lightweight_mode={lightweight_mode})...")
        _global_predictor = SNNPredictor(config_file, lightweight_mode=lightweight_mode)
        print("✅ Global SNNPredictor initialized successfully")
    return _global_predictor

def preprocess_video_data(audio_path: str, config_file: str) -> dict:
    """
    视频数据前处理
    
    Args:
        audio_path: 音频文件路径
        config_file: 配置文件路径
        
    Returns:
        dict: 预处理后的数据（原始格式，不序列化）
    """
    # 使用全局predictor实例
    predictor = get_predictor(config_file)
    
    # 执行前处理
    print(f"    - Calling predictor._preprocess_video_data...")
    preprocess_start = time.time()
    preprocess_data = predictor._preprocess_video_data(audio_path)
    preprocess_end = time.time()
    print(f"    - Raw preprocessing completed in {preprocess_end - preprocess_start:.2f}s")
    
    return preprocess_data

def postprocess_video_data(preprocess_data: dict, synced_video_frames: list, config_file: str) -> np.ndarray:
    """
    视频数据后处理 - 按chunk分批处理以提高性能
    
    Args:
        preprocess_data: 预处理数据
        synced_video_frames: 推理结果列表（从HTTP响应中获取的序列化数据）
        config_file: 配置文件路径
        
    Returns:
        np.ndarray: 最终的视频帧数组
    """
    # 使用全局predictor实例
    predictor = get_predictor(config_file)
    
    # 验证输入数据
    print(f"[POSTPROCESS] Input validation:")
    print(f"  - synced_video_frames length: {len(synced_video_frames)}")
    print(f"  - original_video_frames length: {len(preprocess_data['original_video_frames'])}")
    print(f"  - boxes length: {len(preprocess_data['boxes'])}")
    print(f"  - affine_matrices length: {len(preprocess_data['affine_matrices'])}")
    print(f"  - mask_frames type: {type(preprocess_data.get('mask_frames'))}, length: {len(preprocess_data.get('mask_frames', []))}")
    print(f"  - original_faces_len: {preprocess_data['original_faces_len']}")
    
    # 将序列化的帧数据转换回torch.Tensor
    torch_frames = []
    for i, frame_data in enumerate(synced_video_frames):
        # frame_data是从JSON反序列化后的Python列表
        frame_np = np.array(frame_data, dtype=np.float32)
        frame_tensor = torch.from_numpy(frame_np)
        # 确保tensor在GPU上
        if torch.cuda.is_available():
            frame_tensor = frame_tensor.cuda()
        torch_frames.append(frame_tensor)
        print(f"  - Chunk {i+1}: frame shape {frame_tensor.shape}, device: {frame_tensor.device}")
    
    # 验证帧顺序
    total_frames = sum(frame.shape[0] if len(frame.shape) > 0 else 1 for frame in torch_frames)
    print(f"  - Total frames from chunks: {total_frames}")
    print(f"  - Expected frames: {preprocess_data['original_faces_len']}")
    
    if total_frames != preprocess_data['original_faces_len']:
        print(f"[WARNING] Frame count mismatch! Expected {preprocess_data['original_faces_len']}, got {total_frames}")
        # 合并并裁剪
        merged_frames = torch.cat(torch_frames, dim=0)[:preprocess_data['original_faces_len']]
        # 重新分chunk
        chunk_size = torch_frames[0].shape[0]
        torch_frames = [merged_frames[i:i+chunk_size] for i in range(0, len(merged_frames), chunk_size)]
    
    # 执行后处理
    final_video_frames = predictor._postprocess_video_data(preprocess_data, torch_frames)
    
    print(f"[POSTPROCESS] Output validation:")
    print(f"  - final_video_frames shape: {final_video_frames.shape}")
    
    return final_video_frames

def save_video_output(preprocess_data: dict, final_video_frames: np.ndarray, video_out_path: str, mask_output_path: str = None, config_file: str = None):
    """
    保存视频输出
    
    Args:
        preprocess_data: 预处理数据
        final_video_frames: 最终视频帧数组
        video_out_path: 视频输出路径
        mask_output_path: mask视频输出路径（可选）
        config_file: 配置文件路径
    """
    # 使用全局predictor实例
    predictor = get_predictor(config_file)
    
    # 执行保存
    predictor._save_video_output(preprocess_data, final_video_frames, video_out_path, mask_output_path)

def collect_timing_stats():
    """收集和显示时间统计信息"""
    if not time_list:
        return
    
    max_value = max(time_list)
    min_value = min(time_list)
    mean_value = sum(time_list) / len(time_list)
    
    print(f"\n[TIMING] Detailed Statistics:")
    print(f"  Total requests: {len(time_list)}")
    print(f"  Average time: {mean_value:.2f} ms")
    print(f"  Max time: {max_value:.2f} ms")
    print(f"  Min time: {min_value:.2f} ms")
    
    if len(time_list) > 1:
        std_dev = stdev(time_list)
        print(f"  Std Dev: {std_dev:.2f} ms")
    
    # 显示所有时间（如果测试次数不多）
    if len(time_list) <= 20:
        print(f"  All times: {[f'{t:.2f}' for t in time_list]} ms")

def save_results_to_csv(results_data, output_dir, args):
    """保存结果到CSV文件"""
    if not results_data:
        return
    
    # 创建结果DataFrame
    df = pd.DataFrame(results_data)
    
    # 生成输出文件名
    url_type = "local" if args.local_url else "remote"
    csv_filename = f"video_gen_test_results_{url_type}_iter{args.test_num}.csv"
    output_path = os.path.join(output_dir, csv_filename)
    
    # 保存到CSV
    df.to_csv(output_path, index=False)
    print(f"[INFO] Results saved to: {output_path}")

def make_video_generation_request(url, request_data, test_id, args):
    """发送单个视频生成HTTP请求并使用 orjson 处理响应"""
    try:
        op_begin = time.time()
        # request_data 应该是 prepare_request_payload_with_orjson 返回的 bytes
        headers = {'Content-Type': 'application/json'}
        result = requests.post(url=url, data=request_data, headers=headers, timeout=10)
        op_end = time.time()
        
        request_time = (op_end - op_begin) * 1000.0
        result.raise_for_status()
        
        # 检查 HTTP 状态码，如果不是 200，先处理错误
        # if result.status_code != 200:
        #     print(f"[TEST {test_id}] FAILED - HTTP {result.status_code}: {result.text}")
        #     return {
        #         'test_id': test_id, 'status': 'FAILED', 'request_time_ms': request_time,
        #         'http_status': result.status_code, 'error_message': result.text
        #     }

        # --- 核心修改在这里 ---
        # 使用 orjson.loads() 来解析响应的二进制内容
        response_dict = orjson.loads(result.content)
        if "err_num" in response_dict and response_dict["err_num"] != 0:
            error_msg = f"Server returned an error: {response_dict.get('err_msg', 'Unknown error')}"
            print(f"[TEST {test_id}] FAILED - {error_msg}")
            return {
                'test_id': test_id, 'status': 'FAILED', 'request_time_ms': request_time,
                'http_status': result.status_code, 'error_message': error_msg
            }
        
        if "result" in response_dict and response_dict["result"] is not None:
            # server_results = response_dict["result"]
            server_results_payload = response_dict["result"]
            server_results = orjson.loads(server_results_payload)
            print(f"keys: {server_results.keys()}")
            
            # 3. 在这个嵌套的字典中寻找 decoded_latents
            if "decoded_latents" in server_results:
                decoded_latents_list = server_results["decoded_latents"]
                if not isinstance(decoded_latents_list, list):
                    raise TypeError(f"Expected decoded_latents to be a list, but got {type(decoded_latents_list).__name__}")
                decoded_latents_np = np.array(decoded_latents_list)
                server_results["decoded_latents"] = decoded_latents_np

                print(f"[TEST {test_id}] SUCCESS - Request time: {request_time:.2f} ms")
                
                # 4. 返回成功，并将嵌套的 server_results 作为 response_data
                return {
                    'test_id': test_id, 'status': 'SUCCESS', 'request_time_ms': request_time,
                    'http_status': result.status_code, 
                    'response_data': server_results # <--- 返回 result 内部的字典
                }
        
        # 如果代码走到这里，说明响应结构不符合预期
        error_msg = f"Invalid response structure. 'result' or 'decoded_latents' key missing or invalid. Response keys: {response_dict.keys()}"
        print(f"[TEST {test_id}] FAILED - {error_msg}")
        return {
            'test_id': test_id, 'status': 'FAILED', 'request_time_ms': request_time,
            'http_status': result.status_code, 'error_message': error_msg
        }

    except requests.exceptions.HTTPError as e: # 更具体地捕获 HTTP 错误
        print(f"[TEST {test_id}] FAILED - HTTP Error {e.response.status_code}: {e.response.text}")
        return {
            'test_id': test_id, 'status': 'FAILED', 'request_time_ms': request_time,
            'http_status': e.response.status_code, 'error_message': e.response.text
        }
    except orjson.JSONDecodeError as e:
        print(f"[TEST {test_id}] FAILED - JSON Decode Error: {str(e)}")
        # 在 `result` 变量可能未定义时安全地访问它
        # response_content = result.content[:50] if 'result' in locals() else b''
        print(f"Raw response content that failed to parse: ...")
        return {
            'test_id': test_id, 'status': 'FAILED', 'request_time_ms': request_time,
            'http_status': result.status_code if 'result' in locals() else 0, 'error_message': f"JSON Decode Error: {e}"
        }
    except Exception as e:
        print(f"[TEST {test_id}] FAILED - Exception: {str(e)}")
        return {
            'test_id': test_id, 'status': 'FAILED', 'request_time_ms': (time.time() - op_begin) * 1000.0,
            'http_status': 0, 'error_message': str(e)
        }


def prepare_request_payload(
    chunk_id: int,
    chunk_faces: torch.Tensor,
    chunk_whisper_features: Union[torch.Tensor, List[torch.Tensor]],
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    eta: float,
    height: int,
    width: int,
    weight_dtype: str = "float16",
    debug: bool = False
) -> bytes:
    """
    使用 orjson 高效地准备请求体 (payload)。
    
    这个函数直接将包含 PyTorch Tensors 的数据结构转换为
    一个可供 `requests.post` 使用的 bytes 对象。

    返回:
        bytes: 序列化后的请求体，可直接用于 requests.post(data=...)
    """
    
    # 1. 准备数据：将所有 Tensor 转换为 orjson 能直接处理的 Numpy 数组。
    #    这是一个非常快速的操作，比压缩和base64编码快得多。
    faces_np = chunk_faces.cpu().numpy()

    if isinstance(chunk_whisper_features, list):
        whisper_np = [w.cpu().numpy() for w in chunk_whisper_features]
    else:
        # 保证即使是单个 tensor 也能被正确处理
        whisper_np = chunk_whisper_features.cpu().numpy()

    # 2. 构建包含 Numpy 数组的 Python 字典
    payload_dict = {
        "logid": chunk_id * 1000,
        "clientip": "",
        "data": {
            # 直接把 Numpy 数组放进去
            "chunk_faces": faces_np,
            "chunk_whisper_features": whisper_np,
            
            # 其他元数据
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "weight_dtype": weight_dtype,
            "eta": eta,
            "height": height,
            "width": width,
            "debug": debug
        }
    }
    
    # 3. 使用 orjson 一步完成序列化，得到最终的 bytes 对象
    payload_bytes = orjson.dumps(payload_dict, option=orjson.OPT_SERIALIZE_NUMPY)
    
    return payload_bytes

# def prepare_request_payload(
#     chunk_id,
#     chunk_faces,
#     chunk_whisper_features,
#     num_frames,
#     num_inference_steps,
#     guidance_scale,
#     eta,
#     height,
#     width,
#     weight_dtype="float16",
#     debug=False
# ):
#     # faces
#     chunk_faces_serializable = tensor_to_json_compressed(chunk_faces, compression='gzip')

#     # whisper
#     if isinstance(chunk_whisper_features, list) and len(chunk_whisper_features) > 0:
#         chunk_whisper_serializable = [tensor_to_json_compressed(w, compression='gzip') for w in chunk_whisper_features]
#     else:
#         chunk_whisper_serializable = chunk_whisper_features

#     chunk_request_payload = orjson.dumps({
#         "logid": chunk_id * 1000,
#         "clientip": "",
#         "data": {
#             "chunk_faces": chunk_faces_serializable,
#             "chunk_whisper_features": chunk_whisper_serializable,
#             "num_frames": num_frames,
#             "num_inference_steps": num_inference_steps,
#             "guidance_scale": guidance_scale,
#             "weight_dtype": weight_dtype,
#             "eta": eta,
#             "height": height,
#             "width": width,
#             "debug": debug
#         }
#     })
#     return chunk_request_payload

def main():
    parser = argparse.ArgumentParser(description='HTTP Client for Video Generation Testing')
    parser.add_argument('--audio-file', type=str, required=True,
                       help='Path to the audio file for video generation')
    parser.add_argument('--output-dir', type=str, default='output', 
                       help='Path to the output directory')
    parser.add_argument('--config-file', type=str, default='models/config.yaml',
                       help='Path to the config file')
    parser.add_argument('--local-url', action='store_true', default=False,
                       help='Use local service instead of remote service')
    parser.add_argument('--test-num', type=int, default=1, 
                       help='Number of test iterations to run')
    parser.add_argument('--save-results', action='store_true', default=False,
                       help='Save test results to CSV file')
    
    # 视频生成参数
    parser.add_argument('--target-fps', type=int, default=25,
                       help='Target FPS for video generation')
    parser.add_argument('--inference-steps', type=int, default=20,
                       help='Number of inference steps')
    parser.add_argument('--guidance-scale', type=float, default=1.0,
                       help='Guidance scale')
    parser.add_argument('--seed', type=int, default=1247,
                       help='Random seed')
    parser.add_argument('--debug-pipeline', action='store_true', default=False,
                       help='Enable pipeline debug mode')
    parser.add_argument('--debug-video-merger', action='store_true', default=False,
                       help='Enable video merger debug mode')
    parser.add_argument('--full-mode', action='store_true', default=False,
                       help='Use full mode (load all models) instead of lightweight mode')
    
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 检查音频文件
    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file does not exist: {args.audio_file}")
        return
    
    # 检查音频文件格式
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac']
    file_ext = os.path.splitext(args.audio_file)[1].lower()
    if file_ext not in audio_extensions:
        print(f"❌ Error: File is not a supported audio format: {args.audio_file}")
        print(f"Supported formats: {audio_extensions}")
        return

    # 根据命令行参数选择模式
    lightweight_mode = not args.full_mode
    mode_name = "lightweight" if lightweight_mode else "full"
    
    # 打印配置信息
    print("=" * 60)
    print(f"[CONFIG] Video Generation HTTP Client Test Configuration")
    print(f"  Audio file: {args.audio_file}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Config file: {args.config_file}")
    print(f"  Test iterations: {args.test_num}")
    print(f"  Using local URL: {args.local_url}")
    print(f"  Save results: {args.save_results}")
    print(f"  Target FPS: {args.target_fps}")
    print(f"  Inference steps: {args.inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Seed: {args.seed}")
    print(f"  Debug pipeline: {args.debug_pipeline}")
    print(f"  Debug video merger: {args.debug_video_merger}")
    print(f"  Mode: {mode_name}")
    print("=" * 60)
    
    # 确定URL
    if args.local_url:
        url = local_url
        print(f"[URL] Using LOCAL endpoint: {url}")
    else:
        url = remote_url
        print(f"[URL] Using REMOTE endpoint: {url}")

    # 初始化统计变量
    successful_tests = 0
    failed_tests = 0
    results_data = []
    
    print(f"\n[INFO] Starting {args.test_num} test iterations...")
    
    print(f"Pre-initializing SNNPredictor in {mode_name} mode...")
    init_start_time = time.time()
    get_predictor(args.config_file, lightweight_mode=lightweight_mode)
    init_time = time.time() - init_start_time
    print(f"✅ Predictor initialization completed in {init_time:.2f}s")
    
    # 运行测试循环 - 每次测试都处理完整的音频（所有chunks）
    for i in tqdm.tqdm(range(args.test_num), desc="Running tests"):
        test_id = f"{i+1}/{args.test_num}"
        print(f"\n[TEST {test_id}] Starting complete audio processing...")
        
        try:
            # 步骤1: 前处理
            print(f"[TEST {test_id}] Step 1: Preprocessing data...")
            preprocess_start_time = time.time()
            
            # 详细监控前处理各个步骤
            print(f"[TEST {test_id}]   - Starting preprocessing...")
            preprocess_data = preprocess_video_data(args.audio_file, args.config_file)
            
            preprocess_time = time.time() - preprocess_start_time
            print(f"[TEST {test_id}] Preprocessing completed in {preprocess_time:.2f}s")
            
            # 检查前处理数据的结构
            if preprocess_data:
                print(f"[TEST {test_id}]   - Preprocess data keys: {list(preprocess_data.keys())}")
                if 'faces_by_chunk' in preprocess_data:
                    print(f"[TEST {test_id}]   - faces_by_chunk length: {len(preprocess_data['faces_by_chunk'])}")
                if 'whisper_chunks_by_chunk' in preprocess_data:
                    print(f"[TEST {test_id}]   - whisper_chunks_by_chunk length: {len(preprocess_data['whisper_chunks_by_chunk'])}")
            
            # 步骤2: 逐个chunk发送推理请求
            print(f"[TEST {test_id}] Step 2: Processing all chunks...")
            
            num_chunks = len(preprocess_data['faces_by_chunk'])
            synced_video_frames = [None] * num_chunks  # 预分配列表，确保顺序
            chunk_results = []  # 收集所有chunk的结果
            
            for chunk_idx in range(num_chunks):
                print(f"[TEST {test_id}]   - Processing chunk {chunk_idx+1}/{num_chunks}...")

                chunk_faces = preprocess_data['faces_by_chunk'][chunk_idx]
                chunk_whisper = preprocess_data['whisper_chunks_by_chunk'][chunk_idx]

                # 直接用 prepare_request_payload 生成请求体
                chunk_request_payload = prepare_request_payload(
                    chunk_id=1234567 + i + chunk_idx,
                    chunk_faces=chunk_faces,
                    chunk_whisper_features=chunk_whisper,
                    num_frames=16,
                    num_inference_steps=args.inference_steps,
                    guidance_scale=args.guidance_scale,
                    eta=0.0,
                    height=256,
                    width=256,
                    weight_dtype="float16",
                    debug=args.debug_pipeline
                )

                # 如果debug，打印请求体大小
                if args.debug_pipeline:
                    print(f"[DEBUG] Chunk {chunk_idx+1} request size: {len(chunk_request_payload)/1024/1024:.2f} MB")

                # 发送单个chunk的HTTP请求
                chunk_result = make_video_generation_request(
                    url, chunk_request_payload, f"{test_id}_chunk{chunk_idx+1}", args
                )
                chunk_results.append(chunk_result)
                
                if chunk_result['status'] == 'SUCCESS':
                    # 从响应中获取推理结果
                    response_data = chunk_result['response_data']
                    if response_data is None:
                        raise Exception(f"Chunk {chunk_idx+1} returned None response_data")
                    print(f"type of response_data: {type(response_data)}")
                    decoded_latents = response_data.get('decoded_latents')
                    if decoded_latents is None:
                        raise Exception(f"Chunk {chunk_idx+1} returned None decoded_latents")

                    # 如果debug，打印响应体大小
                    if args.debug_pipeline:
                        resp_bytes_len = len(orjson.dumps(response_data, option=orjson.OPT_SERIALIZE_NUMPY))
                        print(f"[DEBUG] Chunk {chunk_idx+1} response payload size: {resp_bytes_len/1024/1024:.2f} MB")

                    if not isinstance(decoded_latents, np.ndarray):
                        # 添加一个健壮性检查，如果返回的不是预期的 numpy 数组，则报错
                        raise TypeError(f"Expected decoded_latents to be a numpy array, but got {type(decoded_latents)}")
                    
                    frame_tensor = torch.from_numpy(decoded_latents)

                    # 确保按正确顺序存储结果
                    synced_video_frames[chunk_idx] = frame_tensor
                    print(f"[TEST {test_id}]   - Chunk {chunk_idx+1} completed successfully and stored at position {chunk_idx}")
                else:
                    print(f"[TEST {test_id}]   - Chunk {chunk_idx+1} failed: {chunk_result.get('error_message', 'Unknown error')}")
                    # 如果某个chunk失败，整个测试失败
                    raise Exception(f"Chunk {chunk_idx+1} inference failed")
            
            # 验证所有chunks都已处理完成
            if None in synced_video_frames:
                missing_chunks = [i for i, frame in enumerate(synced_video_frames) if frame is None]
                raise Exception(f"Missing chunks: {missing_chunks}")
            
            print(f"[TEST {test_id}] All chunks processed successfully, total: {len(synced_video_frames)}")
            print(f"[TEST {test_id}] Frame order verification:")
            for i, frame in enumerate(synced_video_frames):
                if frame is not None:
                    print(f"[TEST {test_id}]   - Chunk {i+1}: frame shape {frame.shape}")
                else:
                    print(f"[TEST {test_id}]   - Chunk {i+1}: MISSING")
            
            # 步骤3: 后处理
            print(f"[TEST {test_id}] Step 3: Postprocessing video...")
            postprocess_start_time = time.time()
            
            # 直接使用已有的preprocess_data，避免重新前处理
            print(f"[TEST {test_id}]   - Using existing preprocess data for postprocessing...")
            
            # 执行后处理
            final_video_frames = postprocess_video_data(preprocess_data, synced_video_frames, args.config_file)
            postprocess_time = time.time() - postprocess_start_time
            print(f"[TEST {test_id}] Postprocessing completed in {postprocess_time:.2f}s")
            
            # 步骤4: 保存输出
            print(f"[TEST {test_id}] Step 4: Saving output...")
            audio_basename = os.path.splitext(os.path.basename(args.audio_file))[0]
            output_path = os.path.join(args.output_dir, f"{audio_basename}_generated_{i}.mp4")
            mask_output_path = os.path.join(args.output_dir, f"{audio_basename}_mask_{i}.mp4")
            
            save_video_output(preprocess_data, final_video_frames, output_path, mask_output_path, args.config_file)
            print(f"[TEST {test_id}] Output saved to: {output_path}")
            
            successful_tests += 1
            
            # 记录成功结果
            if args.save_results:
                result_record = {
                    'test_iteration': i + 1,
                    'audio_file': args.audio_file,
                    'preprocess_time_s': preprocess_time,
                    'inference_time_ms': sum([r.get('request_time_ms', 0) for r in chunk_results]) if 'chunk_results' in locals() else 0,
                    'postprocess_time_s': postprocess_time,
                    'total_time_s': preprocess_time + postprocess_time,
                    'status': 'SUCCESS',
                    'output_path': output_path,
                    'mask_output_path': mask_output_path,
                    'inference_steps': args.inference_steps,
                    'guidance_scale': args.guidance_scale
                }
                results_data.append(result_record)
            

                    
        except Exception as e:
            failed_tests += 1
            print(f"[TEST {test_id}] Exception occurred: {str(e)}")
            
            # 记录异常结果
            if args.save_results:
                result_record = {
                    'test_iteration': i + 1,
                    'audio_file': args.audio_file,
                    'preprocess_time_s': preprocess_time if 'preprocess_time' in locals() else 0,
                    'inference_time_ms': 0,
                    'postprocess_time_s': 0,
                    'total_time_s': preprocess_time if 'preprocess_time' in locals() else 0,
                    'status': 'EXCEPTION',
                    'output_path': '',
                    'mask_output_path': '',
                    'inference_steps': args.inference_steps,
                    'guidance_scale': args.guidance_scale,
                    'error_message': str(e)
                }
                results_data.append(result_record)

    # 显示最终统计信息
    print("\n" + "=" * 60)
    print(f"[SUMMARY] Video Generation HTTP Client Test Results")
    print("=" * 60)
    print(f"Total tests: {args.test_num}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {(successful_tests/args.test_num)*100:.1f}%")
    print(f"Predictor initialization time: {init_time:.2f}s")
    print(f"Target URL: {url}")
    print(f"Audio file: {args.audio_file}")
    
    # 显示详细时间统计
    collect_timing_stats()
    
    # 保存结果到CSV
    if args.save_results and results_data:
        save_results_to_csv(results_data, args.output_dir, args)
    
    print("=" * 60)

if __name__ == '__main__':
    main()
