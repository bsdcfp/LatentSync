######################################################################
#
# Copyright (c) 2024 Shopee Inc. All Rights Reserved.
#
######################################################################

"""
file: local_test.py
author: songsong.yi@shopee.com
date: 2024-06-12 14:33:31
brief: ---
"""
import os
import sys
import time
import argparse
import pandas as pd
from joblib import Parallel, delayed

import torch
from typing import Dict, List, Optional, Union, Callable
from loguru import logger

# 设置环境变量来避免tokenizers的并行处理警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置 torch.compile 相关环境变量
import torch
torch._dynamo.config.cache_size_limit = 64  # 增加缓存大小从8到64
torch.set_float32_matmul_precision('high')

current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录的父目录添加到 sys.path 中
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from latentsync import NVTXContext, PROFILER_AVAILABLE
from latentsync.backends.snn_predictor import SNNPredictor
from latentsync.src.utils.status_code import PI_STATUS_CODE

def generate_video_workflow(predictor, audio_path: str, output_path: str, mask_output_path: Optional[str] = None) -> Dict:
    """
    视频生成工作流程 - 从SNNPredictor移过来的逻辑
    
    Args:
        predictor: SNNPredictor实例
        audio_path: 音频文件路径
        output_path: 输出视频路径
        mask_output_path: 输出mask视频路径（可选）
    
    Returns:
        Dict: 生成结果
    """
    try:
        logger.info(f"Processing video: {os.path.basename(audio_path)}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if mask_output_path:
            os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
        
        # 调用视频生成流程 - 拆分为四个步骤
        start_time = time.time()
        
        # 步骤1: 前处理
        logger.info(f"Step 1: Preprocessing data for {os.path.basename(audio_path)}")
        preprocess_data = predictor._preprocess_video_data(audio_path)
        
        # 步骤2: 模型推理
        logger.info(f"Step 2: Model inference for {os.path.basename(audio_path)}")
        # 获取视频配置
        video_config = predictor._engine.get_video_config()
        dtype = predictor._engine.get_dtype()
        
        synced_video_frames = predictor._inference_video_chunks(
            preprocess_data=preprocess_data,
            num_frames=16,
            num_inference_steps=video_config.get("inference_steps", 20),
            guidance_scale=video_config.get("guidance_scale", 1.0),
            weight_dtype=dtype,
            eta=0.0,
            generator=None,
            callback=None,
            callback_steps=1,
            height=256,
            width=256,
            debug=video_config.get("debug_pipeline", False),
        )
        
        # 步骤3: 后处理
        logger.info(f"Step 3: Postprocessing video for {os.path.basename(audio_path)}")
        final_video_frames = predictor._postprocess_video_data(preprocess_data, synced_video_frames)
        
        # 步骤4: 保存最终输出
        logger.info(f"Step 4: Saving final output for {os.path.basename(audio_path)}")
        predictor._save_video_output(preprocess_data, final_video_frames, output_path, mask_output_path)
        
        gen_frame_num = preprocess_data['gen_frame_num']
        video_index_list = preprocess_data['video_index_list']
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # 检查输出文件是否存在
        if os.path.exists(output_path):
            result = {
                'status_code': PI_STATUS_CODE.SUCCESS,
                'video_path': output_path,
                'mask_path': mask_output_path if mask_output_path and os.path.exists(mask_output_path) else None,
                'generation_time': generation_time,
                'gen_frame_num': gen_frame_num,
                'error_message': None
            }
            logger.info(f"Video generation successful: {os.path.basename(output_path)} ({generation_time:.2f}s)")
        else:
            result = {
                'status_code': PI_STATUS_CODE.GENERATION_ERROR,
                'video_path': None,
                'mask_path': None,
                'generation_time': generation_time,
                'gen_frame_num': 0,
                'error_message': f"Output video file not found: {output_path}"
            }
            logger.error(f"Video generation failed: output file not found")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {
            'status_code': PI_STATUS_CODE.GENERATION_ERROR,
            'video_path': None,
            'mask_path': None,
            'generation_time': 0,
            'gen_frame_num': 0,
            'error_message': str(e)
        }

def video_worker(audio_files, args):
    """处理视频生成任务 - 直接使用SNNPredictor"""
    results = []
    time_list = []
    
    print(f"[INFO] Processing {len(audio_files)} audio files for video generation")
    
    # 初始化SNNPredictor - 使用models目录下的配置文件
    config_file = os.path.join(current_dir, "..", "models", "config.yaml")
    if not os.path.exists(config_file):
        print(f"❌ Error: Config file not found: {config_file}")
        exit(1)
    
    # 动态更新配置参数
    import yaml
    import json
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # 更新预测器的参数
    if 'snn_torch_predictor' in config_data['predictors']:
        # 基础参数更新
        config_data['predictors']['snn_torch_predictor'].update({
            'model_path': args.model_path,
        })
        
        # 确保业务配置文件路径是绝对路径
        if 'config_file' in config_data['predictors']['snn_torch_predictor']:
            config_file_path = config_data['predictors']['snn_torch_predictor']['config_file']
            if not os.path.isabs(config_file_path):
                # 如果是相对路径，转换为绝对路径
                config_file_path = os.path.join(current_dir, "..", "models", config_file_path)
                config_data['predictors']['snn_torch_predictor']['config_file'] = config_file_path
        
        # 直接使用原始配置文件，命令行参数在SNNTorchBackend中处理
        # 使用绝对路径避免重复拼接
        config_data['predictors']['snn_torch_predictor']['config_file'] = os.path.join(current_dir, "..", "models", "V1.0_latentsync_video_predictor.json")
        
        # 添加命令行参数到配置中
        config_data['predictors']['snn_torch_predictor'].update({
            'target_fps': args.target_fps,
            'inference_steps': args.inference_steps,
            'seed': args.seed,
            'debug_pipeline': args.debug_pipeline,
            'debug_video_merger': args.debug_video_merger,
        })
        
        print(f"[INFO] Using business config: models/V1.0_latentsync_video_predictor.json")
        print(f"[INFO] Command line parameters will be applied during inference:")
        print(f"  - target_fps: {args.target_fps}")
        print(f"  - inference_steps: {args.inference_steps}")
        print(f"  - seed: {args.seed}")
        print(f"  - debug_pipeline: {args.debug_pipeline}")
        print(f"  - debug_video_merger: {args.debug_video_merger}")
    
    # 临时保存更新后的配置到models目录
    temp_config_file = os.path.join(current_dir, "..", "models", "temp_config.yaml")
    with open(temp_config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    config_file = temp_config_file
    
    predictor = SNNPredictor(config_file)
    
    # 批量推理
    op_begin = time.time()
    with NVTXContext(f"video_generation_batch_{len(audio_files)}"):
        for i, audio_file in enumerate(audio_files):
            try:
                # 生成输出路径
                audio_basename = os.path.splitext(os.path.basename(audio_file))[0]
                # 当part为-1时，不显示part号
                part_suffix = f"_{args.part}" if args.part != -1 else ""
                output_path = os.path.join(args.output_dir, f"{audio_basename}_generated{part_suffix}_{i}.mp4")
                mask_output_path = os.path.join(args.output_dir, f"{audio_basename}_mask{part_suffix}_{i}.mp4")
                
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                print(f"[INFO] Processing video {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
                
                # 使用新的视频生成工作流程
                result = generate_video_workflow(predictor, audio_file, output_path, mask_output_path)
                
                # 转换结果格式
                if result['status_code'] == PI_STATUS_CODE.SUCCESS:
                    status_code = "SUCCESS"
                    error_message = None
                    print(f"[INFO] Video generation successful: {os.path.basename(output_path)} ({result['generation_time']:.2f}s)")
                else:
                    status_code = "GENERATION_ERROR"
                    error_message = result['error_message']
                    print(f"[ERROR] Video generation failed: {error_message}")
                
                result_dict = {
                    'audio_file': audio_file,
                    'video_path': result['video_path'],
                    'mask_path': result['mask_path'],
                    'status_code': status_code,
                    'error_message': error_message,
                    'generation_time': result['generation_time']
                }
                results.append(result_dict)
                time_list.append(result['generation_time'])
                
            except Exception as e:
                print(f"[ERROR] Error processing video {i+1}: {str(e)}")
                result_dict = {
                    'audio_file': audio_file,
                    'video_path': None,
                    'mask_path': None,
                    'status_code': "GENERATION_ERROR",
                    'error_message': str(e),
                    'generation_time': 0
                }
                results.append(result_dict)
                time_list.append(0)
    
    op_end = time.time()
    total_time = (op_end - op_begin) * 1000.0  # 转换为毫秒
    
    print(f"[INFO] Total video generation time: {total_time:.2f} ms, avg per video: {total_time/len(audio_files):.2f} ms")
    
    # 计算统计信息
    if time_list:
        total_videos = len(audio_files)
        avg_per_video = sum(time_list) / total_videos
        successful_videos = sum(1 for r in results if r['status_code'] == "SUCCESS")

        print(f"Video generation completed:")
        print(f"  - Total videos: {total_videos}")
        print(f"  - Successful: {successful_videos}")
        print(f"  - Total time: {total_time:.2f}ms")
        print(f"  - Avg per video: {avg_per_video:.2f}ms")
    else:
        print("[WARNING] No timing data collected")
    
    return pd.DataFrame(results, columns=["audio_file", "video_path", "mask_path", "status_code", "error_message", "generation_time"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Generation Test")
    parser.add_argument('--output-dir', type=str, default='output', help='Path to the output directory')
    parser.add_argument('--part', type=int, default=-1, 
        help='index of the target part (default part size is 5000), default=-1 mean: use all part')
    parser.add_argument('--part-size', type=int, default=5000)
    parser.add_argument('--num-worker', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=2)
    parser.add_argument('--model-path', type=str, default='/model_zoo/latentsync_online_simple_auxilary/',
        help='Path to the model directory')
    parser.add_argument('--enable-gpu-monitor', action='store_true', default=False,
        help='Enable GPU monitoring and generate charts')
    parser.add_argument('--gpu-monitor-interval', type=float, default=1.0,
        help='GPU monitoring sample interval in seconds')
    parser.add_argument('--audio-file', type=str, required=True,
        help='Audio file path for video generation')
    parser.add_argument('--target-fps', type=int, default=25,
        help='Target FPS for video generation')
    parser.add_argument('--inference-steps', type=int, default=20,
        help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=1247,
        help='Random seed')
    parser.add_argument('--debug-pipeline', action='store_true', default=False,
        help='Enable pipeline debug mode')
    parser.add_argument('--debug-video-merger', action='store_true', default=False,
        help='Enable video merger debug mode')
    args = parser.parse_args()

    print("=" * 60)
    print(f"[CONFIG] Video Generation Test Configuration")
    print(f"  Audio file: {args.audio_file}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model path: {args.model_path}")
    print(f"  Test number: {args.test_num}")
    print(f"  Workers: {args.num_worker}")
    print(f"  Target FPS: {args.target_fps}")
    print(f"  Inference steps: {args.inference_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Debug pipeline: {args.debug_pipeline}")
    print(f"  Debug video merger: {args.debug_video_merger}")
    print(f"  GPU monitoring: {args.enable_gpu_monitor}")
    if args.enable_gpu_monitor:
        print(f"  GPU monitor interval: {args.gpu_monitor_interval}s")
    print("=" * 60)

    # 创建必要的目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置GPU监控输出目录
    gpu_monitor_dir = os.path.join(args.output_dir, "gpu_monitor")

    # 检查音频文件
    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file does not exist: {args.audio_file}")
        exit(1)
    
    # 检查音频文件格式
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac']
    file_ext = os.path.splitext(args.audio_file)[1].lower()
    if file_ext not in audio_extensions:
        print(f"❌ Error: File is not a supported audio format: {args.audio_file}")
        print(f"Supported formats: {audio_extensions}")
        exit(1)
    
    audio_files = [args.audio_file]
    
    if not audio_files:
        print(f"❌ Error: No audio file specified")
        exit(1)
    
    # 限制测试数量
    audio_files = audio_files[:args.test_num]
    print(f"[INFO] Found {len(audio_files)} audio files for video generation")
    
    num_worker = int(args.num_worker)
    work_size = len(audio_files) // num_worker + int((len(audio_files) % num_worker) != 0)
    
    print(f"[INFO] Starting video processing with {num_worker} worker(s), work size: {work_size}")

    
    results = Parallel(n_jobs=num_worker)(delayed(video_worker)(audio_files[idx*work_size:(idx+1)*work_size], args) for idx in range(num_worker))
    res_df = pd.concat(results, axis=0)

    csv_suffix = f'_video_gen_{args.part}.csv' if args.part != -1 else f'_video_gen_all.csv'
    output_file = os.path.join(args.output_dir, f"video_results{csv_suffix}")
    res_df.to_csv(output_file, index=False)
    
    print(f"[INFO] Video generation results saved to: {output_file}")
    
    print("=" * 60)


