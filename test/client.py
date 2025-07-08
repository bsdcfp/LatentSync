######################################################################
#
# Copyright (c) 2024 Shopee Inc. All Rights Reserved.
#
######################################################################

"""
file: client.py
author: min.yang@shopee.com
date: 2024-04-16 14:33:31
brief: HTTP Client for Product Scene Image Generation Testing
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
import json
import argparse
import glob
import random
import shutil
import time
import cv2
import numpy as np
from statistics import mean, stdev

import requests
import pandas as pd
import tqdm
from scipy.spatial import distance

# 导入图片下载工具
from productscene.src.utils.loader import download_sp_image

local_url = "http://127.0.0.1:80/api/process"
remote_url = "http://us3.aip.mlp.shopee.io/aip-svc-100111/aip-aigc-productbggen-fast/api/process"

# 全局时间统计列表
time_list = []

def download_and_convert_to_base64(image_id):
    """
    下载图片并转换为base64编码的二进制数据
    
    Args:
        image_id: 图片ID
        
    Returns:
        str: base64编码的图片数据，失败时返回None
    """
    try:
        # 使用loader.py中的download_sp_image接口下载BGRA格式图片
        image_array = download_sp_image(image_id, cv2.IMREAD_UNCHANGED)
        
        if image_array is None:
            print(f"[ERROR] Failed to download image: {image_id}")
            return None
        
        # 将numpy数组编码为PNG格式的字节数据
        success, buffer = cv2.imencode('.png', image_array)
        if not success:
            print(f"[ERROR] Failed to encode image: {image_id}")
            return None
        
        # 转换为base64编码
        image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        print(f"[INFO] Downloaded and encoded image {image_id[:20]}...: shape {image_array.shape}, "
              f"dtype {image_array.dtype}, base64 length: {len(image_base64)} chars")
        return image_base64
        
    except Exception as e:
        print(f"[ERROR] Failed to download and convert image {image_id}: {e}")
        return None

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
    pre_download_suffix = "_predownload" if args.pre_download else ""
    csv_filename = f"client_test_results_{url_type}_list{args.list_size}_iter{args.test_num}{pre_download_suffix}.csv"
    output_path = os.path.join(output_dir, csv_filename)
    
    # 保存到CSV
    df.to_csv(output_path, index=False)
    print(f"[INFO] Results saved to: {output_path}")

def make_request(url, request_data, test_id, args):
    """发送单个HTTP请求并处理响应"""
    try:
        op_begin = time.time()
        result = requests.post(url=url, data=request_data, timeout=300)  # 添加超时设置
        op_end = time.time()
        
        request_time = (op_end - op_begin) * 1000.0
        time_list.append(request_time)
        
        if result.status_code == 200:
            results = result.json()["result"]
            print(f"[TEST {test_id}] SUCCESS - Request time: {request_time:.2f} ms")
            
            # 返回成功结果
            return {
                'test_id': test_id,
                'status': 'SUCCESS',
                'request_time_ms': request_time,
                'http_status': result.status_code,
                'response_data': results
            }
        else:
            print(f"[TEST {test_id}] FAILED - HTTP {result.status_code}: {result.text}")
            return {
                'test_id': test_id,
                'status': 'FAILED',
                'request_time_ms': request_time,
                'http_status': result.status_code,
                'error_message': result.text
            }
            
    except Exception as e:
        print(f"[TEST {test_id}] FAILED - Exception: {str(e)}")
        return {
            'test_id': test_id,
            'status': 'FAILED',
            'request_time_ms': 0,
            'http_status': 0,
            'error_message': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='HTTP Client for Product Scene Image Generation Testing')
    parser.add_argument('--csv', type=str, default='examples/test_case/V2.6_online_testdata.csv', 
                       help='Path to the CSV file containing test data')
    parser.add_argument('--output-dir', type=str, default='output', 
                       help='Path to the output directory')
    parser.add_argument('--list-size', type=int, default=1, 
                       help='Number of items in input image list size')
    parser.add_argument('--local-url', action='store_true', default=False,
                       help='Use local service instead of remote service')
    # parser.add_argument('--download-result', action='store_true', default=False,
    #                    help='Download result to local storage')
    parser.add_argument('--test-num', type=int, default=2, 
                       help='Number of test iterations to run')
    parser.add_argument('--data-rows', type=int, default=5, 
                       help='Number of data rows to use from CSV')
    parser.add_argument('--random-data', action='store_true', default=False,
                       help='Use random data selection instead of sequential')
    parser.add_argument('--save-results', action='store_true', default=False,
                       help='Save test results to CSV file')
    parser.add_argument('--pre-download', action='store_true', default=False,
                       help='Pre-download images and send as base64 binary data instead of image IDs')
    args = parser.parse_args()

    # 不需要转换字符串参数为布尔值，store_true自动处理
    # args.local_url = args.local_url.lower() == 'true'
    # args.download_result = args.download_result.lower() == 'true'
    # args.random_data = args.random_data.lower() == 'true'
    # args.save_results = args.save_results.lower() == 'true'
    # args.pre_download = args.pre_download.lower() == 'true'

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置图像缓存目录为 output_dir 的子目录
    image_cache_dir = os.path.join(args.output_dir, "image_cache")
    os.environ["PI_IMAGE_CACHE"] = image_cache_dir
    os.makedirs(image_cache_dir, exist_ok=True)

    # 读取测试数据
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV file not found: {args.csv}")
        return
    
    df = pd.read_csv(args.csv)
    
    # 准备测试数据
    available_rows = min(len(df), args.data_rows)
    test_data = df[:available_rows].to_dict('records')
    
    # 打印配置信息（按照local_test.py的风格）
    print("=" * 60)
    print(f"[CONFIG] HTTP Client Test Configuration")
    print(f"  CSV file: {args.csv}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Image cache directory: {image_cache_dir}")
    print(f"  Test iterations: {args.test_num}")
    print(f"  List size: {args.list_size}")
    print(f"  Using local URL: {args.local_url}")
    # print(f"  Download result: {args.download_result}")
    print(f"  Save results: {args.save_results}")
    print(f"  Pre-download images: {args.pre_download}")
    print(f"  Data rows available: {len(df)}")
    print(f"  Data rows to use: {available_rows}")
    print(f"  Random data selection: {args.random_data}")
    print("=" * 60)
    
    # 确定URL
    url = local_url if args.local_url else remote_url
    print(f"[URL] Target endpoint: {url}")

    # 初始化统计变量
    successful_tests = 0
    failed_tests = 0
    results_data = []
    
    print(f"\n[INFO] Starting {args.test_num} test iterations with varying data...")
    
    # 运行测试循环
    for i in tqdm.tqdm(range(args.test_num), desc="Running tests"):
        test_id = f"{i+1}/{args.test_num}"
        print(f"\n[TEST {test_id}] Starting test iteration...")
        
        # 选择测试数据
        if args.random_data:
            selected_record = random.choice(test_data)
            print(f"[TEST {test_id}] Using random data (row from first {available_rows} rows)")
        else:
            record_index = i % available_rows
            selected_record = test_data[record_index]
            print(f"[TEST {test_id}] Using sequential data (row {record_index + 1}/{available_rows})")
        
        fg_rgba_id = selected_record['aigc_segmentation_image_id']
        generated_config = selected_record['generated_config']
        
        # 根据pre_download参数决定如何处理图片数据
        if args.pre_download:
            print(f"[TEST {test_id}] Pre-downloading image: {fg_rgba_id[:50]}{'...' if len(fg_rgba_id) > 50 else ''}")
            
            # 下载图片并转换为base64
            image_base64_list = []
            download_success = True
            
            for _ in range(args.list_size):
                image_base64 = download_and_convert_to_base64(fg_rgba_id)
                if image_base64 is None:
                    print(f"[TEST {test_id}] FAILED - Failed to download image: {fg_rgba_id}")
                    download_success = False
                    break
                image_base64_list.append(image_base64)
            
            if not download_success:
                failed_tests += 1
                # 记录失败结果
                if args.save_results:
                    result_record = {
                        'test_iteration': i + 1,
                        'image_id': fg_rgba_id,
                        'list_size': args.list_size,
                        'request_time_ms': 0,
                        'status': 'FAILED',
                        'http_status': 0,
                        'generated_config': generated_config,
                        'error_message': 'Failed to download image'
                    }
                    results_data.append(result_record)
                continue
            
            # 使用base64编码的图片数据
            aigc_input = image_base64_list
            print(f"[TEST {test_id}] Using pre-downloaded base64 data: {len(image_base64_list)} images")
        else:
            # 使用图片ID
            aigc_input = [fg_rgba_id] * args.list_size
            print(f"[TEST {test_id}] Using image ID: {fg_rgba_id[:50]}{'...' if len(fg_rgba_id) > 50 else ''}")
        
        # 构建请求
        request_payload = json.dumps({
            "logid": 1234567 + i,  # 每次使用不同的logid
        "clientip": "",
        "data": {
                "aigc_segmentation_image_id": aigc_input,
                "generated_config": [generated_config] * args.list_size,
            }
        })
        
        # 显示请求信息
        if i == 0:
            if args.pre_download:
                print(f"[TEST {test_id}] Sample request structure (with base64 data): logid, clientip, data keys")
                print(f"[TEST {test_id}] Base64 data lengths: {[len(img) for img in aigc_input]} chars")
            else:
                print(f"[TEST {test_id}] Sample request structure: {request_payload[:200]}...")
        else:
            if not args.pre_download:
                print(f"[TEST {test_id}] Image ID: {fg_rgba_id[:50]}{'...' if len(fg_rgba_id) > 50 else ''}")
            # 对于pre_download模式，不再打印具体信息，避免日志过长
        
        # 发送请求
        result = make_request(url, request_payload, test_id, args)
        
        # 统计结果
        if result['status'] == 'SUCCESS':
            successful_tests += 1
            
            # 只在第一次测试时显示详细结果
            if i == 0:
                # 不直接打印response_data，避免打印大量base64数据
                response_keys = list(result['response_data'].keys()) if 'response_data' in result else []
                print(f"[TEST {test_id}] Response keys: {response_keys}")
                if 'response_data' in result:
                    # 只显示status_code等关键信息
                    status_codes = result['response_data'].get('status_code', [])
                    print(f"[TEST {test_id}] Status codes: {status_codes}")
                    # 显示生成的image_id信息（但不显示完整内容，避免过长）
                    if 'aigc_generated_image_id' in result['response_data']:
                        gen_ids = result['response_data']['aigc_generated_image_id']
                        if isinstance(gen_ids, list) and gen_ids:
                            print(f"[TEST {test_id}] Generated {len(gen_ids)} image(s)")
                        else:
                            print(f"[TEST {test_id}] Generated image result available")
            else:
                # 后续测试只显示关键信息
                if 'response_data' in result and 'status_code' in result['response_data']:
                    print(f"[TEST {test_id}] Status codes: {result['response_data']['status_code']}")
        else:
            failed_tests += 1
        
        # 保存结果数据
        if args.save_results:
            result_record = {
                'test_iteration': i + 1,
                'image_id': fg_rgba_id,
                'list_size': args.list_size,
                'pre_download': args.pre_download,
                'request_time_ms': result['request_time_ms'],
                'status': result['status'],
                'http_status': result['http_status'],
                'generated_config': generated_config
            }
            
            if result['status'] == 'SUCCESS' and 'response_data' in result:
                result_record.update({
                    'aigc_generated_image_id': result['response_data'].get('aigc_generated_image_id', [''])[0] if isinstance(result['response_data'].get('aigc_generated_image_id'), list) else result['response_data'].get('aigc_generated_image_id', ''),
                    'aigc_generated_mask_id': result['response_data'].get('aigc_generated_mask_id', [''])[0] if isinstance(result['response_data'].get('aigc_generated_mask_id'), list) else result['response_data'].get('aigc_generated_mask_id', ''),
                    'result_status_code': result['response_data'].get('status_code', [''])[0] if isinstance(result['response_data'].get('status_code'), list) else result['response_data'].get('status_code', '')
                })
            else:
                result_record['error_message'] = result.get('error_message', '')
            
            results_data.append(result_record)

    # 显示最终统计信息（按照local_test.py的风格）
    print("\n" + "=" * 60)
    print(f"[SUMMARY] HTTP Client Test Results")
    print("=" * 60)
    print(f"Total tests: {args.test_num}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {(successful_tests/args.test_num)*100:.1f}%")
    print(f"Data variety: {available_rows} different data records used")
    print(f"Target URL: {url}")
    print(f"Pre-download mode: {args.pre_download}")
    
    # 显示详细时间统计
    collect_timing_stats()
    
    # 保存结果到CSV
    if args.save_results and results_data:
        save_results_to_csv(results_data, args.output_dir, args)
    
    print("=" * 60)

if __name__ == '__main__':
    main()
