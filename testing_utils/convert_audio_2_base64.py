#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频文件转base64格式脚本
支持mp3、wav、m4a等多种音频格式
生成适配algo.py接口的JSON格式数据
"""

import os
import base64
import json
import argparse
from pathlib import Path
import time

def convert_audio_to_base64(audio_path):
    """
    将音频文件转换为base64编码
    
    Args:
        audio_path (str): 音频文件路径
        
    Returns:
        str: base64编码的音频数据
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    # 检查文件大小
    file_size = os.path.getsize(audio_path)
    print(f"📁 文件大小: {file_size / (1024*1024):.2f} MB")
    
    # 读取音频文件并转换为base64
    with open(audio_path, 'rb') as audio_file:
        audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    
    return audio_base64

audio_path = "/home/work/houyi/latentsync_online_simple_auxilary/input_example/beauty_v2_20s.MP3"
audio_base64 = convert_audio_to_base64(audio_path)
print(audio_base64)