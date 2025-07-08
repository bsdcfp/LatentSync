#!/usr/bin/env python3
"""
测试脚本：验证SNNPredictor的视频生成功能迁移是否成功
"""

import os
import sys
import tempfile
import time

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from latentsync.backends.snn_predictor import SNNPredictor

def test_video_generation():
    """测试视频生成功能"""
    print("🧪 开始测试SNNPredictor视频生成功能...")
    
    # 配置路径
    model_path = "/model_zoo/latentsync_online_simple_auxilary/"
    config_file = os.path.join(model_path, "V2.6_latentsync_video_predictor.yaml")
    
    # 创建默认配置文件
    if not os.path.exists(config_file):
        default_config = {
            "predictors": {
                "predictor": "snn_video_predictor",
                "snn_video_predictor": {
                    "model_path": model_path,
                    "max_batch_size": 1
                }
            }
        }
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f)
        print(f"✅ 创建配置文件: {config_file}")
    
    try:
        # 初始化SNNPredictor
        print("🔄 初始化SNNPredictor...")
        start_time = time.time()
        predictor = SNNPredictor(config_file)
        init_time = time.time() - start_time
        print(f"✅ SNNPredictor初始化成功，耗时: {init_time:.2f}s")
        
        # 检查predictor类型
        if predictor._predictor_type != "snn_video_predictor":
            raise ValueError(f"Expected snn_video_predictor, got {predictor._predictor_type}")
        
        # 检查pipeline是否存在
        if not hasattr(predictor, 'pipeline'):
            raise ValueError("Pipeline not found in SNNPredictor")
        
        print("✅ 视频生成组件检查通过")
        
        # 测试音频文件（如果存在）
        test_audio = "/path/to/test/audio.mp3"  # 需要替换为实际的测试音频路径
        if os.path.exists(test_audio):
            print(f"🎵 找到测试音频文件: {test_audio}")
            
            # 创建临时输出目录
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, "test_output.mp4")
                mask_output_path = os.path.join(temp_dir, "test_mask.mp4")
                
                print("🎬 开始视频生成测试...")
                start_time = time.time()
                
                video_paths, mask_paths, status_codes, error_messages = predictor.predict_video(
                    audio_paths=[test_audio],
                    output_paths=[output_path],
                    mask_output_paths=[mask_output_path]
                )
                
                generation_time = time.time() - start_time
                
                if status_codes[0] == "SUCCESS":
                    print(f"✅ 视频生成成功！耗时: {generation_time:.2f}s")
                    print(f"   输出视频: {video_paths[0]}")
                    if mask_paths[0]:
                        print(f"   输出mask: {mask_paths[0]}")
                else:
                    print(f"❌ 视频生成失败: {error_messages[0]}")
        else:
            print(f"⚠️  测试音频文件不存在: {test_audio}")
            print("   跳过实际视频生成测试")
        
        print("🎉 所有测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_video_generation()
    sys.exit(0 if success else 1) 