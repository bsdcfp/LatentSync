#!/bin/bash

# HunyuanDiT Batch Processing Examples
# 展示如何使用新的 --batch-size 参数进行不同规模的批处理测试

echo "=========================================="
echo "HunyuanDiT Batch Processing Test Examples"
echo "=========================================="

MODEL_PATH="/model_zoo/hunyuanDiT/"
TEST_NUM=4  # 测试图片数量

# echo ""
# echo "1. 单张处理模式 (batch-size=1)"
# echo "  传统模式，逐张处理"
# python test/local_test.py \
#     --test-num $TEST_NUM \
#     --input-type image_array \
#     --model-path $MODEL_PATH \
#     --batch-size 1 \
#     --enable-gpu-monitor

echo ""
echo "2. 小批量处理 (max-batch-size=4)"
echo "  测试智能配置分组，相同配置的图片会自动分组"
python test/local_test.py \
    --test-num $TEST_NUM \
    --input-type image_array \
    --model-path $MODEL_PATH \
    --batch-size 2 \
    --enable-gpu-monitor

# echo ""
# echo "3. 中等批量处理 (batch-size=3)" 
# echo "  测试混合配置分组效果"
# python test/local_test.py \
#     --test-num $TEST_NUM \
#     --input-type image_array \
#     --model-path $MODEL_PATH \
#     --batch-size 3 \
#     --enable-gpu-monitor

# echo ""
# echo "4. 大批量处理 (batch-size=6)"
# echo "  一次处理所有图片，查看最大分组效果"
# python test/local_test.py \
#     --test-num $TEST_NUM \
#     --input-type image_array \
#     --model-path $MODEL_PATH \
#     --batch-size 6 \
#     --enable-gpu-monitor

# echo ""
# echo "5. 使用image_id模式测试批处理"
# echo "  测试不同输入类型的批处理性能"
# python test/local_test.py \
#     --test-num $TEST_NUM \
#     --input-type image_id \
#     --model-path $MODEL_PATH \
#     --batch-size 3 \
#     --enable-gpu-monitor

echo ""
echo "=========================================="
echo "批处理测试完成！"
echo "结果文件保存在 output/ 目录中"
echo "GPU监控图表保存在 output/gpu_monitor/ 目录中"
echo "==========================================" 