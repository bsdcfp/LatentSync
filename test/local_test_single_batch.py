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
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录的父目录添加到 sys.path 中
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from productscene.backends.productscene_predictor import ProductScenePredictor

# do uint test
import yaml
import cv2
import time
import argparse
import numpy as np
from PIL import Image

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Image Inference")
    parser.add_argument(
        "--list_size", type=int, default=1, help="num of input image in list"
    )
    args = parser.parse_args()

    pdpredictor = ProductScenePredictor("./models/config.yaml")

    config = yaml.safe_load(open("./models/config.yaml").read())
    predictor = config["predictors"]["predictor"]
    perf_test_times = config["predictors"][predictor]["perf_test_times"]

    images_root = "./examples/test_data/"
    foreground_images = []
    template_cfgs = []
    image_sizes = []
    names = []

    output_size = 768
    for image_name in sorted(os.listdir(images_root)):

        if "foreground" in image_name:
            foreground_path = os.path.join(images_root, image_name)
        else:
            continue

        config_path = os.path.join(
            images_root, "test_template_config.txt"
        )
        foreground_image = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
        template_cfg = json.loads(open(config_path).read())

        foreground_images.append(foreground_image)
        template_cfgs.append(template_cfg)
        image_sizes.append(output_size)

        names.append(image_name)

    run_list_size = min(len(foreground_images), args.list_size)

    warm_times = config["predictors"][predictor]["warm_times"]

    ori_image = foreground_images[0]
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGRA2RGB)
    mask = np.tile(ori_image[:, :, -1][..., np.newaxis], [1, 1, 3])
    template_cfg = template_cfgs[0]
    prompt = template_cfg['text_prompt']
    center = template_cfg['extra_info']['target_center']
    template_h = template_cfg['extra_info']['target_h']
    template_w = template_cfg['extra_info']['target_w']
    if 'sharpen' in template_cfg['extra_info']:
        is_sharpen = template_cfg['extra_info']['sharpen']
        sharpen = False if int(is_sharpen) == 0 else True
    else:
        sharpen = True
    out_h = 768
    out_w = 768
    for i in range(5):
        output_images = pdpredictor.forward(
            image,
            mask,
            prompt,
            sharpen,
            out_h,
            out_w,
            center,
            template_h,
            template_w,
        )

    for idx, image in enumerate(output_images):
        out_name = f"{idx}.jpg"
        cv2.imwrite("./output/generate_" + out_name, image)


