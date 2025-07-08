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
from upload_tool.loader import download,ImageLoader
from upload_tool.upload import upload_single_bytes_to_mms
# do uint test
import yaml
import cv2
import time
import argparse
import numpy as np
from PIL import Image
import csv

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Image Inference")
    parser.add_argument(
        "--list_size", type=int, default=1, help="num of input image in list"
    )
    args = parser.parse_args()

    bg_tryon_predictor = ProductScenePredictor("./models/config.yaml")

    config = yaml.safe_load(open("./models/config.yaml").read())
    predictor = config["predictors"]["predictor"]
    perf_test_times = config["predictors"][predictor]["perf_test_times"]
    out_size = 1024

    test_csv = "./examples/test_case/test_case2.csv"
    images_root = "./examples/test_case"
    #download image
    cv2_loader = ImageLoader(loader_name="cv2")
    with open(test_csv, newline='') as csvfile:
        spamreader = csv.DictReader(csvfile)
        rows = list(spamreader)
        idx = 0
        fieldnames = spamreader.fieldnames + ['opt_image_id']  # Add a new column for result URLs
        l = len(rows)
        for idx,row in enumerate(rows):
            print(f" ====> {idx/l:.2%}")
            url = "https://down-br.img.susercontent.com/"
            rgba_image = url + row['fg_rgba_id']
            foreground_buf = download(url=rgba_image)
            fbuf = cv2_loader(foreground_buf, rgba_flag=True)
            fname = images_root + '/' + f"{idx}" + "_forground.jpg"
            # cv2.imwrite(fname,fbuf)

            gen_jug_brushnet_image = url + row['aigc_generated_image_id']
            res_buf = download(url=gen_jug_brushnet_image)
            rbuf = cv2_loader(res_buf, rgba_flag=True)
            fname = images_root + '/' + f"{idx}" + "_res.jpg"
            # cv2.imwrite(fname,rbuf)

            template_config = row['template_config']
            template_cfg = json.loads(template_config)

            output_images = bg_tryon_predictor.predict(
                            foregrounds=[fbuf],
                            template_configs=[template_cfg],
                            image_sizes=[out_size]
                            )

            result_urls = []
            for res_image in output_images:
                _, generate_image = cv2.imencode(".png", res_image)
                generate_byte_data = generate_image.tobytes()
                generate_id, generate_url = upload_single_bytes_to_mms(
                    generate_byte_data
                )

                result_urls.append(
                    (generate_id, generate_url)
                )
            rows[idx]["opt_image_id"] = result_urls[0][0]

        with open(images_root + '/' + "test_case2_result.csv", 'w') as csvfile:
            spamwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            spamwriter.writeheader()
            spamwriter.writerows(rows)

