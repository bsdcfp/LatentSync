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
    out_size = config["predictors"][predictor]["size"]

    test_csv = "./examples/test_case/test_case.csv"
    images_root = "./examples/test_case"
    #download image
    cv2_loader = ImageLoader(loader_name="cv2")
    with open(test_csv, newline='') as csvfile:
        spamreader = csv.DictReader(csvfile)
        idx = 0
        for row in spamreader:
            print(row)
            url = "https://down-br.img.susercontent.com/"
            rgba_image = url + row['seg_image_id']
            foreground_buf = download(url=rgba_image)
            fbuf = cv2_loader(foreground_buf, rgba_flag=True)
            fname = images_root + '/' + f"{idx}" + "_forground.jpg"
            cv2.imwrite(fname,fbuf)

            gen_jug_brushnet_image = url + row['gen_jug_brushnet']
            res_buf = download(url=gen_jug_brushnet_image)
            rbuf = cv2_loader(res_buf, rgba_flag=True)
            fname = images_root + '/' + f"{idx}" + "_res.jpg"
            cv2.imwrite(fname,rbuf)

            template_config = row['template_config']
            template_cfg = json.loads(template_config)

            output_images = bg_tryon_predictor.predict(
                            foregrounds=[fbuf],
                            template_configs=[template_cfg],
                            image_sizes=[out_size]
                            )
            oname = images_root + '/' + f"{idx}" + "_out.jpg"
            cv2.imwrite(oname, output_images[0])
            tname = "./examples/torch_result/" + f"{idx}" + "_out.jpg"
            if os.path.exists(tname):
                tr = cv2.imread(tname)
                combined_image = np.hstack((rbuf,tr, output_images[0]))
                cv2.imwrite(images_root + "/left_groundtruth_mid_torch_right_out_"+f"{idx}" + ".jpg", combined_image)
            else:
                combined_image = np.hstack((rbuf,output_images[0]))
                cv2.imwrite(images_root + "/left_groundtruth_right_out_"+f"{idx}" + ".jpg", combined_image)
            idx = idx + 1



