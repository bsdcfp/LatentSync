######################################################################
#
# Copyright (c) 2024 Shopee Inc. All Rights Reserved.
#
######################################################################

"""
file: server_http.py
author: songsong.yi@shopee.com
date: 2024-06-19 14:45:00
brief: reference from https://git.garena.com/shopee/MLP/aip/toolchains/oneservice/maas/mass-http-demo/-/blob/master/server_http.py
"""

import cv2
import glob
import os
import json

from aipinfer import logger, exceptions
from typing import Optional, Dict, List, Any

from productscene.backends.productscene_predictor import ProductScenePredictor

from oneservice import BaseProcessor, ResParams, MonitorTimer, ReqParams, server


class Processor(BaseProcessor):
    def __init__(self, config_file: str):

        # NOTE: 创建Predictor
        self._product_predictor = ProductScenePredictor(config_file=config_file)

    def __call__(self, req_params: ReqParams, res_params: ResParams):
        # NOTE: 解析请求参数获取推理所需参数
        with MonitorTimer("decode"):
            data = req_params.data

            check_inputs_none = (
                (data.get("aigc_segmentation_image_id") is None)
                or (data.get("generated_config") is None)
            )

            if check_inputs_none:
                res_params.err_num = exceptions.NUM_ILLEGAL_ARGS
                res_params.err_msg = "aigc_segmentation_image_id/generated_config cann't find in request"
                return

            aigc_segmentation_image_ids = data["aigc_segmentation_image_id"]
            generated_configs = data["generated_config"]

            check_args_valid = (
                isinstance(aigc_segmentation_image_ids, list)
                and len(aigc_segmentation_image_ids) > 0
                and isinstance(generated_configs, list)
                and len(generated_configs) > 0
                and len(aigc_segmentation_image_ids) == len(generated_configs)
                and len(aigc_segmentation_image_ids) <= 8
            )

            # NTOE: 输入错误，请检查输入
            if not check_args_valid:
                res_params.err_num = exceptions.NUM_ILLEGAL_ARGS
                if len(aigc_segmentation_image_ids) > 8:
                    res_params.err_msg = "The input image url list cannot exceed 8"
                else:
                    res_params.err_msg = exceptions.desc_of_num(
                        exceptions.NUM_ILLEGAL_ARGS
                    )
                return

            indices, input_ids, configs = [], [], []
            for idx, aigc_segmentation_image_id in enumerate(aigc_segmentation_image_ids):
                try:
                    input_ids.append(aigc_segmentation_image_id)
                    indices.append(idx)
                    configs.append(generated_configs[idx])

                except Exception:
                    err_indices.append(idx)
                    pass

        with MonitorTimer("predictor"):
            try:
                # NOTE: 执行predictor, 并且设置ResParams
                err_infos = []
                aigc_generated_image_id, aigc_generated_mask_id, status_code, has_nsfw = self._product_predictor.predict(
                    aigc_segmentation_image_id=input_ids,
                    generated_config=configs
                )
                for item in has_nsfw:
                    if item:
                        err_infos.append("detect nsfw !!!")
            except Exception:
                res_params.err_num = exceptions.NUM_BACKEND_ERROR
                res_params.err_msg = exceptions.desc_of_num(
                    exceptions.NUM_BACKEND_ERROR
                )
                return

        # 根据输入类型设置正确的返回字段名
        # 检测第一个输入的类型来确定返回格式
        first_input = input_ids[0] if input_ids else ""
        
        # 简单检测是否为base64数据
        is_base64_input = len(first_input) > 100 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in first_input[:100])
        
        if is_base64_input:
            # 输入是base64，返回base64格式
            res_params.result = {
                "aigc_generated_base64": aigc_generated_image_id, 
                "aigc_generated_mask_base64": aigc_generated_mask_id, 
                "status_code": status_code, 
                "err_infos": err_infos,
                "input_type": "base64"
            }
        else:
            # 输入是image_id，返回image_id格式
            res_params.result = {
                "aigc_generated_image_id": aigc_generated_image_id, 
                "aigc_generated_mask_id": aigc_generated_mask_id, 
                "status_code": status_code, 
                "err_infos": err_infos,
                "input_type": "image_id"
            }

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

# NOTE: Flask的应用程序入口方式
logger.info(f"[SERVER] Initializing ProductScene Processor with config: {config_file}")
server.register(Processor(config_file=config_file), ReqParams, ResParams)
app = server.app

if __name__ == "__main__":
    server.app.run(host="127.0.0.1", port=80, threaded=True)
