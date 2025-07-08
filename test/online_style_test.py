
import cv2
import glob
import os
import json
import time
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录的父目录添加到 sys.path 中
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from aipinfer import logger, exceptions
from typing import Optional, Dict, List, Any
from upload_tool.upload import upload_single_bytes_to_mms
from upload_tool.loader import download, ImageLoader

from productscene.backends.productscene_predictor import ProductScenePredictor

from oneservice import BaseProcessor, ResParams, MonitorTimer, ReqParams, server


class Processor(BaseProcessor):
    def __init__(self, config_file: str):

        # NOTE: 创建Predictor
        self._tryon_predictor = ProductScenePredictor(config_file=config_file)
        self._cv2_loader = ImageLoader(loader_name="cv2")

    def __call__(self, req_params, res_params):
        # NOTE: 解析请求参数获取推理所需参数
        t0 = time.time()
        data = req_params.data

        check_inputs_none = (
            (data.get("foreground_url_list") is None)
            or (data.get("template_config_list") is None)
        )
        if check_inputs_none:
            res_params.err_num = exceptions.NUM_ILLEGAL_ARGS
            res_params.err_msg = "foreground_url_list/template_config_list cann't find in request"
            return

        foreground_urls = data["foreground_url_list"]
        template_configs = data["template_config_list"]

        check_args_valid = (
            isinstance(foreground_urls, list)
            and len(foreground_urls) > 0
            and isinstance(template_configs, list)
            and len(template_configs) > 0
            and len(foreground_urls) == len(template_configs)
            and len(foreground_urls) <= 8
        )
        # NTOE: 输入错误，请检查输入
        if not check_args_valid:
            res_params.err_num = exceptions.NUM_ILLEGAL_ARGS
            if len(foreground_urls) > 8:
                res_params.err_msg = "The input image url list cannot exceed 8"
            else:
                res_params.err_msg = exceptions.desc_of_num(
                    exceptions.NUM_ILLEGAL_ARGS
                )
            return
        indices, foregrounds, template_images, err_indices = [], [], [], []
        for idx, foreground_url in enumerate(foreground_urls):
            try:
                foreground_buf = download(url=foreground_url)

                foregrounds.append(
                    self._cv2_loader(foreground_buf, rgba_flag=True),
                )
                indices.append(idx)
            except Exception:
                err_indices.append(idx)
                pass

        t1 = time.time()
        try:
            # NOTE: 执行predictor, 并且设置ResParams
            res_images = self._tryon_predictor.predict(
                foregrounds=foregrounds,
                template_configs=template_configs,
                image_sizes = [768]*len(foregrounds)
            )
        except Exception:
            res_params.err_num = exceptions.NUM_BACKEND_ERROR
            res_params.err_msg = exceptions.desc_of_num(
                exceptions.NUM_BACKEND_ERROR
            )
            return

        t2 = time.time()
        # NOTE: upload result to cdn
        result_urls = []
        print(" ====>> ", len(res_images))
        for res_image in res_images:
            s0 = time.time()
            _, generate_image = cv2.imencode(".png", res_image)
            s1 = time.time()
            generate_byte_data = generate_image.tobytes()
            s2 = time.time()
            generate_id, generate_url = upload_single_bytes_to_mms(
                generate_byte_data
            )
            s3 = time.time()
            result_urls.append(
                (generate_id, generate_url)
            )
            print(" ====>> upload ", (s3 - s2)*1000, (s2 - s1)*1000 ," | ", (s1 - s0)*1000)
        t3 = time.time()
        image_list = [""] * len(foreground_urls)
        err_infos = [""] * len(foreground_urls)

        for idx, url in zip(indices, result_urls):
            if url[1] == None:
                err_infos[idx] = url[0]
            else:
                image_list[idx] = url[1]

        for idx in err_indices:
            err_infos[idx] = "download request url failure or image file broken"

        res_params.result = {"image_list": image_list, "err_infos": err_infos}
        print(f" ====> all pipeline time  {(t3 - t0) * 1000}( {100 * (t3 - t0)/(t3 - t0)} ) : {(t1 - t0) * 1000}({100 * (t1 - t0)/(t3 - t0)}) | {(t2 - t1) * 1000}({100 * (t2 - t1)/(t3 - t0)}) | {(t3 - t2) * 1000}({100 * (t3 - t2)/(t3 - t0)})")
        return

# NOTE: 从MaaS标准模型路径获取config文件
def find_config_yaml_files(dir_path: str):
    return glob.glob(f"{dir_path}/**/config.yaml", recursive=True)

class DotDict(dict):
    """Dictionary subclass that supports dot notation."""
    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, attr, value):
        self[attr] = value

if __name__ == "__main__":
    aip_model_dir = os.environ.get("AIP_MODEL_PATH", "./models")
    config_file = find_config_yaml_files(aip_model_dir)[0]
    p = Processor(config_file=config_file)
    req_params = {
      "data": {
                "foreground_url_list":
                    ["https://down-br.img.susercontent.com/br-11134253-7r98o-ly1qmp3d40omde"]
                ,

                "template_config_list": [
                                "{\"text_prompt\":\"a foreground item palced on a wooden surface, empty surface, soft light, shadow\",\"extra_info\":{\"ip_adapter_scale\":0.6,\"target_h\":0.8,\"target_w\":0.8,\"target_center\":[0.5,0.5]}}"
                            ]

        }
    }
    res_params = {"err_num":0,"err_msg":"none","result":{}}

    req_params=DotDict(req_params)
    res_params=DotDict(res_params)

    for i in range(0,5):
        p(req_params,res_params)
    print(" ================================================================= after warm =================================")
    p(req_params,res_params)
