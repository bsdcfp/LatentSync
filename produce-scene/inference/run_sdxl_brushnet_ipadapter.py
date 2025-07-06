import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.append('.')
from config.shopee_base import upload_config_4249
from src.api.sdxl_brushnet_ipadapter import SDXLBrushNetIpadapterRunner
from src.utils.loader import download_sp_image, upload_image


def inference(runner, foreground_image, template_image, template_config, sharpen=False):
    """
    args:
        foreground_image (np.ndarray): BGRA, decode type require cv2.IMREAD_UNCHANGED
        template_image (np.ndarray): BGRA, decode type require cv2.IMREAD_UNCHANGED
        template_config (str): output of json.dumps(cfg_dict)
    """
    image = cv2.cvtColor(foreground_image, cv2.COLOR_BGRA2RGB)
    mask = np.tile(foreground_image[:, :, -1][..., np.newaxis], [1, 1, 3])

    ip_image = cv2.cvtColor(template_image, cv2.COLOR_BGRA2RGB)
    ip_mask = 255 - np.tile(template_image[:, :, -1][..., np.newaxis], [1, 1, 3])

    template_config = json.loads(template_config)
    prompt = template_config['text_prompt']
    ip_adapter_scale = template_config['extra_info']['ip_adapter_scale']
    center = template_config['extra_info']['target_center']
    h = template_config['extra_info']['target_h']
    w = template_config['extra_info']['target_w']

    _, gen_addfg = runner(image, mask, ip_image, ip_mask, prompt, ip_adapter_scale, 
                          out_h=768, out_w=768, center=center, template_h=h, template_w=w, sharpen=sharpen)
    return gen_addfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Inference')
    parser.add_argument('--sdxl_ckpt', type=str)
    parser.add_argument('--brushnet_ckpt', type=str)
    parser.add_argument('--ipadapter_ckpt', type=str)
    parser.add_argument('--match_csv', type=str, help='input csv, required col: rgba_id, match_template_image_id, match_template_config')
    parser.add_argument('--logdir', type=str, help='Path to the log directory')
    parser.add_argument('--save_local', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    # init model
    runner = SDXLBrushNetIpadapterRunner(args.sdxl_ckpt, args.brushnet_ckpt, args.ipadapter_ckpt)

    df = pd.read_csv(args.match_csv)
    res = []
    for i, row in df.iterrows():
        # load foreground
        ori_fg_image = download_sp_image(row['rgba_id'], cv2.IMREAD_UNCHANGED) # input1: rgba, foreground
        if ori_fg_image is None: 
            res.append('download_fg_error')
            continue

        # load template image and config
        ori_ip_image = download_sp_image(row['match_template_image_id'], cv2.IMREAD_UNCHANGED)
        if ori_ip_image is None: 
            res.append('download_template_error')
            continue
        template_cfg = row['match_template_config']

        # run
        output_image = inference(runner, ori_fg_image, ori_ip_image, template_cfg)
        
        # save result: output_image is RGB
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        res_id = upload_image(buffer.tobytes(), upload_config_4249)
        res.append(res_id)

        if args.save_local:
            rgba_id = row['rgba_id']
            template_id = row['match_template_image_id']
            save_to = os.path.join(args.logdir, f'{image_name}_{template_id}.jpg')
            cv2.imwrite(save_to, output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
    df['generation_image_id'] = res
    df.to_csv(os.path.join(args.logdir, args.match_csv.replace('.csv', '_generation.csv')), index=False)
