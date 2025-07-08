import argparse
import json
import os
import sys

import cv2
import numpy as np

# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 将当前文件所在目录的父目录添加到 sys.path 中
# parent_dir = os.path.join(current_dir, "..")
# sys.path.append(parent_dir)
from .api.sdxl_brushnet_ipadapter import SDXLBrushNetIpadapterRunner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Inference')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--foreground', type=str, help='Path to the input image, rgba')
    parser.add_argument('--template_image', type=str, help='Path to the input template, rgba')
    parser.add_argument('--template_config', type=str, help='template config, json format')
    parser.add_argument('--logdir', type=str, help='Path to the log directory')
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    # init model
    runner = SDXLBrushNetIpadapterRunner("stabilityai/stable-diffusion-xl-base-1.0", args.ckpt)

    # load foreground
    ori_image = cv2.imread(args.foreground, cv2.IMREAD_UNCHANGED)  # input1: rgba, foreground
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGRA2RGB)
    mask = np.tile(ori_image[:, :, -1][..., np.newaxis], [1, 1, 3])

    # load template image and config
    ori_ip_image = cv2.imread(args.template_image, cv2.IMREAD_UNCHANGED)  # input2: rgba, template image
    ip_image = cv2.cvtColor(ori_ip_image, cv2.COLOR_BGRA2RGB)
    ip_mask = 255 - np.tile(ori_ip_image[:, :, -1][..., np.newaxis], [1, 1, 3])

    template_cfg = json.loads(open(args.template_config).read())  # input3: json str, template config
    # remove
    # ip_adapter_scale = template_cfg['ip_adapter_scale']
    # prompt = template_cfg['text_prompt']
    # template_id = template_cfg['theme'] + '-' + template_cfg['image_prompt'].split('.')[0]
    # bottom = None if 'bottom' not in template_cfg else template_cfg['bottom']

    # update
    prompt = template_cfg['text_prompt']
    ip_adapter_scale = template_cfg['extra_info']['ip_adapter_scale']
    center = template_cfg['extra_info']['target_center']
    h = template_cfg['extra_info']['target_h']
    w = template_cfg['extra_info']['target_w']

    # run
    gen, gen_addfg = runner(image, mask, ip_image, ip_mask, prompt, ip_adapter_scale,
                            out_h=768, out_w=768, center=center, template_h=h, template_w=w)

    # save result
    template_id = "example"
    image_name = args.foreground.split('/')[-1].split('.')[0]
    save_to = os.path.join(args.logdir, f'{image_name}_{template_id}.jpg')
    gen_addfg = cv2.cvtColor(gen_addfg, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_to, gen_addfg, [cv2.IMWRITE_JPEG_QUALITY, 100])
