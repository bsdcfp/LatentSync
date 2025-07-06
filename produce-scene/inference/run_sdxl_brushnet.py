import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.append('.')
from config.shopee_base import upload_config_4249
from src.api.builder import build_runner
from src.utils.loader import download_sp_image, upload_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Inference')
    parser.add_argument('--config', type=str)
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--logdir', type=str, help='Path to the log directory')
    parser.add_argument('--save_local', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    infe_config = json.load(open(args.config))

    # init model
    runner = build_runner(**infe_config['model'])

    df = pd.read_csv(args.input_csv)
    
    results = []
    for i, row in df.iterrows():
        # load foreground
        fg_rgba_id = row[infe_config['input']['keys']['product_fg']]
        foreground_image = download_sp_image(fg_rgba_id, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(foreground_image, cv2.COLOR_BGRA2RGB)
        mask = np.tile(foreground_image[:, :, -1][..., np.newaxis], [1, 1, 3])

        # load template image and config
        template_config_str = row['template_config']
        template_config = json.loads(template_config_str)
        prompt = template_config['text_prompt']
        center = template_config['extra_info']['target_center']
        h = template_config['extra_info']['target_h']
        w = template_config['extra_info']['target_w']
        sharpen = True

        # run
        gen_results = runner(image, mask, prompt, sharpen=sharpen, 
                             out_h=infe_config['output']['out_h'], out_w=infe_config['output']['out_w'], 
                             template_h=h, template_w=w, center=center)
        
        output_image = gen_results['gen_addfg']
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        res_id = upload_image(buffer.tobytes(), upload_config_4249)

        if args.save_local:
            save_to = os.path.join(args.logdir, f'{fg_rgba_id}.jpg')
            cv2.imwrite(save_to, output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
        # results.append([row[infe_config['input']['product_fg']], prompt, res_id])
        results.append(res_id)

    df['gen_image_id'] = results
    df.to_csv(os.path.join(args.logdir, args.input_csv.split('/')[-1].replace('.csv', '_generation.csv')), index=False)
