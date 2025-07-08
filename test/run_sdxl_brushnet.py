import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

# for relative path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from productscene.pipeline.api.sdxl_brushnet import SDXLBrushNetRunner
from upload_tool.loader import download,ImageLoader
from upload_tool.upload import upload_single_bytes_to_mms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Inference')
    parser.add_argument('--config', type=str)
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--logdir', type=str, help='Path to the log directory')
    parser.add_argument('--save_local', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    infe_config = json.load(open(args.config))
    cv2_loader = ImageLoader(loader_name="cv2")
    # init model
    runner = SDXLBrushNetRunner(infe_config['model']['sdxl_ckpt'], infe_config['model']['brushnet_ckpt'],
                                steps=infe_config['model']['steps'],
                                guidance_scale=infe_config['model']['guidance_scale'],
                                multistep_scheduler=infe_config['model']['multistep_scheduler'])

    df = pd.read_csv(args.input_csv)

    url = "https://down-br.img.susercontent.com/"
    results = []
    for i, row in df.iterrows():
        # load foreground
        fg_rgba_id = row[infe_config['input']['keys']['product_fg']]
        rgba_image = url + fg_rgba_id
        foreground_buf = download(url=rgba_image)
        foreground_image = cv2_loader(foreground_buf, rgba_flag=True)
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
        output_image = cv2.cvtColor(output_image[0], cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        generate_byte_data = buffer.tobytes()
        res_id, generate_url = upload_single_bytes_to_mms(generate_byte_data)

        if args.save_local:
            save_to = os.path.join(args.logdir, f'{fg_rgba_id}.jpg')
            cv2.imwrite(save_to, output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

        results.append(res_id)

    df['gen_jug_brushnet'] = results
    df.to_csv(os.path.join(args.logdir, args.input_csv.split('/')[-1].replace('.csv', '_generation.csv')), index=False)
