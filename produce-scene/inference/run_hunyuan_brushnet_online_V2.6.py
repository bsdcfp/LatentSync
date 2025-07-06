import argparse
import copy
import json
import logging
import os
from random import randint
import sys

import cv2
import numpy as np
import pandas as pd
import tqdm
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed

sys.path.append('.')
from config.shopee_base import register_upload_config
from config.status_code import PI_STATUS_CODE
from src.api.builder import build_runner
from src.utils.loader import download_sp_image, upload_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Image Inference')
parser.add_argument('--config', type=str)
parser.add_argument('--csv', type=str)
parser.add_argument('--logdir', type=str, help='Path to the log directory')
parser.add_argument('--part', type=int, default=-1, 
    help='index of the target csv part (default part size is 5000), default=-1 mean: use all part')
parser.add_argument('--part_size', type=int, default=5000)
parser.add_argument('--num_worker', type=int, default=1)
args = parser.parse_args()


class Uploader:
    def __init__(self, cfg):
        self.upload_config = register_upload_config[cfg['gen_output']['uploader']]
        self.upload_mask = True
        
    def __call__(self, row, gen_image, gen_mask=None):
        uploader_executor = ThreadPoolExecutor(max_workers=2)
        try:
            # upload image to sp cdn
            gen_image = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', gen_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            upload_res = uploader_executor.submit(upload_image, buffer.tobytes(), self.upload_config)

            if self.upload_mask:
                _, mask_buffer = cv2.imencode('.png', gen_mask)
                upload_mask_res = uploader_executor.submit(upload_image, mask_buffer.tobytes(), self.upload_config)
                row['aigc_generated_mask_id'] = upload_mask_res.result()
            
            row['aigc_generated_image_id'] = upload_res.result()

        except Exception as e:
            row['status_code'] = PI_STATUS_CODE.UPLOAD_ERROR
            logger.info(f'Upload error: {e}')

        return row


def collecting_result(results, output_keys):
    outputs = []
    
    for result in results:
        output = [result[key] if key in result else '' for key in output_keys]                
        outputs.append(output)

    return outputs


def worker(rows):
    e2e_config = json.load(open(args.config))

    # init models
    runner = build_runner(**e2e_config['gen_model'])
    uploader = Uploader(e2e_config)

    results = []
    for row in tqdm.tqdm(rows):
        result = copy.deepcopy(row)
        result['status_code'] = PI_STATUS_CODE.SUCCESS

        # inputs
        fg_rgba_id = row['aigc_segmentation_image_id']
        foreground_image = download_sp_image(fg_rgba_id, cv2.IMREAD_UNCHANGED)
        if foreground_image is None:
            result['status_code'] = PI_STATUS_CODE.DOWNLOAD_FG_ERROR
            results.append(result)
            continue
        image = cv2.cvtColor(foreground_image, cv2.COLOR_BGRA2RGB)
        mask = np.tile(foreground_image[:, :, -1][..., np.newaxis], [1, 1, 3])

        generated_config = json.loads(row['generated_config'])
        prompt = generated_config['text_prompt']
        center = generated_config['target_center']
        h = generated_config['target_h']
        w = generated_config['target_w']
        out_h = generated_config['out_h']
        out_w = generated_config['out_w']
        seed = generated_config['seed']

        # gen
        gen_results = runner(image, mask, prompt, out_h=out_h, out_w=out_w, 
                             template_h=h, template_w=w, center=center, seed=seed)
        
        # outputs
        if gen_results['has_nsfw_concept']:
            result['status_code'] = PI_STATUS_CODE.NSFW_ERROR
            results.append(result)
            continue

        result = uploader(result, gen_results['gen_addfg'], gen_results['gen_mask'])
        
        # Output results in a unified format
        results.extend(collecting_result([result], e2e_config['gen_output']['keys']))
    return pd.DataFrame(results, columns=e2e_config['gen_output']['keys'])


if __name__ == '__main__':
    os.makedirs(args.logdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(len(df))

    df = df[args.part_size*args.part:args.part_size*(args.part+1)] if args.part != -1 else df
    records = df.to_dict('records')

    print('Data samples count: ', len(records))

    num_worker = int(args.num_worker)
    batch_size = len(records) // num_worker + int((len(records) % num_worker) != 0)
    results = Parallel(n_jobs=num_worker)(delayed(worker)(records[idx*batch_size:(idx+1)*batch_size]) for idx in range(num_worker))
    res_df = pd.concat(results, axis=0)

    csv_suffix = f'_gen_{args.part}.csv' if args.part != -1 else '_gen.csv'
    res_df.to_csv(os.path.join(args.logdir, args.csv.split('/')[-1].replace('.csv', csv_suffix)), index=False)