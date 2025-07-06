import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import tqdm
from joblib import Parallel, delayed

sys.path.append('.')
from config.shopee_base import register_upload_config
from config.status_code import PI_STATUS_CODE
from src.api.white_bg import WhiteBGRunner
from src.utils.loader import download_sp_image, upload_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Image Inference')
parser.add_argument('--config', type=str, default="config/e2e_config/sdxl-brushnet_match.json")
parser.add_argument('--csv', type=str)
parser.add_argument('--logdir', type=str, help='Path to the log directory')
parser.add_argument('--part', type=int, default=-1, 
    help='index of the target csv part (default part size is 5000), default=-1 mean: use all part')
parser.add_argument('--part_size', type=int, default=5000)
parser.add_argument('--num_worker', type=int, default=4)
args = parser.parse_args()


global_url_prefixed = os.environ.get('PI_URL_PREFIXED', 'https://cf.shopee.sg/file/')

# e2e推理config
e2e_config = json.load(open(args.config))
size_overwrite = e2e_config['template']['size_overwrite']
center_overwrite = e2e_config['template']['center_overwrite']
product_fg_key = e2e_config['input']['keys']['product_fg']
product_cate_key = e2e_config['input']['keys']['product_category']
upload_config = register_upload_config[e2e_config['output']['uploader']]

# category mapping
l12id = {
    'Beauty': 100630,
    'Men Bags': 100533,
    'Women Bags': 100016,
    'Men Shoes': 100012,
    'Women Shoes': 100532,
    'Men Clothes': 100011,
    'Women Clothes': 100017,
    'Baby & Kids Fashion': 100633,
}


def process_row(row):
    fg_rgba_id = row[product_fg_key]
    
    template_cfg = {
        'extra_info': {
            'target_h': size_overwrite[0],
            'target_w': size_overwrite[1],
            'target_center': [0.5, 0.5]
        }
    }

    return {
        'fg_rgba_id': fg_rgba_id, 
        'template_config': template_cfg,
        'status_code': PI_STATUS_CODE.SUCCESS
    }


def worker(rows):
    # init model
    runner = WhiteBGRunner(gen_h=e2e_config['model']['gen_h'], gen_w=e2e_config['model']['gen_w'])
    
    results = []
    uploader_executor = ThreadPoolExecutor(max_workers=2)
    with ThreadPoolExecutor(max_workers=4) as executor:
        preprocess_results = [[row[product_fg_key], executor.submit(process_row, row)] for row in rows]
        
        for fg_rgba_id, preprocess_res in tqdm.tqdm(preprocess_results):
            # 生成prompt
            start = time.time()
            try:
                result = preprocess_res.result()
            except Exception as e:
                logger.error(f"{fg_rgba_id} preprocess: {str(e)}")
                results.append({"fg_rgba_id": fg_rgba_id, "status_code": PI_STATUS_CODE.GEN_PROMPT_ERROR})
                continue
            logging.info('gen time: {:.6f}'.format(time.time()-start))
            start = time.time()

            if result['status_code'] != PI_STATUS_CODE.SUCCESS:
                results.append(result)
                continue

            # 图像 & 模版下载
            rgba_id = result['fg_rgba_id']
            template_cfg = result['template_config']
            sharpen = e2e_config['model']['sharpen']

            # 商品前景图
            ori_image = download_sp_image(rgba_id, cv2.IMREAD_UNCHANGED)
            if ori_image is None:
                result['status_code'] = PI_STATUS_CODE.DOWNLOAD_FG_ERROR
                results.append(result)
                continue
            image = cv2.cvtColor(ori_image, cv2.COLOR_BGRA2RGB)
            mask = np.tile(ori_image[:, :, -1][..., np.newaxis], [1, 1, 3])

            # template config
            center = template_cfg['extra_info']['target_center']
            h = template_cfg['extra_info']['target_h']
            w = template_cfg['extra_info']['target_w']
            logger.info('generation preprocess time: {:.6f}'.format(time.time()-start))
            start = time.time()
            result['template_config'] = json.dumps(result['template_config'])

            # run
            gen_results = runner(image, mask, sharpen=sharpen,
                                 out_h=e2e_config['output']['out_h'], out_w=e2e_config['output']['out_w'], 
                                 template_h=h, template_w=w, center=center)
            logger.info('generation time: {}'.format(time.time()-start))
            start = time.time()

            # 上传图片到shopee服务器
            # image
            gen_addfg = cv2.cvtColor(gen_results['gen_addfg'], cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', gen_addfg, [cv2.IMWRITE_JPEG_QUALITY, 100])
            upload_res = uploader_executor.submit(upload_image, buffer.tobytes(), upload_config)
            result['gen_image'] = upload_res

            # mask
            if 'gen_mask' in e2e_config['output']['keys']:
                _, buffer = cv2.imencode('.png', gen_results['gen_mask'])
                upload_res = uploader_executor.submit(upload_image, buffer.tobytes(), upload_config)
                result['gen_mask'] = upload_res
            
            results.append(result)

    # 输出结果到csv文件
    output_keys = list(e2e_config['output']['keys'].keys())
    outputs = []
    for result in results:
        output = []

        # 处理错误case
        if result["status_code"] != PI_STATUS_CODE.SUCCESS:
            for key in output_keys:
                if key in result:
                    output.append(result[key])
                else:
                    output.append('')
        else:
            for key in output_keys:
                if key in ['gen_image', 'gen_mask']:
                    try:
                        img_id = result[key].result()
                        output.append(img_id)
                    except:
                        output.append('')
                        result["status_code"] = PI_STATUS_CODE.UPLOAD_ERROR
                else:
                    output.append(result[key] if key in result else '')
        output.append(result["status_code"])
        outputs.append(output)

    columns = [e2e_config['output']['keys'][k] for k in output_keys] + ['status_code']
    res_df = pd.DataFrame(outputs, columns=columns)
    return res_df


if __name__ == '__main__':
    os.makedirs(args.logdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[args.part_size*args.part:args.part_size*(args.part+1)] if args.part != -1 else df
    records = df[:2].to_dict('records')

    print('Data samples count: ', len(records))

    num_worker = int(args.num_worker)
    batch_size = len(records) // num_worker + int((len(records) % num_worker) != 0)
    results = Parallel(n_jobs=num_worker)(delayed(worker)(records[idx*batch_size:(idx+1)*batch_size]) for idx in range(num_worker))
    res_df = pd.concat(results, axis=0)

    csv_suffix = f'_gen_{args.part}.csv' if args.part != -1 else '_gen.csv'
    res_df.to_csv(os.path.join(args.logdir, args.csv.split('/')[-1].replace('.csv', csv_suffix)), index=False)