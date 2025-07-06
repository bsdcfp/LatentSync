import argparse
import base64
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import requests
import tqdm
from joblib import Parallel, delayed

sys.path.append('.')
from config.shopee_base import register_upload_config
from config.status_code import PI_STATUS_CODE
from src.api.builder import build_runner
from src.utils.loader import download_sp_image, upload_image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Image Inference')
parser.add_argument('--config', type=str, default="config/e2e_config/sdxl-brushnet-ipadapter_scene-dd.json")
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
product_fg_key = e2e_config['input']['keys']['product_fg']
product_cate_key = e2e_config['input']['keys']['product_category']
upload_config = register_upload_config[e2e_config['output']['uploader']]
template_scene = e2e_config['template']['scene']
template_version = e2e_config['template']['version']

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

def api_scene_match(image_data, global_be_category_id,
                    template_scene='dd',
                    size_overwrite=[0.9, 0.9],
                    img_num=1,
                    versions=['dd'],
                    retry=3):
    """
    缓存加速版
    :param image_list:
    :param img_num:
    :param global_be_category_id:
    :param template_scene: seller_tools or dd
    :param size_overwrite:
    :param retry:
    :return:
    """

    image_list = [
        {
            # "image_url": image_data,
            "image_data": image_data,
            "extra_info": "{\"image_type\": \"foreground\"}"
        }
    ]

    url = "http://http-gateway.spex.shopee.sg/sprpc/ai_engine_platform.mmuplt.scenematch.algo"
    # url = "https://http-gateway.spex.test.shopee.sg/sprpc/ai_engine_platform.mmuplt.scenematch.algo"


    header = {
        "Content-Type": "application/json",
        "x-sp-sdu": "ai_engine_platform.mmuplt.controller.global.live.master.default",
        "x-sp-servicekey": "f0dd2d544097d2a938595c1d78949bd3",
        "x-sp-timeout": "60000",
        "shopee-baggage": "CID=br"

        # uat环境
        # "x-sp-sdu": "mmuplt.scenematch.global.uat.master.default",
        # "x-sp-servicekey": "ab36c555f0fd3fb118231d730da6364e",
        # "x-sp-timeout": "300000",
        # "shopee-baggage": "CID=global",
        # "x-sp-debug": '1'
    }

    data = {
        "biz_type": "mmu_algo_test",
        "region": "sg",
        "task": {
            "image_list": image_list,
            "extra_info": json.dumps(
                {
                    "img_num": img_num,
                    "global_be_category_id": global_be_category_id,
                    "template_scene": template_scene,
                    "size_overwrite": size_overwrite,
                    "versions": versions,
                }
            )
        }
    }

    for _ in range(retry):
        try:
            rsp = requests.post(url=url, headers=header, json=data)
            result = rsp.json()
            if result['code'] == 0:
                result = result['task_result']['details']
                return result
        except Exception as e:
            logger.error(f"An exception occurred while calling match template: {e}")
            time.sleep(1)
    return None


def convert_base64_image(cv2_img):
    try:
        _, image_binary = cv2.imencode('.png', cv2_img)
        base64_image = base64.b64encode(image_binary).decode('utf-8')
    except Exception as e:
        logger.error(f"An exception occurred while encode fgimage: {e}")
        return None
    return base64_image


def process_row(row):
    fg_rgba_id = row[product_fg_key]
    category = row[product_cate_key]

    # 缓存图片: us机器下载图片较慢, 异步先缓存, 顺便用于模版请求
    fg_image = download_sp_image(fg_rgba_id, cv2.IMREAD_UNCHANGED)
    image_data = convert_base64_image(fg_image)
    category_id = l12id[category]
    templates = api_scene_match(image_data, category_id, template_scene=template_scene, versions=template_version)
    
    if templates is None:
        return {
            'fg_rgba_id': fg_rgba_id, 
            'status_code': PI_STATUS_CODE.MATCH_ERROR
        }

    template = json.loads(templates[0]['extra_info'])
    template_image_id = template["template_image_id"]
    template_config = template["template_config"]

    return {
        'fg_rgba_id': fg_rgba_id, 
        'template_image_id': template_image_id, 
        'template_config': template_config,
        'template_version': template["template_version"],
        'status_code': PI_STATUS_CODE.SUCCESS
    }


def worker(rows):
    os.makedirs(args.logdir, exist_ok=True)

    # init model
    runner = build_runner(**e2e_config['model'])

    results = []
    uploader_executor = ThreadPoolExecutor(max_workers=2)
    with ThreadPoolExecutor(max_workers=4) as executor:
        preprocess_results = [[row[product_fg_key], executor.submit(process_row, row)] for row in rows]
        
        for fg_rgba_id, preprocess_res in tqdm.tqdm(preprocess_results):
            # 模版匹配
            start = time.time()
            try:
                result = preprocess_res.result()
            except Exception as e:
                logger.error(f"{fg_rgba_id} match error: {str(e)}")
                results.append({"fg_rgba_id": fg_rgba_id, "status_code": PI_STATUS_CODE.MATCH_ERROR})
                continue
            logging.info('match time: {:.6f}'.format(time.time()-start))
            start = time.time()

            if result['status_code'] != PI_STATUS_CODE.SUCCESS:
                results.append(result)
                continue

            # 图像 & 模版下载
            rgba_id = result['fg_rgba_id']
            match_template_image_id = result['template_image_id']
            match_template_config = result['template_config']
            template_cfg = json.loads(match_template_config)
            sharpen = True

            # 商品前景图
            ori_image = download_sp_image(rgba_id, cv2.IMREAD_UNCHANGED)
            if ori_image is None:
                result['status_code'] = PI_STATUS_CODE.DOWNLOAD_FG_ERROR
                results.append(result)
                continue
            image = cv2.cvtColor(ori_image, cv2.COLOR_BGRA2RGB)
            mask = np.tile(ori_image[:, :, -1][..., np.newaxis], [1, 1, 3])

            # 模版图和config
            ori_ip_image = download_sp_image(match_template_image_id, cv2.IMREAD_UNCHANGED)
            if ori_ip_image is None:
                result['status_code'] = PI_STATUS_CODE.DOWNLOAD_TEMPLATE_ERROR
                results.append(result)
                continue
            ip_image = cv2.cvtColor(ori_ip_image, cv2.COLOR_BGRA2RGB)
            ip_mask = 255 - np.tile(ori_ip_image[:, :, -1][..., np.newaxis], [1, 1, 3])

            # template config
            prompt = template_cfg['text_prompt']
            ip_adapter_scale = template_cfg['extra_info']['ip_adapter_scale']
            center = template_cfg['extra_info']['target_center']
            h = template_cfg['extra_info']['target_h']
            w = template_cfg['extra_info']['target_w']
            logger.info('generation preprocess time: {:.6f}'.format(time.time()-start))
            start = time.time()

            # run
            gen_results = runner(image, mask,ip_image, ip_mask, prompt, ip_adapter_scale, 
                                 out_h=e2e_config['output']['out_h'], out_w=e2e_config['output']['out_w'], center=center, template_h=h, template_w=w, sharpen=sharpen)
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
    records = df.to_dict('records')

    print('Data samples count: ', len(records))

    num_worker = int(args.num_worker)
    batch_size = len(records) // num_worker + int((len(records) % num_worker) != 0)
    results = Parallel(n_jobs=num_worker)(delayed(worker)(records[idx*batch_size:(idx+1)*batch_size]) for idx in range(num_worker))
    res_df = pd.concat(results, axis=0)

    csv_suffix = f'_gen_{args.part}.csv' if args.part != -1 else '_gen.csv'
    res_df.to_csv(os.path.join(args.logdir, args.csv.split('/')[-1].replace('.csv', csv_suffix)), index=False)