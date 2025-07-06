import logging
import os
import time

import cv2
import numpy as np
import requests

from core.config.client import init_default_config_mgr
from core.env import env
from imageSdk.protocol import FullUploadRequest, FullUploadFileRequest, Header
from imageSdk.sdk import new_upload_sdk_with_secret


# init logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# init default global image cache
global_image_cache = os.environ.get('PI_IMAGE_CACHE', '/tmp/image_cache')
global_url_prefixed = os.environ.get('PI_URL_PREFIXED', 'https://cf.shopee.sg/file/')


def download_sp_image(image_id, decode_type, use_cache=True, cache_dir=None, url_prefixed=None,
                      retry=3, timeout=30):
    """ download shopee image, and decode it by cv2
    args:
        image_id: shopee image id
        decode_type: enumerate
            cv2.IMREAD_COLOR: bgr image
            cv2.IMREAD_UNCHANGED: bgra image
            cv2.IMREAD_GRAYSCALE: gray image
        use_cache: optional, default = True, 
            when the image already exists in the cache, skip download
        cache_dir: optional, default = '/tmp/image_cache', 
            you can change the default parameters by using ENV 'PI_IMAGE_CACHE'
        url_prefixed: optional, default = 'https://cf.shopee.sg/file/', 
            you can change the default parameters by using ENV 'PI_URL_PREFIXED'
        retry: optional, default = 3,
            number of times to retry downloading the image
        timeout: optional, default = 30,
            timeout value for each download attempt in seconds
    returns:
        image (np.ndarray)
    """
    cache_dir = global_image_cache if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    assert decode_type in [cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE]

    if use_cache:
        cache_path = os.path.join(cache_dir, image_id)
        if os.path.exists(cache_path):
            image = cv2.imread(cache_path, decode_type)
            if image is None:
                os.remove(cache_path)
            else:
                return image
    
    url_prefixed = global_url_prefixed if url_prefixed is None else url_prefixed
    url = url_prefixed + '/' + image_id

    for _ in range(retry):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                image_data = response.content

                # decode
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, decode_type)
                
                if image is not None: 
                    if use_cache:
                        with open(cache_path, "wb") as f:
                            f.write(image_data)
                    return image
                else:
                    logger.warning("Failed to decode the downloaded image data")
            else:
                logger.warning(f"Failed to download image, status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"An exception occurred while downloading the image: {e}")
            time.sleep(1)

    logger.error(f'Failed to download image from {image_id}, max retry {retry}')
    return None


def upload_image(image, sdk_config, retry=3):
    """ upload image to mms
    args:
        image (str | bytes): 
            str: local image path
            bytes: image buffer
        sdk_config (dict): required format = {'cid': str, 'env': str, 'biz': int, 'secret': str}
    returns:
        image_id (str)
    """
    assert isinstance(image, (str, bytes)), "image must be a str or bytes object"

    env.set_cid(sdk_config['cid'])
    env.set_environment(sdk_config['env'])
    biz = sdk_config['biz']
    secret = sdk_config['secret']
    
    for _ in range(retry):
        init_default_config_mgr(biz, secret, skip_scc_init=True)
        sdk = new_upload_sdk_with_secret(biz, secret, public_network=True)
        
        try:
            if isinstance(image, str):
                req = FullUploadFileRequest(_header=Header(), media_path=[image])
                resp = sdk.fullUploadFile(req)
            else:
                req = FullUploadRequest(_header=Header(), media_bytes=[image])
                resp = sdk.fullUpload(req)
            return resp[0].img_id
        except Exception as e:
            logger.error(f"An exception occurred while uploading the image: {e}")
            time.sleep(1)

    logger.error(f'Failed to upload the image, max retry {retry}')
    return None