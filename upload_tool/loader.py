######################################################################
#
# Copyright (c) 2022 Shopee Inc. All Rights Reserved.
#
######################################################################

"""
file: loader.py
author: min.yang@shopee.com
date: 2024-04-17 18:29:00
brief: reference from https://git.garena.com/shopee/MLP/aip/inference/mpi/matching/batch-swin-image-embed/-/blob/master/loader.py
"""
import threading
from io import BytesIO
from typing import List
from concurrent.futures import ThreadPoolExecutor

import cv2
import requests
import numpy as np

from PIL import Image
from PIL import ImageFile

from requests.exceptions import Timeout
from requests.exceptions import ConnectionError

from aipinfer import logger
from urllib.parse import urlparse, urlencode

from typing import Callable
from functools import wraps

def retry(exceptions, *, tries: int = 3, delay: int = 3, backoff: int = 2) -> Callable:
    def wrapper_retry(func: Callable):
        @wraps(func)
        def func_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # pylint: disable=broad-except
                    logger.swarn(mdelay, "{}, Retrying in {} seconds...".format(exc, mdelay))
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return func_retry
    return wrapper_retry

# fix : image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

thread_local = threading.local()

def download(url: str, timeout=3):
    def _get_session():
        if not hasattr(thread_local, "session"):
            adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20,
                                                    max_retries=0, pool_block=True)
            thread_local.session = requests.Session()
            thread_local.session.mount("http://", adapter)
            thread_local.session.mount("https://", adapter)
        return thread_local.session

    @retry((Timeout, ConnectionError), backoff=1)
    def _dl(session, url, timeout=3):
        headers = {'User-Agent': 'Chrome/50.0.2661.102 Safari/537.36'}
        buffer = session.get(url, stream=True, headers=headers, timeout=timeout).content
        return buffer

    if not url.startswith("http") or not url.startswith("https") :
        logger.error(f"unknown supported type: {type(url)}")
        return None

    try:
        return _dl(_get_session(), url, timeout)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"download: url: {url}, error: {exc}")

    return None

class ImageLoader():
    def __init__(self, loader_name="cv2", use_rgb=False):
        assert loader_name in ["cv2", "pil"], "only cv2 and pil are supported"
        self.loader_name = loader_name
        self.use_rgb = use_rgb

    def __call__(self, buffer: bytes, rgba_flag=False):
        if self.loader_name == "cv2":
            data = np.array(bytearray(buffer), dtype=np.uint8)
            if rgba_flag:
                img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            else:
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if self.use_rgb:
                img = img[:, :, ::-1].copy()
        elif self.loader_name == "pil":
            path = BytesIO(buffer)
            img = Image.open(path)
            if len(img.size) == 2:
                img = img.convert("RGB")
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = np.array(img)
            if not self.use_rgb:
                img = img[:, :, ::-1].copy()
        return img


if __name__ == "__main__":
    urls = [
        "https://down-sg.img.susercontent.com/sg-11134228-7rd43-lu7o82kmr0pef2",
        "https://down-sg.img.susercontent.com/sg-11134228-7rd6h-lu7o82pwjaz6c8",
        "https://down-sg.img.susercontent.com/sg-11134228-7rd3m-lu7o82tsdmh614",
    ]
    
    loader = ImageLoader()

    for idx, url in enumerate(urls):
        print(url)
        buffer = loader(download(url))
        cv2.imwrite(str(idx) + ".png", buffer)
