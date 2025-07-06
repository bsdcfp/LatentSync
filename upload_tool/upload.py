######################################################################
#
# Copyright (c) 2024 Shopee Inc. All Rights Reserved.
#
######################################################################

"""
file: upload.py
author: min.yang@shopee.com
date: 2024-04-16 17:45:00
brief: from wenjun.jie@shopee.com desc
"""

from core.env import env
from imageSdk.protocol import Header, FullUploadRequest, FullUploadFileRequest
from imageSdk.sdk import new_upload_sdk_with_secret

env.set_cid("br")
env.set_environment("live")

biz = 4253
secret = '4963a23c02487519007b3f47f8440a3b'

sdk = new_upload_sdk_with_secret(biz, secret, public_network=True)

def upload_single_image_to_mms(f_path):
    req = FullUploadFileRequest(_header=Header(), media_path=[f_path])
    resp = sdk.fullUploadFile(req)
    data = resp[0]
    return data.img_id, data.url

def upload_single_bytes_to_mms(bytes):
    try:
        req = FullUploadRequest(_header=Header(), media_bytes=[bytes])
        resp = sdk.fullUpload(req)
        data = resp[0]
        return data.img_id, data.url
    except:
        try:
            req = FullUploadRequest(_header=Header(), media_bytes=[bytes])
            resp = sdk.fullUpload(req)
            data = resp[0]
            return data.img_id, data.url
        except:
            return "upload cdn error", None


def upload_image_to_mms(img_list):
    resp_list = []
    for f_path in tqdm(img_list):
        try:
            req = FullUploadFileRequest(_header=Header(), media_path=[f_path])
            resp = sdk.fullUploadFile(req)
            resp_list.append(resp[0])
        except:
            try:
                req = FullUploadFileRequest(_header=Header(), media_path=[f_path])
                resp = sdk.fullUploadFile(req)
                resp_list.append(resp[0])
            except:
                print(f_path)
                resp_list.append()
                continue
    return resp_list


if __name__ == '__main__':
    
    # id, url = upload_single_image_to_mms("test/dataset/product/430demo.png")
    # print(id, url)
    # id, url = upload_single_image_to_mms("test/dataset/mask/430demo.png")
    # print(id, url)
    # id, url = upload_single_image_to_mms("test/dataset/background/430demo.png")
    # print(id, url)
    # id, url = upload_single_image_to_mms("test/dataset/product/430demo_rgba.png")
    # print(id, url)

    # image_path = "test/dataset/human/02116.jpg"
    # id, url = upload_single_image_to_mms(image_path)
    # print(image_path, url)
    # image_path = "test/dataset/human/00484.jpg"
    # id, url = upload_single_image_to_mms(image_path)
    # print(image_path, url)
    
    # image_path = "test/dataset/cloth/02116.jpg"
    # id, url = upload_single_image_to_mms(image_path)
    # print(image_path, url)
    # image_path = "test/dataset/cloth/00484.jpg"
    # id, url = upload_single_image_to_mms(image_path)
    # print(image_path, url)
    
    # image_path = "test/dataset/background/02116.png"
    # id, url = upload_single_image_to_mms(image_path)
    # print(image_path, url)
    # image_path = "test/dataset/background/00484.png"
    # id, url = upload_single_image_to_mms(image_path)
    # print(image_path, url)
    
    # image_path = "test/dataset/skeleton/02116.png"
    # id, url = upload_single_image_to_mms(image_path)
    # print(image_path, url)
    # image_path = "test/dataset/skeleton/00484.png"
    # id, url = upload_single_image_to_mms(image_path)
    # print(image_path, url)
    
    # image_path = "test/dataset/mask/02116.png"
    # id, url = upload_single_image_to_mms(image_path)
    # print(image_path, url)
    # image_path = "test/dataset/mask/00484.png"
    # id, url = upload_single_image_to_mms(image_path)
    # print(image_path, url)

    image_path = "examples/test_data/test_foreground.png"
    id, url = upload_single_image_to_mms(image_path)
    print(image_path, url)

    image_path = "examples/test_data/test_template_image.png"
    id, url = upload_single_image_to_mms(image_path)
    print(image_path, url)
    
#     import pandas as pd
#     import os
#     from tqdm import tqdm
#     import glob
    
#     # img_path = "/data/kailin.chen/sd-webui-inpaint-anything-main/images/test_ldm.png"
#     # img_id, img_url = upload_image_to_mms(img_path)
#     # print(img_id, img_url)
    
#     input_dir = "/data/kailin.chen/dataset/ImageOptimizer/outputs/Experiment_Batch1_unclean_BG_150k/iro_cover_image_non_clean_bg_20230823_pinctada_fba_rest_centralised_split/"
#     output_dir = "/data/kailin.chen/dataset/ImageOptimizer/Experiment_Batch1_unclean_BG_150k/"
#     out_file_name = "iro_cover_image_non_clean_bg_20230823_pinctada_fba_rest_centralised_split_url.csv"
    
#     if not os.path.isdir(output_dir): 
#         os.mkdir(output_dir)
    
#     IMG_LIST1 = glob.glob(input_dir + '/*.jpg')
#     IMG_LIST2 = glob.glob(input_dir + '/*.jpeg')
#     IMG_LIST3 = glob.glob(input_dir + '/*.png')
#     IMG_LIST4 = glob.glob(input_dir + '/*.jfif')
#     IMG_LIST5 = glob.glob(input_dir + '/*.JPG')
#     IMG_LIST6 = glob.glob(input_dir + '/*.PNG')
#     IMG_LIST = IMG_LIST1 + IMG_LIST2 + IMG_LIST3 + IMG_LIST4 + IMG_LIST5 + IMG_LIST6
    
#     REG = upload_image_to_mms(IMG_LIST)
    
#     IMG_HASH_DICT = {}
#     ERROR_LIST = []
#     for i, img_file in tqdm(enumerate(IMG_LIST)):
#         filename = os.path.basename(img_file)
#         try:
#             IMG_HASH_DICT[filename] = REG[i].img_id
#         except:
#             print(filename)
#             ERROR_LIST.append(filename)
        
#     LIST_TO_PD = []
    
#     for key in tqdm(IMG_HASH_DICT):
#         try:
#             LIST_TO_PD.append([key, """=IMAGE("http://img.sp.mms.shopee.sg/""" + IMG_HASH_DICT[key] + """",1)""", "http://img.sp.mms.shopee.sg/" + IMG_HASH_DICT[key]])
#         except Exception:
#             print(key, IMG_HASH_DICT[key])
#             ERROR_LIST.append(key)
    
#     print("len(ERROR_LIST):", len(ERROR_LIST))

#     retry = 0
#     while len(ERROR_LIST) != 0 and retry < 20:
#         retry += 1
#         IMG_LIST = [input_dir + name for name in ERROR_LIST]
#         REG = upload_image_to_mms(IMG_LIST)
#         IMG_HASH_DICT = {}
#         for i, img_file in tqdm(enumerate(IMG_LIST)):
#             filename = os.path.basename(img_file)
#             IMG_HASH_DICT[filename] = REG[i].img_id
#         ERROR_LIST = []
#         for key in tqdm(IMG_HASH_DICT):
#             try:
#                 LIST_TO_PD.append([key,
#                                 """=IMAGE("http://img.sp.mms.shopee.sg/""" + IMG_HASH_DICT[key] +
#                                 """",1)"""], "http://img.sp.mms.shopee.sg/" + IMG_HASH_DICT[key])
#             except Exception:
#                 print(key, IMG_HASH_DICT[key])
#                 ERROR_LIST.append(key)
                
#     DF = pd.DataFrame(LIST_TO_PD, columns=['Name', 'Image', 'url'])
#     DF.to_csv(os.path.join(output_dir, out_file_name), index=False)
#     print("len(ERROR_LIST):", len(ERROR_LIST))