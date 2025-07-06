
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_path)

parent_dir = os.path.join(parent_path, "..")
sys.path.append(parent_dir)

from pathlib import Path
from loguru import logger
# from mllm.dialoggen_demo import DialogGen
from hydit.config import get_args
from hydit.inference_controlnet2 import End2End

from torchvision import transforms as T
from PIL import Image, ImageDraw
import numpy as np
import random
import torch
import copy
from cuda import cudart
import time
import csv
import cv2
from io import BytesIO

from upload_tool.loader import download,ImageLoader
from upload_tool.upload import upload_single_bytes_to_mms

from cache_diffusion.cache_diffusion import cachify
from cache_diffusion.cache_diffusion.utils import (
    HUNYUAN_DIT_CONFIG,
)

norm_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

norm_transform2 = T.Compose(
        [
            T.ToTensor(),
        ]
    )

USING_CACHE = True
if USING_CACHE:
    out_dir = "cache"
    csv_file = "hunyuan_result_opt.csv"
else:
    out_dir = "torch"
    csv_file = "hunyuan_result_torch.csv"

def inferencer():
    args = get_args()
    models_root_path = Path(args.model_root)

    # Load models
    gen = End2End(args, models_root_path)

    # # Try to enhance prompt
    # if args.enhance:
    #     logger.info("Loading DialogGen model (for prompt enhancement)...")
    #     enhancer = DialogGen(str(models_root_path / "dialoggen"), args.load_4bit)
    #     logger.info("DialogGen model loaded.")
    # else:
    #     enhancer = None

    # # NOTE: load trt model.
    # model_path = args.model_root
    # device_path = torch.cuda.get_device_name(torch.cuda.current_device()).split()[-1]
    # trt_root_path = os.path.join(model_path, "tensorrt", device_path)

    # batch_size  = 1
    # use_cuda_graph = False
    # verbose = False

    # model_list = [
    #     gen.pipeline,
    # ]

    # for item in model_list:
    #     item.load_trt(trt_root_path,  batch_size, use_cuda_graph, verbose)

    # max_device_memory = 0
    # for item in model_list:
    #     max_device_memory = max(
    #         max_device_memory, item.calculate_max_device_memory()
    #     )

    # _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
    # for item in model_list:
    #     item.activate_engine(shared_device_memory,  batch_size)


    # gen.pipeline.unet.cpu()
    # # gen.pipeline.controlnet.cpu()
    # torch.cuda.empty_cache()
    # print(gen.pipeline.unet)
    # print(gen.pipeline.controlnet)

    if USING_CACHE:
        cachify.prepare(gen.pipeline, HUNYUAN_DIT_CONFIG, "unet")
        cachify.enable(gen.pipeline, "unet")

        cachify.prepare(gen.pipeline, HUNYUAN_DIT_CONFIG, "controlnet")
        cachify.enable(gen.pipeline, "controlnet")

    # print(gen.pipeline.unet)

    return args, gen, None

def mask_random_area(image):
    # 打开原始图片
    image = image.convert('RGBA')

    # 获取图片的宽度和高度
    width, height = image.size

    # 随机生成掩码区域的左上角坐标和大小
    x1 = random.randint(0, width - 100)  # 确保有足够空间生成区域
    y1 = random.randint(0, height - 100)
    x2 = x1 + random.randint(50, 150)
    y2 = y1 + random.randint(50, 150)

    # 创建一个新的透明图像作为掩码
    mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)

    # 在掩码图像上绘制一个矩形区域作为要掩码的区域，设置为完全不透明（alpha = 255）
    draw.rectangle([(x1, y1), (x2, y2)], fill=(0, 0, 0, 255))

    # 将原始图片和掩码进行合成，实现掩码效果
    masked_image = Image.alpha_composite(image, mask)
    masked_image = masked_image.convert('RGB')

    return masked_image


def inference(args, gen, prompt, image):
    tem = image
    resized_tem = tem.resize((768, 768), Image.ANTIALIAS)
    image = np.array(resized_tem)
    print(image.shape)

    text = args.prompt
    ori_image = image[:,:,:3]
    mask = image[:,:,3]

    image_ori = copy.deepcopy(ori_image)

    height, width = mask.shape
    for j in range(height):
        for k in range(width):
            if mask[j, k] != 255:
                ori_image[j, k, :] = 0

    
    condition = Image.fromarray(ori_image)
    mask_image = Image.fromarray(mask)
    image_condition = norm_transform(condition)
    mask_tensor = norm_transform2(mask_image)

    mask_tensor = mask_tensor.unsqueeze(0).cuda()
    

    image_condition = image_condition.unsqueeze(0).cuda()
    results = gen.predict(text,
                        height=height,
                        width=width,
                        image=image_condition,
                        mask=mask_tensor,
                        seed=args.seed,
                        enhanced_prompt=enhanced_prompt,
                        negative_prompt=args.negative,
                        infer_steps=args.infer_steps,
                        guidance_scale=4,
                        batch_size=args.batch_size,
                        src_size_cond=args.size_cond,
                        use_style_cond=args.use_style_cond,
                        )

    return results, mask, image_ori

if __name__ == "__main__":
    args, gen, enhancer = inferencer()

    if enhancer:
        logger.info("Prompt Enhancement...")
        success, enhanced_prompt = enhancer(args.prompt)
        if not success:
            logger.info("Sorry, the prompt is not compliant, refuse to draw.")
            exit()
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
    else:
        enhanced_prompt = None

    # Run inference
    logger.info("Generating images...")
    # height, width = args.image_size
    prefix = 'https://mms.img.susercontent.com/'

    ###############################################
    test_csv = "./examples/test_case/12L1_Benchmark_V1-TestResultsUpdate-V2.6.1_newprompt_v22.csv"
    images_root = "./examples/test_case"
    #download image
    image_loader = ImageLoader(loader_name="pil")
    with open(test_csv, newline='') as csvfile:
        spamreader = csv.DictReader(csvfile)
        rows = list(spamreader)
        idx = 0
        fieldnames = spamreader.fieldnames + ['opt_image_id']  # Add a new column for result URLs
        l = len(rows)
        for idx,row in enumerate(rows):
            print(f" ====> {idx/l:.2%}")
            url = "https://down-br.img.susercontent.com/"
            seg_image = url + row['seg_img_id']
            seg_buf = download(url=seg_image)
            path = BytesIO(seg_buf)
            img = Image.open(path)
            # fname = images_root + '/' + f"{row['seg_img_id']}" + ".png"
            # img.save(fname)
            # cv2.imwrite(fname,fbuf)

            prompt = row['hppromptv22']

            print("[INFO] prompt: ", prompt)
            results, mask, image_ori = inference(args, gen, prompt, img)
            height, width = mask.shape

            images = results['images']
            result_urls = []
            for index, pil_img in enumerate(images):
                pil_img_np = np.array(pil_img)
                for j in range(height):
                    for k in range(width):
                        if mask[j, k] != 0:
                            pil_img_np[j, k, :] = image_ori[j, k, :]
                pil_img = Image.fromarray(pil_img_np, mode='RGB')

                pil_img.save(os.path.join(args.results_dir, out_dir, f"{row['seg_img_id']}" + "_out.png"))

                _, generate_image = cv2.imencode(".png", pil_img_np)
                generate_byte_data = generate_image.tobytes()
                generate_id, generate_url = upload_single_bytes_to_mms(
                    generate_byte_data
                )

                result_urls.append(
                    (generate_id, generate_url)
                )
            rows[idx]["opt_image_id"] = url + result_urls[0][0]
            print("[INFO]", idx)
            if idx > 10:
                break

        with open(images_root + '/' + csv_file, 'w') as csvfile:
            spamwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            spamwriter.writeheader()
            spamwriter.writerows(rows)

    ###############################################



