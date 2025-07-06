import os
import pandas as pd
import cv2
import numpy as np
from .cvcore import max_rect
from .visulization import vis_image_list


def load_template_gallery(image_dir, root_file, target_l1):
    df = pd.read_csv(root_file)
    image2l1 = dict(zip(df['image'].tolist(), df['l1'].tolist()))
    gallery = []
    for image, l1 in list(image2l1.items()):
        template = {}
        if target_l1 in l1:
            template['bg_image_prompt'] = os.path.join(image_dir, image)
            template['bg_image_prompt_mask'] = template['bg_image_prompt'].replace('.jpg', '_mask.png')
            mask = cv2.imread(template['bg_image_prompt_mask'])
            box = max_rect(cv2.imread(template['bg_image_prompt_mask']))
            if np.sum(mask / 255. / 3) / (mask.shape[0] * mask.shape[1]) > 0.6:
                print(template['bg_image_prompt'])
                continue
            params = {}
            params['fg_erode_iter'] = 6
            params['w'] = 800
            params['h'] = 800
            params['bottom'] = 540 + params['h'] // 2
            template['template_cfg'] = params
        else:
            continue
        # print('load template:', template)
        gallery.append(template)
    print('gallery size:', len(gallery))
    return gallery

def search_template(gallery, target_mask):
    print(target_mask)
    target_mask = cv2.imread(target_mask)
    box = max_rect(target_mask)
    target_mask_crop = target_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    target_mask_resized = resize_to_square(target_mask_crop, size=256)

    max_iou = 0
    best_template = None

    for template in gallery:
        template_mask = cv2.imread(template['bg_image_prompt_mask'])
        box = max_rect(template_mask)
        template_mask_crop = template_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        template_mask_resized = resize_to_square(template_mask_crop, size=256)

        iou = calculate_iou(target_mask_resized, template_mask_resized)

        if iou > max_iou:
            max_iou = iou
            best_template = template
    print(best_template, max_iou)
    return best_template

def resize_to_square(image, size):
    h, w = image.shape[:2]
    ratio = size / max(h, w)
    resized = cv2.resize(image, (int(w * ratio), int(h * ratio)))
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    pad_h = (size - resized.shape[0]) // 2
    pad_w = (size - resized.shape[1]) // 2
    padded[pad_h:pad_h + resized.shape[0], pad_w:pad_w + resized.shape[1], :] = resized
    return padded

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# gallery = load_template_gallery('/home/work/pinterest_images', 
#     '/home/work/llm/aigc_image/aigc_productbggen/data/pinterest/pinterest_filter_with_category.csv', 
#     'Beauty')
# search_template(gallery, 
#     '/home/work/llm/aigc_image/aigc_productbggen/data/0508_testdata/Beauty/0a309c7d5b27f8d27a37a3dddf9a4863_mask.png')