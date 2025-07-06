import os

from skimage.measure import label, regionprops
import cv2
import numpy as np
import pandas as pd
from .cvcore import max_rect


def load_template_gallery(image_dir, root_file, target_l1, balck_list):
    df = pd.read_csv(root_file)
    image2l1 = dict(zip([url.split('/')[-1] for url in df['image'].tolist()], df['l1'].tolist()))
    gallery = []
    black_list = [line.strip() for line in open(balck_list).readlines()]

    for image, l1 in list(image2l1.items()):
        template = {}
        
        # black list
        if image in black_list: 
            print(f'skip {image}')
            continue
        
        if isinstance(l1, float): 
            # print(image)
            continue

        # load target template
        if target_l1 in l1:
            template['bg_image_prompt'] = os.path.join(image_dir, image)
            template['bg_image_prompt_mask'] = template['bg_image_prompt'].replace('.jpg', '_mask.png')

            if not os.path.exists(template['bg_image_prompt']):
                print(template['bg_image_prompt'], 'not exist')
                continue
            
            image = cv2.imread(template['bg_image_prompt'])
            mask = cv2.imread(template['bg_image_prompt_mask'])
            
            box = max_rect(mask)
            center = (box[0] + box[2] // 2, box[1] + box[3] // 2)

            mask = cropped_to_square_v2(mask, center, box[2:])
            if mask is None:
                print(template['bg_image_prompt'], 'mask is none')
                continue
            mask = del_small_mask(mask)

            image = cropped_to_square_v2(image, center, box[2:])
            template['bg_image'] = image
            template['bg_image_mask'] = mask

            # if np.sum(mask / 255. / 3) / (mask.shape[0] * mask.shape[1]) > 0.6:
            #     print('fg is large')
            #     continue
        else:
            continue
        gallery.append(template)

    print('gallery size:', len(gallery))
    return gallery


def find_indexes(lst, thr, k):
    # 找到大于thr的元素的索引
    indexes = [i for i, val in enumerate(lst) if val > thr]
    
    if len(indexes) > 0:
        if len(indexes) > k:
            # 对大于阈值的元素的索引按对应值进行排序
            sorted_indexes = sorted(indexes, key=lambda x: lst[x], reverse=True)
            # 只保留前k个索引
            indexes = sorted_indexes[:k]
    else:
        # 找到最大元素的索引
        max_index = max(range(len(lst)), key=lst.__getitem__)
        return [max_index]
    
    return indexes


def search_template_topk(gallery, target_mask, thr=0.75, k=3):
    if isinstance(target_mask, str):
        target_mask = cv2.imread(target_mask)
    box = max_rect(target_mask)
    target_mask_crop = target_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    target_mask_resized = resize_to_square(target_mask_crop, size=256)

    ious = []
    boxs = []

    for template in gallery:
        template_mask = template['bg_image_mask']
        box = max_rect(template_mask)
        template_mask_crop = template_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        template_mask_resized = resize_to_square(template_mask_crop, size=256)

        iou = calculate_iou(target_mask_resized, template_mask_resized)
        ious.append(iou)
        boxs.append(box)

    target_idx = find_indexes(ious, thr, k)
    best_template = [gallery[i] for i in target_idx]
    best_template_box = [boxs[i] for i in target_idx]

    print([ious[i] for i in target_idx])
    return best_template, best_template_box


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


def cropped_to_square(image, center):
    h, w = image.shape[:2]
    x, y = center
    size = min(h, w)

    start_h = y - size // 2
    end_h = start_h + size
    start_w = x - size // 2
    end_w = start_w + size

    if start_h < 0:
        start_h = 0
        end_h = size
    if end_h > h:
        start_h = h - size
        end_h = h
    if start_w < 0:
        start_w = 0
        end_w = size
    if end_w > w:
        start_w = w - size
        end_w = w

    cropped = image[start_h:end_h, start_w:end_w]
    return cropped


def cropped_to_square_v2(image, center, box_size, max_box_ratio=0.9, min_box_ratio=0.7):
    h, w = image.shape[:2]
    x, y = center
    size = min(h, w)

    if size < max(box_size[0], box_size[1]) / max_box_ratio:
        return None

    size = min(size, int(max(box_size[0], box_size[1])/min_box_ratio))

    start_h = y - size // 2
    end_h = start_h + size
    start_w = x - size // 2
    end_w = start_w + size

    if start_h < 0:
        start_h = 0
        end_h = size
    if end_h > h:
        start_h = h - size
        end_h = h
    if start_w < 0:
        start_w = 0
        end_w = size
    if end_w > w:
        start_w = w - size
        end_w = w

    cropped = image[start_h:end_h, start_w:end_w]
    return cropped


def del_small_mask(mask, area_thr=0.01):
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    _, binary = cv2.threshold(opening, 10, 255, cv2.THRESH_BINARY)

    label_image = label(binary)
    regions = regionprops(label_image)

    new_mask = np.zeros_like(mask)
    area_thr = area_thr * new_mask.shape[0] * new_mask.shape[1]
    for region in regions:
        if region.area > area_thr:
            new_mask[label_image == region.label] = 255

    new_mask = mask * (new_mask / 255)
    return new_mask.astype(np.uint8)