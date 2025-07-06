import cv2
import numpy as np

from .cvcore import del_small_mask, max_rect, pad_and_erode, main_rect


def get_boxsize(h, w, max_h, min_h, max_w):
    # TODO: update fg size setting
    aspect_ratio = w / h
    if h > max_h or w > max_w:
        if aspect_ratio > max_w / max_h:
            w = max_w
            h = int(w / aspect_ratio)
        else:
            h = max_h
            w = int(h * aspect_ratio)
    elif h < min_h:
        target_w = int(min_h * aspect_ratio)
        if target_w > max_w:
            w = max_w
            h = int(w / aspect_ratio)
        else:
            w = target_w
            h = min_h
    return h, w


def pad_resize(fg_image, 
               mask_image,
               template_cfg,
               gen_h=768,
               gen_w=768,
               out_h=1080,
               out_w=1080,
    ):
    """
    args:
        fg_image: hxwx3, RGB
        mask_image: hxwx3
    """
    # step 1: crop foreground
    mask_image = del_small_mask(mask_image)
    box = max_rect(mask_image.copy())
    x, y, w, h = box
    
    crop_fg = fg_image[y:(y+h), x:(x+w)]
    crop_mask = mask_image[y:(y+h), x:(x+w)]

    # step 2: get target size and position of final output image (not sd generation)
    target_h, target_w = get_boxsize(h, w, 
                                     template_cfg['h'],
                                     template_cfg['h'],
                                     template_cfg['w'])
    crop_fg_final = cv2.resize(crop_fg.copy(), (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    crop_mask_final = cv2.resize(crop_mask.copy(), (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # position
    target_center = [out_w//2, out_h//2] # (x, y)
    if 'bottom' in template_cfg:
        target_center = (out_w//2, template_cfg['bottom']-(target_h//2))
        
    target_left = target_center[0] - target_w // 2
    target_up = target_center[1] - target_h // 2
    
    # step 3: resize fg and paste to black background for sd input
    ratio_gen2out = (gen_w / out_w)
    sd_w = int(target_w * ratio_gen2out)
    sd_h = int(target_h * ratio_gen2out)
    sd_left = int(target_left * ratio_gen2out)
    sd_up = int(target_up * ratio_gen2out)

    # erode input fg
    crop_mask_sd = pad_and_erode(crop_mask.copy(), iter=template_cfg['fg_erode_iter'])
    crop_mask_sd = cv2.resize(crop_mask_sd, (sd_w, sd_h), interpolation=cv2.INTER_CUBIC)
    
    crop_fg_sd = cv2.resize(crop_fg.copy(), (sd_w, sd_h), interpolation=cv2.INTER_CUBIC)
    crop_fg_sd = (crop_fg_sd * (crop_mask_sd / 255.)).astype(np.uint8)
    
    # paste fg condition to black (or canny) background
    pad_fg = np.zeros((gen_h, gen_w, 3), dtype=np.uint8)
    pad_mask = np.zeros((gen_h, gen_w, 3), dtype=np.uint8)

    pad_fg[sd_up:(sd_up+sd_h), sd_left:(sd_left+sd_w)] = crop_fg_sd
    pad_mask[sd_up:(sd_up+sd_h), sd_left:(sd_left+sd_w)] = crop_mask_sd
    
    return pad_fg, pad_mask, crop_fg_final, crop_mask_final, (target_left, target_up, target_w, target_h)


def pad_resize_v2(fg_image, 
                  mask_image,
                  template_cfg,
                  gen_h=768,
                  gen_w=768,
                  out_h=1080,
                  out_w=1080
    ):
    """ for DD
    args:
        fg_image: hxwx3, RGB
        mask_image: hxwx3
    """
    # step 1: crop foreground
    box = main_rect(mask_image.copy())
    x, y, w, h = box

    crop_fg = fg_image[y:(y+h), x:(x+w)]
    crop_mask = mask_image[y:(y+h), x:(x+w)]

    # step 2: get target size and position of final output image (not sd generation)
    target_h, target_w = get_boxsize(h, w, 
                                     template_cfg['h'],
                                     template_cfg['h'],
                                     template_cfg['w'])

    crop_fg_final = cv2.resize(crop_fg.copy(), (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    crop_mask_final = crop_mask.copy()
    crop_mask_final = cv2.GaussianBlur(crop_mask_final, (3, 3), 0)
    crop_mask_final = cv2.GaussianBlur(crop_mask_final, (3, 3), 0)
    crop_mask_final = cv2.resize(crop_mask_final, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # position
    target_center = [out_w//2, out_h//2] # (x, y)
    if 'center' in template_cfg:
        target_center = template_cfg['center']
    elif 'bottom' in template_cfg:
        target_center = (out_w//2, template_cfg['bottom']-(target_h//2))
        
    target_left = target_center[0] - target_w // 2
    target_up = target_center[1] - target_h // 2
    
    # step 3: resize fg and paste to black background for sd input
    ratio_gen2out = (gen_w / out_w)
    sd_w = int(target_w * ratio_gen2out)
    sd_h = int(target_h * ratio_gen2out)
    sd_left = int(target_left * ratio_gen2out)
    sd_up = int(target_up * ratio_gen2out)

    # erode input fg
    crop_mask_sd = pad_and_erode(crop_mask.copy(), iter=template_cfg['fg_erode_iter'])
    crop_mask_sd = cv2.resize(crop_mask_sd, (sd_w, sd_h), interpolation=cv2.INTER_CUBIC)
    
    crop_fg_sd = cv2.resize(crop_fg.copy(), (sd_w, sd_h), interpolation=cv2.INTER_CUBIC)
    crop_fg_sd = (crop_fg_sd * (crop_mask_sd / 255.)).astype(np.uint8)
    
    # paste fg condition to black (or canny) background
    pad_fg = np.zeros((gen_h, gen_w, 3), dtype=np.uint8)
    pad_mask = np.zeros((gen_h, gen_w, 3), dtype=np.uint8)

    pad_fg[sd_up:(sd_up+sd_h), sd_left:(sd_left+sd_w)] = crop_fg_sd
    pad_mask[sd_up:(sd_up+sd_h), sd_left:(sd_left+sd_w)] = crop_mask_sd
    
    return pad_fg, pad_mask, crop_fg_final, crop_mask_final, (target_left, target_up, target_w, target_h)


def get_cropbox(mask, target_h, target_w, center, crop_tags):
    fg_w, fg_h = main_rect(mask)[2:]
    fg_ratio = fg_h / fg_w
    target_ratio = target_h / target_w
    # print(fg_w, fg_h, fg_ratio)

    if 'top' in crop_tags and 'bottom' in crop_tags:
        # assert fg_ratio >= 1
        target_h = 1
        target_w = target_h / target_ratio
        center[1] = 0.5
    if 'left' in crop_tags and 'right' in crop_tags:
        # assert fg_ratio <= 1
        target_w = 1
        target_h = target_ratio * target_w
        center[0] = 0.5

    if 'left' in crop_tags and 'right' not in crop_tags:
        target_w = target_h / fg_ratio if fg_ratio > target_ratio else target_w
        center[0] = target_w / 2
    if 'right' in crop_tags and 'left' not in crop_tags:
        target_w = target_h / fg_ratio if fg_ratio > target_ratio else target_w
        center[0] = 1 - target_w / 2

    if 'top' in crop_tags and 'bottom' not in crop_tags:
        target_h = target_w * fg_ratio if fg_ratio < target_ratio else target_h
        center[1] = target_h / 2
    if 'bottom' in crop_tags and 'top' not in crop_tags:
        target_h = target_w * fg_ratio if fg_ratio < target_ratio else target_h
        center[1] = 1 - target_h / 2

    return target_h, target_w, center