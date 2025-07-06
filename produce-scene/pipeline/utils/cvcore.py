import cv2
import numpy as np
from skimage.measure import label, regionprops


def pad_and_erode(crop_box, iter=1):
    pad_box = np.zeros((crop_box.shape[0]+2, crop_box.shape[1]+2, 3), crop_box.dtype)
    pad_box[1:(crop_box.shape[0]+1), 1:(crop_box.shape[1]+1)] = crop_box
    pad_box = cv2.erode(pad_box, np.ones((3, 3), np.uint8), iterations=iter)
    return pad_box[1:(crop_box.shape[0]+1), 1:(crop_box.shape[1]+1)]


def max_rect(mask):
    if len(mask.shape) != 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box = None
    max_area = -1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            x, y, w, h = cv2.boundingRect(contour)
            box = [x, y, w, h]
            max_area = area
    return box


def main_rect(mask):
    if len(mask.shape) != 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    x, y, w, h = cv2.boundingRect(contours[0])
    x1, y1, x2, y2 = x, y, x+w, y+h

    for contour in contours[1:]:
        temp_x, temp_y, temp_w, temp_h = cv2.boundingRect(contour)
        x1 = min(x1, temp_x)
        y1 = min(y1, temp_y)
        x2 = max(x2, temp_x + temp_w)
        y2 = max(y2, temp_y + temp_h)
    return (x1, y1, x2-x1, y2-y1)


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