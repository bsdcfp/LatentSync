import numpy as np
import cv2
import pywt


def get_gray_img_np(img_rgb_np):
    return cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2GRAY)


def get_sharpness(img_rgb_np):
    # A Fast Wavelet-Based Algorithm for Global and Local Image Sharpness Estimation
    img_gray_np = get_gray_img_np(img_rgb_np)
    gray = np.float32(img_gray_np)
    coeffs2 = pywt.wavedec2(data=gray, wavelet="bior4.4", level=3)
    LL3, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs2
    subband = [LH1, HL1, HH1, LH2, HL2, HH2, LH3, HL3, HH3, LL3]

    mse_subband = [(band**2).mean() for band in subband]
    energy_subband = [np.log10(1 + mse) for mse in mse_subband]
    e1 = 0.2 * (energy_subband[0] + energy_subband[1]) / 2 + 0.8 * energy_subband[2]
    e2 = 0.2 * (energy_subband[3] + energy_subband[4]) / 2 + 0.8 * energy_subband[5]
    e3 = 0.2 * (energy_subband[6] + energy_subband[7]) / 2 + 0.8 * energy_subband[8]
    e_sum = 4 * e1 + 2 * e2 + 1 * e3
    return e_sum / 100


def ndarr_image_sharpening(img_rgb_arr, ks=0, weight=1.5, sigma=2, threshold=0):
    """ sharp on luminance channel of HLS color space with an unsharp mask """
    img_hls = cv2.cvtColor(img_rgb_arr, cv2.COLOR_RGB2HLS)
    img_l = img_hls[:, :, 1]
    blur_l = cv2.GaussianBlur(img_l, (ks, ks), sigma)
    img_l_s_np = cv2.addWeighted(img_l, weight, blur_l, 1 - weight, 0)
    if threshold > 0:
        low_contrast_mask = np.absolute(img_l - blur_l) < threshold
        np.copyto(img_l_s_np, img_l, where=low_contrast_mask)

    img_hls[:, :, 1] = img_l_s_np
    img_s_np = cv2.cvtColor(img_hls, cv2.COLOR_HLS2RGB)
    return img_s_np


def ndarr_image_sharpening_lab(img_rgb_arr, ks=0, weight=1.5, sigma=2, threshold=0):
    """ sharp on luminance channel of LAB color space with an unsharp mask """
    img_lab = cv2.cvtColor(img_rgb_arr, cv2.COLOR_RGB2LAB)
    img_l = img_lab[:, :, 0]
    blur_l = cv2.GaussianBlur(img_l, (ks, ks), sigma)
    img_l_s_np = cv2.addWeighted(img_l, weight, blur_l, -(weight - 1), 0)
    if threshold > 0:
        low_contrast_mask = np.absolute(img_l - blur_l) < threshold
        np.copyto(img_l_s_np, img_l, where=low_contrast_mask)

    img_lab[:, :, 0] = img_l_s_np
    img_s_np = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return img_s_np

def ndarr_image_sharpening_hsv(img_rgb_arr, ks=0, weight=1.5, sigma=2, threshold=0):
    """ sharp on value channel of HSV color space with an unsharp mask """
    img_hsv = cv2.cvtColor(img_rgb_arr, cv2.COLOR_RGB2HSV)
    img_v = img_hsv[:, :, -1]
    blur_v = cv2.GaussianBlur(img_v, (ks, ks), sigma)
    img_v_s_np = cv2.addWeighted(img_v, weight, blur_v, -(weight - 1), 0)
    if threshold > 0:
        low_contrast_mask = np.absolute(img_v - blur_v) < threshold
        np.copyto(img_v_s_np, img_v, where=low_contrast_mask)

    img_hsv[:, :, -1] = img_v_s_np
    img_s_np = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return img_s_np


def img_sharpening_v2(img_rgb_arr, color_space="HSV"):
    """
    sharp on luminance channel of HLS color space with an unsharp mask
    use a dynamic kernel size according to image dimension,
    and a sharpening weight according to original image's sharpness
    """
    # inp_pil original image/ thumbnail
    height, width = img_rgb_arr.shape[:2]
    size = min(height, width)
    img_sharpness = round(get_sharpness(img_rgb_arr), 3)
    kernel_size = int(size * 11 / 800)  # thumbnail ks --> 5
    kernel_size = kernel_size + [1, 0][kernel_size % 2 != 0]
    s_weight = round(-8.82 * img_sharpness + 3.1, 3)
    s_weight = np.clip(s_weight, 1.1, 2.5)
    if color_space == "HLS":
        img_s_arr = ndarr_image_sharpening(
            img_rgb_arr, ks=kernel_size, weight=s_weight, sigma=0, threshold=5
        )
    elif color_space == "LAB":
        img_s_arr = ndarr_image_sharpening_lab(
            img_rgb_arr, ks=kernel_size, weight=s_weight, sigma=0, threshold=5
        )
    else:
        img_s_arr = ndarr_image_sharpening_hsv(
            img_rgb_arr, ks=kernel_size, weight=s_weight, sigma=0, threshold=5
        )
    # return arr2pil(img_s_arr)
    return img_s_arr

def img_sharpening_v2(img_rgb_arr, color_space="HSV"):
    """
    sharp on luminance channel of HLS color space with an unsharp mask
    use a dynamic kernel size according to image dimension,
    and a sharpening weight according to original image's sharpness
    """
    # inp_pil original image/ thumbnail
    height, width = img_rgb_arr.shape[:2]
    size = min(height, width)
    img_sharpness = round(get_sharpness(img_rgb_arr), 3)
    kernel_size = int(size * 11 / 800)  # thumbnail ks --> 5
    kernel_size = kernel_size + [1, 0][kernel_size % 2 != 0]
    s_weight = round(-8.82 * img_sharpness + 3.1, 3)
    s_weight = np.clip(s_weight, 1.1, 2.5)
    if color_space == "HLS":
        img_s_arr = ndarr_image_sharpening(
            img_rgb_arr, ks=kernel_size, weight=s_weight, sigma=0, threshold=5
        )
    elif color_space == "LAB":
        img_s_arr = ndarr_image_sharpening_lab(
            img_rgb_arr, ks=kernel_size, weight=s_weight, sigma=0, threshold=5
        )
    else:
        img_s_arr = ndarr_image_sharpening_hsv(
            img_rgb_arr, ks=kernel_size, weight=s_weight, sigma=0, threshold=5
        )
    # return arr2pil(img_s_arr)
    return img_s_arr
