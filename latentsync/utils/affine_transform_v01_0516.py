# Adapted from https://github.com/guanjz20/StyleSync/blob/main/utils.py

import numpy as np
import cv2
import torch
import torch.nn.functional as F  # Add this line
from einops import rearrange
import torchvision.transforms as transforms

def transformation_from_points(points1, points0, smooth=True, p_bias=None):
    points2 = np.array(points0)
    points2 = points2.astype(np.float64)
    points1 = points1.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(np.matmul(points1.T, points2))
    R = (np.matmul(U, Vt)).T
    sR = (s2 / s1) * R
    T = c2.reshape(2, 1) - (s2 / s1) * np.matmul(R, c1.reshape(2, 1))
    M = np.concatenate((sR, T), axis=1)
    if smooth:
        bias = points2[2] - points1[2]
        if p_bias is None:
            p_bias = bias
        else:
            bias = p_bias * 0.2 + bias * 0.8
        p_bias = bias
        M[:, 2] = M[:, 2] + bias
    return M, p_bias

# 0611

def invert_affine_transform_torch_simple(affine_matrix):
    """简化版仿射变换矩阵求逆"""
    if isinstance(affine_matrix, np.ndarray):
        affine_matrix = torch.from_numpy(affine_matrix).float()
    
    bottom_row = torch.tensor([[0., 0., 1.]], device=affine_matrix.device, dtype=affine_matrix.dtype)
    affine_3x3 = torch.cat([affine_matrix, bottom_row], dim=0)
    inv_affine_3x3 = torch.inverse(affine_3x3)
    return inv_affine_3x3[:2, :]

def erode_torch_simple(img, kernel_size):
    """简化版torch腐蚀操作"""
    original_shape = img.shape
    
    if len(original_shape) == 2:
        img_tensor = img.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif len(original_shape) == 3:
        if original_shape[2] == 3:  # HWC格式，3通道
            img_tensor = img.permute(2, 0, 1).unsqueeze(0)  # [H, W, 3] -> [1, 3, H, W]
        else:  # HWC格式，1通道
            # 修复Bug：需要先squeeze最后一维，再添加正确的维度
            img_tensor = img.squeeze(-1).unsqueeze(0).unsqueeze(0)  # [H, W, 1] -> [H, W] -> [1, 1, H, W]
    else:
        img_tensor = img
    
    pool_size = max(1, kernel_size)
    # 使用精确的padding确保输出尺寸不变
    padding = pool_size // 2
    eroded = -F.max_pool2d(-img_tensor, pool_size, stride=1, padding=padding)
    
    # 如果尺寸不匹配，强制调整到原始尺寸
    target_h = img_tensor.shape[2]
    target_w = img_tensor.shape[3]
    if eroded.shape[2] != target_h or eroded.shape[3] != target_w:
        eroded = F.interpolate(eroded, size=(target_h, target_w), mode='bilinear', align_corners=True)
    
    # 恢复到原始格式
    if len(original_shape) == 2:
        eroded = eroded.squeeze(0).squeeze(0)  # [1, 1, H, W] -> [H, W]
    elif len(original_shape) == 3:
        if original_shape[2] == 3:  # 原始是HWC，3通道
            eroded = eroded.squeeze(0).permute(1, 2, 0)  # [1, 3, H, W] -> [H, W, 3]
        else:  # 原始是HWC，1通道
            eroded = eroded.squeeze(0).squeeze(0).unsqueeze(-1)  # [1, 1, H, W] -> [H, W, 1]
    
    return eroded

def gaussian_blur_torch_simple(img, kernel_size, sigma=0.0):
    """简化版torch高斯模糊"""
    if sigma == 0.0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    original_shape = img.shape
    is_hwc_format = len(original_shape) == 3 and original_shape[2] in [1, 3, 4]
    
    if len(img.shape) == 2:
        img_tensor = img.unsqueeze(0).unsqueeze(0)
    elif is_hwc_format:
        img_tensor = img.permute(2, 0, 1).unsqueeze(0)
    elif len(img.shape) == 3:
        img_tensor = img.unsqueeze(0)
    else:
        img_tensor = img
    
    x = torch.arange(kernel_size, device=img.device, dtype=img.dtype) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2) / torch.exp(-0.5 * (x / sigma) ** 2).sum()
    
    pad_size = kernel_size // 2
    kernel_1d_h = kernel_1d.view(1, 1, 1, kernel_size)
    img_tensor = F.pad(img_tensor, (pad_size, pad_size, 0, 0), mode='reflect')
    blurred = F.conv2d(img_tensor, kernel_1d_h, groups=img_tensor.shape[1])
    
    kernel_1d_v = kernel_1d.view(1, 1, kernel_size, 1)
    blurred = F.pad(blurred, (0, 0, pad_size, pad_size), mode='reflect')
    blurred = F.conv2d(blurred, kernel_1d_v, groups=blurred.shape[1])
    
    if len(original_shape) == 2:
        blurred = blurred.squeeze(0).squeeze(0)
    elif is_hwc_format:
        blurred = blurred.squeeze(0).permute(1, 2, 0)
    else:
        blurred = blurred.squeeze(0)
    
    return blurred


def warp_affine_torch_simple(src, M, dsize, is_mask=False, fill_value=127.0):
    """简化版torch仿射变换"""
    if src.dtype != torch.float32:
        src = src.to(dtype=torch.float32)
    
    if len(src.shape) == 3:
        src = src.unsqueeze(0)
    
    out_h, out_w = dsize[1], dsize[0]
    y, x = torch.meshgrid(torch.arange(out_h, device=src.device, dtype=torch.float32), 
                          torch.arange(out_w, device=src.device, dtype=torch.float32), indexing='ij')
    homogeneous_coords = torch.stack([x.flatten(), y.flatten(), torch.ones_like(x.flatten())], dim=0)
    
    if isinstance(M, np.ndarray):
        M_tensor = torch.from_numpy(M).to(device=src.device, dtype=torch.float32)
    else:
        M_tensor = M.to(device=src.device, dtype=torch.float32)
    
    M_3x3 = torch.cat([M_tensor, torch.tensor([[0., 0., 1.]], device=src.device)], dim=0)
    M_inv = torch.inverse(M_3x3)[:2, :]
    src_coords = M_inv @ homogeneous_coords
    
    src_x = src_coords[0].reshape(out_h, out_w)
    src_y = src_coords[1].reshape(out_h, out_w)
    src_h, src_w = src.shape[2:]
    
    normalized_x = 2.0 * src_x / (src_w - 1) - 1.0
    normalized_y = 2.0 * src_y / (src_h - 1) - 1.0
    grid = torch.stack([normalized_x, normalized_y], dim=-1).unsqueeze(0).to(dtype=src.dtype)
    
    padding_mode = 'zeros' if is_mask else 'zeros'
    output = F.grid_sample(src, grid, mode='bilinear', padding_mode=padding_mode, align_corners=True)
    
    if not is_mask:
        out_of_bounds_mask = (normalized_x < -1) | (normalized_x > 1) | (normalized_y < -1) | (normalized_y > 1)
        out_of_bounds_mask = out_of_bounds_mask.unsqueeze(0).unsqueeze(0)
        fill_tensor = torch.full((1, output.shape[1], output.shape[2], output.shape[3]), 
                                fill_value, device=output.device, dtype=output.dtype)
        output = torch.where(out_of_bounds_mask, fill_tensor, output)
    
    # 确保输出尺寸精确匹配
    if output.shape[2] != out_h or output.shape[3] != out_w:
        output = F.interpolate(output, size=(out_h, out_w), mode='bilinear', align_corners=True)
    
    return output.squeeze(0)


class AlignRestore(object):
    def __init__(self, align_points=3, device="cuda"):
        self.device = torch.device(device)

        if align_points == 3:
            self.upscale_factor = 1
            ratio = 2.8
            self.crop_ratio = (ratio, ratio)
            self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
            self.face_template = self.face_template * ratio
            self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
            self.p_bias = None
            self.mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)
            self.mask_t = torch.ones((1,self.face_size[1], self.face_size[0]), device=self.device)


    def process(self, img, lmk_align=None, smooth=True, align_points=3):
        aligned_face, affine_matrix = self.align_warp_face(img, lmk_align, smooth)
        restored_img = self.restore_img(img, aligned_face, affine_matrix)
        cv2.imwrite("restored.jpg", restored_img)
        cv2.imwrite("aligned.jpg", aligned_face)
        return aligned_face, restored_img

    def align_warp_face(self, img, lmks3, smooth=True, border_mode="constant"):
        affine_matrix, self.p_bias = transformation_from_points(lmks3, self.face_template, smooth, self.p_bias)
        if border_mode == "constant":
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == "reflect101":
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == "reflect":
            border_mode = cv2.BORDER_REFLECT

        cropped_face = cv2.warpAffine(
            img,
            affine_matrix,
            self.face_size,
            flags=cv2.INTER_LANCZOS4,
            borderMode=border_mode,
            borderValue=[127, 127, 127],
        )
        return cropped_face, affine_matrix


    def restore_img(self, input_img, face, affine_matrix):
        h, w, _ = input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        upsample_img = cv2.resize(input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        inverse_affine *= self.upscale_factor
        if self.upscale_factor > 1:
            extra_offset = 0.5 * self.upscale_factor
        else:
            extra_offset = 0
        inverse_affine[:, 2] += extra_offset
        inv_restored = cv2.warpAffine(face, inverse_affine, (w_up, h_up), flags=cv2.INTER_LANCZOS4)
        mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)
        inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
        inv_mask_erosion = cv2.erode(
            inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8)
        )
        pasted_face = inv_mask_erosion[:, :, None] * inv_restored
        total_face_area = np.sum(inv_mask_erosion)
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = w_edge * 2
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
        inv_soft_mask = inv_soft_mask[:, :, None]
        upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img
        if np.max(upsample_img) > 256:
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)
        return upsample_img

    # @profile
    def restore_img_torch(self, input_img, face, affine_matrix):
        h, w, _ = input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        upsample_img = cv2.resize(input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        inverse_affine *= self.upscale_factor

        extra_offset = 0.5 * self.upscale_factor if self.upscale_factor > 1 else 0
        inverse_affine[:, 2] += extra_offset

        inv_restored = cv2.warpAffine(face, inverse_affine, (w_up, h_up), flags=cv2.INTER_LANCZOS4)
        # mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)
        inv_mask = cv2.warpAffine(self.mask, inverse_affine, (w_up, h_up)) # cv2.imwrite('/home/work/houyi/pj_25_q1/mmuplt-avatarv3/hy_test/inv_mask.png', (inv_mask * 255).astype(np.uint8))

        # valid_points = self.get_transformed_mask_coordinates(inverse_affine, w_up, h_up, debug=True)

        inv_mask_erosion = cv2.erode(inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8))

        inv_mask_erosion_t = torch.from_numpy(inv_mask_erosion).to(self.device).unsqueeze(-1)
        inv_restored_t = torch.from_numpy(inv_restored).to(self.device)
        pasted_face_t = inv_mask_erosion_t * inv_restored_t

        total_face_area = torch.sum(inv_mask_erosion_t).item()
        w_edge = int(total_face_area**0.5) // 20
        w_edge = max(1, w_edge)  # 确保w_edge至少为1
        erosion_radius = w_edge * 2
        
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = w_edge * 2
        blur_size = max(1, blur_size)  # 确保blur_size至少为1
        if blur_size % 2 == 0:  # 确保blur_size为奇数
            blur_size += 1
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size, blur_size), 0)
        inv_soft_mask = inv_soft_mask[:, :, None]
        
        inv_soft_mask_t = torch.from_numpy(inv_soft_mask).to(self.device)
        upsample_img_t = torch.from_numpy(upsample_img).to(self.device)
        
        result_t = inv_soft_mask_t * pasted_face_t + (1 - inv_soft_mask_t) * upsample_img_t
        result = result_t.cpu().numpy()
        
        if np.max(result) > 256:
            result = result.astype(np.uint16)
        else:
            result = result.astype(np.uint8)
        return result

    # @profile
    def restore_img_torch_simple(self, input_img, face, affine_matrix):
        """
        简化版完全使用torch tensor操作实现，代码简洁
        """
        # print("restore_img_torch_simple version")
        face_device = face.device
        h, w, _ = input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        
        # 上采样输入图像，保持CHW格式
        input_img_tensor = torch.from_numpy(input_img).to(device=face_device, dtype=torch.float32)
        input_img_chw = rearrange(input_img_tensor, "h w c -> c h w")
        resize_transform = transforms.Resize((h_up, w_up), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        input_img_processed = resize_transform(input_img_chw)

        try:
            # 仿射变换矩阵求逆
            affine_matrix_tensor = torch.from_numpy(affine_matrix).to(device=face_device, dtype=torch.float32)
            inverse_affine_tensor = invert_affine_transform_torch_simple(affine_matrix_tensor)
            inverse_affine_tensor *= self.upscale_factor
            extra_offset = 0.5 * self.upscale_factor if self.upscale_factor > 1 else 0
            inverse_affine_tensor[:, 2] += extra_offset
            
            # 变换face和mask
            inv_face = warp_affine_torch_simple(face, inverse_affine_tensor, (w_up, h_up), is_mask=False, fill_value=127.0)
            inv_face = (inv_face / 2 + 0.5).clamp(0, 1) * 255
            
            mask_on_device = self.mask_t.to(dtype=torch.float32, device=face_device)
            inv_mask = warp_affine_torch_simple(mask_on_device, inverse_affine_tensor, (w_up, h_up), is_mask=True)
            inv_mask_2d = inv_mask.squeeze()

            # # 第一次腐蚀
            # inv_mask_2d = inv_mask
            inv_mask_erosion_2d = erode_torch_simple(inv_mask_2d, int(2 * self.upscale_factor))

            # 计算pasted_face
            inv_mask_erosion_3d = inv_mask_erosion_2d.unsqueeze(0).expand_as(inv_face)
            pasted_face = inv_mask_erosion_3d * inv_face
            
            # 计算face区域参数
            total_face_area = torch.sum(inv_mask_erosion_2d.float())
            w_edge = int(total_face_area**0.5) // 20
            erosion_radius = w_edge * 2

            # 第二次腐蚀和高斯模糊
            inv_mask_erosion_2d_for_second = inv_mask_erosion_2d
            inv_mask_center_tensor = erode_torch_simple(inv_mask_erosion_2d_for_second, erosion_radius)
            
            blur_size = w_edge * 2 + 1
            sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8
            inv_soft_mask_2d = gaussian_blur_torch_simple(inv_mask_center_tensor, blur_size, sigma)
            
            # 最终融合
            inv_soft_mask_3d = inv_soft_mask_2d.unsqueeze(0).expand_as(inv_face)
            img_back = inv_soft_mask_3d * pasted_face + (1 - inv_soft_mask_3d) * input_img_processed

            # 转换回numpy格式
            img_back = rearrange(img_back, "c h w -> h w c").contiguous().to(dtype=torch.uint8)
            img_back = torch.clamp(img_back, 0, 255).to(dtype=torch.uint8)
            return img_back.cpu().numpy()
            
        except Exception as e:
            # 回退到原实现 - 确保face数据格式正确
            if isinstance(face, torch.Tensor):
                face_np = (face.permute(1, 2, 0).cpu().numpy() / 2 + 0.5) * 255
                face_np = np.clip(face_np, 0, 255).astype(np.uint8)
            else:
                face_np = face
            return self.restore_img_torch(input_img, face_np, affine_matrix)

class laplacianSmooth:
    def __init__(self, smoothAlpha=0.3):
        self.smoothAlpha = smoothAlpha
        self.pts_last = None

    def smooth(self, pts_cur):
        if self.pts_last is None:
            self.pts_last = pts_cur.copy()
            return pts_cur.copy()
        x1 = min(pts_cur[:, 0])
        x2 = max(pts_cur[:, 0])
        y1 = min(pts_cur[:, 1])
        y2 = max(pts_cur[:, 1])
        width = x2 - x1
        pts_update = []
        for i in range(len(pts_cur)):
            x_new, y_new = pts_cur[i]
            x_old, y_old = self.pts_last[i]
            tmp = (x_new - x_old) ** 2 + (y_new - y_old) ** 2
            w = np.exp(-tmp / (width * self.smoothAlpha))
            x = x_old * w + x_new * (1 - w)
            y = y_old * w + y_new * (1 - w)
            pts_update.append([x, y])
        pts_update = np.array(pts_update)
        self.pts_last = pts_update.copy()

        return pts_update
