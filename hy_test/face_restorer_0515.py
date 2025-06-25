import cv2
import numpy as np
import line_profiler
import time
import os
import torch
import torch.nn.functional as F  # Add this line


# from line_profiler import LineProfiler

# class SecondLineProfiler(LineProfiler):  # 正确继承 LineProfiler
#     def print_stats(self, stream=None):
#         # 调用原始的统计方法
#         lstats = self.get_stats()
        
#         # 输出统计信息，但将时间单位改为秒
#         if stream is None:
#             import sys
#             stream = sys.stdout
            
#         stream.write('Timer unit: 1 s\n\n')

#         # 遍历每个函数的统计信息
#         for func, (code, timings) in lstats.timings.items():
#             filename = code.co_filename
#             lineno = code.co_firstlineno
#             func_name = code.co_name

#             stream.write(f'File: {filename}\n')
#             stream.write(f'Function: {func_name} at line {lineno}\n')

#             total_time = sum(t[2] for t in timings) / 1e9  # 转换为秒
#             stream.write(f'Total time: {total_time:.6f} s\n\n')

#             template = '%8s %8s %10s %10s %10s  %s\n'
#             stream.write(template % ('Line #', 'Hits', 'Time', 'Per Hit', '% Time', 'Line Contents'))
#             stream.write('=' * 80 + '\n')

#             # 提取代码行内容
#             lines = code.co_code.splitlines(keepends=False)
            
#             for timing in timings:
#                 line_no = timing[0]
#                 nhits = timing[1]
#                 time_ns = timing[2]
#                 time_s = time_ns / 1e9
#                 per_hit = time_s / nhits if nhits > 0 else 0
#                 percent = (time_s / total_time * 100) if total_time > 0 else 0
#                 line_content = lines[line_no - code.co_firstlineno] if line_no <= len(lines) else ''

#                 stream.write(template % (
#                     line_no,
#                     nhits,
#                     f'{time_s:.6f}',
#                     f'{per_hit:.6f}' if nhits > 0 else 'N/A',
#                     f'{percent:.1f}%',
#                     line_content
#                 ))
#             stream.write('\n')


# torch实现的腐蚀操作
def erode_torch(img, kernel_size):
    # input： img torch.Tensor
    # output: eroded_img torch.Tensor
    img_tensor = img
    # 确保输入是4D张量 [B, C, H, W]
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    elif len(img_tensor.shape) == 3:
        if img_tensor.shape[2] == 3:  # HWC格式
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # 转为NCHW
        else:
            img_tensor = img_tensor.unsqueeze(0)  # 已经是CHW格式，添加批次维度
    
    # 为2D最小池化创建核大小
    pool_size = kernel_size
    
    # 使用最大池化的负操作来实现腐蚀
    # 腐蚀 = -max_pool2d(-input)
    img_tensor_neg = -img_tensor
    # 确保padding不会改变输出尺寸
    eroded_neg = F.max_pool2d(img_tensor_neg, pool_size, stride=1, padding=pool_size//2)
    eroded = -eroded_neg
    
    # 还原原始维度
    if len(img.shape) == 2:
        eroded = eroded.squeeze(0).squeeze(0)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        eroded = eroded.squeeze(0).permute(1, 2, 0)
    else:
        # 确保返回与输入相同的形状
        if isinstance(img, np.ndarray):
            eroded = eroded.squeeze(0).squeeze(0)
    
    # # 如果输入是numpy数组，转回numpy
    # if isinstance(img, np.ndarray):
    #     eroded_np = eroded.cpu().numpy()
    #     # 确保输出的形状与输入相同
    #     if eroded_np.shape != img.shape:
    #         # 如果尺寸不同，裁剪或填充到原始尺寸
    #         if eroded_np.shape[0] > img.shape[0] or eroded_np.shape[1] > img.shape[1]:
    #             eroded_np = eroded_np[:img.shape[0], :img.shape[1]]
    #         elif eroded_np.shape[0] < img.shape[0] or eroded_np.shape[1] < img.shape[1]:
    #             # 创建与原始图像相同大小的零矩阵
    #             result = np.zeros_like(img)
    #             # 将eroded_np复制到result中
    #             result[:eroded_np.shape[0], :eroded_np.shape[1]] = eroded_np
    #             eroded_np = result
    #     return eroded_np
    
    # 确保输出的形状与输入相同
    if eroded.shape != img_tensor.shape and not isinstance(img, np.ndarray):
        # 裁剪或填充到原始尺寸
        if eroded.shape[-2] > img_tensor.shape[-2] or eroded.shape[-1] > img_tensor.shape[-1]:
            eroded = eroded[..., :img_tensor.shape[-2], :img_tensor.shape[-1]]
    
    return eroded

# 将cv2.GaussianBlur替换为torch版本
def gaussian_blur_torch(img, kernel_size, sigma=0):
    if sigma == 0:
        # 根据OpenCV的实现，如果sigma=0，则根据kernel_size计算
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    
    # 确保kernel_size是奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 创建1D高斯核
    x = torch.arange(kernel_size, device=img.device, dtype=torch.float32) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # 转换图像为torch tensor
    if isinstance(img, np.ndarray):
        img_tensor = torch.from_numpy(img).to(device="cuda")
    else:
        img_tensor = img
    
    # 添加批次和通道维度
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    elif len(img_tensor.shape) == 3 and img_tensor.shape[2] == 3:  # HWC格式
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # 转为NCHW
    
    # 使用可分离卷积实现高斯模糊
    # 水平方向
    pad_size = kernel_size // 2
    kernel_1d_h = kernel_1d.view(1, 1, 1, kernel_size)
    img_tensor = F.pad(img_tensor, (pad_size, pad_size, 0, 0), mode='reflect')
    blurred = F.conv2d(img_tensor, kernel_1d_h, groups=img_tensor.shape[1])
    
    # 垂直方向
    kernel_1d_v = kernel_1d.view(1, 1, kernel_size, 1)
    blurred = F.pad(blurred, (0, 0, pad_size, pad_size), mode='reflect')
    blurred = F.conv2d(blurred, kernel_1d_v, groups=blurred.shape[1])
    
    # 还原维度
    if len(img.shape) == 2:
        blurred = blurred.squeeze(0).squeeze(0)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        blurred = blurred.squeeze(0).permute(1, 2, 0)
    
    # 如果输入是numpy数组，转回numpy
    if isinstance(img, np.ndarray):
        return blurred.cpu().numpy()
    
    return blurred

    

class FaceRestorer:
    def __init__(self, upscale_factor=2, face_size=(256, 256)):
        self.upscale_factor = 1
        ratio = 2.8
        self.crop_ratio = (ratio, ratio)
        self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
        self.face_template = self.face_template * ratio
        self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
        self.p_bias = None

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



class FaceRestorer_torch:
    def __init__(self, upscale_factor=2, face_size=(256, 256)):
        self.upscale_factor = 1
        ratio = 2.8
        self.crop_ratio = (ratio, ratio)
        self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
        self.face_template = self.face_template * ratio
        self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
        self.p_bias = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)

    def restore_img(self, input_img, face, affine_matrix):
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
        erosion_radius = w_edge * 2
        
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = w_edge * 2
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
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


    def get_transformed_mask_coordinates(self, inverse_affine, w_up, h_up, debug=False):
        # 获取原始 mask 的非零坐标点
        y_coords, x_coords = np.nonzero(self.mask)
        points = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=1)
        
        # 使用仿射变换矩阵转换坐标点
        transformed_points = cv2.transform(points.reshape(-1, 1, 3), inverse_affine).reshape(-1, 2)
        
        # # 过滤掉超出图像边界的点
        # valid_points = transformed_points[
        #     (transformed_points[:, 0] >= 0) & 
        #     (transformed_points[:, 0] < w_up) & 
        #     (transformed_points[:, 1] >= 0) & 
        #     (transformed_points[:, 1] < h_up)
        # ]
        # 找到四个顶点
        # 左上角点 (x最小, y最小)
        top_left = transformed_points[np.argmin(transformed_points[:, 0] + transformed_points[:, 1])]
        # 右上角点 (x最大, y最小)
        top_right = transformed_points[np.argmin(transformed_points[:, 1] - transformed_points[:, 0])]
        # 左下角点 (x最小, y最大)
        bottom_left = transformed_points[np.argmin(transformed_points[:, 0] - transformed_points[:, 1])]
        # 右下角点 (x最大, y最大)
        bottom_right = transformed_points[np.argmax(transformed_points[:, 0] + transformed_points[:, 1])]
        
        corners = np.array([top_left, top_right, bottom_left, bottom_right])

        if debug:
            # 创建可视化图像
            vis_img = np.zeros((h_up, w_up, 3), dtype=np.uint8)
            
            # 在坐标点位置画红点
            for x, y in transformed_points:
                cv2.circle(vis_img, (int(x), int(y)), 5, (0, 0, 255), -1)  # 红色点，半径为5
            
            # 在顶点位置画蓝点
            for x, y in corners:
                cv2.circle(vis_img, (int(x), int(y)), 10, (255, 0, 0), -1)  # 蓝色点，半径为10
            
            # 连接顶点画线
            # cv2.polylines(vis_img, [corners.astype(np.int32)], True, (0, 255, 0), 2)  # 绿色线
            
            # 保存带有红点的图像
            cv2.imwrite('/home/work/houyi/pj_25_q1/mmuplt-avatarv3/hy_test/mask_points_blue.png', vis_img)
        
        return transformed_points




class FaceRestorer_torch_v2:
    def __init__(self, upscale_factor=2, face_size=(256, 256)):
        self.upscale_factor = 1
        ratio = 2.8
        self.crop_ratio = (ratio, ratio)
        self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
        self.face_template = self.face_template * ratio
        self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
        self.p_bias = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)

    def restore_img(self, input_img, face, affine_matrix):
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
        erosion_radius = w_edge * 2
        
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = w_edge * 2
        

        # inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
        # inv_soft_mask = inv_soft_mask[:, :, None]
        # inv_soft_mask_t = torch.from_numpy(inv_soft_mask).to(self.device)

        # 改用torch
        inv_mask_center_t= torch.from_numpy(inv_mask_center).to(self.device)
        inv_soft_mask_t = gaussian_blur_torch(inv_mask_center_t, blur_size + 1)
        inv_soft_mask_t = inv_soft_mask_t.unsqueeze(-1)  # 添加通道维度
        
        upsample_img_t = torch.from_numpy(upsample_img).to(self.device)
        
        result_t = inv_soft_mask_t * pasted_face_t + (1 - inv_soft_mask_t) * upsample_img_t
        result_t = result_t.to(dtype=torch.uint8)
        result = result_t.cpu().numpy()
        
        if np.max(result) > 256:
            result = result.astype(np.uint16)
        else:
            result = result.astype(np.uint8)
        return result





class FaceRestorer_torch_v3:
    def __init__(self, upscale_factor=2, face_size=(256, 256)):
        self.upscale_factor = 1
        ratio = 2.8
        self.crop_ratio = (ratio, ratio)
        self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
        self.face_template = self.face_template * ratio
        self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
        self.p_bias = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)

    def restore_img(self, input_img, face, affine_matrix):
        h, w, _ = input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        upsample_img = cv2.resize(input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        upsample_img_t = torch.from_numpy(upsample_img).to(self.device) 

        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        inverse_affine *= self.upscale_factor

        extra_offset = 0.5 * self.upscale_factor if self.upscale_factor > 1 else 0
        inverse_affine[:, 2] += extra_offset

        inv_restored = cv2.warpAffine(face, inverse_affine, (w_up, h_up), flags=cv2.INTER_LANCZOS4)
        # mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)
        inv_mask = cv2.warpAffine(self.mask, inverse_affine, (w_up, h_up)) # cv2.imwrite('/home/work/houyi/pj_25_q1/mmuplt-avatarv3/hy_test/inv_mask.png', (inv_mask * 255).astype(np.uint8))

        # 使用erode_torch替换cv2.erode
        # inv_mask_erosion = erode_torch(inv_mask, int(2 * self.upscale_factor))
        inv_mask_erosion = cv2.erode(inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8))

        inv_mask_erosion_t = torch.from_numpy(inv_mask_erosion).to(self.device).unsqueeze(-1)
        inv_restored_t = torch.from_numpy(inv_restored).to(self.device)
        pasted_face_t = inv_mask_erosion_t * inv_restored_t

        total_face_area = torch.sum(inv_mask_erosion_t).item()
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2
        
        # 使用erode_torch替换cv2.erode
        # print(f"inv_mask_erosion shape: {inv_mask_erosion.shape}")
        inv_mask_erosion_t = torch.from_numpy(inv_mask_erosion).to(self.device)
        inv_mask_center_t = erode_torch(inv_mask_erosion_t, erosion_radius)
        # print(f"inv_mask_center shape: {inv_mask_center.shape}")
        blur_size = w_edge * 2
        

        # inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
        # inv_soft_mask = inv_soft_mask[:, :, None]
        # inv_soft_mask_t = torch.from_numpy(inv_soft_mask).to(self.device)

        # 改用torch
        # inv_mask_center_t = torch.from_numpy(inv_mask_center).to(self.device) if isinstance(inv_mask_center, np.ndarray) else inv_mask_center
        inv_mask_center_t = inv_mask_center_t
        inv_soft_mask_t = gaussian_blur_torch(inv_mask_center_t, blur_size + 1)
        inv_soft_mask_t = inv_soft_mask_t.unsqueeze(-1)  # 添加通道维度
                
        result_t = inv_soft_mask_t * pasted_face_t + (1 - inv_soft_mask_t) * upsample_img_t
        result_t = result_t.to(dtype=torch.uint8)
        result = result_t.to(device='cpu', non_blocking=True).numpy()
        
        if np.max(result) > 256:
            result = result.astype(np.uint16)
        else:
            result = result.astype(np.uint8)
        return result


import numpy as np
import cv2
import torch
from einops import rearrange
import kornia


class FaceRestorer_new_0514:
    def __init__(self, upscale_factor=2, face_size=(256, 256), device="cuda"):
        self.upscale_factor = 1
        ratio = 2.8
        self.crop_ratio = (ratio, ratio)
        self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
        self.face_template = self.face_template * ratio
        self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
        self.p_bias = None
        # 预计算常用的结构元素以避免重复创建
        self.erosion_kernels = {}
        self.dtype = torch.float16
        self.device = device
        self.fill_value = torch.tensor([127, 127, 127], device=device, dtype=self.dtype)
        self.mask = torch.ones((1, 1, self.face_size[1], self.face_size[0]), device=device, dtype=self.dtype)

    def restore_img(self, input_img, face, affine_matrix):
        h, w, _ = input_img.shape

        if isinstance(affine_matrix, np.ndarray):
            affine_matrix = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)

        if isinstance(face, np.ndarray):
            face = torch.from_numpy(face).to(device=self.device)
            face = rearrange(face, "h w c -> c h w")
            # print(orch.max(face))
            face = face / 127.5 - 1.0
    

        inv_affine_matrix = kornia.geometry.transform.invert_affine_transform(affine_matrix)
        face = face.to(dtype=self.dtype).unsqueeze(0)

        inv_face = kornia.geometry.transform.warp_affine(
            face, inv_affine_matrix, (h, w), mode="bilinear", padding_mode="fill", fill_value=self.fill_value
        ).squeeze(0)
        inv_face = (inv_face / 2 + 0.5).clamp(0, 1) * 255

        input_img = rearrange(torch.from_numpy(input_img).to(device=self.device, dtype=self.dtype), "h w c -> c h w")
        inv_mask = kornia.geometry.transform.warp_affine(
            self.mask, inv_affine_matrix, (h, w), padding_mode="zeros"
        )  # (1, 1, h_up, w_up)

        inv_mask_erosion = kornia.morphology.erosion(
            inv_mask,
            torch.ones(
                (int(2 * self.upscale_factor), int(2 * self.upscale_factor)), device=self.device, dtype=self.dtype
            ),
        )

        inv_mask_erosion_t = inv_mask_erosion.squeeze(0).expand_as(inv_face)
        pasted_face = inv_mask_erosion_t * inv_face
        total_face_area = torch.sum(inv_mask_erosion.float())
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2

        # This step will consume a large amount of GPU memory.
        # inv_mask_center = kornia.morphology.erosion(
        #     inv_mask_erosion, torch.ones((erosion_radius, erosion_radius), device=self.device, dtype=self.dtype)
        # )

        # Run on CPU to avoid consuming a large amount of GPU memory.
        inv_mask_erosion = inv_mask_erosion.squeeze().cpu().numpy().astype(np.float32)
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        inv_mask_center = torch.from_numpy(inv_mask_center).to(device=self.device, dtype=self.dtype)[None, None, ...]

        blur_size = w_edge * 2 + 1
        sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8
        inv_soft_mask = kornia.filters.gaussian_blur2d(
            inv_mask_center, (blur_size, blur_size), (sigma, sigma)
        ).squeeze(0)
        inv_soft_mask_3d = inv_soft_mask.expand_as(inv_face)
        img_back = inv_soft_mask_3d * pasted_face + (1 - inv_soft_mask_3d) * input_img

        img_back = rearrange(img_back, "c h w -> h w c").contiguous().to(dtype=torch.uint8)
        img_back = img_back.cpu().numpy()
        return img_back

# 创建测试数据
def create_test_data():
    # 创建一个测试输入图像
    input_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    # 创建一个假的面部图像
    face = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # 创建仿射变换矩阵
    affine_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    
    return input_img, face, affine_matrix




# if __name__ == "__main__":
#     # 创建测试数据
#     input_img, face, affine_matrix = create_test_data()
#     input_img = np.load("/home/work/houyi/pj_25_q1/hy_test/frame.npy") # (2160, 2160, 3)
#     face = np.load("/home/work/houyi/pj_25_q1/hy_test/face.npy") # (280, 210, 3)
#     affine_matrix = np.load("/home/work/houyi/pj_25_q1/hy_test/affine_matrix.npy")

#     # # 创建FaceRestorer实例
#     # restorer = FaceRestorer_new_0514(upscale_factor=1., device="cuda")
#     # # 准备保存目录
#     # save_dir = "/home/work/houyi/pj_25_q1/hy_test/0515_new_restorer"
#     # os.makedirs(save_dir, exist_ok=True)    

#     # 创建FaceRestorer实例
#     restorer = FaceRestorer_torch_v3(upscale_factor=1.)
#     # 准备保存目录
#     save_dir = "/home/work/houyi/pj_25_q1/hy_test/0515_torch_v3"
#     os.makedirs(save_dir, exist_ok=True)    

#     # 使用line_profiler进行分析
#     profile = line_profiler.LineProfiler()
#     # profile = SecondLineProfiler()
#     profile.add_function(restorer.restore_img)
    
    
#     original_face_path = os.path.join(save_dir, "original_face.png")
#     original_face = cv2.cvtColor(face.copy(), cv2.COLOR_BGR2RGB)
#     cv2.imwrite(original_face_path, original_face)

#     original_img_path = os.path.join(save_dir, "original_img.png")
#     original_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_BGR2RGB)
#     cv2.imwrite(original_img_path, original_img)

#     # 执行函数几次以确保统计数据准确
#     times = 1
#     for i in range(times):
#         result = profile.runcall(restorer.restore_img, input_img, face, affine_matrix)
#         # 每10轮保存一次结果
#         if i % 1 == 0:
#             save_path = os.path.join(save_dir, f"restored_face_{i}.png")
#             cv2.imwrite(save_path, cv2.cvtColor(result.copy(), cv2.COLOR_BGR2RGB))
#             print(f"第{i}轮结果图像已保存到: {save_path}")
    
#     # 保存最后一轮的结果
#     save_path = os.path.join(save_dir, "restored_face_final.png")
#     cv2.imwrite(save_path, cv2.cvtColor(result.copy(), cv2.COLOR_BGR2RGB))
#     print(f"最终结果图像已保存到: {save_path}")
    
#     # 打印分析结果
#     profile.print_stats()




if __name__ == "__main__":
    # 创建测试数据
    input_img, face, affine_matrix = create_test_data()
    input_img = np.load("/home/work/avatar/mmuplt-avatarv8/latent_sync/hy_test/frame.npy") # (2160, 2160, 3)
    face = np.load("/home/work/avatar/mmuplt-avatarv8/latent_sync/hy_test/face.npy") # (280, 210, 3)
    affine_matrix = np.load("/home/work/avatar/mmuplt-avatarv8/latent_sync/hy_test/affine_matrix.npy")

    # # 创建FaceRestorer实例
    # restorer = FaceRestorer_new_0514(upscale_factor=1., device="cuda")
    # # 准备保存目录
    # save_dir = "/home/work/houyi/pj_25_q1/hy_test/0515_new_restorer"
    # os.makedirs(save_dir, exist_ok=True)    

    # 创建FaceRestorer实例
    restorer = FaceRestorer_torch_v3(upscale_factor=1.)
    # restorer = FaceRestorer_new_0514(upscale_factor=1., device="cuda")
    print(f"实例的类名：{restorer.__class__.__name__}")
    # # 准备保存目录
    # save_dir = "/home/work/houyi/pj_25_q1/hy_test/0515_torch"
    # os.makedirs(save_dir, exist_ok=True)    

    # 使用line_profiler进行分析
    profile = line_profiler.LineProfiler()
    # profile = SecondLineProfiler()
    profile.add_function(restorer.restore_img)
    
    
    # original_face_path = os.path.join(save_dir, "original_face.png")
    # original_face = cv2.cvtColor(face.copy(), cv2.COLOR_BGR2RGB)
    # cv2.imwrite(original_face_path, original_face)

    # original_img_path = os.path.join(save_dir, "original_img.png")
    # original_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_BGR2RGB)
    # cv2.imwrite(original_img_path, original_img)

    # 执行函数几次以确保统计数据准确
    times = 100
    print(f"开始计时执行 {times} 次...")
    start_time = time.time()
    for i in range(times):
        result = profile.runcall(restorer.restore_img, input_img, face, affine_matrix)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"总执行时间: {total_time:.4f} 秒")
    print(f"平均每次执行时间: {total_time/times:.4f} 秒")
    
    # 打印分析结果
    profile.print_stats()
