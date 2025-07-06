from datetime import datetime
from typing import Dict, List, Optional
from productscene.config.shopee_base import register_upload_config
import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from productscene import NVTXContext
from productscene.config.status_code import PI_STATUS_CODE
from productscene.src.utils.loader import upload_image

class Uploader:
    """
    图片上传器
    """
    def __init__(self, cfg: Dict):
        self.upload_config = register_upload_config[cfg['gen_output']['uploader']]
        self.upload_mask = True
        
    def __call__(self, row: Dict, gen_image: np.ndarray, gen_mask: Optional[np.ndarray] = None) -> Dict:
        """
        上传生成的图片和mask
        
        Args:
            row: 结果字典
            gen_image: 生成的图片 (RGB格式)
            gen_mask: 生成的mask
            
        Returns:
            Dict: 包含上传结果的字典
        """
        uploader_executor = ThreadPoolExecutor(max_workers=2)
        try:
            # 将RGB图片转换为BGR格式并编码为JPG
            gen_image_bgr = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', gen_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
            upload_res = uploader_executor.submit(upload_image, buffer.tobytes(), self.upload_config)

            # 上传mask（如果存在）
            if self.upload_mask and gen_mask is not None:
                _, mask_buffer = cv2.imencode('.png', gen_mask)
                upload_mask_res = uploader_executor.submit(upload_image, mask_buffer.tobytes(), self.upload_config)
                row['aigc_generated_mask_id'] = upload_mask_res.result()
            
            row['aigc_generated_image_id'] = upload_res.result()

        except Exception as e:
            row['status_code'] = PI_STATUS_CODE.UPLOAD_ERROR
            logger.error(f'Upload error: {e}')
        finally:
            uploader_executor.shutdown(wait=True)

        return row


class DebugSaver:
    """
    Debug图片保存器
    """
    def __init__(self, enabled: bool = False, output_path: str = './output/'):
        self.enabled = enabled
        self.output_path = output_path
        
        if self.enabled:
            os.makedirs(self.output_path, exist_ok=True)
    
    @NVTXContext
    def save_debug_image(self, image: np.ndarray, filename: str, image_format: str = 'RGB') -> bool:
        """
        保存debug图片到本地
        
        Args:
            image: 图片数组
            filename: 文件名（不包含扩展名）
            image_format: 图片格式 ('RGB' 或 'BGR')
            
        Returns:
            bool: 是否保存成功
        """
        if not self.enabled:
            return False
            
        try:
            # 根据输入格式转换为BGR（OpenCV默认格式）
            if image_format == 'RGB':
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 精确到秒
            
            # 构建完整路径，加入时间戳
            debug_image_path = os.path.join(self.output_path, f'{filename}_{timestamp}.png')
            
            # 保存图片
            success = cv2.imwrite(debug_image_path, image_bgr)
            
            if success:
                logger.info(f"Saved debug image to {debug_image_path}")
                return True
            else:
                logger.error(f"Failed to save debug image: {debug_image_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving debug image {filename}: {e}")
            return False
    
    def save_debug_mask(self, mask: np.ndarray, filename: str) -> bool:
        """
        保存debug mask到本地
        
        Args:
            mask: mask数组
            filename: 文件名（不包含扩展名）
            
        Returns:
            bool: 是否保存成功
        """
        if not self.enabled:
            return False
            
        try:
            # 如果mask是3通道，取第一个通道
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 精确到秒
            
            # 构建完整路径，加入时间戳
            debug_mask_path = os.path.join(self.output_path, f'{filename}_mask_{timestamp}.png')
            
            # 保存mask
            success = cv2.imwrite(debug_mask_path, mask)
            
            if success:
                logger.info(f"Saved debug mask to {debug_mask_path}")
                return True
            else:
                logger.error(f"Failed to save debug mask: {debug_mask_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving debug mask {filename}: {e}")
            return False

