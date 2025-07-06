import cv2
import numpy as np
import base64
from typing import List, Tuple, Optional, Union
from loguru import logger

from latentsync import NVTXContext, PROFILER_AVAILABLE
from latentsync.src.utils.status_code import PI_STATUS_CODE

class SNNPreprocessor:
    """
    负责图片的下载、读取和预处理功能
    """
    
    def __init__(self):
        pass
    
    def _is_base64_data(self, data: str) -> bool:
        """
        检测输入数据是否为base64编码的图片数据
        
        Args:
            data: 输入数据字符串
            
        Returns:
            bool: 是否为base64数据
        """
        # base64数据通常很长且包含特定字符
        if len(data) < 100:  # 图片的base64数据通常很长
            return False
        
        # 检查是否包含base64字符集
        import string
        base64_chars = string.ascii_letters + string.digits + '+/='
        
        # 如果90%以上的字符都是base64字符，认为是base64数据
        valid_chars = sum(1 for c in data if c in base64_chars)
        return (valid_chars / len(data)) > 0.9
    
    def _decode_base64_image(self, base64_data: str) -> Optional[np.ndarray]:
        """
        解码base64图片数据
        
        Args:
            base64_data: base64编码的图片数据
            
        Returns:
            np.ndarray: 解码后的图片数组，失败时返回None
        """
        try:
            # 解码base64数据
            image_bytes = base64.b64decode(base64_data)
            
            # 将字节数据转换为numpy数组
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # 使用cv2解码图片
            image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            if image is None:
                logger.error("Failed to decode base64 image data")
                return None
            
            logger.info(f"Successfully decoded base64 image: shape {image.shape}, dtype {image.dtype}")
            return image
            
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            return None
    
    @NVTXContext
    def process_input_data(self, input_data: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        处理输入数据（可能是image_id或base64数据）
        
        Args:
            input_data: 输入数据（image_id或base64编码的图片）
            
        Returns:
            Tuple[image, mask, status_code]: 
            - image: RGB格式的图片数组
            - mask: mask数组 
            - status_code: 状态码
        """
        try:
            # 检测输入类型
            if self._is_base64_data(input_data):
                logger.info("Detected base64 image data, decoding...")
                # 解码base64数据
                foreground_image = self._decode_base64_image(input_data)
                if foreground_image is None:
                    return None, None, PI_STATUS_CODE.DOWNLOAD_FG_ERROR
            else:
                logger.info(f"Detected image_id, downloading: {input_data[:20]}...")
                # 下载图片
                foreground_image = download_sp_image(input_data, cv2.IMREAD_UNCHANGED)
                if foreground_image is None:
                    logger.error(f"Failed to download image: {input_data[:20]}...")
                    return None, None, PI_STATUS_CODE.DOWNLOAD_FG_ERROR
            
            # 处理图片格式
            image, mask = self._process_image_format(foreground_image)
            
            return image, mask, PI_STATUS_CODE.SUCCESS
            
        except Exception as e:
            logger.error(f"Error processing input data: {e}")
            return None, None, PI_STATUS_CODE.DOWNLOAD_FG_ERROR
    
    @NVTXContext
    def _process_image_format(self, foreground_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将下载的图片转换为模型输入格式
        
        Args:
            foreground_image: BGRA格式的原始图片
            
        Returns:
            Tuple[image, mask]: RGB图片和mask
        """
        # 将图片转换为RGB格式
        image = cv2.cvtColor(foreground_image, cv2.COLOR_BGRA2RGB)
        
        # 提取alpha通道作为mask，并扩展为3通道
        alpha_channel = foreground_image[:, :, -1]
        mask = np.tile(alpha_channel[..., np.newaxis], [1, 1, 3])
        
        return image, mask
    
    @NVTXContext 
    def batch_process_input_data(self, input_data_list: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[int]]:
        """
        批量处理输入数据（可能是image_id或base64数据）
        
        Args:
            input_data_list: 输入数据列表（image_id或base64编码的图片）
            
        Returns:
            Tuple[images, masks, valid_input_data, status_codes]:
            - images: 成功处理的图片列表
            - masks: 对应的mask列表
            - valid_input_data: 成功处理的输入数据标识列表
            - status_codes: 所有数据的状态码列表
        """
        images = []
        masks = []
        valid_input_data = []
        status_codes = []
        
        for i, input_data in enumerate(input_data_list):
            # 为了避免打印长base64数据，生成简短的标识
            if self._is_base64_data(input_data):
                data_id = f"base64_data_{i}"
            else:
                data_id = input_data[:20] + "..." if len(input_data) > 20 else input_data
            
            image, mask, status_code = self.process_input_data(input_data)
            status_codes.append(status_code)
            
            if status_code == PI_STATUS_CODE.SUCCESS:
                images.append(image)
                masks.append(mask)
                valid_input_data.append(data_id)
            else:
                logger.warning(f"Failed to process input data {data_id}, status: {status_code}")
        
        return images, masks, valid_input_data, status_codes 