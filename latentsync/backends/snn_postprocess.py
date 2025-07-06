import cv2
import numpy as np
import os
import json
import base64
from typing import Dict, List, Optional
from loguru import logger
from datetime import datetime

from latentsync import NVTXContext, PROFILER_AVAILABLE
from latentsync.src.utils.status_code import PI_STATUS_CODE


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


class SNNPostprocessor:
    """
    负责debug保存和结果后处理功能
    """
    
    def __init__(self, upload_config: Optional[Dict] = None, debug_enabled: bool = False, debug_output_path: str = './output/'):
        """
        初始化后处理器
        
        Args:
            upload_config: 上传配置（已废弃，保留参数兼容性）
            debug_enabled: 是否启用debug保存
            debug_output_path: debug输出路径
        """
        self.debug_saver = DebugSaver(debug_enabled, debug_output_path)
        
    @NVTXContext
    def process_single_result(self, gen_results: Dict, image_id: str, index: int = 0, input_type: str = 'image_id') -> Dict:
        """
        处理单个推理结果
        
        Args:
            gen_results: 模型推理结果
            image_id: 原始图片ID
            index: 图片索引（用于debug文件命名）
            input_type: 输入类型 ('image_id'、'base64' 或 'image_array')
            
        Returns:
            Dict: 处理后的结果
        """
        with NVTXContext("SNNPostprocessor.process_single_result"):
            result = {'status_code': PI_STATUS_CODE.SUCCESS}
            
            # 检查NSFW
            if gen_results.get('has_nsfw_concept', False):
                result['status_code'] = PI_STATUS_CODE.NSFW_ERROR
                logger.warning(f"NSFW content detected for image {image_id}")
            
            # Debug保存
            if self.debug_saver.enabled:
                self._save_debug_files(gen_results, image_id, index)
            
            # 根据输入类型决定处理方式
            if input_type == 'image_id':
                # 输入是image_id，返回image_id（简化处理）
                result['aigc_generated_image_id'] = f"generated_{image_id}"
                result['aigc_generated_mask_id'] = f"mask_{image_id}"
            elif input_type == 'base64':
                # 输入是base64，返回base64
                result['aigc_generated_base64'] = self._to_base64(gen_results.get('gen_addfg', np.zeros((256, 256, 3), dtype=np.uint8)))
                result['aigc_generated_mask_base64'] = self._to_base64(gen_results.get('gen_mask', np.zeros((256, 256), dtype=np.uint8)), format='png')
            else:  # input_type == 'image_array'
                # 输入是image_array，返回image_array
                result['aigc_generated_image_array'] = gen_results.get('gen_addfg', np.zeros((256, 256, 3), dtype=np.uint8))
                result['aigc_generated_mask_array'] = gen_results.get('gen_mask', np.zeros((256, 256), dtype=np.uint8))
            
            return result
    
    def _save_debug_files(self, gen_results: Dict, image_id: str, index: int):
        """
        保存debug文件
        
        Args:
            gen_results: 推理结果
            image_id: 图片ID
            index: 索引
        """
        if not self.debug_saver.enabled:
            return
        
        # 保存生成的图片
        if 'gen_addfg' in gen_results:
            self.debug_saver.save_debug_image(
                gen_results['gen_addfg'], 
                f"{image_id}_gen_{index}", 
                'RGB'
            )
        
        # 保存生成的mask
        if 'gen_mask' in gen_results:
            self.debug_saver.save_debug_mask(
                gen_results['gen_mask'], 
                f"{image_id}_gen_{index}"
            )
    
    def _to_base64(self, data: np.ndarray, format: str = 'jpg', quality: int = 95) -> str:
        """
        将numpy数组转换为base64字符串
        
        Args:
            data: numpy数组
            format: 图片格式 ('jpg' 或 'png')
            quality: JPEG质量 (1-100)
            
        Returns:
            str: base64编码的字符串
        """
        try:
            # 根据格式设置编码参数
            if format.lower() == 'jpg':
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            else:  # png
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
            
            # 编码图片
            success, buffer = cv2.imencode(f'.{format}', data, encode_params)
            
            if not success:
                logger.error(f"Failed to encode image to {format}")
                return ""
            
            # 转换为base64
            base64_str = base64.b64encode(buffer).decode('utf-8')
            return base64_str
            
        except Exception as e:
            logger.error(f"Error converting to base64: {e}")
            return ""
    
    @NVTXContext
    def process_batch_results(self, gen_results_list: List[Dict], image_ids: List[str], input_type: str = 'image_id') -> List[Dict]:
        """
        批量处理推理结果
        
        Args:
            gen_results_list: 推理结果列表
            image_ids: 图片ID列表
            input_type: 输入类型 ('image_id'、'base64' 或 'image_array')
            
        Returns:
            List[Dict]: 处理后的结果列表
        """
        processed_results = []
        
        for idx, (gen_results, image_id) in enumerate(zip(gen_results_list, image_ids)):
            result = self.process_single_result(gen_results, image_id, idx, input_type)
            processed_results.append(result)
        
        return processed_results 