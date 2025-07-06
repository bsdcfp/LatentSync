import torch
import os
import threading
from typing import Optional, Dict, List
import numpy as np
import cv2

from loguru import logger
# from cuda import cudart

from oneservice import InputDictType, OutputDictType, BackendConfig
from oneservice.backends.base import BaseBackend

import torch.nn.functional as F

class SNNFastBackend(BaseBackend):
    
    def __init__(self, config: BackendConfig):
        super(SNNFastBackend, self).__init__(config)
        model_path = config.model_path

        self._use_fp16 = config.fp16
        self._device = config.device
        self._size = config.size
        
    #    # NOTE: do load model process
    #     config_path = model_path + "/infer_768_v0510.yaml"
    #     ckpt_path = model_path + "/tryon_infer_0510.ckpt"
    #     matth_model_path = model_path + "/SGHM-ResNet50.pth"

    #     # Load tensorrt engine
    #     use_cuda_graph = config.use_cuda_graph
    #     batch_size = config.batch_size
    #     verbose = config.verbose
    #     device = torch.cuda.get_device_name(torch.cuda.current_device())
    #     device_path = 'A100' if 'A100' in device else 'T4'
    #     trt_root_path = os.path.join(model_path, 'tensorrt', device_path)

    #     # model_list = [self._model.model, self._model.first_stage_model, self._model.conditioner.embedders[0]]
    #     model_list = [self._model.model, self._model.first_stage_model, self._model.conditioner.embedders[0]]

    #     for item in model_list:
    #         item.load_trt(trt_root_path, batch_size, use_cuda_graph, verbose)

        

    #     max_device_memory = 0
    #     for item in model_list:
    #         max_device_memory = max(max_device_memory, item.calculate_max_device_memory())
        
    #     _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
    #     for item in model_list:
    #         item.activate_engine(shared_device_memory, batch_size)

        self._lock = threading.Lock()


    def forward(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray], 
        generated_config: List[Dict],
    ):
        """
        Args:
            images: List of RGB images as numpy arrays
            masks: List of mask arrays 
            generated_config: List of generation configuration dictionaries
        
        Returns:
            List of generation results with keys: 'gen_addfg', 'gen_mask', 'has_nsfw_concept'
        """
        results = []
        
        with self._lock:
            for idx, (image, mask, config) in enumerate(zip(images, masks, generated_config)):
                logger.info(f"[INFO] SNNFastBackend processing image {idx + 1}/{len(images)}")
                
                # For now, return dummy results since the actual implementation is not complete
                # You would implement the actual TensorRT inference here
                dummy_result = {
                    'gen_addfg': image,  # Return original image as placeholder
                    'gen_mask': mask[:, :, 0],  # Return first channel of mask as placeholder
                    'has_nsfw_concept': False
                }
                results.append(dummy_result)
                
        return results
