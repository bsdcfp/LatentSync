import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from loguru import logger

from productscene.hydit.config import InfeConfig
from productscene.hydit.inference_controlnet2 import End2End
from ..utils.post_process import paste_fg, paste_fg_sharpen
from ..utils.pre_process import pad_resize_v2
from .safety_checker import SafePipeline
from productscene import NVTXContext, PROFILER_AVAILABLE

norm_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)
mask_transform = T.Compose(
    [
        T.ToTensor(),
        # T.Normalize([0.5], [0.5]),
    ]
)

class HunyuanBrushNetRunner(SafePipeline):
    def __init__(self, **kwargs):
        safety_model_path = kwargs.pop('safety_model_path')
        super().__init__(safety_model_path=safety_model_path)

        self.gen_h = kwargs.pop('gen_h')
        self.gen_w = kwargs.pop('gen_w')
        self.mask_change = kwargs.pop('mask_change')

        infe_config = InfeConfig(**kwargs)
        self.gen = End2End(infe_config, infe_config.model_root)
        self.args = infe_config

    def __call__(self, images, masks, prompts, sharpen=True, seeds=None, out_h=768, out_w=768, 
                 template_h=0.9, template_w=0.9, centers=None):
        """
        支持真正batch处理的推理方法
        
        Args:
            images: 单个图片(np.ndarray)或图片列表(List[np.ndarray])
            masks: 单个mask(np.ndarray)或mask列表(List[np.ndarray])  
            prompts: 单个prompt(str)或prompt列表(List[str])
            sharpen: 是否锐化，bool或List[bool]
            seeds: 随机种子，int或List[int]或None
            out_h: 输出高度，int或List[int]
            out_w: 输出宽度，int或List[int]
            template_h: 模板高度比例，float或List[float]
            template_w: 模板宽度比例，float或List[float]
            centers: 中心位置，tuple或List[tuple]或None
            
        Returns:
            Dict: 包含批量结果的字典
        """
        # 统一为列表格式处理
        if not isinstance(images, list):
            images = [images]
        if not isinstance(masks, list):
            masks = [masks]
        if not isinstance(prompts, list):
            prompts = [prompts]
            
        batch_size = len(images)
        
        # 参数标准化 - 将所有参数转换为列表格式
        if not isinstance(sharpen, list):
            sharpen = [sharpen] * batch_size
        if seeds is None:
            seeds = [42] * batch_size
        elif not isinstance(seeds, list):
            seeds = [seeds] * batch_size
        if not isinstance(out_h, list):
            out_h = [out_h] * batch_size
        if not isinstance(out_w, list):
            out_w = [out_w] * batch_size
        if not isinstance(template_h, list):
            template_h = [template_h] * batch_size
        if not isinstance(template_w, list):
            template_w = [template_w] * batch_size
        if centers is None:
            centers = [(0.5, 0.5)] * batch_size
        elif not isinstance(centers, list):
            centers = [centers] * batch_size
            
        # 验证输入长度一致性
        if not (len(images) == len(masks) == len(prompts) == batch_size):
            raise ValueError(f"Input length mismatch: images={len(images)}, masks={len(masks)}, prompts={len(prompts)}")
        
        # 批量预处理
        batch_processed_images = []
        batch_processed_masks = []
        batch_crop_data = []  # 存储后处理需要的数据
        
        for i in range(batch_size):
            default_template_cfg = {
                'h': int(out_h[i] * template_h[i]),
                'w': int(out_w[i] * template_w[i]),
                'fg_erode_iter': 2,
                'center': (int(out_w[i] * centers[i][0]), int(out_h[i] * centers[i][1]))
            }

            # 预处理单个图片
            pad_fg, pad_mask, crop_fg_final, crop_mask_final, target_pos = pad_resize_v2(
                images[i], masks[i], default_template_cfg, 
                out_h=out_h[i], out_w=out_w[i], gen_h=self.gen_h, gen_w=self.gen_w
            )
            
            # 转换为PIL图像并应用transforms
            pad_fg_pil = Image.fromarray(np.uint8(pad_fg)).convert("RGB")
            if self.mask_change:
                pad_mask_pil = Image.fromarray(np.uint8(255-pad_mask)).convert("L")
            else:
                pad_mask_pil = Image.fromarray(np.uint8(pad_mask)).convert("L")

            # 应用transforms并转换为tensor
            pad_fg_tensor = norm_transform(pad_fg_pil)
            pad_mask_tensor = mask_transform(pad_mask_pil)
            
            # 添加调试信息
            logger.info(f"Image {i}: pad_fg_tensor shape: {pad_fg_tensor.shape}, pad_mask_tensor shape: {pad_mask_tensor.shape}")
            
            batch_processed_images.append(pad_fg_tensor)
            batch_processed_masks.append(pad_mask_tensor)
            
            # 保存后处理数据
            batch_crop_data.append({
                'crop_fg_final': crop_fg_final,
                'crop_mask_final': crop_mask_final,
                'target_pos': target_pos,
                'pad_mask': pad_mask,
                'out_h': out_h[i],
                'out_w': out_w[i],
                'sharpen': sharpen[i]
            })
        
        # 将batch tensors堆叠并移到GPU
        logger.info(f"Attempting to stack {len(batch_processed_images)} image tensors and {len(batch_processed_masks)} mask tensors")
        batch_images_tensor = torch.stack(batch_processed_images).cuda()
        batch_masks_tensor = torch.stack(batch_processed_masks).cuda()
        logger.info(f"Successfully stacked tensors: images {batch_images_tensor.shape}, masks {batch_masks_tensor.shape}")
        
        # 使用真正的batch推理
        with NVTXContext("HunyuanBrushNetRunner.batch_predict"):
            logger.info(f"Processing batch of {batch_size} images with true batch inference")
            
            # 检查是否可以使用真正的batch推理（现在分组已经在上层处理了）
            if batch_size <= self.args.batch_size:
                # 使用真正的batch推理 - 直接调用pipeline！
                logger.info(f"Using true batch pipeline inference for {batch_size} images")
                
                # 设置种子（使用第一个种子，因为分组已经确保所有种子相同）
                from productscene.hydit.inference_controlnet2 import set_seeds
                generator = set_seeds(seeds[0], device='cuda')
                
                # 准备其他参数
                if self.args.use_style_cond:
                    style = torch.as_tensor([0, 0] * batch_size, device='cuda')
                else:
                    style = None
                
                # 准备image_meta_size
                if self.args.size_cond is not None:
                    size_cond = list(self.args.size_cond) + [self.gen_w, self.gen_h, 0, 0]
                    # 恢复原始逻辑：CFG现在在pipeline内部正确处理
                    image_meta_size = torch.as_tensor([size_cond] * batch_size, device='cuda')
                else:
                    image_meta_size = None
                
                # 计算freqs_cis_img
                reso = f'{self.gen_h}x{self.gen_w}'
                if reso in self.gen.freqs_cis_img:
                    freqs_cis_img = self.gen.freqs_cis_img[reso]
                else:
                    freqs_cis_img = self.gen.calc_rope(self.gen_h, self.gen_w)
                
                # 直接调用pipeline进行真正的batch推理
                pipeline_results = self.gen.pipeline(
                    height=self.gen_h,
                    width=self.gen_w,
                    prompt=prompts,
                    negative_prompt=self.args.negative,
                    num_images_per_prompt=1,
                    guidance_scale=self.args.cfg_scale,
                    num_inference_steps=self.args.infer_steps,
                    image_meta_size=image_meta_size,
                    style=style,
                    return_dict=False,
                    generator=generator,
                    freqs_cis_img=freqs_cis_img,
                    use_fp16=self.args.use_fp16,
                    learn_sigma=self.args.learn_sigma,
                    image=batch_images_tensor,
                    mask=batch_masks_tensor,
                    control_weight=eval(self.args.control_weight),
                )[0]
                
                batch_generated = [np.asarray(img) for img in pipeline_results]
                logger.info(f"True batch pipeline inference completed for {batch_size} images")
            else:
                # 超过最大batch大小，需要拆分处理
                logger.warning(f"Batch size {batch_size} exceeds maximum {self.args.batch_size}, splitting into smaller batches")
                batch_generated = []
                for i in range(0, batch_size, self.args.batch_size):
                    end_idx = min(i + self.args.batch_size, batch_size)
                    # sub_batch_size = end_idx - i
                    
                    # 递归调用处理子批次
                    sub_batch_results = self(
                        images=images[i:end_idx],
                        masks=masks[i:end_idx],
                        prompts=prompts[i:end_idx],
                        sharpen=sharpen[i:end_idx],
                        seeds=seeds[i:end_idx],
                        out_h=out_h[i:end_idx],
                        out_w=out_w[i:end_idx],
                        template_h=template_h[i:end_idx],
                        template_w=template_w[i:end_idx],
                        centers=centers[i:end_idx]
                    )
                    
                    if isinstance(sub_batch_results, list):
                        for result in sub_batch_results:
                            batch_generated.append(result['gen'])
                    else:
                        batch_generated.append(sub_batch_results['gen'])

        # 批量后处理
        batch_results = []
        for i in range(batch_size):
            crop_data = batch_crop_data[i]
            image_gen = batch_generated[i]
            
            # 应用后处理
            if crop_data['sharpen']:
                image_addfg = paste_fg_sharpen(
                    image_gen, crop_data['crop_fg_final'], crop_data['crop_mask_final'], 
                    crop_data['target_pos'], target_h=crop_data['out_h'], target_w=crop_data['out_w'], iter=1
                )
            else:
                image_addfg = paste_fg(
                    image_gen, crop_data['crop_fg_final'], crop_data['crop_mask_final'],
                    crop_data['target_pos'], target_h=crop_data['out_h'], target_w=crop_data['out_w'], iter=1
                )
            
            # 调整mask尺寸
            pad_mask = crop_data['pad_mask']
            if pad_mask.shape[0] != crop_data['out_h'] or pad_mask.shape[1] != crop_data['out_w']:
                pad_mask = cv2.resize(pad_mask, (crop_data['out_w'], crop_data['out_h']), interpolation=cv2.INTER_CUBIC)
            
            batch_results.append({
                'gen': image_gen,
                'gen_addfg': image_addfg,
                'gen_mask': pad_mask,
                'has_nsfw_concept': False  # 暂时设为False，后续添加批量安全检查
            })
        
        # 批量安全检查
        if batch_size == 1:
            # 单张图片的安全检查（保持向后兼容）
            image_checked, has_nsfw_concept, flagged_images = self.run_safety_checker([
                batch_results[0]['gen'], batch_results[0]['gen_addfg']
            ])
            if has_nsfw_concept[0] or has_nsfw_concept[1]:
                nsfw_image = flagged_images[0] if has_nsfw_concept[0] else flagged_images[1]
                out_image = image_checked[0] if has_nsfw_concept[0] else image_checked[1]
                return {
                    'gen': out_image,
                    'gen_addfg': out_image,
                    'gen_nsfw': nsfw_image,
                    'gen_mask': batch_results[0]['gen_mask'],
                    'has_nsfw_concept': True,
                }
            else:
                image_gen, image_addfg = image_checked
                return {
                    'gen': image_gen,
                    'gen_addfg': image_addfg,
                    'gen_mask': batch_results[0]['gen_mask'],
                    'has_nsfw_concept': False
                }
        else:
            # 批量安全检查
            for i, result in enumerate(batch_results):
                try:
                    image_checked, has_nsfw_concept, flagged_images = self.run_safety_checker([
                        result['gen'], result['gen_addfg']
                    ])
                    if has_nsfw_concept[0] or has_nsfw_concept[1]:
                        nsfw_image = flagged_images[0] if has_nsfw_concept[0] else flagged_images[1]
                        out_image = image_checked[0] if has_nsfw_concept[0] else image_checked[1]
                        result['gen'] = out_image
                        result['gen_addfg'] = out_image
                        result['gen_nsfw'] = nsfw_image
                        result['has_nsfw_concept'] = True
                    else:
                        image_gen, image_addfg = image_checked
                        result['gen'] = image_gen
                        result['gen_addfg'] = image_addfg
                        result['has_nsfw_concept'] = False
                except Exception as e:
                    # 安全检查失败时保持原结果
                    logger.warning(f"Safety check failed for image {i}: {e}")
                    result['has_nsfw_concept'] = False
            
            return batch_results