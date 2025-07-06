import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (
    BrushNetModel,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    StableDiffusionXLBrushNetPipeline,
)

from ..utils.post_process import paste_fg, paste_fg_sharpen
from ..utils.pre_process import pad_resize_v2
from .safety_checker import SafePipeline


class SDXLBrushNetRunner(SafePipeline):
    def __init__(self, base_model_path, brushnet_model_path, lora_config=None,
            steps=30, guidance_scale=5, multistep_scheduler=True, gen_h=768, gen_w=768,
            pos_prompt_prefix='High Resolution, Perfect for creating dynamic, ',
            neg_prompt="Naked, Nude, suspended objects, incomplete table, floating table, shining light, clear reflection, blurred, cartoon plant, sofa, complicated, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
            **kwargs
        ):
        super().__init__(**kwargs)

        brushnet = BrushNetModel.from_pretrained(brushnet_model_path, torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLBrushNetPipeline.from_pretrained(base_model_path, brushnet=brushnet, 
                torch_dtype=torch.float16, low_cpu_mem_usage=False, use_safetensors=False
            )
        if lora_config is not None:
            self.pipe.load_lora_weights(lora_config['model_path'], adapter_name=lora_config['lora_name'])
            self.pipe.set_adapters(adapter_names=[lora_config['lora_name']], adapter_weights=[lora_config['lora_weight']])
        if multistep_scheduler:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to('cuda')
        self.gen_h = gen_h
        self.gen_w = gen_w
        self.prefix = pos_prompt_prefix
        self.neg_prompt = neg_prompt
        self.num_inference_steps = steps
        self.guidance_scale = guidance_scale
        self.brushnet_conditioning_scale = 1.0

    def __call__(self, image, mask, prompt, use_original_input=False,
                 out_h=1024, out_w=1024, bottom=None, template_h=0.8, template_w=0.8, center=None, sharpen=False):
        # inputs of inpaint model
        if not use_original_input: 
            default_template_cfg = {
                'h': int(out_h*template_h),
                'w': int(out_w*template_w),
                'fg_erode_iter': 3
            }
            if bottom is not None: 
                default_template_cfg['bottom'] = int(out_h * bottom)
            if center is not None:
                default_template_cfg['center'] = (int(out_w*center[0]), int(out_h*center[1]))

            pad_fg, pad_mask, crop_fg_final, crop_mask_final, target_pos = pad_resize_v2(
                image, mask, default_template_cfg, out_h=out_h, out_w=out_w, gen_h=self.gen_h, gen_w=self.gen_w)
            pad_fg = Image.fromarray(np.uint8(pad_fg)).convert("RGB")
            pad_mask_input = Image.fromarray(np.uint8(255-pad_mask)).convert("RGB")
        else:
            pad_fg = Image.fromarray(np.uint8(image)).convert("RGB")
            pad_mask = mask.copy()
            pad_mask_input = Image.fromarray(np.uint8(255-mask)).convert("RGB")
            crop_fg_final = image.copy()
            crop_mask_final = mask.copy()
            target_pos = [0, 0, image.shape[1], image.shape[0]]
            out_h = image.shape[0]
            out_w = image.shape[1]
        prompt = self.prefix + prompt

        # inference inpaint model
        image_gen = self.pipe(
            image=pad_fg,
            mask=pad_mask_input,
            prompt=prompt,
            negative_prompt=self.neg_prompt,
            brushnet_conditioning_scale=self.brushnet_conditioning_scale,
            generator=None,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale
        ).images[0]

        image_gen = np.asarray(image_gen)
        if sharpen:
            image_addfg = paste_fg_sharpen(image_gen, crop_fg_final, crop_mask_final, target_pos, 
                                           target_h=out_h, target_w=out_w, iter=2)
        else:
            image_addfg = paste_fg(image_gen, crop_fg_final, crop_mask_final, target_pos, 
                                target_h=out_h, target_w=out_w, iter=2)
        if pad_mask.shape[0] != out_h or pad_mask.shape[1] != out_w:
            pad_mask = cv2.resize(pad_mask, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

        # safety checker
        image_checked, has_nsfw_concept, flagged_images = self.run_safety_checker([image_gen, image_addfg])

        if has_nsfw_concept[0] or has_nsfw_concept[1]:
            nsfw_image = flagged_images[0] if has_nsfw_concept[0] else flagged_images[1]
            out_image = image_checked[0] if has_nsfw_concept[0] else image_checked[1]
            return {
                'gen': out_image,
                'gen_addfg': out_image,
                'gen_nsfw': nsfw_image,
                'gen_mask': pad_mask,
                'has_nsfw_concept': True,
            }
        else:
            image_gen, image_addfg = image_checked
            return {
                'gen': image_gen,
                'gen_addfg': image_addfg,
                'gen_mask': pad_mask,
                'has_nsfw_concept': False
            }