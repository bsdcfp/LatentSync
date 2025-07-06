import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, BrushNetModel, DPMSolverMultistepScheduler

from ..modules.clip_masked_vit import MaskedCLIPVisionModelWithProjection
from ..pipelines.pipeline_brushnet_sd_xl_mask import (
    MaskedStableDiffusionXLBrushNetPipeline,
)
from ..utils.post_process import paste_fg, paste_fg_sharpen
from ..utils.pre_process import pad_resize_v2


class SDXLBrushNetIpadapterRunner():
    def __init__(self, base_model_path, brushnet_model_path, ipadapter_model_path):
        image_encoder = MaskedCLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16
        )
        brushnet = BrushNetModel.from_pretrained(brushnet_model_path, torch_dtype=torch.float16)
        self.pipe = MaskedStableDiffusionXLBrushNetPipeline.from_pretrained(
            base_model_path, image_encoder=image_encoder, brushnet=brushnet, 
            torch_dtype=torch.float16, low_cpu_mem_usage=False, use_safetensors=True
        )
        self.pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        ipadapter_model_path = ipadapter_model_path.split("/")
        self.pipe.load_ip_adapter('/'.join(ipadapter_model_path[:-2]), subfolder=ipadapter_model_path[-2], weight_name=ipadapter_model_path[-1])
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.suffix = "photography, ultra highres, best quality, high quality"
        self.neg_prompt = "human, suspended objects, incomplete table, floating table, shining light, clear reflection, blurred, cartoon plant, sofa, complicated, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
        self.num_inference_steps = 30
        self.brushnet_conditioning_scale = 1.0

    def __call__(self, image, mask, ip_image, ip_image_mask, prompt, ip_adapter_scale, use_original_input=False,
                 out_h=1080, out_w=1080, bottom=None, template_h=0.8, template_w=0.8, center=None, sharpen=False):
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
                image, mask, default_template_cfg, out_h=out_h, out_w=out_w)
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
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        ip_image = Image.fromarray(np.uint8(ip_image)).convert("RGB")
        ip_image_mask = Image.fromarray(np.uint8(ip_image_mask)).convert("RGB")

        prompt = prompt + self.suffix

        image_gen = self.pipe(
            image=pad_fg,
            mask=pad_mask_input,
            ip_adapter_image=ip_image,
            ip_adapter_image_mask=ip_image_mask,
            prompt=prompt,
            negative_prompt=self.neg_prompt,
            brushnet_conditioning_scale=self.brushnet_conditioning_scale,
            generator=None,
            num_inference_steps=self.num_inference_steps
        ).images[0]

        if sharpen:
            image_addfg = paste_fg_sharpen(np.asarray(image_gen), crop_fg_final, crop_mask_final, target_pos, 
                                           target_h=out_h, target_w=out_w, iter=2)
        else:
            image_addfg = paste_fg(np.asarray(image_gen), crop_fg_final, crop_mask_final, target_pos, 
                                   target_h=out_h, target_w=out_w, iter=2)

        return {
            'gen': np.asarray(image_gen),
            'gen_addfg': image_addfg,
            'gen_mask': pad_mask
        }