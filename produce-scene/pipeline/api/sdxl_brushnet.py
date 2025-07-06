import numpy as np
import random
import torch
import os
import sys
from cuda import cudart

from PIL import Image
from diffusers import (
    BrushNetModel,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    StableDiffusionXLBrushNetPipeline,
)

from ..utils.post_process import paste_fg, paste_fg_sharpen
from ..utils.pre_process import pad_resize_v2

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "../..")
sys.path.append(parent_dir)

from trt_backend import TrtExecutor
class SDXLBrushNetRunner():
    def __init__(self, base_model_path, brushnet_model_path,
            steps=6, guidance_scale=2, multistep_scheduler=False,
            gen_h=1024,gen_w=1024,
            pos_prompt_prefix='High Resolution, Perfect for creating dynamic, ',
            neg_prompt="Naked, Nude, suspended objects, incomplete table, floating table, shining light, clear reflection, blurred, cartoon plant, sofa, complicated, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
        ):
        brushnet = BrushNetModel.from_pretrained(brushnet_model_path, torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLBrushNetPipeline.from_pretrained(base_model_path, brushnet=brushnet,
                torch_dtype=torch.float16, low_cpu_mem_usage=False, use_safetensors=False
            )
        if multistep_scheduler:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(self.pipe.scheduler.config)

        # self.pipe.to('cuda')
        self.pipe.enable_model_cpu_offload()
        self.prefix = pos_prompt_prefix
        self.neg_prompt = neg_prompt
        self.num_inference_steps = steps
        self.pipe.step_to_skip_cfg = steps
        self.guidance_scale = guidance_scale
        self.brushnet_conditioning_scale = 1.0
        self.gen_h = gen_h
        self.gen_w = gen_w

        fix_generator = False
        if fix_generator:
            self.generator = torch.Generator("cuda").manual_seed(1024)
            torch.manual_seed(1024)
            torch.cuda.manual_seed_all(1024)
            np.random.seed(1024)
            random.seed(1024)
        else:
            self.generator = None

        self.pipe.use_trt = True
        batch_size = 1
        # model_parent_dir = os.environ.get("AIP_MODEL_PATH", "./models")
        model_parent_dir = os.path.dirname(brushnet_model_path)
        print(" =====> model_parent_dir ", model_parent_dir)
        if self.pipe.use_trt:
            # unet
            self.pipe.unet_trt = TrtExecutor(["unet"])
            engine_root_path = os.path.join(model_parent_dir,"tensorrt/T4/")
            print(" =====> engine_root_path ", engine_root_path)
            self.pipe.unet_trt.loadEngines(engine_root_path)
            _, shared_device_memory = cudart.cudaMalloc(self.pipe.unet_trt.calculateMaxDeviceMemory())
            self.pipe.unet_trt.activateEngines(shared_device_memory)
            self.pipe.unet_trt.loadResources()
            self.pipe.unet_trt.reshape(batch_size = 2 * batch_size)
            self.pipe.unet.cpu()

            # brush net
            self.pipe.brushnet_trt = TrtExecutor(["brushnet"])
            engine_root_path = os.path.join(model_parent_dir,"tensorrt/T4/")
            print(" =====> engine_root_path ", engine_root_path)
            self.pipe.brushnet_trt.loadEngines(engine_root_path)
            _, shared_device_memory = cudart.cudaMalloc(self.pipe.brushnet_trt.calculateMaxDeviceMemory())
            self.pipe.brushnet_trt.activateEngines(shared_device_memory)
            self.pipe.brushnet_trt.loadResources()
            self.pipe.brushnet_trt.reshape(batch_size = 2 * batch_size)
            self.pipe.brushnet.cpu()

            # text encoder 2
            self.pipe.text_encoder_2_trt = TrtExecutor(["text_encoder2"])
            engine_root_path = os.path.join(model_parent_dir,"tensorrt/T4/")
            self.pipe.text_encoder_2_trt.loadEngines(engine_root_path)
            _, shared_device_memory = cudart.cudaMalloc(self.pipe.text_encoder_2_trt.calculateMaxDeviceMemory())
            self.pipe.text_encoder_2_trt.activateEngines(shared_device_memory)
            self.pipe.text_encoder_2_trt.loadResources()
            self.pipe.text_encoder_2_trt.reshape(batch_size = batch_size)
            self.pipe.text_encoder_2.cpu()

            # vae decoder
            # self.pipe.vae_decoder_trt = TrtExecutor(["vae_decoder"])
            # engine_root_path = os.path.join(model_parent_dir,"tensorrt/T4/")
            # self.pipe.vae_decoder_trt.loadEngines(engine_root_path)
            # _, shared_device_memory = cudart.cudaMalloc(self.pipe.vae_decoder_trt.calculateMaxDeviceMemory())
            # self.pipe.vae_decoder_trt.activateEngines(shared_device_memory)
            # self.pipe.vae_decoder_trt.loadResources()
            # self.pipe.vae_decoder_trt.reshape(batch_size = batch_size)
            # self.pipe.vae.decoder.cpu()
            # self.pipe.vae.post_quant_conv.cpu()


    def __call__(self, image, mask, prompt, use_original_input=False,
                 out_h=1024, out_w=1024, bottom=None, template_h=0.8, template_w=0.8, center=None, sharpen=False):

        # make input batchs
        batch_size = 1
        if isinstance(image,list) and isinstance(mask,list) and isinstance(prompt,list):
            batch_size = len(image)
            if len(image) != len(mask) or len(image) != len(prompt):
                raise ValueError(f"got error length of inputs image: f{len(image)} mask: f{len(mask)} prompt: f{len(prompt)}")
        else:
            image   = [image]*batch_size
            mask    = [mask]*batch_size
            prompt  = [prompt]*batch_size

        if not isinstance(out_h,list):
            out_h = [out_h] * batch_size
        if not isinstance(out_w,list):
            out_w = [out_w] * batch_size
        if not isinstance(template_h,list):
            template_h = [template_h] * batch_size
        if not isinstance(template_w,list):
            template_w = [template_w] * batch_size
        if not isinstance(center[0],list):
            center = [center] * batch_size
        if not isinstance(sharpen,list):
            sharpen = [sharpen] * batch_size
        # inputs of inpaint model
        pad_fgs = []
        pad_masks = []
        pad_mask_inputs = []
        crop_fg_finals = []
        crop_mask_finals= []
        target_poss= []
        for i in range(batch_size):
            if not use_original_input:
                default_template_cfg = {
                    'h': int(out_h[i]*template_h[i]),
                    'w': int(out_w[i]*template_w[i]),
                    'fg_erode_iter': 3
                }
                if bottom is not None:
                    default_template_cfg['bottom'] = int(out_h[i] * bottom[i])
                if center is not None:
                    default_template_cfg['center'] = (int(out_w[i]*center[i][0]), int(out_h[i]*center[i][1]))

                pad_fg, pad_mask, crop_fg_final, crop_mask_final, target_pos = pad_resize_v2(
                    image[i], mask[i], default_template_cfg, out_h=out_h[i], out_w=out_w[i], gen_h=self.gen_h, gen_w=self.gen_w)
                pad_fg = Image.fromarray(np.uint8(pad_fg)).convert("RGB")
                pad_mask_input = Image.fromarray(np.uint8(255-pad_mask)).convert("RGB")
                pad_fgs.append(pad_fg)
                pad_masks.append(pad_mask)
                pad_mask_inputs.append(pad_mask_input)
                crop_fg_finals.append(crop_fg_final)
                crop_mask_finals.append(crop_mask_final)
                target_poss.append(target_pos)
            else:
                pad_fg = Image.fromarray(np.uint8(image)).convert("RGB")
                pad_mask = mask.copy()
                pad_mask_input = Image.fromarray(np.uint8(255-mask)).convert("RGB")
                crop_fg_final = image.copy()
                crop_mask_final = mask.copy()
                target_pos[i] = [0, 0, image.shape[1], image.shape[0]]
                out_h[i] = image.shape[0]
                out_w[i] = image.shape[1]
                pad_fgs.append(pad_fg)
                pad_masks.append(pad_mask)
                pad_mask_inputs.append(pad_mask_input)
                crop_fg_finals.append(crop_fg_final)
                crop_mask_finals.append(crop_mask_final)
                target_poss.append(target_pos)

        prompts = [self.prefix + p for p in prompt]

        # inference inpaint model
        image_gen = self.pipe(
            image=pad_fgs,
            mask=pad_mask_inputs,
            prompt=prompts,
            negative_prompt=[self.neg_prompt]*batch_size,
            brushnet_conditioning_scale=self.brushnet_conditioning_scale,
            generator=self.generator,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale
        ).images

        image_addfgs = []
        for i in range(batch_size):
            if sharpen[i]:
                image_addfg = paste_fg_sharpen(np.asarray(image_gen[i]), crop_fg_finals[i], crop_mask_finals[i], target_poss[i],
                                            target_h=out_h[i], target_w=out_w[i], iter=2)
            else:
                image_addfg = paste_fg(np.asarray(image_gen[i]), crop_fg_finals[i], crop_mask_finals[i], target_poss[i],
                                    target_h=out_h[i], target_w=out_w[i], iter=2)
            image_addfgs.append(image_addfg)
        return {
            'gen': [np.asarray(im) for im in image_gen],
            'gen_addfg': image_addfgs,
            'gen_mask': pad_masks
        }
