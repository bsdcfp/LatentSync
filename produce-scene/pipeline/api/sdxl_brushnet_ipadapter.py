import numpy as np
from PIL import Image
import torch
import os
import time
from diffusers import BrushNetModel, DPMSolverMultistepScheduler, AutoencoderKL
from ..modules.clip_masked_vit import MaskedCLIPVisionModelWithProjection
from ..modules.pipeline_brushnet_sd_xl_mask import MaskedStableDiffusionXLBrushNetPipeline
from ..utils.pre_process import pad_resize_v2
from ..utils.post_process import paste_fg
from ...trt_backend import TrtExecutor
from cuda import cudart
import random

class SDXLBrushNetIpadapterRunner():
    def __init__(self, base_model_path, brushnet_model_path):
        image_encoder = MaskedCLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16
        )
        model_parent_dir = os.environ.get("AIP_MODEL_PATH", "./models")
        print(" =====> model_parent_dir ", model_parent_dir)
        brushnet = BrushNetModel.from_pretrained(brushnet_model_path, torch_dtype=torch.float16)
        self.pipe = MaskedStableDiffusionXLBrushNetPipeline.from_pretrained(
            base_model_path, image_encoder=image_encoder, brushnet=brushnet,
            torch_dtype=torch.float16, low_cpu_mem_usage=False, use_safetensors=True
        )
        self.pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,attn_implementation="sdpa")
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.encode_image_stream = torch.cuda.Stream()
        self.pipe.use_trt = False
        fix_generator = True
        if fix_generator:
            self.generator = torch.Generator("cuda").manual_seed(1024)
            torch.manual_seed(1024)
            torch.cuda.manual_seed_all(1024)
            np.random.seed(1024)
            random.seed(1024)
        else:
            self.generator = None

        if self.pipe.use_trt:
            # unet
            self.pipe.unet_trt = TrtExecutor(["unet"])
            engine_root_path = os.path.join(model_parent_dir,"tensorrt/T4/")
            print(" =====> engine_root_path ", engine_root_path)
            self.pipe.unet_trt.loadEngines(engine_root_path)
            _, shared_device_memory = cudart.cudaMalloc(self.pipe.unet_trt.calculateMaxDeviceMemory())
            self.pipe.unet_trt.activateEngines(shared_device_memory)
            self.pipe.unet_trt.loadResources()
            self.pipe.unet.cpu()

            # brush net
            self.pipe.brushnet_trt = TrtExecutor(["brushnet"])
            engine_root_path = os.path.join(model_parent_dir,"tensorrt/T4/")
            print(" =====> engine_root_path ", engine_root_path)
            self.pipe.brushnet_trt.loadEngines(engine_root_path)
            _, shared_device_memory = cudart.cudaMalloc(self.pipe.brushnet_trt.calculateMaxDeviceMemory())
            self.pipe.brushnet_trt.activateEngines(shared_device_memory)
            self.pipe.brushnet_trt.loadResources()
            self.pipe.brushnet_trt.reshape(batch_size=2)
            self.pipe.brushnet.cpu()

            self.pipe.text_encoder_2_trt = TrtExecutor(["text_encoder2"])
            engine_root_path = "models/tensorrt/T4/"
            self.pipe.text_encoder_2_trt.loadEngines(engine_root_path)
            _, shared_device_memory = cudart.cudaMalloc(self.pipe.text_encoder_2_trt.calculateMaxDeviceMemory())
            self.pipe.text_encoder_2_trt.activateEngines(shared_device_memory)
            self.pipe.text_encoder_2_trt.loadResources()
            self.pipe.text_encoder_2_trt.reshape(batch_size=1)
            self.pipe.text_encoder_2.cpu()
        else:
            pass

        self.pipe.time_list = []
        self.pipe.enable_model_cpu_offload()
        self.suffix = ", photography, ultra highres, best quality, high quality"
        self.neg_prompt = "human, suspended objects, incomplete table, floating table, shining light, clear reflection, blurred, cartoon plant, sofa, complicated, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
        self.num_inference_steps = 30
        self.pipe.step_to_skip_cfg = 30
        self.brushnet_conditioning_scale = 1.0

    def __call__(self, image, mask, ip_image, ip_image_mask, prompt, ip_adapter_scale,
                 out_h=1080, out_w=1080, bottom=None, template_h=0.8, template_w=0.8, center=None):
        default_template_cfg = {
            'h': int(out_h*template_h),
            'w': int(out_w*template_w),
            'fg_erode_iter': 3
        }
        if bottom is not None:
            default_template_cfg['bottom'] = int(out_h * bottom)
        if center is not None:
            default_template_cfg['center'] = (int(out_w*center[0]), int(out_h*center[1]))
        t0 = time.time()
        pad_fg, pad_mask, crop_fg_final, crop_mask_final, target_pos = pad_resize_v2(
            image, mask, default_template_cfg, out_h=out_h, out_w=out_w)
        pad_fg = Image.fromarray(np.uint8(pad_fg)).convert("RGB")
        pad_mask = Image.fromarray(np.uint8(255-pad_mask)).convert("RGB")

        ip_adapter_scale_tensor = torch.FloatTensor([ip_adapter_scale])

        ip_image = Image.fromarray(np.uint8(ip_image)).convert("RGB")
        ip_image_mask = Image.fromarray(np.uint8(ip_image_mask)).convert("RGB")

        prompt = prompt + self.suffix
        t1 = time.time()
        print(" preprocess pipeline time ", (t1 - t0) * 1000)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        image_gen = self.pipe(
            image=pad_fg,
            mask=pad_mask,
            ip_adapter_image=ip_image,
            ip_adapter_image_mask=ip_image_mask,
            prompt=prompt,
            negative_prompt=self.neg_prompt,
            brushnet_conditioning_scale=self.brushnet_conditioning_scale,
            generator=self.generator,
            num_inference_steps=self.num_inference_steps,
            ip_adapter_scale=ip_adapter_scale_tensor
        ).images[0]

        end.record()
        torch.cuda.synchronize()
        print("[INFO] pipeline time {:.2f}".format(start.elapsed_time(end)), "ms")
        t0 = time.time()
        image_addfg = paste_fg(np.asarray(image_gen), crop_fg_final, crop_mask_final, target_pos,
                               target_h=out_h, target_w=out_w, iter=2)
        t1 = time.time()
        print(" post result time ", (t1 - t0) * 1000)
        return np.asarray(image_gen), image_addfg
