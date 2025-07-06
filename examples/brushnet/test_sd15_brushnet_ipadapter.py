from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
import torch
import cv2
import numpy as np
from PIL import Image

# choose the base model here
base_model_path = "/home/work/llm/BrushNet/data/ckpt/realisticVisionV60B1_v51VAE"
# base_model_path = "runwayml/stable-diffusion-v1-5"

# input brushnet ckpt path
brushnet_path = "/home/work/llm/BrushNet/data/ckpt/segmentation_mask_brushnet_ckpt"

# choose whether using blended operation
blended = True

# input source image / mask image path and the text prompt
image_path = '/home/work/llm/aigc_image/aigc_productbggen/data/0508_testdata/Beauty/18a0eb25277f73fe70559d018eaa0459.jpg'
mask_path = '/home/work/llm/aigc_image/aigc_productbggen/data/0508_testdata/Beauty/18a0eb25277f73fe70559d018eaa0459_mask.png'
caption="a foreground item stand on the marble surface, empty surface, empty surface. master light, shadow"

# conditioning scale
brushnet_conditioning_scale=1.0
from transformers import CLIPVisionModelWithProjection
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16
)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, image_encoder=image_encoder, torch_dtype=torch.float16, low_cpu_mem_usage=False
)
# pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.safetensors")

pipe.set_ip_adapter_scale(0.4)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

init_image = cv2.imread(image_path)[:,:,::-1]
mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)
mask_image = mask_image[:,:,np.newaxis]

init_image = init_image * mask_image

init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
mask_image = Image.fromarray(255-mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")

from diffusers.utils import load_image
ip_image = load_image('/home/work/llm/aigc_image/aigc_productbggen/data/bg_prompt_240415_filter/室内/室内-小空间-客厅-台子/室内-小空间-客厅-台子-1_image2image_text_31.png')

generator = torch.Generator("cuda").manual_seed(1234)

image = pipe(
    caption, 
    init_image, 
    mask_image, 
    num_inference_steps=50, 
    generator=generator,
    brushnet_conditioning_scale=brushnet_conditioning_scale,
    ip_adapter_image=ip_image
).images[0]

if blended:
    image_np=np.array(image)
    init_image_np=cv2.imread(image_path)[:,:,::-1]
    mask_np = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
    mask_np = 1- mask_np
    # blur, you can adjust the parameters for better performance
    mask_blurred = cv2.GaussianBlur(mask_np*255, (21, 21), 0)/255
    mask_blurred = mask_blurred[:,:,np.newaxis]
    mask_np = 1-(1-mask_np) * (1-mask_blurred)

    image_pasted=init_image_np * (1-mask_np) + image_np*mask_np
    image_pasted=image_pasted.astype(image_np.dtype)
    image=Image.fromarray(image_pasted)

image.save("runs/pipeline/output.png")