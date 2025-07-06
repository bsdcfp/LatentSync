import numpy as np
import torch
from diffusers.pipelines.stable_diffusion_safe.safety_checker import (
    SafeStableDiffusionSafetyChecker,
)
from PIL import Image
from transformers import CLIPImageProcessor


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.

    args: images = list(numpy array)
    """
    pil_images = []
    for image in images:
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image, mode="L")
        elif image.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_image = Image.fromarray(image.squeeze(), mode="L")
        else:
            pil_image = Image.fromarray(image)
        pil_images.append(pil_image)
    return pil_images


class SafePipeline():
    def __init__(self, safety_model_path='CompVis/stable-diffusion-safety-checker', device='cuda:0'):
        self.safety_checker = SafeStableDiffusionSafetyChecker.from_pretrained(safety_model_path)
        self.safety_checker = self.safety_checker.to(device)
        self.feature_extractor = CLIPImageProcessor.from_pretrained(safety_model_path, device=device)
        self.device = device
        self.dtype = torch.float32

    def run_safety_checker(self, images):
        """
        args: image = numpy array | list(numpy array)
        """
        if not isinstance(images, list):
            images = list(images)

        if self.safety_checker is not None:
            pil_images = numpy_to_pil(images)
            safety_checker_input = self.feature_extractor(pil_images, return_tensors="pt").to(self.device)
            images, has_nsfw_concept = self.safety_checker(images=images, clip_input=safety_checker_input.pixel_values.to(self.dtype))
            flagged_images = []
            if any(has_nsfw_concept):
                # print("Potential NSFW content was detected in one or more images. A black image will be returned.")
                for idx, has_nsfw in enumerate(has_nsfw_concept):
                    if has_nsfw:
                        flagged_images.append(images[idx])
                        images[idx] = np.zeros(images[idx].shape, dtype=np.uint8)  # black image
                    else:
                        flagged_images.append(None)
        else:
            has_nsfw_concept = [False] * len(images)
            flagged_images = [None] * len(images)

        return images, has_nsfw_concept, flagged_images
