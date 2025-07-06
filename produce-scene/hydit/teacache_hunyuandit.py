from typing import Any, Dict, Optional, Tuple, Union
# from diffusers import DiffusionPipeline
from .diffusion.pipeline_controlnet2 import StableDiffusionControlNetPipeline
# from diffusers.models import FluxTransformer2DModel
from .modules.models_reference3 import HunYuanDiT, HUNYUAN_DIT_MODELS, HunYuanReferenceNet, HUNYUAN_DIT_CONFIG
from .modules.embedders import timestep_embedding

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
import torch
import numpy as np
from datetime import datetime
import time

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def teacache_hunyuandit_forward(self,
                x,
                t,
                encoder_hidden_states=None,
                text_embedding_mask=None,
                encoder_hidden_states_t5=None,
                text_embedding_mask_t5=None,
                image_meta_size=None,
                style=None,
                cos_cis_img=None,
                sin_cis_img=None,
                return_dict=True,
                controls=None,
                ):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x: torch.Tensor
            (B, D, H, W)
        t: torch.Tensor
            (B)
        encoder_hidden_states: torch.Tensor
            CLIP text embedding, (B, L_clip, D)
        text_embedding_mask: torch.Tensor
            CLIP text embedding mask, (B, L_clip)
        encoder_hidden_states_t5: torch.Tensor
            T5 text embedding, (B, L_t5, D)
        text_embedding_mask_t5: torch.Tensor
            T5 text embedding mask, (B, L_t5)
        image_meta_size: torch.Tensor
            (B, 6)
        style: torch.Tensor
            (B)
        cos_cis_img: torch.Tensor
        sin_cis_img: torch.Tensor
        return_dict: bool
            Whether to return a dictionary.
        """
        text_states = encoder_hidden_states                     # Already CFG processed in pipeline
        text_states_t5 = encoder_hidden_states_t5               # Already CFG processed in pipeline
        text_states_mask = text_embedding_mask.bool()           # Already CFG processed in pipeline
        text_states_t5_mask = text_embedding_mask_t5.bool()     # Already CFG processed in pipeline
        
        b_t5, l_t5, c_t5 = text_states_t5.shape
        text_states_t5 = self.mlp_t5(text_states_t5.view(-1, c_t5))
        text_states = torch.cat([text_states, text_states_t5.view(b_t5, l_t5, -1)], dim=1)  
        clip_t5_mask = torch.cat([text_states_mask, text_states_t5_mask], dim=-1)

        clip_t5_mask = clip_t5_mask
        text_states = torch.where(clip_t5_mask.unsqueeze(2), text_states, self.text_embedding_padding.to(text_states))

        _, _, oh, ow = x.shape
        th, tw = oh // self.patch_size, ow // self.patch_size

        # ========================= Build time and image embedding =========================
        t = self.t_embedder(t)
        x = self.x_embedder(x)
        # Get image RoPE embedding according to `reso`lution.
        freqs_cis_img = (cos_cis_img, sin_cis_img)

        # ========================= Concatenate all extra vectors =========================
        # Build text tokens with pooling
        extra_vec = self.pooler(encoder_hidden_states_t5)

        if self.args.size_cond == None:
            image_meta_size = None
        self.check_condition_validation(image_meta_size, style)
        # Build image meta size tokens if applicable
        if image_meta_size is not None:
            image_meta_size = timestep_embedding(image_meta_size.view(-1), 256)   # [B * 6, 256]
            if self.args.use_fp16:
                image_meta_size = image_meta_size.half()
            image_meta_size = image_meta_size.view(-1, 6 * 256)
            extra_vec = torch.cat([extra_vec, image_meta_size], dim=1)  # [B, D + 6 * 256]

        # Build style tokens
        if style is not None:
            style_embedding = self.style_embedder(style)
            extra_vec = torch.cat([extra_vec, style_embedding], dim=1)

        # Concatenate all extra vectors
        c = t + self.extra_embedder(extra_vec)  # [B, D]

        # ========================= TeaCache Logic =========================
        if self.enable_teacache:
            # Create a copy of input for comparison
            inp = x.clone()
            c_ = c.clone()
            
            # Get modulated input from first block's norm1 for comparison
            # This mimics the self-attention input preparation in HunYuanDiTBlock._forward
            shift_msa = self.blocks[0].default_modulation(c_).unsqueeze(dim=1)
            modulated_inp = self.blocks[0].norm1(inp) + shift_msa
            
            if self.cnt < 5 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else: 
                coefficients = [7.33226126e+02, -4.01131952e+02,  6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = modulated_inp 
            self.cnt += 1 
            if self.cnt == self.num_steps:
                self.cnt = 0

        # ========================= Forward pass through HunYuanDiT blocks =========================
        if self.enable_teacache:
            if not should_calc:
                # Use cached residual
                x += self.previous_residual
                # # Still need to apply final layer and unpatchify to maintain correct dimensions
                # x = self.final_layer(x, c)                              # (N, L, patch_size ** 2 * out_channels)
                # x = self.unpatchify(x, th, tw)                          # (N, out_channels, H, W)
            else:
                # Full computation - store the original input before any processing
                ori_x = x.clone()
                skips = []
                for layer, block in enumerate(self.blocks):
                    if controls is not None:
                        x = x + controls.pop(0)
                    if layer > self.depth // 2:
                        skip = skips.pop()
                        x = block(x, c, text_states, freqs_cis_img, skip)   # (N, L, D)
                    else:
                        x = block(x, c, text_states, freqs_cis_img)         # (N, L, D)
                    if layer < (self.depth // 2 - 1):
                        skips.append(x)
                if controls is not None and len(controls) != 0:
                    raise ValueError("The number of controls is not equal to the number of skip connections.")

                # Cache the residual BEFORE final layer and unpatchify (which change dimensions)
                self.previous_residual = x - ori_x
                
                # # ========================= Final layer =========================
                # x = self.final_layer(x, c)                              # (N, L, patch_size ** 2 * out_channels)
                # x = self.unpatchify(x, th, tw)                          # (N, out_channels, H, W)
        else:
            # Original computation without caching
            skips = []
            for layer, block in enumerate(self.blocks):
                if controls is not None:
                    x = x + controls.pop(0)
                if layer > self.depth // 2:
                    skip = skips.pop()
                    x = block(x, c, text_states, freqs_cis_img, skip)   # (N, L, D)
                else:
                    x = block(x, c, text_states, freqs_cis_img)         # (N, L, D)
                if layer < (self.depth // 2 - 1):
                    skips.append(x)
            if controls is not None and len(controls) != 0:
                raise ValueError("The number of controls is not equal to the number of skip connections.")

        # ========================= Final layer =========================
        x = self.final_layer(x, c)                              # (N, L, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, th, tw)                          # (N, out_channels, H, W)

        if return_dict:
            return {'x': x}
        return x

# HunYuanDiT.forward = teacache_hunyuandit_forward
# num_inference_steps = 18
# seed = 42
# prompt = "An image of a squirrel in Picasso style"
# # model_path = "black-forest-labs/FLUX.1-dev"
# model_path = "/model_zoo/hunyuanDiT/"
# pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
# pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
# # pipeline.enable_sequential_cpu_offload()
# # pipeline.enable_vae_slicing()

# # TeaCache
# pipeline.transformer.__class__.enable_teacache = True
# pipeline.transformer.__class__.cnt = 0
# pipeline.transformer.__class__.num_steps = num_inference_steps
# pipeline.transformer.__class__.rel_l1_thresh = 0.6 # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
# pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
# pipeline.transformer.__class__.previous_modulated_input = None
# pipeline.transformer.__class__.previous_residual = None


# # pipeline.to("cuda")
# start_time = time.time()
# img = pipeline(
#     prompt, 
#     num_inference_steps=num_inference_steps,
#     generator=torch.Generator("cpu").manual_seed(seed)
#     ).images[0]
# infer_time = time.time() - start_time
# format_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# img.save("{}.png".format('TeaCache_' + prompt + '_' + format_time))
# print(f"Inference time: {infer_time} seconds")