from latentsync.models.unet import UNet3DConditionModel
from scripts.inference_online_0618 import _compute_unet_blocks
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL

torch_inference_modes = ["default", "reduce-overhead", "max-autotune"]


# FIXME update callsites after serialization support for torch.compile is added
def optimize_checkpoint(model, torch_inference):
    if not torch_inference or torch_inference == "eager":
        return model
    assert torch_inference in torch_inference_modes
    return torch.compile(model, mode=torch_inference, dynamic=False, fullgraph=False)


class CacheBlock(torch.nn.Module):
    def __init__(self, down_blocks, mid_block, up_blocks):
        super(CacheBlock, self).__init__()
        self.down_blocks = down_blocks
        self.mid_block = mid_block
        self.up_blocks = up_blocks

    def forward(
        self,
        sample,
        emb,
        encoder_hidden_states,
        attention_mask,
        down_block_additional_residuals,
        mid_block_additional_residual,
        forward_upsample_size,
    ):
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            down_block_res_samples += res_samples

        # 支持ControlNet
        down_block_res_samples = list(down_block_res_samples)
        if down_block_additional_residuals is not None:
            for i, down_block_additional_residual in enumerate(
                down_block_additional_residuals
            ):
                if down_block_additional_residual.dim() == 4:  # 广播
                    down_block_additional_residual = (
                        down_block_additional_residual.unsqueeze(2)
                    )
                down_block_res_samples[i] = (
                    down_block_res_samples[i] + down_block_additional_residual
                )

        # Mid block
        sample = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )

        # 支持ControlNet
        if mid_block_additional_residual is not None:
            if mid_block_additional_residual.dim() == 4:  # 广播
                mid_block_additional_residual = mid_block_additional_residual.unsqueeze(
                    2
                )
            sample = sample + mid_block_additional_residual

        # Up blocks
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            else:
                upsample_size = None

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    encoder_hidden_states=encoder_hidden_states,
                )

        return sample


def get_unet_model(model_config, model_path, torch_inference=""):
    config = OmegaConf.load(model_config)
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        model_path,  # load checkpoint
    )
    unet = unet.to(dtype=torch.float16, device="cuda")

    cache_model = CacheBlock(unet.down_blocks, unet.mid_block, unet.up_blocks)

    class UNETPS(torch.nn.Module):
        @torch.no_grad()
        def forward(self, sample, emb, encoder_hidden_states):
            return self.model(
                sample=sample,
                emb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                down_block_additional_residuals=None,
                mid_block_additional_residual=None,
                forward_upsample_size=False,
            )

    cache_model = optimize_checkpoint(cache_model, torch_inference)
    _unet = UNETPS()
    _unet.model = cache_model
    return _unet


def get_vae_decode_model(model_path, torch_inference=""):
    vae = AutoencoderKL.from_pretrained(
        model_path, device="cuda", torch_dtype=torch.float16
    )

    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    vae.eval()

    vae = optimize_checkpoint(vae, torch_inference)

    class VaeDecode(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        @torch.no_grad()
        def forward(self, latents):
            return self.vae.decode(latents, return_dict=False)

    _vae = VaeDecode(vae)

    return _vae

def get_vae_encode_model(model_path, torch_inference=""):
    vae = AutoencoderKL.from_pretrained(
        model_path, device="cuda", torch_dtype=torch.float16
    )

    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    vae.eval()

    vae = optimize_checkpoint(vae, torch_inference)

    class VaeEncode(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        @torch.no_grad()
        def forward(self, images):
            return self.vae.encode(images).latent_dist.sample(generator=None)

    _vae = VaeEncode(vae)

    return _vae

