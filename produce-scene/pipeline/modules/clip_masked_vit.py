from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import CLIPVisionModelWithProjection
from transformers.modeling_outputs import ModelOutput
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings, CLIPEncoder
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


@dataclass
class MaskedCLIPVisionModelOutput(ModelOutput):
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    attention_mask: Optional[torch.LongTensor] = None


@dataclass
class MaskedBaseModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    attention_mask: Optional[torch.LongTensor] = None


class MaskedCLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
    
    def encode_mask(self, mask, grid_num):
        mask = F.interpolate(mask, (224, 224))
        mask = F.max_pool2d(mask, 25, 1, 12)
        attn_mask = F.interpolate(mask, (grid_num, grid_num))

        attn_mask = torch.mean(attn_mask, dim=1)
        attn_mask = attn_mask.reshape(-1, grid_num*grid_num)
        
        out_attn_mask = torch.zeros_like(attn_mask)
        out_attn_mask[attn_mask==0] = 1
        cls_token_mask = torch.ones(attn_mask.size(0), 1, dtype=torch.bool, device=attn_mask.device)
        out_attn_mask = torch.cat([cls_token_mask, out_attn_mask], dim=1)

        return out_attn_mask

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, MaskedBaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        if pixel_mask is not None:
            attn_mask = self.encode_mask(pixel_mask, int(pow(hidden_states.shape[1]-1, 0.5)))
        else:
            attn_mask = torch.ones(hidden_states.size(0), hidden_states.size(1), dtype=hidden_states.dtype ,device=hidden_states.device)

        if attn_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            input_attn_mask = _prepare_4d_attention_mask(attn_mask, hidden_states.dtype)
        else:
            input_attn_mask = None

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attention_mask=input_attn_mask
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return MaskedBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            attention_mask=attn_mask
        )


class MaskedCLIPVisionModelWithProjection(CLIPVisionModelWithProjection):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)

        self.vision_model = MaskedCLIPVisionTransformer(config)

        self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, MaskedCLIPVisionModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_mask=pixel_mask
        )

        pooled_output = vision_outputs[1]  # pooled_output

        image_embeds = self.visual_projection(pooled_output)

        if not return_dict:
            outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return MaskedCLIPVisionModelOutput(
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
            attention_mask=vision_outputs.attention_mask
        )
