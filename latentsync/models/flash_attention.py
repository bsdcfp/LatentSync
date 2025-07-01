# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from einops import rearrange
from torch.nn import functional as F

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

__all__ = [
    'flash_attention',
]


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, attention_mask=None):
    # if FLASH_ATTN_3_AVAILABLE:
    #     q = rearrange(q, "(b n) s d -> b s n d", n=num_heads)
    #     k = rearrange(k, "(b n) s d -> b s n d", n=num_heads)
    #     v = rearrange(v, "(b n) s d -> b s n d", n=num_heads)
    #     
    #     # Flash Attention 3 的 attention_mask 处理
    #     if attention_mask is not None:
    #         # attention_mask 从 [B*N, L, L] 转换为 [B, L, L] (广播到所有头)
    #         attention_mask = rearrange(attention_mask, "(b n) s1 s2 -> b s1 s2", n=num_heads)
    #         # Flash Attention 3 支持 attn_bias
    #         x = flash_attn_interface.flash_attn_func(q, k, v, attn_bias=attention_mask)
    #     else:
    #         x = flash_attn_interface.flash_attn_func(q, k, v)
    #     
    #     x = rearrange(x, "b s n d -> (b n) s d", n=num_heads)
    if FLASH_ATTN_2_AVAILABLE:
        # print("==> FLASH_ATTN_2_AVAILABLE")
        q = rearrange(q, "(b n) s d -> b s n d", n=num_heads)
        k = rearrange(k, "(b n) s d -> b s n d", n=num_heads)
        v = rearrange(v, "(b n) s d -> b s n d", n=num_heads)
        
        # Flash Attention 2 的 attention_mask 处理
        if attention_mask is not None:
            # attention_mask 从 [B*N, L, L] 转换为 [B, L, L] (广播到所有头)
            attention_mask = rearrange(attention_mask, "(b n) s1 s2 -> b s1 s2", n=num_heads)
            # Flash Attention 2 期望 causal mask 或 None
            # 如果 attention_mask 不是 causal，可能需要特殊处理
            if attention_mask.shape[1] == attention_mask.shape[2]:  # 方阵
                # 检查是否为 causal mask
                is_causal = torch.allclose(
                    attention_mask, 
                    torch.triu(attention_mask, diagonal=1), 
                    atol=1e-6
                )
                if is_causal:
                    x = flash_attn.flash_attn_func(q, k, v, causal=True)
                else:
                    # 非 causal mask，使用 attn_bias
                    x = flash_attn.flash_attn_func(q, k, v, attn_bias=attention_mask)
            else:
                x = flash_attn.flash_attn_func(q, k, v)
        else:
            x = flash_attn.flash_attn_func(q, k, v)
        
        x = rearrange(x, "b s n d -> (b n) s d", n=num_heads)
    else:
        # 检查输入格式
        if q.shape[0] % num_heads == 0:
            # 输入已经是 [B*N, L, D] 格式
            batch_size = q.shape[0] // num_heads
            q = rearrange(q, "(b n) s d -> b n s d", n=num_heads)
            k = rearrange(k, "(b n) s d -> b n s d", n=num_heads)
            v = rearrange(v, "(b n) s d -> b n s d", n=num_heads)
            if attention_mask is not None:
                attention_mask = rearrange(attention_mask, "(b n) s1 s2 -> b n s1 s2", n=num_heads)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
            x = rearrange(x, "b n s d -> (b n) s d", n=num_heads)
        else:
            # 输入是 [B, L, C] 格式，需要先转换为 [B*N, L, D]
            # 这里应该不会到达，因为 CrossAttention 已经处理了格式转换
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
    
    return x

