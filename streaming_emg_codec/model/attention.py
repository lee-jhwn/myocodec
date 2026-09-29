from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_with_kvcache

    FLASH_ATTN_AVAILABLE = True
except Exception:
    flash_attn_qkvpacked_func = None
    flash_attn_with_kvcache = None
    FLASH_ATTN_AVAILABLE = False


SDPA_AVAILABLE = hasattr(F, "scaled_dot_product_attention")


def _require_flash_attention(x: torch.Tensor) -> None:
    if x.device.type == "cuda" and not (FLASH_ATTN_AVAILABLE or SDPA_AVAILABLE):
        raise RuntimeError(
            "CUDA execution requires flash-attn or torch>=2.0 SDPA. Install flash-attn "
            "with: pip install flash-attn --no-build-isolation"
        )


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, qkv: torch.Tensor, seqlen_offset: int = 0) -> torch.Tensor:
        seqlen = qkv.shape[1]
        positions = torch.arange(
            seqlen_offset,
            seqlen_offset + seqlen,
            device=qkv.device,
            dtype=torch.float32,
        )
        freqs = torch.outer(positions, self.inv_freq.to(device=qkv.device))
        cos = freqs.cos().to(dtype=qkv.dtype).view(1, seqlen, 1, 1, -1)
        sin = freqs.sin().to(dtype=qkv.dtype).view(1, seqlen, 1, 1, -1)
        first = qkv[..., 0::2]
        second = qkv[..., 1::2]
        rotated = torch.empty_like(qkv)
        rotated[..., 0::2] = first * cos - second * sin
        rotated[..., 1::2] = first * sin + second * cos
        return rotated


@dataclass
class InferenceCache:
    batch_size: int
    max_seqlen: int
    n_heads: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self) -> None:
        self._alloc(self.max_seqlen)
        self.seqlen_offset = 0
        self.filled = 0

    def _alloc(self, size: int) -> None:
        self.kv_cache = torch.empty(
            self.batch_size,
            size,
            2,
            self.n_heads,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.max_seqlen = size

    def reserve(self, seqlen: int, window_size: int = 0) -> None:
        """Make room for ``seqlen`` new entries."""
        if window_size <= 0:
            required = self.filled + seqlen
            if required <= self.max_seqlen:
                return
            new_max = max(self.max_seqlen, 1)
            while new_max < required:
                new_max *= 2
            old, filled = self.kv_cache, self.filled
            self._alloc(new_max)
            self.kv_cache[:, :filled] = old[:, :filled]
            return

        headroom = max(seqlen, window_size)
        capacity = window_size + headroom
        if self.max_seqlen < capacity:
            old, filled = self.kv_cache, self.filled
            self._alloc(capacity)
            keep = min(filled, capacity)
            if keep:
                self.kv_cache[:, :keep] = old[:, filled - keep : filled]
            self.filled = keep
        if self.filled + seqlen > self.max_seqlen:
            drop = self.filled + seqlen - self.max_seqlen
            keep = self.filled - drop
            self.kv_cache[:, :keep] = self.kv_cache[:, drop : self.filled].clone()
            self.filled = keep

    def ensure_capacity(self, additional: int) -> None:
        self.reserve(additional, 0)

    def update_position(self, seqlen: int) -> None:
        self.seqlen_offset += seqlen
        self.filled = min(self.filled + seqlen, self.max_seqlen)

    def reset(self) -> None:
        self.seqlen_offset = 0
        self.filled = 0


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        head_dim: int,
        dropout_rate: float,
        window_size: int,
        rotary_base: float,
    ):
        super().__init__()
        if n_embd != n_heads * head_dim:
            raise ValueError(f"n_embd must equal n_heads * head_dim, got {n_embd}")
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate
        self.window_size = window_size
        self.softmax_scale = 1.0 / math.sqrt(head_dim)
        self.qkv = nn.Linear(n_embd, 3 * n_heads * head_dim)
        self.out = nn.Linear(n_embd, n_embd)
        self.rotary = RotaryEmbedding(head_dim, rotary_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_flash_attention(x)
        batch, seqlen, _ = x.shape
        original_dtype = x.dtype
        qkv = self.qkv(x).view(batch, seqlen, 3, self.n_heads, self.head_dim)
        qkv = self.rotary(qkv, seqlen_offset=0)
        if x.device.type == "cuda":
            qkv = qkv.to(torch.bfloat16)
            if FLASH_ATTN_AVAILABLE:
                context = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.dropout_rate if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=True,
                    window_size=(self.window_size, 0) if self.window_size > 0 else (-1, 0),
                )
            else:
                context = self._sdpa_attention(qkv, seqlen_offset=0)
        else:
            context = self._torch_attention(qkv, seqlen_offset=0)
        context = context.contiguous().view(batch, seqlen, self.n_embd).to(original_dtype)
        return self.out(context)

    def decode(self, x: torch.Tensor, cache: InferenceCache) -> torch.Tensor:
        _require_flash_attention(x)
        batch, seqlen, _ = x.shape
        if batch != cache.batch_size:
            raise ValueError(f"Cache batch size {cache.batch_size} does not match input batch {batch}")
        cache.reserve(seqlen, self.window_size)
        original_dtype = x.dtype
        qkv = self.qkv(x).view(batch, seqlen, 3, self.n_heads, self.head_dim)
        qkv = self.rotary(qkv, seqlen_offset=cache.seqlen_offset)
        qkv = qkv.to(cache.dtype)

        if x.device.type == "cuda":
            if cache.seqlen_offset == 0:
                if FLASH_ATTN_AVAILABLE:
                    context = flash_attn_qkvpacked_func(
                        qkv,
                        dropout_p=0.0,
                        softmax_scale=self.softmax_scale,
                        causal=True,
                        window_size=(self.window_size, 0) if self.window_size > 0 else (-1, 0),
                    )
                else:
                    context = self._sdpa_attention(qkv, seqlen_offset=0)
                cache.kv_cache[:, cache.filled : cache.filled + seqlen] = qkv[:, :, 1:3]
            elif not FLASH_ATTN_AVAILABLE:
                cache.kv_cache[:, cache.filled : cache.filled + seqlen] = qkv[:, :, 1:3]
                context = self._sdpa_attention_with_cache(qkv[:, :, 0], cache, seqlen)
            else:
                context = flash_attn_with_kvcache(
                    qkv[:, :, 0].contiguous(),
                    cache.kv_cache[:, :, 0],
                    cache.kv_cache[:, :, 1],
                    k=qkv[:, :, 1].contiguous(),
                    v=qkv[:, :, 2].contiguous(),
                    cache_seqlens=cache.filled,
                    softmax_scale=self.softmax_scale,
                    causal=True,
                    window_size=(self.window_size, 0) if self.window_size > 0 else (-1, 0),
                )
        else:
            cache.kv_cache[:, cache.filled : cache.filled + seqlen] = qkv[:, :, 1:3]
            context = self._torch_attention_with_cache(qkv[:, :, 0], cache, seqlen)

        cache.update_position(seqlen)
        context = context.contiguous().view(batch, seqlen, self.n_embd).to(original_dtype)
        return self.out(context)

    def _window_mask(
        self, q_pos: torch.Tensor, k_pos: torch.Tensor
    ) -> torch.Tensor:
        mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        if self.window_size > 0:
            mask = mask & ((q_pos.unsqueeze(1) - k_pos.unsqueeze(0)) <= self.window_size)
        return mask

    def _sdpa_attention(self, qkv: torch.Tensor, seqlen_offset: int) -> torch.Tensor:
        # qkv: [B, T, 3, H, D] -> context [B, T, H, D]
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        seqlen = q.shape[1]
        pos = torch.arange(seqlen_offset, seqlen_offset + seqlen, device=q.device)
        mask = self._window_mask(pos, pos).view(1, 1, seqlen, seqlen)
        context = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
            dropout_p=self.dropout_rate if self.training else 0.0,
            scale=self.softmax_scale,
        )
        return context.transpose(1, 2)

    def _sdpa_attention_with_cache(
        self, q: torch.Tensor, cache: InferenceCache, seqlen: int
    ) -> torch.Tensor:
        end = cache.filled + seqlen
        if self.window_size > 0:
            start = max(0, end - seqlen - self.window_size)
        else:
            start = 0
        k = cache.kv_cache[:, start:end, 0]
        v = cache.kv_cache[:, start:end, 1]
        q_pos = torch.arange(end - seqlen, end, device=q.device)
        k_pos = torch.arange(start, end, device=q.device)
        mask = self._window_mask(q_pos, k_pos).view(1, 1, seqlen, end - start)
        context = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
            dropout_p=0.0,
            scale=self.softmax_scale,
        )
        return context.transpose(1, 2)

    def _torch_attention(self, qkv: torch.Tensor, seqlen_offset: int) -> torch.Tensor:
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        scores = torch.einsum("bthd,bshd->btsh", q, k) * self.softmax_scale
        t = torch.arange(seqlen_offset, seqlen_offset + q.shape[1], device=q.device)
        s = torch.arange(seqlen_offset, seqlen_offset + k.shape[1], device=q.device)
        mask = s.unsqueeze(0) <= t.unsqueeze(1)
        if self.window_size > 0:
            mask = mask & ((t.unsqueeze(1) - s.unsqueeze(0)) <= self.window_size)
        scores = scores.masked_fill(~mask.view(1, q.shape[1], k.shape[1], 1), float("-inf"))
        weights = F.softmax(scores, dim=2)
        return torch.einsum("btsh,bshd->bthd", weights, v)

    def _torch_attention_with_cache(
        self,
        q: torch.Tensor,
        cache: InferenceCache,
        seqlen: int,
    ) -> torch.Tensor:
        end = cache.filled + seqlen
        if self.window_size > 0:
            start = max(0, end - seqlen - self.window_size)
        else:
            start = 0
        k = cache.kv_cache[:, start:end, 0].contiguous()
        v = cache.kv_cache[:, start:end, 1].contiguous()
        scores = torch.einsum("bthd,bshd->btsh", q, k) * self.softmax_scale
        t = torch.arange(end - seqlen, end, device=q.device)
        s = torch.arange(start, end, device=q.device)
        mask = s.unsqueeze(0) <= t.unsqueeze(1)
        if self.window_size > 0:
            mask = mask & ((t.unsqueeze(1) - s.unsqueeze(0)) <= self.window_size)
        scores = scores.masked_fill(~mask.view(1, seqlen, k.shape[1], 1), float("-inf"))
        weights = F.softmax(scores, dim=2)
        return torch.einsum("btsh,bshd->bthd", weights, v)


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(
            n_embd=config.n_embd,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            dropout_rate=config.dropout_rate,
            window_size=config.window_size,
            rotary_base=config.rotary_base,
        )
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, config.n_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.n_hidden, config.n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def decode(self, x: torch.Tensor, cache: InferenceCache) -> torch.Tensor:
        x = x + self.attn.decode(self.norm1(x), cache)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.n_embd)
        self.default_cache_length = max(2048, config.window_size * 2 if config.window_size > 0 else 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def allocate_inference_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
        length: int | None = None,
    ) -> list[InferenceCache]:
        if dtype is None:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        length = length or self.default_cache_length
        return [
            InferenceCache(
                batch_size=batch_size,
                max_seqlen=length,
                n_heads=self.config.n_heads,
                head_dim=self.config.head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in self.layers
        ]

    def decode(self, x: torch.Tensor, inference_cache: list[InferenceCache] | None = None):
        if inference_cache is None:
            inference_cache = self.allocate_inference_cache(x.shape[0], x.device)
        if len(inference_cache) != len(self.layers):
            raise ValueError("Inference cache must contain one cache per Transformer layer")
        for layer, cache in zip(self.layers, inference_cache, strict=True):
            x = layer.decode(x, cache)
        return self.norm(x), inference_cache
