from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from streaming_emg_codec.config import ModelConfig
from streaming_emg_codec.model.attention import InferenceCache, TransformerStack
from streaming_emg_codec.model.rvq import ResidualVectorQuantizer


@dataclass
class StreamingState:
    encoder_cache: list[InferenceCache] | None = None
    decoder_cache: list[InferenceCache] | None = None

    def reset(self) -> None:
        for cache_list in (self.encoder_cache, self.decoder_cache):
            if cache_list is not None:
                for cache in cache_list:
                    cache.reset()


class StreamingEMGCodec(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate
        self.frame_size = config.frame_size
        self.linear_in = nn.Linear(config.frame_size, config.encoder.n_embd)
        self.encoder = TransformerStack(config.encoder)
        self.rvq_in = nn.Linear(config.encoder.n_embd, config.rvq.embedding_dim)
        self.rvq = ResidualVectorQuantizer(config.rvq)
        self.rvq_out = nn.Linear(config.rvq.embedding_dim, config.decoder.n_embd)
        self.decoder = TransformerStack(config.decoder)
        self.linear_out = nn.Linear(config.decoder.n_embd, config.frame_size)

    def forward(self, x: torch.Tensor, n_codebooks: torch.Tensor | int | None = None):
        framed, shape = self._frame(x)
        n_codebooks = self._expand_n_codebooks(n_codebooks, shape)
        encoded = self.rvq_in(self.encoder(self.linear_in(framed)))
        quantized, indices, vq_loss, commit_loss = self.rvq(encoded, n_codebooks)
        decoded = self.linear_out(self.decoder(self.rvq_out(quantized)))
        reconstruction = self._unframe(decoded, shape, crop_to=x.shape[-1])
        indices = self._restore_indices(indices, shape)
        return {
            "reconstruction": reconstruction,
            "indices": indices,
            "vq_loss": vq_loss,
            "commit_loss": commit_loss,
        }

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        n_codebooks: torch.Tensor | int | None = None,
        inference_cache: list[InferenceCache] | None = None,
    ):
        framed, shape = self._frame(x)
        n_codebooks = self._expand_n_codebooks(n_codebooks, shape)
        hidden, inference_cache = self.encoder.decode(self.linear_in(framed), inference_cache)
        indices = self.rvq.encode(self.rvq_in(hidden), n_codebooks)
        return self._restore_indices(indices, shape), inference_cache

    @torch.no_grad()
    def decode(
        self,
        indices: torch.Tensor,
        n_codebooks: torch.Tensor | int | None = None,
        inference_cache: list[InferenceCache] | None = None,
    ):
        flat_indices, shape = self._flatten_indices(indices)
        n_codebooks = self._expand_n_codebooks(n_codebooks, shape)
        quantized = self.rvq.decode(flat_indices, n_codebooks)
        hidden, inference_cache = self.decoder.decode(self.rvq_out(quantized), inference_cache)
        decoded = self.linear_out(hidden)
        return self._unframe(decoded, shape), inference_cache

    @torch.no_grad()
    def streaming_step(
        self,
        chunk: torch.Tensor,
        state: StreamingState | None = None,
        n_codebooks: torch.Tensor | int | None = None,
    ):
        if chunk.shape[-1] != self.frame_size:
            raise ValueError(f"Streaming chunks must be exactly {self.frame_size} samples")
        state = state or StreamingState()
        indices, state.encoder_cache = self.encode(chunk, n_codebooks, state.encoder_cache)
        reconstruction, state.decoder_cache = self.decode(indices, n_codebooks, state.decoder_cache)
        return reconstruction, indices, state

    def _frame(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, channels, time], got shape {tuple(x.shape)}")
        batch, channels, time = x.shape
        if self.config.channels is not None and channels != self.config.channels:
            raise ValueError(f"Expected {self.config.channels} channels, got {channels}")
        pad = (-time) % self.frame_size
        if pad:
            x = F.pad(x, (0, pad))
        frames = x.shape[-1] // self.frame_size
        framed = x.reshape(batch * channels, frames, self.frame_size)
        return framed, {"batch": batch, "channels": channels, "frames": frames, "time": time + pad}

    def _unframe(self, framed: torch.Tensor, shape: dict[str, int], crop_to: int | None = None) -> torch.Tensor:
        x = framed.reshape(shape["batch"], shape["channels"], shape["frames"] * self.frame_size)
        if crop_to is not None:
            x = x[..., :crop_to]
        return x

    def _restore_indices(self, indices: torch.Tensor, shape: dict[str, int]) -> torch.Tensor:
        return indices.reshape(shape["batch"], shape["channels"], shape["frames"], indices.shape[-1])

    def _flatten_indices(self, indices: torch.Tensor):
        if indices.ndim != 4:
            raise ValueError(
                f"Expected token indices [batch, channels, frames, codebooks], got {tuple(indices.shape)}"
            )
        batch, channels, frames, codebooks = indices.shape
        flat = indices.reshape(batch * channels, frames, codebooks)
        return flat, {"batch": batch, "channels": channels, "frames": frames, "time": frames * self.frame_size}

    def _expand_n_codebooks(self, n_codebooks, shape: dict[str, int]):
        if n_codebooks is None or isinstance(n_codebooks, int):
            return n_codebooks
        if n_codebooks.ndim == 0:
            return n_codebooks.view(1).expand(shape["batch"] * shape["channels"])
        if n_codebooks.numel() == shape["batch"]:
            return n_codebooks.view(shape["batch"], 1).expand(-1, shape["channels"]).reshape(-1)
        if n_codebooks.numel() == shape["batch"] * shape["channels"]:
            return n_codebooks.reshape(-1)
        raise ValueError(
            "n_codebooks must be scalar, [batch], or [batch * channels]; "
            f"got {tuple(n_codebooks.shape)} for batch={shape['batch']} channels={shape['channels']}"
        )
