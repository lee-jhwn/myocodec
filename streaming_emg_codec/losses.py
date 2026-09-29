from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from streaming_emg_codec.config import LossConfig


class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, fft_sizes: tuple[int, ...] | list[int]):
        super().__init__()
        self.fft_sizes = tuple(int(size) for size in fft_sizes)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
        pred = prediction.float().reshape(-1, prediction.shape[-1])
        tgt = target.float().reshape(-1, target.shape[-1])
        total = pred.new_zeros(())
        for n_fft in self.fft_sizes:
            pred_padded = _pad_to_fft(pred, n_fft)
            tgt_padded = _pad_to_fft(tgt, n_fft)
            hop = max(1, n_fft // 4)
            window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
            pred_stft = torch.stft(
                pred_padded,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                return_complex=True,
            )
            tgt_stft = torch.stft(
                tgt_padded,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                return_complex=True,
            )
            pred_mag = pred_stft.abs().clamp_min(1e-7)
            tgt_mag = tgt_stft.abs().clamp_min(1e-7)
            mag_loss = F.l1_loss(pred_mag.log(), tgt_mag.log())
            convergence = (pred_mag - tgt_mag).norm(p="fro") / tgt_mag.norm(p="fro").clamp_min(1e-7)
            phase_loss = F.l1_loss(torch.angle(pred_stft), torch.angle(tgt_stft))
            total = total + mag_loss + convergence + 0.1 * phase_loss
        return total / len(self.fft_sizes)


class CodecLoss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        self.spectral = MultiScaleSpectralLoss(config.stft_fft_sizes)

    def forward(self, outputs: dict[str, torch.Tensor], target: torch.Tensor) -> dict[str, torch.Tensor]:
        reconstruction = outputs["reconstruction"].float()
        target = target.float()
        huber = F.smooth_l1_loss(reconstruction, target)
        spectral = self.spectral(reconstruction, target)
        vq_loss = outputs["vq_loss"]
        commit_loss = outputs["commit_loss"]
        total = (
            self.config.huber_weight * huber
            + self.config.spectral_weight * spectral
            + self.config.vq_weight * vq_loss
            + self.config.commit_weight * commit_loss
        )
        return {
            "loss": total,
            "huber": huber.detach(),
            "spectral": spectral.detach(),
            "vq": vq_loss.detach(),
            "commit": commit_loss.detach(),
        }


def _pad_to_fft(x: torch.Tensor, n_fft: int) -> torch.Tensor:
    if x.shape[-1] >= n_fft:
        return x
    return F.pad(x, (0, n_fft - x.shape[-1]))
