from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def _flatten_emg_channels(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.float()
    if x.ndim != 3:
        raise ValueError(f"Expected EMG tensor [batch, channels, time], got {tuple(x.shape)}")
    x = x.float()
    traces = [
        x.mean(dim=1),
        x.square().mean(dim=1).clamp_min(1e-8).sqrt(),
    ]
    if x.shape[1] > 1:
        traces.append(x.std(dim=1, unbiased=False))
    if x.shape[1] >= 4:
        midpoint = x.shape[1] // 2
        traces.append(x[:, :midpoint].mean(dim=1))
        traces.append(x[:, midpoint:].mean(dim=1))
    return torch.cat(traces, dim=0)


def _padding_2d(kernel_size: tuple[int, int], dilation: tuple[int, int] = (1, 1)) -> tuple[int, int]:
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


class EMGPeriodDiscriminator(nn.Module):
    """HiFi-GAN-style multi-period discriminator branch."""

    def __init__(
        self,
        period: int,
        base_filters: int = 8,
        max_filters: int = 128,
        kernel_size: int = 5,
        stride: int = 3,
    ):
        super().__init__()
        self.period = int(period)
        channels = [
            base_filters,
            min(base_filters * 4, max_filters),
            min(base_filters * 8, max_filters),
            max_filters,
            max_filters,
        ]
        in_channels = [1] + channels[:-1]
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1) if idx < 4 else (1, 1),
                        padding=(kernel_size // 2, 0),
                    )
                )
                for idx, (in_ch, out_ch) in enumerate(zip(in_channels, channels))
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(channels[-1], 1, kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = _flatten_emg_channels(x).unsqueeze(1)
        batch, channels, time = x.shape
        if time % self.period:
            pad = self.period - (time % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            time += pad
        x = x.reshape(batch, channels, time // self.period, self.period)

        feature_maps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)
        return torch.flatten(x, 1), feature_maps


class EMGMultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: Sequence[int], base_filters: int = 8, max_filters: int = 128):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                EMGPeriodDiscriminator(period, base_filters=base_filters, max_filters=max_filters)
                for period in periods
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outputs = []
        feature_maps = []
        for discriminator in self.discriminators:
            output, fmap = discriminator(x)
            outputs.append(output)
            feature_maps.append(fmap)
        return outputs, feature_maps


class EMGSTFTDiscriminator(nn.Module):
    """Single STFT discriminator scale for normalized EMG traces."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        filters: int = 16,
        max_filters: int = 128,
        kernel_size: tuple[int, int] = (7, 3),
        stride: tuple[int, int] = (1, 2),
        dilations: Sequence[int] = (1, 2, 4),
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.convs = nn.ModuleList()
        self.convs.append(
            weight_norm(nn.Conv2d(2, filters, kernel_size=kernel_size, padding=_padding_2d(kernel_size)))
        )
        in_ch = filters
        for index, dilation in enumerate(dilations):
            out_ch = min(filters * (2 ** (index + 1)), max_filters)
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=(1, int(dilation)),
                        padding=_padding_2d(kernel_size, (1, int(dilation))),
                    )
                )
            )
            in_ch = out_ch
        self.convs.append(
            weight_norm(nn.Conv2d(in_ch, max_filters, kernel_size=(3, 3), padding=_padding_2d((3, 3))))
        )
        self.conv_post = weight_norm(nn.Conv2d(max_filters, 1, kernel_size=(3, 3), padding=_padding_2d((3, 3))))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = _flatten_emg_channels(x)
        if x.shape[-1] < self.n_fft:
            x = F.pad(x, (0, self.n_fft - x.shape[-1]), mode="reflect")
        window = torch.hann_window(self.win_length, device=x.device, dtype=torch.float32)
        z = torch.stft(
            x.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        z = torch.view_as_real(z).permute(0, 3, 1, 2).contiguous()
        feature_maps = []
        for conv in self.convs:
            z = F.leaky_relu(conv(z), 0.2)
            feature_maps.append(z)
        z = self.conv_post(z)
        feature_maps.append(z)
        return z, feature_maps


class EMGMultiScaleSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        n_ffts: Sequence[int],
        hop_lengths: Sequence[int],
        win_lengths: Sequence[int],
        filters: int = 16,
        max_filters: int = 128,
    ):
        super().__init__()
        if not (len(n_ffts) == len(hop_lengths) == len(win_lengths)):
            raise ValueError("n_ffts, hop_lengths, and win_lengths must have matching lengths")
        self.discriminators = nn.ModuleList(
            [
                EMGSTFTDiscriminator(
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=win,
                    filters=filters,
                    max_filters=max_filters,
                )
                for n_fft, hop, win in zip(n_ffts, hop_lengths, win_lengths)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outputs = []
        feature_maps = []
        for discriminator in self.discriminators:
            output, fmap = discriminator(x)
            outputs.append(output)
            feature_maps.append(fmap)
        return outputs, feature_maps


class EMGDiscriminator(nn.Module):
    """Combined EMG discriminator: multi-period + multi-scale STFT branches."""

    def __init__(self, config):
        super().__init__()
        self.mpd = EMGMultiPeriodDiscriminator(
            periods=config.periods,
            base_filters=config.mpd_base_filters,
            max_filters=config.mpd_max_filters,
        )
        self.msstft = EMGMultiScaleSTFTDiscriminator(
            n_ffts=config.stft_fft_sizes,
            hop_lengths=config.stft_hop_lengths,
            win_lengths=config.stft_win_lengths,
            filters=config.stft_filters,
            max_filters=config.stft_max_filters,
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        mpd_outputs, mpd_fmaps = self.mpd(x)
        stft_outputs, stft_fmaps = self.msstft(x)
        return mpd_outputs + stft_outputs, mpd_fmaps + stft_fmaps


@dataclass
class DiscriminatorLossStats:
    real_loss: torch.Tensor
    fake_loss: torch.Tensor
    real_accuracy: torch.Tensor
    fake_accuracy: torch.Tensor


def discriminator_loss(
    real_outputs: Sequence[torch.Tensor],
    fake_outputs: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, DiscriminatorLossStats]:
    loss = real_outputs[0].new_zeros(())
    real_loss_total = real_outputs[0].new_zeros(())
    fake_loss_total = real_outputs[0].new_zeros(())
    real_acc_total = real_outputs[0].new_zeros(())
    fake_acc_total = real_outputs[0].new_zeros(())
    count = 0
    for real, fake in zip(real_outputs, fake_outputs):
        real_loss = torch.mean((1.0 - real) ** 2)
        fake_loss = torch.mean(fake**2)
        loss = loss + real_loss + fake_loss
        real_loss_total = real_loss_total + real_loss.detach()
        fake_loss_total = fake_loss_total + fake_loss.detach()
        real_acc_total = real_acc_total + (real >= 0.5).float().mean().detach()
        fake_acc_total = fake_acc_total + (fake < 0.5).float().mean().detach()
        count += 1
    denom = max(count, 1)
    return loss, DiscriminatorLossStats(
        real_loss=real_loss_total / denom,
        fake_loss=fake_loss_total / denom,
        real_accuracy=real_acc_total / denom,
        fake_accuracy=fake_acc_total / denom,
    )


def generator_adversarial_loss(fake_outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    loss = fake_outputs[0].new_zeros(())
    for fake in fake_outputs:
        loss = loss + torch.mean((1.0 - fake) ** 2)
    return loss


def feature_matching_loss(
    real_feature_maps: Sequence[Sequence[torch.Tensor]],
    fake_feature_maps: Sequence[Sequence[torch.Tensor]],
) -> torch.Tensor:
    loss = real_feature_maps[0][0].new_zeros(())
    for real_maps, fake_maps in zip(real_feature_maps, fake_feature_maps):
        for real, fake in zip(real_maps, fake_maps):
            loss = loss + F.l1_loss(fake, real.detach())
    return loss


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)
