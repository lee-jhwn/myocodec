from __future__ import annotations

import glob
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from streaming_emg_codec.config import DataConfig


class ShardDataset(torch.utils.data.IterableDataset):
    """Low-RAM block-wise streamer over float16 HDF5 window shards."""

    BLOCK = 16384

    def __init__(self, config: DataConfig, training: bool = True):
        del training
        self.shards = sorted(glob.glob(os.path.join(config.root, "*.h5")))
        if not self.shards:
            raise ValueError(f"no shards in {config.root}")

    def __iter__(self):
        import h5py

        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        rng = random.Random(int(info.seed) if info else 0)
        while True:
            order = self.shards[:]
            rng.shuffle(order)
            mine = [s for i, s in enumerate(order) if i % nw == wid] or [order[wid % len(order)]]
            for path in mine:
                try:
                    handle = h5py.File(path, "r")
                    windows = handle["windows"]
                except Exception:
                    continue
                n = windows.shape[0]
                for start in range(0, n, self.BLOCK):
                    block = np.asarray(windows[start:start + self.BLOCK], dtype=np.float32)
                    idx = list(range(block.shape[0]))
                    rng.shuffle(idx)
                    for j in idx:
                        yield {"emg": torch.from_numpy(block[j][None, :].copy())}
                handle.close()


class SyntheticEMGDataset(Dataset):
    """Band-limited sinusoids plus noise."""

    def __init__(self, config: DataConfig, length: int = 1024):
        self.channels = config.channels or 16
        self.samples = int(config.window_seconds * config.sample_rate)
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        time = torch.linspace(0, 1, self.samples)
        freqs = torch.linspace(15, 140, self.channels).unsqueeze(1)
        phase = torch.rand(self.channels, 1) * 2 * math.pi
        signal = torch.sin(2 * math.pi * freqs * time.unsqueeze(0) + phase)
        envelope = 0.5 + torch.rand(self.channels, 1)
        noise = 0.05 * torch.randn_like(signal)
        return {"emg": (signal * envelope + noise).float()}


def build_dataset(config: DataConfig, training: bool = True) -> Dataset:
    kind = config.kind.lower()
    if kind == "shards":
        return ShardDataset(config, training=training)
    if kind == "synthetic":
        return SyntheticEMGDataset(config)
    raise ValueError(f"Unknown dataset kind: {config.kind}")
