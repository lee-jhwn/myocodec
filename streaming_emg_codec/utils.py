from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return output


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def snr_db(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-window SNR in dB of ``estimate`` against ``reference``."""
    ref = reference.detach().flatten(1)
    err = (estimate.detach().float() - reference.detach().float()).flatten(1)
    ref_pow = ref.float().pow(2).mean(dim=1).clamp_min(1e-12)
    err_pow = err.pow(2).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(ref_pow / err_pow)
