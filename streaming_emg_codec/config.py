from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AttentionConfig:
    n_embd: int = 128
    n_heads: int = 4
    head_dim: int = 32
    n_layers: int = 4
    n_hidden: int = 512
    dropout_rate: float = 0.0
    window_size: int = 256           # in frames
    rotary_base: float = 10000.0


@dataclass
class RVQConfig:
    num_codebooks: int = 6
    codebook_size: int = 256
    embedding_dim: int = 16
    commitment_weight: float = 0.25
    revive_every: int = 250
    codebook_init: str = "uniform"
    dead_code_restart: bool = False
    dead_criteria: float = 0.9
    counts_decay: float = 0.99
    rotation_trick: bool = False


@dataclass
class ModelConfig:
    sample_rate: int = 1000
    frame_size: int = 20             # samples per frame
    channels: int | None = None
    encoder: AttentionConfig = field(default_factory=AttentionConfig)
    decoder: AttentionConfig = field(default_factory=AttentionConfig)
    rvq: RVQConfig = field(default_factory=RVQConfig)


@dataclass
class DataConfig:
    kind: str = "shards"
    root: str | None = None
    sample_rate: int = 2000
    channels: int | None = 1
    window_seconds: float = 5.0
    split: str = "train"
    batch_size: int = 8
    num_workers: int = 2


@dataclass
class LossConfig:
    huber_weight: float = 1.0
    spectral_weight: float = 1.0
    vq_weight: float = 1.0
    commit_weight: float = 0.25
    stft_fft_sizes: tuple[int, ...] = (64, 128, 256)


@dataclass
class DiscriminatorConfig:
    enabled: bool = False
    start_step: int = 10_000
    ramp_steps: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    adversarial_weight: float = 0.1
    feature_matching_weight: float = 2.0
    periods: tuple[int, ...] = (2, 3, 5, 7, 11)
    mpd_base_filters: int = 8
    mpd_max_filters: int = 128
    stft_filters: int = 16
    stft_max_filters: int = 128
    stft_fft_sizes: tuple[int, ...] = (64, 128, 256, 512)
    stft_hop_lengths: tuple[int, ...] = (17, 31, 67, 131)
    stft_win_lengths: tuple[int, ...] = (64, 128, 256, 512)


@dataclass
class TrainConfig:
    device: str = "cuda"
    seed: int = 1337
    optimizer: str = "adamw"
    lr: float = 3e-4
    muon_lr: float = 0.02
    weight_decay: float = 1e-2
    max_steps: int = 100_000
    log_every: int = 50
    save_every: int = 5_000
    output_dir: str = "checkpoints"
    grad_clip: float = 1.0
    quantizer_dropout: float = 0.0
    use_amp: bool = True
    max_gpu_memory_gb: float = 40.0


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "streaming-emg-codec"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
    dir: str = "wandb"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply `--override a.b=value` strings onto the raw YAML dict."""
    if not overrides:
        return data
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got {override!r}")
        key, raw_value = override.split("=", 1)
        cursor = data
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce_value(raw_value)
    return data


def _dataclass_from_dict(cls: type, data: dict[str, Any]):
    kwargs = {}
    field_map = {field_.name: field_ for field_ in fields(cls)}
    for key, value in data.items():
        if key not in field_map:
            raise ValueError(f"Unknown config key {cls.__name__}.{key}")
        default_factory = getattr(field_map[key], "default_factory", None)
        if isinstance(value, dict) and default_factory is not None:
            try:
                nested_default = default_factory()
            except TypeError:
                nested_default = None
            if is_dataclass(nested_default):
                kwargs[key] = _dataclass_from_dict(type(nested_default), value)
                continue
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = apply_overrides(raw, overrides)
    return _dataclass_from_dict(ExperimentConfig, raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_dict(config), handle, sort_keys=False)


def _to_dict(value):
    if is_dataclass(value):
        return {field_.name: _to_dict(getattr(value, field_.name)) for field_ in fields(value)}
    if isinstance(value, tuple):
        return list(value)
    return value
