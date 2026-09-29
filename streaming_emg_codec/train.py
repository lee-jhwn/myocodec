from __future__ import annotations

import argparse
import fcntl
import signal
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from streaming_emg_codec.config import load_config, save_config
from streaming_emg_codec.data import build_dataset
from streaming_emg_codec.losses import CodecLoss
from streaming_emg_codec.model import StreamingEMGCodec
from streaming_emg_codec.model.discriminator import (
    EMGDiscriminator,
    feature_matching_loss,
    generator_adversarial_loss,
    set_requires_grad,
)
from streaming_emg_codec.utils import ensure_dir, move_batch_to_device, set_seed, snr_db


def build_optimizer(model, config):
    """AdamW, or Muon with its auxiliary Adam when config.train.optimizer == "muon"."""
    lr = config.train.lr
    wd = config.train.weight_decay
    kind = str(getattr(config.train, "optimizer", "adamw")).lower()
    if kind not in ("adamw", "muon"):
        raise ValueError(f"unknown optimizer {kind!r}")
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    from muon import SingleDeviceMuonWithAuxAdam

    muon_p, adam_p = [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        is_codebook = "rvq" in name and prm.ndim == 2 and "codebook" in name
        is_edge = name.startswith(("linear_in", "linear_out", "rvq_in", "rvq_out"))
        if prm.ndim == 2 and not is_codebook and not is_edge:
            muon_p.append(prm)
        else:
            adam_p.append(prm)
    muon_lr = float(getattr(config.train, "muon_lr", 0.02))
    groups = [
        dict(params=muon_p, use_muon=True, lr=muon_lr, weight_decay=wd),
        dict(params=adam_p, use_muon=False, lr=lr, betas=(0.9, 0.95), weight_decay=wd),
    ]
    n_m = sum(q.numel() for q in muon_p) / 1e6
    n_a = sum(q.numel() for q in adam_p) / 1e6
    print(f"[optim] Muon on {len(muon_p)} matrices ({n_m:.2f} M params, lr {muon_lr}); "
          f"aux Adam on {len(adam_p)} tensors ({n_a:.2f} M, lr {lr})", flush=True)
    return SingleDeviceMuonWithAuxAdam(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a streaming EMG neural codec")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--resume",
        default="auto",
        help="Checkpoint path, 'auto' for output_dir/latest.pt, or 'none' to start fresh.",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    set_seed(config.train.seed)
    device = torch.device(config.train.device)
    output_dir = ensure_dir(config.train.output_dir)
    lock_handle = _acquire_training_lock(output_dir)
    _configure_cuda_memory_limit(device, config.train.max_gpu_memory_gb)

    dataset = build_dataset(config.data, training=True)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset),
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    def _iterate_forever(dl):
        while True:
            for batch in dl:
                yield batch
    iterator = _iterate_forever(loader)

    model = StreamingEMGCodec(config.model).to(device)
    criterion = CodecLoss(config.loss).to(device)
    optimizer = build_optimizer(model, config)
    discriminator = EMGDiscriminator(config.discriminator).to(device) if config.discriminator.enabled else None
    discriminator_optimizer = (
        torch.optim.AdamW(
            discriminator.parameters(),
            lr=config.discriminator.lr,
            weight_decay=config.discriminator.weight_decay,
            betas=(0.8, 0.99),
        )
        if discriminator is not None
        else None
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.train.use_amp and device.type == "cuda")

    resume_path = _resolve_resume_path(args.resume, output_dir)
    start_step = _load_checkpoint_if_available(
        resume_path,
        model,
        optimizer,
        scaler,
        device,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
    )
    save_config(config, output_dir / "resolved_config.yaml")
    wandb_run = _init_wandb(config, output_dir, start_step)
    latest_state = {"step": start_step, "saved": False}

    def handle_signal(signum, frame):
        print(f"received signal {signum}; saving checkpoint before exit", flush=True)
        _save_checkpoint(
            output_dir,
            model,
            optimizer,
            scaler,
            config,
            latest_state["step"],
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer,
        )
        _finish_wandb(wandb_run)
        latest_state["saved"] = True
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    if start_step >= config.train.max_steps:
        print(f"checkpoint already at step {start_step} >= max_steps {config.train.max_steps}; exiting")
        _finish_wandb(wandb_run)
        lock_handle.close()
        return

    start = time.time()
    model.train()
    if discriminator is not None:
        discriminator.train()

    for step in range(start_step + 1, config.train.max_steps + 1):
        latest_state["step"] = step
        batch = move_batch_to_device(next(iterator), device)
        emg = batch["emg"].float()
        n_codebooks = _sample_n_codebooks(config, emg, device)
        adversarial_active = discriminator is not None and step >= config.discriminator.start_step

        disc_metrics = None
        if adversarial_active:
            disc_metrics = _update_discriminator(
                model=model,
                discriminator=discriminator,
                discriminator_optimizer=discriminator_optimizer,
                scaler=scaler,
                emg=emg,
                n_codebooks=n_codebooks,
                config=config,
                device=device,
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=config.train.use_amp and device.type == "cuda",
        ):
            outputs = model(emg, n_codebooks=n_codebooks)
            losses = criterion(outputs, emg)
        codec_loss = losses["loss"]

        if adversarial_active:
            set_requires_grad(discriminator, False)
            with torch.no_grad():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=config.train.use_amp and device.type == "cuda",
                ):
                    _, real_fmaps = discriminator(emg.detach())
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=config.train.use_amp and device.type == "cuda",
            ):
                fake_outputs, fake_fmaps = discriminator(outputs["reconstruction"])
                adv_loss = generator_adversarial_loss(fake_outputs)
                fm_loss = feature_matching_loss(real_fmaps, fake_fmaps)
                adv_ramp = _adversarial_ramp(step, config.discriminator)
                generator_loss = (
                    codec_loss
                    + adv_ramp * config.discriminator.adversarial_weight * adv_loss
                    + adv_ramp * config.discriminator.feature_matching_weight * fm_loss
                )
            losses["codec"] = codec_loss.detach()
            losses["adv"] = adv_loss.detach()
            losses["fm"] = fm_loss.detach()
            losses["adv_ramp"] = emg.new_tensor(adv_ramp)
            losses["disc"] = disc_metrics["disc"]
            losses["disc_real"] = disc_metrics["disc_real"]
            losses["disc_fake"] = disc_metrics["disc_fake"]
            losses["disc_real_acc"] = disc_metrics["disc_real_acc"]
            losses["disc_fake_acc"] = disc_metrics["disc_fake_acc"]
            losses["loss"] = generator_loss
        else:
            generator_loss = codec_loss

        scaler.scale(generator_loss).backward()
        if config.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if discriminator is not None:
            set_requires_grad(discriminator, True)
        _check_cuda_memory(device, config.train.max_gpu_memory_gb)

        if step % config.train.log_every == 0 or step == 1:
            elapsed = max(time.time() - start, 1e-6)
            steps_since_start = max(step - start_step, 1)
            samples_per_sec = steps_since_start * emg.shape[0] / elapsed
            metrics = {key: float(value.detach().cpu()) for key, value in losses.items()}
            _vqs = getattr(getattr(model, "rvq", None), "vqs", None)
            if _vqs is not None:
                _u = [float(v.usage) for v in _vqs if hasattr(v, "usage")]
                if _u:
                    metrics["cb_usage"] = sum(_u) / len(_u)
                    metrics["cb_usage_min"] = min(_u)
            with torch.no_grad():
                recon = outputs["reconstruction"].detach().float()
                metrics["snr_clean"] = float(snr_db(recon, emg).mean().cpu())
            memory = _cuda_memory_string(device)
            print(
                f"step={step} samples_per_sec={samples_per_sec:.2f} "
                + " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
                + (f" {memory}" if memory else ""),
                flush=True,
            )
            _log_wandb(
                wandb_run,
                step=step,
                samples_per_sec=samples_per_sec,
                metrics=metrics,
                memory_metrics=_cuda_memory_metrics(device),
                active_codebooks=n_codebooks,
                adversarial_active=adversarial_active,
            )

        if step % config.train.save_every == 0 or step == config.train.max_steps:
            _save_checkpoint(
                output_dir,
                model,
                optimizer,
                scaler,
                config,
                step,
                discriminator=discriminator,
                discriminator_optimizer=discriminator_optimizer,
            )

    _finish_wandb(wandb_run)
    lock_handle.close()


def _adversarial_ramp(step: int, disc_config) -> float:
    ramp_steps = getattr(disc_config, "ramp_steps", 0)
    if ramp_steps and ramp_steps > 0:
        progress = (step - disc_config.start_step) / float(ramp_steps)
        return float(min(1.0, max(0.0, progress)))
    return 1.0


def _init_wandb(config, output_dir: Path, start_step: int):
    if not getattr(config.wandb, "enabled", False):
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"wandb disabled: import failed: {type(exc).__name__}: {exc}", flush=True)
        return None
    if not hasattr(wandb, "init"):
        print("wandb disabled: imported module has no wandb.init; install the wandb package", flush=True)
        return None

    wandb_dir = ensure_dir(config.wandb.dir)
    run_id_path = output_dir / "wandb_run_id.txt"
    run_id = run_id_path.read_text(encoding="utf-8").strip() if run_id_path.exists() else None
    init_kwargs = {
        "project": config.wandb.project,
        "config": asdict(config),
        "dir": str(wandb_dir),
        "mode": config.wandb.mode,
        "resume": "allow",
    }
    if config.wandb.entity:
        init_kwargs["entity"] = config.wandb.entity
    if config.wandb.name:
        init_kwargs["name"] = config.wandb.name
    if config.wandb.group:
        init_kwargs["group"] = config.wandb.group
    if config.wandb.tags:
        init_kwargs["tags"] = list(config.wandb.tags)
    if run_id:
        init_kwargs["id"] = run_id

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        if config.wandb.mode != "offline":
            print(
                f"wandb online init failed: {type(exc).__name__}: {exc}; retrying with mode=offline",
                flush=True,
            )
            init_kwargs["mode"] = "offline"
            try:
                run = wandb.init(**init_kwargs)
            except Exception as offline_exc:
                print(
                    f"wandb disabled: offline init failed: {type(offline_exc).__name__}: {offline_exc}",
                    flush=True,
                )
                return None
        else:
            print(f"wandb disabled: init failed: {type(exc).__name__}: {exc}", flush=True)
            return None

    try:
        if hasattr(wandb, "define_metric"):
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")
        if run is not None and not run_id_path.exists() and getattr(run, "id", None):
            run_id_path.write_text(str(run.id) + "\n", encoding="utf-8")
        print(
            f"wandb enabled project={config.wandb.project} name={config.wandb.name or getattr(run, 'name', None)} "
            f"id={getattr(run, 'id', None)} start_step={start_step}",
            flush=True,
        )
        return run
    except Exception as exc:
        print(f"wandb disabled: post-init setup failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def _log_wandb(
    run,
    *,
    step: int,
    samples_per_sec: float,
    metrics: dict[str, float],
    memory_metrics: dict[str, float],
    active_codebooks: torch.Tensor | int | None,
    adversarial_active: bool,
) -> None:
    if run is None:
        return
    payload = {"step": step, "train/samples_per_sec": samples_per_sec, "train/adversarial_active": int(adversarial_active)}
    payload.update({f"train/{key}": value for key, value in metrics.items()})
    payload.update({f"memory/{key}": value for key, value in memory_metrics.items()})
    payload.update(_active_codebook_metrics(active_codebooks))
    try:
        run.log(payload, step=step)
    except Exception as exc:
        print(f"wandb log failed at step {step}: {type(exc).__name__}: {exc}", flush=True)


def _active_codebook_metrics(active_codebooks: torch.Tensor | int | None) -> dict[str, float]:
    if active_codebooks is None:
        return {}
    if isinstance(active_codebooks, int):
        return {
            "train/active_codebooks_mean": float(active_codebooks),
            "train/active_codebooks_min": float(active_codebooks),
            "train/active_codebooks_max": float(active_codebooks),
        }
    values = active_codebooks.detach().float()
    return {
        "train/active_codebooks_mean": float(values.mean().cpu()),
        "train/active_codebooks_min": float(values.min().cpu()),
        "train/active_codebooks_max": float(values.max().cpu()),
    }


def _finish_wandb(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        print(f"wandb finish failed: {type(exc).__name__}: {exc}", flush=True)


def _sample_n_codebooks(config, emg: torch.Tensor, device: torch.device) -> torch.Tensor | None:
    if config.train.quantizer_dropout <= 0:
        return None
    batch = emg.shape[0]
    full = torch.full((batch,), config.model.rvq.num_codebooks, device=device, dtype=torch.long)
    drop = torch.rand(batch, device=device) < config.train.quantizer_dropout
    sampled = torch.randint(1, config.model.rvq.num_codebooks + 1, (batch,), device=device)
    return torch.where(drop, sampled, full)


def _update_discriminator(
    model: torch.nn.Module,
    discriminator: torch.nn.Module,
    discriminator_optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    emg: torch.Tensor,
    n_codebooks: torch.Tensor | int | None,
    config,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    set_requires_grad(discriminator, True)
    discriminator_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=config.train.use_amp and device.type == "cuda",
        ):
            fake = model(emg, n_codebooks=n_codebooks)["reconstruction"].detach()

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=config.train.use_amp and device.type == "cuda",
    ):
        real_outputs, _ = discriminator(emg.detach())
        real_loss = _disc_real_loss(real_outputs)
    scaler.scale(real_loss).backward()
    real_acc = _disc_accuracy(real_outputs, real=True).detach()
    del real_outputs

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=config.train.use_amp and device.type == "cuda",
    ):
        fake_outputs, _ = discriminator(fake)
        fake_loss = _disc_fake_loss(fake_outputs)
    scaler.scale(fake_loss).backward()
    fake_acc = _disc_accuracy(fake_outputs, real=False).detach()
    del fake_outputs, fake

    if config.discriminator.grad_clip > 0:
        scaler.unscale_(discriminator_optimizer)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.discriminator.grad_clip)
    scaler.step(discriminator_optimizer)
    discriminator_optimizer.zero_grad(set_to_none=True)
    return {
        "disc": (real_loss.detach() + fake_loss.detach()),
        "disc_real": real_loss.detach(),
        "disc_fake": fake_loss.detach(),
        "disc_real_acc": real_acc,
        "disc_fake_acc": fake_acc,
    }


def _disc_real_loss(outputs: list[torch.Tensor]) -> torch.Tensor:
    loss = outputs[0].new_zeros(())
    for output in outputs:
        loss = loss + torch.mean((1.0 - output) ** 2)
    return loss


def _disc_fake_loss(outputs: list[torch.Tensor]) -> torch.Tensor:
    loss = outputs[0].new_zeros(())
    for output in outputs:
        loss = loss + torch.mean(output**2)
    return loss


def _disc_accuracy(outputs: list[torch.Tensor], real: bool) -> torch.Tensor:
    acc = outputs[0].new_zeros(())
    for output in outputs:
        acc = acc + ((output >= 0.5) if real else (output < 0.5)).float().mean()
    return acc / max(len(outputs), 1)


def _acquire_training_lock(output_dir: Path):
    lock_path = output_dir / "train.lock"
    handle = lock_path.open("w", encoding="utf-8")
    print(f"acquiring training lock {lock_path}", flush=True)
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    handle.write(f"pid={__import__('os').getpid()}\n")
    handle.flush()
    print("training lock acquired", flush=True)
    return handle


def _resolve_resume_path(resume_arg: str, output_dir: Path) -> Path | None:
    if resume_arg.lower() in {"none", "false", "0", "no"}:
        return None
    if resume_arg == "auto":
        path = output_dir / "latest.pt"
        return path if path.exists() else None
    return Path(resume_arg)


def _load_checkpoint_if_available(
    checkpoint_path: Path | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    discriminator: torch.nn.Module | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
) -> int:
    if checkpoint_path is None:
        print("starting from scratch; no resume checkpoint found", flush=True)
        return 0
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint and checkpoint["scaler"]:
        scaler.load_state_dict(checkpoint["scaler"])
    if discriminator is not None:
        if "discriminator" in checkpoint:
            discriminator.load_state_dict(checkpoint["discriminator"])
            print("loaded discriminator state", flush=True)
        else:
            print("checkpoint has no discriminator state; initializing discriminator from scratch", flush=True)
    if discriminator_optimizer is not None:
        if "discriminator_optimizer" in checkpoint:
            discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
            print("loaded discriminator optimizer state", flush=True)
        else:
            print("checkpoint has no discriminator optimizer state; initializing discriminator optimizer", flush=True)
    step = int(checkpoint.get("step", 0))
    print(f"resumed from {checkpoint_path} at step {step}", flush=True)
    return step


def _save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config,
    step: int,
    discriminator: torch.nn.Module | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "config": asdict(config),
    }
    if discriminator is not None:
        checkpoint["discriminator"] = discriminator.state_dict()
    if discriminator_optimizer is not None:
        checkpoint["discriminator_optimizer"] = discriminator_optimizer.state_dict()
    tmp_latest = output_dir / "latest.pt.tmp"
    torch.save(checkpoint, output_dir / f"step_{step}.pt")
    torch.save(checkpoint, tmp_latest)
    tmp_latest.replace(output_dir / "latest.pt")


def _configure_cuda_memory_limit(device: torch.device, max_gpu_memory_gb: float | None) -> None:
    if device.type != "cuda" or not max_gpu_memory_gb or max_gpu_memory_gb <= 0:
        return
    device_index = torch.cuda.current_device() if device.index is None else device.index
    props = torch.cuda.get_device_properties(device_index)
    total_gb = props.total_memory / 1024**3
    fraction = min(1.0, max_gpu_memory_gb / total_gb)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
    print(f"cuda memory cap set to {max_gpu_memory_gb:.2f} GB ({fraction:.3f} of {total_gb:.2f} GB)", flush=True)


def _check_cuda_memory(device: torch.device, max_gpu_memory_gb: float | None) -> None:
    if device.type != "cuda" or not max_gpu_memory_gb or max_gpu_memory_gb <= 0:
        return
    device_index = torch.cuda.current_device() if device.index is None else device.index
    reserved_gb = torch.cuda.memory_reserved(device_index) / 1024**3
    allocated_gb = torch.cuda.memory_allocated(device_index) / 1024**3
    max_reserved_gb = torch.cuda.max_memory_reserved(device_index) / 1024**3
    if max(reserved_gb, allocated_gb, max_reserved_gb) > max_gpu_memory_gb:
        raise RuntimeError(
            "GPU memory exceeded configured cap: "
            f"allocated={allocated_gb:.2f}GB reserved={reserved_gb:.2f}GB "
            f"max_reserved={max_reserved_gb:.2f}GB cap={max_gpu_memory_gb:.2f}GB"
        )


def _cuda_memory_string(device: torch.device) -> str:
    metrics = _cuda_memory_metrics(device)
    if not metrics:
        return ""
    return (
        f"cuda_alloc_gb={metrics['cuda_alloc_gb']:.2f} "
        f"cuda_reserved_gb={metrics['cuda_reserved_gb']:.2f} "
        f"cuda_max_reserved_gb={metrics['cuda_max_reserved_gb']:.2f}"
    )


def _cuda_memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    device_index = torch.cuda.current_device() if device.index is None else device.index
    return {
        "cuda_alloc_gb": torch.cuda.memory_allocated(device_index) / 1024**3,
        "cuda_reserved_gb": torch.cuda.memory_reserved(device_index) / 1024**3,
        "cuda_max_reserved_gb": torch.cuda.max_memory_reserved(device_index) / 1024**3,
    }


if __name__ == "__main__":
    main()
