from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embedding_dim: int, *, codebook_init: str = "uniform",
                 dead_code_restart: bool = False, dead_criteria: float = 0.9,
                 counts_decay: float = 0.99, rotation_trick: bool = False):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        if codebook_init == "randn":
            nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0)
        else:
            nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        self.dead_code_restart = dead_code_restart
        self.dead_criteria = dead_criteria
        self.counts_decay = counts_decay
        self.rotation_trick = rotation_trick
        self.register_buffer("counts_avg", torch.ones(codebook_size), persistent=False)
        self.register_buffer("usage", torch.ones(()), persistent=False)

    @torch.no_grad()
    def _counts_ema(self, indices: torch.Tensor) -> None:
        counts = torch.bincount(indices.reshape(-1), minlength=self.codebook_size).float()
        self.counts_avg.mul_(self.counts_decay).add_(counts, alpha=1.0 - self.counts_decay)
        self.usage.fill_((self.counts_avg > self.dead_criteria).float().mean())

    @torch.no_grad()
    def expires_code(self, batch_samples: torch.Tensor) -> int:
        """Teleport entries whose EMA usage is below threshold onto real encoder outputs."""
        dead = self.counts_avg < self.dead_criteria
        n = int(dead.sum())
        if n == 0:
            return 0
        flat = batch_samples.detach().reshape(-1, self.embedding_dim)
        if flat.shape[0] == 0:
            return 0
        pick = torch.randint(0, flat.shape[0], (self.codebook_size,), device=flat.device)
        self.codebook.weight.data.copy_(
            torch.where(dead.unsqueeze(-1), flat[pick].to(self.codebook.weight.dtype),
                        self.codebook.weight.data))
        self.counts_avg.masked_fill_(dead, self.dead_criteria * 2.0)
        return n

    def _rotate(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        xn = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        qn = q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        u, v, e = x / xn, q / qn, x
        w = ((u + v) / (u + v).norm(dim=-1, keepdim=True).clamp(min=1e-6)).detach()
        eww = (e * w).sum(-1, keepdim=True) * w
        euq = (e * u.detach()).sum(-1, keepdim=True) * v.detach()
        return (e - 2 * eww + 2 * euq) * (qn / xn).detach()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, self.embedding_dim)
        codebook = self.codebook.weight
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        return indices.view(*x.shape[:-1])

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codebook(indices)

    def forward(self, x: torch.Tensor):
        indices = self.encode(x)
        quantized = self.decode(indices)
        vq_loss = (quantized - x.detach()).pow(2).mean(dim=-1)
        commit_loss = (x - quantized.detach()).pow(2).mean(dim=-1)
        if self.rotation_trick:
            quantized_st = self._rotate(x, quantized)
        else:
            quantized_st = x + (quantized - x).detach()
        if self.training and self.dead_code_restart:
            self._counts_ema(indices)
            self.expires_code(x)
        return quantized_st, indices, vq_loss, commit_loss


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_codebooks = config.num_codebooks
        self.codebook_size = config.codebook_size
        self.embedding_dim = config.embedding_dim
        self.commitment_weight = config.commitment_weight
        self.vqs = nn.ModuleList(
            [
                VectorQuantizer(
                    config.codebook_size, config.embedding_dim,
                    codebook_init=getattr(config, "codebook_init", "uniform"),
                    dead_code_restart=getattr(config, "dead_code_restart", False),
                    dead_criteria=getattr(config, "dead_criteria", 0.9),
                    counts_decay=getattr(config, "counts_decay", 0.99),
                    rotation_trick=getattr(config, "rotation_trick", False),
                )
                for _ in range(config.num_codebooks)
            ]
        )

    def forward(self, x: torch.Tensor, n_codebooks: torch.Tensor | int | None = None):
        n_codebooks_tensor = self._normalize_n_codebooks(x, n_codebooks)
        residual = x
        quantized_total = torch.zeros_like(x)
        indices = []
        vq_loss = x.new_zeros(())
        commit_loss = x.new_zeros(())
        mask = self._codebook_mask(n_codebooks_tensor, x.device, x.dtype)
        for idx, vq in enumerate(self.vqs):
            quantized, selected, this_vq_loss, this_commit_loss = vq(residual)
            residual = residual - quantized.detach()
            active = mask[:, idx].view(-1, 1, 1)
            loss_active = mask[:, idx].view(-1, 1)
            denom = loss_active.sum().clamp_min(1.0) * this_vq_loss.shape[1]
            quantized_total = quantized_total + quantized * active
            vq_loss = vq_loss + (this_vq_loss * loss_active).sum() / denom
            commit_loss = commit_loss + (this_commit_loss * loss_active).sum() / denom
            indices.append(selected)
        return quantized_total, torch.stack(indices, dim=-1), vq_loss, commit_loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor, n_codebooks: torch.Tensor | int | None = None) -> torch.Tensor:
        n_codebooks_tensor = self._normalize_n_codebooks(x, n_codebooks)
        max_books = int(n_codebooks_tensor.max().item())
        residual = x
        indices = []
        for idx in range(max_books):
            selected = self.vqs[idx].encode(residual)
            quantized = self.vqs[idx].decode(selected)
            residual = residual - quantized
            indices.append(selected)
        return torch.stack(indices, dim=-1)

    @torch.no_grad()
    def decode(self, indices: torch.Tensor, n_codebooks: torch.Tensor | int | None = None) -> torch.Tensor:
        batch, steps, available_books = indices.shape
        probe = self.vqs[0].decode(indices[:, :, 0])
        output = torch.zeros(batch, steps, self.embedding_dim, device=indices.device, dtype=probe.dtype)
        n_codebooks_tensor = self._normalize_n_codebooks(output, n_codebooks, default=available_books)
        max_books = int(n_codebooks_tensor.max().item())
        if max_books > available_books:
            raise ValueError(f"Requested {max_books} codebooks but indices only contain {available_books}")
        mask = self._codebook_mask(n_codebooks_tensor, indices.device, output.dtype)
        for idx in range(max_books):
            decoded = self.vqs[idx].decode(indices[:, :, idx])
            output = output + decoded * mask[:, idx].view(-1, 1, 1)
        return output

    def _normalize_n_codebooks(
        self,
        x: torch.Tensor,
        n_codebooks: torch.Tensor | int | None,
        default: int | None = None,
    ) -> torch.Tensor:
        batch = x.shape[0]
        if n_codebooks is None:
            n_codebooks = default or self.num_codebooks
        if isinstance(n_codebooks, int):
            n_codebooks = torch.full((batch,), n_codebooks, device=x.device, dtype=torch.long)
        else:
            n_codebooks = n_codebooks.to(device=x.device, dtype=torch.long).view(-1)
            if n_codebooks.numel() == 1 and batch > 1:
                n_codebooks = n_codebooks.expand(batch)
        if n_codebooks.shape[0] != batch:
            raise ValueError(f"n_codebooks batch {n_codebooks.shape[0]} does not match {batch}")
        if n_codebooks.min().item() < 1 or n_codebooks.max().item() > self.num_codebooks:
            raise ValueError(f"n_codebooks must be in [1, {self.num_codebooks}]")
        return n_codebooks

    def _codebook_mask(self, n_codebooks: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return (
            torch.arange(self.num_codebooks, device=device).unsqueeze(0) < n_codebooks.unsqueeze(1)
        ).to(dtype=dtype)
