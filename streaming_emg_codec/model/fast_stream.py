from __future__ import annotations

import torch
import torch.nn.functional as F

from streaming_emg_codec.model.attention import FLASH_ATTN_AVAILABLE
from streaming_emg_codec.model.codec import StreamingEMGCodec, StreamingState

if FLASH_ATTN_AVAILABLE:
    from flash_attn import flash_attn_with_kvcache
else:
    flash_attn_with_kvcache = None


class StreamingSession:
    """Frame-synchronous encode+decode with captured CUDA graphs."""

    def __init__(
        self,
        model: StreamingEMGCodec,
        batch_size: int = 1,
        device: torch.device | str = "cuda",
        cache_dtype: torch.dtype | None = None,
        use_cuda_graph: bool = True,
        exact_mask: bool = True,
        compile: bool = False,
        backend: str = "auto",
        capacity: int = 512,
    ):
        self.model = model.eval()
        self.device = torch.device(device)
        self.batch = int(batch_size)
        self.frame_size = model.frame_size
        self.exact_mask = exact_mask
        if cache_dtype is None:
            cache_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.cache_dtype = cache_dtype

        enc_cfg, dec_cfg = model.config.encoder, model.config.decoder
        if enc_cfg.window_size <= 0 or dec_cfg.window_size <= 0:
            raise ValueError("StreamingSession requires a bounded attention window")
        self.window = {"enc": enc_cfg.window_size, "dec": dec_cfg.window_size}
        self.enc_window = enc_cfg.window_size
        self.dec_window = dec_cfg.window_size
        self.n_codebooks = model.config.rvq.num_codebooks
        self.embedding_dim = model.config.rvq.embedding_dim

        if backend == "auto":
            backend = "flash" if (FLASH_ATTN_AVAILABLE and self.device.type == "cuda") else "sdpa"
        if backend == "flash" and not (FLASH_ATTN_AVAILABLE and self.device.type == "cuda"):
            raise ValueError("backend='flash' requires flash-attn and CUDA")
        if backend not in ("flash", "sdpa"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend

        self.use_cuda_graph = bool(use_cuda_graph) and self.device.type == "cuda"
        self.capacity = int(capacity)
        self._compile = bool(compile)
        self.reset()


    def reset(self) -> None:
        """Start a new stream: drop all context and re-enter the warmup phase."""
        self._n = 0
        self._ref_state = StreamingState()
        self._graphs = {}
        self._fns = {}
        self._ready = False
        self._warm_frames = max(self.enc_window, self.dec_window) + 1

    @property
    def warmup_frames(self) -> int:
        """Frames served by the reference path before the fast path takes over."""
        return self._warm_frames

    @torch.no_grad()
    def step(self, frame: torch.Tensor):
        """One frame in, one frame out."""
        b, c = self._check(frame)
        if not self._ready:
            recon, idx, self._ref_state = self.model.streaming_step(
                frame, state=self._ref_state, n_codebooks=None
            )
            self._after_warm_frame()
            return recon, idx

        self._maybe_compact(("enc", "dec"))
        self._in.copy_(frame.reshape(self.batch, 1, self.frame_size))
        self._run("both", self._compute_both, ("enc", "dec"))
        return (self._recon.reshape(b, c, self.frame_size),
                self._idx.reshape(b, c, 1, self.n_codebooks))

    @torch.no_grad()
    def step_encode(self, frame: torch.Tensor):
        """Encoder half only: one frame in, one token vector out."""
        b, c = self._check(frame)
        if not self._ready:
            idx, self._ref_state.encoder_cache = self.model.encode(
                frame, None, self._ref_state.encoder_cache)
            _, self._ref_state.decoder_cache = self.model.decode(
                idx, None, self._ref_state.decoder_cache)
            self._after_warm_frame()
            return idx

        self._maybe_compact(("enc",))
        self._in.copy_(frame.reshape(self.batch, 1, self.frame_size))
        self._run("encode", self._compute_encode, ("enc",))
        return self._idx.reshape(b, c, 1, self.n_codebooks)

    @torch.no_grad()
    def step_decode(self, indices: torch.Tensor):
        """Decoder half only: one token vector in, one frame out."""
        if not self._ready:
            raise RuntimeError("step_decode needs a primed session; drive the warmup "
                               "frames through step() or step_encode() first")
        shape = indices.shape
        self._maybe_compact(("dec",))
        self._idx_in.copy_(indices.reshape(self.batch, 1, self.n_codebooks))
        self._run("decode", self._compute_decode, ("dec",))
        return self._recon.reshape(shape[0], shape[1], self.frame_size)


    def _check(self, frame: torch.Tensor):
        if frame.shape[-1] != self.frame_size:
            raise ValueError(f"expected {self.frame_size} samples per frame, got {frame.shape[-1]}")
        b, c = frame.shape[0], frame.shape[1]
        if b * c != self.batch:
            raise ValueError(f"batch*channels ({b * c}) does not match session batch {self.batch}")
        return b, c

    def _after_warm_frame(self) -> None:
        self._n += 1
        if self._n == self._warm_frames:
            self._seed_from_reference()

    def _run(self, name, compute, tags) -> None:
        if self.use_cuda_graph:
            if name not in self._graphs:
                self._capture(name, compute)
            self._graphs[name].replay()
        else:
            self._fns.setdefault(name, self._maybe_compile(compute))()
            self._advance(tags)
        for tag in tags:
            self._seq_host[tag] += 1

    def _maybe_compile(self, fn):
        return torch.compile(fn, mode="max-autotune-no-cudagraphs", dynamic=False) \
            if self._compile else fn


    def _seed_from_reference(self) -> None:
        dev = self.device
        self._in = torch.zeros(self.batch, 1, self.frame_size, device=dev, dtype=torch.float32)
        self._idx_in = torch.zeros(self.batch, 1, self.n_codebooks, device=dev, dtype=torch.long)
        self._rope_pos = {t: torch.tensor(float(self._n), device=dev, dtype=torch.float32)
                          for t in ("enc", "dec")}
        if self.backend == "flash":
            self._seed_flash()
        else:
            self._seed_rings()
        self._ready = True

    def _seed_flash(self) -> None:
        dev, dt = self.device, self.cache_dtype
        self._kc, self._vc, self._seqlens, self._seq_host = {}, {}, {}, {}
        for tag, caches in (("enc", self._ref_state.encoder_cache),
                            ("dec", self._ref_state.decoder_cache)):
            ks, vs = [], []
            for cache in caches:
                k = torch.zeros(self.batch, self.capacity, cache.n_heads, cache.head_dim,
                                device=dev, dtype=dt)
                v = torch.zeros_like(k)
                n = cache.filled
                k[:, :n].copy_(cache.kv_cache[:, :n, 0].to(dt))
                v[:, :n].copy_(cache.kv_cache[:, :n, 1].to(dt))
                ks.append(k); vs.append(v)
            self._kc[tag], self._vc[tag] = ks, vs
            filled = caches[0].filled
            self._seqlens[tag] = torch.full((self.batch,), filled, device=dev, dtype=torch.int32)
            self._seq_host[tag] = filled

    def _seed_rings(self) -> None:
        dev, dt = self.device, self.cache_dtype
        self._rings, self._slot, self._seq_host = {}, {}, {}
        for tag, caches in (("enc", self._ref_state.encoder_cache),
                            ("dec", self._ref_state.decoder_cache)):
            size = self.window[tag] + 1
            rings = []
            for cache in caches:
                ring = torch.zeros(self.batch, size, 2, cache.n_heads, cache.head_dim,
                                   device=dev, dtype=dt)
                ring.copy_(cache.kv_cache[:, cache.filled - size:cache.filled].to(dt))
                rings.append(ring)
            self._rings[tag] = rings
            self._slot[tag] = torch.zeros(1, device=dev, dtype=torch.long)
            self._seq_host[tag] = caches[0].filled
        if self.exact_mask:
            self._mask = {t: torch.ones(1, 1, 1, self.window[t] + 1, device=dev, dtype=torch.bool)
                          for t in ("enc", "dec")}
        else:
            self._mask = {"enc": None, "dec": None}

    def _maybe_compact(self, tags) -> None:
        if self.backend != "flash":
            return
        for tag in tags:
            if self._seq_host[tag] >= self.capacity - 1:
                self._compact_flash(tag)

    def _compact_flash(self, tag: str) -> None:
        keep = self.window[tag] + 1
        n = self._seq_host[tag]
        for k, v in zip(self._kc[tag], self._vc[tag]):
            k[:, :keep] = k[:, n - keep:n].clone()
            v[:, :keep] = v[:, n - keep:n].clone()
            k[:, keep:].zero_(); v[:, keep:].zero_()
        self._seqlens[tag].fill_(keep)
        self._seq_host[tag] = keep


    def _rope(self, qkv: torch.Tensor, inv_freq: torch.Tensor, tag: str) -> torch.Tensor:
        freqs = (self._rope_pos[tag] * inv_freq).view(1, 1, 1, 1, -1)
        cos, sin = freqs.cos().to(qkv.dtype), freqs.sin().to(qkv.dtype)
        first, second = qkv[..., 0::2], qkv[..., 1::2]
        return torch.stack((first * cos - second * sin,
                            first * sin + second * cos), dim=-1).flatten(-2)

    def _attend_flash(self, layer, x, k_cache, v_cache, tag):
        attn = layer.attn
        qkv = attn.qkv(x).view(self.batch, 1, 3, attn.n_heads, attn.head_dim)
        qkv = self._rope(qkv, attn.rotary.inv_freq, tag).to(self.cache_dtype)
        ctx = flash_attn_with_kvcache(
            qkv[:, :, 0].contiguous(), k_cache, v_cache,
            k=qkv[:, :, 1].contiguous(), v=qkv[:, :, 2].contiguous(),
            cache_seqlens=self._seqlens[tag],
            softmax_scale=attn.softmax_scale, causal=True,
            window_size=(self.window[tag], 0),
        )
        ctx = ctx.contiguous().view(self.batch, 1, attn.n_embd).to(x.dtype)
        return attn.out(ctx)

    def _attend_sdpa(self, layer, x, ring, tag):
        attn = layer.attn
        qkv = attn.qkv(x).view(self.batch, 1, 3, attn.n_heads, attn.head_dim)
        qkv = self._rope(qkv, attn.rotary.inv_freq, tag).to(self.cache_dtype)
        ring.index_copy_(1, self._slot[tag], qkv[:, :, 1:3])
        q = qkv[:, :, 0].transpose(1, 2)                       # [B, H, 1, D]
        k = ring[:, :, 0].transpose(1, 2)                      # [B, H, W+1, D]
        v = ring[:, :, 1].transpose(1, 2)
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask[tag],
                                             dropout_p=0.0, scale=attn.softmax_scale)
        ctx = ctx.transpose(1, 2).contiguous().view(self.batch, 1, attn.n_embd).to(x.dtype)
        return attn.out(ctx)

    def _stack(self, stack, x, tag):
        if self.backend == "flash":
            for layer, k, v in zip(stack.layers, self._kc[tag], self._vc[tag]):
                x = x + self._attend_flash(layer, layer.norm1(x), k, v, tag)
                x = x + layer.ffn(layer.norm2(x))
        else:
            for layer, ring in zip(stack.layers, self._rings[tag]):
                x = x + self._attend_sdpa(layer, layer.norm1(x), ring, tag)
                x = x + layer.ffn(layer.norm2(x))
        return stack.norm(x)

    def _rvq_encode(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.reshape(-1, self.embedding_dim)
        out = []
        for vq in self.model.rvq.vqs:
            cb = vq.codebook.weight
            d = (residual.pow(2).sum(dim=1, keepdim=True)
                 - 2 * residual @ cb.t()
                 + cb.pow(2).sum(dim=1).unsqueeze(0))
            i = d.argmin(dim=1)
            residual = residual - vq.codebook(i)
            out.append(i)
        return torch.stack(out, dim=-1).view(self.batch, 1, self.n_codebooks)

    def _rvq_decode(self, indices: torch.Tensor) -> torch.Tensor:
        vqs = self.model.rvq.vqs
        out = vqs[0].codebook(indices[:, :, 0])
        for idx in range(1, self.n_codebooks):
            out = out + vqs[idx].codebook(indices[:, :, idx])
        return out

    def _encode_core(self) -> torch.Tensor:
        m = self.model
        h = self._stack(m.encoder, m.linear_in(self._in), "enc")
        return self._rvq_encode(m.rvq_in(h))

    def _decode_core(self, indices: torch.Tensor) -> torch.Tensor:
        m = self.model
        h = self._stack(m.decoder, m.rvq_out(self._rvq_decode(indices)), "dec")
        return m.linear_out(h).reshape(self.batch, 1, self.frame_size)

    def _compute_both(self) -> None:
        idx = self._encode_core()
        self._idx = idx
        self._recon = self._decode_core(idx)

    def _compute_encode(self) -> None:
        self._idx = self._encode_core()

    def _compute_decode(self) -> None:
        self._recon = self._decode_core(self._idx_in)

    def _advance(self, tags) -> None:
        for tag in tags:
            self._rope_pos[tag] += 1.0
            if self.backend == "flash":
                self._seqlens[tag].add_(1)
            else:
                self._slot[tag].add_(1).remainder_(self.window[tag] + 1)


    def _state_snapshot(self):
        pos = {t: p.cpu() for t, p in self._rope_pos.items()}
        host = dict(self._seq_host)
        if self.backend == "flash":
            return ("flash", pos, host,
                    {t: [k.to("cpu", copy=True) for k in ks] for t, ks in self._kc.items()},
                    {t: [v.to("cpu", copy=True) for v in vs] for t, vs in self._vc.items()},
                    {t: s.cpu() for t, s in self._seqlens.items()})
        return ("sdpa", pos, host,
                {t: [r.to("cpu", copy=True) for r in rs] for t, rs in self._rings.items()},
                {t: s.cpu() for t, s in self._slot.items()})

    def _state_restore(self, snap) -> None:
        kind, pos, host = snap[0], snap[1], snap[2]
        for t, p in pos.items():
            self._rope_pos[t].copy_(p)
        self._seq_host.update(host)
        if kind == "flash":
            _, _, _, ks, vs, seq = snap
            for tag in ks:
                for dst, src in zip(self._kc[tag], ks[tag]):
                    dst.copy_(src)
                for dst, src in zip(self._vc[tag], vs[tag]):
                    dst.copy_(src)
                self._seqlens[tag].copy_(seq[tag])
            return
        _, _, _, rings, slots = snap
        for tag, rs in rings.items():
            for dst, src in zip(self._rings[tag], rs):
                dst.copy_(src)
            self._slot[tag].copy_(slots[tag])

    def _capture(self, name, compute) -> None:
        tags = {"both": ("enc", "dec"), "encode": ("enc",), "decode": ("dec",)}[name]
        fn = self._fns.setdefault(name, self._maybe_compile(compute))
        snap = self._state_snapshot()
        torch.cuda.synchronize()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
                self._advance(tags)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
            self._advance(tags)
        torch.cuda.synchronize()

        self._state_restore(snap)
        self._graphs[name] = graph
