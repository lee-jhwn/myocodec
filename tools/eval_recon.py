import argparse, glob, math, sys
import h5py, numpy as np, torch
from scipy.signal import resample_poly


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def si_sdr(est, ref, eps=1e-8):
    ref = ref - ref.mean(-1, keepdim=True); est = est - est.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    t = a * ref; n = est - t
    return 10 * torch.log10((t.pow(2).sum(-1) + eps) / (n.pow(2).sum(-1) + eps))


def pearson(est, ref, eps=1e-8):
    e = est - est.mean(-1, keepdim=True); r = ref - ref.mean(-1, keepdim=True)
    return (e * r).sum(-1) / (e.norm(dim=-1) * r.norm(dim=-1) + eps)


def snr(est, ref, eps=1e-8):
    return 10 * torch.log10((ref.pow(2).sum(-1) + eps) / ((ref - est).pow(2).sum(-1) + eps))


def rms_env(x, win):
    """Non-overlapping RMS envelope; x is [B, T]."""
    T = (x.shape[-1] // win) * win
    return x[..., :T].reshape(x.shape[0], -1, win).pow(2).mean(-1).clamp_min(1e-12).sqrt()


class Accum:
    def __init__(self):
        self.n = 0
        self.sisdr = self.pear = self.snr = self.envr = 0.0
        self.ss_res = self.ss_tot_sq = self.sum_ref = self.abs_res = 0.0
        self.n_samp = 0

    def add(self, rec, ref, env_win):
        self.n += ref.shape[0]
        self.sisdr += float(si_sdr(rec, ref).sum())
        self.pear += float(pearson(rec, ref).sum())
        self.snr += float(snr(rec, ref).sum())
        self.envr += float(pearson(rms_env(rec, env_win), rms_env(ref, env_win)).sum())
        d = (rec - ref).double()
        r = ref.double()
        self.ss_res += float(d.pow(2).sum())
        self.abs_res += float(d.abs().sum())
        self.sum_ref += float(r.sum())
        self.ss_tot_sq += float(r.pow(2).sum())
        self.n_samp += r.numel()

    def report(self):
        mean_ref = self.sum_ref / self.n_samp
        ss_tot = self.ss_tot_sq - self.n_samp * mean_ref ** 2
        var = ss_tot / self.n_samp
        return dict(
            n=self.n,
            si_sdr=self.sisdr / self.n, pearson=self.pear / self.n,
            snr=self.snr / self.n, envr=self.envr / self.n,
            r2=1.0 - self.ss_res / ss_tot,
            nrmse=math.sqrt(self.ss_res / self.n_samp) / math.sqrt(var),
            mae=self.abs_res / self.n_samp,
        )


def codebook_stats(counts, L, K):
    utils, perps = [], []
    for c in range(L):
        tot = int(counts[c].sum())
        utils.append(int((counts[c] > 0).sum()) / K)
        if tot > 0:
            p = counts[c].float() / tot
            nz = p[p > 0]
            perps.append(math.exp(float(-(nz * nz.log()).sum())))
        else:
            perps.append(0.0)
    return utils, perps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["myocodec", "biocodec"], required=True)
    ap.add_argument("--config"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-root", default="data/shards_val")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--biocodec-repo", default=None,
                    help="path to a clone of the BioCodec release (only for --model biocodec)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if a.model == "myocodec":
        from streaming_emg_codec.config import load_config
        from streaming_emg_codec.model import StreamingEMGCodec
        cfg = load_config(a.config)
        model = StreamingEMGCodec(cfg.model).to(dev)
        ck = torch.load(a.ckpt, map_location=dev)
        model.load_state_dict(ck["model"]); model.eval()
        L = cfg.model.rvq.num_codebooks; K = cfg.model.rvq.codebook_size
        fs = float(cfg.model.sample_rate)
        tps = cfg.model.sample_rate / cfg.model.frame_size
        step = ck.get("step", "?")
        use_amp = cfg.train.use_amp
    else:
        if not a.biocodec_repo:
            raise SystemExit("--model biocodec requires --biocodec-repo (the baseline is not "
                             "redistributed here)")
        sys.path.insert(0, a.biocodec_repo)
        import biocodec.modules as m, biocodec.quantization as qt
        from biocodec.model import BioCodecModel
        L, K, HOP = 6, 256, 36
        fs = 1000.0
        enc = m.SEANetEncoder(channels=1, norm="weight_norm", causal=True, ratios=[3, 3, 2, 2],
                              n_residual_layers=2, true_skip=True, compress=1)
        dec = m.SEANetDecoder(channels=1, norm="weight_norm", causal=True, ratios=[3, 3, 2, 2],
                              n_residual_layers=2, true_skip=True, compress=1)
        quant = qt.ResidualVectorQuantizer(dimension=enc.dimension, n_q=L, bins=K)
        model = BioCodecModel(enc, dec, quant, sample_rate=1000, channels=1,
                              normalize=False, segment=None, name="bc")
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()}, strict=True)
        model.eval().to(dev)
        tps = fs / HOP
        step = "epoch2"
        use_amp = False

    env_win = int(round(0.020 * fs))          # 20 ms
    acc = Accum()
    counts = torch.zeros(L, K, dtype=torch.long)

    with torch.inference_mode():
        for sp in sorted(glob.glob(f"{a.val_root}/shard_*.h5")):
            with h5py.File(sp, "r") as f:
                W = f["windows"]
                for b0 in range(0, W.shape[0], a.batch_size):
                    if a.max_windows and acc.n >= a.max_windows:
                        break
                    x = np.asarray(W[b0:b0 + a.batch_size], dtype=np.float32)[:, None, :]
                    if a.model == "myocodec":
                        emg = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
                        with torch.autocast(dev.type, dtype=torch.bfloat16,
                                            enabled=use_amp and dev.type == "cuda"):
                            out = model(emg, n_codebooks=None)
                        rec = out["reconstruction"].float()[:, 0, :]
                        ref = emg.float()[:, 0, :]
                        idx = out["indices"].detach().cpu()
                        for c in range(L):
                            counts[c] += torch.bincount(idx[..., c].reshape(-1), minlength=K)
                    else:
                        x1k = resample_poly(x, 1, 2, axis=-1)
                        emg = torch.from_numpy(np.ascontiguousarray(x1k)).to(dev)
                        emg = (emg - emg.mean(-1, keepdim=True)) / emg.std(-1, keepdim=True).clamp_min(1e-5)
                        emg = emg.clamp(-10.0, 10.0)
                        frames = model.encode(emg); codes = frames[0][0]
                        rec = model.decode(frames)[:, :, :emg.shape[-1]].float()[:, 0, :]
                        ref = emg.float()[:, 0, :]
                        for c in range(L):
                            counts[c] += torch.bincount(codes[:, c, :].reshape(-1).cpu(), minlength=K)
                    acc.add(rec, ref, env_win)
            if a.max_windows and acc.n >= a.max_windows:
                break

    r = acc.report()
    utils, perps = codebook_stats(counts, L, K)
    bps = L * math.log2(K) * tps
    print(f"{a.tag} step={step} n={r['n']} SI-SDR={r['si_sdr']:.3f} Pearson={r['pearson']:.4f} "
          f"SNR={r['snr']:.3f} R2={r['r2']:.4f} NRMSE={r['nrmse']:.4f} MAE={r['mae']:.4f} "
          f"EnvR={r['envr']:.4f} cb_util={sum(utils)/L:.4f} cb_perp={sum(perps)/L:.2f} "
          f"cb_util_min={min(utils):.4f} cb_perp_min={min(perps):.2f} "
          f"tok/s={tps:.1f} bits/s/ch={bps:.0f} env_win={env_win}", flush=True)
    print("PERCB_UTIL|" + "|".join(f"{u:.4f}" for u in utils), flush=True)
    print("PERCB_PERP|" + "|".join(f"{p:.2f}" for p in perps), flush=True)


if __name__ == "__main__":
    main()
