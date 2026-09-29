import argparse, glob, h5py, numpy as np, torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streaming_emg_codec.config import load_config
from streaming_emg_codec.model import StreamingEMGCodec, StreamingState

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--val-root", default="data/shards_val")
ap.add_argument("--nwin", type=int, default=8)
ap.add_argument("--tag", default="model")
ap.add_argument("--amp", action="store_true",
                help="also report the bfloat16 autocast used during training")
a = ap.parse_args()

cfg = load_config(a.config)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StreamingEMGCodec(cfg.model).to(dev).eval()
model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
frame = cfg.model.frame_size

sp = sorted(glob.glob(f"{a.val_root}/shard_*.h5"))[0]
with h5py.File(sp, "r") as f:
    x = np.asarray(f["windows"][:a.nwin], dtype=np.float32)[:, None, :]
emg = torch.from_numpy(x).to(dev)
pad = (-emg.shape[-1]) % frame
if pad:
    emg = torch.nn.functional.pad(emg, (0, pad))

print(f"{a.tag}: windows={a.nwin} frame={frame} frames/window={emg.shape[-1] // frame}")

modes = [("float32", False)] + ([("bfloat16", True)] if a.amp else [])
ok = True
for label, use_amp in modes:
    amp = dict(device_type=dev.type, dtype=torch.bfloat16,
               enabled=use_amp and dev.type == "cuda")
    with torch.no_grad():
        with torch.autocast(**amp):
            out = model(emg, n_codebooks=None)
        tok_off, rec_off = out["indices"], out["reconstruction"].float()

        state = StreamingState(); recs = []; toks = []
        for s in range(0, emg.shape[-1], frame):
            with torch.autocast(**amp):
                rec, tok, state = model.streaming_step(emg[..., s:s + frame], state=state,
                                                       n_codebooks=None)
            recs.append(rec.float()); toks.append(tok)
        rec_str = torch.cat(recs, dim=-1); tok_str = torch.cat(toks, dim=2)

    tok_eq = (tok_off == tok_str)
    frac = tok_eq.float().mean().item()
    percb = [round(tok_eq[..., c].float().mean().item(), 5) for c in range(tok_off.shape[-1])]
    maxdiff = (rec_off - rec_str).abs().max().item()
    rel = ((rec_off - rec_str).norm() / rec_off.norm().clamp_min(1e-8)).item()
    print(f"  [{label}]")
    print(f"    TOKENS: exact_all={bool(tok_eq.all())}  frac_match={frac:.6f}  per_codebook={percb}")
    print(f"    RECON : max_abs_diff={maxdiff:.3e}  rel_l2_diff={rel:.3e}")
    if label == "float32":
        ok = frac > 0.999 and rel < 1e-2

print(f"  VERDICT[{a.tag}]: {'PASS (streaming == offline in fp32)' if ok else 'MISMATCH - investigate'}")
