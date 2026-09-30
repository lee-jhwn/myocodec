# MyoCodec: A Streaming Neural Codec for Electromyography

Jihwan Lee<sup>1</sup>, Kleanthis Avramidis<sup>1</sup>, Junhyeok Lee<sup>2</sup>, Tiantian Feng<sup>1</sup>, Najim Dehak<sup>2</sup>, Shrikanth Narayanan<sup>1</sup>

<sup>1</sup>Signal Analysis and Interpretation Lab (SAIL), University of Southern California, USA<br><sup>2</sup>Center for Language and Speech Processing, Johns Hopkins University, USA

[Paper](https://arxiv.org/abs/2609.36687) | [Audio samples](https://lee-jhwn.github.io/myocodec/) | [Checkpoints](https://huggingface.co/lee-jhwn/myocodec)

Official PyTorch implementation.

## Install

```bash
pip install -r requirements.txt
```

`flash-attn` and `wandb` are optional; remove them from `requirements.txt` if they fail to install.

## Checkpoint

The pretrained weights are on Hugging Face at [lee-jhwn/myocodec](https://huggingface.co/lee-jhwn/myocodec):

```bash
hf download lee-jhwn/myocodec myocodec_step200000_model.pt --local-dir .
```

## Train

```bash
scripts/train.sh --config configs/pretrain_stage1.yaml \
  --override data.root=$EMG_SHARD_ROOT --override train.output_dir=checkpoints/myocodec
scripts/train.sh --config configs/pretrain_stage2.yaml \
  --override data.root=$EMG_SHARD_ROOT --override train.output_dir=checkpoints/myocodec
```

Stage 2 resumes from stage 1's latest checkpoint in the same `output_dir`; after that, use only
the stage-2 config on it.

## Evaluate

```bash
python tools/eval_recon.py --model myocodec --config configs/pretrain_stage2.yaml \
  --ckpt myocodec_step200000_model.pt --val-root $EMG_SHARD_ROOT_VAL
python tools/verify_streaming.py --config configs/pretrain_stage2.yaml \
  --ckpt myocodec_step200000_model.pt --val-root $EMG_SHARD_ROOT_VAL
```

## Inference

```python
import torch
from huggingface_hub import hf_hub_download
from streaming_emg_codec.config import load_config
from streaming_emg_codec.model import StreamingEMGCodec
from streaming_emg_codec.model.fast_stream import StreamingSession

cfg = load_config("configs/pretrain_stage2.yaml")
model = StreamingEMGCodec(cfg.model).eval()
ckpt = hf_hub_download("lee-jhwn/myocodec", "myocodec_step200000_model.pt")
model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])

out = model(x)

# Streaming on GPU, one 40-sample (20 ms) frame at a time
session = StreamingSession(model.cuda(), batch_size=16, device="cuda")  # batch * channels
recon, codes = session.step(frame)                                     # frame: [1, 16, 40]
```
