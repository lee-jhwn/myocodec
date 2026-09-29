#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec python -m streaming_emg_codec.train "$@"
