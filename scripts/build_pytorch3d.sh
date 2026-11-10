#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Requires CUDA 12.8 matching the local torch 2.7.0+cu128 runtime.
# .cuda is an isolated toolkit prefix; no system CUDA changes are made.
export CUDA_HOME="${PSGNAV_CUDA_HOME:-$PWD/.cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
export PATH="$PWD/.venv/bin:$PATH"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo 'Install cuda-nvcc=12.8 and cuda-cudart-dev=12.8 into .cuda first.' >&2
  exit 1
fi
.venv/bin/python -m pip install ninja
if [[ ! -d .deps/pytorch3d/.git ]]; then
  git clone https://github.com/facebookresearch/pytorch3d.git .deps/pytorch3d
  git -C .deps/pytorch3d checkout 88e182f989c80836f4bd744e0d9cb1852762ce01
fi
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import nvidia
cuda=Path(os.environ['CUDA_HOME'])
headers=cuda/'targets/x86_64-linux/include'
if headers.is_dir():
    for root in nvidia.__path__:
        for folder in Path(root).glob('*/include'):
            for child in folder.iterdir():
                target=headers/child.name
                if not target.exists():target.symlink_to(child)
    for child in headers.iterdir():
        target=cuda/'include'/child.name
        if not target.exists():target.symlink_to(child.resolve())
PY
.venv/bin/python -m pip install --no-build-isolation ./.deps/pytorch3d
