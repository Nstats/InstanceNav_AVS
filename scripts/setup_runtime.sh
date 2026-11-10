#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Reuse a Python 3.9 installation with Habitat-Sim and a Blackwell-capable PyTorch.
base_python="${PSGNAV_BASE_PYTHON:-python3.9}"
if [[ ! -x .venv/bin/python ]]; then
  "$base_python" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install 'numpy<2' 'scikit-image<0.25' scikit-learn scikit-fmm \
  yacs faiss-cpu==1.8.0.post1 nltk inflect prettytable ftfy einops pycocotools \
  python-dotenv requests Flask Pillow pytest
if [[ ! -d GLIP/.git ]]; then
  git clone https://github.com/microsoft/GLIP.git GLIP
  git -C GLIP checkout 9dda9558c1ef59bb6cdc8e896e2bcab775a68ff0
fi
if git -C GLIP apply --check ../patches/glip_torch2.patch 2>/dev/null; then
  git -C GLIP apply ../patches/glip_torch2.patch
elif ! git -C GLIP apply --reverse --check ../patches/glip_torch2.patch 2>/dev/null; then
  echo 'GLIP patch does not match this checkout; inspect local dependency changes.' >&2
  exit 1
fi
.venv/bin/python - <<'PY'
import nltk
for name in ('punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng'):
    if not nltk.download(name,download_dir='data/nltk_data',quiet=True):
        raise SystemExit('NLTK asset download failed: '+name)
PY
