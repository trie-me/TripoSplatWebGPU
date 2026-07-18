#!/usr/bin/env bash
set -euo pipefail

OFFICIAL_COMMIT="a78fa12d06dbf1381ca548bfac32bb68cb8c451d"
CACHE_ROOT="${TRIPOSPLAT_V2_CACHE:-${HOME}/.cache/triposplat-webgpu-v2}"
OFFICIAL_REPO="${CACHE_ROOT}/official"
VENV="${CACHE_ROOT}/venv"
CKPTS="${CACHE_ROOT}/ckpts"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12 is required. Install it first (for example: brew install python@3.12)." >&2
  exit 1
fi

mkdir -p "${CACHE_ROOT}"
if [[ ! -d "${OFFICIAL_REPO}/.git" ]]; then
  git clone https://github.com/VAST-AI-Research/TripoSplat.git "${OFFICIAL_REPO}"
fi
git -C "${OFFICIAL_REPO}" fetch origin "${OFFICIAL_COMMIT}"
git -C "${OFFICIAL_REPO}" checkout --detach "${OFFICIAL_COMMIT}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3.12 -m venv "${VENV}"
fi
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install \
  numpy==2.5.1 \
  pillow==12.3.0 \
  safetensors==0.8.0 \
  torch==2.13.0 \
  torchvision==0.28.0 \
  tqdm==4.68.4 \
  huggingface_hub==1.24.0

mkdir -p "${CKPTS}/diffusion_models"
"${VENV}/bin/python" - "${CKPTS}" <<'PY'
import sys
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="VAST-AI/TripoSplat",
    filename="diffusion_models/triposplat_fp16.safetensors",
    local_dir=sys.argv[1],
)
PY

echo
echo "Exact Mac v2 dependencies are ready in ${CACHE_ROOT}."
echo "Start the hosted runner with:"
echo "  scripts/triposplat/start_mac_20_step_v2.sh"
