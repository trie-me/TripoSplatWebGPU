#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE_ROOT="${TRIPOSPLAT_V2_CACHE:-${HOME}/.cache/triposplat-webgpu-v2}"
export TRIPOSPLAT_PYTHON="${TRIPOSPLAT_PYTHON:-${CACHE_ROOT}/venv/bin/python}"
export TRIPOSPLAT_OFFICIAL_REPO="${TRIPOSPLAT_OFFICIAL_REPO:-${CACHE_ROOT}/official}"
export TRIPOSPLAT_FLOW_WEIGHTS="${TRIPOSPLAT_FLOW_WEIGHTS:-${CACHE_ROOT}/ckpts/diffusion_models/triposplat_fp16.safetensors}"
export TRIPOSPLAT_RUNNER_URL="${TRIPOSPLAT_RUNNER_URL:-https://yosun-triposplat-webgpu-v2.static.hf.space/}"
exec bash "${ROOT}/scripts/triposplat/start_mac_20_step.sh"
