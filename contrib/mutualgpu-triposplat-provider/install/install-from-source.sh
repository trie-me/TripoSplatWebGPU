#!/usr/bin/env bash
set -euo pipefail
umask 077

repository="${TRIPOSPLAT_PROVIDER_REPOSITORY:-https://github.com/trie-me/TripoSplatWebGPU.git}"
ref=""
backend="auto"
destination="${XDG_DATA_HOME:-$HOME/.local/share}/triposplat-webgpu-provider/source"
model_dir="${XDG_DATA_HOME:-$HOME/.local/share}/triposplat-webgpu-provider/models/de3b99ab2627d565a8d5fc40f2db52557b82b974"
download_models=false
dry_run=false

usage() {
  cat <<'EOF'
Usage: install-from-source.sh --ref BRANCH_OR_TAG [--backend auto|cuda|rocm] [--download-models] [--destination PATH] [--model-dir PATH] [--dry-run]

This is a development bootstrap for a visible Git branch. It clones the
specified source ref and creates a local frozen CUDA or ROCm environment. For
an immutable production install, use the signed release installer instead.
EOF
}
fail() { echo "TripoSplat source installer: $1" >&2; exit 2; }

while (($#)); do
  case "$1" in
    --ref) ref="${2:-}"; shift ;;
    --backend) backend="${2:-}"; shift ;;
    --destination) destination="${2:-}"; shift ;;
    --model-dir) model_dir="${2:-}"; shift ;;
    --download-models) download_models=true ;;
    --dry-run) dry_run=true ;;
    --help|-h) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
  shift
done
[[ -n "$ref" && "$ref" =~ ^[A-Za-z0-9._/-]+$ ]] || fail "--ref must be an explicit safe branch or tag name"
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || fail "this provider supports Linux x86_64 only"
[[ "$backend" == auto || "$backend" == cuda || "$backend" == rocm ]] || fail "--backend must be auto, cuda, or rocm"
if [[ "$backend" == auto ]]; then
  if command -v rocminfo >/dev/null 2>&1 && rocminfo >/dev/null 2>&1; then backend=rocm
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then backend=cuda
  else fail "could not detect a usable NVIDIA CUDA or AMD ROCm runtime"; fi
fi
worker_dir="$destination/contrib/mutualgpu-triposplat-provider"
if "$dry_run"; then
  echo "would clone $repository at $ref into $destination"
  echo "would run npm ci and frozen uv sync for $backend"
  "$download_models" && echo "would download five SHA-256-pinned model files into $model_dir"
  exit 0
fi
command -v git >/dev/null || fail "git is required"
command -v npm >/dev/null || fail "npm is required"
command -v uv >/dev/null || fail "uv 0.11 is required"
if [[ -e "$destination" ]]; then fail "destination already exists: $destination"; fi
git clone --depth 1 --branch "$ref" "$repository" "$destination"
npm ci --prefix "$worker_dir"
uv sync --project "$worker_dir/runtime/$backend" --frozen --no-dev
if "$download_models"; then python3 "$worker_dir/install/model-download.py" --manifest "$worker_dir/model-manifest.json" --model-dir "$model_dir"; fi
echo "Source installation complete. Run:"
echo "  $worker_dir/run-worker.sh run --provider-key-file /secure/provider.key --backend $backend --model-dir $model_dir"
