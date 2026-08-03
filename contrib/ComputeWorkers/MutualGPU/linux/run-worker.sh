#!/usr/bin/env bash
set -euo pipefail

worker_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIPOSPLAT_ERROR_PREFIX="TripoSplat worker"
# Reuse the installer's no-guess readiness checks for direct/source runs too.
# shellcheck source=install/lib.sh
source "$worker_dir/install/lib.sh"
model_revision="de3b99ab2627d565a8d5fc40f2db52557b82b974"
backend="${MUTUALGPU_TRIPOSPLAT_BACKEND:-auto}"
provider_key_file="${MUTUALGPU_PROVIDER_KEY_FILE:-}"
model_dir="${MUTUALGPU_TRIPOSPLAT_MODEL_DIR:-}"
api_url="${MUTUALGPU_API_URL:-https://mutualgpu.com}"

usage() {
  cat <<'EOF'
Usage: mutualgpu-triposplat run --provider-key-file PATH [--backend auto|cuda|rocm] [--model-dir PATH] [--api-url HTTPS_URL]

Runs a single foreground Linux provider. The opaque provider key must be stored
in a chmod-600 file containing one key; it is read by Node and never passed to
the Python GPU subprocess.
EOF
}
fail() { echo "TripoSplat worker: $1" >&2; exit 2; }

while (($#)); do
  case "$1" in
    run) ;;
    --provider-key-file) provider_key_file="${2:-}"; shift ;;
    --backend) backend="${2:-}"; shift ;;
    --model-dir) model_dir="${2:-}"; shift ;;
    --api-url) api_url="${2:-}"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
  shift
done

[[ "$(uname -s)" == "Linux" ]] || fail "this foreground worker supports Linux only"
[[ "$(uname -m)" == "x86_64" ]] || fail "this release currently supports Linux x86_64 only"
[[ "$api_url" == https://* ]] || fail "--api-url must use https://"
[[ -n "$provider_key_file" ]] || fail "--provider-key-file is required"
[[ -r "$provider_key_file" ]] || fail "provider key file is not readable"
[[ "$(stat -c '%a' "$provider_key_file")" =~ ^[0-7]*[0-7][0-7]$ ]] || fail "could not inspect provider key file mode"
mode="$(stat -c '%a' "$provider_key_file")"
(( (8#$mode & 8#077) == 0 )) || fail "provider key file must be chmod 600"

triposplat_detect_backends
triposplat_select_backend "$backend"
backend="$TRIPOSPLAT_SELECTED_BACKEND"
if [[ -z "$model_dir" && -r "$worker_dir/model-dir" ]]; then model_dir="$(<"$worker_dir/model-dir")"; fi
model_dir="${model_dir:-$worker_dir/models/$model_revision}"

runtime_project="$worker_dir/runtime/$backend"
python="$runtime_project/.venv/bin/python"
[[ -f "$runtime_project/uv.lock" ]] || fail "the $backend frozen uv lock is missing; reinstall this exact release"
[[ -x "$python" ]] || fail "the $backend environment is missing; rerun the deterministic installer"
[[ -d "$model_dir" ]] || fail "model directory is missing; rerun install with --download-models or pass --model-dir"
[[ -d "$worker_dir/node_modules/@mutualgpu/provider-core" ]] || fail "bundled provider SDK is missing; reinstall this exact release"

export MUTUALGPU_API_URL="$api_url"
export MUTUALGPU_PROVIDER_KEY_FILE="$provider_key_file"
export MUTUALGPU_TRIPOSPLAT_BACKEND="$backend"
export MUTUALGPU_TRIPOSPLAT_PYTHON="$python"
export MUTUALGPU_TRIPOSPLAT_MODEL_DIR="$model_dir"
export MUTUALGPU_TRIPOSPLAT_MODEL_MANIFEST="$worker_dir/model-manifest.json"

echo "[preflight] Linux $backend worker; verifying the GPU, models, and optional Vulkan diagnostics before enrollment."
exec env -u SSL_CERT_DIR node "$worker_dir/src/index.mjs"
