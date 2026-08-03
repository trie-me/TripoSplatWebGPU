#!/usr/bin/env bash
set -euo pipefail
umask 077

# BASH_SOURCE has no element when this program is read from standard input by
# `curl ... | bash -s --`.  Do not derive a directory until there is a real
# script file; the matching helper is fetched below for that deliberately
# supported execution mode.
script_source="${BASH_SOURCE[0]:-}"
installer_library="${TRIPOSPLAT_INSTALLER_LIBRARY:-}"
if [[ -z "$installer_library" && -n "$script_source" && -f "$script_source" ]]; then
  installer_dir="$(cd -- "$(dirname -- "$script_source")" && pwd -P)"
  installer_library="$installer_dir/lib.sh"
fi
library_temporary=""
early_fail() { echo "TripoSplat source installer: $1" >&2; exit 2; }

repository="${TRIPOSPLAT_PROVIDER_REPOSITORY:-https://github.com/trie-me/TripoSplatWebGPU.git}"
ref=""
requested_backend="auto"
destination="${XDG_DATA_HOME:-$HOME/.local/share}/triposplat-webgpu-provider/source"
model_dir="${XDG_DATA_HOME:-$HOME/.local/share}/triposplat-webgpu-provider/models/de3b99ab2627d565a8d5fc40f2db52557b82b974"
download_models=false
dry_run=false
provider_key_file=""
provider_key_supplied=false
api_url="${MUTUALGPU_API_URL:-https://mutualgpu.com}"
run_foreground=true

usage() {
  cat <<'EOF'
Usage: install-from-source.sh --ref BRANCH_OR_TAG [--backend auto|cuda|rocm] [--download-models] [--destination PATH] [--model-dir PATH] [--provider-key-file PATH] [--api-url HTTPS_URL] [--no-run] [--dry-run]

This is a development bootstrap for a visible Git branch. It clones the
specified source ref and creates a local frozen CUDA or ROCm environment. For
an immutable production install, use the signed release installer instead.

Unless --no-run is supplied, a successful interactive bootstrap securely
prompts for the actual MutualGPU provider credential when no key file is
supplied, then starts the foreground worker. The credential is never echoed,
logged, or passed to the Python GPU process.
EOF
}
while (($#)); do
  case "$1" in
    --ref) ref="${2:-}"; shift ;;
    --backend) requested_backend="${2:-}"; shift ;;
    --destination) destination="${2:-}"; shift ;;
    --model-dir) model_dir="${2:-}"; shift ;;
    --provider-key-file) provider_key_file="${2:-}"; provider_key_supplied=true; shift ;;
    --api-url) api_url="${2:-}"; shift ;;
    --download-models) download_models=true ;;
    --no-run) run_foreground=false ;;
    --dry-run) dry_run=true ;;
    --help|-h) usage; exit 0 ;;
    *) triposplat_fail "unknown argument: $1" ;;
  esac
  shift
done
[[ -n "$ref" && "$ref" =~ ^[A-Za-z0-9._/-]+$ ]] || early_fail "--ref must be an explicit safe branch or tag name"
[[ "$api_url" == https://* ]] || early_fail "--api-url must use https://"
if [[ -r "$installer_library" ]]; then
  # shellcheck source=lib.sh
  source "$installer_library"
else
  command -v curl >/dev/null || early_fail "curl is required when this development bootstrap is piped from GitHub"
  library_temporary="$(mktemp "${TMPDIR:-/tmp}/mutualgpu-triposplat-lib.XXXXXX")"
  cleanup_library() { rm -f -- "$library_temporary"; }
  trap cleanup_library EXIT
  raw_base="${TRIPOSPLAT_PROVIDER_RAW_BASE:-https://raw.githubusercontent.com/trie-me/TripoSplatWebGPU}"
  curl --fail --location --proto '=https' --tlsv1.2 --retry 3 --output "$library_temporary" \
    "$raw_base/$ref/contrib/ComputeWorkers/MutualGPU/linux/install/lib.sh"
  # The development URL is branch-scoped by design; immutable production
  # installs use the separately checksummed release bootstrap instead.
  source "$library_temporary"
fi
TRIPOSPLAT_ERROR_PREFIX="TripoSplat source installer"
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || triposplat_fail "this provider supports Linux x86_64 only"
triposplat_print_host_inventory
triposplat_detect_backends
triposplat_select_backend "$requested_backend"
backend="$TRIPOSPLAT_SELECTED_BACKEND"
worker_dir="$destination/contrib/ComputeWorkers/MutualGPU/linux"
if "$dry_run"; then
  echo "would clone $repository at $ref into $destination"
  echo "would run npm ci and frozen uv sync for $backend"
  echo "would run the final $backend PyTorch GPU probe before allowing a foreground run"
  "$download_models" && echo "would download five SHA-256-pinned model files into $model_dir"
  "$run_foreground" && echo "would securely obtain an actual provider key and start the foreground worker"
  exit 0
fi
command -v git >/dev/null || triposplat_fail "git is required"
command -v npm >/dev/null || triposplat_fail "npm is required"
command -v uv >/dev/null || triposplat_fail "uv 0.11 is required"
if [[ -e "$destination" ]]; then triposplat_fail "destination already exists: $destination"; fi
git clone --depth 1 --branch "$ref" "$repository" "$destination"
# A source checkout can come from an archive, a filesystem that does not
# retain Git modes, or a repository whose executable bits were normalized by
# an intermediary.  The command printed below must be runnable without asking
# the operator to repair permissions themselves.
chmod 0755 "$worker_dir/run-worker.sh"
npm ci --prefix "$worker_dir"
triposplat_sync_frozen_environment "$worker_dir/runtime/$backend"
"$worker_dir/runtime/$backend/.venv/bin/python" "$worker_dir/install/probe-pytorch.py" --backend "$backend"
if "$download_models"; then triposplat_download_models "$worker_dir/install/model-download.py" "$worker_dir/model-manifest.json" "$model_dir"; fi

if ! "$run_foreground"; then
  echo "Source installation complete; the foreground worker was not started (--no-run)."
  exit 0
fi

if [[ -z "$provider_key_file" ]]; then
  provider_key_file="${XDG_CONFIG_HOME:-$HOME/.config}/mutualgpu/triposplat/provider.key"
fi

source_provider_key_file() {
  local file="$1" mode key
  [[ -f "$file" && -O "$file" && -r "$file" ]] || triposplat_fail "--provider-key-file must name a readable regular file owned by this user"
  mode="$(triposplat_mode "$file")" || triposplat_fail "could not inspect provider key file mode"
  [[ "$mode" =~ ^[0-7]+$ ]] && (( (8#$mode & 8#077) == 0 )) || triposplat_fail "provider key file must be chmod 600"
  key="$(<"$file")"
  [[ -n "$key" && ! "$key" =~ [[:space:]] ]] || triposplat_fail "provider key file must contain exactly one non-whitespace provider key"
}

if [[ -e "$provider_key_file" ]]; then
  source_provider_key_file "$provider_key_file"
elif "$provider_key_supplied"; then
  triposplat_fail "--provider-key-file does not exist: $provider_key_file"
else
  key_directory="$(dirname -- "$provider_key_file")"
  [[ "$key_directory" == /* ]] || triposplat_fail "XDG_CONFIG_HOME must be an absolute path"
  mkdir -p "$key_directory"
  chmod 0700 "$key_directory"
  triposplat_private_directory "$key_directory" || triposplat_fail "the provider-key directory must be private and owned by this user"
  [[ -r /dev/tty && -w /dev/tty ]] || triposplat_fail "an actual MutualGPU provider key is required; rerun from an interactive terminal with --provider-key-file PATH"
  printf 'Paste the actual MutualGPU provider key (input is hidden): ' >/dev/tty
  IFS= read -r -s provider_key </dev/tty || triposplat_fail "could not read the provider key from the terminal"
  printf '\n' >/dev/tty
  [[ -n "$provider_key" && ! "$provider_key" =~ [[:space:]] ]] || triposplat_fail "the provider key must be one non-whitespace value"
  provider_key_temporary="$(mktemp "$key_directory/.provider-key.XXXXXX")"
  printf '%s\n' "$provider_key" >"$provider_key_temporary"
  unset provider_key
  chmod 0600 "$provider_key_temporary"
  mv -f "$provider_key_temporary" "$provider_key_file"
  echo "[credentials] stored the provider key at a private local path; its contents were not echoed or logged."
fi

echo "Source installation complete. Starting the foreground worker; Ctrl-C stops it."
exec "$worker_dir/run-worker.sh" run --provider-key-file "$provider_key_file" --backend "$backend" --model-dir "$model_dir" --api-url "$api_url"
