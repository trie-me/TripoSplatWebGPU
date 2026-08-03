#!/usr/bin/env bash
set -euo pipefail
umask 077

installer_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$installer_dir/lib.sh"

version=""
requested_backend="auto"
download_models=false
dry_run=false
yes=false
uninstall=false
rollback_version=""
data_root="${XDG_DATA_HOME:-$HOME/.local/share}/mutualgpu/triposplat"
config_root="${XDG_CONFIG_HOME:-$HOME/.config}/mutualgpu/triposplat"
model_root="${XDG_DATA_HOME:-$HOME/.local/share}/mutualgpu/triposplat/models"
release_base="${MUTUALGPU_TRIPOSPLAT_RELEASE_BASE_URL:-https://github.com/trie-me/TripoSplatWebGPU/releases/download}"
provider_key_file=""
shim_dir_override=""
api_url="${MUTUALGPU_API_URL:-https://mutualgpu.com}"

usage() {
  cat <<'EOF'
Usage: install.sh --version triposplat-v0.1.0 [options]
  --backend auto|cuda|rocm       Safely infer a ready PyTorch backend, or select one explicitly (default: auto)
  --install-dir PATH            Versioned installation root (default: XDG data directory)
  --config-dir PATH             Configuration and launcher root (default: XDG config directory)
  --model-dir PATH              Model root (default: XDG data directory)
  --shim-dir PATH               Existing private user-owned PATH directory for user-global commands
  --provider-key-file PATH      Remember only this private key-file path; never copies or reads the key
  --api-url HTTPS_URL           MutualGPU endpoint (default: https://mutualgpu.com)
  --download-models             Fetch and SHA-256 verify the 3.78 GB pinned models
  --dry-run                     Scan and explain the host and print immutable actions without changing files
  --rollback VERSION            Atomically activate an already installed version
  --uninstall                   Remove one installed version (requires --version)
  --yes                         Do not ask before destructive uninstall

The installer inventories graphics hardware through inxi, lspci, and sysfs;
uses nvidia-smi for NVIDIA readiness; and uses rocminfo plus amd-smi/rocm-smi
for AMD readiness. It does not install drivers, invoke sudo, or require Vulkan.
After the frozen environment is created it runs a backend-specific PyTorch GPU
probe. Mixed ready CUDA and ROCm hosts must select --backend explicitly.

The bootstrap is designed for an immutable GitHub release. It verifies the
release archive SHA-256 before activation. For a verify-then-run workflow,
download this script and its published SHA-256 first, verify it locally, then
run: bash install.sh --version ... . The bootstrap itself cannot verify bytes
before bash evaluates them, which is the tradeoff of curl | bash.
EOF
}

checksum() {
  if triposplat_command sha256sum; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}

safe_version() {
  [[ "$1" =~ ^triposplat-v[0-9]+\.[0-9]+\.[0-9]+$ ]] || triposplat_fail "--version must be an immutable triposplat-vX.Y.Z tag"
}

private_key_file() {
  local file="$1" mode
  [[ -f "$file" && -O "$file" ]] || triposplat_fail "--provider-key-file must name a regular file owned by this user"
  mode="$(triposplat_mode "$file")" || triposplat_fail "could not inspect provider key file mode"
  [[ "$mode" =~ ^[0-7]+$ ]] && (( (8#$mode & 8#077) == 0 )) || triposplat_fail "provider key file must be chmod 600"
}

while (($#)); do
  case "$1" in
    --version) version="${2:-}"; shift ;;
    --backend) requested_backend="${2:-}"; shift ;;
    --install-dir) data_root="${2:-}"; shift ;;
    --config-dir) config_root="${2:-}"; shift ;;
    --model-dir) model_root="${2:-}"; shift ;;
    --shim-dir) shim_dir_override="${2:-}"; shift ;;
    --provider-key-file) provider_key_file="${2:-}"; shift ;;
    --api-url) api_url="${2:-}"; shift ;;
    --download-models) download_models=true ;;
    --dry-run) dry_run=true ;;
    --uninstall) uninstall=true ;;
    --rollback) rollback_version="${2:-}"; shift ;;
    --yes) yes=true ;;
    --help|-h) usage; exit 0 ;;
    *) triposplat_fail "unknown argument: $1" ;;
  esac
  shift
done

[[ "$(uname -s)" == "Linux" ]] || triposplat_fail "this installer supports Linux only"
[[ "$(uname -m)" == "x86_64" ]] || triposplat_fail "this release currently supports Linux x86_64 only"
[[ "$api_url" == https://* ]] || triposplat_fail "--api-url must use https://"

if [[ -n "$rollback_version" ]]; then
  safe_version "$rollback_version"
  target="$data_root/versions/$rollback_version"
  [[ -d "$target" ]] || triposplat_fail "rollback version is not installed"
  if "$dry_run"; then echo "would atomically set $data_root/current -> $target"; exit 0; fi
  mkdir -p "$data_root"
  new_current="$data_root/.current-new.$$"
  ln -s "versions/$rollback_version" "$new_current"
  mv -Tf "$new_current" "$data_root/current"
  echo "Activated $rollback_version. The user-global commands continue to run in the foreground."
  exit 0
fi

[[ -n "$version" ]] || triposplat_fail "--version is required"
safe_version "$version"
version_dir="$data_root/versions/$version"
if "$uninstall"; then
  [[ -d "$version_dir" ]] || triposplat_fail "version is not installed"
  if ! "$yes"; then read -r -p "Remove $version_dir? [y/N] " answer; [[ "$answer" == y || "$answer" == Y ]] || exit 0; fi
  rm -rf -- "$version_dir"
  if [[ "$(readlink "$data_root/current" 2>/dev/null || true)" == "versions/$version" ]]; then rm -f -- "$data_root/current"; fi
  echo "Removed $version; configuration, commands, and model files remain for another installed version or a later reinstall."
  exit 0
fi

triposplat_print_host_inventory
triposplat_detect_backends
triposplat_select_backend "$requested_backend"
backend="$TRIPOSPLAT_SELECTED_BACKEND"
triposplat_resolve_shim_directory "$shim_dir_override"
shim_dir="$TRIPOSPLAT_SHIM_DIR"

asset_version="${version#triposplat-}"
archive_name="mutualgpu-triposplat-${asset_version}-linux-x86_64.tar.gz"
archive_url="$release_base/$version/$archive_name"
checksum_url="$release_base/$version/$archive_name.sha256"
model_revision="de3b99ab2627d565a8d5fc40f2db52557b82b974"
model_dir="$model_root/$model_revision"
if "$dry_run"; then
  echo "[commands] safe user-global shim directory: $shim_dir ($TRIPOSPLAT_SHIM_SOURCE)."
  [[ "$TRIPOSPLAT_SHIM_SOURCE" == "fallback" ]] && echo "[commands] ~/.local/bin is not currently a safe PATH entry; it would be created. Add it to PATH before opening a new shell."
  echo "would download immutable archive: $archive_url"
  echo "would verify release checksum: $checksum_url"
  echo "would install: $version_dir"
  echo "would sync frozen $backend environment: $version_dir/runtime/$backend/uv.lock"
  echo "would run the final $backend PyTorch GPU probe before activation"
  echo "would write non-secret config and executable launcher beneath: $config_root"
  echo "would install foreground commands: $shim_dir/mutualgpu-triposplat and $shim_dir/mutualgpu-triposplat-restart"
  "$download_models" && echo "would download five SHA-256-pinned model files into $model_dir"
  [[ -n "$provider_key_file" ]] && echo "would remember only the key-file path in private configuration; the key bytes would not be copied or logged"
  exit 0
fi

[[ -n "$provider_key_file" ]] && private_key_file "$provider_key_file"
command -v curl >/dev/null || triposplat_fail "curl is required"
command -v tar >/dev/null || triposplat_fail "tar is required"
command -v uv >/dev/null || triposplat_fail "uv 0.11 is required for the frozen Python environment"
if [[ "$TRIPOSPLAT_SHIM_SOURCE" == "fallback" ]]; then
  mkdir -p "$shim_dir"
  chmod 0700 "$shim_dir"
fi
triposplat_private_directory "$shim_dir" || triposplat_fail "the selected command directory is not a safe user-owned location"
mkdir -p "$data_root/versions"
if [[ ! -d "$config_root" ]]; then
  mkdir -p "$config_root"
  chmod 0700 "$config_root"
fi
triposplat_private_directory "$config_root" || triposplat_fail "--config-dir must be a private directory owned by this user"
temporary="$(mktemp -d "${TMPDIR:-/tmp}/mutualgpu-triposplat.XXXXXX")"
stage=""
cleanup() {
  rm -rf -- "$temporary"
  if [[ -n "$stage" && -d "$stage" ]]; then rm -rf -- "$stage"; fi
}
trap cleanup EXIT
curl --fail --location --proto '=https' --tlsv1.2 --retry 3 --output "$temporary/$archive_name" "$archive_url"
curl --fail --location --proto '=https' --tlsv1.2 --retry 3 --output "$temporary/$archive_name.sha256" "$checksum_url"
expected="$(awk '{print $1}' "$temporary/$archive_name.sha256")"
[[ "$expected" =~ ^[a-fA-F0-9]{64}$ ]] || triposplat_fail "release checksum file is invalid"
[[ "$(checksum "$temporary/$archive_name")" == "$expected" ]] || triposplat_fail "release archive checksum mismatch"
tar -tzf "$temporary/$archive_name" | awk 'BEGIN { bad=0 } $0 ~ /^\// || $0 ~ /(^|\/)\.\.($|\/)/ { bad=1 } END { exit bad }' || triposplat_fail "release archive contains an unsafe path"
stage="$data_root/versions/.${version}.staging.$$"
rm -rf -- "$stage"
mkdir -p "$stage"
tar -xzf "$temporary/$archive_name" -C "$stage" --strip-components=1
[[ -f "$stage/runtime/$backend/uv.lock" && -f "$stage/run-worker.sh" && -f "$stage/install/model-download.py" && -f "$stage/install/probe-pytorch.py" ]] || triposplat_fail "release archive is incomplete"
chmod 0755 "$stage/run-worker.sh" "$stage/install/model-download.py" "$stage/install/probe-pytorch.py"
triposplat_sync_frozen_environment "$stage/runtime/$backend"
"$stage/runtime/$backend/.venv/bin/python" "$stage/install/probe-pytorch.py" --backend "$backend"
if [[ -e "$version_dir" ]]; then
  rm -rf -- "$stage"
  echo "$version is already installed; keeping its immutable bytes."
else
  mv "$stage" "$version_dir"
fi
printf '%s\n' "$model_dir" >"$version_dir/model-dir"
triposplat_write_config "$config_root" "$data_root" "$backend" "$model_dir" "$provider_key_file" "$api_url"
triposplat_write_launcher "$config_root"
triposplat_write_shim "$shim_dir/mutualgpu-triposplat" "$config_root/launcher" "run"
triposplat_write_shim "$shim_dir/mutualgpu-triposplat-restart" "$config_root/launcher" "restart"
new_current="$data_root/.current-new.$$"
ln -s "versions/$version" "$new_current"
mv -Tf "$new_current" "$data_root/current"
if "$download_models"; then
  triposplat_download_models "$version_dir/install/model-download.py" "$version_dir/model-manifest.json" "$model_dir"
fi
echo "Installed and activated $version ($backend)."
echo "Foreground run command: mutualgpu-triposplat"
echo "Foreground fresh-start command: mutualgpu-triposplat-restart"
if [[ -z "$provider_key_file" ]]; then
  echo "Before the first run, create a chmod-600 provider-key file and pass its path once:"
  echo "  mutualgpu-triposplat run --provider-key-file /secure/provider.key"
else
  echo "The private config remembers only the provider-key file path; no provider key was copied or logged."
fi
