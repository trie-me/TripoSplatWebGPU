#!/usr/bin/env bash
set -euo pipefail
umask 077

version=""
backend="auto"
download_models=false
dry_run=false
yes=false
uninstall=false
rollback_version=""
data_root="${XDG_DATA_HOME:-$HOME/.local/share}/mutualgpu/triposplat"
config_root="${XDG_CONFIG_HOME:-$HOME/.config}/mutualgpu/triposplat"
bin_dir="${HOME}/.local/bin"
model_root="${XDG_DATA_HOME:-$HOME/.local/share}/mutualgpu/triposplat/models"
release_base="${MUTUALGPU_TRIPOSPLAT_RELEASE_BASE_URL:-https://github.com/trie-me/TripoSplatWebGPU/releases/download}"
provider_key_file=""

usage() {
  cat <<'EOF'
Usage: install.sh --version triposplat-v0.1.0 [options]
  --backend auto|cuda|rocm       Select frozen PyTorch runtime (default: auto)
  --install-dir PATH            Version root (default: XDG data directory)
  --model-dir PATH              Model root (default: XDG data directory)
  --provider-key-file PATH      Record no secret; prints the subsequent run command
  --download-models             Fetch and SHA-256 verify the 3.78 GB pinned models
  --dry-run                     Print the exact immutable actions without changing files
  --rollback VERSION            Atomically activate an already installed version
  --uninstall                   Remove one installed version (requires --version)
  --yes                         Do not ask before destructive uninstall

The bootstrap is designed for an immutable GitHub release. It verifies the
release archive SHA-256 before activation. For a verify-then-run workflow,
download this script and its published SHA-256 first, verify it locally, then
run: bash install.sh --version ... . The bootstrap itself cannot verify bytes
before bash evaluates them, which is the tradeoff of curl | bash.
EOF
}
fail() { echo "TripoSplat installer: $1" >&2; exit 2; }
checksum() { if command -v sha256sum >/dev/null; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
safe_version() { [[ "$1" =~ ^triposplat-v[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "--version must be an immutable triposplat-vX.Y.Z tag"; }

while (($#)); do
  case "$1" in
    --version) version="${2:-}"; shift ;;
    --backend) backend="${2:-}"; shift ;;
    --install-dir) data_root="${2:-}"; shift ;;
    --model-dir) model_root="${2:-}"; shift ;;
    --provider-key-file) provider_key_file="${2:-}"; shift ;;
    --download-models) download_models=true ;;
    --dry-run) dry_run=true ;;
    --uninstall) uninstall=true ;;
    --rollback) rollback_version="${2:-}"; shift ;;
    --yes) yes=true ;;
    --help|-h) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
  shift
done

[[ "$(uname -s)" == "Linux" ]] || fail "this installer supports Linux only"
[[ "$(uname -m)" == "x86_64" ]] || fail "this release currently supports Linux x86_64 only"
[[ "$backend" == "auto" || "$backend" == "cuda" || "$backend" == "rocm" ]] || fail "--backend must be auto, cuda, or rocm"
if [[ -n "$rollback_version" ]]; then
  safe_version "$rollback_version"
  target="$data_root/versions/$rollback_version"
  [[ -d "$target" ]] || fail "rollback version is not installed"
  if "$dry_run"; then echo "would atomically set $data_root/current -> $target"; exit 0; fi
  mkdir -p "$data_root"
  ln -s "versions/$rollback_version" "$data_root/.current-new"
  mv -Tf "$data_root/.current-new" "$data_root/current"
  echo "Activated $rollback_version."
  exit 0
fi
[[ -n "$version" ]] || fail "--version is required"
safe_version "$version"
version_dir="$data_root/versions/$version"
if "$uninstall"; then
  [[ -d "$version_dir" ]] || fail "version is not installed"
  if ! "$yes"; then read -r -p "Remove $version_dir? [y/N] " answer; [[ "$answer" == y || "$answer" == Y ]] || exit 0; fi
  rm -rf -- "$version_dir"
  if [[ "$(readlink "$data_root/current" 2>/dev/null || true)" == "versions/$version" ]]; then rm -f -- "$data_root/current"; fi
  echo "Removed $version; model files remain in $model_root."
  exit 0
fi
if [[ "$backend" == auto ]]; then
  if command -v rocminfo >/dev/null 2>&1 && rocminfo >/dev/null 2>&1; then backend=rocm
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then backend=cuda
  else fail "could not detect a usable NVIDIA CUDA or AMD ROCm runtime"; fi
fi
asset_version="${version#triposplat-}"
archive_name="mutualgpu-triposplat-${asset_version}-linux-x86_64.tar.gz"
archive_url="$release_base/$version/$archive_name"
checksum_url="$release_base/$version/$archive_name.sha256"
if "$dry_run"; then
  echo "would download immutable archive: $archive_url"
  echo "would verify release checksum: $checksum_url"
  echo "would install: $version_dir"
  echo "would sync frozen $backend environment: $version_dir/runtime/$backend/uv.lock"
  "$download_models" && echo "would download five SHA-256-pinned model files into $model_root/de3b99ab2627d565a8d5fc40f2db52557b82b974"
  [[ -n "$provider_key_file" ]] && echo "next command: mutualgpu-triposplat run --provider-key-file $provider_key_file --backend $backend"
  exit 0
fi
command -v curl >/dev/null || fail "curl is required"
command -v tar >/dev/null || fail "tar is required"
command -v uv >/dev/null || fail "uv 0.11 is required for the frozen Python environment"
temporary="$(mktemp -d "${TMPDIR:-/tmp}/mutualgpu-triposplat.XXXXXX")"
cleanup() { rm -rf -- "$temporary"; }
trap cleanup EXIT
curl --fail --location --proto '=https' --tlsv1.2 --retry 3 --output "$temporary/$archive_name" "$archive_url"
curl --fail --location --proto '=https' --tlsv1.2 --retry 3 --output "$temporary/$archive_name.sha256" "$checksum_url"
expected="$(awk '{print $1}' "$temporary/$archive_name.sha256")"
[[ "$expected" =~ ^[a-fA-F0-9]{64}$ ]] || fail "release checksum file is invalid"
[[ "$(checksum "$temporary/$archive_name")" == "$expected" ]] || fail "release archive checksum mismatch"
tar -tzf "$temporary/$archive_name" | awk 'BEGIN { bad=0 } $0 ~ /^\// || $0 ~ /(^|\/)\.\.($|\/)/ { bad=1 } END { exit bad }' || fail "release archive contains an unsafe path"
mkdir -p "$data_root/versions" "$bin_dir" "$config_root"
stage="$data_root/versions/.${version}.staging.$$"
rm -rf -- "$stage"
mkdir -p "$stage"
tar -xzf "$temporary/$archive_name" -C "$stage" --strip-components=1
[[ -f "$stage/runtime/$backend/uv.lock" && -f "$stage/run-worker.sh" && -f "$stage/install/model-download.py" ]] || fail "release archive is incomplete"
chmod 0755 "$stage/run-worker.sh" "$stage/install/model-download.py"
uv sync --project "$stage/runtime/$backend" --frozen --no-dev
if [[ -e "$version_dir" ]]; then rm -rf -- "$stage"; echo "$version is already installed."; else mv "$stage" "$version_dir"; fi
printf '%s\n' "$model_root/de3b99ab2627d565a8d5fc40f2db52557b82b974" >"$version_dir/model-dir"
ln -s "versions/$version" "$data_root/.current-new"
mv -Tf "$data_root/.current-new" "$data_root/current"
{
  printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail'
  printf 'exec %q "${@}"\n' "$data_root/current/run-worker.sh"
} >"$bin_dir/mutualgpu-triposplat"
chmod 0755 "$bin_dir/mutualgpu-triposplat"
if "$download_models"; then
  python3 "$version_dir/install/model-download.py" --manifest "$version_dir/model-manifest.json" --model-dir "$model_root/de3b99ab2627d565a8d5fc40f2db52557b82b974"
fi
echo "Installed and activated $version ($backend)."
echo "Run in the foreground: mutualgpu-triposplat run --provider-key-file /path/to/provider.key --backend $backend"
[[ -n "$provider_key_file" ]] && echo "Requested key file remains local and was not copied: $provider_key_file"
