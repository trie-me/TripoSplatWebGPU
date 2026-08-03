#!/usr/bin/env bash
# Shared, dependency-free installer helpers. This file is intentionally kept
# shell-only so the bootstrap can explain the host before it downloads anything.

triposplat_fail() {
  echo "${TRIPOSPLAT_ERROR_PREFIX:-TripoSplat installer}: $1" >&2
  exit 2
}

triposplat_command() {
  command -v "$1" >/dev/null 2>&1
}

triposplat_mode() {
  stat -c '%a' "$1" 2>/dev/null || stat -f '%Lp' "$1" 2>/dev/null
}

triposplat_owner() {
  stat -c '%u' "$1" 2>/dev/null || stat -f '%u' "$1" 2>/dev/null
}

triposplat_private_directory() {
  local directory="$1" mode owner
  [[ -d "$directory" ]] || return 1
  owner="$(triposplat_owner "$directory")" || return 1
  mode="$(triposplat_mode "$directory")" || return 1
  [[ "$owner" == "$(id -u)" ]] || return 1
  [[ "$mode" =~ ^[0-7]+$ ]] || return 1
  (( (8#$mode & 8#022) == 0 ))
}

triposplat_canonical_directory() {
  (cd -- "$1" && pwd -P)
}

triposplat_resolve_shim_directory() {
  local requested="${1:-}" entry canonical fallback
  TRIPOSPLAT_SHIM_SOURCE="PATH"
  if [[ -n "$requested" ]]; then
    [[ "$requested" == /* ]] || triposplat_fail "--shim-dir must be an absolute path"
    triposplat_private_directory "$requested" || triposplat_fail "--shim-dir must name an existing directory owned by this user and not writable by group or other users"
    TRIPOSPLAT_SHIM_DIR="$(triposplat_canonical_directory "$requested")"
    TRIPOSPLAT_SHIM_SOURCE="override"
    return
  fi

  local old_ifs="$IFS"
  IFS=:
  for entry in $PATH; do
    [[ "$entry" == /* ]] || continue
    if triposplat_private_directory "$entry"; then
      canonical="$(triposplat_canonical_directory "$entry")"
      TRIPOSPLAT_SHIM_DIR="$canonical"
      IFS="$old_ifs"
      return
    fi
  done
  IFS="$old_ifs"

  [[ "$HOME" == /* ]] || triposplat_fail "HOME must be an absolute path to select a safe command directory"
  fallback="$HOME/.local/bin"
  # A conventional fallback is only used when HOME itself is private. It is
  # deliberately reported as a fallback because the operator may need to add
  # it to PATH before the commands are available in a new shell.
  triposplat_private_directory "$HOME" || triposplat_fail "no safe user-writable directory was found in PATH, and HOME is not private enough for the ~/.local/bin fallback"
  TRIPOSPLAT_SHIM_DIR="$fallback"
  TRIPOSPLAT_SHIM_SOURCE="fallback"
}

triposplat_print_host_inventory() {
  echo "[host] platform: $(uname -s) $(uname -m)"
  if triposplat_command inxi; then
    echo "[host] inxi graphics inventory:"
    inxi -Gxx 2>/dev/null || echo "[host] inxi could not read graphics inventory."
  else
    echo "[host] inxi not installed; continuing with lspci and sysfs inventory."
  fi
  if triposplat_command lspci; then
    echo "[host] PCI display devices:"
    lspci -nnk 2>/dev/null | awk '/VGA compatible controller|3D controller|Display controller/ { print "[host]   " $0 }' || true
  else
    echo "[host] lspci not installed; continuing with sysfs inventory."
  fi

  local device vendor product
  for device in /sys/class/drm/card*/device; do
    [[ -r "$device/vendor" ]] || continue
    vendor="$(<"$device/vendor")"
    product="unknown"
    [[ -r "$device/device" ]] && product="$(<"$device/device")"
    case "$vendor" in
      0x10de) echo "[host] sysfs GPU: NVIDIA vendor=$vendor device=$product" ;;
      0x1002) echo "[host] sysfs GPU: AMD vendor=$vendor device=$product" ;;
      *) echo "[host] sysfs GPU: vendor=$vendor device=$product" ;;
    esac
  done

  if triposplat_command vulkaninfo && vulkaninfo --summary >/dev/null 2>&1; then
    echo "[host] optional Vulkan diagnostics: vulkaninfo --summary is available (not used for inference)."
  else
    echo "[host] optional Vulkan diagnostics: unavailable; this does not block native CUDA or ROCm inference."
  fi
}

triposplat_detect_backends() {
  TRIPOSPLAT_CUDA_READY=false
  TRIPOSPLAT_ROCM_READY=false
  local nvidia_summary rocm_manager=""

  if triposplat_command nvidia-smi; then
    if nvidia_summary="$(nvidia-smi -L 2>/dev/null)" && [[ -n "$nvidia_summary" ]]; then
      TRIPOSPLAT_CUDA_READY=true
      echo "[cuda] NVIDIA driver readiness: nvidia-smi reports a GPU."
      nvidia_summary="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
      [[ -n "$nvidia_summary" ]] && echo "[cuda] detected: $nvidia_summary"
    else
      echo "[cuda] NVIDIA hardware may be present, but nvidia-smi could not access a usable driver. No driver will be installed automatically."
    fi
  else
    echo "[cuda] nvidia-smi is unavailable; CUDA is not a viable default."
  fi

  local rocm_agents=false rocm_management=false
  if triposplat_command rocminfo && rocminfo >/dev/null 2>&1; then
    rocm_agents=true
    echo "[rocm] ROCm compute readiness: rocminfo reports an accessible agent."
  elif triposplat_command rocminfo; then
    echo "[rocm] rocminfo is installed but cannot access an AMD ROCm agent. No driver will be installed automatically."
  else
    echo "[rocm] rocminfo is unavailable; ROCm is not a viable default."
  fi
  if triposplat_command amd-smi && amd-smi list >/dev/null 2>&1; then
    rocm_management=true
    rocm_manager="amd-smi"
  elif triposplat_command rocm-smi && rocm-smi >/dev/null 2>&1; then
    rocm_management=true
    rocm_manager="rocm-smi"
  fi
  if [[ "$rocm_management" == true ]]; then
    echo "[rocm] AMD management readiness: $rocm_manager can inspect the device."
  else
    echo "[rocm] neither amd-smi nor rocm-smi could inspect an AMD device."
  fi
  if [[ "$rocm_agents" == true && "$rocm_management" == true ]]; then TRIPOSPLAT_ROCM_READY=true; fi
}

triposplat_select_backend() {
  local requested="$1"
  case "$requested" in
    cuda)
      [[ "$TRIPOSPLAT_CUDA_READY" == true ]] || triposplat_fail "--backend cuda was requested, but NVIDIA readiness failed. Install or repair the driver yourself, then rerun; this installer never invokes sudo or installs drivers."
      TRIPOSPLAT_SELECTED_BACKEND="cuda"
      ;;
    rocm)
      [[ "$TRIPOSPLAT_ROCM_READY" == true ]] || triposplat_fail "--backend rocm was requested, but rocminfo plus amd-smi/rocm-smi did not report a ready AMD device. Install or repair ROCm yourself, then rerun; this installer never invokes sudo or installs drivers."
      TRIPOSPLAT_SELECTED_BACKEND="rocm"
      ;;
    auto)
      if [[ "$TRIPOSPLAT_CUDA_READY" == true && "$TRIPOSPLAT_ROCM_READY" == true ]]; then
        triposplat_fail "both CUDA and ROCm look viable. Refusing to silently guess on a mixed-backend host; rerun with --backend cuda or --backend rocm."
      elif [[ "$TRIPOSPLAT_CUDA_READY" == true ]]; then
        TRIPOSPLAT_SELECTED_BACKEND="cuda"
      elif [[ "$TRIPOSPLAT_ROCM_READY" == true ]]; then
        TRIPOSPLAT_SELECTED_BACKEND="rocm"
      else
        triposplat_fail "no usable NVIDIA CUDA or AMD ROCm runtime was detected. Review the readiness report above; this installer does not install drivers or use sudo."
      fi
      ;;
    *) triposplat_fail "--backend must be auto, cuda, or rocm" ;;
  esac
  echo "[backend] selected $TRIPOSPLAT_SELECTED_BACKEND${requested:+ (requested $requested)}."
}

triposplat_sync_frozen_environment() {
  local project="$1"
  # SSL_CERT_FILE and SSL_CERT_DIR override uv's normal trust source. Start
  # from the Linux system store so a stale shell override cannot break a
  # bootstrap on a host whose system store already trusts its network proxy.
  if env -u SSL_CERT_FILE -u SSL_CERT_DIR uv sync --project "$project" --frozen --no-dev --system-certs; then return; fi
  if [[ -n "${SSL_CERT_FILE:-}" || -n "${SSL_CERT_DIR:-}" ]]; then
    echo "[tls] system trust did not validate the download; retrying with the explicitly configured certificate bundle." >&2
    uv sync --project "$project" --frozen --no-dev --system-certs
    return
  fi
  triposplat_fail "uv could not validate its download with the Linux system trust store. Repair the host certificate trust; this installer will not disable TLS verification."
}

triposplat_download_models() {
  local downloader="$1" manifest="$2" model_directory="$3"
  env -u SSL_CERT_FILE -u SSL_CERT_DIR python3 "$downloader" --manifest "$manifest" --model-dir "$model_directory"
}

triposplat_write_config() {
  local config_directory="$1" install_root="$2" selected_backend="$3" model_directory="$4" provider_key_file="$5" api_url="$6"
  local config_file temporary
  config_file="$config_directory/config.env"
  temporary="$(mktemp "$config_directory/.config.env.XXXXXX")"
  {
    printf '%s=%q\n' "TRIPOSPLAT_INSTALL_ROOT" "$install_root"
    printf '%s=%q\n' "MUTUALGPU_TRIPOSPLAT_BACKEND" "$selected_backend"
    printf '%s=%q\n' "MUTUALGPU_TRIPOSPLAT_MODEL_DIR" "$model_directory"
    printf '%s=%q\n' "MUTUALGPU_TRIPOSPLAT_MODEL_MANIFEST" "$install_root/current/model-manifest.json"
    printf '%s=%q\n' "MUTUALGPU_API_URL" "$api_url"
    [[ -n "$provider_key_file" ]] && printf '%s=%q\n' "MUTUALGPU_PROVIDER_KEY_FILE" "$provider_key_file"
  } >"$temporary"
  chmod 0600 "$temporary"
  mv -f "$temporary" "$config_file"
}

triposplat_write_launcher() {
  local config_directory="$1" launcher temporary
  launcher="$config_directory/launcher"
  temporary="$(mktemp "$config_directory/.launcher.XXXXXX")"
  cat >"$temporary" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

config_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
config_file="$config_directory/config.env"
fail() { echo "TripoSplat launcher: $1" >&2; exit 2; }
[[ -f "$config_file" && -O "$config_file" ]] || fail "configuration is missing or is not owned by this user"
mode="$(stat -c '%a' "$config_file" 2>/dev/null || stat -f '%Lp' "$config_file" 2>/dev/null || true)"
[[ "$mode" =~ ^[0-7]+$ ]] && (( (8#$mode & 8#077) == 0 )) || fail "configuration must not be readable by group or other users"
# The installer writes this private, non-secret file using Bash-safe quoting.
# shellcheck disable=SC1090
source "$config_file"
export MUTUALGPU_TRIPOSPLAT_BACKEND MUTUALGPU_TRIPOSPLAT_MODEL_DIR MUTUALGPU_TRIPOSPLAT_MODEL_MANIFEST MUTUALGPU_API_URL
export MUTUALGPU_PROVIDER_KEY_FILE="${MUTUALGPU_PROVIDER_KEY_FILE:-}"
command="${1:-run}"
if [[ "$command" == "run" || "$command" == "restart" ]]; then
  [[ $# -gt 0 ]] && shift
else
  fail "usage: mutualgpu-triposplat [run|restart] [worker options]"
fi
if [[ "$command" == "restart" ]]; then
  echo "[launcher] no daemon is installed; restart starts a fresh foreground worker."
fi
[[ -x "$TRIPOSPLAT_INSTALL_ROOT/current/run-worker.sh" ]] || fail "the active version is missing; rerun the immutable installer or roll back"
exec "$TRIPOSPLAT_INSTALL_ROOT/current/run-worker.sh" run "$@"
EOF
  chmod 0700 "$temporary"
  mv -f "$temporary" "$launcher"
}

triposplat_write_shim() {
  local destination="$1" launcher="$2" mode="$3" temporary marker="# Managed by MutualGPU TripoSplat installer"
  if [[ -e "$destination" || -L "$destination" ]]; then
    [[ -f "$destination" ]] && grep -Fqx "$marker" "$destination" || triposplat_fail "refusing to overwrite an unrelated command: $destination"
  fi
  temporary="$(mktemp "$(dirname -- "$destination")/.mutualgpu-triposplat.XXXXXX")"
  {
    printf '%s\n%s\n%s\n' '#!/usr/bin/env bash' "$marker" 'set -euo pipefail'
    if [[ "$mode" == "restart" ]]; then
      printf 'exec %q restart "$@"\n' "$launcher"
    else
      printf 'exec %q "$@"\n' "$launcher"
    fi
  } >"$temporary"
  chmod 0755 "$temporary"
  mv -f "$temporary" "$destination"
}
