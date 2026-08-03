# Native Linux MutualGPU provider for TripoSplat

This contribution adds a standalone, Linux-native provider for the existing `tripo-splat` MutualGPU capability. It keeps the official TripoSplat PyTorch implementation as the numerical source of truth and serves as a fallback for Linux browsers whose WebGPU-over-Vulkan path is unavailable or unreliable.

NVIDIA uses native CUDA; AMD uses native ROCm/HIP. Vulkan is captured only as optional browser-compatibility diagnostics through `vulkaninfo --summary`; it is not an inference dependency. No Windows, DirectML/DX12, macOS, Metal, or MPS runtime is included.

The provider is deliberately an ad hoc foreground process. It installs no daemon, `systemd` unit, launch agent, cron job, or persistent container service.

## Branch bootstrap

The following command is a **development bootstrap** for this branch. It clones the branch, selects CUDA or ROCm, uses the committed frozen environment, and optionally downloads the five checksum-pinned TripoSplat weights. It is curlable after this branch is pushed; it is not a substitute for the immutable release installer below.

```bash
curl --proto '=https' --tlsv1.2 -LsSf \
  https://raw.githubusercontent.com/trie-me/TripoSplatWebGPU/codex/compute-workers-mutualgpu-linux/contrib/ComputeWorkers/MutualGPU/linux/install/install-from-source.sh \
  | bash -s -- \
    --ref codex/compute-workers-mutualgpu-linux \
    --backend cuda \
    --download-models
```

After GPU and model verification, it invisibly prompts once for the **actual**
MutualGPU provider credential and writes it only to a mode-`0600` file beneath
`${XDG_CONFIG_HOME:-$HOME/.config}/mutualgpu/triposplat/`. It then starts the
foreground worker; Ctrl-C stops it. The key is read only by the Node
controller, never logged, and never passed into the Python GPU subprocess. To
use an existing protected credential instead, add
`--provider-key-file /absolute/path/to/actual-provider.key`; the file must be
owned by the current user and mode `0600`.

## Immutable release installation

Once a `triposplat-vX.Y.Z` release has been published, use the release asset
bootstrap:

```bash
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/trie-me/TripoSplatWebGPU/releases/download/triposplat-v0.1.0/install.sh \
  | bash -s -- --version triposplat-v0.1.0 --backend cuda --download-models
```

That bootstrap verifies the release archive SHA-256 before activation, maintains the previous version for rollback, uses the relevant `uv.lock`, and never resolves dependencies while serving. The verify-then-run procedure is documented in [install/install.sh](install/install.sh); it is the preferred path for production. A release is not created by this branch alone.

`triposplat-v0.1.0` is not published yet, so the example release URL currently
returns 404 and must not be used as an installation command. Until publication,
use the branch development bootstrap above; it retrieves the matching helper
only when the bootstrap itself is piped from GitHub.

## Host-aware foreground installation

The immutable installer first explains the Linux host rather than hiding backend
selection in environment variables. It reports graphics inventory from `inxi`,
`lspci`, and sysfs when available, then checks NVIDIA with `nvidia-smi` and AMD
with both `rocminfo` and `amd-smi` or `rocm-smi`. `vulkaninfo --summary` is an
optional browser diagnostic only; it is never an inference requirement. The
selected frozen environment must also pass its matching CUDA or ROCm/HIP
PyTorch probe before it is activated. The installer never calls `sudo` or
installs drivers.

For Python downloads, the installer asks `uv` to use the Linux system
certificate store. This supports hosts whose organization adds a trusted proxy
or root CA there, without disabling TLS verification. It clears inherited
`SSL_CERT_FILE` and `SSL_CERT_DIR` for that first attempt because either can
silently override system trust. If a deliberately configured custom CA bundle
is needed, the installer retries with it. If neither trust source validates the
issuer, repair the host trust configuration; do not use an insecure-download
override.

`--backend auto` selects the only ready backend. If both CUDA and ROCm are
ready, it stops and asks for an explicit `--backend cuda` or `--backend rocm`;
it will not silently guess on mixed hardware. `--install-dir`, `--config-dir`,
`--model-dir`, `--shim-dir`, and `--api-url` are explicit path/endpoint
overrides for operators who need them.

On success, non-secret configuration and an executable launcher are written
beneath `${XDG_CONFIG_HOME:-$HOME/.config}/mutualgpu/triposplat`. The installer
scans `PATH` for a private user-owned directory before adding these foreground
commands:

```bash
mutualgpu-triposplat
mutualgpu-triposplat-restart
```

The first command defaults to `run`; `restart` starts a fresh foreground
worker. Neither command controls a daemon or creates any persistence. If a
private `--provider-key-file` is supplied at install time, only its pathname is
kept in mode-`0600` configuration; the key bytes are never copied, logged, or
passed to Python. Otherwise pass the protected file path on the first run:

```bash
mutualgpu-triposplat run --provider-key-file /absolute/path/to/actual-provider.key
```

See [MIGRATION.md](MIGRATION.md) for the canary test gates and the exact
software/service rollback procedure. Keep the browser provider available until
the Linux host has completed that qualification matrix.

## Exact provider compatibility

The capability remains exactly `tripo-splat`:

- Required `image_url`: PNG, JPEG, or WebP.
- Defaults: 262,144 Gaussians, 20 steps, guidance 3, PLY, and a random unsigned 32-bit seed when omitted.
- Supported sampling schedules: 4 or 20 steps.
- `enable_safety_checker` must be `false` until a qualified checker exists.
- Both `scene.ply` and `scene.splat` are always emitted, together with `manifest.json` in a store-mode ZIP using the browser-compatible `triposplat-webgpu-result` envelope.

Node validates the HTTPS input descriptor, streaming byte count, SHA-256, and file signature before sending a private local path to Python. Python validates pixel dimensions, verifies all five model SHA-256 values, loads the official pipeline before enrollment, and runs one task at a time. Cancellation or timeout terminates and warms the GPU child before it re-enrolls the provider session. Ambiguous result uploads are not blindly retried.

## Source and licenses

This is intentionally self-contained so it can be cloned from this repository. The narrow MutualGPU provider SDK dependency is vendored under `vendor/mutualgpu-sdk/` under its AGPL-3.0-only terms. The official TripoSplat Python model/pipeline files are vendored under `vendor/triposplat/` at the pinned upstream revision and retain their MIT license. See [NOTICE](NOTICE).

## Checks and qualification

```bash
npm ci
npm run check
uv lock --check --project runtime/cuda
uv lock --check --project runtime/rocm
```

The checks require no model download or GPU. The remaining gate is real Linux qualification: start with the RTX 4080 CUDA matrix (4/20-step inference, output validation, cancellation, reconnect, upload ambiguity, rollback, and a live task from the WebRunner), then run the equal ROCm matrix on supported AMD hardware.
