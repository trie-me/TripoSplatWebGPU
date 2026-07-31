# Native Linux MutualGPU provider for TripoSplat

This contribution adds a standalone, Linux-native provider for the existing `tripo-splat` MutualGPU capability. It keeps the official TripoSplat PyTorch implementation as the numerical source of truth and serves as a fallback for Linux browsers whose WebGPU-over-Vulkan path is unavailable or unreliable.

NVIDIA uses native CUDA; AMD uses native ROCm/HIP. Vulkan is captured only as optional browser-compatibility diagnostics through `vulkaninfo --summary`; it is not an inference dependency. No Windows, DirectML/DX12, macOS, Metal, or MPS runtime is included.

The provider is deliberately an ad hoc foreground process. It installs no daemon, `systemd` unit, launch agent, cron job, or persistent container service.

## Branch bootstrap

The following command is a **development bootstrap** for this branch. It clones the branch, selects CUDA or ROCm, uses the committed frozen environment, and optionally downloads the five checksum-pinned TripoSplat weights. It is curlable after this branch is pushed; it is not a substitute for the immutable release installer below.

```bash
curl --proto '=https' --tlsv1.2 -LsSf \
  https://raw.githubusercontent.com/trie-me/TripoSplatWebGPU/codex/native-linux-provider/contrib/mutualgpu-triposplat-provider/install/install-from-source.sh \
  | bash -s -- \
    --ref codex/native-linux-provider \
    --backend cuda \
    --download-models
```

It prints the exact foreground command after setup. Store the provider key in a one-line `chmod 600` file and run:

```bash
~/.local/share/triposplat-webgpu-provider/source/contrib/mutualgpu-triposplat-provider/run-worker.sh run \
  --provider-key-file /secure/provider.key \
  --backend cuda
```

The key is read only by the Node controller, never accepted as a command-line value, never logged, and never passed into the Python GPU subprocess.

## Immutable release installation

For a published `triposplat-vX.Y.Z` release, use the release asset bootstrap:

```bash
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/trie-me/TripoSplatWebGPU/releases/download/triposplat-v0.1.0/install.sh \
  | bash -s -- --version triposplat-v0.1.0 --backend cuda --download-models
```

That bootstrap verifies the release archive SHA-256 before activation, maintains the previous version for rollback, uses the relevant `uv.lock`, and never resolves dependencies while serving. The verify-then-run procedure is documented in [install/install.sh](install/install.sh); it is the preferred path for production. A release is not created by this branch alone.

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
