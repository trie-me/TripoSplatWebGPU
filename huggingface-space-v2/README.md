---
title: TripoSplat WebGPU V2 · Exact Mac 20-Step
short_description: Bit-exact 20-step TripoSplat runner for Apple Silicon.
emoji: 🫧
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
models:
  - VAST-AI/TripoSplat
  - Yosun/TripoSplat-WebGPU
---

# TripoSplat WebGPU V2

This v2 runner fixes the 20-step quality path on Apple Silicon. Image conditioning, octree construction, Gaussian decoding, preview, and export remain in browser WebGPU. The complete 20-step / 40-call flow sampler runs through the untouched official fp32 PyTorch model on the visitor's local Mac MPS device.

The unchanged official fixture passes bit-for-bit:

- latent maximum error: **0**
- camera maximum error: **0**
- qualification gate: **passed**
- strict gate: **passed**
- recorded Apple M3 Max flow inference: **347.09 seconds**
- historical conservative WebGPU flow: **676.67 seconds**

[Machine-readable validation report](./validation/2026-07-18-flow20-mac-mps-service.json)

Hugging Face hosts only this static runner. It does not receive the source image or provide inference compute. The authenticated flow request stays on `127.0.0.1`; the local service rejects non-loopback access, incorrect source revisions, incorrect weight hashes, and unqualified dependency versions.

## Start on a Mac

Requirements: Apple Silicon macOS, desktop Chrome with WebGPU, Git, and Python 3.12.

```bash
git clone https://huggingface.co/spaces/Yosun/TripoSplat-WebGPU-v2
cd TripoSplat-WebGPU-v2/mac

# One-time official source, dependency, and flow-weight setup.
bash scripts/triposplat/setup_mac_20_step_v2.sh

# Starts the authenticated MPS service and opens this hosted runner.
bash scripts/triposplat/start_mac_20_step_v2.sh
```

Keep the launcher terminal open while generating. The startup token is moved into tab-scoped session storage and removed from the address bar. The browser skips the 1.64 GB WebGPU DiT artifact, but still verifies and caches the other browser model stages.

## Scope

The complete flow boundary is exact on the recorded fixture. The surrounding browser DINO, VAE, octree, and Gaussian stages retain their separately recorded validation status; this release does not claim whole-scene or rendered-pixel parity for arbitrary images. The portable v1 WebGPU DiT fallback remains available in the source repository but is deliberately disabled in this v2 runner because its accumulated 20-step latent drift fails the official gate.

[Source and technical status](https://github.com/yosun/TripoSplatWebGPU) · [Official TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)
