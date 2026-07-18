# RTX 3090-class 20-step performance and optimization direction

**Status:** community field evidence and engineering plan, updated 2026-07-18. This is not a repository-controlled hardware qualification.

## Field report

A user identifying the device as an RTX 3090 supplied a warm-OPFS public-runner report from Linux and Chrome 148. Chromium exposed the adapter only as `nvidia ampere`, so the report supports the GPU family but does not independently prove the exact SKU.

| Field | Value |
| --- | ---: |
| End-to-end | 472.2 s |
| Schedule | 20 steps / 40 CFG DiT calls |
| Precision | fp32 |
| Cache | OPFS, 6.02 GiB before and after |
| Downloaded this run | 0 B |
| Model phase | 59.23 s |
| Conditioning wall | 69.3 s |
| Sampling wall | 305.0 s |
| Decode wall | 35.95 s |
| Preview/export wall | 2.73 s |
| DINO inference | 3.01 s |
| VAE inference | 1.01 s |
| DiT inference | 301.9 s |
| DiT readback | 0.007 s |
| Octree inference | 3.02 s |
| Gaussian inference | 5.12 s |

The machine-readable transcription is [`benchmarks/2026-07-18-rtx3090-community-warm-opfs-fp32.json`](benchmarks/2026-07-18-rtx3090-community-warm-opfs-fp32.json). The report did not include an official-reference comparison, peak VRAM, ONNX Runtime version, driver version, or a profiler trace.

## What the report establishes

DiT inference consumes 301.9 seconds, approximately 64% of end-to-end wall time. Forty calls average about 7.55 seconds each. Sampling wall exceeds reported DiT inference by only 3.1 seconds, while all DiT output readbacks total 7 milliseconds. The dominant repeat-run cost is therefore fp32 DiT execution, not tensor download or the host CFG/Euler loops.

The `nvidia ampere` adapter and the stage timings provide no sign that the entire model ran through WASM. The public runner configures only the WebGPU execution provider. Per-node profiling is still required before claiming that every operation uses an efficient GPU kernel.

The remaining non-DiT time is material. Subtracting reported neural inference from conditioning and decode leaves about 93 seconds attributable to graph/session setup, artifact access, allocation, compilation, and unreported orchestration. The 59-second model phase on a zero-download run is consistent with cached-artifact inspection or validation after a reload. These categories need finer instrumentation before assigning all of that time to shader compilation.

## Why 20 steps can look better than four

Four-step numerical parity and four-step visual quality are different gates. Four large Euler updates provide only four opportunities for image conditioning and CFG to steer the noisy latent. Twenty smaller updates follow the learned flow field more closely and can produce materially better geometry even when the WebGPU trajectory is not bit-equivalent to official PyTorch.
The current 20-step implementation also accumulates a reduction-order discrepancy in `context_refiner.0`, especially on the zero-conditioned branch. Numerical distance from the official 20-step trajectory can grow while the result remains perceptually preferable to the official or WebGPU four-step trajectory. The discrete octree then amplifies modest latent differences into topology changes. Therefore step count, numerical parity, and perceived quality must be reported as separate axes.

The product direction is to preserve 20 steps and reduce per-call cost. Four steps remains a diagnostic/fast schedule, not an acceptable substitute where it visibly degrades output.

## Hypotheses resolved or narrowed

| Hypothesis | Assessment from this report and code |
| --- | --- |
| Wrong GPU | Unlikely; Chromium reported `nvidia ampere` |
| Configured WASM fallback | Ruled out for the public runner; it requests WebGPU only |
| GPU-to-CPU readback dominates | Ruled out; 7 ms total DiT readback |
| JavaScript CFG/Euler dominates | Ruled out; sampling overhead above DiT inference was about 3.1 s |
| Session recreated every DiT call | Ruled out; one DiT session serves all 40 calls |
| K=256 candidate active | Ruled out; the canonical manifest remains deployed |
| fp32 40-call workload dominates | Confirmed |
| Many small/unfused WebGPU dispatches | Plausible; dispatch-level profiling is not yet recorded |
| Session construction and shader compilation matter | Likely and material, but current stage categories do not isolate them |

## Optimization order

### 1. Profile one warmed DiT call

Enable ONNX Runtime WebGPU profiling only in a diagnostic lab and configure it before session creation. Run an unprofiled warm-up, then separately profile a deterministic conditional call and the known zero-conditioned call. Record wall time, profile-record or dispatch count, summed GPU duration where exposed, median and p95 duration, top kernels by cumulative time, and wall time not explained by GPU records. Add per-session `GraphInfo.loadMs` to reports.

The canonical pipeline explicitly uses `graphOptimizationLevel: 'disabled'` because exported `Add(0)` barriers are parity-sensitive. A lab may compare disabled, basic, and all optimization, but no optimized setting can be promoted without the existing numerical trajectory gates.

ONNX Runtime documents `ort.env.webgpu.profiling`, `ort.env.trace`, graph capture, and GPU tensor placement in its [Web performance diagnosis](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html) and [environment/session options](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html) guides. WebGPU timing does not expose true NVIDIA SM occupancy. Coarse utilization may be observed externally; hardware occupancy claims require an appropriate GPU profiler or native reproduction.

### 2. Add explicit session-retention policies

The current correctness-first pipeline disposes DINO, VAE, DiT, octree, and Gaussian sessions after their stages. Add an opt-in policy rather than silently changing memory behavior:

```ts
sessionRetention: 'stage' | 'dit' | 'all'
```

Keep `stage` as the default. Qualify `dit` first because it retains the most repeatedly useful session with less residency than `all`. Image-specific retained conditioning can be replaced safely because the worker disposes previous tensors for the same reusable-input ID. Retention benefits later generations in the same tab; browser reloads cannot preserve GPU sessions.

A same-tab, all-session steady-state run might remove much of the approximately 93 seconds of observed setup overhead, but it cannot remove the five minutes of DiT compute. Raw model bytes are not peak VRAM, so neither a 24 GiB GPU nor browser `maxBufferSize` is proof that `all` is safe.

### 3. A/B graph capture after profiling

The DiT has fixed shapes and repeats 40 times, making it structurally suitable for an `enableGraphCapture` experiment. Capture replays command preparation; it does not reduce FLOPs, fuse kernels, or correct the attention reduction. It is valuable only if profiling finds substantial CPU submission gaps. Keep it lab-only until initialization, memory behavior, cancellation, and numerical output pass.

### 4. Develop mixed FP16/FP32 DiT, not an automatic all-FP16 path

The RTX 3090 may benefit from lower weight and activation bandwidth, but WebGPU `shader-f16` does not by itself guarantee Tensor Core use. Profile first. Decouple graph compute precision from sampler arithmetic so an FP16 graph does not automatically force host CFG/Euler rounding to FP16.

The first candidate should keep public latent/camera tensors, CFG/Euler, Q/K normalization, logits, softmax, and especially `context_refiner.0` probability×V accumulation in fp32 while testing fp16 weights, ordinary GEMMs, and validated intermediate storage. Qualify stage output, teacher-forced conditional/unconditional calls, complete 20-step state, octree topology, final scene, and fixed-camera renders on NVIDIA and at least one non-NVIDIA WebGPU device.

### 5. Keep GPU-resident CFG/Euler as a later architectural experiment

Removing 7 ms of readback is not a performance objective. GPU-resident state matters only if it removes synchronization and enables more continuous command submission. It requires GPU tensor I/O binding, explicit buffer lifetime management, a fused CFG/Euler shader, cancellation/device-loss handling, and parity validation. Profile and graph-capture results should justify that complexity first.

## Decision

Preserve the 20-step public quality path. Profile before changing precision or runtime structure; then pursue retained DiT sessions for repeat-run latency and a carefully mixed-precision DiT for single-run latency. Do not promote K=256, collapsed context, graph optimization, graph capture, or FP16 from timing alone.

External ONNX Runtime documentation descriptions above are paraphrased for licensing compliance.