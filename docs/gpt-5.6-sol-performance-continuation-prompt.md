# GPT-5.6 Sol continuation prompt: fix and accelerate the 20-step path

Copy the prompt below into a new GPT-5.6 Sol coding session.

```text
Continue the TripoSplatWebGPU 20-step quality/parity and performance investigation in this repository.

Primary goals
1. Fix the accumulated fp32 20-step trajectory defect without reducing the schedule, changing official sampler semantics, or relaxing gates.
2. Make the corrected 20-step / 40-call path faster, guided by measured WebGPU profiling rather than assumptions.

Product constraint
- Keep the public quality path at 20 steps / 40 CFG DiT calls. A user reports that 20 steps produces visibly acceptable geometry while four steps is too low quality. Four-step numerical parity does not make four-step visual quality acceptable.
- Do not conceal quality problems by changing guidance, shifted schedule, timestep scaling, seed behavior, tolerance, Gaussian count, octree policy, or output filtering.
- Treat visual quality, official-reference numerical parity, and runtime as separate gates.

Read first
- docs/current-status.md
- docs/20-step-quality-investigation.md
- docs/context0-value-reduction-candidate.md
- docs/rtx3090-20-step-performance.md
- docs/compatibility-and-benchmarks.md
- ATTENTION_REDUCTION_PARITY.md
- packages/triposplat-webgpu/src/pipeline.ts
- packages/triposplat-webgpu/src/worker.ts
- packages/triposplat-webgpu/src/runtime.ts
- packages/triposplat-webgpu/src/sampler.ts
- existing export, probe, fixture, and browser-lab scripts relevant to `context_refiner.0`

Established quality diagnosis
- The official shifted schedule, `1000 * timestep`, CFG expression, fp32 host rounding order, and Euler update are not the initiating defect.
- Standalone octree occupancy and Gaussian decoder gates pass.
- Four-step final state passes its qualification envelope but misses a stricter diagnostic. The 20-step final latent fails qualification and strict gates after 40 calls.
- Teacher-forced replay removes autoregressive input drift: all conditional calls pass the strict gate; the zero-conditioned call passes at step 1 and fails from step 2 onward.
- Fresh CPU ONNX Runtime reproduces most of the invocation-8 unconditional error, so the initiating defect is not WebGPU-only, session reuse, or host sampling.
- The first material split is `context_refiner.0` self-attention residual. Detailed probes identify probability×V accumulation as the actionable reduction boundary for 4,101 repeated zero-condition tokens.
- K=256 balanced reduction improved full 20-step max/mean/RMSE but remained outside tolerance and was slower. Online-softmax, collapsed-token, stable-RMS, RoPE, and projection variants already failed to solve the complete trajectory. Do not repeat them without new evidence.

New performance evidence
- Self-reported RTX 3090, exposed by Chrome as `nvidia ampere`.
- Linux x86_64, Chrome 148, 32 GiB device-memory hint.
- Warm OPFS cache: 6.02 GiB before/after, 0 B downloaded.
- fp32, 20 steps, 262,144 Gaussians.
- End-to-end 472.2 s; model 59.23 s; conditioning 69.3 s; sampling 305.0 s; decode 35.95 s; preview 2.73 s.
- DINO 3.01 s; VAE 1.01 s; DiT 301.9 s; DiT readback 7 ms; octree 3.02 s; Gaussian 5.12 s.
- Readback and host CFG/Euler are not primary bottlenecks. Forty DiT calls average about 7.55 s each.

Track A — fix the fp32 20-step trajectory
1. Reproduce the canonical invocation-7 conditional and invocation-8 zero-conditioned teacher fixtures before changing a graph.
2. Reuse or extend existing boundary probes for Q/K normalization, logits, softmax, probability×V, output projection, attention residual, and complete DiT prediction.
3. Determine the untouched official PyTorch accumulation behavior at probability×V and identify exactly where adapted PyTorch, CPU ORT, and WebGPU first diverge. Do not infer a fix only from final latent error.
4. Design one bounded runtime-compatible candidate preserving official operation and multiplicity semantics while controlling reduction order. Prefer a focused fused/custom lowering or equivalent exact reduction over token collapse or the rejected K=256 graph expansion.
5. Keep public inputs, outputs, CFG, and Euler state fp32. Do not introduce FP16 while isolating the fp32 correctness defect.
6. Gate the candidate in order: official versus adapted internal boundaries; CPU ORT conditional/unconditional fixtures; browser teacher-forced calls; all 40 autoregressive calls; octree and Gaussian scene; then fixed-camera render/topology comparison.
7. Reject a candidate that only improves invocation 8, misses the existing full-trajectory gate, worsens important maxima/topology, or lacks a controlled visual comparison. Do not change tolerances to promote it.
8. Keep candidate routing opt-in until every required gate passes. If no candidate passes, record the failed result and preserve the canonical manifest.

Track B — profile and accelerate the corrected quality path
1. Add opt-in diagnostic instrumentation, disabled by default, for one deterministic DiT invocation.
2. Configure ONNX Runtime WebGPU profiling before any session is created and capture profiling records through `ort.env.webgpu.profiling.ondata`.
3. Preserve `ort.env.trace` as a separate opt-in diagnostic.
4. Record per-session `GraphInfo.loadMs` in benchmark output.
5. Run one unprofiled warm-up, then profile a warmed conditional fixture and the known zero-conditioned fixture separately.
6. Report wall time, summed GPU kernel time where exposed, profile-record/dispatch count, median and p95 duration, top kernels by cumulative time, and unaccounted wall time.
7. Add a lab-only A/B for canonical `graphOptimizationLevel: 'disabled'` with graph capture off versus on. Do not promote graph capture unless static-shape execution succeeds and existing numerical gates remain unchanged.
8. If practical, add lab-only `basic` and `all` graph-optimization variants. The canonical path must remain disabled because exported Add(0) layout barriers are parity-sensitive.

Do not
- Do not change the public default to four steps.
- Do not claim that a structurally valid scene or one visually plausible image establishes official parity.
- Do not deploy K=256 or collapsed-context experiments.
- Do not create or advertise an FP16 production manifest before profiling identifies the bottleneck and numerical qualification exists.
- Do not move CFG/Euler to GPU merely to remove readback; measured DiT readback is only 7 ms total.
- Do not claim true NVIDIA SM occupancy from WebGPU timing records. Distinguish kernel timing, coarse GPU utilization, and hardware occupancy counters.

Follow-on design after the profiling milestone
- Propose `sessionRetention: 'stage' | 'dit' | 'all'`, preserving `stage` as the default.
- Implement and validate `dit` before `all`. Retention is same-tab only and must not be described as surviving reloads.
- Report session load savings separately from inference savings.
- Treat raw model bytes, browser buffer limits, and JavaScript heap as insufficient evidence of available VRAM.
- Use profile evidence to decide whether graph capture/fusion or mixed precision is next.
- For mixed precision, decouple graph compute precision from sampler arithmetic. Keep public latent/camera state and CFG/Euler fp32. Prefer fp32 Q/K normalization, logits, softmax, and `context_refiner.0` probability×V accumulation while testing fp16 weights/GEMMs elsewhere.
- Keep GPU-resident CFG/Euler as a later experiment because its purpose would be synchronization/queue continuity, not eliminating the measured 7 ms readback.

Validation and evidence
- Do not add an unqualified production switch.
- Preserve cancellation, retry, worker disposal, reusable conditioning replacement, and device-loss behavior.
- Run targeted package tests, typecheck affected packages, and the deterministic DiT/flow gates available locally.
- Store machine-readable benchmark/profiling results under docs/benchmarks or docs/validation and document exact browser, adapter, runtime version, graph hashes, warm-up policy, and profile overhead.
- Distinguish repository-controlled results from community reports.
- If browser/GPU execution is unavailable, implement only instrumentation that can be statically validated and provide exact manual run instructions; do not fabricate performance conclusions.

Deliverables for the first milestone
- A paired invocation-7/invocation-8 internal-boundary report confirming the exact first reduction divergence.
- One evidence-backed fp32 correction candidate, or a documented rejection explaining why no bounded candidate is justified yet.
- Minimal opt-in DiT profiling and per-session load timing.
- Canonical-versus-candidate numerical and runtime comparison using unchanged gates.
- A full 20-step autoregressive result if the candidate passes the earlier boundary gates.
- Documentation of results and the next decision; no unrelated refactor or unqualified production model change.

Use the repository's existing conventions and source-of-truth documents. Inspect current code before editing, make the smallest coherent changes, and validate them.
```