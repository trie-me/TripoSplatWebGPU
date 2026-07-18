# Twenty-step quality and parity investigation

**Status:** diagnostic conclusion and performance direction updated 2026-07-18. No experimental graph described here is deployed.

## Executive conclusion

The 20-step WebGPU path is structurally correct and often visually acceptable, but it accumulates a value-dependent DiT error that is largest on the all-zero unconditional CFG branch. The first material divergence is in `context_refiner.0` attention, where 4,101 identical zero-condition embeddings create a long repeated-token reduction. Small prediction differences compound across 40 DiT calls and are amplified by the discrete eight-level octree into visible topology and Gaussian-scale changes.

The official shifted schedule, `1000 * timestep`, CFG expression, float32 rounding order, and Euler update are not implicated. Standalone octree occupancy and Gaussian decoder validation pass. Improving the quality path therefore requires a more faithful attention reduction, not a changed sampler or relaxed tolerance.

## Established evidence

- FAL and WebGPU PLYs are valid binary little-endian files with 262,144 vertices and the same 17 Gaussian properties; there is no malformed header, stride error, NaN contamination, or gross decoder corruption.
- The 20-step result tends toward slightly smaller Gaussian scales than the 4-step result. On some subjects this produces thinner geometry or holes.
- Browser final-state max latent error grows from `0.0044521` at four steps to `0.0487093` at twenty steps, approximately 10.9× across 40 DiT calls.
- Teacher-forced replay proves that the defect is not only autoregressive accumulation: later unconditional calls diverge even when each call receives the exact official state and timestep.
- Paired block probes localize the first material conditional/unconditional separation to `context_refiner_00_attention_residual`: ORT-versus-official max error is `3.0994e-6` conditional and `1.1063e-4` unconditional, a 35.69× ratio.
- At that boundary, adapted PyTorch remains close to untouched official PyTorch while ONNX Runtime diverges from both. Real RoPE, static Sobol positions, and host CFG/Euler arithmetic are not the initiating cause.
- FAL PLYs contain no provenance metadata. Exact service parity cannot be claimed without matching image preparation, features, initial noise, model revision, random stream, and export conventions.

Primary evidence: [`flow20 browser benchmark`](benchmarks/2026-07-15-flow20-fp32-webgpu.json) and [`invocation 7/8 block probes`](validation/2026-07-15-dit-flow4-invocations07-08-block-probes.json).

## Visual quality and performance consequence

A community RTX 3090-class report adds an important product observation: the 20-step output was visibly acceptable, while four steps produced inadequate quality. This does not conflict with the four-step qualification pass or the 20-step parity failure. Four-step parity measures agreement with the official four-step trajectory, not whether four large Euler updates are sufficient for the desired geometry. Twenty smaller updates apply conditioning and CFG repeatedly and can produce a perceptually better scene even while small per-call ONNX reduction differences accumulate relative to the official 20-step state. The octree can then amplify either useful latent refinement or numerical drift into discrete topology changes.

The quality path must therefore remain 20 steps while performance work targets the cost of each DiT call. The warm-cache report measured 301.9 seconds of DiT inference, about 64% of a 472.2-second run, while all DiT readback totaled only 7 milliseconds. The next bounded work is dispatch/kernel profiling, per-session load timing, opt-in retained sessions, a graph-capture A/B, and only then a mixed FP16/FP32 DiT. Details: [`RTX 3090-class 20-step performance`](rtx3090-20-step-performance.md).

## Experiments and outcomes

| Experiment | Invocation-40 max latent error | Outcome |
| --- | ---: | --- |
| Canonical fp32 ONNX | `0.00105971` | Baseline |
| Context key-chunked online softmax | `0.00106657` | Slightly worse; removed |
| First-token selection for repeated context | `0.00105458` | No material change; removed |
| Stable RMS rewrite | Comparable unconditional failures | Not causal; not deployed |
| One representative plus exact multiplicity bias | `0.00092310` | Modest improvement, but adapter parity failed |
| Sixteen representatives plus `4101/16` bias | `0.00092727` | No further benefit; temporary change reverted |

The one-representative ONNX graph closely matched its adapted PyTorch specialization (`5.97e-5` max latent error), but that specialization differed from untouched official PyTorch by `9.29e-4`. The problem is therefore the changed floating-point reduction order, not an ONNX transcription failure in that candidate. Its 1.633 GB sidecar was byte-identical to the canonical sidecar, proving a future specialized graph need not duplicate weights.

## K=256 probability×V candidate findings

A bounded candidate replaced only `context_refiner.0` probability×V accumulation with seventeen 256-key MatMuls and a balanced Add tree. It preserved scores, softmax, weights, sampler, CFG, guidance, and tolerances. The candidate established several important findings:

- Probability×V is the actionable numerical boundary. Against untouched official fp32 PyTorch, candidate weighted-value RMSE was `1.6715e-7` for invocation 7 and `4.5358e-5` for invocation 8. Invocation 8 post-projection RMSE remained `6.0785e-5`, so downstream amplification was still material.
- Isolated invocation changes were mixed. Invocation 7 latent maximum error regressed from `4.3750e-5` to `4.4525e-5`, while invocation 8 improved from `7.6425e-4` to `7.5686e-4`.
- The complete 20-step trajectory improved materially: final latent max `0.0487093` → `0.0416024`, mean `0.000351717` → `0.000240853`, and RMSE `0.000812052` → `0.000566224`. Despite the 14.6–31.5% reductions, it still failed the existing trajectory tolerance.
- Runtime cost was unacceptable for production: teacher-forced execution was about 6.96% slower, and the candidate full-flow run was about 20.84% slower than the historical canonical run. No reliable browser peak-GPU-memory API was available.
- The candidate completed Gaussian decode with 262,144 finite Gaussians, valid PLY and `.splat` exports, and a ready viewer. Two clean canonical E2E attempts reset before export, so no controlled fixed-camera or geometry comparison exists and visual improvement is not claimed.
- Extending the tree to both context-refiner layers was rejected because invocation 7 and invocation 8 mean/RMSE regressed. The two-layer option was removed rather than retained as an unqualified surface.

**Decision:** no-go. Keep the canonical graph and manifest. The result is evidence that reduction order can improve the full autoregressive trajectory, but this implementation is too slow, remains outside tolerance, and lacks paired visual qualification. Detailed evidence: [`context0 K=256 candidate`](context0-value-reduction-candidate.md) and [`machine-readable result`](validation/2026-07-17-context0-k256-candidate.json).

## Decisions and next direction

1. Keep the canonical graph for both CFG passes; do not add runtime routing for the collapsed candidate.
2. Preserve the official sampler and public float32 state. Do not tune guidance, schedule, or tolerances to conceal graph error.
3. Keep collapsed-context support diagnostic-only until it passes untouched-official PyTorch and Chrome/Edge WebGPU gates.
4. Focus the next bounded investigation inside `context_refiner.0`: Q/K normalization, logits, softmax, value accumulation, and output projection for an exact unconditional invocation.
5. Prefer a custom fused WebGPU online-softmax/value-accumulation kernel, or another implementation that preserves the official reduction behavior, over additional graph-level token-collapse variants.
6. Qualify any candidate in order: untouched official vs adapted PyTorch, ORT CPU vs official, several unconditional trajectory calls, browser teacher-forced replay, full autoregressive 20-step state, then fixed-camera PLY renders against current WebGPU, Hugging Face, and FAL.

Until those gates pass, the correct product statement is: 20-step WebGPU generation completes and may look acceptable, but it is not yet quality/parity-qualified against the official implementation.