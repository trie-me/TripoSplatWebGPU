# Context0 K=256 value-reduction candidate

## Decision
**No-go: do not deploy.** The candidate preserves score and softmax semantics, but it does not meet the complete 20-step tolerance and adds substantial browser cost. The canonical production manifest and graph are unchanged.

## Candidate
`--context0-attention-value-chunk 256` replaces only `context_refiner.0` probability×V accumulation with 17 256-key MatMuls followed by a balanced Add tree. Scores, softmax, model weights, sampler, CFG, guidance, and tolerances remain unchanged. The graph adds 393,100 bytes (11,078,714 bytes; SHA-256 `bbc1df17ddbf77326e516ad8a5f0f8de6bc7be4b65e0e8a9857dcbca7fb4ce4e`); its 1,633,210,368-byte sidecar is unchanged.

Earlier query-layout and fused-MHA approaches, and sequential, pairwise, blockwise, float64, and online reductions, were discarded because none robustly improved both invocations. Applying the tree to two context layers was also rejected: invocation 7 latent max regressed from `4.45246696e-5` to `4.50611145e-5`; invocation 8 mean/RMSE regressed from `4.30320606e-5` / `5.66209284e-5` to `4.30375116e-5` / `5.66324887e-5`.

## Exact attention boundaries
Against untouched official fp32 PyTorch on recorded trajectory calls, the final one-layer tree measured:

| Invocation | Probability×V max / RMSE | Post-projection max / RMSE |
|---|---:|---:|
| 7 conditional | `1.54972076e-6` / `1.67148644e-7` | `1.09672546e-5` / `2.98181018e-7` |
| 8 unconditional | `2.00510025e-4` / `4.53580119e-5` | `8.39233398e-4` / `6.07854848e-5` |

The capture uses the final Add-tree output, not a partial MatMul. It improves the previously captured WebGPU weighted-value RMSE from `1.92427655e-7` to `1.66964243e-7` (7) and from `5.95244344e-5` to `4.53580119e-5` (8); post-projection/full-output parity remains insufficient.

## Invocation and trajectory results
ORT CPU latent error, canonical → K=256: invocation 7 max `4.37498093e-5` → `4.45246696e-5`, mean `4.03755653e-6` → `4.03502491e-6`, RMSE `5.25906759e-6` → `5.25668856e-6`; invocation 8 max `7.64250755e-4` → `7.56859779e-4`, mean `4.30371181e-5` → `4.30320606e-5`, RMSE `5.66413873e-5` → `5.66209284e-5`.

The complete browser 20-step flow improved final latent error: max `0.0487092733` → `0.0416023731` (14.59%), mean `0.000351716981` → `0.000240852894` (31.52%), RMSE `0.000812051848` → `0.000566224326` (30.27%). It still fails the existing full-trajectory tolerance. Teacher-forced invocation 8 max/RMSE changed `8.44243215e-4` / `7.07704406e-5` → `8.36440129e-4` / `7.07570603e-5`; invocation 2 regressed slightly.

## Decode and visual qualification
The candidate completed one deterministic 20-step decode: 262,144 finite Gaussians, valid 17,826,208-byte PLY (`1047…a7fa`), valid 8,388,608-byte `.splat` (`d5b3…ce58`), and a ready fixed-camera viewer. Candidate geometry has no non-finite values, 21,995 active 128³ voxels, mean scale `0.00263693`, and mean opacity `0.838405`. No canonical PLY was produced: two clean canonical browser attempts reset during DiT before exporting. Therefore there is **no controlled visual/geometry comparison and no claim of visual improvement**. The prior 20-step structural `passed:false` was a gate bug (`steps === 4`); it now correctly accepts both supported 4- and 20-step schedules.

## Cost and next path
Candidate full-flow inference was 817,523.6 ms versus a 676,525.5 ms historical canonical run (~20.84% cross-run slower); teacher-forced inference was 6.96% slower. Memory is algorithmic only: a 256-query partial is ~1 MiB, 17 first-level partials ~17 MiB, and concurrent old/new tree levels ~26 MiB without reuse; score and probability tensors are each ~64 MiB. Browser peak GPU memory is unavailable.

Keep canonical production behavior. The next best path is a runtime-compatible attention lowering that improves the **full autoregressive trajectory** without the K=256 tree’s overhead, then rerun a stable controlled canonical/candidate E2E pair before considering deployment.
