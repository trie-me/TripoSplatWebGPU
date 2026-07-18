# Context0 attention candidate queue

Canonical graph and `manifest.json` remain the production path until a candidate clears paired invocation, trajectory, runtime, and E2E gates.

| Rank | Candidate | State | Evidence / next action |
|---:|---|---|---|
| 1 | `wgsl-online-softmax-v1` split ONNX + single-dispatch online accumulation | **Rejected at early invocation-8 gate** | It loads and runs on Chrome WebGPU without context loss, but invocation 8 latent RMSE is `0.2988153` and max is `3.3437576`. ORT exposes a GPU buffer allocation but no packed-layout/stride contract for a generic WGSL reader. See `validation/2026-07-17-context0-wgsl-online-smoke.json`. |
| 2 | Repeated-V bypass for zero context | **Rejected before graph build** | Invocation 8 V rows are exactly repeated, but untouched official SDPA is not exactly the representative V: bypass RMSE is `4.5447353e-5`, slightly worse than canonical ORT CPU `4.5354315e-5`. Invocation 7 is not repeated, so the bypass also fails the paired-operation requirement. See `validation/2026-07-18-context0-repeated-v-invariant.json`. |
| 3 | `context_refiner.0` fixed q16 chunks | **Rejected at paired gate** | Exact captured MPS replay finds a threshold at query length 16, but the canonical 256-query chunks are already above it. The q16 graph is larger, leaves CPU ORT unchanged, and produces invocation-7/8 WebGPU metrics exactly identical to canonical `optimization=all`; invocation 8 still fails. See `validation/2026-07-18-context0-query-geometry-q16.json`. |
| 4 | ORT-compatible full-query tiled/pairwise value reduction | **Next** | Preserve canonical score/softmax, multiplicity, and full 4,101-query geometry. Use an ONNX lowering that ORT owns end-to-end, or add the smallest ORT custom-op/layout-view support needed to consume packed Q/K/V safely. Gate invocation 7 and 8 before any trajectory work. |
| 5 | Compensated fused accumulation | Queued | Consider only if candidate 4 gives a valid packed layout but misses invocation 8 numerically. |
| 6 | Selective precision / exporter layout change | Queued | Consider only after a real tile/reduction candidate fails its early gate. |

No arbitrary key-chunk sweep is authorized. A candidate advances only when invocation 8 does not materially regress and invocation 7 does not regress.
