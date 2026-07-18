# Context0 attention candidate queue

Canonical graph and `manifest.json` remain the production path until a candidate clears paired invocation, trajectory, runtime, and E2E gates.

| Rank | Candidate | State | Evidence / next action |
|---:|---|---|---|
| 1 | `wgsl-online-softmax-v1` split ONNX + single-dispatch online accumulation | **Rejected at early invocation-8 gate** | It loads and runs on Chrome WebGPU without context loss, but invocation 8 latent RMSE is `0.2988153` and max is `3.3437576`. ORT exposes a GPU buffer allocation but no packed-layout/stride contract for a generic WGSL reader. See `validation/2026-07-17-context0-wgsl-online-smoke.json`. |
| 2 | ORT-compatible tiled/pairwise value reduction | **Next** | Preserve canonical score/softmax and use an ONNX lowering that ORT owns end-to-end, or add the smallest ORT custom-op/layout-view support needed to consume packed Q/K/V safely. Gate invocation 7 and 8 before any trajectory work. |
| 3 | Compensated fused accumulation | Queued | Consider only if candidate 2 gives a valid packed layout but misses invocation 8 numerically. |
| 4 | Selective precision / exporter layout change | Queued | Consider only after a real tile/reduction candidate fails its early gate. |

No arbitrary key-chunk sweep is authorized. A candidate advances only when invocation 8 does not materially regress and invocation 7 does not regress.
