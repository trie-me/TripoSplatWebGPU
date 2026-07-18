# DiT WebGPU profiling and bounded optimization A/B

Updated 2026-07-18. The public quality path remains the official 20-step, 40-DiT-call fp32 schedule. This work adds diagnostics; it does not change sampler semantics, numerical tolerances, the canonical graph, or the production session policy.

## What is implemented

The low-level runtime has two startup-only diagnostics:

- `configuration.webgpuProfiling` installs `ort.env.webgpu.profiling.ondata` before the first session is created;
- `configuration.trace` controls `ort.env.trace` as a separate stream.

An individual `runGraph()` call opts into record collection with `profileWebGpu: true`. Its result includes wall inference time, raw structured-clone-safe records, dispatch distribution, cumulative top-kernel timing, and wall time not represented by summed kernel timestamps. Profiling-enabled workers globally serialize diagnostic runs so callbacks cannot be attributed to a concurrent session.

`GraphInfo.loadMs` is now propagated into built-in scene metadata as `dinoLoadMs`, `vaeLoadMs`, `ditLoadMs`, `octreeLoadMs`, and `gaussianLoadMs`, separate from inference timing.

Profiling is disabled by default. It must be enabled when the runtime is constructed, before graph loading:

```ts
import { createRuntime } from '@ai3d/triposplat-webgpu/low-level'

const runtime = createRuntime({
  configuration: {
    webgpuProfiling: true,
    trace: false,
  },
})

const graphInfo = await runtime.loadGraph('dit-diagnostic', graph, {
  graphOptimizationLevel: 'disabled',
})

await runtime.runGraph('dit-diagnostic', warmupInputs)
const measured = await runtime.runGraph('dit-diagnostic', inputs, {
  profileWebGpu: true,
  tag: 'deterministic-dit-invocation',
})
```

## Browser lab

Start Vite with `pnpm dev`, then open:

```text
/dit-profile-lab.html?autorun=1
```

The lab loads the official flow-four teacher fixture and:

1. runs invocation 7 once without profiling as warm-up;
2. measures a second unprofiled invocation 7 baseline;
3. requests an isolated profile for invocation 7 conditional;
4. requests an isolated profile for invocation 8 zero-conditioned;
5. applies the unchanged strict output gates to both calls.

Query parameters:

- `optimization=disabled|basic|all`;
- `capture=1` for the lab-only graph-capture attempt;
- `trace=1` for the separate ORT trace stream;
- `model=` and `fixture=` for explicit local artifacts.

The complete result, including raw records when available, is retained on `window.__TRIPOSPLAT_DIT_PROFILE_RESULT__`.

## Repository-controlled result

Chrome 150 on the recorded Apple `metal-3` adapter executed the canonical and `all` optimization variants. The model revision and graph hashes match the canonical artifacts.

| Configuration | Session load | Warmed invocation 7 | Invocation 7 strict gate | Invocation 8 strict gate |
| --- | ---: | ---: | --- | --- |
| `disabled`, capture off | 15,557.4 ms | 12,993.7 ms | Pass | Fail |
| `all`, capture off | 10,251.4 ms | 11,844.3 ms | Pass | Fail |

`all` reduced this run's session load by 34.11% and warmed invocation-7 wall time by 8.85%. Invocation 8 remained outside the existing strict gate: latent RMSE changed only from `7.0770441e-5` to `7.0733219e-5`.

The complete 20-step follow-up did not sustain that speed result. All 40 calls completed without fallback or device loss, but `all` took 731,902.9 ms for sampling versus the historical 676,669.1 ms conservative run, an 8.16% increase. Session load improved from 12,510.0 to 9,552.0 ms. Final latent max/mean/RMSE improved modestly to `0.0483210` / `0.000320900` / `0.000748286`, but both qualification and strict gates still failed. Because these are separate long runs, the timing comparison includes thermal and run-to-run variance; it is nevertheless sufficient to reject an automatic Mac promotion.

The adapter exposed `timestamp-query`, but the installed ONNX Runtime 1.27 native WebGPU entrypoint emitted zero `env.webgpu.profiling.ondata` records. The report records availability as `no-records`; it does not interpret the empty list as zero GPU time and does not claim profiling overhead or hardware occupancy. `ort.env.trace` remains available separately.

Graph capture was rejected before inference with:

```text
Not supported preferred output location: cpu. Only 'gpu-buffer' location is supported when enableGraphCapture is true.
```

The current CPU output/readback contract is intentional for host fp32 CFG/Euler. Graph capture stays off until a lab explicitly supplies GPU-buffer inputs and outputs, preserves cancellation/disposal behavior, and passes the complete trajectory gate.

Machine-readable evidence: [`2026-07-18-dit-webgpu-profile-apple-metal3.json`](benchmarks/2026-07-18-dit-webgpu-profile-apple-metal3.json) and [`2026-07-18-flow20-fp32-webgpu-optimization-all.json`](benchmarks/2026-07-18-flow20-fp32-webgpu-optimization-all.json).

## Decision

No unqualified production switch is justified. On Mac, the complete `all` run is now a no-go for both numerical qualification and sustained latency. The next useful performance gate is a thermally controlled, immediately paired 20-step comparison on target NVIDIA hardware. Session retention remains a separate repeat-generation optimization; it cannot reduce the 40 calls inside one generation and requires a measured memory/lifecycle qualification before implementation.
