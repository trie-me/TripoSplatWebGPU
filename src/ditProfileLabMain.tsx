import { StrictMode, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'

import type { WebGpuRunProfile } from '../packages/triposplat-webgpu/src/profiling'
import { compareFloat32 } from './models/triposplat/tensorMath'
import {
  OrtWorkerClient,
  type OrtRunSessionResult,
  type OrtWorkerStatus,
} from './runtime/OrtWorkerClient'
import { createTensorPayload } from './runtime/tensors'

const DEFAULT_MODEL = '/models/triposplat/dit_step_webgpu_fp32.onnx'
const DEFAULT_FIXTURE = '/fixtures/generated/flow4-fp32-trajectory'
const SESSION_ID = 'triposplat/dit-profile'
const STRICT_TOLERANCE = {
  absolute: 0.0001,
  relative: 0.001,
  minimumCosineSimilarity: 0.99999999,
}
const SHAPES = {
  latent: [1, 8192, 16],
  camera: [1, 1, 5],
  t: [1],
  feature1: [1, 4101, 1280],
  feature2: [1, 4101, 128],
  pred_latent: [1, 8192, 16],
  pred_camera: [1, 1, 5],
} as const

type OptimizationLevel = 'disabled' | 'basic' | 'all'
type ProfiledInvocation = 7 | 8

interface TrajectoryTensor {
  path: string
  sha256?: string
}

interface TrajectoryInvocation {
  invocation: number
  step: number
  pass: 'conditional' | 'unconditional'
  tensors: Record<
    'sample_latent' | 'sample_camera' | 't' | 'pred_latent' | 'pred_camera',
    TrajectoryTensor
  >
}

interface InvocationFixture {
  invocation: ProfiledInvocation
  pass: 'conditional' | 'unconditional'
  latent: Float32Array
  camera: Float32Array
  t: Float32Array
  predLatent: Float32Array
  predCamera: Float32Array
}

interface OutputGate {
  maxAbsoluteError: number
  meanAbsoluteError: number
  rmse: number
  cosineSimilarity: number
  fractionWithinTolerance: number
  passed: boolean
}

interface ProfiledCall {
  invocation: ProfiledInvocation
  pass: 'conditional' | 'unconditional'
  inferenceMs: number
  readbackMs: number
  totalMs: number
  latent: OutputGate
  camera: OutputGate
  profile: WebGpuRunProfile
}

interface DitProfileResult {
  completed: true
  passed: boolean
  component: 'dit_webgpu_profile'
  configuration: {
    graphOptimizationLevel: OptimizationLevel
    enableGraphCapture: boolean
    trace: boolean
    warmup: 'one-unprofiled-invocation-7'
  }
  model: {
    url: string
    loadMs: number
    executionProvider: string
    revision?: string
    graphSha256?: string
    sidecarSha256?: string
  }
  unprofiled: {
    warmupInvocation7Ms: number
    warmedInvocation7Ms: number
  }
  profiled: {
    invocation7: ProfiledCall
    invocation8: ProfiledCall
    conditionalProfileWallDeltaMs: number
    conditionalProfileOverheadMs?: number
  }
  environment: {
    userAgent: string
    crossOriginIsolated: boolean
    webgpu: boolean
    logicalCpuCores: number
    deviceMemoryGiB?: number
    adapter?: Record<string, string>
    webgpuFeatures?: string[]
    runtime: 'onnxruntime-web 1.27.0'
  }
}

declare global {
  interface Window {
    __TRIPOSPLAT_DIT_PROFILE_RESULT__?: DitProfileResult
  }
}

function elementCount(shape: readonly number[]): number {
  return shape.reduce((product, value) => product * value, 1)
}

async function fetchFloat32(url: string, expectedElements: number): Promise<Float32Array> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Could not fetch ${url}: HTTP ${response.status}`)
  const buffer = await response.arrayBuffer()
  if (buffer.byteLength !== expectedElements * Float32Array.BYTES_PER_ELEMENT) {
    throw new Error(
      `${url} has ${buffer.byteLength} bytes; expected `
        + `${expectedElements * Float32Array.BYTES_PER_ELEMENT}.`,
    )
  }
  return new Float32Array(buffer)
}

function initialParameter(name: string, fallback: string): string {
  return new URLSearchParams(window.location.search).get(name) ?? fallback
}

function initialBoolean(name: string): boolean {
  return initialParameter(name, '0') === '1'
}

function initialOptimization(): OptimizationLevel {
  const value = initialParameter('optimization', 'disabled')
  if (value === 'disabled' || value === 'basic' || value === 'all') return value
  throw new Error(`Unsupported optimization '${value}'. Use disabled, basic, or all.`)
}

function gate(reference: Float32Array, candidate: Float32Array): OutputGate {
  const comparison = compareFloat32(reference, candidate)
  let within = 0
  for (let index = 0; index < reference.length; index += 1) {
    const allowed = (
      STRICT_TOLERANCE.absolute
      + STRICT_TOLERANCE.relative * Math.abs(reference[index])
    )
    if (Math.abs(reference[index] - candidate[index]) <= allowed) within += 1
  }
  const fractionWithinTolerance = within / reference.length
  return {
    maxAbsoluteError: comparison.maxAbsoluteError,
    meanAbsoluteError: comparison.meanAbsoluteError,
    rmse: comparison.rmse,
    cosineSimilarity: comparison.cosineSimilarity,
    fractionWithinTolerance,
    passed: comparison.finite
      && fractionWithinTolerance === 1
      && comparison.cosineSimilarity >= STRICT_TOLERANCE.minimumCosineSimilarity,
  }
}

async function loadTrajectory(fixtureUrl: string): Promise<{
  feature1: Float32Array
  feature2: Float32Array
  invocations: Record<ProfiledInvocation, InvocationFixture>
}> {
  const response = await fetch(`${fixtureUrl}/flow.json`)
  if (!response.ok) throw new Error(`Could not fetch ${fixtureUrl}/flow.json: HTTP ${response.status}`)
  const manifest = await response.json() as { trajectory?: TrajectoryInvocation[] }
  const selected = ([7, 8] as const).map((number) => {
    const invocation = manifest.trajectory?.find((value) => value.invocation === number)
    if (!invocation) throw new Error(`Trajectory fixture has no invocation ${number}.`)
    return invocation
  })
  const path = (invocation: TrajectoryInvocation, name: keyof TrajectoryInvocation['tensors']) =>
    `${fixtureUrl}/${invocation.tensors[name].path}`
  const [feature1, feature2, ...values] = await Promise.all([
    fetchFloat32(`${fixtureUrl}/feature1.f32`, elementCount(SHAPES.feature1)),
    fetchFloat32(`${fixtureUrl}/feature2.f32`, elementCount(SHAPES.feature2)),
    ...selected.flatMap((invocation) => [
      fetchFloat32(path(invocation, 'sample_latent'), elementCount(SHAPES.latent)),
      fetchFloat32(path(invocation, 'sample_camera'), elementCount(SHAPES.camera)),
      fetchFloat32(path(invocation, 't'), 1),
      fetchFloat32(path(invocation, 'pred_latent'), elementCount(SHAPES.pred_latent)),
      fetchFloat32(path(invocation, 'pred_camera'), elementCount(SHAPES.pred_camera)),
    ]),
  ])
  const invocations = {} as Record<ProfiledInvocation, InvocationFixture>
  for (const [index, invocation] of selected.entries()) {
    const offset = index * 5
    const invocationNumber = invocation.invocation as ProfiledInvocation
    invocations[invocationNumber] = {
      invocation: invocationNumber,
      pass: invocation.pass,
      latent: values[offset],
      camera: values[offset + 1],
      t: values[offset + 2],
      predLatent: values[offset + 3],
      predCamera: values[offset + 4],
    }
  }
  return { feature1, feature2, invocations }
}

function outputFloat32(response: OrtRunSessionResult, name: 'pred_latent' | 'pred_camera') {
  const output = response.outputs[name]
  if (!output || output.type !== 'float32') {
    throw new Error(`DiT did not return float32 output '${name}'.`)
  }
  return output.data
}

function inputsFor(
  fixture: InvocationFixture,
  feature1: Float32Array,
  feature2: Float32Array,
) {
  const conditional = fixture.pass === 'conditional'
  return {
    latent: createTensorPayload('float32', new Float32Array(fixture.latent), SHAPES.latent),
    camera: createTensorPayload('float32', new Float32Array(fixture.camera), SHAPES.camera),
    t: createTensorPayload('float32', new Float32Array(fixture.t), SHAPES.t),
    feature1: createTensorPayload(
      'float32',
      conditional ? new Float32Array(feature1) : new Float32Array(feature1.length),
      SHAPES.feature1,
    ),
    feature2: createTensorPayload(
      'float32',
      conditional ? new Float32Array(feature2) : new Float32Array(feature2.length),
      SHAPES.feature2,
    ),
  }
}

async function modelProvenance(modelUrl: string): Promise<{
  revision?: string
  graphSha256?: string
  sidecarSha256?: string
}> {
  try {
    const response = await fetch(new URL('manifest.json', new URL(modelUrl, document.baseURI)))
    if (!response.ok) return {}
    const manifest = await response.json() as {
      modelRevision?: string
      graphs?: {
        dit?: {
          integrity?: { digest?: string }
          externalData?: Array<{ integrity?: { digest?: string } }>
        }
      }
    }
    return {
      ...(manifest.modelRevision === undefined ? {} : { revision: manifest.modelRevision }),
      ...(manifest.graphs?.dit?.integrity?.digest === undefined
        ? {}
        : { graphSha256: manifest.graphs.dit.integrity.digest }),
      ...(manifest.graphs?.dit?.externalData?.[0]?.integrity?.digest === undefined
        ? {}
        : { sidecarSha256: manifest.graphs.dit.externalData[0].integrity.digest }),
    }
  } catch {
    return {}
  }
}

async function adapterDetails(): Promise<{
  info: Record<string, string>
  features: string[]
} | undefined> {
  const gpu = (navigator as Navigator & {
    gpu?: {
      requestAdapter(): Promise<{
        features: ReadonlySet<string>
        info: {
          vendor?: string
          architecture?: string
          device?: string
          description?: string
        }
      } | null>
    }
  }).gpu
  if (!gpu) return undefined
  const adapter = await gpu.requestAdapter()
  if (!adapter) return undefined
  const info = adapter.info
  return {
    info: Object.fromEntries(
      Object.entries({
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      }).filter((entry): entry is [string, string] => typeof entry[1] === 'string' && entry[1] !== ''),
    ),
    features: Array.from(adapter.features).sort(),
  }
}

async function profiledCall(
  client: OrtWorkerClient,
  fixture: InvocationFixture,
  feature1: Float32Array,
  feature2: Float32Array,
): Promise<ProfiledCall> {
  const response = await client.runSession({
    sessionId: SESSION_ID,
    inputs: inputsFor(fixture, feature1, feature2),
    outputs: ['pred_latent', 'pred_camera'],
    tag: `profile-invocation-${fixture.invocation}-${fixture.pass}`,
    profileWebGpu: true,
  })
  if (!response.profile) {
    throw new Error(`Invocation ${fixture.invocation} returned no WebGPU profile.`)
  }
  return {
    invocation: fixture.invocation,
    pass: fixture.pass,
    inferenceMs: response.timings.inferenceMs,
    readbackMs: response.timings.readbackMs,
    totalMs: response.timings.totalMs,
    latent: gate(fixture.predLatent, outputFloat32(response, 'pred_latent')),
    camera: gate(fixture.predCamera, outputFloat32(response, 'pred_camera')),
    profile: response.profile,
  }
}

function displayResult(result: DitProfileResult): unknown {
  const withoutRecords = (call: ProfiledCall) => ({
    ...call,
    profile: {
      ...call.profile,
      records: `[${call.profile.records.length} records retained on window.__TRIPOSPLAT_DIT_PROFILE_RESULT__]`,
    },
  })
  return {
    ...result,
    profiled: {
      ...result.profiled,
      invocation7: withoutRecords(result.profiled.invocation7),
      invocation8: withoutRecords(result.profiled.invocation8),
    },
  }
}

export function DitProfileLab() {
  const clientRef = useRef<OrtWorkerClient | null>(null)
  const runRef = useRef<() => Promise<void>>(async () => undefined)
  const autoRunStarted = useRef(false)
  const [modelUrl, setModelUrl] = useState(() => initialParameter('model', DEFAULT_MODEL))
  const [fixtureUrl, setFixtureUrl] = useState(() => initialParameter('fixture', DEFAULT_FIXTURE))
  const [optimization, setOptimization] = useState<OptimizationLevel>(initialOptimization)
  const [graphCapture, setGraphCapture] = useState(() => initialBoolean('capture'))
  const [trace, setTrace] = useState(() => initialBoolean('trace'))
  const [status, setStatus] = useState('Ready to warm and profile deterministic invocations 7 and 8.')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DitProfileResult | null>(null)

  useEffect(() => () => {
    const client = clientRef.current
    clientRef.current = null
    if (client) void client.dispose()
  }, [])

  const run = async () => {
    if (busy) return
    setBusy(true)
    setError(null)
    setResult(null)
    delete window.__TRIPOSPLAT_DIT_PROFILE_RESULT__
    let client: OrtWorkerClient | undefined
    let finalStatus: string | undefined
    try {
      if (clientRef.current) await clientRef.current.dispose()
      const onStatus = (event: OrtWorkerStatus) => setStatus(event.message)
      client = new OrtWorkerClient({
        onStatus,
        runtime: { webgpuProfiling: true, trace },
      })
      clientRef.current = client
      setStatus('Fetching the paired official teacher fixtures…')
      const [fixture, provenance, adapter] = await Promise.all([
        loadTrajectory(fixtureUrl),
        modelProvenance(modelUrl),
        adapterDetails(),
      ])
      const sidecarUrl = `${modelUrl}.data`
      const filename = new URL(modelUrl, document.baseURI).pathname.split('/').at(-1)
      if (!filename) throw new Error(`Could not derive the external-data path from ${modelUrl}.`)
      const loaded = await client.loadSession({
        sessionId: SESSION_ID,
        manifest: {
          graphUrl: modelUrl,
          externalData: [{ path: `${decodeURIComponent(filename)}.data`, url: sidecarUrl }],
        },
        options: {
          allowWasmFallback: false,
          graphOptimizationLevel: optimization,
          enableGraphCapture: graphCapture,
        },
      })
      if (loaded.executionProvider !== 'webgpu') {
        throw new Error(`Expected WebGPU, loaded ${loaded.executionProvider}.`)
      }
      setStatus('Running one unprofiled warm-up…')
      const warmup = await client.runSession({
        sessionId: SESSION_ID,
        inputs: inputsFor(fixture.invocations[7], fixture.feature1, fixture.feature2),
        outputs: ['pred_latent', 'pred_camera'],
        tag: 'warmup-invocation-7',
        profileWebGpu: false,
      })
      setStatus('Measuring the warmed unprofiled conditional baseline…')
      const baseline = await client.runSession({
        sessionId: SESSION_ID,
        inputs: inputsFor(fixture.invocations[7], fixture.feature1, fixture.feature2),
        outputs: ['pred_latent', 'pred_camera'],
        tag: 'baseline-invocation-7',
        profileWebGpu: false,
      })
      setStatus('Profiling warmed invocation 7 (conditional)…')
      const invocation7 = await profiledCall(
        client,
        fixture.invocations[7],
        fixture.feature1,
        fixture.feature2,
      )
      setStatus('Profiling warmed invocation 8 (zero-conditioned)…')
      const invocation8 = await profiledCall(
        client,
        fixture.invocations[8],
        fixture.feature1,
        fixture.feature2,
      )
      const deviceMemory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory
      const next: DitProfileResult = {
        completed: true,
        passed: invocation7.latent.passed
          && invocation7.camera.passed
          && invocation8.latent.passed
          && invocation8.camera.passed,
        component: 'dit_webgpu_profile',
        configuration: {
          graphOptimizationLevel: optimization,
          enableGraphCapture: graphCapture,
          trace,
          warmup: 'one-unprofiled-invocation-7',
        },
        model: {
          url: modelUrl,
          loadMs: loaded.loadMs,
          executionProvider: loaded.executionProvider,
          ...provenance,
        },
        unprofiled: {
          warmupInvocation7Ms: warmup.timings.inferenceMs,
          warmedInvocation7Ms: baseline.timings.inferenceMs,
        },
        profiled: {
          invocation7,
          invocation8,
          conditionalProfileWallDeltaMs: invocation7.inferenceMs - baseline.timings.inferenceMs,
          ...(invocation7.profile.availability === 'available'
            ? {
                conditionalProfileOverheadMs:
                  invocation7.inferenceMs - baseline.timings.inferenceMs,
              }
            : {}),
        },
        environment: {
          userAgent: navigator.userAgent,
          crossOriginIsolated: self.crossOriginIsolated,
          webgpu: 'gpu' in navigator,
          logicalCpuCores: navigator.hardwareConcurrency,
          ...(deviceMemory === undefined ? {} : { deviceMemoryGiB: deviceMemory }),
          ...(adapter === undefined
            ? {}
            : { adapter: adapter.info, webgpuFeatures: adapter.features }),
          runtime: 'onnxruntime-web 1.27.0',
        },
      }
      window.__TRIPOSPLAT_DIT_PROFILE_RESULT__ = next
      setResult(next)
      finalStatus = next.passed
        ? 'PASS: both profiled calls preserve the strict numerical gate.'
        : 'PROFILE COMPLETE: at least one existing strict numerical gate remains failed.'
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
      finalStatus = 'DiT profiling failed.'
    } finally {
      if (client) await client.dispose().catch(() => undefined)
      if (clientRef.current === client) clientRef.current = null
      if (finalStatus) setStatus(finalStatus)
      setBusy(false)
    }
  }

  runRef.current = run
  useEffect(() => {
    const timeout = window.setTimeout(() => {
      if (initialBoolean('autorun') && !autoRunStarted.current) {
        autoRunStarted.current = true
        void runRef.current()
      }
    }, 0)
    return () => window.clearTimeout(timeout)
  }, [])

  return (
    <main>
      <h1>TripoSplat DiT · WebGPU profiler</h1>
      <p>
        Runs an unprofiled warm-up, a warmed baseline, then profiles canonical
        invocation 7 and zero-conditioned invocation 8 separately.
      </p>
      <label>
        ONNX graph
        <input value={modelUrl} onChange={(event) => setModelUrl(event.target.value)} />
      </label>
      <label>
        Teacher fixture
        <input value={fixtureUrl} onChange={(event) => setFixtureUrl(event.target.value)} />
      </label>
      <label>
        Graph optimization
        <select
          value={optimization}
          onChange={(event) => setOptimization(event.target.value as OptimizationLevel)}
        >
          <option value="disabled">disabled (canonical)</option>
          <option value="basic">basic (lab only)</option>
          <option value="all">all (lab only)</option>
        </select>
      </label>
      <label className="check">
        <input
          type="checkbox"
          checked={graphCapture}
          onChange={(event) => setGraphCapture(event.target.checked)}
        />
        enableGraphCapture (lab only)
      </label>
      <label className="check">
        <input type="checkbox" checked={trace} onChange={(event) => setTrace(event.target.checked)} />
        ORT trace (separate diagnostic stream)
      </label>
      <button type="button" disabled={busy} onClick={() => void run()}>
        {busy ? 'Profiling…' : 'Run paired DiT profile'}
      </button>
      <p role="status" data-testid="dit-profile-status">{status}</p>
      {error ? <pre className="error" data-testid="dit-profile-error">{error}</pre> : null}
      {result
        ? <pre data-testid="dit-profile-result">{JSON.stringify(displayResult(result), null, 2)}</pre>
        : null}
    </main>
  )
}

const style = document.createElement('style')
style.textContent = `
  :root { color: #ececf3; background: #101014; font: 15px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  body { margin: 0; }
  main { max-width: 1060px; margin: 0 auto; padding: 48px 24px; }
  h1 { font: 600 28px/1.2 system-ui, sans-serif; }
  label { display: grid; gap: 6px; margin: 18px 0; }
  label.check { display: flex; align-items: center; gap: 10px; }
  label.check input { width: auto; }
  input, select { box-sizing: border-box; width: 100%; padding: 10px; color: inherit; background: #1b1b22; border: 1px solid #3a3a48; border-radius: 6px; }
  button { padding: 10px 16px; color: #08080a; background: #f8cf00; border: 0; border-radius: 6px; font-weight: 700; cursor: pointer; }
  button:disabled { opacity: .55; cursor: wait; }
  pre { overflow: auto; padding: 16px; background: #18181f; border-radius: 8px; }
  .error { color: #ff9b9b; }
`
document.head.append(style)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <DitProfileLab />
  </StrictMode>,
)
