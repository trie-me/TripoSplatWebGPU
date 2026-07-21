import {
  getModelCacheStatus,
  runWebGpuModelChecks,
  type CacheBackend,
  type ModelCacheBackendStatus,
  type WebGpuCheckStage,
  type WebGpuModelCheckReport,
} from '../packages/triposplat-webgpu/dist/index.js'
import {
  ModelArtifactManager,
  createRuntime,
  elementCount,
  fetchModelManifest,
  modelCacheNamespace,
  TRIPOSPLAT_CAMERA_SHAPE,
  TRIPOSPLAT_FEATURE1_SHAPE,
  TRIPOSPLAT_FEATURE2_SHAPE,
  TRIPOSPLAT_LATENT_SHAPE,
  withVerifiedModelArtifacts,
  type ModelCacheEntry,
  type ResolvedGraphManifestEntry,
  type ResolvedTripoSplatModelManifest,
  type RuntimeStatus,
} from '../packages/triposplat-webgpu/dist/low-level.js'

const DEFAULT_MODEL_BASE = 'https://huggingface.co/Yosun/TripoSplat-WebGPU/resolve/main/triposplat-webgpu/0.1.0-fp32.20260715/'
const DIT_SESSION_ID = 'checks-dit'

type PersistentBackend = Exclude<CacheBackend, 'none'>

interface CachedDitCandidate {
  backend: PersistentBackend
  manifest: ResolvedTripoSplatModelManifest
}

interface CheckReport {
  generatedAt: string
  page: string
  browser: string
  secureContext: boolean
  modelManifestUrl?: string
  modelNamespace?: string
  declaredModelBytes?: number
  webgpu?: WebGpuModelCheckReport
  cache?: Awaited<ReturnType<typeof getModelCacheStatus>>
  runtimeEvents: RuntimeStatus[]
  cachedDitProbe?: Record<string, unknown>
  error?: string
}

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector)
  if (!element) throw new Error(`Missing checks-runner element '${selector}'.`)
  return element
}

const modelBaseInput = required<HTMLInputElement>('#model-base')
const runChecksButton = required<HTMLButtonElement>('#run-checks')
const runDitButton = required<HTMLButtonElement>('#run-dit')
const copyReportButton = required<HTMLButtonElement>('#copy-report')
const overall = required<HTMLElement>('#overall')
const stageList = required<HTMLOListElement>('#stage-list')
const manifestStatus = required<HTMLElement>('#manifest-status')
const cacheRows = required<HTMLTableSectionElement>('#cache-rows')
const reportElement = required<HTMLElement>('#report')
const ditHelp = required<HTMLElement>('#dit-help')

let report: CheckReport = createReport()
let cachedDitCandidate: CachedDitCandidate | undefined
let running = false
function createReport(): CheckReport {
  return {
    generatedAt: new Date().toISOString(),
    page: location.href,
    browser: navigator.userAgent,
    secureContext: window.isSecureContext,
    runtimeEvents: [],
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1_024) return `${bytes} B`
  if (bytes < 1_024 ** 2) return `${(bytes / 1_024).toFixed(1)} KiB`
  if (bytes < 1_024 ** 3) return `${(bytes / 1_024 ** 2).toFixed(1)} MiB`
  return `${(bytes / 1_024 ** 3).toFixed(2)} GiB`
}

function formatDuration(milliseconds: number): string {
  return milliseconds < 1_000 ? `${milliseconds.toFixed(1)} ms` : `${(milliseconds / 1_000).toFixed(2)} s`
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function normalizedModelBase(): string {
  const url = new URL(modelBaseInput.value.trim())
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['localhost', '127.0.0.1'].includes(url.hostname))) {
    throw new Error('The model server must use HTTPS, except on localhost.')
  }
  if (!url.pathname.endsWith('/')) url.pathname += '/'
  return url.href
}

function declaredModelBytes(manifest: ResolvedTripoSplatModelManifest): number {
  if (manifest.estimatedModelBytes !== undefined) return manifest.estimatedModelBytes
  let total = 0
  for (const graph of Object.values(manifest.graphs)) {
    if (!graph) continue
    total += graph.byteLength ?? 0
    for (const external of graph.externalData ?? []) total += external.byteLength ?? 0
  }
  return total
}

function expectedDitArtifacts(graph: ResolvedGraphManifestEntry): Array<{
  label: string
  byteLength?: number
  sha256?: string
}> {
  return [
    { label: 'dit:graph', byteLength: graph.byteLength, sha256: graph.integrity?.digest },
    ...(graph.externalData ?? []).map((external) => ({
      label: `dit:external:${external.path}`,
      byteLength: external.byteLength,
      sha256: external.integrity?.digest,
    })),
  ]
}

function hasArtifact(entries: readonly ModelCacheEntry[], expected: ReturnType<typeof expectedDitArtifacts>[number]): boolean {
  return entries.some((entry) => (
    entry.label === expected.label
    && entry.integrityVerified
    && (expected.byteLength === undefined || entry.byteLength === expected.byteLength)
    && (expected.sha256 === undefined || entry.sha256 === expected.sha256)
  ))
}

function backendHasCompleteDit(status: ModelCacheBackendStatus, graph: ResolvedGraphManifestEntry): boolean {
  return status.available && expectedDitArtifacts(graph).every((artifact) => hasArtifact(status.entries, artifact))
}

function setOverall(state: 'running' | 'passed' | 'failed' | 'warning', text: string): void {
  overall.dataset.state = state
  overall.textContent = text
}

function renderReport(): void {
  reportElement.textContent = JSON.stringify(report, null, 2)
  copyReportButton.disabled = false
}
function renderStage(stage: WebGpuCheckStage): void {
  const item = document.createElement('li')
  item.dataset.state = stage.status
  const name = document.createElement('strong')
  name.textContent = stage.id.replaceAll('-', ' ')
  const description = document.createElement('span')
  description.textContent = stage.message
  const duration = document.createElement('small')
  duration.textContent = formatDuration(stage.durationMs)
  item.append(name, description, duration)
  stageList.append(item)
}

function renderCache(
  cache: Awaited<ReturnType<typeof getModelCacheStatus>>,
  manifest: ResolvedTripoSplatModelManifest,
): void {
  const dit = manifest.graphs.dit
  cacheRows.replaceChildren(...cache.backends.map((backend) => {
    const row = document.createElement('tr')
    const complete = dit ? backendHasCompleteDit(backend, dit) : false
    const values = [
      backend.backend.toUpperCase(),
      backend.available ? 'Yes' : `No${backend.error ? ` · ${backend.error}` : ''}`,
      String(backend.entryCount),
      formatBytes(backend.totalBytes),
      complete ? 'Ready' : 'Missing artifacts',
    ]
    for (const value of values) {
      const cell = document.createElement('td')
      cell.textContent = value
      row.append(cell)
    }
    return row
  }))
}

function setBusy(value: boolean): void {
  running = value
  runChecksButton.disabled = value
  runDitButton.disabled = value || cachedDitCandidate === undefined
  modelBaseInput.disabled = value
}

async function inspectManifestAndCache(): Promise<void> {
  const base = normalizedModelBase()
  const manifestUrl = new URL('manifest.json', base)
  manifestStatus.textContent = `Fetching only ${manifestUrl.href}…`
  const manifest = await fetchModelManifest(manifestUrl)
  const namespace = modelCacheNamespace(manifest)
  const cache = await getModelCacheStatus({ namespace })
  const bytes = declaredModelBytes(manifest)
  report.modelManifestUrl = manifest.sourceUrl
  report.modelNamespace = namespace
  report.declaredModelBytes = bytes
  report.cache = cache
  renderCache(cache, manifest)
  manifestStatus.textContent = `${formatBytes(bytes)} declared by ${manifest.name} ${manifest.version}. Found ${formatBytes(cache.totalBytes)} across existing caches. No model artifact was requested.`

  const dit = manifest.graphs.dit
  const readyBackend = dit
    ? cache.backends.find((backend) => backendHasCompleteDit(backend, dit))
    : undefined
  cachedDitCandidate = readyBackend
    ? { backend: readyBackend.backend, manifest }
    : undefined
  runDitButton.disabled = running || cachedDitCandidate === undefined
  ditHelp.textContent = cachedDitCandidate
    ? `A complete DiT cache was found in ${cachedDitCandidate.backend.toUpperCase()}. The optional probe hard-disables network artifact fetches.`
    : 'No single cache backend contains every declared DiT artifact. The deep probe is disabled and this page will not download the missing files.'
}

async function runChecks(): Promise<void> {
  if (running) return
  report = createReport()
  cachedDitCandidate = undefined
  stageList.replaceChildren()
  cacheRows.innerHTML = '<tr><td colspan="5">Inspecting OPFS and Cache API…</td></tr>'
  setBusy(true)
  setOverall('running', 'Running WebGPU, ONNX Runtime, manifest, and cache checks…')
  try {
    const [webgpu] = await Promise.all([
      runWebGpuModelChecks({
        powerPreference: 'high-performance',
        onStage: renderStage,
        onRuntimeStatus: (status) => report.runtimeEvents.push(status),
      }),
      inspectManifestAndCache(),
    ])
    report.webgpu = webgpu
    if (webgpu.supported) {
      setOverall(cachedDitCandidate ? 'passed' : 'warning', cachedDitCandidate
        ? 'ONNX Runtime WebGPU passed; a cached-only full DiT probe is available.'
        : 'ONNX Runtime WebGPU passed; full-model execution is not yet proven on this device.')
    } else {
      setOverall('failed', 'This browser failed a required WebGPU or ONNX Runtime check.')
    }
  } catch (error) {
    report.error = errorMessage(error)
    setOverall('failed', report.error)
  } finally {
    report.generatedAt = new Date().toISOString()
    renderReport()
    setBusy(false)
  }
}
function zeroInput(graph: ResolvedGraphManifestEntry, shape: readonly number[]) {
  const length = elementCount(shape)
  return graph.inputPrecision === 'fp16'
    ? { type: 'float16' as const, dims: [...shape], data: new Uint16Array(length) }
    : { type: 'float32' as const, dims: [...shape], data: new Float32Array(length) }
}

async function runCachedDitProbe(): Promise<void> {
  if (running || !cachedDitCandidate) return
  const { backend, manifest } = cachedDitCandidate
  const graph = manifest.graphs.dit
  if (!graph) throw new Error('The manifest has no DiT graph.')
  setBusy(true)
  setOverall('running', `Verifying cached ${backend.toUpperCase()} artifacts and running one real DiT invocation…`)
  const startedAt = performance.now()
  const denyNetwork: typeof fetch = async (input) => {
    throw new Error(`Network model download blocked by checks runner: ${String(input)}`)
  }
  const manager = new ModelArtifactManager({
    backend,
    namespace: modelCacheNamespace(manifest),
    fetch: denyNetwork,
  })
  const runtime = withVerifiedModelArtifacts(
    createRuntime({
      executionProviders: ['webgpu'],
      onStatus: (status) => report.runtimeEvents.push(status),
    }),
    manager,
    { [DIT_SESSION_ID]: 'dit' },
  )
  try {
    const graphInfo = await runtime.loadGraph(DIT_SESSION_ID, graph, { graphOptimizationLevel: 'disabled' })
    const result = await runtime.runGraph(DIT_SESSION_ID, {
      feature1: zeroInput(graph, TRIPOSPLAT_FEATURE1_SHAPE),
      feature2: zeroInput(graph, TRIPOSPLAT_FEATURE2_SHAPE),
      latent: zeroInput(graph, TRIPOSPLAT_LATENT_SHAPE),
      camera: zeroInput(graph, TRIPOSPLAT_CAMERA_SHAPE),
      t: zeroInput(graph, [1]),
    }, {
      outputs: ['pred_latent', 'pred_camera'],
      tag: 'cached-only-dit-compatibility-probe',
    })
    const latent = result.outputs.pred_latent
    const camera = result.outputs.pred_camera
    if (!latent || !camera) throw new Error('The DiT probe did not return both required outputs.')
    report.cachedDitProbe = {
      passed: true,
      backend,
      networkModelDownloadsAllowed: false,
      executionProvider: graphInfo.executionProvider,
      graphLoadMs: graphInfo.loadMs,
      inferenceMs: result.timings.inferenceMs,
      readbackMs: result.timings.readbackMs,
      totalProbeMs: performance.now() - startedAt,
      latentElements: latent.data.length,
      cameraElements: camera.data.length,
    }
    setOverall('passed', 'A cached production DiT graph loaded and completed one WebGPU invocation.')
  } catch (error) {
    report.cachedDitProbe = {
      passed: false,
      backend,
      networkModelDownloadsAllowed: false,
      totalProbeMs: performance.now() - startedAt,
      error: errorMessage(error),
    }
    setOverall('failed', `Cached DiT probe failed: ${errorMessage(error)}`)
  } finally {
    await runtime.dispose().catch(() => undefined)
    report.generatedAt = new Date().toISOString()
    renderReport()
    setBusy(false)
  }
}

runChecksButton.addEventListener('click', () => { void runChecks() })
runDitButton.addEventListener('click', () => { void runCachedDitProbe() })
copyReportButton.addEventListener('click', () => {
  void navigator.clipboard.writeText(reportElement.textContent ?? '').then(() => {
    copyReportButton.textContent = 'Copied'
    window.setTimeout(() => { copyReportButton.textContent = 'Copy report' }, 1_500)
  })
})

const suppliedModelBase = new URLSearchParams(location.search).get('modelBaseUrl')
modelBaseInput.value = suppliedModelBase ?? DEFAULT_MODEL_BASE
void runChecks()
