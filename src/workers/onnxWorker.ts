/// <reference lib="WebWorker" />

import * as ort from 'onnxruntime-web/webgpu'

import {
  CONTEXT0_ATTENTION_BYTES,
  CONTEXT0_ATTENTION_SHAPE,
  createContext0AttentionExecutor,
  type Context0AttentionExecutor,
  type WebGpuBuffer,
  type WebGpuDevice,
} from './context0Attention'
import { assertModelManifest } from '../runtime/modelManifest'
import type { OnnxModelManifest } from '../runtime/modelManifest'
import type {
  OrtConfigureRuntimeResult,
  OrtExecutionProvider,
  OrtLoadContext0SplitRequest,
  OrtLoadContext0SplitResult,
  OrtLoadSessionRequest,
  OrtLoadSessionResult,
  OrtRunSessionRequest,
  OrtRunSessionResult,
  OrtRuntimeConfiguration,
  OrtSessionLoadOptions,
  OrtValueMetadata,
  OrtWorkerOperation,
  OrtWorkerReply,
  OrtWorkerRequest,
  OrtWorkerResultMap,
  OrtWorkerStage,
  OrtWorkerStatus,
  SerializedWorkerError,
} from '../runtime/OrtWorkerClient'
import type { TensorPayload, TensorPayloadMap } from '../runtime/tensors'
import {
  assertTensorPayloadMap,
  createTensorPayload,
  tensorPayloadTransferables,
} from '../runtime/tensors'

const workerScope = self as DedicatedWorkerGlobalScope

interface LoadedSession {
  session: ort.InferenceSession
  executionProvider: OrtExecutionProvider
  metadata: Omit<OrtLoadSessionResult, 'loadMs'>
  loadMs: number
}

interface SessionRecord {
  fingerprint: string
  loading: Promise<LoadedSession>
  runTail: Promise<void>
  disposed: boolean
}

interface LoadedContext0Split {
  pre: LoadedSession
  post: LoadedSession
  executor: Context0AttentionExecutor
  metadata: Omit<OrtLoadContext0SplitResult, 'loadMs'>
  loadMs: number
}

interface Context0SplitRecord {
  fingerprint: string
  loading: Promise<LoadedContext0Split>
  runTail: Promise<void>
  disposed: boolean
}

const sessions = new Map<string, SessionRecord>()
const context0Splits = new Map<string, Context0SplitRecord>()
let runtimeConfiguration: OrtConfigureRuntimeResult | undefined

function serializeError(error: unknown): SerializedWorkerError {
  if (error instanceof Error) {
    return { name: error.name, message: error.message, stack: error.stack }
  }
  return { name: 'Error', message: String(error) }
}

function postMessageSafe(message: OrtWorkerReply | OrtWorkerStatus, transfer: Transferable[] = []): void {
  workerScope.postMessage(message, transfer)
}

function postStatus(
  stage: OrtWorkerStage,
  message: string,
  requestId?: string,
  sessionId?: string,
  executionProvider?: OrtExecutionProvider,
  progress?: number,
): void {
  postMessageSafe({
    type: 'status',
    stage,
    message,
    timestampMs: Date.now(),
    requestId,
    sessionId,
    executionProvider,
    progress,
  })
}

function postSuccess<Operation extends OrtWorkerOperation>(
  operation: Operation,
  requestId: string,
  result: OrtWorkerResultMap[Operation],
  transfer: Transferable[] = [],
): void {
  const reply = {
    type: 'reply',
    operation,
    requestId,
    ok: true,
    result,
  } as OrtWorkerReply
  postMessageSafe(reply, transfer)
}

function postError(operation: OrtWorkerOperation, requestId: string, error: unknown): void {
  const reply = {
    type: 'reply',
    operation,
    requestId,
    ok: false,
    error: serializeError(error),
  } as OrtWorkerReply
  postMessageSafe(reply)
}

function assertSessionId(sessionId: unknown): asserts sessionId is string {
  if (typeof sessionId !== 'string' || sessionId.trim().length === 0 || sessionId.includes('\0')) {
    throw new TypeError('sessionId must be a non-empty string without null characters.')
  }
}

function defaultRuntimeConfiguration(): OrtConfigureRuntimeResult {
  const baseUrl = new URL(`${import.meta.env.BASE_URL}ort/`, workerScope.location.origin).href
  return {
    wasmThreads: workerScope.crossOriginIsolated
      ? Math.max(1, Math.min(4, workerScope.navigator.hardwareConcurrency || 2))
      : 1,
    wasmSimd: true,
    wasmPaths: {
      mjs: new URL('ort-wasm-simd-threaded.asyncify.mjs', baseUrl).href,
      wasm: new URL('ort-wasm-simd-threaded.asyncify.wasm', baseUrl).href,
    },
  }
}

function normalizeRuntimeConfiguration(configuration: OrtRuntimeConfiguration): OrtConfigureRuntimeResult {
  const defaults = defaultRuntimeConfiguration()
  const wasmThreads = configuration.wasmThreads ?? defaults.wasmThreads
  if (!Number.isInteger(wasmThreads) || wasmThreads < 1) {
    throw new RangeError('wasmThreads must be a positive integer.')
  }

  const wasmSimd = configuration.wasmSimd ?? defaults.wasmSimd
  if (
    typeof wasmSimd !== 'boolean'
    && wasmSimd !== 'fixed'
    && wasmSimd !== 'relaxed'
  ) {
    throw new TypeError("wasmSimd must be boolean, 'fixed', or 'relaxed'.")
  }

  let wasmPaths: OrtConfigureRuntimeResult['wasmPaths']
  if (configuration.wasmPaths === undefined) {
    wasmPaths = defaults.wasmPaths
  } else if (typeof configuration.wasmPaths === 'string') {
    if (configuration.wasmPaths.trim().length === 0) {
      throw new TypeError('wasmPaths prefix must be non-empty.')
    }
    wasmPaths = new URL(configuration.wasmPaths, workerScope.location.href).href
  } else {
    if (configuration.wasmPaths.mjs === undefined && configuration.wasmPaths.wasm === undefined) {
      throw new TypeError('wasmPaths must include at least one of mjs or wasm.')
    }
    wasmPaths = {
      mjs: configuration.wasmPaths.mjs === undefined
        ? undefined
        : new URL(configuration.wasmPaths.mjs, workerScope.location.href).href,
      wasm: configuration.wasmPaths.wasm === undefined
        ? undefined
        : new URL(configuration.wasmPaths.wasm, workerScope.location.href).href,
    }
  }

  return { wasmThreads, wasmSimd, wasmPaths }
}

function configureRuntime(
  configuration: OrtRuntimeConfiguration,
  requestId?: string,
): OrtConfigureRuntimeResult {
  const normalized = normalizeRuntimeConfiguration(configuration)
  if (runtimeConfiguration) {
    if (JSON.stringify(runtimeConfiguration) !== JSON.stringify(normalized)) {
      throw new Error('ONNX Runtime is already configured; start a new worker to use different WASM settings.')
    }
    return runtimeConfiguration
  }

  postStatus('runtime-configuring', 'Configuring ONNX Runtime.', requestId)
  ort.env.wasm.numThreads = normalized.wasmThreads
  ort.env.wasm.simd = normalized.wasmSimd
  ort.env.wasm.wasmPaths = normalized.wasmPaths
  runtimeConfiguration = normalized
  postStatus('runtime-ready', 'ONNX Runtime is configured.', requestId)
  return normalized
}

function ensureRuntimeConfigured(requestId: string): OrtConfigureRuntimeResult {
  return runtimeConfiguration ?? configureRuntime({}, requestId)
}

function toMetadata(metadata: readonly ort.InferenceSession.ValueMetadata[]): OrtValueMetadata[] {
  return metadata.map((value) => value.isTensor
    ? {
        name: value.name,
        isTensor: true,
        type: value.type,
        shape: Array.from(value.shape),
      }
    : { name: value.name, isTensor: false })
}

function sessionFingerprint(request: OrtLoadSessionRequest): string {
  return JSON.stringify({ manifest: request.manifest, options: request.options ?? {} })
}

function createSessionOptions(
  manifest: OnnxModelManifest,
  options: OrtSessionLoadOptions | undefined,
  provider: OrtExecutionProvider,
  preferredOutputLocation: 'cpu' | 'gpu-buffer' = 'cpu',
  device?: WebGpuDevice,
): ort.InferenceSession.SessionOptions {
  const common: ort.InferenceSession.SessionOptions = {
    graphOptimizationLevel: options?.graphOptimizationLevel ?? 'all',
    preferredOutputLocation,
    externalData: manifest.externalData?.map(({ path, url }) => ({ path, data: url })),
  }

  if (options?.freeDimensionOverrides) {
    common.freeDimensionOverrides = options.freeDimensionOverrides
  }
  if (options?.logSeverityLevel !== undefined) {
    common.logSeverityLevel = options.logSeverityLevel
  }

  if (provider === 'wasm') {
    common.executionProviders = ['wasm']
    return common
  }

  const webgpu: ort.InferenceSession.WebGpuExecutionProviderOption = {
    name: 'webgpu',
    device,
    preferredLayout: options?.webgpu?.preferredLayout,
    forceCpuNodeNames: options?.webgpu?.forceCpuNodeNames,
    validationMode: options?.webgpu?.validationMode,
  }
  common.executionProviders = [webgpu]
  if (options?.enableGraphCapture !== undefined) {
    common.enableGraphCapture = options.enableGraphCapture
  }
  return common
}

async function loadOrtSession(
  request: OrtLoadSessionRequest,
  requestId: string,
  preferredOutputLocation: 'cpu' | 'gpu-buffer' = 'cpu',
  device?: WebGpuDevice,
): Promise<LoadedSession> {
  ensureRuntimeConfigured(requestId)
  const { sessionId, manifest, options } = request
  const startedAt = performance.now()
  postStatus('session-loading', `Loading ONNX session '${sessionId}'.`, requestId, sessionId, 'webgpu')

  const heartbeatStartedAt = performance.now()
  const heartbeat = setInterval(() => {
    const seconds = Math.floor((performance.now() - heartbeatStartedAt) / 1000)
    postStatus(
      'session-loading',
      `Loading ONNX session '${sessionId}' (${seconds}s elapsed).`,
      requestId,
      sessionId,
      'webgpu',
    )
  }, 1000)

  let session: ort.InferenceSession
  let executionProvider: OrtExecutionProvider = 'webgpu'
  try {
    try {
      session = await ort.InferenceSession.create(
        manifest.graphUrl,
        createSessionOptions(manifest, options, 'webgpu', preferredOutputLocation, device),
      )
    } catch (webgpuError) {
      if (!options?.allowWasmFallback) {
        throw webgpuError
      }
      executionProvider = 'wasm'
      postStatus(
        'session-fallback',
        `WebGPU could not load '${sessionId}'; retrying with WASM.`,
        requestId,
        sessionId,
        'wasm',
      )
      try {
        session = await ort.InferenceSession.create(
          manifest.graphUrl,
          createSessionOptions(manifest, options, 'wasm'),
        )
      } catch (wasmError) {
        throw new Error(
          `Could not create '${sessionId}' with WebGPU (${String(webgpuError)}) or WASM (${String(wasmError)}).`,
        )
      }
    }
  } finally {
    clearInterval(heartbeat)
  }

  const loadMs = performance.now() - startedAt
  const metadata: Omit<OrtLoadSessionResult, 'loadMs'> = {
    sessionId,
    executionProvider,
    inputNames: Array.from(session.inputNames),
    outputNames: Array.from(session.outputNames),
    inputMetadata: toMetadata(session.inputMetadata),
    outputMetadata: toMetadata(session.outputMetadata),
  }
  postStatus(
    'session-ready',
    `ONNX session '${sessionId}' is ready.`,
    requestId,
    sessionId,
    executionProvider,
    1,
  )
  return { session, executionProvider, metadata, loadMs }
}

async function loadSession(request: OrtLoadSessionRequest, requestId: string): Promise<OrtLoadSessionResult> {
  assertSessionId(request.sessionId)
  assertModelManifest(request.manifest)
  const fingerprint = sessionFingerprint(request)
  const existing = sessions.get(request.sessionId)
  if (existing) {
    if (existing.fingerprint !== fingerprint) {
      throw new Error(`Session '${request.sessionId}' is already loaded with a different manifest or options.`)
    }
    const loaded = await existing.loading
    postStatus(
      'session-ready',
      `ONNX session '${request.sessionId}' was already loaded.`,
      requestId,
      request.sessionId,
      loaded.executionProvider,
      1,
    )
    return { ...loaded.metadata, loadMs: loaded.loadMs }
  }

  const record: SessionRecord = {
    fingerprint,
    loading: Promise.resolve(undefined as never),
    runTail: Promise.resolve(),
    disposed: false,
  }
  record.loading = loadOrtSession(request, requestId).catch((error: unknown) => {
    if (sessions.get(request.sessionId) === record) {
      sessions.delete(request.sessionId)
    }
    throw error
  })
  sessions.set(request.sessionId, record)

  const loaded = await record.loading
  return { ...loaded.metadata, loadMs: loaded.loadMs }
}

let webGpuDevice: WebGpuDevice | undefined

function isWebGpuDevice(value: unknown): value is WebGpuDevice {
  return typeof value === 'object' && value !== null
    && 'createBuffer' in value && 'createComputePipeline' in value && 'queue' in value
}

async function sharedWebGpuDevice(): Promise<WebGpuDevice> {
  if (webGpuDevice) return webGpuDevice
  const device = await ort.env.webgpu.device
  if (!isWebGpuDevice(device)) {
    throw new Error('ONNX Runtime did not expose its WebGPU device after pre-session initialization.')
  }
  webGpuDevice = device
  void device.lost.then((info) => {
    postStatus('webgpu-context-lost', `WebGPU device was lost: ${info.message || info.reason}.`)
  })
  return device
}

function context0SplitFingerprint(request: OrtLoadContext0SplitRequest): string {
  return JSON.stringify({ graphs: request.graphs, options: request.options ?? {} })
}

async function loadContext0Split(
  request: OrtLoadContext0SplitRequest,
  requestId: string,
): Promise<OrtLoadContext0SplitResult> {
  assertSessionId(request.sessionId)
  assertModelManifest(request.graphs.pre)
  assertModelManifest(request.graphs.post)
  if (sessions.has(request.sessionId)) {
    throw new Error(`Session '${request.sessionId}' is already loaded as a regular ONNX session.`)
  }
  const fingerprint = context0SplitFingerprint(request)
  const existing = context0Splits.get(request.sessionId)
  if (existing) {
    if (existing.fingerprint !== fingerprint) {
      throw new Error(`Context0 split '${request.sessionId}' is already loaded with different graphs or options.`)
    }
    const loaded = await existing.loading
    return { ...loaded.metadata, loadMs: loaded.loadMs }
  }
  const record: Context0SplitRecord = {
    fingerprint,
    loading: Promise.resolve(undefined as never),
    runTail: Promise.resolve(),
    disposed: false,
  }
  record.loading = (async () => {
    const startedAt = performance.now()
    const options = { ...request.options, allowWasmFallback: false }
    const pre = await loadOrtSession(
      { sessionId: `${request.sessionId}/pre`, manifest: request.graphs.pre, options },
      requestId,
      'gpu-buffer',
    )
    let device: WebGpuDevice
    let post: LoadedSession
    try {
      device = await sharedWebGpuDevice()
      post = await loadOrtSession(
        { sessionId: `${request.sessionId}/post`, manifest: request.graphs.post, options },
        requestId,
        'cpu',
      )
    } catch (error) {
      await pre.session.release()
      throw error
    }
    if (pre.executionProvider !== 'webgpu' || post.executionProvider !== 'webgpu') {
      await Promise.all([pre.session.release(), post.session.release()])
      throw new Error('Context0 split requires WebGPU for both ONNX sessions.')
    }
    const metadata: Omit<OrtLoadContext0SplitResult, 'loadMs'> = {
      sessionId: request.sessionId,
      executionProvider: 'webgpu',
      inputNames: Array.from(pre.session.inputNames),
      outputNames: Array.from(post.session.outputNames),
      preOutputNames: Array.from(pre.session.outputNames),
      postInputNames: Array.from(post.session.inputNames),
    }
    return {
      pre,
      post,
      executor: createContext0AttentionExecutor(device),
      metadata,
      loadMs: performance.now() - startedAt,
    }
  })().catch((error: unknown) => {
    if (context0Splits.get(request.sessionId) === record) context0Splits.delete(request.sessionId)
    throw error
  })
  context0Splits.set(request.sessionId, record)
  const loaded = await record.loading
  return { ...loaded.metadata, loadMs: loaded.loadMs }
}

function requireGpuBuffer(tensor: ort.Tensor, name: string): WebGpuBuffer {
  if (tensor.type !== 'float32' || tensor.location !== 'gpu-buffer') {
    throw new Error(`Context0 split output '${name}' must be a float32 GPU buffer.`)
  }
  if (
    tensor.dims.length !== CONTEXT0_ATTENTION_SHAPE.length
    || tensor.dims.some((value, index) => value !== CONTEXT0_ATTENTION_SHAPE[index])
  ) {
    throw new Error(
      `Context0 split output '${name}' has shape [${tensor.dims.join(', ')}]; `
      + `expected [${CONTEXT0_ATTENTION_SHAPE.join(', ')}].`,
    )
  }
  const buffer = tensor.gpuBuffer
  if (!buffer || buffer.size < CONTEXT0_ATTENTION_BYTES) {
    throw new Error(
      `Context0 split output '${name}' has ${buffer?.size ?? 0} bytes; `
      + `expected at least ${CONTEXT0_ATTENTION_BYTES}.`,
    )
  }
  return buffer as unknown as WebGpuBuffer
}

async function runContext0Split(
  request: OrtRunSessionRequest,
  requestId: string,
): Promise<OrtRunSessionResult> {
  assertSessionId(request.sessionId)
  assertTensorPayloadMap(request.inputs, 'request.inputs')
  const record = context0Splits.get(request.sessionId)
  if (!record || record.disposed) throw new Error(`Context0 split '${request.sessionId}' is not loaded.`)
  postStatus('inference-queued', `Queued Context0 split inference for '${request.sessionId}'.`, requestId, request.sessionId, 'webgpu')
  return enqueueSessionRun(record, async () => {
    if (record.disposed) throw new Error(`Context0 split '${request.sessionId}' was disposed before inference started.`)
    const loaded = await record.loading
    const requestedOutputs = request.outputs === undefined ? undefined : Array.from(request.outputs)
    if (requestedOutputs?.some((name) => !loaded.post.session.outputNames.includes(name))) {
      throw new Error(`Context0 post session '${request.sessionId}' does not provide every requested output.`)
    }
    const preFeeds: Record<string, ort.Tensor> = {}
    for (const [name, payload] of Object.entries(request.inputs)) preFeeds[name] = toOrtTensor(payload)
    const totalStartedAt = performance.now()
    let inferenceMs = 0
    let preOutputs: ort.InferenceSession.ReturnType | undefined
    let attended: ort.Tensor | undefined
    try {
      const preStartedAt = performance.now()
      preOutputs = await loaded.pre.session.run(preFeeds, request.tag ? { tag: `${request.tag}/pre` } : {})
      inferenceMs += performance.now() - preStartedAt
      const q = preOutputs.triposplat_context0_q
      const k = preOutputs.triposplat_context0_k
      const v = preOutputs.triposplat_context0_v
      if (!q || !k || !v) throw new Error('Context0 pre graph did not produce Q, K, and V GPU outputs.')
      const output = loaded.executor.dispatch({
        q: requireGpuBuffer(q, 'triposplat_context0_q'),
        k: requireGpuBuffer(k, 'triposplat_context0_k'),
        v: requireGpuBuffer(v, 'triposplat_context0_v'),
      })
      attended = ort.Tensor.fromGpuBuffer(output, {
        dataType: 'float32',
        dims: Array.from(CONTEXT0_ATTENTION_SHAPE),
        dispose: () => output.destroy(),
      })
      const postFeeds: Record<string, ort.Tensor> = {}
      for (const inputName of loaded.post.session.inputNames) {
        if (inputName === 'triposplat_context0_attended') {
          postFeeds[inputName] = attended
          continue
        }
        const value = preOutputs[inputName] ?? preFeeds[inputName]
        if (!value) {
          throw new Error(
            `Context0 pre graph did not produce and caller did not provide post input '${inputName}'.`,
          )
        }
        postFeeds[inputName] = value
      }
      const postStartedAt = performance.now()
      const ortOutputs = requestedOutputs === undefined
        ? await loaded.post.session.run(postFeeds, request.tag ? { tag: `${request.tag}/post` } : {})
        : await loaded.post.session.run(postFeeds, requestedOutputs, request.tag ? { tag: `${request.tag}/post` } : {})
      inferenceMs += performance.now() - postStartedAt
      const readbackStartedAt = performance.now()
      const outputEntries = await Promise.all(Object.entries(ortOutputs).map(async ([name, tensor]) => {
        try {
          return [name, await outputTensorPayload(name, tensor)] as const
        } finally {
          tensor.dispose()
        }
      }))
      const readbackMs = performance.now() - readbackStartedAt
      postStatus('inference-complete', `Context0 split inference for '${request.sessionId}' completed.`, requestId, request.sessionId, 'webgpu', 1)
      return {
        sessionId: request.sessionId,
        outputs: Object.fromEntries(outputEntries) as TensorPayloadMap,
        timings: { inferenceMs, readbackMs, totalMs: performance.now() - totalStartedAt },
      }
    } finally {
      for (const tensor of Object.values(preFeeds)) tensor.dispose()
      attended?.dispose()
      if (preOutputs) {
        for (const tensor of Object.values(preOutputs)) tensor.dispose()
      }
    }
  })
}

function toOrtTensor(payload: TensorPayload): ort.Tensor {
  switch (payload.type) {
    case 'float32':
      return new ort.Tensor('float32', payload.data, payload.dims)
    case 'float16':
      return new ort.Tensor('float16', payload.data, payload.dims)
    case 'int32':
      return new ort.Tensor('int32', payload.data, payload.dims)
    case 'int64':
      return new ort.Tensor('int64', payload.data, payload.dims)
  }
}

async function outputTensorPayload(name: string, tensor: ort.Tensor): Promise<TensorPayload> {
  const data = await tensor.getData(true)
  switch (tensor.type) {
    case 'float32': {
      if (!(data instanceof Float32Array)) {
        throw new TypeError(`Output '${name}' declared float32 but returned a different storage type.`)
      }
      const copy = new Float32Array(data.length)
      copy.set(data)
      return createTensorPayload('float32', copy, tensor.dims)
    }
    case 'float16': {
      if (!(data instanceof Uint16Array)) {
        throw new TypeError(`Output '${name}' declared float16 but returned a different storage type.`)
      }
      const copy = new Uint16Array(data.length)
      copy.set(data)
      return createTensorPayload('float16', copy, tensor.dims)
    }
    case 'int32': {
      if (!(data instanceof Int32Array)) {
        throw new TypeError(`Output '${name}' declared int32 but returned a different storage type.`)
      }
      const copy = new Int32Array(data.length)
      copy.set(data)
      return createTensorPayload('int32', copy, tensor.dims)
    }
    case 'int64': {
      if (!(data instanceof BigInt64Array)) {
        throw new TypeError(`Output '${name}' declared int64 but returned a different storage type.`)
      }
      const copy = new BigInt64Array(data.length)
      copy.set(data)
      return createTensorPayload('int64', copy, tensor.dims)
    }
    default:
      throw new TypeError(
        `Output '${name}' uses unsupported dtype '${tensor.type}'. Supported worker payloads are float32, float16, int32, and int64.`,
      )
  }
}

function enqueueSessionRun<T>(record: { runTail: Promise<void> }, task: () => Promise<T>): Promise<T> {
  const result = record.runTail.then(task)
  record.runTail = result.then(() => undefined, () => undefined)
  return result
}

async function runSession(request: OrtRunSessionRequest, requestId: string): Promise<OrtRunSessionResult> {
  assertSessionId(request.sessionId)
  assertTensorPayloadMap(request.inputs, 'request.inputs')
  const record = sessions.get(request.sessionId)
  if (!record || record.disposed) {
    throw new Error(`ONNX session '${request.sessionId}' is not loaded.`)
  }

  postStatus('inference-queued', `Queued inference for '${request.sessionId}'.`, requestId, request.sessionId)
  return enqueueSessionRun(record, async () => {
    if (record.disposed) {
      throw new Error(`ONNX session '${request.sessionId}' was disposed before inference started.`)
    }
    const loaded = await record.loading
    const requestedOutputs = request.outputs === undefined ? undefined : Array.from(request.outputs)
    if (requestedOutputs) {
      const available = new Set(loaded.session.outputNames)
      const seen = new Set<string>()
      for (const output of requestedOutputs) {
        if (typeof output !== 'string' || output.trim().length === 0 || seen.has(output)) {
          throw new TypeError('Requested output names must be unique, non-empty strings.')
        }
        if (!available.has(output)) {
          throw new Error(
            `Session '${request.sessionId}' has no output '${output}'. Available outputs: ${loaded.session.outputNames.join(', ')}.`,
          )
        }
        seen.add(output)
      }
    }

    const feeds: Record<string, ort.Tensor> = {}
    for (const [name, payload] of Object.entries(request.inputs)) {
      feeds[name] = toOrtTensor(payload)
    }

    const totalStartedAt = performance.now()
    const inferenceStartedAt = performance.now()
    postStatus(
      'inference-running',
      `Running inference for '${request.sessionId}'.`,
      requestId,
      request.sessionId,
      loaded.executionProvider,
    )

    let ortOutputs: ort.InferenceSession.ReturnType
    try {
      const runOptions: ort.InferenceSession.RunOptions = request.tag ? { tag: request.tag } : {}
      ortOutputs = requestedOutputs === undefined
        ? await loaded.session.run(feeds, runOptions)
        : await loaded.session.run(feeds, requestedOutputs, runOptions)
    } finally {
      for (const tensor of Object.values(feeds)) {
        tensor.dispose()
      }
    }
    const inferenceMs = performance.now() - inferenceStartedAt

    const readbackStartedAt = performance.now()
    const entries = Object.entries(ortOutputs)
    postStatus(
      'outputs-reading',
      `Reading ${entries.length} output tensor${entries.length === 1 ? '' : 's'} from '${request.sessionId}'.`,
      requestId,
      request.sessionId,
      loaded.executionProvider,
      entries.length === 0 ? 1 : 0,
    )

    const outputEntries = await Promise.all(entries.map(async ([name, tensor], index) => {
      try {
        const payload = await outputTensorPayload(name, tensor)
        postStatus(
          'outputs-reading',
          `Read output '${name}' from '${request.sessionId}'.`,
          requestId,
          request.sessionId,
          loaded.executionProvider,
          entries.length === 0 ? 1 : (index + 1) / entries.length,
        )
        return [name, payload] as const
      } finally {
        tensor.dispose()
      }
    }))
    const outputs = Object.fromEntries(outputEntries) as TensorPayloadMap
    const readbackMs = performance.now() - readbackStartedAt
    const totalMs = performance.now() - totalStartedAt
    postStatus(
      'inference-complete',
      `Inference for '${request.sessionId}' completed.`,
      requestId,
      request.sessionId,
      loaded.executionProvider,
      1,
    )
    return {
      sessionId: request.sessionId,
      outputs,
      timings: { inferenceMs, readbackMs, totalMs },
    }
  })
}

async function disposeContext0Split(sessionId: string, requestId?: string): Promise<boolean> {
  const record = context0Splits.get(sessionId)
  if (!record) return false
  record.disposed = true
  context0Splits.delete(sessionId)
  postStatus('session-disposing', `Disposing Context0 split '${sessionId}'.`, requestId, sessionId)
  await record.runTail
  let loaded: LoadedContext0Split | undefined
  try {
    loaded = await record.loading
  } catch {
    // A failed split load owns no releasable pair of sessions.
  }
  if (loaded) await Promise.all([loaded.pre.session.release(), loaded.post.session.release()])
  postStatus('session-disposed', `Disposed Context0 split '${sessionId}'.`, requestId, sessionId, undefined, 1)
  return true
}

async function disposeSession(sessionId: string, requestId?: string): Promise<boolean> {
  assertSessionId(sessionId)
  if (context0Splits.has(sessionId)) return disposeContext0Split(sessionId, requestId)
  const record = sessions.get(sessionId)
  if (!record) {
    return false
  }

  record.disposed = true
  sessions.delete(sessionId)
  postStatus('session-disposing', `Disposing ONNX session '${sessionId}'.`, requestId, sessionId)
  await record.runTail
  let loaded: LoadedSession | undefined
  try {
    loaded = await record.loading
  } catch {
    // A failed load has no session resources left to release. Preserve disposal semantics.
  }
  if (loaded) {
    await loaded.session.release()
  }
  postStatus('session-disposed', `Disposed ONNX session '${sessionId}'.`, requestId, sessionId, undefined, 1)
  return true
}

async function disposeAll(requestId: string): Promise<string[]> {
  const sessionIds = [...sessions.keys(), ...context0Splits.keys()]
  postStatus('worker-disposing', `Disposing ${sessionIds.length} ONNX session(s).`, requestId)
  await Promise.all(sessionIds.map((sessionId) => disposeSession(sessionId, requestId)))
  postStatus('worker-disposed', 'All ONNX sessions are disposed.', requestId, undefined, undefined, 1)
  return sessionIds
}

async function dispatch(request: OrtWorkerRequest): Promise<void> {
  try {
    switch (request.type) {
      case 'configure-runtime': {
        const result = configureRuntime(request.payload, request.requestId)
        postSuccess(request.type, request.requestId, result)
        return
      }
      case 'load-session': {
        const result = await loadSession(request.payload, request.requestId)
        postSuccess(request.type, request.requestId, result)
        return
      }
      case 'load-context0-split': {
        const result = await loadContext0Split(request.payload, request.requestId)
        postSuccess(request.type, request.requestId, result)
        return
      }
      case 'run-session': {
        const result = await runSession(request.payload, request.requestId)
        postSuccess(request.type, request.requestId, result, tensorPayloadTransferables(result.outputs))
        return
      }
      case 'run-context0-split': {
        const result = await runContext0Split(request.payload, request.requestId)
        postSuccess(request.type, request.requestId, result, tensorPayloadTransferables(result.outputs))
        return
      }
      case 'dispose-session': {
        const disposed = await disposeSession(request.payload.sessionId, request.requestId)
        postSuccess(request.type, request.requestId, { sessionId: request.payload.sessionId, disposed })
        return
      }
      case 'dispose-all': {
        const disposedSessionIds = await disposeAll(request.requestId)
        postSuccess(request.type, request.requestId, { disposedSessionIds })
        return
      }
    }
  } catch (error) {
    postError(request.type, request.requestId, error)
  }
}

workerScope.onmessage = (event: MessageEvent<OrtWorkerRequest>) => {
  void dispatch(event.data)
}
