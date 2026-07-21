import { createRuntime, type CreateRuntimeOptions, type RuntimeStatus } from './runtime.js'

const CANARY_ONNX_BASE64 = 'CAgSEXRyaXBvc3BsYXQtd2ViZ3B1OnEKGQoFaW5wdXQKA3R3bxIGb3V0cHV0IgNNdWwSGHRyaXBvc3BsYXRfd2ViZ3B1X2NhbmFyeSoPCAEQASIEAAAAQEIDdHdvWhMKBWlucHV0EgoKCAgBEgQKAggEYhQKBm91dHB1dBIKCggIARIECgIIBEIECgAQEQ=='

const GPU_MAP_READ = 0x0001
const GPU_COPY_SRC = 0x0004
const GPU_COPY_DST = 0x0008
const GPU_STORAGE = 0x0080

const WEBGPU_LIMIT_NAMES = [
  'maxTextureDimension1D',
  'maxTextureDimension2D',
  'maxTextureDimension3D',
  'maxTextureArrayLayers',
  'maxBindGroups',
  'maxBindingsPerBindGroup',
  'maxDynamicUniformBuffersPerPipelineLayout',
  'maxDynamicStorageBuffersPerPipelineLayout',
  'maxSampledTexturesPerShaderStage',
  'maxSamplersPerShaderStage',
  'maxStorageBuffersPerShaderStage',
  'maxStorageTexturesPerShaderStage',
  'maxUniformBuffersPerShaderStage',
  'maxUniformBufferBindingSize',
  'maxStorageBufferBindingSize',
  'maxVertexBuffers',
  'maxBufferSize',
  'maxVertexAttributes',
  'maxVertexBufferArrayStride',
  'maxInterStageShaderComponents',
  'maxColorAttachments',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
  'maxComputeWorkgroupsPerDimension',
] as const

export type WebGpuCheckStageId =
  | 'secure-context'
  | 'api'
  | 'adapter'
  | 'requirements'
  | 'device'
  | 'compute'
  | 'ort-canary'

export interface WebGpuModelRequirements {
  requiredFeatures?: readonly string[]
  requiredLimits?: Readonly<Record<string, number>>
}

export interface WebGpuCheckStage {
  id: WebGpuCheckStageId
  status: 'passed' | 'failed' | 'skipped'
  message: string
  durationMs: number
  details?: Record<string, unknown>
}

export interface WebGpuModelCheckOptions {
  requirements?: WebGpuModelRequirements
  powerPreference?: 'low-power' | 'high-performance'
  forceFallbackAdapter?: boolean
  runOrtCanary?: boolean
  workerUrl?: string | URL
  baseUrl?: string | URL
  signal?: AbortSignal
  onStage?: (stage: WebGpuCheckStage) => void
  onRuntimeStatus?: (status: RuntimeStatus) => void
}

export interface WebGpuModelCheckReport {
  supported: boolean
  qualification: 'blocked' | 'compute-verified' | 'ort-webgpu-verified'
  adapterName?: string | undefined
  features: string[]
  limits: Record<string, number>
  stages: WebGpuCheckStage[]
  warnings: string[]
}
interface BufferLike {
  getMappedRange(): ArrayBuffer
  mapAsync(mode: number): Promise<void>
  unmap(): void
  destroy(): void
}

interface ComputePassLike {
  setPipeline(pipeline: PipelineLike): void
  setBindGroup(index: number, group: unknown): void
  dispatchWorkgroups(count: number): void
  end(): void
}

interface CommandEncoderLike {
  beginComputePass(): ComputePassLike
  copyBufferToBuffer(source: BufferLike, sourceOffset: number, destination: BufferLike, destinationOffset: number, size: number): void
  finish(): unknown
}

interface PipelineLike {
  getBindGroupLayout(index: number): unknown
}

interface DeviceLike {
  queue: { submit(commands: unknown[]): void }
  createShaderModule(descriptor: { code: string }): unknown
  createComputePipeline(descriptor: Record<string, unknown>): PipelineLike
  createBuffer(descriptor: { size: number; usage: number; mappedAtCreation?: boolean }): BufferLike
  createBindGroup(descriptor: Record<string, unknown>): unknown
  createCommandEncoder(): CommandEncoderLike
  pushErrorScope?(filter: 'validation'): void
  popErrorScope?(): Promise<{ message?: string } | null>
  destroy(): void
}

interface AdapterLike {
  limits: Record<string, number>
  features?: Iterable<string>
  info?: { description?: string; vendor?: string; architecture?: string }
  requestDevice(descriptor?: {
    requiredFeatures?: string[]
    requiredLimits?: Record<string, number>
  }): Promise<DeviceLike>
}

interface GpuLike {
  requestAdapter(options?: {
    powerPreference?: 'low-power' | 'high-performance'
    forceFallbackAdapter?: boolean
  }): Promise<AdapterLike | null>
}

function now(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now()
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw signal.reason ?? new DOMException('Operation aborted.', 'AbortError')
}

function gpu(): GpuLike | undefined {
  if (typeof navigator === 'undefined') return undefined
  return (navigator as Navigator & { gpu?: GpuLike }).gpu
}

function adapterFeatures(adapter: AdapterLike): string[] {
  return adapter.features ? Array.from(adapter.features).sort() : []
}

function adapterLimits(adapter: AdapterLike, additionalNames: readonly string[] = []): Record<string, number> {
  const result: Record<string, number> = {}
  const names = new Set<string>([
    ...WEBGPU_LIMIT_NAMES,
    ...Object.keys(adapter.limits),
    ...additionalNames,
  ])
  for (const name of names) {
    const value = adapter.limits[name]
    if (typeof value === 'number') result[name] = value
  }
  return result
}

function canaryBytes(): Uint8Array<ArrayBuffer> {
  const binary = atob(CANARY_ONNX_BASE64)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index)
  return bytes
}

function expected(values: Float32Array, target: readonly number[]): boolean {
  return values.length === target.length && target.every((value, index) => Math.abs(values[index] - value) < 1e-6)
}
async function runComputeCanary(device: DeviceLike): Promise<number[]> {
  const input = new Float32Array([1, 2, 3, 4])
  const byteLength = input.byteLength
  const inputBuffer = device.createBuffer({
    size: byteLength,
    usage: GPU_STORAGE | GPU_COPY_DST,
    mappedAtCreation: true,
  })
  const outputBuffer = device.createBuffer({ size: byteLength, usage: GPU_STORAGE | GPU_COPY_SRC })
  const readbackBuffer = device.createBuffer({ size: byteLength, usage: GPU_COPY_DST | GPU_MAP_READ })
  try {
    new Float32Array(inputBuffer.getMappedRange()).set(input)
    inputBuffer.unmap()
    device.pushErrorScope?.('validation')
    const module = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> input_values: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
        @compute @workgroup_size(4)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          if (id.x < 4u) { output_values[id.x] = input_values[id.x] * 2.0; }
        }
      `,
    })
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    })
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
      ],
    })
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(1)
    pass.end()
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, byteLength)
    device.queue.submit([encoder.finish()])
    await readbackBuffer.mapAsync(GPU_MAP_READ)
    const values = Array.from(new Float32Array(readbackBuffer.getMappedRange().slice(0)))
    readbackBuffer.unmap()
    const validationError = await device.popErrorScope?.()
    if (validationError) throw new Error(validationError.message || 'WebGPU validation failed.')
    if (!expected(new Float32Array(values), [2, 4, 6, 8])) {
      throw new Error(`WebGPU compute returned [${values.join(', ')}], expected [2, 4, 6, 8].`)
    }
    return values
  } finally {
    inputBuffer.destroy()
    outputBuffer.destroy()
    readbackBuffer.destroy()
  }
}

async function runOrtCanary(options: WebGpuModelCheckOptions): Promise<Record<string, unknown>> {
  const modelUrl = URL.createObjectURL(new Blob([canaryBytes()], { type: 'application/octet-stream' }))
  const runtimeOptions: CreateRuntimeOptions = { executionProviders: ['webgpu'] }
  if (options.onRuntimeStatus !== undefined) runtimeOptions.onStatus = options.onRuntimeStatus
  if (options.workerUrl !== undefined) runtimeOptions.workerUrl = options.workerUrl
  if (options.baseUrl !== undefined) runtimeOptions.baseUrl = options.baseUrl
  const runtime = createRuntime(runtimeOptions)
  try {
    const graph = await runtime.loadGraph('webgpu-compatibility-canary', { url: modelUrl })
    if (graph.executionProvider !== 'webgpu') {
      throw new Error(`ONNX Runtime selected '${graph.executionProvider}' instead of WebGPU.`)
    }
    const result = await runtime.runGraph('webgpu-compatibility-canary', {
      input: { type: 'float32', dims: [4], data: new Float32Array([1, 2, 3, 4]) },
    }, { transferInputs: false, tag: 'webgpu-compatibility-canary' })
    const output = result.outputs.output
    if (!output || output.type !== 'float32' || !expected(output.data, [2, 4, 6, 8])) {
      throw new Error('ONNX Runtime WebGPU canary returned an unexpected output.')
    }
    return {
      executionProvider: graph.executionProvider,
      graphLoadMs: graph.loadMs,
      inferenceMs: result.timings.inferenceMs,
      readbackMs: result.timings.readbackMs,
      output: Array.from(output.data),
    }
  } finally {
    await runtime.dispose().catch(() => undefined)
    URL.revokeObjectURL(modelUrl)
  }
}
export async function runWebGpuModelChecks(
  options: WebGpuModelCheckOptions = {},
): Promise<WebGpuModelCheckReport> {
  const stages: WebGpuCheckStage[] = []
  const warnings = [
    'Browsers do not expose reliable total GPU or unified-memory capacity.',
    'A passing canary does not prove that every full TripoSplat graph fits in memory.',
  ]
  let features: string[] = []
  let limits: Record<string, number> = {}
  let qualification: WebGpuModelCheckReport['qualification'] = 'blocked'

  const record = (
    id: WebGpuCheckStageId,
    status: WebGpuCheckStage['status'],
    stageMessage: string,
    startedAt: number,
    details?: Record<string, unknown>,
  ): void => {
    const stage: WebGpuCheckStage = {
      id,
      status,
      message: stageMessage,
      durationMs: now() - startedAt,
    }
    if (details !== undefined) stage.details = details
    stages.push(stage)
    options.onStage?.(stage)
  }

  let startedAt = now()
  if (typeof globalThis.isSecureContext === 'boolean' && !globalThis.isSecureContext) {
    record('secure-context', 'failed', 'WebGPU requires HTTPS or localhost.', startedAt)
    return { supported: false, qualification, features, limits, stages, warnings }
  }
  record('secure-context', 'passed', 'Secure browser context available.', startedAt)

  startedAt = now()
  const webGpu = gpu()
  if (!webGpu) {
    record('api', 'failed', 'navigator.gpu is unavailable.', startedAt)
    return { supported: false, qualification, features, limits, stages, warnings }
  }
  record('api', 'passed', 'WebGPU API is exposed.', startedAt)

  throwIfAborted(options.signal)
  startedAt = now()
  let adapter: AdapterLike | null
  try {
    adapter = await webGpu.requestAdapter({
      powerPreference: options.powerPreference ?? 'high-performance',
      forceFallbackAdapter: options.forceFallbackAdapter ?? false,
    })
  } catch (error) {
    record('adapter', 'failed', `WebGPU adapter request failed: ${message(error)}`, startedAt)
    return { supported: false, qualification, features, limits, stages, warnings }
  }
  if (!adapter) {
    record('adapter', 'failed', 'No WebGPU adapter was returned.', startedAt)
    return { supported: false, qualification, features, limits, stages, warnings }
  }
  features = adapterFeatures(adapter)
  limits = adapterLimits(adapter, Object.keys(options.requirements?.requiredLimits ?? {}))
  const adapterName = adapter.info?.description
    || [adapter.info?.vendor, adapter.info?.architecture].filter(Boolean).join(' ')
    || undefined
  record('adapter', 'passed', adapterName ? `Using ${adapterName}.` : 'Hardware WebGPU adapter acquired.', startedAt)

  throwIfAborted(options.signal)
  startedAt = now()
  const requiredFeatures = Array.from(options.requirements?.requiredFeatures ?? [])
  const requiredLimits = { ...(options.requirements?.requiredLimits ?? {}) }
  const missingFeatures = requiredFeatures.filter((feature) => !features.includes(feature))
  const insufficientLimits = Object.entries(requiredLimits).filter(([name, required]) => {
    const available = limits[name] ?? 0
    return !Number.isFinite(required) || required < 0 || available < required
  })
  if (missingFeatures.length > 0 || insufficientLimits.length > 0) {
    record('requirements', 'failed', 'The adapter does not satisfy the declared model requirements.', startedAt, {
      missingFeatures,
      insufficientLimits: insufficientLimits.map(([name, required]) => ({
        name,
        required,
        available: limits[name] ?? 0,
      })),
    })
    return { supported: false, qualification, adapterName, features, limits, stages, warnings }
  }
  record('requirements', 'passed', 'Declared WebGPU features and limits are available.', startedAt, {
    requiredFeatures,
    requiredLimits,
  })

  throwIfAborted(options.signal)
  startedAt = now()
  let device: DeviceLike
  try {
    device = await adapter.requestDevice({ requiredFeatures, requiredLimits })
    record('device', 'passed', 'The browser granted a WebGPU device.', startedAt)
  } catch (error) {
    record('device', 'failed', `WebGPU device request failed: ${message(error)}`, startedAt)
    return { supported: false, qualification, adapterName, features, limits, stages, warnings }
  }

  throwIfAborted(options.signal)
  startedAt = now()
  try {
    const output = await runComputeCanary(device)
    qualification = 'compute-verified'
    record('compute', 'passed', 'Compute, copy, submit, and readback succeeded.', startedAt, { output })
  } catch (error) {
    record('compute', 'failed', `WebGPU compute canary failed: ${message(error)}`, startedAt)
    return { supported: false, qualification: 'blocked', adapterName, features, limits, stages, warnings }
  } finally {
    device.destroy()
  }

  if (options.runOrtCanary === false) {
    record('ort-canary', 'skipped', 'ONNX Runtime WebGPU canary was disabled.', now())
  } else {
    throwIfAborted(options.signal)
    startedAt = now()
    try {
      const details = await runOrtCanary(options)
      qualification = 'ort-webgpu-verified'
      record('ort-canary', 'passed', 'ONNX Runtime created and ran a WebGPU session.', startedAt, details)
    } catch (error) {
      record('ort-canary', 'failed', `ONNX Runtime WebGPU canary failed: ${message(error)}`, startedAt)
      return { supported: false, qualification, adapterName, features, limits, stages, warnings }
    }
  }

  return { supported: true, qualification, adapterName, features, limits, stages, warnings }
}
