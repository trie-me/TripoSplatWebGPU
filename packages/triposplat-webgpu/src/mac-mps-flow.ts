import {
  assertLength,
  elementCount,
  TRIPOSPLAT_CAMERA_SHAPE,
  TRIPOSPLAT_FEATURE1_SHAPE,
  TRIPOSPLAT_FEATURE2_SHAPE,
  TRIPOSPLAT_LATENT_SHAPE,
} from './contracts.js'
import { InferenceError, throwIfAborted } from './errors.js'
import type { MacMpsFlowBackendOptions } from './types.js'

const PROTOCOL_VERSION = '1'
const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT
const LATENT_ELEMENTS = elementCount(TRIPOSPLAT_LATENT_SHAPE)
const CAMERA_ELEMENTS = elementCount(TRIPOSPLAT_CAMERA_SHAPE)
const FEATURE1_ELEMENTS = elementCount(TRIPOSPLAT_FEATURE1_SHAPE)
const FEATURE2_ELEMENTS = elementCount(TRIPOSPLAT_FEATURE2_SHAPE)
const REQUEST_ELEMENTS = LATENT_ELEMENTS + CAMERA_ELEMENTS + FEATURE1_ELEMENTS + FEATURE2_ELEMENTS
const RESPONSE_ELEMENTS = LATENT_ELEMENTS + CAMERA_ELEMENTS

export interface MacMpsFlowRequest {
  latent: Float32Array
  camera: Float32Array
  feature1: Float32Array
  feature2: Float32Array
  steps: 4 | 20
  guidanceScale: number
  shift: number
  signal?: AbortSignal
}

export interface MacMpsFlowResult {
  latent: Float32Array
  camera: Float32Array
  wallMs: number
  inferenceMs: number
  modelLoadMs?: number
  sourceCommit?: string
}

function endpoint(serviceUrl: string): URL {
  let url: URL
  try {
    url = new URL(serviceUrl)
  } catch {
    throw new TypeError('macMpsFlow.serviceUrl must be an absolute HTTP URL.')
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new TypeError('macMpsFlow.serviceUrl must use HTTP or HTTPS.')
  }
  if (!['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)) {
    throw new TypeError('macMpsFlow.serviceUrl must use a loopback hostname.')
  }
  if (!url.pathname.endsWith('/')) url.pathname += '/'
  return new URL('v1/flow', url)
}

function finitePositive(value: number, label: string): void {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${label} must be positive and finite.`)
}

function copyRequest(request: MacMpsFlowRequest): ArrayBuffer {
  assertLength('MPS latent', request.latent, LATENT_ELEMENTS)
  assertLength('MPS camera', request.camera, CAMERA_ELEMENTS)
  assertLength('MPS feature1', request.feature1, FEATURE1_ELEMENTS)
  assertLength('MPS feature2', request.feature2, FEATURE2_ELEMENTS)
  const packed = new Float32Array(REQUEST_ELEMENTS)
  let offset = 0
  for (const values of [request.latent, request.camera, request.feature1, request.feature2]) {
    packed.set(values, offset)
    offset += values.length
  }
  return packed.buffer
}

function numericHeader(headers: Headers, name: string): number | undefined {
  const raw = headers.get(name)
  if (raw === null) return undefined
  const value = Number(raw)
  return Number.isFinite(value) && value >= 0 ? value : undefined
}

async function errorDetail(response: Response): Promise<string> {
  const fallback = `${response.status} ${response.statusText}`.trim()
  const text = await response.text().catch(() => '')
  if (!text) return fallback
  try {
    const parsed = JSON.parse(text) as { error?: unknown }
    return typeof parsed.error === 'string' ? parsed.error : fallback
  } catch {
    return text.slice(0, 512)
  }
}

/**
 * Run the complete official shifted-flow sampler through the authenticated
 * localhost fp32 MPS service. The service owns all 40 DiT calls, CFG, and Euler
 * updates, so the browser receives the exact final latent/camera state.
 */
export async function runMacMpsFlow(
  backend: MacMpsFlowBackendOptions,
  request: MacMpsFlowRequest,
): Promise<MacMpsFlowResult> {
  if (!backend || typeof backend.serviceUrl !== 'string' || backend.serviceUrl.trim() === '') {
    throw new TypeError('macMpsFlow.serviceUrl must be non-empty.')
  }
  if (typeof backend.token !== 'string' || backend.token.length < 16) {
    throw new TypeError('macMpsFlow.token must contain at least 16 characters.')
  }
  if (request.steps !== 4 && request.steps !== 20) {
    throw new RangeError('MPS flow steps must be 4 or 20.')
  }
  finitePositive(request.guidanceScale, 'MPS guidanceScale')
  finitePositive(request.shift, 'MPS shift')
  throwIfAborted(request.signal)
  const body = copyRequest(request)
  const flowEndpoint = endpoint(backend.serviceUrl)
  const startedAt = performance.now()
  let response: Response
  try {
    response = await fetch(flowEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Triposplat-Protocol': PROTOCOL_VERSION,
        'X-Triposplat-Token': backend.token,
        'X-Triposplat-Steps': String(request.steps),
        'X-Triposplat-Guidance-Scale': String(request.guidanceScale),
        'X-Triposplat-Shift': String(request.shift),
      },
      body,
      ...(request.signal === undefined ? {} : { signal: request.signal }),
    })
  } catch (error) {
    throwIfAborted(request.signal)
    throw new InferenceError('Could not reach the local Mac MPS flow service.', {
      cause: error,
      diagnostics: { serviceUrl: backend.serviceUrl },
    })
  }
  if (!response.ok) {
    throw new InferenceError(`Mac MPS flow service failed: ${await errorDetail(response)}`, {
      diagnostics: { serviceUrl: backend.serviceUrl, status: response.status },
    })
  }
  const buffer = await response.arrayBuffer()
  if (buffer.byteLength !== RESPONSE_ELEMENTS * FLOAT_BYTES) {
    throw new InferenceError(
      `Mac MPS flow service returned ${buffer.byteLength} bytes; expected ${RESPONSE_ELEMENTS * FLOAT_BYTES}.`,
      { diagnostics: { serviceUrl: backend.serviceUrl } },
    )
  }
  const values = new Float32Array(buffer)
  const latent = new Float32Array(values.subarray(0, LATENT_ELEMENTS))
  const camera = new Float32Array(values.subarray(LATENT_ELEMENTS))
  const wallMs = performance.now() - startedAt
  const modelLoadMs = numericHeader(response.headers, 'X-Triposplat-Model-Load-Ms')
  const sourceCommit = response.headers.get('X-Triposplat-Source-Commit')
  return {
    latent,
    camera,
    wallMs,
    inferenceMs: numericHeader(response.headers, 'X-Triposplat-Inference-Ms') ?? wallMs,
    ...(modelLoadMs === undefined ? {} : { modelLoadMs }),
    ...(sourceCommit === null ? {} : { sourceCommit }),
  }
}
