export interface WebGpuProfilingTensorMetadata {
  dims: number[]
  dataType: string
}

/** Structured-clone-safe copy of one ONNX Runtime WebGPU timestamp record. */
export interface WebGpuProfilingRecord {
  version: 1
  inputsMetadata: WebGpuProfilingTensorMetadata[]
  outputsMetadata: WebGpuProfilingTensorMetadata[]
  kernelId: number
  kernelType: string
  kernelName: string
  programName: string
  /** Nanoseconds from ONNX Runtime's profiling time base. */
  startTime: number
  /** Nanoseconds from ONNX Runtime's profiling time base. */
  endTime: number
}

export interface WebGpuKernelProfile {
  kernelType: string
  kernelName: string
  programName: string
  dispatchCount: number
  totalMs: number
  medianMs: number
  p95Ms: number
}

export interface WebGpuProfilingSummary {
  recordCount: number
  dispatchCount: number
  summedGpuKernelMs: number
  medianDispatchMs: number
  p95DispatchMs: number
  topKernels: WebGpuKernelProfile[]
}

export interface WebGpuRunProfile {
  requested: true
  availability: 'available' | 'timestamp-query-unavailable' | 'no-records'
  timestampQuerySupported: boolean
  records: WebGpuProfilingRecord[]
  summary: WebGpuProfilingSummary
  /** Wall time spent inside InferenceSession.run. */
  inferenceWallMs: number
  /**
   * InferenceSession.run wall time not represented by summed GPU timestamps.
   * This is queue/submission/runtime time, not a hardware-occupancy counter.
   */
  unaccountedWallMs: number
}

function durationMs(record: WebGpuProfilingRecord): number {
  const duration = (record.endTime - record.startTime) / 1_000_000
  return Number.isFinite(duration) && duration >= 0 ? duration : 0
}

function percentile(sorted: readonly number[], fraction: number): number {
  if (sorted.length === 0) return 0
  const index = Math.max(0, Math.ceil(sorted.length * fraction) - 1)
  return sorted[Math.min(index, sorted.length - 1)]
}

function median(sorted: readonly number[]): number {
  if (sorted.length === 0) return 0
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle]
}

export function summarizeWebGpuProfiling(
  records: readonly WebGpuProfilingRecord[],
  topKernelCount = 20,
): WebGpuProfilingSummary {
  if (!Number.isInteger(topKernelCount) || topKernelCount < 0) {
    throw new RangeError('topKernelCount must be a non-negative integer.')
  }
  const durations = records.map(durationMs).sort((left, right) => left - right)
  const grouped = new Map<string, {
    kernelType: string
    kernelName: string
    programName: string
    durations: number[]
  }>()
  for (const record of records) {
    const key = JSON.stringify([record.kernelType, record.kernelName, record.programName])
    let group = grouped.get(key)
    if (!group) {
      group = {
        kernelType: record.kernelType,
        kernelName: record.kernelName,
        programName: record.programName,
        durations: [],
      }
      grouped.set(key, group)
    }
    group.durations.push(durationMs(record))
  }
  const topKernels = Array.from(grouped.values(), (group): WebGpuKernelProfile => {
    const values = group.durations.sort((left, right) => left - right)
    return {
      kernelType: group.kernelType,
      kernelName: group.kernelName,
      programName: group.programName,
      dispatchCount: values.length,
      totalMs: values.reduce((sum, value) => sum + value, 0),
      medianMs: median(values),
      p95Ms: percentile(values, 0.95),
    }
  })
    .sort((left, right) => right.totalMs - left.totalMs)
    .slice(0, topKernelCount)
  return {
    recordCount: records.length,
    dispatchCount: records.length,
    summedGpuKernelMs: durations.reduce((sum, value) => sum + value, 0),
    medianDispatchMs: median(durations),
    p95DispatchMs: percentile(durations, 0.95),
    topKernels,
  }
}
