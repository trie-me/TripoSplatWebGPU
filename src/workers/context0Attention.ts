export const CONTEXT0_ATTENTION_TOKENS = 4101
export const CONTEXT0_ATTENTION_HEADS = 16
export const CONTEXT0_ATTENTION_HEAD_DIM = 64
export const CONTEXT0_ATTENTION_SHAPE = [
  1,
  CONTEXT0_ATTENTION_HEADS,
  CONTEXT0_ATTENTION_TOKENS,
  CONTEXT0_ATTENTION_HEAD_DIM,
] as const
export const CONTEXT0_ATTENTION_ELEMENTS = (
  CONTEXT0_ATTENTION_TOKENS * CONTEXT0_ATTENTION_HEADS * CONTEXT0_ATTENTION_HEAD_DIM
)
export const CONTEXT0_ATTENTION_BYTES = CONTEXT0_ATTENTION_ELEMENTS * Float32Array.BYTES_PER_ELEMENT

export interface WebGpuBuffer {
  readonly size: number
  readonly mapState: 'unmapped' | 'pending' | 'mapped'
  destroy(): void
}

export interface WebGpuDevice {
  readonly lost: Promise<{ message: string; reason: string }>
  readonly queue: { submit(commands: readonly unknown[]): void }
  createShaderModule(descriptor: { code: string; label: string }): unknown
  createComputePipeline(descriptor: {
    label: string
    layout: 'auto'
    compute: { module: unknown; entryPoint: string }
  }): { getBindGroupLayout(index: number): unknown }
  createBuffer(descriptor: { label: string; size: number; usage: number }): WebGpuBuffer
  createBindGroup(descriptor: {
    layout: unknown
    entries: ReadonlyArray<{ binding: number; resource: { buffer: WebGpuBuffer } }>
  }): unknown
  createCommandEncoder(descriptor: { label: string }): {
    beginComputePass(descriptor: { label: string }): {
      setPipeline(pipeline: unknown): void
      setBindGroup(index: number, group: unknown): void
      dispatchWorkgroups(x: number, y?: number, z?: number): void
      end(): void
    }
    finish(): unknown
  }
}

export interface Context0AttentionBuffers {
  q: WebGpuBuffer
  k: WebGpuBuffer
  v: WebGpuBuffer
}

/**
 * One workgroup owns one (head, query) row. Its 64 lanes reduce the Q·K score
 * and retain one output channel each, so no [query,key] probability buffer is
 * materialized. Q and K arrive as normalized fp32 tensors in [H,L,D] layout.
 */
const shader = /* wgsl */ `
struct Buffer { values: array<f32>; };
@group(0) @binding(0) var<storage, read> q: Buffer;
@group(0) @binding(1) var<storage, read> k: Buffer;
@group(0) @binding(2) var<storage, read> v: Buffer;
@group(0) @binding(3) var<storage, read_write> output: Buffer;
var<workgroup> dot: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) workgroup: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>,
) {
  let lane = local.x;
  let head = workgroup.y;
  let query = workgroup.x;
  let rowBase = (head * 4101u + query) * 64u;
  let qValue = q.values[rowBase + lane];
  var runningMax = -3.402823466e+38;
  var denominator = 0.0;
  var accumulator = 0.0;

  for (var key = 0u; key < 4101u; key = key + 1u) {
    let keyBase = (head * 4101u + key) * 64u;
    // Preserve canonical separate Q/K scale-rounding before the dot reduction.
    dot[lane] = (qValue * 0.3535533905932738) * (k.values[keyBase + lane] * 0.3535533905932738);
    workgroupBarrier();
    for (var width = 32u; width > 0u; width = width / 2u) {
      if (lane < width) {
        dot[lane] = dot[lane] + dot[lane + width];
      }
      workgroupBarrier();
    }
    let score = dot[0u];
    let nextMax = max(runningMax, score);
    let oldScale = exp(runningMax - nextMax);
    let weight = exp(score - nextMax);
    accumulator = accumulator * oldScale + weight * v.values[keyBase + lane];
    denominator = denominator * oldScale + weight;
    runningMax = nextMax;
  }
  output.values[rowBase + lane] = accumulator / denominator;
}
`

export interface Context0AttentionExecutor {
  dispatch(buffers: Context0AttentionBuffers): WebGpuBuffer
}

export function createContext0AttentionExecutor(device: WebGpuDevice): Context0AttentionExecutor {
  const module = device.createShaderModule({ code: shader, label: 'triposplat-context0-online-attention' })
  const pipeline = device.createComputePipeline({
    label: 'triposplat-context0-online-attention',
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  })
  const bindGroupLayout = pipeline.getBindGroupLayout(0)
  return {
    dispatch(buffers) {
      const output = device.createBuffer({
        label: 'triposplat-context0-attended',
        size: CONTEXT0_ATTENTION_BYTES,
        usage: 0x80 | 0x04,
      })
      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: buffers.q } },
          { binding: 1, resource: { buffer: buffers.k } },
          { binding: 2, resource: { buffer: buffers.v } },
          { binding: 3, resource: { buffer: output } },
        ],
      })
      const encoder = device.createCommandEncoder({ label: 'triposplat-context0-online-attention' })
      const pass = encoder.beginComputePass({ label: 'triposplat-context0-online-attention' })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(CONTEXT0_ATTENTION_TOKENS, CONTEXT0_ATTENTION_HEADS)
      pass.end()
      device.queue.submit([encoder.finish()])
      return output
    },
  }
}
