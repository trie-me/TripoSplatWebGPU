import { createRoot } from 'react-dom/client'
import { useEffect, useRef, type ReactElement } from 'react'

type Vector = readonly [number, number, number]

const PHI = (1 + Math.sqrt(5)) / 2
const vertices: readonly Vector[] = [
  [-1, PHI, 0], [1, PHI, 0], [-1, -PHI, 0], [1, -PHI, 0],
  [0, -1, PHI], [0, 1, PHI], [0, -1, -PHI], [0, 1, -PHI],
  [PHI, 0, -1], [PHI, 0, 1], [-PHI, 0, -1], [-PHI, 0, 1],
]
const faces = [
  [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
] as const
const edgeKeys = new Set<string>()
for (const face of faces) {
  for (let index = 0; index < face.length; index += 1) {
    const point = face[index]
    const next = face[(index + 1) % face.length]
    edgeKeys.add(`${Math.min(point, next)}-${Math.max(point, next)}`)
  }
}
const edges = [...edgeKeys].map((edge) => edge.split('-').map(Number) as [number, number])

function rotate([x, y, z]: Vector, ax: number, ay: number, az: number): Vector {
  const cx = Math.cos(ax), sx = Math.sin(ax), cy = Math.cos(ay), sy = Math.sin(ay), cz = Math.cos(az), sz = Math.sin(az)
  const y1 = y * cx - z * sx, z1 = y * sx + z * cx, x2 = x * cy + z1 * sy, z2 = -x * sy + z1 * cy
  return [x2 * cz - y1 * sz, x2 * sz + y1 * cz, z2]
}

function project([x, y, z]: Vector, width: number, height: number): readonly [number, number, number] {
  const scale = Math.min(width, height) * .26, perspective = 1 + z * .12
  return [width / 2 + x * scale * perspective, height * .58 + z * scale * .54 - y * scale, z]
}

function pagePath(context: CanvasRenderingContext2D, width: number, height: number): void {
  context.beginPath()
  context.moveTo(width * .29, height * .36)
  context.lineTo(width * .71, height * .36)
  context.lineTo(width * .91, height * .84)
  context.lineTo(width * .09, height * .84)
  context.closePath()
}

function drawPaper(context: CanvasRenderingContext2D, width: number, height: number): void {
  pagePath(context, width, height)
  context.fillStyle = '#0a101d'
  context.fill()
  context.strokeStyle = 'rgba(146, 165, 202, .38)'
  context.lineWidth = 1
  context.stroke()
  context.save()
  pagePath(context, width, height)
  context.clip()
  context.strokeStyle = 'rgba(88, 230, 255, .07)'
  for (let row = 0; row < 14; row += 1) {
    const t = row / 13, y = height * (.36 + t * .48)
    context.beginPath(); context.moveTo(width * (.29 - t * .2), y); context.lineTo(width * (.71 + t * .2), y); context.stroke()
  }
  for (let column = -8; column <= 8; column += 1) {
    context.beginPath(); context.moveTo(width * .5 + column * width * .026, height * .36); context.lineTo(width * .5 + column * width * .051, height * .84); context.stroke()
  }
  context.restore()
}

function drawWireframe(context: CanvasRenderingContext2D, points: readonly Vector[], width: number, height: number): void {
  const flat = points.map(([x, , z]) => project([x, 0, z], width, height))
  context.save(); pagePath(context, width, height); context.clip()
  context.strokeStyle = 'rgba(88, 230, 255, .48)'; context.lineWidth = 1.25; context.lineJoin = 'round'
  for (const [from, to] of edges) {
    context.beginPath(); context.moveTo(flat[from][0], flat[from][1]); context.lineTo(flat[to][0], flat[to][1]); context.stroke()
  }
  context.restore()
}

function drawMesh(context: CanvasRenderingContext2D, points: readonly Vector[], width: number, height: number): void {
  const sortable = faces.map((face, index) => ({ face, index, depth: face.reduce<number>((total, point) => total + points[point][2], 0) / face.length })).sort((a, b) => a.depth - b.depth)
  for (const { face, index } of sortable) {
    const projected = face.map((point) => project(points[point], width, height))
    context.beginPath(); context.moveTo(projected[0][0], projected[0][1]); projected.slice(1).forEach(([x, y]) => context.lineTo(x, y)); context.closePath()
    context.fillStyle = `hsla(${202 + index * 2}, 45%, ${31 + index * .7}%, .92)`
    context.strokeStyle = 'rgba(214, 247, 255, .42)'; context.lineWidth = 1; context.fill(); context.stroke()
  }
}

export function HeroAnimation(): ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    if (!context) return
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const normalized = vertices.map((vertex) => {
      const radius = Math.hypot(...vertex)
      return rotate([vertex[0] / radius, vertex[1] / radius, vertex[2] / radius], -.38, .76, -.08)
    })
    const emergenceDuration = 3_200
    const zRotationSpeed = 0.0001
    let frame = 0
    let width = 0
    let height = 0
    let lastTime = 0
    let animationStartTime: number | undefined
    const render = (time = lastTime) => {
      lastTime = time
      if (!reducedMotion && animationStartTime === undefined && time > 0) animationStartTime = time
      const elapsed = reducedMotion ? emergenceDuration : Math.max(0, time - (animationStartTime ?? time))
      const emergenceProgress = Math.min(1, elapsed / emergenceDuration)
      const eased = 1 - Math.pow(1 - emergenceProgress, 3)
      const zRotation = emergenceProgress >= 1 ? (elapsed - emergenceDuration) * zRotationSpeed : 0
      const mesh = normalized.map((vertex) => {
        const [x, y, z] = rotate(vertex, 0, 0, zRotation)
        return [x, y + (-1.18 + eased * 2.5), z] as Vector
      })
      context.clearRect(0, 0, width, height)
      drawPaper(context, width, height)
      drawWireframe(context, normalized, width, height)
      drawMesh(context, mesh, width, height)
      if (!reducedMotion) frame = requestAnimationFrame(render)
    }
    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      width = rect.width; height = rect.height
      canvas.width = Math.max(1, Math.round(width * dpr)); canvas.height = Math.max(1, Math.round(height * dpr))
      context.setTransform(dpr, 0, 0, dpr, 0, 0); render()
    }
    const observer = new ResizeObserver(resize)
    observer.observe(canvas)
    return () => { observer.disconnect(); cancelAnimationFrame(frame) }
  }, [])
  return <div className="hero-animation"><span className="hero-animation-label">LOCAL SPATIAL STUDY</span><canvas ref={canvasRef} aria-hidden="true" /></div>
}

const mount = document.querySelector<HTMLElement>('#hero-animation-root')
if (mount) createRoot(mount).render(<HeroAnimation />)
