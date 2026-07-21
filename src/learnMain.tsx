import { StrictMode, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import './learn.css'

type Stage = {
  index: string
  title: string
  plain: string
  technical: string
  tone: string
}

const stages: Stage[] = [
  { index: '01', title: 'See', plain: 'Read the picture two ways', technical: 'DINOv3 + Flux VAE', tone: 'cyan' },
  { index: '02', title: 'Imagine', plain: 'Build a compact 3D plan', technical: 'Rectified-flow DiT', tone: 'violet' },
  { index: '03', title: 'Grow', plain: 'Put detail where it matters', technical: 'Dynamic octree', tone: 'lime' },
  { index: '04', title: 'Paint', plain: 'Turn points into soft 3D marks', technical: 'Gaussian decoder', tone: 'amber' },
  { index: '05', title: 'Show', plain: 'Render or export the scene', technical: 'PLY / .splat', tone: 'rose' },
]

const gaussianParts = [
  ['μ', 'Position', 'where its center lives in 3D'],
  ['s', 'Scale', 'how wide it spreads on three axes'],
  ['q', 'Rotation', 'which way its ellipsoid points'],
  ['α', 'Opacity', 'how strongly it covers what is behind it'],
  ['c', 'Color', 'what light it contributes to the view'],
]

function ArrowIcon() {
  return <span aria-hidden="true">↗</span>
}

function Eyebrow({ children }: { children: ReactNode }) {
  return <p className="eyebrow"><span aria-hidden="true">◆</span>{children}</p>
}

function StageRail({ technical = false }: { technical?: boolean }) {
  return <ol className="stage-rail" aria-label="TripoSplat generation stages">
    {stages.map((stage, index) => <li key={stage.index} className={`stage-card tone-${stage.tone}`}>
      <span className="stage-index">{stage.index}</span>
      <div className="stage-orbit" aria-hidden="true"><i></i><i></i><i></i></div>
      <h3>{stage.title}</h3>
      <p>{technical ? stage.technical : stage.plain}</p>
      {index < stages.length - 1 && <span className="stage-arrow" aria-hidden="true">→</span>}
    </li>)}
  </ol>
}

function AudienceChooser() {
  return <nav className="audience-chooser" aria-label="Choose an explanation level">
    <a href="#generalist"><span>01</span><strong>Generalist</strong><small>No equations · about 6 min</small><b aria-hidden="true">Start with intuition →</b></a>
    <a href="#undergrad"><span>02</span><strong>Undergrad math</strong><small>Linear algebra + calculus · about 12 min</small><b aria-hidden="true">See the machinery →</b></a>
    <a href="#researcher"><span>03</span><strong>Researcher</strong><small>Tensor contracts + porting · about 15 min</small><b aria-hidden="true">Inspect the system →</b></a>
  </nav>
}

export function App() {
  return <>
    <a className="learn-skip" href="#learning-paths">Skip to learning paths</a>
    <header className="learn-topbar">
      <a className="brand learn-brand" href="/" aria-label="TripoSplat WebGPU home"><span className="brand-mark" aria-hidden="true"><span></span></span><span>TRIPOSPLAT <b>LEARN</b></span></a>
      <nav className="learn-nav" aria-label="Learning levels">
        <a href="#generalist">Generalist</a><a href="#undergrad">Math</a><a href="#researcher">Research</a><a href="#platform">Portability</a><a className="run-link" href="/e2e-web">Run it <ArrowIcon /></a>
      </nav>
    </header>

    <main className="learn-main">
      <section className="learn-hero learn-shell" aria-labelledby="learn-title">
        <div className="hero-grid">
          <div>
            <Eyebrow>One image in. A spatial scene out.</Eyebrow>
            <h1 id="learn-title">How does a picture become <em>3D?</em></h1>
            <p className="hero-deck">Follow the same system through three lenses: first as a visual story, then as mathematics, and finally as a browser inference architecture.</p>
            <div className="hero-actions"><a className="primary-link" href="#learning-paths">Choose your level <span aria-hidden="true">↓</span></a><a className="quiet-link" href="#five-stage-map">See the pipeline</a></div>
          </div>
          <div className="hero-object" aria-label="Stylized illustration of a 2D image becoming a cloud of 3D Gaussian ellipsoids">
            <div className="image-plane"><span>2D</span><i></i><i></i><i></i></div>
            <div className="transfer-lines" aria-hidden="true"><i></i><i></i><i></i><i></i></div>
            <div className="splat-cloud" aria-hidden="true">{Array.from({ length: 18 }, (_, index) => <i key={index}></i>)}</div>
            <span className="object-label label-input">source image</span><span className="object-label label-output">spatial scene</span>
          </div>
        </div>
        <div id="five-stage-map" className="hero-pipeline"><span>THE WHOLE IDEA</span><StageRail /></div>
      </section>

      <section id="learning-paths" className="path-section learn-shell" aria-labelledby="paths-title">
        <Eyebrow>Pick the right altitude</Eyebrow><h2 id="paths-title">Three explanations. One pipeline.</h2>
        <p className="section-lead">Each level is complete on its own. Read downward for a progressively more technical account.</p>
        <AudienceChooser />
      </section>

      <article id="generalist" className="level-section level-generalist">
        <div className="learn-shell">
          <header className="level-heading"><div><span className="level-number">LEVEL 01</span><Eyebrow>For the curious generalist</Eyebrow><h2>Think in soft, colored fireflies.</h2></div><p>TripoSplat does not carve a traditional solid model directly. It predicts a cloud of many tiny, translucent 3D ellipsoids—Gaussian “splats.” From a camera, those soft marks overlap into a convincing image.</p></header>

          <section className="split-story" aria-labelledby="guess-title">
            <div className="story-copy"><span className="chapter">01 · THE INFERENCE PROBLEM</span><h3 id="guess-title">One view, many possible objects.</h3><p>A photograph shows only the surfaces facing the camera. The back, exact depth, and hidden details are missing. TripoSplat therefore performs <strong>learned inference</strong>, not measurement: it uses patterns learned from 3D examples to propose a plausible full object consistent with the visible image.</p><aside><b>Important distinction</b><span>The result is a generated 3D hypothesis. It is not a scan and cannot recover information that the image never contained.</span></aside></div>
            <div className="ambiguity-figure" aria-label="One silhouette leading to several possible depth profiles"><div className="photo-card"><span>What we see</span><i></i></div><span className="fork" aria-hidden="true">→</span><div className="possibilities"><i></i><i></i><i></i><span>What might exist</span></div></div>
          </section>

          <section className="lesson-block" aria-labelledby="splat-title">
            <div className="lesson-intro"><span className="chapter">02 · THE REPRESENTATION</span><h3 id="splat-title">A splat is a tiny soft 3D brushstroke.</h3><p>Each Gaussian carries just enough information to occupy space and contribute appearance. Hundreds of thousands can describe fine structure without first turning the object into a mesh of hard triangles.</p></div>
            <div className="gaussian-grid">{gaussianParts.map(([symbol, title, description]) => <div className="gaussian-part" key={title}><span>{symbol}</span><h4>{title}</h4><p>{description}</p></div>)}</div>
            <div className="plain-render"><div className="ellipsoid-demo" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div><p><strong>Rendering is organized transparency.</strong> Project each ellipsoid onto the screen, order contributions along each ray, then blend color from front to back. Soft edges merge into surfaces.</p></div>
          </section>

          <section className="lesson-block" aria-labelledby="journey-title">
            <div className="lesson-intro"><span className="chapter">03 · THE JOURNEY</span><h3 id="journey-title">Five jobs turn pixels into space.</h3><p>The model first understands the image, then invents a compact 3D plan, spends more points on detailed regions, and decodes those points into renderable Gaussians.</p></div>
            <StageRail />
            <div className="translation-grid">
              <div><b>See</b><p>Two image encoders collect complementary clues: semantic structure and local appearance.</p></div>
              <div><b>Imagine</b><p>A transformer repeatedly refines random latent values into a coherent object-and-camera description.</p></div>
              <div><b>Grow</b><p>An octree subdivides promising regions, concentrating spatial samples where geometry needs detail.</p></div>
              <div><b>Paint</b><p>A decoder assigns position offsets, scale, rotation, opacity, and color to the selected points.</p></div>
              <div><b>Show</b><p>The browser packages the canonical scene as PLY or <code>.splat</code> and hands it to a viewer.</p></div>
            </div>
          </section>

          <section className="browser-story" aria-labelledby="browser-story-title">
            <div><span className="chapter">04 · WHY WEBGPU?</span><h3 id="browser-story-title">The browser becomes the inference computer.</h3><p>The original research code targets Python, PyTorch, and CUDA. TripoSplatWebGPU separates that numerical recipe into browser-loadable graphs plus TypeScript orchestration. ONNX Runtime Web maps supported tensor operations to WebGPU; the host code handles image preparation, the sampling loop, the adaptive octree, caching, cancellation, and export.</p></div>
            <div className="local-flow"><span className="device">YOUR DEVICE</span><div><b>Image</b><i>stays local</i></div><em>→</em><div><b>Browser</b><i>runs 5 stages</i></div><em>→</em><div><b>3D files</b><i>you control</i></div><p>The static website serves interface code. A model CDN supplies versioned weights. Inference happens in the browser—not in a site-owned inference API.</p></div>
          </section>
          <a className="next-level" href="#undergrad"><span>Next: expose the mathematics</span><b>LEVEL 02 ↓</b></a>
        </div>
      </article>

      <article id="undergrad" className="level-section level-undergrad">
        <div className="learn-shell">
          <header className="level-heading"><div><span className="level-number">LEVEL 02</span><Eyebrow>For undergraduates</Eyebrow><h2>Geometry, probability, and an ODE.</h2></div><p>You need linear algebra, multivariable calculus, basic probability, and the idea of a neural network as a learned function. The exact training objective is not required to understand browser inference.</p></header>

          <section className="math-section" aria-labelledby="gaussian-math-title">
            <div className="math-copy"><span className="chapter">01 · 3D GAUSSIANS</span><h3 id="gaussian-math-title">From five attributes to a projected footprint.</h3><p>A Gaussian centered at <i>μ</i> uses a positive-definite covariance <i>Σ</i>. TripoSplat represents its shape with positive axis scales and a quaternion rotation, which guarantees a valid ellipsoid when assembled as:</p></div>
            <div className="equation-card"><span>shape</span><code>Σ = R(q) diag(sₓ², sᵧ², s_z²) R(q)ᵀ</code><p><i>R(q)</i> converts quaternion <i>q</i> to a rotation matrix.</p></div>
            <div className="equation-card"><span>density</span><code>G(x) = exp[−½ (x−μ)ᵀ Σ⁻¹ (x−μ)]</code><p>Density decays smoothly away from the center.</p></div>
            <div className="equation-card wide"><span>front-to-back compositing</span><code>C = Σᵢ Tᵢ αᵢ cᵢ &nbsp;&nbsp; with &nbsp;&nbsp; Tᵢ = ∏ⱼ&lt;ᵢ (1−αⱼ)</code><p><i>Tᵢ</i> is the light not already blocked by earlier splats. Differentiable projection and blending let a training system learn Gaussian parameters from rendered errors.</p></div>
          </section>

          <section className="math-section math-flow" aria-labelledby="flow-math-title">
            <div className="math-copy"><span className="chapter">02 · RECTIFIED FLOW</span><h3 id="flow-math-title">Turn noise into a structured latent.</h3><p>At inference time, the DiT predicts a velocity field over latent state <i>z</i>. Conceptually, generation solves an ordinary differential equation conditioned on image features <i>c</i>.</p></div>
            <div className="equation-card"><span>continuous view</span><code>dz / dt = vθ(z, t | c)</code><p>The learned vector field says how the state should move at time <i>t</i>.</p></div>
            <div className="equation-card"><span>Euler update</span><code>zₖ₊₁ = zₖ + Δt · v_cfg(zₖ, tₖ)</code><p>The browser controls this iterative schedule in TypeScript.</p></div>
            <div className="equation-card wide"><span>classifier-free guidance</span><code>v_cfg = v_u + w (v_c − v_u)</code><p>The same DiT is evaluated unconditionally (<i>v<sub>u</sub></i>) and with image conditioning (<i>v<sub>c</sub></i>). Guidance scale <i>w</i> strengthens image agreement. Thus 20 sampling steps require 40 DiT calls in the official path.</p></div>
          </section>

          <section className="math-section" aria-labelledby="octree-title">
            <div className="math-copy"><span className="chapter">03 · LEARNED DENSITY CONTROL</span><h3 id="octree-title">Resolution is a decision, not a fixed grid.</h3><p>A dense high-resolution 3D lattice wastes work in empty space. An octree starts coarse. For each active cell, an occupancy network predicts eight child logits. High-probability children survive; their cells can subdivide again.</p></div>
            <div className="octree-visual" aria-label="Three successive octree subdivision levels"><div><i></i><span>coarse</span></div><em>→</em><div className="level-two">{Array.from({ length: 8 }, (_, i) => <i key={i}></i>)}<span>select</span></div><em>→</em><div className="level-three">{Array.from({ length: 18 }, (_, i) => <i key={i}></i>)}<span>refine</span></div></div>
            <div className="equation-card wide"><span>one child decision</span><code>p(child occupied) = σ(ℓ) = 1 / (1 + e⁻ℓ)</code><p>The learned logits guide where the representation spends its point budget. This is how one compact latent can support variable output density.</p></div>
          </section>

          <section className="math-section" aria-labelledby="conditioning-title">
            <div className="math-copy"><span className="chapter">04 · CONDITIONING + DECODING</span><h3 id="conditioning-title">Two image feature streams constrain one 3D latent.</h3><p>DINOv3 emits high-dimensional tokens carrying semantic/spatial cues. A Flux VAE emits lower-dimensional appearance features. The DiT jointly attends to these features while evolving 8,192 latent tokens and a camera state. The octree then proposes 3D points; a Gaussian transformer maps each point plus the latent condition to 480 raw channels, which host code activates and packs into scene attributes.</p></div>
            <div className="tensor-story"><div><span>ImageNet-normalized RGB</span><b>1 × 3 × 1024 × 1024</b></div><em>↙ ↘</em><div><span>DINO tokens</span><b>4,101 × 1,280</b></div><div><span>VAE tokens</span><b>4,101 × 128</b></div><em>↘ ↙</em><div><span>flow latent</span><b>8,192 × 16</b></div><em>↓</em><div><span>variable points</span><b>P × 3 → P × 480</b></div></div>
          </section>
          <a className="next-level" href="#researcher"><span>Next: inspect contracts and porting boundaries</span><b>LEVEL 03 ↓</b></a>
        </div>
      </article>

      <article id="researcher" className="level-section level-researcher">
        <div className="learn-shell">
          <header className="level-heading"><div><span className="level-number">LEVEL 03</span><Eyebrow>For researchers + systems engineers</Eyebrow><h2>The port is a graph partition.</h2></div><p>TripoSplatWebGPU preserves the official PyTorch implementation as the numerical oracle. The browser system decomposes the executable into static ONNX graph islands and explicit host-controlled algorithms, then validates boundaries stage by stage.</p></header>

          <section className="research-grid" aria-labelledby="contracts-title">
            <div className="research-copy"><span className="chapter">01 · GRAPH CONTRACTS</span><h3 id="contracts-title">Five graph families, explicit tensor boundaries.</h3><p>Fixed graph interfaces make parity measurable and lifetime management possible. Sessions are staged rather than retained simultaneously.</p></div>
            <div className="contract-table-wrap"><table className="contract-table"><thead><tr><th>Graph</th><th>Principal input</th><th>Principal output</th><th>Host responsibility</th></tr></thead><tbody>
              <tr><td>DINOv3</td><td><code>[1,3,1024,1024]</code></td><td><code>[1,4101,1280]</code></td><td>resize, alpha geometry, normalization</td></tr>
              <tr><td>Flux VAE</td><td>RGB + explicit ε</td><td><code>[1,4101,128]</code></td><td>sample ε; preserve stochastic semantics</td></tr>
              <tr><td>DiT invocation</td><td>latent, camera, t, features</td><td>latent + camera velocity</td><td>CFG, schedule, Euler update</td></tr>
              <tr><td>Occupancy</td><td><code>[1,8192,3]</code> centers</td><td><code>[1,8192,8]</code> logits</td><td>traverse, compact, resample, jitter</td></tr>
              <tr><td>Gaussian decoder</td><td><code>[1,P,3]</code> + latent</td><td><code>[1,P,480]</code></td><td>activate, canonicalize, export</td></tr>
            </tbody></table></div>
          </section>

          <section className="research-grid" aria-labelledby="runtime-title">
            <div className="research-copy"><span className="chapter">02 · BROWSER RUNTIME</span><h3 id="runtime-title">Separate numerical graphs from product infrastructure.</h3><p>The reusable platform is deliberately tensor-neutral. Model-specific schedules and coordinate conventions stay above it; viewers consume a canonical Gaussian scene below it.</p></div>
            <ol className="stack-diagram">
              <li><span>05</span><div><b>Application adapter</b><p>input UX · progress · cancellation · downloads · lifecycle</p></div></li>
              <li><span>04</span><div><b>Model orchestrator</b><p>TripoSplat preprocessing · CFG/Euler · octree · activation</p></div></li>
              <li><span>03</span><div><b>Verified artifact layer</b><p>manifest · SHA-256 · byte length · OPFS / Cache API</p></div></li>
              <li><span>02</span><div><b>Tensor RPC + session runtime</b><p>module worker · transferable buffers · graph IDs · disposal</p></div></li>
              <li><span>01</span><div><b>Execution + scene boundaries</b><p>ONNX Runtime Web / WebGPU · canonical GaussianScene</p></div></li>
            </ol>
          </section>

          <section className="research-grid" aria-labelledby="constraints-title">
            <div className="research-copy"><span className="chapter">03 · PORTING PRESSURE</span><h3 id="constraints-title">WebGPU changes the feasible lowering.</h3><p>A graph that exports is not necessarily a graph that fits, runs, or matches. Browser kernels, fixed shapes, readbacks, worker copies, and opaque GPU memory all influence the partition.</p></div>
            <div className="constraint-grid">
              <div><span>Memory</span><b>Attention dominates</b><p>A naïve full score matrix can reach gigabytes. Query chunking or a controlled reduction is a correctness requirement, not merely tuning.</p></div>
              <div><span>Control flow</span><b>Keep dynamic loops on host</b><p>Sampling and adaptive octree traversal remain explicit TypeScript where variable iteration and compaction are easier to inspect.</p></div>
              <div><span>Numerics</span><b>Validate boundaries</b><p>Execution providers may change reduction order, fusion, precision, or unsupported-op fallback. End-to-end resemblance alone cannot locate drift.</p></div>
              <div><span>Delivery</span><b>Weights are infrastructure</b><p>The manifest names every graph and external-data shard. A model CDN needs CORS, byte ranges, immutable revisions, and correct ONNX sidecar paths.</p></div>
            </div>
          </section>

          <section className="status-panel" aria-labelledby="status-title">
            <div><span className="chapter">STATUS · 2026-07-18</span><h3 id="status-title">Demonstrated is not the same as fully qualified.</h3><p>The prepared-image browser path completes all five stages and returns 262,144 finite Gaussians with valid PLY and <code>.splat</code> exports on the recorded Chrome / Apple M3 Max environment. DINO, VAE, one DiT call, all octree frontiers, and the raw Gaussian boundary have passing checks.</p></div>
            <ul><li className="pass"><b>Structural path</b><span>passes on the recorded system</span></li><li className="mixed"><b>4-step flow</b><span>qualification pass; stricter diagnostic fail</span></li><li className="fail"><b>20-step flow</b><span>executes, but remains numerically unqualified</span></li><li className="open"><b>Whole scene</b><span>official rendered-pixel parity remains open</span></li></ul>
            <a href="https://github.com/ai3d-dev/TripoSplatWebGPU/blob/main/docs/current-status.md" target="_blank" rel="noreferrer">Read the live release-gate ledger <ArrowIcon /></a>
          </section>
        </div>
      </article>

      <section id="platform" className="platform-section">
        <div className="learn-shell">
          <header className="platform-heading"><div><span className="level-number">GENERALIZED PLATFORM</span><Eyebrow>Beyond one model</Eyebrow><h2>Which AI pipelines can move to WebGPU?</h2></div><p>Portability is not a property of a model name. It is a property of its operators, tensor sizes, control flow, numerical tolerance, artifact layout, and target device. The TripoSplatWebGPU stack is a pattern for an entire class of client-side inference systems.</p></header>

          <div className="family-grid">
            <article><span>01</span><h3>Feed-forward reconstructors</h3><p>Image → point cloud, Gaussian set, depth, normal, or compact scene in one bounded pass.</p><b>Usually the cleanest fit</b></article>
            <article><span>02</span><h3>Latent diffusion + flow</h3><p>Iterative image, video, audio, or 3D generators with graph calls orchestrated by a host sampler.</p><b>Fit depends on attention + steps</b></article>
            <article><span>03</span><h3>Sparse spatial decoders</h3><p>Octrees, occupancy fields, point transformers, triplanes, and voxel refiners with adaptive host logic.</p><b>Good with careful partitioning</b></article>
            <article><span>04</span><h3>Classical vision pipelines</h3><p>Segmentation, embeddings, pose, depth, matting, restoration, and feature extraction.</p><b>Often directly portable</b></article>
            <article><span>05</span><h3>Neural rendering systems</h3><p>Gaussian, point, or compact neural-field decoders whose outputs feed a browser renderer.</p><b>Separate inference from rendering</b></article>
            <article><span>06</span><h3>Mesh generators</h3><p>Models producing vertices, faces, implicit fields, or oriented primitives, followed by host meshing/export.</p><b>Possible; topology is the challenge</b></article>
          </div>

          <section className="portability-ladder" aria-labelledby="ladder-title">
            <div className="ladder-heading"><span className="chapter">A PRACTICAL PORTABILITY SIEVE</span><h3 id="ladder-title">Start with the graph, not the hype.</h3></div>
            <div className="ladder-row easy"><div><span>A</span><b>Strong candidate</b></div><p>Inference-only; bounded tensors; ONNX-supported operators; moderate memory; browser-safe preprocessing; deterministic output contract.</p><small>Encoders · segmentation · feed-forward regressors · compact decoders</small></div>
            <div className="ladder-row adapt"><div><span>B</span><b>Candidate with partitioning</b></div><p>Iterative samplers, dynamic sparse structures, several graph stages, large external weights, or limited dynamic shapes.</p><small>Diffusion / flow · octrees · multi-stage 3D · multi-model apps</small></div>
            <div className="ladder-row hard"><div><span>C</span><b>Needs re-architecture</b></div><p>Custom CUDA ops, unfused giant attention, Python data-dependent control flow, unsupported sparse kernels, or excessive CPU↔GPU readback.</p><small>Rewrite kernels · chunk tensors · move loops to host · change representation</small></div>
            <div className="ladder-row poor"><div><span>D</span><b>Poor browser target</b></div><p>Multi-GPU training, workloads exceeding client storage/memory, secret server-side weights, or latency dominated by unavoidable huge transfers.</p><small>Keep server-side, distill, quantize, or design a smaller client model</small></div>
          </section>

          <section className="recipe-section" aria-labelledby="recipe-title">
            <div className="lesson-intro"><span className="chapter">THE REPEATABLE RECIPE</span><h3 id="recipe-title">A model-to-browser port in eight checks.</h3></div>
            <ol className="recipe-grid">
              <li><span>01</span><b>Freeze the oracle</b><p>Pin source revision, weights, inputs, seeds, and reference tensors.</p></li>
              <li><span>02</span><b>Map the executable</b><p>Inventory operators, shapes, loops, custom kernels, and peak intermediates.</p></li>
              <li><span>03</span><b>Choose graph islands</b><p>Export stable tensor functions; keep dynamic orchestration explicit.</p></li>
              <li><span>04</span><b>Design contracts</b><p>Name tensors, dtypes, dimensions, ownership, coordinates, and randomness.</p></li>
              <li><span>05</span><b>Package artifacts</b><p>Manifest every graph and sidecar with size, digest, revision, and capabilities.</p></li>
              <li><span>06</span><b>Validate vertically</b><p>Compare preprocessing, graph boundaries, loop states, decoded attributes, and renders.</p></li>
              <li><span>07</span><b>Engineer lifecycle</b><p>Stage sessions; cache safely; transfer buffers; support abort, retry, and dispose.</p></li>
              <li><span>08</span><b>Qualify real devices</b><p>Record browser, adapter, memory behavior, timing, accuracy, and failure envelopes.</p></li>
            </ol>
          </section>
        </div>
      </section>

      <section id="sources" className="sources-section learn-shell" aria-labelledby="sources-title">
        <div><Eyebrow>Continue learning</Eyebrow><h2 id="sources-title">Sources, evidence, and runnable code.</h2><p>This guide distinguishes the official research implementation from the browser port. The source repository remains the numerical authority; the WebGPU status ledger records what the port can and cannot currently demonstrate.</p></div>
        <div className="source-links">
          <a href="https://github.com/VAST-AI-Research/TripoSplat" target="_blank" rel="noreferrer"><span>Official implementation</span><b>VAST-AI-Research / TripoSplat</b><ArrowIcon /></a>
          <a href="https://www.tripo3d.ai/research/triposplat" target="_blank" rel="noreferrer"><span>Research overview</span><b>Generative 3D Gaussians with learned density control</b><ArrowIcon /></a>
          <a href="https://github.com/ai3d-dev/TripoSplatWebGPU/blob/main/docs/architecture-audit.md" target="_blank" rel="noreferrer"><span>Architecture</span><b>Graph contracts and browser boundaries</b><ArrowIcon /></a>
          <a href="https://github.com/ai3d-dev/TripoSplatWebGPU/blob/main/docs/current-status.md" target="_blank" rel="noreferrer"><span>Validation</span><b>Current release-gate ledger</b><ArrowIcon /></a>
          <a href="https://github.com/ai3d-dev/TripoSplatWebGPU/blob/main/docs/model-hosting.md" target="_blank" rel="noreferrer"><span>Deployment</span><b>Model hosting, CORS, and caching</b><ArrowIcon /></a>
          <a href="https://github.com/ai3d-dev/TripoSplatWebGPU" target="_blank" rel="noreferrer"><span>Browser port</span><b>Source, labs, fixtures, and benchmarks</b><ArrowIcon /></a>
        </div>
      </section>

      <section className="learn-cta"><div className="learn-shell"><div><span>READY TO SEE IT RUN?</span><h2>Turn one image into a local 3D Gaussian scene.</h2></div><a href="/e2e-web">Open the WebGPU runner <ArrowIcon /></a></div></section>
    </main>
    <footer className="learn-footer"><div className="learn-shell"><span>TRIPOSPLAT LEARN · ENGINEERING PREVIEW</span><nav aria-label="Footer"><a href="#generalist">Generalist</a><a href="#undergrad">Math</a><a href="#researcher">Research</a><a href="#platform">Platform</a><a href="https://github.com/ai3d-dev/TripoSplatWebGPU">GitHub</a></nav></div></footer>
  </>
}

const root = document.querySelector<HTMLDivElement>('#learn-root')
if (!root) throw new Error('Learning subsite root was not found.')
createRoot(root).render(<StrictMode><App /></StrictMode>)
