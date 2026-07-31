import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { TripoSplatRuntime } from "../src/inference-runtime.mjs";

test("runtime keeps the Python process resident and returns the interoperable ZIP envelope", async () => {
  process.env.MUTUALGPU_PROVIDER_KEY = "must-not-reach-python";
  process.env.MUTUALGPU_PROVIDER_KEY_FILE = "/must-not-reach-python";
  const runtime = new TripoSplatRuntime({ python: "python3", scriptPath: fileURLToPath(new URL("./fake-inference.py", import.meta.url)), backend: "cuda", modelDir: "/models", modelManifest: "/models/manifest.json", sdkVersion: "fake", startupTimeoutMs: 5_000, generationTimeoutMs: 5_000 });
  try {
    assert.deepEqual(await runtime.start(), { device: "cuda", backend: "cuda", modelRevision: "fake-revision", diagnostics: {} });
    const progress = [];
    const result = await runtime.generate({ inputPath: "/private/input.png", numGaussians: 32768, steps: 4, guidanceScale: 3, seed: 42 }, { onProgress: update => progress.push(update) });
    assert.equal(result.resultZip.subarray(0, 2).toString(), "PK");
    assert.equal(result.metadata.format, "triposplat-webgpu-result");
    assert.equal(progress[0].percent, 50);
  } finally { runtime.close(); delete process.env.MUTUALGPU_PROVIDER_KEY; delete process.env.MUTUALGPU_PROVIDER_KEY_FILE; }
});

