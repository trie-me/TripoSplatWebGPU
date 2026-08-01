import assert from "node:assert/strict";
import { chmod, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { buildEnrollment, loadConfig, parseGenerationRequest, readProviderKey } from "../src/config.mjs";

test("exact tripo-splat enrollment contract remains stable", () => {
  const config = loadConfig(environment());
  const enrollment = buildEnrollment(config);
  assert.equal(enrollment.capabilities[0].name, "tripo-splat");
  assert.deepEqual(enrollment.capabilities[0].inputs.map(input => input.key), ["image_url", "num_gaussians", "num_inference_steps", "guidance_scale", "output_format", "seed", "enable_safety_checker"]);
  assert.equal(enrollment.capabilities[0].inputs[0].type, "Image");
  assert.deepEqual(enrollment.capabilities[0].inputs[0].contentTypes, ["image/png", "image/jpeg", "image/webp"]);
  assert.deepEqual(enrollment.capabilities[0].inputs[2].allowedValues, undefined);
  assert.equal(enrollment.capabilities[0].inputs[2].minimum, 4);
  assert.equal(enrollment.capabilities[0].inputs[2].maximum, 20);
  assert.deepEqual(enrollment.capabilities[0].output, { hasMetadata: true });
});

test("request scalars use browser-compatible defaults and reject unsupported variants", () => {
  assert.deepEqual(parseGenerationRequest({}, () => 42), { numGaussians: 262144, requestedGaussians: 262144, steps: 20, guidanceScale: 3, outputFormat: "ply", seed: 42, enableSafetyChecker: false });
  assert.equal(parseGenerationRequest({ num_gaussians: "32769" }).numGaussians, 32768);
  assert.equal(parseGenerationRequest({ num_inference_steps: "4", output_format: "splat", seed: "4294967295" }).seed, 4294967295);
  assert.throws(() => parseGenerationRequest({ num_inference_steps: "5" }), /4 or 20/);
  assert.throws(() => parseGenerationRequest({ enable_safety_checker: "true" }), /not supported/);
});

test("provider credentials must come from a private one-line file", async () => {
  const directory = await mkdtemp(join(tmpdir(), "triposplat-key-"));
  const keyFile = join(directory, "provider.key");
  try {
    await writeFile(keyFile, "opaque-key\n", { mode: 0o600 });
    await chmod(keyFile, 0o600);
    assert.equal(await readProviderKey(keyFile), "opaque-key");
    await chmod(keyFile, 0o644);
    await assert.rejects(readProviderKey(keyFile), /chmod 600/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

function environment() {
  return {
    MUTUALGPU_PROVIDER_KEY_FILE: "/private/provider.key",
    MUTUALGPU_TRIPOSPLAT_MODEL_DIR: "/models",
    MUTUALGPU_TRIPOSPLAT_MODEL_MANIFEST: "/models/manifest.json"
  };
}
import assert from "node:assert/strict";
import { chmod, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { buildEnrollment, loadConfig, parseGenerationRequest, readProviderKey } from "../src/config.mjs";

test("exact tripo-splat enrollment contract remains stable", () => {
  const config = loadConfig(environment());
  const enrollment = buildEnrollment(config);
  assert.equal(enrollment.capabilities[0].name, "tripo-splat");
  assert.deepEqual(enrollment.capabilities[0].inputs.map(input => input.key), ["image_url", "num_gaussians", "num_inference_steps", "guidance_scale", "output_format", "seed", "enable_safety_checker"]);
  assert.equal(enrollment.capabilities[0].inputs[0].type, "Image");
  assert.deepEqual(enrollment.capabilities[0].inputs[0].contentTypes, ["image/png", "image/jpeg", "image/webp"]);
  assert.deepEqual(enrollment.capabilities[0].output, { hasMetadata: true });
});

test("request scalars use browser-compatible defaults and reject unsupported variants", () => {
  assert.deepEqual(parseGenerationRequest({}, () => 42), { numGaussians: 262144, requestedGaussians: 262144, steps: 20, guidanceScale: 3, outputFormat: "ply", seed: 42, enableSafetyChecker: false });
  assert.equal(parseGenerationRequest({ num_gaussians: "32769" }).numGaussians, 32768);
  assert.equal(parseGenerationRequest({ num_inference_steps: "4", output_format: "splat", seed: "4294967295" }).seed, 4294967295);
  assert.throws(() => parseGenerationRequest({ num_inference_steps: "5" }), /4 or 20/);
  assert.throws(() => parseGenerationRequest({ enable_safety_checker: "true" }), /not supported/);
});

test("provider credentials must come from a private one-line file", async () => {
  const directory = await mkdtemp(join(tmpdir(), "triposplat-key-"));
  const keyFile = join(directory, "provider.key");
  try {
    await writeFile(keyFile, "opaque-key\n", { mode: 0o600 });
    await chmod(keyFile, 0o600);
    assert.equal(await readProviderKey(keyFile), "opaque-key");
    await chmod(keyFile, 0o644);
    await assert.rejects(readProviderKey(keyFile), /chmod 600/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

function environment() {
  return {
    MUTUALGPU_PROVIDER_KEY_FILE: "/private/provider.key",
    MUTUALGPU_TRIPOSPLAT_MODEL_DIR: "/models",
    MUTUALGPU_TRIPOSPLAT_MODEL_MANIFEST: "/models/manifest.json"
  };
}

