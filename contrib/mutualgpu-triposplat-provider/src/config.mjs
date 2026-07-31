import { randomInt } from "node:crypto";
import { readFile, stat } from "node:fs/promises";

const MEDIA_TYPES = Object.freeze(["image/png", "image/jpeg", "image/webp"]);
const MACHINE_TIERS = new Set(["Small", "Medium", "Large", "ExtraLarge"]);

export class AssignmentValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = "AssignmentValidationError";
  }
}

export function loadConfig(environment = process.env) {
  const machineTier = tier(environment.MUTUALGPU_TRIPOSPLAT_MACHINE_TIER ?? "Large", "MUTUALGPU_TRIPOSPLAT_MACHINE_TIER");
  const computeTier = tier(environment.MUTUALGPU_TRIPOSPLAT_COMPUTE_TIER ?? machineTier, "MUTUALGPU_TRIPOSPLAT_COMPUTE_TIER");
  const backend = oneOf(environment.MUTUALGPU_TRIPOSPLAT_BACKEND ?? "auto", "MUTUALGPU_TRIPOSPLAT_BACKEND", ["auto", "cuda", "rocm"]);
  return Object.freeze({
    apiUrl: requiredUrl(environment.MUTUALGPU_API_URL ?? "https://mutualgpu.com", "MUTUALGPU_API_URL"),
    providerKeyFile: required(environment, "MUTUALGPU_PROVIDER_KEY_FILE"),
    backend,
    machineTier,
    computeTier,
    memoryGiB: integer(environment.MUTUALGPU_TRIPOSPLAT_MEMORY_GIB ?? "32", "MUTUALGPU_TRIPOSPLAT_MEMORY_GIB", 1, 1024),
    python: environment.MUTUALGPU_TRIPOSPLAT_PYTHON ?? "python3",
    modelDir: required(environment, "MUTUALGPU_TRIPOSPLAT_MODEL_DIR"),
    modelManifest: required(environment, "MUTUALGPU_TRIPOSPLAT_MODEL_MANIFEST"),
    heartbeatMs: integer(environment.MUTUALGPU_TRIPOSPLAT_HEARTBEAT_SECONDS ?? "15", "MUTUALGPU_TRIPOSPLAT_HEARTBEAT_SECONDS", 1, 3600) * 1_000,
    startupTimeoutMs: integer(environment.MUTUALGPU_TRIPOSPLAT_STARTUP_TIMEOUT_SECONDS ?? "1800", "MUTUALGPU_TRIPOSPLAT_STARTUP_TIMEOUT_SECONDS", 60, 7200) * 1_000,
    generationTimeoutMs: integer(environment.MUTUALGPU_TRIPOSPLAT_TIMEOUT_SECONDS ?? "1800", "MUTUALGPU_TRIPOSPLAT_TIMEOUT_SECONDS", 60, 7200) * 1_000,
    sdkVersion: environment.MUTUALGPU_TRIPOSPLAT_SDK_VERSION ?? "0.1.0-preview.4",
    collectVulkanDiagnostics: boolean(environment.MUTUALGPU_TRIPOSPLAT_VULKAN_DIAGNOSTICS ?? "true", "MUTUALGPU_TRIPOSPLAT_VULKAN_DIAGNOSTICS")
  });
}

/** Reads the opaque provider credential only in the Node process, never the child. */
export async function readProviderKey(file) {
  const metadata = await stat(file);
  if (!metadata.isFile()) throw new TypeError("MUTUALGPU_PROVIDER_KEY_FILE must name a regular file.");
  if ((metadata.mode & 0o077) !== 0) throw new TypeError("MUTUALGPU_PROVIDER_KEY_FILE must not be readable by group or other users (chmod 600).");
  const key = (await readFile(file, "utf8")).trim();
  if (!key || /\s/.test(key)) throw new TypeError("MUTUALGPU_PROVIDER_KEY_FILE must contain one non-whitespace provider key.");
  return key;
}

export function buildEnrollment(config) {
  return {
    machine: {
      tier: config.machineTier,
      specifications: { computeTier: config.computeTier, memoryGiB: config.memoryGiB }
    },
    capabilities: [{
      name: "tripo-splat",
      description: "Convert one image into an interoperable TripoSplat Gaussian-splat result.",
      inputs: [
        { key: "image_url", type: "Image", required: true, label: "Image", description: "Input image to convert into a 3D Gaussian splat.", contentTypes: MEDIA_TYPES, displayOrder: 0 },
        { key: "num_gaussians", type: "Integer", required: false, label: "Number of Gaussians", description: "Target Gaussian count; values are rounded by the official pipeline to a multiple of 32.", default: "262144", minimum: 32768, maximum: 262144, displayOrder: 1 },
        { key: "num_inference_steps", type: "Integer", required: false, label: "Inference steps", description: "The supported production schedules are 4 and 20 steps.", default: "20", allowedValues: ["4", "20"], displayOrder: 2 },
        { key: "guidance_scale", type: "Number", required: false, label: "Guidance scale", description: "Classifier-free guidance strength.", default: "3", minimum: 0, maximum: 20, displayOrder: 3 },
        { key: "output_format", type: "String", required: false, label: "Output format", description: "The requested primary Gaussian-splat format; both PLY and SPLAT are included for compatibility.", default: "ply", allowedValues: ["ply", "splat"], displayOrder: 4 },
        { key: "seed", type: "Integer", required: false, label: "Seed", description: "Optional unsigned 32-bit seed. A random seed is selected when omitted.", minimum: 0, maximum: 4294967295, displayOrder: 5 },
        { key: "enable_safety_checker", type: "Boolean", required: false, label: "Enable safety checker", description: "No qualified native safety checker is bundled; false is required.", default: "false", displayOrder: 6 }
      ],
      output: { hasMetadata: true }
    }]
  };
}

export function parseGenerationRequest(scalars = {}, chooseSeed = () => randomInt(0, 0x1_0000_0000)) {
  const requestedGaussians = scalarInteger(scalars.num_gaussians ?? "262144", "num_gaussians", 32768, 262144);
  const numGaussians = Math.round(requestedGaussians / 32) * 32;
  const steps = scalarInteger(scalars.num_inference_steps ?? "20", "num_inference_steps", 1, 100);
  if (steps !== 4 && steps !== 20) throw new AssignmentValidationError("num_inference_steps must be 4 or 20");
  const guidanceScale = scalarNumber(scalars.guidance_scale ?? "3", "guidance_scale", 0, 20);
  const outputFormat = oneOf(scalars.output_format ?? "ply", "output_format", ["ply", "splat"], AssignmentValidationError);
  const seed = scalars.seed == null || scalars.seed === "" ? chooseSeed() : scalarInteger(scalars.seed, "seed", 0, 0xffff_ffff);
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffff_ffff) throw new AssignmentValidationError("seed must be an unsigned 32-bit integer");
  const safety = scalarBoolean(scalars.enable_safety_checker ?? "false", "enable_safety_checker");
  if (safety) throw new AssignmentValidationError("enable_safety_checker=true is not supported by this provider");
  return Object.freeze({ numGaussians, requestedGaussians, steps, guidanceScale, outputFormat, seed, enableSafetyChecker: false });
}

export const acceptedImageContentTypes = MEDIA_TYPES;

function required(environment, name) {
  const value = environment[name];
  if (typeof value !== "string" || value.trim() === "") throw new TypeError(`${name} is required.`);
  return value;
}

function requiredUrl(value, name) {
  let url;
  try { url = new URL(value); } catch { throw new TypeError(`${name} must be a URL.`); }
  if (url.protocol !== "https:") throw new TypeError(`${name} must use https.`);
  return url;
}

function tier(value, name) { return oneOf(value, name, MACHINE_TIERS); }
function oneOf(value, name, allowed, ErrorType = TypeError) {
  if (!allowed.includes?.(value) && !allowed.has?.(value)) throw new ErrorType(`${name} must be one of: ${[...allowed].join(", ")}.`);
  return value;
}
function integer(value, name, minimum, maximum) {
  if (!/^\d+$/.test(String(value))) throw new TypeError(`${name} must be a whole number.`);
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) throw new TypeError(`${name} must be between ${minimum} and ${maximum}.`);
  return parsed;
}
function scalarInteger(value, name, minimum, maximum) {
  if (!/^\d+$/.test(String(value))) throw new AssignmentValidationError(`${name} must be an integer`);
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) throw new AssignmentValidationError(`${name} must be between ${minimum} and ${maximum}`);
  return parsed;
}
function scalarNumber(value, name, minimum, maximum) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < minimum || parsed > maximum) throw new AssignmentValidationError(`${name} must be between ${minimum} and ${maximum}`);
  return parsed;
}
function scalarBoolean(value, name) {
  if (value === "true") return true;
  if (value === "false") return false;
  throw new AssignmentValidationError(`${name} must be true or false`);
}
function boolean(value, name) {
  if (value === "true") return true;
  if (value === "false") return false;
  throw new TypeError(`${name} must be true or false.`);
}

