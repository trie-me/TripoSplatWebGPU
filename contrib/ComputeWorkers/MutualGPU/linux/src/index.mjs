import { fileURLToPath } from "node:url";
import { ProviderClient } from "@mutualgpu/provider-core";
import { NodeGrpcTransport } from "@mutualgpu/provider-node";
import { buildEnrollment, loadConfig, readProviderKey } from "./config.mjs";
import { TripoSplatRuntime } from "./inference-runtime.mjs";
import { createTaskHandler } from "./provider-binding.mjs";

const config = loadConfig();
const providerKey = await readProviderKey(config.providerKeyFile);
const activity = { state: "starting", taskId: null, completed: 0, failed: 0, lastEventAt: new Date() };
const runtime = new TripoSplatRuntime({
  python: config.python,
  scriptPath: fileURLToPath(new URL("./inference.py", import.meta.url)),
  backend: config.backend,
  modelDir: config.modelDir,
  modelManifest: config.modelManifest,
  sdkVersion: config.sdkVersion,
  collectVulkanDiagnostics: config.collectVulkanDiagnostics,
  startupTimeoutMs: config.startupTimeoutMs,
  generationTimeoutMs: config.generationTimeoutMs,
  log: message => process.stderr.write(message)
});
const provider = new ProviderClient(new NodeGrpcTransport(config.apiUrl, providerKey), {
  onDiagnostic: event => {
    // Connection state is useful to operators, but task IDs and server responses
    // can be sensitive operational data and are intentionally not rendered.
    if (["disconnect_observed", "recovery_expired"].includes(event.event)) console.warn(`[provider] ${event.event}.`);
  }
});

let stopping = false;
let heartbeat = null;
let recovery = null;
let handler;
const stop = signal => {
  if (stopping) return;
  stopping = true;
  if (heartbeat) clearInterval(heartbeat);
  console.info(`Stopping foreground TripoSplat worker after ${signal}.`);
  provider.close();
  runtime.close();
};
process.once("SIGINT", () => stop("SIGINT"));
process.once("SIGTERM", () => stop("SIGTERM"));

const scheduleRuntimeRecovery = () => {
  if (stopping || recovery) return;
  recovery = (async () => {
    // Let ProviderClient finish and detach the terminal assignment before closing
    // its transport. A new session is only advertised after Python is warm again.
    await new Promise(resolve => setTimeout(resolve, 0));
    if (stopping) return;
    console.warn("[recovery] restarting the GPU subprocess and provider session before accepting another task.");
    provider.close();
    await runtime.restart();
    await provider.enroll(buildEnrollment(config));
    await provider.connect(handler);
    activity.state = "idle";
    console.info("[recovery] warm TripoSplat provider session restored.");
  })().catch(error => {
    console.error(`[recovery] failed (${safeErrorCategory(error)}). Rerun the foreground command after correcting the problem.`);
    process.exitCode = 1;
    stop("recovery failure");
  }).finally(() => { recovery = null; });
};

handler = createTaskHandler({ runtime, activity, onRuntimeReset: scheduleRuntimeRecovery });
let startupStage = "TripoSplat model warmup";
try {
  console.info("[1/4] Verifying model files and loading the official TripoSplat pipeline before enrollment.");
  const runtimeInfo = await runtime.start();
  startupStage = "MutualGPU enrollment";
  console.info(`[2/4] Runtime ready on ${runtimeInfo.backend}/${runtimeInfo.device}; enrolling tripo-splat.`);
  await provider.enroll(buildEnrollment(config));
  startupStage = "MutualGPU provider connection";
  console.info("[3/4] Enrollment accepted; opening the foreground provider session.");
  await provider.connect(handler);
  activity.state = "idle";
  console.info(`[4/4] Provider connected with model revision ${runtimeInfo.modelRevision}. Press Ctrl-C to stop.`);
  const connectedAt = Date.now();
  heartbeat = setInterval(() => {
    const task = activity.taskId ? " task=active" : "";
    console.info(`[heartbeat] state=${activity.state}${task} uptime=${Math.floor((Date.now() - connectedAt) / 1000)}s completed=${activity.completed} failed=${activity.failed}`);
  }, config.heartbeatMs);
  heartbeat.unref?.();
} catch (error) {
  runtime.close();
  provider.close();
  const category = safeErrorCategory(error);
  console.error(`TripoSplat worker failed during ${startupStage} (${category}).`);
  console.error(remediation(category));
  process.exitCode = 1;
}

function safeErrorCategory(error) {
  if (error?.name && error.name !== "Error") return error.name;
  const message = typeof error?.message === "string" ? error.message : "";
  const grpcStatus = /MutualGPU enrollment failed \((\d+)\)/.exec(message)?.[1];
  if (grpcStatus) return `GrpcStatus${grpcStatus}`;
  if (/startup timed out/i.test(message)) return "StartupTimeout";
  if (/certificate|\bTLS\b|\bSSL\b/i.test(message)) return "TlsError";
  return "Error";
}
function remediation(category) {
  switch (category) {
    case "GpuUnavailable": return "No compatible Linux CUDA or ROCm PyTorch runtime is available to this process.";
    case "ModelVerificationFailed": return "Run the model download command again; every model file must match the pinned manifest.";
    case "ModelLoadFailed": return "Check free VRAM, GPU driver support, and the selected backend's frozen environment.";
    case "UnsupportedPlatform": return "This worker intentionally supports Linux only.";
    case "GrpcStatus16": return "The provider credential was not accepted. Check the protected key file without printing its contents.";
    case "StartupTimeout": return "Pipeline warmup exceeded its allowance. Check model storage, GPU availability, and increase the startup timeout only if necessary.";
    default: return "Review the safe stage diagnostic above; provider credentials and requester data are not written to logs.";
  }
}

