import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { collectResult } from "./artifact-validation.mjs";

export class RuntimeRestartRequired extends Error {
  constructor(message) { super(message); this.name = "RuntimeRestartRequired"; }
}

/** A single persistent, secret-free Python process owns the GPU and model weights. */
export class TripoSplatRuntime {
  #options;
  #child = null;
  #ready = null;
  #pending = null;
  #nextId = 1;
  #closed = false;

  constructor(options) { this.#options = options; }
  get info() { return this.#ready?.info; }

  async start() {
    if (this.#child && this.#ready?.info) return this.#ready.info;
    this.#closed = false;
    const child = spawn(this.#options.python, ["-u", this.#options.scriptPath, "--serve"], {
      stdio: ["pipe", "pipe", "pipe"], env: runtimeEnvironment(this.#options)
    });
    this.#child = child;
    const ready = new Promise((resolve, reject) => { this.#ready = { resolve, reject, info: null }; });
    createInterface({ input: child.stdout }).on("line", line => this.#receiveLine(child, line));
    child.stderr.on("data", chunk => {
      const message = chunk.toString();
      if (message) (this.#options.log ?? (value => process.stderr.write(value)))(`[python] ${message}`);
    });
    child.once("error", error => this.#processEnded(child, restartError(error)));
    child.once("exit", (code, signal) => this.#processEnded(child, new RuntimeRestartRequired(`TripoSplat runtime exited (${signal ?? code ?? "unknown"}).`)));
    const timeout = setTimeout(() => {
      this.#processEnded(child, new RuntimeRestartRequired("TripoSplat runtime startup timed out."));
      child.kill("SIGTERM");
    }, this.#options.startupTimeoutMs ?? 1_800_000);
    timeout.unref?.();
    try { return await ready; } finally { clearTimeout(timeout); }
  }

  async generate(request, { onProgress = () => {}, signal } = {}) {
    if (this.#pending) throw new Error("TripoSplat already owns an active generation.");
    await this.start();
    if (signal?.aborted) throw abortError();
    const outputDirectory = await mkdtemp(join(tmpdir(), "mutualgpu-triposplat-"));
    const id = String(this.#nextId++);
    let abortListener;
    try {
      const result = new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          const error = new RuntimeRestartRequired("TripoSplat generation timed out.");
          reject(error);
          this.#child?.kill("SIGTERM");
        }, this.#options.generationTimeoutMs ?? 1_800_000);
        timeout.unref?.();
        this.#pending = { id, outputDirectory, onProgress, resolve, reject, timeout };
      });
      abortListener = () => {
        this.#pending?.reject(new RuntimeRestartRequired("TripoSplat generation was cancelled."));
        this.#child?.kill("SIGTERM");
      };
      signal?.addEventListener("abort", abortListener, { once: true });
      this.#child.stdin.write(`${JSON.stringify({ type: "generate", id, outputDirectory, ...request })}\n`);
      return await result;
    } finally {
      signal?.removeEventListener("abort", abortListener);
      if (this.#pending?.id === id) this.#clearPending();
      await rm(outputDirectory, { recursive: true, force: true });
    }
  }

  async restart() {
    this.close();
    return this.start();
  }

  close() {
    this.#closed = true;
    const child = this.#child;
    this.#child = null;
    this.#ready?.reject(new Error("TripoSplat runtime is closing."));
    this.#ready = null;
    this.#pending?.reject(new Error("TripoSplat runtime is closing."));
    this.#clearPending();
    child?.kill("SIGTERM");
  }

  #receiveLine(child, line) {
    if (child !== this.#child) return;
    let message;
    try { message = JSON.parse(line); } catch {
      this.#processEnded(child, new RuntimeRestartRequired("TripoSplat runtime emitted an invalid protocol message."));
      child.kill("SIGTERM");
      return;
    }
    if (message.type === "ready") {
      const info = Object.freeze({ device: message.device, backend: message.backend, modelRevision: message.modelRevision, diagnostics: message.diagnostics ?? {} });
      if (this.#ready) this.#ready.info = info;
      this.#ready?.resolve(info);
      return;
    }
    if (message.type === "startup_error") {
      const error = new Error("TripoSplat runtime preflight failed.");
      error.name = message.category ?? "TripoSplatStartupError";
      this.#processEnded(child, error);
      child.kill("SIGTERM");
      return;
    }
    const pending = this.#pending;
    if (!pending || message.id !== pending.id) return;
    if (message.type === "progress") {
      void Promise.resolve(pending.onProgress({
        phase: message.phase ?? "inference",
        percent: Math.max(1, Math.min(98, Number(message.percent) || 1)),
        message: message.message ?? "Running TripoSplat inference."
      })).catch(() => {});
      return;
    }
    if (message.type === "error") {
      const error = new Error("TripoSplat inference failed.");
      error.name = message.category ?? "TripoSplatError";
      pending.reject(error);
      this.#clearPending();
      return;
    }
    if (message.type === "result") {
      void collectResult(pending.outputDirectory).then(pending.resolve, pending.reject).finally(() => this.#clearPending());
    }
  }

  #processEnded(child, error) {
    if (child !== this.#child) return;
    this.#child = null;
    this.#ready?.reject(error);
    this.#ready = null;
    this.#pending?.reject(error);
    this.#clearPending();
  }

  #clearPending() {
    if (!this.#pending) return;
    clearTimeout(this.#pending.timeout);
    this.#pending = null;
  }
}

function runtimeEnvironment(options) {
  const environment = { ...process.env };
  for (const name of Object.keys(environment)) if (/^MUTUALGPU_.*(?:KEY|PASSWORD|SECRET|TOKEN)(?:_|$)/i.test(name)) delete environment[name];
  return {
    ...environment,
    MUTUALGPU_TRIPOSPLAT_BACKEND: options.backend,
    MUTUALGPU_TRIPOSPLAT_MODEL_DIR: options.modelDir,
    MUTUALGPU_TRIPOSPLAT_MODEL_MANIFEST: options.modelManifest,
    MUTUALGPU_TRIPOSPLAT_SDK_VERSION: options.sdkVersion,
    MUTUALGPU_TRIPOSPLAT_VULKAN_DIAGNOSTICS: String(options.collectVulkanDiagnostics)
  };
}
function restartError(error) { return error instanceof RuntimeRestartRequired ? error : new RuntimeRestartRequired("TripoSplat runtime could not start."); }
function abortError() { const error = new Error("TripoSplat generation was cancelled."); error.name = "AbortError"; return error; }

