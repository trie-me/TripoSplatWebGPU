import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AssignmentValidationError, parseGenerationRequest } from "./config.mjs";
import { downloadVerifiedInput, InputDownloadError, validateDescriptor } from "./input-download.mjs";

export function createTaskHandler({ runtime, logger = console, chooseSeed, activity = {}, progressRetryMs = 1_050, onRuntimeReset = async () => {}, inputDownload = downloadVerifiedInput, fetchImpl }) {
  if (!runtime || typeof runtime.generate !== "function") throw new TypeError("A TripoSplat runtime is required.");
  return async task => {
    const progress = createProgressReporter({ task, logger, activity, retryMs: progressRetryMs });
    activity.state = "validating";
    activity.taskId = task.taskId;
    let request;
    try {
      request = parseGenerationRequest(task.scalars, chooseSeed);
      if (!task.input) throw new AssignmentValidationError("image_url is required");
      validateDescriptor(task.input);
    } catch (error) {
      if (!(error instanceof AssignmentValidationError) && !(error instanceof InputDownloadError)) throw error;
      await task.reject(`Invalid TripoSplat inputs: ${error.message}.`);
      idle(activity);
      logger.warn?.(`[task ${task.taskId}] assignment rejected: invalid inputs.`);
      return;
    }
    await task.accept();
    await progress.flush({ phase: "input", percent: 1, message: "Downloading and verifying the input image." });
    const inputDirectory = await mkdtemp(join(tmpdir(), "mutualgpu-triposplat-input-"));
    let resetRequired = false;
    try {
      let inputPath;
      try {
        inputPath = await inputDownload(task.input, inputDirectory, { signal: task.signal, fetchImpl });
      } catch (error) {
        if (error instanceof InputDownloadError && error.category === "expired") {
          const refreshedUrl = await task.refreshInputDownload();
          inputPath = await inputDownload({ ...task.input, url: refreshedUrl }, inputDirectory, { signal: task.signal, fetchImpl });
        } else throw error;
      }
      activity.state = "inference";
      const generated = await runtime.generate({ ...request, inputPath }, { onProgress: update => progress.push(update), signal: task.signal });
      await progress.flush({ phase: "upload", percent: 99, message: "Uploading the verified TripoSplat result." });
      activity.state = "uploading";
      const published = await task.uploadResult({ resultZip: generated.resultZip, metadata: generated.metadata, logs: generated.logs });
      await task.complete(published.receipt);
      activity.completed = (activity.completed ?? 0) + 1;
      logger.info?.(`Completed TripoSplat task ${task.taskId}.`);
    } catch (error) {
      resetRequired = error?.name === "RuntimeRestartRequired";
      if (task.signal?.aborted) {
        logger.info?.(`[task ${task.taskId}] cancelled.`);
      } else {
        const mapped = failureFor(error);
        resetRequired ||= mapped.resetRuntime;
        logger.error?.(`TripoSplat task ${task.taskId} failed during ${mapped.step}: ${safeLocalError(error)}`);
        try { await task.fail(mapped.step, mapped.reason); } catch { /* cancellation/rebind owns terminal state */ }
        activity.failed = (activity.failed ?? 0) + 1;
      }
    } finally {
      await rm(inputDirectory, { recursive: true, force: true });
      await progress.close();
      idle(activity);
    }
    if (resetRequired) await onRuntimeReset();
  };
}

function failureFor(error) {
  if (error instanceof InputDownloadError) return { step: "input", reason: "The provider could not verify the input image.", resetRuntime: false };
  if (error?.name === "AbortError") return { step: "cancelled", reason: "The task was cancelled.", resetRuntime: true };
  if (error?.name === "RuntimeRestartRequired") return { step: "inference", reason: "TripoSplat inference stopped before producing a result.", resetRuntime: true };
  if (error?.name === "ProviderClientError" && error?.code === "upload_outcome_unknown") return { step: "upload", reason: "The result upload outcome is unknown and requires reconciliation.", resetRuntime: false };
  return { step: "inference", reason: "TripoSplat inference failed on the provider.", resetRuntime: false };
}

function createProgressReporter({ task, logger, activity, retryMs }) {
  let latest = null; let inFlight = null; let timer = null; let closed = false;
  const queue = update => { if (!closed) { latest = update; activity.lastEventAt = new Date(); logger.info?.(`[task ${task.taskId}] ${update.phase}: ${Number(update.percent).toFixed(1)}% — ${update.message}`); } };
  const schedule = () => { if (!closed && !timer && latest) { timer = setTimeout(() => { timer = null; void send(); }, retryMs); timer.unref?.(); } };
  const send = async () => {
    if (closed || !latest) return true;
    if (inFlight) return inFlight;
    const update = latest;
    inFlight = (async () => {
      try { const delivered = await task.reportProgress(update); if (delivered === false) { schedule(); return false; } if (latest === update) latest = null; return true; }
      catch (error) { if (latest === update) latest = null; logger.warn?.(`Progress update was not delivered: ${safeLocalError(error)}`); return false; }
      finally { inFlight = null; if (!closed && latest && latest !== update) void send(); }
    })();
    return inFlight;
  };
  return {
    async push(update) { queue(update); await send(); },
    async flush(update) { if (update) queue(update); if (timer) { clearTimeout(timer); timer = null; } while (!closed && latest) { const delivered = await send(); if (!delivered && latest) await new Promise(resolve => setTimeout(resolve, retryMs)); } },
    async close() { closed = true; latest = null; if (timer) clearTimeout(timer); if (inFlight) await inFlight; }
  };
}

function idle(activity) { activity.state = "idle"; activity.taskId = null; activity.lastEventAt = new Date(); }
function safeLocalError(error) { return typeof error?.name === "string" ? error.name : "Error"; }

