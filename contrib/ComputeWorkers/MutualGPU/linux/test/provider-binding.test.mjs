import assert from "node:assert/strict";
import test from "node:test";
import { createTaskHandler } from "../src/provider-binding.mjs";
import { InputDownloadError } from "../src/input-download.mjs";

test("binding rejects invalid browser-compatible scalars before accepting", async () => {
  const events = [];
  const handler = createTaskHandler({ runtime: { generate: async () => assert.fail("must not run") } });
  await handler(task({ num_inference_steps: "8" }, events));
  assert.deepEqual(events, [["reject", "Invalid TripoSplat inputs: num_inference_steps must be 4 or 20."]]);
});

test("binding accepts, uses a one-time input refresh, and publishes only result ZIP metadata", async () => {
  const events = [];
  let calls = 0;
  const runtime = { generate: async (request, { onProgress }) => {
    assert.equal(request.inputPath, "/private/input.png");
    await onProgress({ phase: "sampling", percent: 50, message: "step" });
    return { resultZip: Uint8Array.of(0x50, 0x4b), metadata: { format: "triposplat-webgpu-result", version: 1, count: 262144, elapsedMs: 1, sdkVersion: "fake" }, logs: "done\n" };
  } };
  const handler = createTaskHandler({ runtime, inputDownload: async input => {
    calls += 1;
    if (calls === 1) throw new InputDownloadError("expired", "expired");
    assert.equal(input.url, "https://refreshed.example/input");
    return "/private/input.png";
  }, logger: silentLogger });
  const assigned = task({}, events);
  assigned.refreshInputDownload = async () => "https://refreshed.example/input";
  await handler(assigned);
  assert.equal(calls, 2);
  assert.equal(events[0][0], "accept");
  assert.deepEqual(events.find(event => event[0] === "upload")[1], { resultZip: Uint8Array.of(0x50, 0x4b), metadata: { format: "triposplat-webgpu-result", version: 1, count: 262144, elapsedMs: 1, sdkVersion: "fake" }, logs: "done\n" });
  assert.deepEqual(events.at(-1), ["complete", "receipt-1"]);
});

function task(scalars, events) {
  return {
    taskId: "task-1", scalars, input: { url: "https://input.example/image", contentType: "image/png", length: 12, sha256: "a".repeat(64) }, signal: new AbortController().signal,
    accept: async () => events.push(["accept"]), reject: async reason => events.push(["reject", reason]), reportProgress: async update => { events.push(["progress", update]); return true; },
    refreshInputDownload: async () => "https://refreshed.example/input", uploadResult: async result => { events.push(["upload", result]); return { receipt: "receipt-1", ignoredParts: [] }; }, complete: async receipt => events.push(["complete", receipt]), fail: async (step, reason) => events.push(["fail", step, reason])
  };
}
const silentLogger = { info() {}, warn() {}, error() {} };

