import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { downloadVerifiedInput, InputDownloadError, validateDescriptor } from "../src/input-download.mjs";

test("input downloader streams an exact, digest-checked PNG to a private file", async () => {
  const bytes = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3, 4]);
  const directory = await mkdtemp(join(tmpdir(), "triposplat-input-"));
  const input = descriptor(bytes, "image/png");
  try {
    const path = await downloadVerifiedInput(input, directory, { fetchImpl: async () => new Response(bytes, { status: 200, headers: { "content-length": String(bytes.length) } }) });
    assert.deepEqual(await readFile(path), bytes);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("input downloader classifies expired signed URLs without exposing the URL", async () => {
  const bytes = Buffer.from([0xff, 0xd8, 0xff, 0]);
  const directory = await mkdtemp(join(tmpdir(), "triposplat-input-"));
  try {
    await assert.rejects(downloadVerifiedInput(descriptor(bytes, "image/jpeg"), directory, { fetchImpl: async () => new Response(null, { status: 403 }) }), error => error instanceof InputDownloadError && error.category === "expired");
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("input descriptor requires HTTPS, a bounded declared size, and SHA-256", () => {
  assert.throws(() => validateDescriptor({ url: "http://example.test/a", contentType: "image/png", length: 1, sha256: "a".repeat(64) }), /HTTPS/);
  assert.throws(() => validateDescriptor({ url: "https://example.test/a", contentType: "image/gif", length: 1, sha256: "a".repeat(64) }), /unsupported/);
});

function descriptor(bytes, contentType) {
  return { url: "https://input.example/object", contentType, length: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") };
}

