import { createHash, randomUUID } from "node:crypto";
import { open } from "node:fs/promises";
import { join } from "node:path";
import { acceptedImageContentTypes } from "./config.mjs";

export const MAX_INPUT_BYTES = 25 * 1024 * 1024;

export class InputDownloadError extends Error {
  constructor(category, message) {
    super(message);
    this.name = "InputDownloadError";
    this.category = category;
  }
}

export async function downloadVerifiedInput(input, directory, { fetchImpl = globalThis.fetch, signal, maxBytes = MAX_INPUT_BYTES } = {}) {
  validateDescriptor(input, maxBytes);
  let response;
  try {
    response = await fetchImpl(input.url, { headers: { Accept: input.contentType }, redirect: "error", signal });
  } catch (error) {
    if (signal?.aborted) throw abortError();
    throw new InputDownloadError("download", "The provider could not download the input image.");
  }
  if (response.status === 401 || response.status === 403 || response.status === 404) {
    throw new InputDownloadError("expired", "The input URL could no longer be used.");
  }
  if (!response.ok) throw new InputDownloadError("download", "The provider could not download the input image.");
  const contentLength = response.headers?.get?.("content-length");
  if (contentLength != null && contentLength !== "" && (!/^\d+$/.test(contentLength) || Number(contentLength) !== input.length)) {
    throw new InputDownloadError("integrity", "The input image length did not match its descriptor.");
  }
  const path = join(directory, `input-${randomUUID()}.${extensionFor(input.contentType)}`);
  const output = await open(path, "wx", 0o600);
  const hash = createHash("sha256");
  let length = 0;
  try {
    if (!response.body) throw new InputDownloadError("download", "The input image had no body.");
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = Buffer.from(value);
      length += chunk.length;
      if (length > maxBytes || length > input.length) throw new InputDownloadError("integrity", "The input image exceeded its declared size.");
      hash.update(chunk);
      await output.write(chunk);
    }
  } finally {
    await output.close();
  }
  if (length !== input.length || hash.digest("hex") !== input.sha256.toLowerCase()) {
    throw new InputDownloadError("integrity", "The input image did not match its descriptor.");
  }
  const source = await open(path, "r");
  let bytes;
  try { bytes = (await source.read({ length: 12, position: 0 })).buffer; }
  finally { await source.close(); }
  if (!hasExpectedMagic(bytes, input.contentType)) throw new InputDownloadError("integrity", "The input image did not match its declared media type.");
  return path;
}

export function validateDescriptor(input, maxBytes = MAX_INPUT_BYTES) {
  if (!input || typeof input !== "object") throw new InputDownloadError("validation", "The assignment does not contain an image input.");
  let url;
  try { url = new URL(input.url); } catch { throw new InputDownloadError("validation", "The input image URL is invalid."); }
  if (url.protocol !== "https:") throw new InputDownloadError("validation", "The input image URL must use HTTPS.");
  if (!acceptedImageContentTypes.includes(input.contentType)) throw new InputDownloadError("validation", "The input image type is unsupported.");
  if (!Number.isSafeInteger(input.length) || input.length <= 0 || input.length > maxBytes) throw new InputDownloadError("validation", "The input image size is unsupported.");
  if (typeof input.sha256 !== "string" || !/^[a-f0-9]{64}$/i.test(input.sha256)) throw new InputDownloadError("validation", "The input image checksum is invalid.");
}

function extensionFor(contentType) {
  return { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp" }[contentType];
}
function hasExpectedMagic(bytes, contentType) {
  if (contentType === "image/png") return bytes.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
  if (contentType === "image/jpeg") return bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
  return bytes.subarray(0, 4).toString("ascii") === "RIFF" && bytes.subarray(8, 12).toString("ascii") === "WEBP";
}
function abortError() { const error = new Error("Input download was cancelled."); error.name = "AbortError"; return error; }

