import { readFile } from "node:fs/promises";
import { join } from "node:path";

export const MAX_RESULT_BYTES = 50 * 1024 * 1024;

export async function collectResult(directory) {
  const resultZip = await readFile(join(directory, "result.zip"));
  if (resultZip.length < 4 || resultZip.length > MAX_RESULT_BYTES || resultZip[0] !== 0x50 || resultZip[1] !== 0x4b) {
    throw new Error("TripoSplat result ZIP has an invalid format or size.");
  }
  const metadata = JSON.parse(await readFile(join(directory, "metadata.json"), "utf8"));
  if (metadata?.format !== "triposplat-webgpu-result" || metadata?.version !== 1) {
    throw new Error("TripoSplat result metadata does not use the interoperable envelope.");
  }
  return { resultZip, metadata, logs: `TripoSplat completed in ${metadata.elapsedMs ?? "unknown"} ms.\n` };
}

