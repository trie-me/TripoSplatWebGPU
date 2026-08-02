import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import test from "node:test";

const install = fileURLToPath(new URL("../install/install.sh", import.meta.url));
const run = fileURLToPath(new URL("../run-worker.sh", import.meta.url));

test("installer is Linux-only, immutable-versioned, frozen, and foreground-only", async () => {
  const text = await readFile(install, "utf8");
  assert.match(text, /triposplat-v/);
  assert.match(text, /uv sync --project .*--frozen --no-dev/);
  assert.match(text, /--download-models/);
  assert.doesNotMatch(text, /systemctl|systemd|launchctl|cron|docker run/);
});

test("foreground launcher has CUDA and ROCm detection but no Windows or macOS routes", async () => {
  const text = await readFile(run, "utf8");
  assert.match(text, /rocminfo/);
  assert.match(text, /nvidia-smi/);
  assert.match(text, /Linux only/);
  assert.doesNotMatch(text, /mps|metal|directml|dx12/i);
});

