import assert from "node:assert/strict";
import { chmod, mkdtemp, mkdir, readFile, realpath, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import test from "node:test";

const worker = fileURLToPath(new URL("..", import.meta.url));
const install = join(worker, "install", "install.sh");
const sourceInstall = join(worker, "install", "install-from-source.sh");
const installerLibrary = join(worker, "install", "lib.sh");
const fixtureDirectory = join(worker, "test", "fixtures", "installer");

test("fixture-driven NVIDIA detection selects CUDA, writes no files in dry-run, and skips unsafe PATH entries", async () => {
  const fixture = await loadFixture("nvidia-ready");
  await withFixture(fixture, async context => {
    const result = await runInstaller(context);
    assert.equal(result.code, 0, result.stderr);
    assert.match(result.stdout, /inxi graphics inventory/);
    assert.match(result.stdout, /NVIDIA driver readiness/);
    assert.match(result.stdout, new RegExp(`selected ${fixture.expectedBackend} \\(requested auto\\)`));
    assert.match(result.stdout, new RegExp(`safe user-global shim directory: ${escape(context.safeShim)} \\(PATH\\)`));
    await assert.rejects(stat(context.dataRoot));
    await assert.rejects(stat(context.configRoot));
  });
});

test("fixture-driven ROCm detection requires both rocminfo and AMD management readiness", async () => {
  const fixture = await loadFixture("rocm-ready");
  await withFixture(fixture, async context => {
    const result = await runInstaller(context);
    assert.equal(result.code, 0, result.stderr);
    assert.match(result.stdout, /ROCm compute readiness/);
    assert.match(result.stdout, /AMD management readiness: amd-smi/);
    assert.match(result.stdout, new RegExp(`selected ${fixture.expectedBackend} \\(requested auto\\)`));
  });
});

test("fixture-driven ROCm detection rejects rocminfo without an AMD management probe", async () => {
  const fixture = await loadFixture("rocm-unmanaged");
  await withFixture(fixture, async context => {
    const result = await runInstaller(context);
    assert.equal(result.code, 2);
    assert.match(result.stdout, /ROCm compute readiness/);
    assert.match(result.stdout, /neither amd-smi nor rocm-smi could inspect/);
    assert.match(result.stderr, /no usable NVIDIA CUDA or AMD ROCm runtime/);
  });
});

test("fixture-driven mixed hardware is never silently guessed and an explicit override works", async () => {
  const fixture = await loadFixture("mixed-ready");
  await withFixture(fixture, async context => {
    const automatic = await runInstaller(context);
    assert.equal(automatic.code, 2);
    assert.match(automatic.stderr, /both CUDA and ROCm look viable/);

    const selected = await runInstaller(context, ["--backend", "rocm"]);
    assert.equal(selected.code, 0, selected.stderr);
    assert.match(selected.stdout, /selected rocm \(requested rocm\)/);
  });
});

test("branch bootstrap piped from GitHub retrieves its matching helper before a dry-run", async () => {
  const fixture = await loadFixture("nvidia-ready");
  await withFixture(fixture, async context => {
    const curl = join(context.root, "commands", "curl");
    await writeFile(curl, [
      "#!/bin/sh",
      "set -eu",
      "output=''",
      "while [ $# -gt 0 ]; do if [ \"$1\" = --output ]; then output=\"$2\"; shift 2; else shift; fi; done",
      "cp \"$TRIPOSPLAT_TEST_INSTALLER_LIBRARY\" \"$output\""
    ].join("\n"));
    await chmod(curl, 0o755);
    const result = await run("/bin/bash", [
      "-s", "--",
      "--ref", "codex/compute-workers-mutualgpu-linux",
      "--backend", "cuda",
      "--dry-run"
    ], {
      ...context.environment,
      TRIPOSPLAT_PROVIDER_RAW_BASE: "https://raw.fixture/trie-me/TripoSplatWebGPU",
      TRIPOSPLAT_TEST_INSTALLER_LIBRARY: installerLibrary
    }, await readFile(sourceInstall, "utf8"));
    assert.equal(result.code, 0, result.stderr);
    assert.match(result.stdout, /selected cuda \(requested cuda\)/);
    assert.match(result.stdout, /would clone https:\/\/github.com\/trie-me\/TripoSplatWebGPU.git/);
  });
});

test("source bootstrap restores the foreground launcher permission and can install without starting", async () => {
  const fixture = await loadFixture("nvidia-ready");
  await withFixture(fixture, async context => {
    const sourceContent = join(context.root, "source-content");
    const sourceWorker = join(sourceContent, "contrib", "ComputeWorkers", "MutualGPU", "linux");
    const destination = join(context.root, "source-destination");
    await mkdir(join(sourceWorker, "runtime", "cuda", ".venv", "bin"), { recursive: true });
    await mkdir(join(sourceWorker, "install"), { recursive: true });
    await writeFile(join(sourceWorker, "run-worker.sh"), "#!/bin/sh\nexit 0\n", { mode: 0o644 });
    await writeFile(join(sourceWorker, "runtime", "cuda", ".venv", "bin", "python"), "#!/bin/sh\nexit 0\n", { mode: 0o755 });
    await chmod(join(sourceWorker, "runtime", "cuda", ".venv", "bin", "python"), 0o755);
    await writeFile(join(context.root, "commands", "git"), [
      "#!/bin/sh",
      "set -eu",
      "for argument in \"$@\"; do destination=\"$argument\"; done",
      "cp -R \"$TRIPOSPLAT_TEST_SOURCE_CONTENT/.\" \"$destination\""
    ].join("\n"));
    await writeFile(join(context.root, "commands", "npm"), "#!/bin/sh\nexit 0\n");
    await writeFile(join(context.root, "commands", "uv"), "#!/bin/sh\nexit 0\n");
    await Promise.all(["git", "npm", "uv"].map(name => chmod(join(context.root, "commands", name), 0o755)));
    const result = await run("/bin/bash", [
      sourceInstall,
      "--ref", "codex/compute-workers-mutualgpu-linux",
      "--backend", "cuda",
      "--destination", destination,
      "--no-run"
    ], { ...context.environment, TRIPOSPLAT_TEST_SOURCE_CONTENT: sourceContent });
    assert.equal(result.code, 0, result.stderr);
    assert.match(result.stdout, /foreground worker was not started/);
    assert.equal((await stat(join(destination, "contrib", "ComputeWorkers", "MutualGPU", "linux", "run-worker.sh"))).mode & 0o777, 0o755);
  });
});

test("configuration is private, contains only the key-file path, and the PATH resolver selects a safe user directory", async () => {
  const fixture = await loadFixture("nvidia-ready");
  await withFixture(fixture, async context => {
    const configuration = join(context.root, "configuration");
    const installation = join(context.root, "installed");
    await mkdir(configuration, { mode: 0o700 });
    await mkdir(join(installation, "current"), { recursive: true, mode: 0o700 });
    const fakeWorker = join(installation, "current", "run-worker.sh");
    await writeFile(fakeWorker, "#!/usr/bin/env bash\nprintf 'backend=%s model=%s key=%s args=%s\\n' \"$MUTUALGPU_TRIPOSPLAT_BACKEND\" \"$MUTUALGPU_TRIPOSPLAT_MODEL_DIR\" \"$MUTUALGPU_PROVIDER_KEY_FILE\" \"$*\"\n");
    await chmod(fakeWorker, 0o755);
    const keyPath = join(context.root, "private-provider.key");
    const result = await run("/bin/bash", ["-c", [
      "source \"$1\"",
      "triposplat_resolve_shim_directory ''",
      "test \"$TRIPOSPLAT_SHIM_DIR\" = \"$2\"",
      "triposplat_write_config \"$3\" \"$4\" cuda \"$5\" \"$6\" https://mutualgpu.example",
      "triposplat_write_launcher \"$3\"",
      "triposplat_write_shim \"$2/mutualgpu-triposplat\" \"$3/launcher\" run",
      "triposplat_write_shim \"$2/mutualgpu-triposplat-restart\" \"$3/launcher\" restart"
    ].join("; "), "--", installerLibrary, context.safeShim, configuration, installation, join(context.root, "models"), keyPath], context.environment);
    assert.equal(result.code, 0, result.stderr);
    const config = await readFile(join(configuration, "config.env"), "utf8");
    assert.match(config, /MUTUALGPU_TRIPOSPLAT_BACKEND=cuda/);
    assert.match(config, new RegExp(`MUTUALGPU_PROVIDER_KEY_FILE=${escape(keyPath)}`));
    assert.doesNotMatch(config, /opaque-provider-secret/);
    assert.equal((await stat(join(configuration, "config.env"))).mode & 0o777, 0o600);
    assert.equal((await stat(join(configuration, "launcher"))).mode & 0o777, 0o700);
    assert.match(await readFile(join(context.safeShim, "mutualgpu-triposplat"), "utf8"), /Managed by MutualGPU TripoSplat installer/);
    assert.match(await readFile(join(context.safeShim, "mutualgpu-triposplat-restart"), "utf8"), /restart/);
    const launch = await run(join(context.safeShim, "mutualgpu-triposplat"), [], context.environment);
    assert.equal(launch.code, 0, launch.stderr);
    assert.match(launch.stdout, /backend=cuda .*args=run/);
    const restart = await run(join(context.safeShim, "mutualgpu-triposplat-restart"), [], context.environment);
    assert.equal(restart.code, 0, restart.stderr);
    assert.match(restart.stdout, /backend=cuda .*args=run/);
    assert.match(restart.stdout, /no daemon is installed; restart starts a fresh foreground worker/);
  });
});

test("installer remains Linux-only, immutable-versioned, frozen, and foreground-only", async () => {
  const text = await readFile(install, "utf8");
  const sourceText = await readFile(sourceInstall, "utf8");
  const library = await readFile(installerLibrary, "utf8");
  assert.match(text, /triposplat-v/);
  assert.match(text, /triposplat_sync_frozen_environment/);
  assert.match(sourceText, /triposplat_sync_frozen_environment/);
  assert.match(sourceText, /chmod 0755 "\$worker_dir\/run-worker\.sh"/);
  assert.match(sourceText, /Paste the actual MutualGPU provider key/);
  assert.match(sourceText, /exec "\$worker_dir\/run-worker\.sh" run/);
  assert.match(sourceText, /--no-run/);
  assert.match(text, /probe-pytorch.py/);
  assert.match(library, /inxi/);
  assert.match(library, /lspci/);
  assert.match(library, /rocminfo/);
  assert.match(library, /nvidia-smi/);
  assert.match(library, /env -u SSL_CERT_FILE -u SSL_CERT_DIR uv sync .*--system-certs/);
  assert.match(library, /triposplat_download_models/);
  assert.match(text, /mutualgpu-triposplat-restart/);
  assert.doesNotMatch(`${text}\n${library}`, /systemctl|systemd|launchctl|cron|docker run/);
});

test("foreground launcher has CUDA and ROCm detection but no Windows or macOS routes", async () => {
  const text = await readFile(join(worker, "run-worker.sh"), "utf8");
  const library = await readFile(installerLibrary, "utf8");
  assert.match(text, /source "\$worker_dir\/install\/lib.sh"/);
  assert.match(library, /rocminfo/);
  assert.match(library, /nvidia-smi/);
  assert.match(text, /Linux only/);
  assert.doesNotMatch(text, /mps|metal|directml|dx12/i);
});

async function loadFixture(name) {
  return JSON.parse(await readFile(join(fixtureDirectory, `${name}.json`), "utf8"));
}

async function withFixture(fixture, callback) {
  const root = await mkdtemp(join(tmpdir(), "triposplat-installer-"));
  try {
    const home = join(root, "home");
    const unsafeShim = join(root, "unsafe-bin");
    const safeShim = join(root, "safe-bin");
    const commands = join(root, "commands");
    const dataRoot = join(root, "data", "mutualgpu", "triposplat");
    const configRoot = join(root, "config", "mutualgpu", "triposplat");
    await Promise.all([mkdir(home, { mode: 0o700 }), mkdir(unsafeShim, { mode: 0o777 }), mkdir(safeShim, { mode: 0o700 }), mkdir(commands, { mode: 0o700 })]);
    await chmod(unsafeShim, 0o777);
    await writeFixtureCommands(commands, fixture);
    const environment = {
      ...process.env,
      HOME: home,
      XDG_DATA_HOME: join(root, "data"),
      XDG_CONFIG_HOME: join(root, "config"),
      PATH: `${unsafeShim}:${safeShim}:${commands}:${process.env.PATH}`
    };
    await callback({ root, safeShim: await realpath(safeShim), dataRoot, configRoot, environment });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

async function writeFixtureCommands(directory, fixture) {
  const scripts = {
    uname: `case "$1" in -s) echo Linux ;; -m) echo x86_64 ;; *) exit 2 ;; esac`,
    inxi: `echo "Graphics: ${fixture.graphics}"`,
    lspci: `echo "01:00.0 VGA compatible controller: ${fixture.graphics}"`
  };
  if (fixture.cuda) {
    scripts["nvidia-smi"] = `if [ "$1" = "-L" ]; then echo "GPU 0: Fixture NVIDIA (UUID: GPU-fixture)"; else echo "Fixture NVIDIA, 555.42, 16384"; fi`;
  }
  if (fixture.rocmInfo) {
    scripts.rocminfo = "echo '  Name: gfx1100'";
  }
  if (fixture.amdManagement) {
    scripts["amd-smi"] = "test \"$1\" = list";
  }
  await Promise.all(Object.entries(scripts).map(async ([name, body]) => {
    const file = join(directory, name);
    await writeFile(file, `#!/bin/sh\nset -eu\n${body}\n`);
    await chmod(file, 0o755);
  }));
}

function runInstaller(context, extra = []) {
  return run("/bin/bash", [install, "--version", "triposplat-v0.1.0", "--dry-run", ...extra], context.environment);
}

function run(command, args, environment, input = "") {
  return new Promise(resolve => {
    const child = spawn(command, args, { env: environment, stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", value => { stdout += value; });
    child.stderr.on("data", value => { stderr += value; });
    child.stdin.end(input);
    child.on("close", code => resolve({ code, stdout, stderr }));
  });
}

function escape(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
