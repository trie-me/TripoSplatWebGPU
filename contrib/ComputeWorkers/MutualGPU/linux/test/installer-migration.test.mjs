import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { chmod, mkdtemp, mkdir, readFile, readdir, readlink, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import test from "node:test";

const worker = fileURLToPath(new URL("..", import.meta.url));
const installer = join(worker, "install", "install.sh");
const releaseBuild = join(worker, "scripts", "build-release.sh");

test("release migration preserves the active version on failure, upgrades atomically, and rolls back", async () => {
  await withReleaseFixture(async fixture => {
    const first = await install(fixture, "triposplat-v0.1.0");
    assert.equal(first.code, 0, first.stderr);
    await assertCurrent(fixture, "triposplat-v0.1.0");
    assert.equal((await stat(join(fixture.configRoot, "config.env"))).mode & 0o777, 0o600);

    const incomplete = await install(fixture, "triposplat-v0.2.0", { TRIPOSPLAT_TEST_RELEASE_LAYOUT: "incomplete" });
    assert.equal(incomplete.code, 2);
    assert.match(incomplete.stderr, /release archive is incomplete/);
    await assertCurrent(fixture, "triposplat-v0.1.0");
    assert.deepEqual((await readdir(join(fixture.dataRoot, "versions"))).filter(name => name.includes(".staging.")), []);

    const upgraded = await install(fixture, "triposplat-v0.2.0");
    assert.equal(upgraded.code, 0, upgraded.stderr);
    await assertCurrent(fixture, "triposplat-v0.2.0");

    const rollback = await run("/bin/bash", [installer, "--rollback", "triposplat-v0.1.0"], fixture.environment);
    assert.equal(rollback.code, 0, rollback.stderr);
    await assertCurrent(fixture, "triposplat-v0.1.0");
    assert.match(rollback.stdout, /user-global commands continue to run in the foreground/);

    const command = await run(join(fixture.shimDirectory, "mutualgpu-triposplat"), [], fixture.environment);
    assert.equal(command.code, 0, command.stderr);
    assert.match(command.stdout, /fixture worker backend=cuda args=run/);
  });
});

test("release build emits a self-contained curl bootstrap and ships the rollback runbook", async () => {
  const output = await mkdtemp(join(tmpdir(), "triposplat-release-build-"));
  try {
    const built = await run("/bin/bash", [releaseBuild, "triposplat-v0.1.0"], {
      ...process.env,
      TRIPOSPLAT_RELEASE_OUT_DIR: output
    });
    assert.equal(built.code, 0, built.stderr);
    const bootstrap = await readFile(join(output, "install.sh"), "utf8");
    assert.match(bootstrap, /triposplat_print_host_inventory/);
    assert.doesNotMatch(bootstrap, /installer_dir=|source "\$installer_dir\/lib\.sh"/);
    const syntax = await run("/bin/bash", ["-n", join(output, "install.sh")], process.env);
    assert.equal(syntax.code, 0, syntax.stderr);
    const archive = join(output, "mutualgpu-triposplat-v0.1.0-linux-x86_64.tar.gz");
    const contents = await run("tar", ["-tzf", archive], process.env);
    assert.equal(contents.code, 0, contents.stderr);
    assert.match(contents.stdout, /MIGRATION\.md/);
    assert.match(contents.stdout, /install\/lib\.sh/);
  } finally {
    await rm(output, { recursive: true, force: true });
  }
});

async function withReleaseFixture(callback) {
  const root = await mkdtemp(join(tmpdir(), "triposplat-migration-"));
  try {
    const home = join(root, "home");
    const shimDirectory = join(root, "bin");
    const commands = join(root, "commands");
    const content = join(root, "release-content");
    const dataRoot = join(root, "data", "mutualgpu", "triposplat");
    const configRoot = join(root, "config", "mutualgpu", "triposplat");
    const archive = join(root, "release.tar.gz");
    const checksum = join(root, "release.tar.gz.sha256");
    const keyFile = join(root, "provider.key");
    await Promise.all([
      mkdir(home, { mode: 0o700 }),
      mkdir(shimDirectory, { mode: 0o700 }),
      mkdir(commands, { mode: 0o700 }),
      createReleaseContent(content),
      writeFile(archive, "fixture release archive\n"),
      writeFile(keyFile, "fixture-provider-key\n", { mode: 0o600 })
    ]);
    await chmod(keyFile, 0o600);
    const digest = createHash("sha256").update(await readFile(archive)).digest("hex");
    await writeFile(checksum, `${digest}  fixture-release.tar.gz\n`);
    await writeFixtureCommands(commands);
    const environment = {
      ...process.env,
      HOME: home,
      XDG_DATA_HOME: join(root, "data"),
      XDG_CONFIG_HOME: join(root, "config"),
      MUTUALGPU_TRIPOSPLAT_RELEASE_BASE_URL: "https://release.fixture/triposplat",
      TRIPOSPLAT_TEST_RELEASE_ARCHIVE: archive,
      TRIPOSPLAT_TEST_RELEASE_CHECKSUM: checksum,
      TRIPOSPLAT_TEST_RELEASE_CONTENT: content,
      SSL_CERT_FILE: "/fixture/stale-cert.pem",
      SSL_CERT_DIR: "/fixture/stale-certs",
      PATH: `${shimDirectory}:${commands}:${process.env.PATH}`
    };
    await callback({ root, dataRoot, configRoot, shimDirectory, keyFile, environment });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

async function createReleaseContent(content) {
  await mkdir(join(content, "runtime", "cuda", ".venv", "bin"), { recursive: true });
  await mkdir(join(content, "runtime", "rocm"), { recursive: true });
  await mkdir(join(content, "install"), { recursive: true });
  await writeFile(join(content, "runtime", "cuda", "uv.lock"), "version = 1\n");
  await writeFile(join(content, "runtime", "rocm", "uv.lock"), "version = 1\n");
  await writeFile(join(content, "runtime", "cuda", ".venv", "bin", "python"), "#!/bin/sh\necho 'fixture PyTorch backend probe'\n");
  await writeFile(join(content, "run-worker.sh"), "#!/bin/sh\nprintf 'fixture worker backend=%s args=%s\\n' \"$MUTUALGPU_TRIPOSPLAT_BACKEND\" \"$*\"\n");
  await writeFile(join(content, "install", "model-download.py"), "#!/usr/bin/env python3\n");
  await writeFile(join(content, "install", "probe-pytorch.py"), "#!/usr/bin/env python3\n");
  await writeFile(join(content, "model-manifest.json"), "{}\n");
  await Promise.all([
    chmod(join(content, "runtime", "cuda", ".venv", "bin", "python"), 0o755),
    chmod(join(content, "run-worker.sh"), 0o755)
  ]);
}

async function writeFixtureCommands(directory) {
  const scripts = {
    uname: "case \"$1\" in -s) echo Linux ;; -m) echo x86_64 ;; *) exit 2 ;; esac",
    "nvidia-smi": "if [ \"$1\" = -L ]; then echo 'GPU 0: Fixture NVIDIA'; else echo 'Fixture NVIDIA, 555.42, 16384'; fi",
    curl: [
      "output=''",
      "while [ $# -gt 0 ]; do",
      "  if [ \"$1\" = --output ]; then output=\"$2\"; shift 2; else shift; fi",
      "done",
      "case \"$output\" in *.sha256) cp \"$TRIPOSPLAT_TEST_RELEASE_CHECKSUM\" \"$output\" ;; *) cp \"$TRIPOSPLAT_TEST_RELEASE_ARCHIVE\" \"$output\" ;; esac"
    ].join("\n"),
    tar: [
      "case \"$1\" in",
      "  -tzf)",
      "    printf '%s\\n' fixture/runtime/cuda/uv.lock fixture/runtime/rocm/uv.lock fixture/run-worker.sh fixture/install/model-download.py fixture/install/probe-pytorch.py fixture/model-manifest.json",
      "    ;;",
      "  -xzf)",
      "    destination='' previous=''",
      "    for value in \"$@\"; do if [ \"$previous\" = -C ]; then destination=\"$value\"; fi; previous=\"$value\"; done",
      "    cp -R \"$TRIPOSPLAT_TEST_RELEASE_CONTENT/.\" \"$destination/\"",
      "    if [ \"${TRIPOSPLAT_TEST_RELEASE_LAYOUT:-complete}\" = incomplete ]; then rm -f \"$destination/install/probe-pytorch.py\"; fi",
      "    ;;",
      "  *) exit 2 ;;",
      "esac"
    ].join("\n"),
    uv: [
      "if [ -n \"${SSL_CERT_FILE:-}\" ] || [ -n \"${SSL_CERT_DIR:-}\" ]; then echo 'stale SSL override leaked into uv' >&2; exit 2; fi",
      "for value in \"$@\"; do if [ \"$value\" = --system-certs ]; then exit 0; fi; done",
      "echo 'missing --system-certs' >&2",
      "exit 2"
    ].join("\n"),
    mv: [
      "source_path='' destination=''",
      "for value in \"$@\"; do case \"$value\" in -*) ;; *) if [ -z \"$source_path\" ]; then source_path=\"$value\"; else destination=\"$value\"; fi ;; esac; done",
      "if [ -e \"$destination\" ] || [ -L \"$destination\" ]; then rm -f \"$destination\"; fi",
      "exec /bin/mv \"$source_path\" \"$destination\""
    ].join("\n")
  };
  await Promise.all(Object.entries(scripts).map(async ([name, body]) => {
    const file = join(directory, name);
    await writeFile(file, `#!/bin/sh\nset -eu\n${body}\n`);
    await chmod(file, 0o755);
  }));
}

function install(fixture, version, extraEnvironment = {}) {
  return run("/bin/bash", [
    installer,
    "--version", version,
    "--backend", "cuda",
    "--provider-key-file", fixture.keyFile
  ], { ...fixture.environment, ...extraEnvironment });
}

async function assertCurrent(fixture, version) {
  assert.equal(await readlink(join(fixture.dataRoot, "current")), `versions/${version}`);
}

function run(command, args, environment) {
  return new Promise(resolve => {
    const child = spawn(command, args, { env: environment, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", value => { stdout += value; });
    child.stderr.on("data", value => { stderr += value; });
    child.on("close", code => resolve({ code, stdout, stderr }));
  });
}
