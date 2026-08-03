# Linux native-worker migration and rollback

This worker is an additional foreground MutualGPU provider. It does not replace
the browser provider, alter the `tripo-splat` capability name, or change the
browser build. Keep the browser provider available until the native provider
has passed the full Linux qualification matrix.

## Test gates before a migration

1. Run `npm run check`, both frozen-lock checks, and the root repository test
   suite from the exact release commit.
2. Build the release archive and verify that its archive checksum, bootstrap
   checksum, SBOM, and provenance describe the same immutable version.
3. On an otherwise-unused Linux GPU host, use `--dry-run` first. Capture the
   displayed PCI/sysfs inventory, NVIDIA or AMD readiness, selected backend,
   and optional Vulkan diagnostic result.
4. Install a canary version with a provider key dedicated to that host. Confirm
   the final PyTorch CUDA or ROCm/HIP probe before entering the key or starting
   the foreground command.
5. Run 4- and 20-step tasks with seed 42. Verify the ZIP contains finite,
   browser-loadable `scene.ply`, `scene.splat`, and `manifest.json` with the
   exact `tripo-splat` result envelope.
6. Exercise cancellation, reconnect/rebind, failed upload, lost completion
   acknowledgement, Ctrl-C, and a clean manual relaunch. Only then admit the
   canary provider to normal traffic.

The automated migration fixture covers the installer transaction without a GPU:
an incomplete release leaves the previous `current` version active, a complete
upgrade changes it atomically, and `--rollback` returns to the previous version.
It is not a substitute for the real CUDA and ROCm qualification gates above.

## Immediate rollback

If the foreground native worker fails, stop it with Ctrl-C. The browser
provider remains unaffected and is the immediate service fallback. Do not
restart the native worker until the host report and its safe failure category
are understood.

To roll back installed worker software, choose a version that remains under
`versions/` and run the installer retained by the active immutable version:

```bash
${XDG_DATA_HOME:-$HOME/.local/share}/mutualgpu/triposplat/current/install/install.sh \
  --rollback triposplat-vX.Y.Z
```

This changes only the `current` symlink; it does not delete models, download
anything, start a background service, or alter drivers. Verify with a foreground
canary task before using the reverted version for normal work. If the original
installation used `--install-dir`, pass that same value to the rollback command.

If the worker needs to be removed from service entirely, stop the foreground
process and revoke or disable its dedicated provider credential through the
normal MutualGPU operator process. Removing an installed version is optional:

```bash
${XDG_DATA_HOME:-$HOME/.local/share}/mutualgpu/triposplat/current/install/install.sh \
  --version triposplat-vX.Y.Z --uninstall --yes
```

This removes only that immutable software version. It deliberately leaves model
data and the private configuration in place, so recovery does not require
re-downloading 3.78 GB of weights. Never use `--uninstall` on the last known
good version until the replacement has completed the qualification matrix.
