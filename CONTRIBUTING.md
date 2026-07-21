# Contributing

Thank you for helping improve TripoSplat WebGPU. This is an engineering preview with strict numerical, privacy, and provenance requirements.

## Before opening a change

1. Read the [developer guide](docs/developer-guide.md) and the detailed [contribution and release process](docs/contributing-and-release.md).
2. Open an issue before a large API, model, numerical, or deployment change.
3. Do not commit model weights, generated fixtures, credentials, signed URLs, proprietary inputs, or machine-local paths.
4. Keep benchmark claims tied to a machine-readable report, exact environment, and unchanged acceptance thresholds.

## Local checks

Use Node.js 22 or newer and the pinned pnpm version:

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm audit:public
pnpm typecheck
pnpm test
pnpm --filter @ai3d/triposplat-webgpu test
pnpm lint
pnpm build
pnpm test:package-consumer
```

Browser numerical changes also require the relevant deterministic WebGPU validation described in the detailed contribution guide.

## Pull requests

Keep changes focused, explain user-visible and numerical impact, list validation performed, and disclose generated or AI-assisted changes that need extra review. By contributing, you confirm that you have the right to submit the code and assets. Do not assume that repository visibility grants a blanket license; see the [licensing section](README.md#licensing) and [third-party notices](THIRD_PARTY_NOTICES.md).
