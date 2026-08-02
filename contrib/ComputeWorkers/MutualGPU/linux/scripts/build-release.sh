#!/usr/bin/env bash
set -euo pipefail
umask 077

worker_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(git -C "$worker_dir" rev-parse --show-toplevel)"
version="${1:-}"
[[ "$version" =~ ^triposplat-v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: build-release.sh triposplat-vX.Y.Z" >&2; exit 2; }
asset_version="${version#triposplat-}"
for backend in cuda rocm; do [[ -f "$worker_dir/runtime/$backend/uv.lock" ]] || { echo "missing frozen lock: runtime/$backend/uv.lock" >&2; exit 2; }; done
out_dir="$worker_dir/dist"
stage="$(mktemp -d "${TMPDIR:-/tmp}/triposplat-mutualgpu-release.XXXXXX")"
cleanup() { rm -rf -- "$stage"; }
trap cleanup EXIT
archive_root="$stage/mutualgpu-triposplat-$asset_version"
mkdir -p "$archive_root/node_modules/@mutualgpu/provider-core/src" "$archive_root/node_modules/@mutualgpu/provider-node/src"
cp -R "$worker_dir/src" "$worker_dir/vendor/triposplat" "$worker_dir/runtime" "$worker_dir/install" "$archive_root/"
cp "$worker_dir/model-manifest.json" "$worker_dir/run-worker.sh" "$worker_dir/README.md" "$worker_dir/package.json" "$worker_dir/LICENSE" "$worker_dir/NOTICE" "$archive_root/"
cp -R "$worker_dir/vendor/mutualgpu-sdk/provider-core/src/." "$archive_root/node_modules/@mutualgpu/provider-core/src/"
cp "$worker_dir/vendor/mutualgpu-sdk/provider-core/package.json" "$archive_root/node_modules/@mutualgpu/provider-core/package.json"
cp -R "$worker_dir/vendor/mutualgpu-sdk/provider-node/src/." "$archive_root/node_modules/@mutualgpu/provider-node/src/"
cp "$worker_dir/vendor/mutualgpu-sdk/provider-node/package.json" "$archive_root/node_modules/@mutualgpu/provider-node/package.json"
printf '%s\n' "$(git -C "$repo_dir" rev-parse HEAD)" >"$archive_root/TRIPOSPLAT_WEBGPU_REVISION"
mkdir -p "$out_dir"
archive="$out_dir/mutualgpu-triposplat-${asset_version}-linux-x86_64.tar.gz"
tar -C "$stage" -czf "$archive" "$(basename "$archive_root")"
if command -v sha256sum >/dev/null; then sha256sum "$archive" >"$archive.sha256"; else shasum -a 256 "$archive" >"$archive.sha256"; fi
cp "$worker_dir/install/install.sh" "$out_dir/install.sh"
if command -v sha256sum >/dev/null; then sha256sum "$out_dir/install.sh" >"$out_dir/install.sh.sha256"; else shasum -a 256 "$out_dir/install.sh" >"$out_dir/install.sh.sha256"; fi
archive_sha="$(awk '{print $1}' "$archive.sha256")"
bootstrap_sha="$(awk '{print $1}' "$out_dir/install.sh.sha256")"
source_revision="$(git -C "$repo_dir" rev-parse HEAD)"
printf '{\n  "format": "triposplat-mutualgpu-provider-release",\n  "version": "%s",\n  "sourceRevision": "%s",\n  "archive": {"name": "%s", "sha256": "%s"},\n  "bootstrap": {"name": "install.sh", "sha256": "%s"}\n}\n' \
  "$version" "$source_revision" "$(basename "$archive")" "$archive_sha" "$bootstrap_sha" >"$out_dir/release-manifest.json"
printf '{\n  "bomFormat": "CycloneDX",\n  "specVersion": "1.5",\n  "version": 1,\n  "components": [\n    {"type":"application","name":"@ai3d/triposplat-mutualgpu-provider","version":"%s","licenses":[{"license":{"id":"AGPL-3.0-only"}}]},\n    {"type":"library","name":"@mutualgpu/provider-core","version":"0.1.0-preview.4","licenses":[{"license":{"id":"AGPL-3.0-only"}}]},\n    {"type":"library","name":"@mutualgpu/provider-node","version":"0.1.0-preview.4","licenses":[{"license":{"id":"AGPL-3.0-only"}}]},\n    {"type":"library","name":"TripoSplat","version":"a78fa12d06dbf1381ca548bfac32bb68cb8c451d","licenses":[{"license":{"id":"MIT"}}]}\n  ]\n}\n' \
  "$version" >"$out_dir/sbom.cdx.json"
printf '{\n  "predicateType": "https://slsa.dev/provenance/v1",\n  "subject": [{"name": "%s", "digest": {"sha256": "%s"}}],\n  "buildDefinition": {"externalParameters": {"sourceRevision": "%s", "workerVersion": "%s"}}\n}\n' \
  "$(basename "$archive")" "$archive_sha" "$source_revision" "$version" >"$out_dir/provenance.json"
echo "Built $archive, immutable checksums, release manifest, SBOM, and provenance."
