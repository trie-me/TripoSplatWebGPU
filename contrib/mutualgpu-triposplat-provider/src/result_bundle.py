"""Interoperable TripoSplat result validation and ZIP construction."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

MAX_RESULT_BYTES = 50 * 1024 * 1024
REQUIRED_ENTRIES = ("scene.ply", "scene.splat", "manifest.json")


def build_result_bundle(output_directory: Path, gaussian: Any, count: int, elapsed_ms: int, sdk_version: str, provenance: dict[str, Any]) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    ply_path = output_directory / "scene.ply"
    splat_path = output_directory / "scene.splat"
    gaussian.save_ply(ply_path)
    gaussian.save_splat(splat_path)
    validate_scene_files(ply_path, splat_path, count)
    manifest = {
        "format": "triposplat-webgpu-result",
        "version": 1,
        "count": count,
        "elapsedMs": elapsed_ms,
        "sdkVersion": sdk_version,
        "provenance": provenance,
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    result_path = output_directory / "result.zip"
    with zipfile.ZipFile(result_path, "w", compression=zipfile.ZIP_STORED, strict_timestamps=False) as archive:
        for entry in REQUIRED_ENTRIES:
            archive.write(output_directory / entry, entry)
    validate_result_bundle(result_path, count)
    metadata_path = output_directory / "metadata.json"
    metadata_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return manifest


def validate_scene_files(ply_path: Path, splat_path: Path, count: int) -> None:
    header = ply_path.read_bytes()[:8192]
    if not header.startswith(b"ply\nformat binary_little_endian 1.0\n"):
        raise RuntimeError("scene.ply is not a binary little-endian PLY file")
    if f"element vertex {count}\n".encode("ascii") not in header:
        raise RuntimeError("scene.ply Gaussian count does not match the requested result")
    if splat_path.stat().st_size != count * 32:
        raise RuntimeError("scene.splat does not contain exactly 32 bytes per Gaussian")


def validate_result_bundle(result_path: Path, count: int) -> None:
    if result_path.stat().st_size <= 0 or result_path.stat().st_size > MAX_RESULT_BYTES:
        raise RuntimeError("result ZIP exceeds MutualGPU's 50 MiB limit")
    with zipfile.ZipFile(result_path) as archive:
        if tuple(archive.namelist()) != REQUIRED_ENTRIES:
            raise RuntimeError("result ZIP must contain only scene.ply, scene.splat, and manifest.json")
        manifest = json.loads(archive.read("manifest.json"))
    if manifest.get("format") != "triposplat-webgpu-result" or manifest.get("version") != 1 or manifest.get("count") != count:
        raise RuntimeError("result ZIP manifest is not compatible with the WebGPU requestor")

