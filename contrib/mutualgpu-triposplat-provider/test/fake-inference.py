#!/usr/bin/env python3
import json
import sys
import zipfile
from pathlib import Path

if any(name.startswith("MUTUALGPU_") and any(part in name for part in ("KEY", "PASSWORD", "SECRET", "TOKEN")) for name in __import__("os").environ):
    emit = lambda value: print(json.dumps(value, separators=(",", ":")), flush=True)
    emit({"type": "startup_error", "category": "SecretLeakDetected"})
    raise SystemExit(1)


def emit(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)


emit({"type": "ready", "device": "cuda", "backend": "cuda", "modelRevision": "fake-revision", "diagnostics": {}})
for line in sys.stdin:
    request = json.loads(line)
    if request.get("type") != "generate":
        continue
    if request.get("steps") == 999:
        emit({"type": "error", "id": request["id"], "category": "ModelLoadFailed"})
        continue
    output = Path(request["outputDirectory"])
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"format": "triposplat-webgpu-result", "version": 1, "count": request["numGaussians"], "elapsedMs": 1, "sdkVersion": "fake"}
    (output / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with zipfile.ZipFile(output / "result.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("scene.ply", "ply\n")
        archive.writestr("scene.splat", b"x")
        archive.writestr("manifest.json", json.dumps(metadata))
    emit({"type": "progress", "id": request["id"], "phase": "sampling", "percent": 50, "message": "step"})
    emit({"type": "result", "id": request["id"], "metadata": metadata})

