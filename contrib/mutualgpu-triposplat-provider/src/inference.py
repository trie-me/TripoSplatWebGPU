#!/usr/bin/env python3
"""Persistent Linux CUDA/ROCm TripoSplat inference subprocess.

The provider key intentionally never enters this process. Node downloads and
verifies the requester image, then sends only a private local file path and
validated scalar values on the JSON-lines protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from result_bundle import build_result_bundle

WORKER_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = WORKER_DIR / "vendor" / "triposplat"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

PIPELINE: Any = None
TORCH: Any = None
DEVICE = ""
BACKEND = ""
MODEL_DIR = Path(os.environ.get("MUTUALGPU_TRIPOSPLAT_MODEL_DIR", ""))
MODEL_MANIFEST = Path(os.environ.get("MUTUALGPU_TRIPOSPLAT_MODEL_MANIFEST", ""))
SDK_VERSION = os.environ.get("MUTUALGPU_TRIPOSPLAT_SDK_VERSION", "unknown")
MAX_INPUT_PIXELS = 16_000_000


def emit(message: dict[str, Any]) -> None:
    print(json.dumps(message, separators=(",", ":")), flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def initialize() -> dict[str, Any]:
    global TORCH, DEVICE, BACKEND
    if sys.platform != "linux":
        raise RuntimeError("The TripoSplat native worker supports Linux only")
    try:
        import torch
        import numpy  # noqa: F401
        import safetensors  # noqa: F401
        import torchvision  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as error:
        raise RuntimeError("TripoSplat Python dependencies are not installed") from error
    TORCH = torch
    requested = os.environ.get("MUTUALGPU_TRIPOSPLAT_BACKEND", "auto").lower()
    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    if not torch.cuda.is_available():
        raise RuntimeError("No supported CUDA or ROCm accelerator is available")
    detected = "rocm" if hip else "cuda" if cuda else ""
    if requested not in ("auto", "cuda", "rocm"):
        raise RuntimeError("The requested TripoSplat backend is invalid")
    if not detected or (requested != "auto" and requested != detected):
        raise RuntimeError("The requested TripoSplat backend is unavailable")
    BACKEND = detected
    DEVICE = "cuda"
    diagnostics = runtime_diagnostics(torch)
    log(f"runtime ready: python={sys.version_info.major}.{sys.version_info.minor} torch={torch.__version__} backend={BACKEND} device={torch.cuda.get_device_name(0)}")
    return diagnostics


def model_manifest() -> dict[str, Any]:
    if not MODEL_MANIFEST.is_file():
        raise RuntimeError("TripoSplat model manifest is missing")
    try:
        manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
        if not isinstance(manifest.get("files"), list) or not manifest.get("revision"):
            raise ValueError("invalid manifest")
        return manifest
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("TripoSplat model manifest is invalid") from error


def verify_models(manifest: dict[str, Any]) -> None:
    if not MODEL_DIR.is_dir():
        raise RuntimeError("TripoSplat model directory is missing")
    for item in manifest["files"]:
        relative = item.get("path")
        expected_size = item.get("bytes")
        expected_hash = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_size, int) or not isinstance(expected_hash, str):
            raise RuntimeError("TripoSplat model manifest is invalid")
        path = MODEL_DIR / relative
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError("A TripoSplat model file is missing or has the wrong size")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise RuntimeError("A TripoSplat model file failed SHA-256 verification")


def load_pipeline(manifest: dict[str, Any]) -> Any:
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE
    try:
        from triposplat import TripoSplatPipeline
        log("loading verified official TripoSplat pipeline")
        PIPELINE = TripoSplatPipeline(
            ckpt_path=str(MODEL_DIR / "diffusion_models/triposplat_fp16.safetensors"),
            decoder_path=str(MODEL_DIR / "vae/triposplat_vae_decoder_fp16.safetensors"),
            dinov3_path=str(MODEL_DIR / "clip_vision/dino_v3_vit_h.safetensors"),
            flux2_vae_encoder_path=str(MODEL_DIR / "vae/flux2-vae.safetensors"),
            rmbg_path=str(MODEL_DIR / "background_removal/birefnet.safetensors"),
            device=DEVICE,
        )
    except Exception as error:
        PIPELINE = None
        raise RuntimeError("The official TripoSplat pipeline could not be loaded") from error
    return PIPELINE


def validate_input(path: str) -> None:
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = MAX_INPUT_PIXELS
        with Image.open(path) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_INPUT_PIXELS:
                raise RuntimeError("input image dimensions exceed the provider limit")
            image.verify()
    except Exception as error:
        raise RuntimeError("The verified input image could not be decoded safely") from error


def generate(request: dict[str, Any]) -> None:
    identifier = str(request["id"])
    output_directory = Path(request["outputDirectory"])
    input_path = str(request["inputPath"])
    started = time.monotonic()
    validate_input(input_path)
    pipeline = load_pipeline(model_manifest())
    seed = int(request["seed"])
    steps = int(request["steps"])
    count = int(request["numGaussians"])
    guidance = float(request["guidanceScale"])
    emit({"type": "progress", "id": identifier, "phase": "preprocessing", "percent": 3, "message": "Preparing the input image."})
    generator = TORCH.Generator(device=DEVICE).manual_seed(seed)
    prepared = pipeline.preprocess_image(input_path)
    emit({"type": "progress", "id": identifier, "phase": "dino", "percent": 8, "message": "Encoding image features with DINOv3."})
    cond = pipeline.encode_image(prepared, generator=generator)
    emit({"type": "progress", "id": identifier, "phase": "flux-vae", "percent": 15, "message": "Encoding Flux VAE features."})

    def progress(step: int, total: int) -> None:
        percent = 21 + (step / total) * 59
        emit({"type": "progress", "id": identifier, "phase": "sampling", "percent": percent, "message": f"Sampling step {step} of {total}."})

    latent = pipeline.sample_latent(cond, steps=steps, guidance_scale=guidance, generator=generator, callback=progress)
    emit({"type": "progress", "id": identifier, "phase": "octree", "percent": 81, "message": "Decoding the occupancy octree."})
    gaussian = pipeline.decode_latent(latent["latent"], num_gaussians=count)
    emit({"type": "progress", "id": identifier, "phase": "gaussian-decoder", "percent": 96, "message": "Packing Gaussian features."})
    elapsed_ms = round((time.monotonic() - started) * 1000)
    manifest = build_result_bundle(output_directory, gaussian, count, elapsed_ms, SDK_VERSION, {
        "backend": BACKEND,
        "torch": TORCH.__version__,
        "torchCuda": getattr(TORCH.version, "cuda", None),
        "torchHip": getattr(TORCH.version, "hip", None),
        "modelRevision": model_manifest()["revision"],
        "seed": seed,
        "steps": steps,
        "guidanceScale": guidance,
    })
    emit({"type": "progress", "id": identifier, "phase": "packing", "percent": 98, "message": "Validated PLY, SPLAT, and result manifest."})
    emit({"type": "result", "id": identifier, "metadata": manifest})


def runtime_diagnostics(torch: Any) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "backend": BACKEND,
        "torchCuda": getattr(torch.version, "cuda", None),
        "torchHip": getattr(torch.version, "hip", None),
        "gpu": torch.cuda.get_device_name(0),
        "vramBytes": int(torch.cuda.get_device_properties(0).total_memory),
    }
    if os.environ.get("MUTUALGPU_TRIPOSPLAT_VULKAN_DIAGNOSTICS", "true").lower() == "true" and shutil.which("vulkaninfo"):
        try:
            result = subprocess.run(["vulkaninfo", "--summary"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20, check=False)
            diagnostics["vulkanSummary"] = result.stdout[-12_000:] if result.returncode == 0 else "unavailable"
        except (OSError, subprocess.SubprocessError):
            diagnostics["vulkanSummary"] = "unavailable"
    return diagnostics


def safe_category(error: BaseException) -> str:
    messages: list[str] = []
    current: BaseException | None = error
    while current is not None and len(messages) < 8:
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    message = " ".join(messages).lower()
    if "dependencies are not installed" in message:
        return "MissingDependencies"
    if "linux only" in message:
        return "UnsupportedPlatform"
    if "accelerator is available" in message or "backend is unavailable" in message:
        return "GpuUnavailable"
    if "out of memory" in message:
        return "GpuOutOfMemory"
    if "model" in message and ("missing" in message or "verification" in message or "wrong size" in message):
        return "ModelVerificationFailed"
    if "pipeline could not be loaded" in message:
        return "ModelLoadFailed"
    if "input image" in message:
        return "InputValidationFailed"
    return type(error).__name__


def serve() -> None:
    diagnostics = initialize()
    manifest = model_manifest()
    verify_models(manifest)
    load_pipeline(manifest)
    emit({"type": "ready", "device": DEVICE, "backend": BACKEND, "modelRevision": manifest["revision"], "diagnostics": diagnostics})
    log("TripoSplat model warmup complete; accepting one task at a time")
    for line in sys.stdin:
        request: Any = None
        try:
            request = json.loads(line)
            if request.get("type") == "generate":
                generate(request)
        except BaseException as error:
            emit({"type": "error", "id": str(request.get("id", "")) if isinstance(request, dict) else "", "category": safe_category(error)})


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "--probe"
    diagnostics = initialize()
    if command == "--serve":
        serve()
        return
    if command == "--verify-models":
        verify_models(model_manifest())
    emit({"type": "ready", "device": DEVICE, "backend": BACKEND, "modelRevision": model_manifest().get("revision") if MODEL_MANIFEST.is_file() else None, "diagnostics": diagnostics})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        emit({"type": "startup_error", "category": safe_category(error)})
        raise SystemExit(1) from None

