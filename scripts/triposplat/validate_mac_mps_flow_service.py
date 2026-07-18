#!/usr/bin/env python3
"""Run the unchanged official 20-step fixture through the Mac MPS service."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from dit_common import comparison_metrics
from run_mac_mps_flow_service import (
    CAMERA_ELEMENTS,
    FEATURE1_ELEMENTS,
    FEATURE2_ELEMENTS,
    LATENT_ELEMENTS,
    PROTOCOL_VERSION,
    RESPONSE_BYTES,
)

QUALIFICATION = {"absolute": 0.005, "relative": 0.003, "minimum_cosine_similarity": 0.99999998}
STRICT = {"absolute": 0.0001, "relative": 0.001, "minimum_cosine_similarity": 0.99999999}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", default="http://127.0.0.1:8765/")
    parser.add_argument("--token", required=True)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("public/fixtures/generated/flow20-fp32-trajectory"),
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def read_f32(path: Path, elements: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != elements:
        raise ValueError(f"{path} has {values.size} fp32 values; expected {elements}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values")
    return values


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def endpoint(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def read_health(base: str) -> dict[str, Any]:
    with urlopen(endpoint(base, "v1/health"), timeout=30) as response:
        return json.load(response)


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict[str, Any]:
    operating_system = command_output(["sw_vers", "-productVersion"])
    build = command_output(["sw_vers", "-buildVersion"])
    return {
        "hardware": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "operatingSystem": (
            f"macOS {operating_system} ({build})"
            if operating_system is not None and build is not None
            else platform.platform()
        ),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def post_fixture(base: str, token: str, payload: bytes) -> tuple[bytes, dict[str, str], float]:
    request = Request(
        endpoint(base, "v1/flow"),
        method="POST",
        data=payload,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Triposplat-Protocol": PROTOCOL_VERSION,
            "X-Triposplat-Token": token,
            "X-Triposplat-Steps": "20",
            "X-Triposplat-Guidance-Scale": "3",
            "X-Triposplat-Shift": "3",
        },
    )
    started = time.perf_counter()
    with urlopen(request, timeout=900) as response:
        body = response.read()
        headers = {name.lower(): value for name, value in response.headers.items()}
    return body, headers, (time.perf_counter() - started) * 1000


def qualified(metrics: dict[str, Any], tolerance: dict[str, float]) -> bool:
    return bool(
        metrics.get("passed")
        and metrics.get("cosine_similarity", -1) >= tolerance["minimum_cosine_similarity"]
    )


def gate(
    reference: np.ndarray,
    candidate: np.ndarray,
    tolerance: dict[str, float],
) -> dict[str, Any]:
    metrics = comparison_metrics(
        reference,
        candidate,
        atol=tolerance["absolute"],
        rtol=tolerance["relative"],
    )
    return {
        "tolerance": tolerance,
        **metrics,
        "passed": qualified(metrics, tolerance),
    }


def main() -> None:
    args = parse_args()
    fixture = args.fixture_dir.expanduser().resolve()
    flow_manifest = json.loads((fixture / "flow.json").read_text(encoding="utf-8"))
    arrays = [
        read_f32(fixture / "latent.f32", LATENT_ELEMENTS),
        read_f32(fixture / "camera.f32", CAMERA_ELEMENTS),
        read_f32(fixture / "feature1.f32", FEATURE1_ELEMENTS),
        read_f32(fixture / "feature2.f32", FEATURE2_ELEMENTS),
    ]
    payload = b"".join(np.ascontiguousarray(values, dtype="<f4").tobytes() for values in arrays)
    health = read_health(args.service_url)
    response, headers, wall_ms = post_fixture(args.service_url, args.token, payload)
    if len(response) != RESPONSE_BYTES:
        raise RuntimeError(f"service returned {len(response)} bytes; expected {RESPONSE_BYTES}")

    candidate = np.frombuffer(response, dtype="<f4")
    candidate_latent = candidate[:LATENT_ELEMENTS]
    candidate_camera = candidate[LATENT_ELEMENTS:]
    reference_latent = read_f32(fixture / "flow20_latent.f32", LATENT_ELEMENTS)
    reference_camera = read_f32(fixture / "flow20_camera.f32", CAMERA_ELEMENTS)

    qualification = {
        "latent": gate(reference_latent, candidate_latent, QUALIFICATION),
        "camera": gate(reference_camera, candidate_camera, QUALIFICATION),
    }
    qualification["passed"] = all(item["passed"] for item in qualification.values())
    strict = {
        "latent": gate(reference_latent, candidate_latent, STRICT),
        "camera": gate(reference_camera, candidate_camera, STRICT),
    }
    strict["passed"] = all(item["passed"] for item in strict.values())
    expected_hashes = {
        "latent": flow_manifest["outputs"]["latent"]["sha256"],
        "camera": flow_manifest["outputs"]["camera"]["sha256"],
    }
    actual_hashes = {
        "latent": sha256_bytes(candidate_latent.tobytes()),
        "camera": sha256_bytes(candidate_camera.tobytes()),
    }
    report = {
        "date": time.strftime("%Y-%m-%d"),
        "stage": "TripoSplat 20-step fp32 CFG/Euler flow loop through localhost PyTorch/MPS",
        "passed": qualification["passed"],
        "strictPassed": strict["passed"],
        "bitExact": actual_hashes == expected_hashes,
        "environment": environment(),
        "fixture": {
            "path": str(fixture),
            "sourceCommit": flow_manifest["source"]["commit"],
            "expectedSha256": expected_hashes,
        },
        "service": {
            "url": args.service_url,
            "health": health,
            "sourceCommitHeader": headers.get("x-triposplat-source-commit"),
        },
        "settings": {
            "steps": 20,
            "guidanceScale": 3,
            "shift": 3,
            "conditionalInvocations": 20,
            "unconditionalInvocations": 20,
        },
        "timingsMs": {
            "requestWall": wall_ms,
            "inference": float(headers["x-triposplat-inference-ms"]),
            "modelLoad": float(headers["x-triposplat-model-load-ms"]),
        },
        "outputSha256": actual_hashes,
        "qualification": qualification,
        "strictDiagnostic": strict,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not qualification["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
