#!/usr/bin/env python3
"""Serve the exact official fp32 TripoSplat flow sampler over localhost.

The browser uploads fixed-shape fp32 latent, camera, DINO and VAE tensors once.
This service runs all conditional/unconditional DiT calls plus official CFG/Euler
on native PyTorch MPS and returns the final fp32 latent and camera. It binds to
loopback by default and requires a startup token on every inference request.
"""

from __future__ import annotations

import argparse
import gc
import hmac
import json
import secrets
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from dit_common import (
    CAMERA_SHAPE,
    FEATURE1_SHAPE,
    FEATURE2_SHAPE,
    LATENT_SHAPE,
    choose_torch_device,
    load_official_flow_model,
    resolved_file,
    sha256_file,
    source_revision,
    synchronize_torch,
)

PROTOCOL_VERSION = "1"
FLOAT_BYTES = np.dtype("<f4").itemsize
LATENT_ELEMENTS = int(np.prod(LATENT_SHAPE))
CAMERA_ELEMENTS = int(np.prod(CAMERA_SHAPE))
FEATURE1_ELEMENTS = int(np.prod(FEATURE1_SHAPE))
FEATURE2_ELEMENTS = int(np.prod(FEATURE2_SHAPE))
REQUEST_ELEMENTS = LATENT_ELEMENTS + CAMERA_ELEMENTS + FEATURE1_ELEMENTS + FEATURE2_ELEMENTS
RESPONSE_ELEMENTS = LATENT_ELEMENTS + CAMERA_ELEMENTS
REQUEST_BYTES = REQUEST_ELEMENTS * FLOAT_BYTES
RESPONSE_BYTES = RESPONSE_ELEMENTS * FLOAT_BYTES
OFFICIAL_COMMIT = "a78fa12d06dbf1381ca548bfac32bb68cb8c451d"
OFFICIAL_WEIGHTS_SHA256 = "c870b97ac1d6bc9177608a5ec625e19ef9f3c5019aa68f64b0fb7803abcd6d20"
OFFICIAL_TORCH_VERSION = "2.13.0"
OFFICIAL_NUMPY_VERSION = "2.5.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triposplat-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", help="At least 16 characters; random when omitted.")
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Exact browser Origin. Repeat as needed; localhost Vite origins are allowed by default.",
    )
    parser.add_argument("--allow-dirty-source", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.token is not None and len(args.token) < 16:
        parser.error("--token must contain at least 16 characters")
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("--host must remain loopback-only (127.0.0.1 or localhost)")
    return args


def split_request(body: bytes) -> dict[str, np.ndarray]:
    if len(body) != REQUEST_BYTES:
        raise ValueError(f"request has {len(body)} bytes; expected {REQUEST_BYTES}")
    values = np.frombuffer(body, dtype="<f4")
    offset = 0
    result: dict[str, np.ndarray] = {}
    for name, count, shape in (
        ("latent", LATENT_ELEMENTS, LATENT_SHAPE),
        ("camera", CAMERA_ELEMENTS, CAMERA_SHAPE),
        ("feature1", FEATURE1_ELEMENTS, FEATURE1_SHAPE),
        ("feature2", FEATURE2_ELEMENTS, FEATURE2_SHAPE),
    ):
        # Copy out of the immutable HTTP request and enforce contiguous native fp32.
        result[name] = np.array(values[offset : offset + count], dtype=np.float32, copy=True).reshape(shape)
        offset += count
    return result


def pack_response(result: dict[str, Any]) -> bytes:
    arrays = []
    for name, shape in (("latent", LATENT_SHAPE), ("camera", CAMERA_SHAPE)):
        tensor = result.get(name)
        if tensor is None:
            raise RuntimeError(f"official sampler did not return {name}")
        array = np.ascontiguousarray(tensor.detach().float().cpu().numpy(), dtype="<f4")
        if array.shape != shape:
            raise RuntimeError(f"official sampler {name} shape is {array.shape}; expected {shape}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"official sampler {name} contains non-finite values")
        arrays.append(array.reshape(-1))
    payload = np.concatenate(arrays).astype("<f4", copy=False).tobytes()
    if len(payload) != RESPONSE_BYTES:
        raise RuntimeError(f"response has {len(payload)} bytes; expected {RESPONSE_BYTES}")
    return payload


class MpsFlowApplication:
    def __init__(
        self,
        torch: Any,
        model: Any,
        source: Any,
        device: Any,
        token: str,
        source_commit: str,
        source_dirty: bool | None,
        weights: Path,
        weights_sha256: str,
        model_load_ms: float,
        allowed_origins: set[str],
    ) -> None:
        self.torch = torch
        self.model = model
        self.source = source
        self.device = device
        self.token = token
        self.source_commit = source_commit
        self.source_dirty = source_dirty
        self.weights = weights
        self.weights_sha256 = weights_sha256
        self.model_load_ms = model_load_ms
        self.allowed_origins = allowed_origins
        self.inference_lock = threading.Lock()

    def origin_allowed(self, origin: str | None) -> bool:
        if origin is None:
            return True
        if origin in self.allowed_origins:
            return True
        parsed = urlparse(origin)
        return (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        )

    def authorized(self, token: str | None) -> bool:
        return token is not None and hmac.compare_digest(token, self.token)

    def run(self, arrays: dict[str, np.ndarray], steps: int, guidance: float, shift: float) -> tuple[bytes, float]:
        torch = self.torch
        with self.inference_lock:
            condition = {
                "feature1": torch.from_numpy(arrays["feature1"]).to(device=self.device, dtype=torch.float32),
                "feature2": torch.from_numpy(arrays["feature2"]).to(device=self.device, dtype=torch.float32),
            }
            negative = {name: torch.zeros_like(value) for name, value in condition.items()}
            noise = {
                "latent": torch.from_numpy(arrays["latent"]).to(device=self.device, dtype=torch.float32),
                "camera": torch.from_numpy(arrays["camera"]).to(device=self.device, dtype=torch.float32),
            }
            sampler = self.source.pipeline_module.FlowEulerCfgSampler()
            synchronize_torch(torch, self.device)
            started = time.perf_counter()
            with torch.inference_mode():
                result = sampler.sample(
                    self.model,
                    {name: value.clone() for name, value in noise.items()},
                    cond=condition,
                    neg_cond=negative,
                    steps=steps,
                    guidance_scale=guidance,
                    shift=shift,
                )
            synchronize_torch(torch, self.device)
            inference_ms = (time.perf_counter() - started) * 1000
            payload = pack_response(result)
        del result, sampler, noise, negative, condition, arrays
        gc.collect()
        return payload, inference_ms

    def health(self) -> dict[str, Any]:
        return {
            "ready": True,
            "protocol": PROTOCOL_VERSION,
            "device": str(self.device),
            "precision": "fp32",
            "torchVersion": self.torch.__version__,
            "sourceCommit": self.source_commit,
            "sourceDirty": self.source_dirty,
            "weightsSha256": self.weights_sha256,
            "modelLoadMs": self.model_load_ms,
            "requestBytes": REQUEST_BYTES,
            "responseBytes": RESPONSE_BYTES,
            "schedule": [4, 20],
        }


def handler_class(application: MpsFlowApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "TripoSplatMPS/1"

        def log_message(self, message: str, *args: Any) -> None:
            sys.stderr.write(
                f"{self.log_date_time_string()} {self.client_address[0]} {message % args}\n"
            )

        def cors(self) -> bool:
            origin = self.headers.get("Origin")
            if not application.origin_allowed(origin):
                self.error(HTTPStatus.FORBIDDEN, f"Origin is not allowed: {origin}")
                return False
            if origin is not None:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            return True

        def error(self, status: HTTPStatus, message: str) -> None:
            payload = json.dumps({"error": message}).encode("utf-8")
            self.send_response(status)
            if not self.cors_for_error():
                return
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def cors_for_error(self) -> bool:
            origin = self.headers.get("Origin")
            if origin is not None and application.origin_allowed(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            return True

        def do_OPTIONS(self) -> None:
            origin = self.headers.get("Origin")
            if not application.origin_allowed(origin):
                self.error(HTTPStatus.FORBIDDEN, f"Origin is not allowed: {origin}")
                return
            self.send_response(HTTPStatus.NO_CONTENT)
            if not self.cors():
                return
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header(
                "Access-Control-Allow-Headers",
                "Content-Type, X-Triposplat-Protocol, X-Triposplat-Token, "
                "X-Triposplat-Steps, X-Triposplat-Guidance-Scale, X-Triposplat-Shift",
            )
            self.send_header(
                "Access-Control-Expose-Headers",
                "X-Triposplat-Inference-Ms, X-Triposplat-Model-Load-Ms, "
                "X-Triposplat-Source-Commit",
            )
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_GET(self) -> None:
            if urlparse(self.path).path != "/v1/health":
                self.error(HTTPStatus.NOT_FOUND, "not found")
                return
            origin = self.headers.get("Origin")
            if not application.origin_allowed(origin):
                self.error(HTTPStatus.FORBIDDEN, f"Origin is not allowed: {origin}")
                return
            payload = json.dumps(application.health(), sort_keys=True).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            if not self.cors():
                return
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/v1/flow":
                self.error(HTTPStatus.NOT_FOUND, "not found")
                return
            if not application.origin_allowed(self.headers.get("Origin")):
                self.error(HTTPStatus.FORBIDDEN, "origin is not allowed")
                return
            if not application.authorized(self.headers.get("X-Triposplat-Token")):
                self.error(HTTPStatus.UNAUTHORIZED, "invalid service token")
                return
            if self.headers.get("X-Triposplat-Protocol") != PROTOCOL_VERSION:
                self.error(HTTPStatus.BAD_REQUEST, "unsupported protocol version")
                return
            try:
                content_length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self.error(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
                return
            if content_length != REQUEST_BYTES:
                self.error(
                    HTTPStatus.BAD_REQUEST,
                    f"request has {content_length} bytes; expected {REQUEST_BYTES}",
                )
                return
            try:
                steps = int(self.headers.get("X-Triposplat-Steps", ""))
                guidance = float(self.headers.get("X-Triposplat-Guidance-Scale", ""))
                shift = float(self.headers.get("X-Triposplat-Shift", ""))
                if steps not in (4, 20):
                    raise ValueError("steps must be 4 or 20")
                if not np.isfinite(guidance) or guidance <= 0:
                    raise ValueError("guidance scale must be positive and finite")
                if not np.isfinite(shift) or shift <= 0:
                    raise ValueError("shift must be positive and finite")
                body = self.rfile.read(content_length)
                arrays = split_request(body)
            except ValueError as error:
                self.error(HTTPStatus.BAD_REQUEST, str(error))
                return
            try:
                payload, inference_ms = application.run(arrays, steps, guidance, shift)
            except Exception as error:
                self.log_error("inference failed: %s", error)
                self.error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
                return
            self.send_response(HTTPStatus.OK)
            if not self.cors():
                return
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Triposplat-Inference-Ms", f"{inference_ms:.6f}")
            self.send_header("X-Triposplat-Model-Load-Ms", f"{application.model_load_ms:.6f}")
            self.send_header("X-Triposplat-Source-Commit", application.source_commit)
            self.send_header(
                "Access-Control-Expose-Headers",
                "X-Triposplat-Inference-Ms, X-Triposplat-Model-Load-Ms, "
                "X-Triposplat-Source-Commit",
            )
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def main() -> None:
    try:
        import torch
    except ImportError as error:
        raise SystemExit("PyTorch is required. Use the repository's TripoSplat Python environment.") from error

    args = parse_args()
    torch_version = str(torch.__version__).split("+", 1)[0]
    if torch_version != OFFICIAL_TORCH_VERSION:
        raise RuntimeError(
            f"PyTorch version is {torch.__version__}; exact v2 requires {OFFICIAL_TORCH_VERSION}"
        )
    if np.__version__ != OFFICIAL_NUMPY_VERSION:
        raise RuntimeError(
            f"NumPy version is {np.__version__}; exact v2 requires {OFFICIAL_NUMPY_VERSION}"
        )
    repository = args.triposplat_repo.expanduser().resolve()
    weights = resolved_file(args.weights, "TripoSplat flow-model weights")
    weights_sha256 = sha256_file(weights)
    if weights_sha256 != OFFICIAL_WEIGHTS_SHA256:
        raise RuntimeError(
            f"flow-model weights sha256 is {weights_sha256}; expected {OFFICIAL_WEIGHTS_SHA256}"
        )
    commit, dirty = source_revision(repository)
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"official source commit is {commit}; expected {OFFICIAL_COMMIT}")
    if dirty and not args.allow_dirty_source:
        raise RuntimeError("official model.py or triposplat.py is dirty; pass --allow-dirty-source only for diagnostics")
    device = choose_torch_device(torch, "mps")
    token = args.token or secrets.token_urlsafe(32)
    print(f"Loading untouched official fp32 flow model at {commit} on {device}", flush=True)
    started = time.perf_counter()
    model, source = load_official_flow_model(
        torch=torch,
        triposplat_repo=repository,
        weights=weights,
        device=device,
        internal_precision="fp32",
        low_memory_construction=True,
    )
    synchronize_torch(torch, device)
    model_load_ms = (time.perf_counter() - started) * 1000
    application = MpsFlowApplication(
        torch=torch,
        model=model,
        source=source,
        device=device,
        token=token,
        source_commit=commit,
        source_dirty=dirty,
        weights=weights,
        weights_sha256=weights_sha256,
        model_load_ms=model_load_ms,
        allowed_origins=set(args.allow_origin),
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_class(application))
    server.daemon_threads = True
    print(
        json.dumps(
            {
                "ready": True,
                "serviceUrl": f"http://{args.host}:{args.port}/",
                "token": token,
                **application.health(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
