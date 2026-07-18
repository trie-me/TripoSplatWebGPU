#!/usr/bin/env python3
"""Probe MPS SDPA query-geometry sensitivity with exact captured Q/K/V tensors.

The first four query rows and every K/V row come from untouched official
TripoSplat captures. Only the total query count changes. Because attention rows
are mathematically independent, changes in the retained four outputs identify
backend geometry/tiling behavior rather than a model or sampler change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

DEFAULT_QUERY_LENGTHS = (
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    3072,
    4095,
    4096,
    4097,
    4100,
    4101,
)
HEADS = 16
HEAD_DIM = 64
CONDITION_TOKENS = 4101
RETAINED_ROWS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--invocation", action="append", type=int, dest="invocations")
    parser.add_argument(
        "--query-length",
        action="append",
        type=int,
        dest="query_lengths",
        help="Total query count to replay; repeat for multiple values.",
    )
    parser.add_argument(
        "--tail",
        choices=("actual-prefix", "repeat-first", "repeat-four", "zero"),
        default="actual-prefix",
        help="Content used after the retained first four queries.",
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.invocations = args.invocations or [7, 8]
    args.query_lengths = args.query_lengths or list(DEFAULT_QUERY_LENGTHS)
    if any(value not in (7, 8) for value in args.invocations):
        parser.error("--invocation currently accepts only the paired captures 7 and 8")
    if len(set(args.invocations)) != len(args.invocations):
        parser.error("--invocation values must be unique")
    if any(value < RETAINED_ROWS or value > CONDITION_TOKENS for value in args.query_lengths):
        parser.error(
            f"--query-length must be in [{RETAINED_ROWS}, {CONDITION_TOKENS}]"
        )
    args.query_lengths = sorted(set(args.query_lengths))
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def metrics(reference: Any, candidate: Any) -> dict[str, Any]:
    import numpy as np

    expected = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(candidate, dtype=np.float32)
    if expected.shape != actual.shape:
        raise ValueError(f"Shape mismatch: {expected.shape} versus {actual.shape}")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(delta)
    denominator = math.sqrt(
        float(np.dot(expected.reshape(-1).astype(np.float64), expected.reshape(-1)))
        * float(np.dot(actual.reshape(-1).astype(np.float64), actual.reshape(-1)))
    )
    return {
        "maxAbsoluteError": float(absolute.max()),
        "meanAbsoluteError": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "cosineSimilarity": (
            float(
                np.dot(
                    expected.reshape(-1).astype(np.float64),
                    actual.reshape(-1).astype(np.float64),
                )
                / denominator
            )
            if denominator
            else 1.0
        ),
        "candidateSha256": array_sha256(actual),
    }


def query_prefix(q: Any, length: int, tail: str) -> Any:
    import numpy as np

    if tail == "actual-prefix":
        return np.ascontiguousarray(q[:, :, :length, :])
    result = np.empty((q.shape[0], q.shape[1], length, q.shape[3]), dtype=np.float32)
    result[:, :, :RETAINED_ROWS, :] = q[:, :, :RETAINED_ROWS, :]
    if length == RETAINED_ROWS:
        return result
    if tail == "repeat-first":
        result[:, :, RETAINED_ROWS:, :] = q[:, :, :1, :]
    elif tail == "repeat-four":
        indices = np.arange(length - RETAINED_ROWS) % RETAINED_ROWS
        result[:, :, RETAINED_ROWS:, :] = q[:, :, indices, :]
    elif tail == "zero":
        result[:, :, RETAINED_ROWS:, :] = 0
    else:
        raise AssertionError(f"Unhandled tail mode {tail}")
    return result


def load_capture(root: Path, invocation: int) -> tuple[Any, Any, Any, Any, dict[str, str]]:
    import numpy as np

    base = root / f"invocation_{invocation:02d}" / "untouched_official"
    paths = {
        "qNormalized": base / "q_normalized.npy",
        "kTransposed": base / "k_transposed.npy",
        "vTransposed": base / "v_transposed.npy",
        "weightedValueTransposed": base / "weighted_value_transposed.npy",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing capture files: {missing}")
    q_normalized = np.load(paths["qNormalized"], allow_pickle=False)
    q = np.ascontiguousarray(q_normalized.transpose(0, 2, 1, 3))
    k = np.ascontiguousarray(np.load(paths["kTransposed"], allow_pickle=False))
    v = np.ascontiguousarray(np.load(paths["vTransposed"], allow_pickle=False))
    reference = np.ascontiguousarray(
        np.load(paths["weightedValueTransposed"], allow_pickle=False)
    )
    expected_shapes = {
        "q": (1, HEADS, CONDITION_TOKENS, HEAD_DIM),
        "k": (1, HEADS, CONDITION_TOKENS, HEAD_DIM),
        "v": (1, HEADS, CONDITION_TOKENS, HEAD_DIM),
        "reference": (1, HEADS, RETAINED_ROWS, HEAD_DIM),
    }
    for name, value in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("reference", reference),
    ):
        if value.shape != expected_shapes[name] or value.dtype != np.float32:
            raise ValueError(
                f"{name} is {value.shape}/{value.dtype}; "
                f"expected {expected_shapes[name]}/float32"
            )
    return q, k, v, reference, {
        name: file_sha256(path) for name, path in paths.items()
    }


def run_invocation(
    torch: Any,
    root: Path,
    invocation: int,
    query_lengths: list[int],
    tail: str,
    device: Any,
    warmup: bool,
) -> dict[str, Any]:
    q, k, v, reference, hashes = load_capture(root, invocation)
    k_device = torch.from_numpy(k).to(device)
    v_device = torch.from_numpy(v).to(device)
    synchronize = torch.mps.synchronize if device.type == "mps" else lambda: None
    if warmup:
        warmup_q = torch.from_numpy(q[:, :, :RETAINED_ROWS, :]).to(device)
        with torch.inference_mode():
            torch.nn.functional.scaled_dot_product_attention(
                warmup_q, k_device, v_device
            )
        synchronize()
        del warmup_q
    records = []
    for length in query_lengths:
        q_value = query_prefix(q, length, tail)
        q_device = torch.from_numpy(q_value).to(device)
        synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = torch.nn.functional.scaled_dot_product_attention(
                q_device, k_device, v_device
            )
        synchronize()
        duration_ms = (time.perf_counter() - started) * 1000.0
        retained = output[:, :, :RETAINED_ROWS, :].detach().float().cpu().numpy()
        records.append(
            {
                "queryLength": length,
                "durationMs": duration_ms,
                "retainedRowsVersusFullOfficial": metrics(reference, retained),
            }
        )
        del q_device, output, retained
    del k_device, v_device
    if device.type == "mps":
        torch.mps.empty_cache()
    best = min(
        records,
        key=lambda item: item["retainedRowsVersusFullOfficial"]["rmse"],
    )
    return {
        "invocation": invocation,
        "captureSha256": hashes,
        "referenceSha256": array_sha256(reference),
        "records": records,
        "bestQueryLengthByRmse": best["queryLength"],
        "bestRmse": best["retainedRowsVersusFullOfficial"]["rmse"],
    }


def main() -> None:
    import numpy as np
    import torch

    args = parse_args()
    capture_root = args.capture_root.expanduser().resolve()
    report = args.report.expanduser().resolve()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    device = torch.device(args.device)
    result = {
        "schemaVersion": 1,
        "component": "context_refiner.0_attention_query_geometry",
        "status": "diagnostic_only",
        "device": str(device),
        "torchVersion": torch.__version__,
        "numpyVersion": np.__version__,
        "tailPolicy": args.tail,
        "retainedRows": RETAINED_ROWS,
        "keyValueRows": CONDITION_TOKENS,
        "queryLengths": args.query_lengths,
        "warmup": args.warmup,
        "captureRoot": str(capture_root),
        "invocations": [
            run_invocation(
                torch,
                capture_root,
                invocation,
                args.query_lengths,
                args.tail,
                device,
                args.warmup,
            )
            for invocation in args.invocations
        ],
        "interpretation": (
            "Only total query geometry and optional non-retained query content change. "
            "The retained first four Q rows and all K/V rows are exact untouched-official "
            "captures; attention rows are mathematically independent."
        ),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
