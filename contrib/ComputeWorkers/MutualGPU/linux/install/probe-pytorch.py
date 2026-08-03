#!/usr/bin/env python3
"""Fail closed unless the selected frozen environment reaches the expected GPU backend."""

from __future__ import annotations

import argparse
import sys


def fail(message: str) -> None:
    print(f"PyTorch backend probe: {message}", file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cuda", "rocm"), required=True)
    args = parser.parse_args()
    try:
        import torch
    except Exception as error:  # pragma: no cover - exercised by installer hosts
        fail(f"could not import torch ({type(error).__name__}).")
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is false; the selected GPU runtime is not usable.")
    cuda_version = getattr(torch.version, "cuda", None)
    hip_version = getattr(torch.version, "hip", None)
    if args.backend == "cuda":
        if not cuda_version or hip_version:
            fail(f"expected a CUDA PyTorch build, found cuda={cuda_version!r} hip={hip_version!r}.")
    elif not hip_version:
        fail(f"expected a ROCm/HIP PyTorch build, found cuda={cuda_version!r} hip={hip_version!r}.")
    try:
        device = torch.cuda.get_device_name(0)
        memory_mib = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except Exception as error:  # pragma: no cover - exercised by installer hosts
        fail(f"could not query GPU 0 ({type(error).__name__}).")
    print(
        f"PyTorch backend probe: backend={args.backend} device={device} "
        f"memoryMiB={memory_mib} torch={torch.__version__} cuda={cuda_version or '-'} hip={hip_version or '-'}"
    )


if __name__ == "__main__":
    main()
