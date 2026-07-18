#!/usr/bin/env python3
"""Compare two TripoSplat 17-float PLYs with geometry and fixed-camera proxies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SH_C0 = 0.28209479177387814
CAMERAS = {
    "front": ([0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    "side": ([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    "isometric": ([1.5, 1.5, 1.5], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args()


def load_ply(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    marker = b"end_header\n"
    end = raw.find(marker)
    if end < 0:
        raise ValueError(f"{path} has no end_header marker")
    header_end = end + len(marker)
    header = raw[:header_end].decode("ascii")
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path} is not binary little-endian PLY")
    count_line = next(line for line in header.splitlines() if line.startswith("element vertex "))
    count = int(count_line.rsplit(" ", 1)[1])
    values = np.frombuffer(raw, dtype="<f4", count=count * 17, offset=header_end).reshape(count, 17)
    if header_end + values.nbytes != len(raw):
        raise ValueError(f"{path} byte length does not match 17-float vertex layout")
    return {
        "positions": values[:, 0:3].copy(),
        "colors": np.clip(0.5 + SH_C0 * values[:, 6:9], 0.0, 1.0),
        "opacities": (1.0 / (1.0 + np.exp(-values[:, 9].astype(np.float64)))).astype(np.float32),
        "scales": np.exp(values[:, 10:13].astype(np.float64)).astype(np.float32),
        "rotations": values[:, 13:17].copy(),
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(
            ("p01", "p10", "p50", "p90", "p99"),
            np.quantile(values, (0.01, 0.1, 0.5, 0.9, 0.99)),
        )
    }


def geometry_stats(scene: dict[str, np.ndarray]) -> dict[str, Any]:
    positions = scene["positions"].astype(np.float64)
    scales = scene["scales"].astype(np.float64)
    opacities = scene["opacities"].astype(np.float64)
    center = positions.mean(axis=0)
    radial = np.linalg.norm(positions - center, axis=1)
    minimum_scale = scales.min(axis=1)
    anisotropy = scales.max(axis=1) / np.maximum(minimum_scale, 1e-12)
    finite = sum(int(np.size(value) - np.isfinite(value).sum()) for value in scene.values())
    active = opacities >= 0.01
    active_positions = positions[active]
    voxels = np.unique(np.floor(np.clip((active_positions + 1.0) * 64.0, 0, 127)).astype(np.int16), axis=0)
    return {
        "pointCount": int(positions.shape[0]),
        "nonFiniteValues": finite,
        "positionBounds": {"minimum": positions.min(axis=0).tolist(), "maximum": positions.max(axis=0).tolist()},
        "positionCentroid": center.tolist(),
        "positionSpread": positions.std(axis=0).tolist(),
        "radialDistance": {"mean": float(radial.mean()), **quantiles(radial)},
        "scale": {
            "mean": float(scales.mean()),
            **quantiles(scales),
            "minimumAxis": quantiles(minimum_scale),
            "fractionBelow0.002": float(np.mean(minimum_scale < 0.002)),
            "fractionBelow0.005": float(np.mean(minimum_scale < 0.005)),
            "fractionAnisotropyAbove10": float(np.mean(anisotropy > 10.0)),
        },
        "opacity": {
            "mean": float(opacities.mean()),
            **quantiles(opacities),
            "fractionAtLeast0.01": float(np.mean(opacities >= 0.01)),
            "fractionAtLeast0.1": float(np.mean(opacities >= 0.1)),
            "fractionAtLeast0.5": float(np.mean(opacities >= 0.5)),
        },
        "activeVoxelCount128": int(voxels.shape[0]),
        "activeVoxels": voxels,
    }

def render(scene: dict[str, np.ndarray], camera: tuple[list[float], list[float], list[float]], size: int) -> tuple[np.ndarray, dict[str, float]]:
    eye, target, up = (np.asarray(value, dtype=np.float64) for value in camera)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    vertical = np.cross(right, forward)
    relative = scene["positions"].astype(np.float64) - eye
    depth = relative @ forward
    focal = 1.0 / math.tan(math.radians(45.0) / 2.0)
    x = (relative @ right) * focal / depth
    y = (relative @ vertical) * focal / depth
    valid = (depth > 0.01) & (np.abs(x) <= 1.0) & (np.abs(y) <= 1.0)
    pixel_x = np.clip(((x[valid] + 1.0) * 0.5 * (size - 1)).astype(np.int64), 0, size - 1)
    pixel_y = np.clip(((1.0 - y[valid]) * 0.5 * (size - 1)).astype(np.int64), 0, size - 1)
    indices = pixel_y * size + pixel_x
    projected_radius = scene["scales"][valid].max(axis=1) * focal * size / (2.0 * depth[valid])
    weights = scene["opacities"][valid].astype(np.float64) * np.clip(projected_radius, 0.5, 8.0) ** 2
    weight_sum = np.bincount(indices, weights=weights, minlength=size * size)
    color_sum = np.stack([
        np.bincount(indices, weights=weights * scene["colors"][valid, channel], minlength=size * size)
        for channel in range(3)
    ], axis=1)
    colors = color_sum / np.maximum(weight_sum[:, None], 1e-12)
    alpha = 1.0 - np.exp(-0.15 * weight_sum)
    image = np.concatenate((colors * alpha[:, None], alpha[:, None]), axis=1).reshape(size, size, 4)
    mask = alpha.reshape(size, size) >= 0.05
    occupied = np.argwhere(mask)
    holes = 1.0
    if occupied.size:
        low = occupied.min(axis=0)
        high = occupied.max(axis=0) + 1
        holes = float(1.0 - mask[low[0]:high[0], low[1]:high[1]].mean())
    return image, {
        "projectedPoints": int(valid.sum()),
        "coverageFraction": float(mask.mean()),
        "boundingBoxHoleFraction": holes,
    }


def image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = candidate[:, :, :3] - reference[:, :, :3]
    rmse = float(np.sqrt(np.mean(difference * difference)))
    x = reference[:, :, :3].mean(axis=2).ravel()
    y = candidate[:, :, :3].mean(axis=2).ravel()
    c1, c2 = 0.01**2, 0.03**2
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    ssim = ((2 * x.mean() * y.mean() + c1) * (2 * covariance + c2)) / ((x.mean() ** 2 + y.mean() ** 2 + c1) * (x.var() + y.var() + c2))
    mask_x = reference[:, :, 3] >= 0.05
    mask_y = candidate[:, :, 3] >= 0.05
    union = np.logical_or(mask_x, mask_y).sum()
    return {
        "rgbRmse": rmse,
        "rgbPsnrDb": None if rmse == 0 else float(-20.0 * math.log10(rmse)),
        "globalLuminanceSsim": float(ssim),
        "coverageMaskIou": float(np.logical_and(mask_x, mask_y).sum() / max(union, 1)),
        "changedPixelFractionAt1Over255": float(np.mean(np.max(np.abs(difference), axis=2) > 1 / 255)),
    }

def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive")
    reference = load_ply(args.reference)
    candidate = load_ply(args.candidate)
    reference_stats = geometry_stats(reference)
    candidate_stats = geometry_stats(candidate)
    reference_voxels = {tuple(value) for value in reference_stats.pop("activeVoxels")}
    candidate_voxels = {tuple(value) for value in candidate_stats.pop("activeVoxels")}
    args.render_dir.mkdir(parents=True, exist_ok=True)
    views: dict[str, Any] = {}
    for name, camera in CAMERAS.items():
        reference_image, reference_view = render(reference, camera, args.size)
        candidate_image, candidate_view = render(candidate, camera, args.size)
        Image.fromarray(np.rint(reference_image * 255).astype(np.uint8), "RGBA").save(args.render_dir / f"reference-{name}.png")
        Image.fromarray(np.rint(candidate_image * 255).astype(np.uint8), "RGBA").save(args.render_dir / f"candidate-{name}.png")
        views[name] = {
            "camera": {"position": camera[0], "target": camera[1], "up": camera[2], "verticalFovDegrees": 45},
            "reference": reference_view,
            "candidate": candidate_view,
            "comparison": image_metrics(reference_image, candidate_image),
        }
    report = {
        "method": "Decoded linear scale/opacity geometry plus deterministic opacity-weighted center projections; projection images are qualification proxies, not the production Gaussian renderer.",
        "reference": {"path": str(args.reference.resolve()), "geometry": reference_stats},
        "candidate": {"path": str(args.candidate.resolve()), "geometry": candidate_stats},
        "topology": {
            "activeVoxelResolution": 128,
            "activeVoxelIou": len(reference_voxels & candidate_voxels) / max(len(reference_voxels | candidate_voxels), 1),
            "candidateOnlyActiveVoxels": len(candidate_voxels - reference_voxels),
            "referenceOnlyActiveVoxels": len(reference_voxels - candidate_voxels),
        },
        "fixedCameraViews": views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
