#!/usr/bin/env python3
"""
Batch denoising for 3D light-sheet TIFF stacks.

Data model
----------
- One TIFF file = one 3D volume in Z,Y,X order.
- One folder = one organoid/position.
- Many TIFFs in that folder = timepoints of the same organoid.
- Denoising is applied to every Z slice in every TIFF. Slices are not treated as
  independent files; they are saved back as one 3D stack.

This script is intended to run after registration and background subtraction.
It deliberately avoids uint8 movie-style conversion and preserves quantitative
uint16 intensity by default.

Recommended conservative first pass:
python3 script/denoise.py \
  --input-root result/bgsub/position_1 \
  --output-root result/denoised \
  --glob-pattern "*.tif*" \
  --method median_gaussian2d \
  --median-size 3 \
  --sigma-xy 0.6 \
  --intensity-floor 5 \
  --compression zlib \
  --overwrite

python3 script/denoise.py \
  --input-root result/bgsub/position_1 \
  --output-root result/denoised_mild \
  --glob-pattern "*.tif*" \
  --method median_gaussian2d \
  --median-size 3 \
  --sigma-xy 0.3 \
  --intensity-floor 3 \
  --output-dtype same \
  --compression zlib \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, median_filter

_NATURAL_SORT_RE = re.compile(r"(\d+)")
TIFF_EXTS = {".tif", ".tiff"}


@dataclass(frozen=True)
class DenoiseConfig:
    method: str
    median_size: int
    sigma_xy: float
    sigma_z: float
    intensity_floor: float
    output_dtype: str
    compression: str | None
    overwrite: bool
    crop_to_foreground: bool
    crop_padding_xy: int
    crop_padding_z: int
    mask_percentile: float
    min_size_voxels: int


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    return tuple(int(tok) if tok.isdigit() else tok.lower() for tok in _NATURAL_SORT_RE.split(path.name))


def is_tiff(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tif") or name.endswith(".tiff") or name.endswith(".ome.tif") or name.endswith(".ome.tiff")


def discover_positions(input_root: Path, glob_pattern: str) -> dict[str, list[Path]]:
    subdirs = sorted((d for d in input_root.iterdir() if d.is_dir() and not d.name.startswith(".")), key=lambda p: p.name)
    positions: dict[str, list[Path]] = {}
    for subdir in subdirs:
        files = sorted((p for p in subdir.glob(glob_pattern) if p.is_file() and is_tiff(p)), key=natural_sort_key)
        if files:
            positions[subdir.name] = files
    if positions:
        return positions
    files = sorted((p for p in input_root.glob(glob_pattern) if p.is_file() and is_tiff(p)), key=natural_sort_key)
    if files:
        positions[input_root.name] = files
    return positions


def _can_stack_tiff_pages_as_z(tf: tifffile.TiffFile) -> bool:
    if len(tf.pages) < 2:
        return False
    first = tf.pages[0]
    if len(first.shape) != 2:
        return False
    first_shape = tuple(first.shape)
    first_dtype = np.dtype(first.dtype)
    return all(tuple(p.shape) == first_shape and np.dtype(p.dtype) == first_dtype for p in tf.pages)


def _read_tiff_pages_as_zyx(tf: tifffile.TiffFile) -> np.ndarray:
    n_z = len(tf.pages)
    y, x = tf.pages[0].shape
    dtype = np.dtype(tf.pages[0].dtype)
    arr = np.empty((n_z, y, x), dtype=dtype)
    for z, page in enumerate(tf.pages):
        arr[z] = page.asarray()
    return arr


def read_volume(path: str | Path) -> np.ndarray:
    path = str(path)
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes = getattr(series, "axes", "") or ""
        arr = series.asarray()
        if arr.ndim == 2 and axes == "YX" and _can_stack_tiff_pages_as_z(tf):
            arr = _read_tiff_pages_as_zyx(tf)
            axes = "ZYX"

    # Remove singleton axes and interpret common TIFF axes.
    if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
        arr = arr[..., 0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected one 3D ZYX stack, got shape {arr.shape} axes={axes!r}")

    if axes and set("ZYX").issubset(set(axes)) and len(axes) == 3 and axes != "ZYX":
        order = [axes.index(ax) for ax in "ZYX"]
        arr = np.transpose(arr, order)
    return np.ascontiguousarray(arr)


def save_volume(path: str | Path, arr: np.ndarray, compression: str | None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr, bigtiff=True, compression=compression, metadata=None)


def cast_output(arr: np.ndarray, input_dtype: np.dtype, output_dtype: str) -> np.ndarray:
    output_dtype = output_dtype.lower()
    if output_dtype == "float32":
        return arr.astype(np.float32, copy=False)
    if output_dtype == "uint8":
        positive = arr[arr > 0]
        hi = float(np.percentile(positive, 99.9)) if positive.size else 1.0
        hi = max(hi, 1e-6)
        return (np.clip(arr, 0, hi) / hi * 255).astype(np.uint8)
    if output_dtype == "uint16":
        return np.rint(np.clip(arr, 0, 65535)).astype(np.uint16)
    if output_dtype == "same":
        dtype = np.dtype(input_dtype)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return np.rint(np.clip(arr, info.min, info.max)).astype(dtype)
        return arr.astype(dtype)
    raise ValueError("--output-dtype must be same, uint16, uint8, or float32")


def cv2_median2d(frame: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return frame.astype(np.float32, copy=False)
    k = int(ksize)
    if k % 2 == 0:
        k += 1
    # OpenCV median is much faster for uint8/uint16; fall back to scipy otherwise.
    if frame.dtype in (np.dtype("uint8"), np.dtype("uint16")):
        return cv2.medianBlur(frame, k).astype(np.float32)
    return median_filter(frame.astype(np.float32), size=k, mode="nearest")


def denoise_volume(arr: np.ndarray, cfg: DenoiseConfig) -> np.ndarray:
    x = arr
    method = cfg.method

    if method == "none":
        out = x.astype(np.float32, copy=False)
    elif method in {"median2d", "gaussian2d", "median_gaussian2d"}:
        out = np.empty(x.shape, dtype=np.float32)
        for z in range(x.shape[0]):
            if z % max(1, x.shape[0] // 10) == 0:
                print(f"    slice {z}/{x.shape[0]-1}", flush=True)
            f: np.ndarray = x[z]
            if method in {"median2d", "median_gaussian2d"}:
                f = cv2_median2d(f, cfg.median_size)
            else:
                f = f.astype(np.float32, copy=False)
            if method in {"gaussian2d", "median_gaussian2d"} and cfg.sigma_xy > 0:
                f = cv2.GaussianBlur(f.astype(np.float32), (0, 0), float(cfg.sigma_xy))
            out[z] = f.astype(np.float32, copy=False)
    elif method == "gaussian3d":
        # Anisotropic 3D smoothing. Use small sigma_z because Z spacing is usually much larger than XY.
        out = gaussian_filter(x.astype(np.float32), sigma=(cfg.sigma_z, cfg.sigma_xy, cfg.sigma_xy), mode="nearest")
    else:
        raise ValueError(f"Unknown method: {method}")

    if cfg.intensity_floor > 0:
        out = out.copy()
        out[out < float(cfg.intensity_floor)] = 0
    out[out < 0] = 0
    return out.astype(np.float32, copy=False)


def make_foreground_bbox_for_position(files: list[Path], cfg: DenoiseConfig) -> dict[str, int] | None:
    if not cfg.crop_to_foreground:
        return None

    z0s: list[int] = []
    z1s: list[int] = []
    y0s: list[int] = []
    y1s: list[int] = []
    x0s: list[int] = []
    x1s: list[int] = []

    print("  estimating shared crop box from all timepoints", flush=True)
    for path in files:
        arr = read_volume(path)
        pos = arr[arr > 0]
        if pos.size < 100:
            continue
        thr = float(np.percentile(pos, cfg.mask_percentile))
        mask = arr > thr
        coords = np.argwhere(mask)
        if coords.shape[0] < cfg.min_size_voxels:
            continue
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0) + 1
        z0s.append(int(z0)); z1s.append(int(z1)); y0s.append(int(y0)); y1s.append(int(y1)); x0s.append(int(x0)); x1s.append(int(x1))
        del arr, mask, coords

    if not z0s:
        return None

    shape = read_volume(files[0]).shape
    z, y, x = shape
    return {
        "z0": max(0, min(z0s) - cfg.crop_padding_z),
        "z1": min(z, max(z1s) + cfg.crop_padding_z),
        "y0": max(0, min(y0s) - cfg.crop_padding_xy),
        "y1": min(y, max(y1s) + cfg.crop_padding_xy),
        "x0": max(0, min(x0s) - cfg.crop_padding_xy),
        "x1": min(x, max(x1s) + cfg.crop_padding_xy),
    }


def crop_volume(arr: np.ndarray, bbox: dict[str, int] | None) -> np.ndarray:
    if bbox is None:
        return arr
    return arr[bbox["z0"]:bbox["z1"], bbox["y0"]:bbox["y1"], bbox["x0"]:bbox["x1"]]


def process_file(path: Path, out_dir: Path, cfg: DenoiseConfig, bbox: dict[str, int] | None) -> dict[str, Any]:
    t0 = time.time()
    out_path = out_dir / path.name
    meta_path = out_dir / f"{path.stem}.denoise.json"
    if out_path.exists() and not cfg.overwrite:
        return {"input": str(path), "output": str(out_path), "status": "skipped_exists"}

    in_bytes = path.stat().st_size if path.exists() else None
    print(f"  reading {path.name}", flush=True)
    arr = read_volume(path)
    input_shape = tuple(int(v) for v in arr.shape)
    input_dtype = str(arr.dtype)
    print(f"  denoising {path.name}: shape={input_shape} dtype={input_dtype}", flush=True)
    den = denoise_volume(arr, cfg)
    output_float_shape = tuple(int(v) for v in den.shape)
    den = crop_volume(den, bbox)
    out = cast_output(den, arr.dtype, cfg.output_dtype)
    print(f"  writing {out_path}", flush=True)
    save_volume(out_path, out, cfg.compression)
    out_bytes = out_path.stat().st_size if out_path.exists() else None

    result = {
        "input": str(path),
        "output": str(out_path),
        "status": "ok",
        "method": cfg.method,
        "input_shape_zyx": list(input_shape),
        "denoised_shape_before_crop_zyx": list(output_float_shape),
        "output_shape_zyx": list(out.shape),
        "input_dtype": input_dtype,
        "output_dtype": str(out.dtype),
        "median_size": cfg.median_size,
        "sigma_xy": cfg.sigma_xy,
        "sigma_z": cfg.sigma_z,
        "intensity_floor": cfg.intensity_floor,
        "bbox_zyx": bbox,
        "compression": cfg.compression,
        "input_bytes": in_bytes,
        "output_bytes": out_bytes,
        "size_ratio_output_vs_input": float(out_bytes / in_bytes) if in_bytes and out_bytes else None,
        "elapsed_sec": time.time() - t0,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch denoise 3D light-sheet TIFF stacks.")
    p.add_argument("--input-root", required=True, help="Folder containing position folders, or TIFFs for one position.")
    p.add_argument("--output-root", required=True)
    p.add_argument("--glob-pattern", default="*.tif*")
    p.add_argument("--method", default="median_gaussian2d", choices=["none", "median2d", "gaussian2d", "median_gaussian2d", "gaussian3d"])
    p.add_argument("--median-size", type=int, default=3, help="2D median kernel size. Use 3 for mild despeckling.")
    p.add_argument("--sigma-xy", type=float, default=0.6, help="Gaussian sigma in XY pixels. Use 0.3-0.8 for mild denoising.")
    p.add_argument("--sigma-z", type=float, default=0.0, help="Only used for gaussian3d. Keep small because Z spacing is coarse.")
    p.add_argument("--intensity-floor", type=float, default=5.0, help="Set intensities below this to 0 to reduce residual background and file size.")
    p.add_argument("--output-dtype", default="same", choices=["same", "uint16", "uint8", "float32"])
    p.add_argument("--compression", default="zlib", help="tifffile compression: zlib, lzw, zstd, or none.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--crop-to-foreground", action="store_true", help="Optional extra shared crop per position. Usually not needed after BGS.")
    p.add_argument("--crop-padding-xy", type=int, default=40)
    p.add_argument("--crop-padding-z", type=int, default=4)
    p.add_argument("--mask-percentile", type=float, default=70.0)
    p.add_argument("--min-size-voxels", type=int, default=10000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    compression = None if str(args.compression).lower() in {"none", "no", "false", "0"} else args.compression
    cfg = DenoiseConfig(
        method=args.method,
        median_size=max(1, int(args.median_size)),
        sigma_xy=max(0.0, float(args.sigma_xy)),
        sigma_z=max(0.0, float(args.sigma_z)),
        intensity_floor=max(0.0, float(args.intensity_floor)),
        output_dtype=args.output_dtype,
        compression=compression,
        overwrite=args.overwrite,
        crop_to_foreground=args.crop_to_foreground,
        crop_padding_xy=max(0, int(args.crop_padding_xy)),
        crop_padding_z=max(0, int(args.crop_padding_z)),
        mask_percentile=float(args.mask_percentile),
        min_size_voxels=max(1, int(args.min_size_voxels)),
    )

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    positions = discover_positions(input_root, args.glob_pattern)
    if not positions:
        raise SystemExit(f"No TIFF files found under {input_root} with pattern {args.glob_pattern!r}")

    print(f"Discovered {len(positions)} position(s):")
    for pos, files in positions.items():
        print(f"  {pos}: {len(files)} TIFF file(s); first={files[0].name}")
    print(f"Method={cfg.method}; median={cfg.median_size}; sigma_xy={cfg.sigma_xy}; floor={cfg.intensity_floor}; compression={cfg.compression}")

    if args.dry_run:
        return

    all_results: list[dict[str, Any]] = []
    t_start = time.time()
    for position, files in positions.items():
        print(f"\n[position {position}]", flush=True)
        out_dir = output_root / position
        out_dir.mkdir(parents=True, exist_ok=True)
        bbox = make_foreground_bbox_for_position(files, cfg)
        if bbox is not None:
            print(f"  shared crop bbox: {bbox}", flush=True)
        for i, path in enumerate(files, start=1):
            print(f"  file {i}/{len(files)}: {path.name}", flush=True)
            try:
                res = process_file(path, out_dir, cfg, bbox)
            except Exception as exc:
                res = {"input": str(path), "status": "error", "error": str(exc), "position": position, "timepoint": path.stem}
            res["position"] = position
            res["timepoint"] = path.stem
            all_results.append(res)
            print(json.dumps({k: v for k, v in res.items() if k not in {"bbox_zyx"}}, default=str), flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "denoising_summary.csv"
    write_summary_csv(summary_csv, all_results)
    manifest = {
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "glob_pattern": args.glob_pattern,
        "positions": {p: [str(f.resolve()) for f in files] for p, files in positions.items()},
        "settings": cfg.__dict__,
        "summary_csv": str(summary_csv),
        "results": all_results,
        "elapsed_sec": time.time() - t_start,
    }
    with (output_root / "denoising_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_csv}")
    print(f"Saved manifest: {output_root / 'denoising_manifest.json'}")


if __name__ == "__main__":
    main()
