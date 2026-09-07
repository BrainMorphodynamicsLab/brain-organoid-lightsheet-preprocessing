#!/usr/bin/env python3
"""
Fast 3D drift correction for light-sheet organoid TIFF stacks.

Data model
----------
- One TIFF = one 3D volume, array order (Z, Y, X).
- One folder/position = one organoid/replicate.
- Many TIFFs inside one position folder = timepoints for that organoid.
- Different positions/organoids are processed independently and are NOT globally registered.

Why this script exists
----------------------
Elastix/transformix is accurate but can be very slow on >2 GB stacks, especially
when the full-resolution volume is resampled and written with compression. For
longitudinal organoid drift correction, a much faster and often sufficient
strategy is to estimate one 3D translation from a downsampled copy and apply the
same 3D translation to the entire full-resolution stack.

This script follows the practical idea used in large light-sheet pipelines:
use the organoid signal/bounding information at reduced resolution to estimate
drift, then apply one transform consistently to all Z slices of each timepoint.
It does not register slices independently.

Outputs
-------
For every input TIFF, writes one registered TIFF to:
  <output-root>/<position>/<same filename>.tif

Also writes:
  <output-root>/fast_registration_summary.csv
  <output-root>/<position>/<filename>.json
  <output-root>/registration_manifest.json

Recommended first test
----------------------
python3 script/registration.py \
  --input-root original \
  --output-root result/registered_fast_xy4 \
  --ome-xml original/position_1/ome-tiff.companion.ome \
  --glob-pattern "*.tif*" \
  --downsample-xy 4 \
  --downsample-z 1 \
  --estimate-method center_phase_mip \
  --apply-mode integer \
  --compression none \
  --overwrite

Then validate with your existing validate_3d_registration.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import exposure
from skimage.registration import phase_cross_correlation
from skimage.transform import downscale_local_mean

_NATURAL_SORT_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class RunConfig:
    spacing_x: float
    spacing_y: float
    spacing_z: float
    padding_z: int
    padding_y: int
    padding_x: int
    downsample_xy: int
    downsample_z: int
    clip_low: float
    clip_high: float
    center_threshold_percentile: float
    estimate_method: str
    apply_mode: str
    phase_upsample: int
    series_index: int
    channel_index: int | None
    overwrite: bool
    compression: str | None
    save_dtype: str
    max_shift_z: float | None
    max_shift_yx: float | None


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    return tuple(int(tok) if tok.isdigit() else tok.lower() for tok in _NATURAL_SORT_RE.split(path.name))


def parse_ome_spacing(ome_path: str) -> tuple[float, float, float] | None:
    tree = ET.parse(ome_path)
    root = tree.getroot()
    pixels = None
    for elem in root.iter():
        if elem.tag.rsplit("}", 1)[-1] == "Pixels":
            pixels = elem
            break
    if pixels is None:
        return None
    try:
        return (
            float(pixels.attrib["PhysicalSizeX"]),
            float(pixels.attrib["PhysicalSizeY"]),
            float(pixels.attrib["PhysicalSizeZ"]),
        )
    except KeyError:
        return None


def discover_positions(input_root: Path, glob_pattern: str) -> dict[str, list[Path]]:
    """Find TIFF timepoints. If input_root has subfolders, each subfolder is a position.

    Only files matching glob_pattern are returned. With glob-pattern '*.tif*', OME XML and AVI
    files are ignored because they do not match the TIFF extension pattern.
    """
    subdirs = sorted((d for d in input_root.iterdir() if d.is_dir() and not d.name.startswith(".")), key=lambda d: d.name)
    positions: dict[str, list[Path]] = {}

    for subdir in subdirs:
        files = sorted((p for p in subdir.glob(glob_pattern) if p.is_file()), key=natural_sort_key)
        if files:
            positions[subdir.name] = files

    if positions:
        loose = sorted((p for p in input_root.glob(glob_pattern) if p.is_file()), key=natural_sort_key)
        if loose:
            print(
                f"Warning: {len(loose)} TIFF(s) directly under {input_root} are ignored because position subfolders were found."
            )
        return positions

    files = sorted((p for p in input_root.glob(glob_pattern) if p.is_file()), key=natural_sort_key)
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
    return all(tuple(page.shape) == first_shape and np.dtype(page.dtype) == first_dtype for page in tf.pages)


def _read_tiff_pages_as_zyx(tf: tifffile.TiffFile, source: str) -> np.ndarray:
    if not _can_stack_tiff_pages_as_z(tf):
        raise ValueError(f"{source}: pages are not uniform 2D planes; cannot stack as ZYX")
    n_z = len(tf.pages)
    y, x = tf.pages[0].shape
    dtype = np.dtype(tf.pages[0].dtype)
    arr = np.empty((n_z, y, x), dtype=dtype)
    for z, page in enumerate(tf.pages):
        arr[z] = page.asarray()
    return arr


def _normalize_axes_to_zyx(arr: np.ndarray, axes: str, *, channel_index: int | None, source: str) -> np.ndarray:
    axes_list = list(axes)
    out = arr
    for axis_i in range(len(axes_list) - 1, -1, -1):
        ax = axes_list[axis_i]
        size = out.shape[axis_i]
        if ax in {"Z", "Y", "X"}:
            continue
        if ax == "C" and channel_index is not None:
            if not 0 <= channel_index < size:
                raise ValueError(f"{source}: channel_index={channel_index} outside C size {size}")
            out = np.take(out, channel_index, axis=axis_i)
            axes_list.pop(axis_i)
            continue
        if size == 1:
            out = np.take(out, 0, axis=axis_i)
            axes_list.pop(axis_i)
            continue
        raise ValueError(
            f"{source}: non-spatial axis {ax!r} with size {size}. "
            "Pass --channel-index for multi-channel TIFFs, or export one single-channel RFP stack."
        )

    axes_now = "".join(axes_list)
    if out.ndim != 3:
        raise ValueError(f"{source}: expected one 3D stack after axis cleanup, got {out.shape}, axes {axes_now!r}")
    if axes_now == "ZYX":
        return np.ascontiguousarray(out)
    if len(axes_now) == 3 and axes_now[-2:] == "YX":
        return np.ascontiguousarray(out)
    if set(axes_now) == {"Z", "Y", "X"}:
        order = [axes_now.index(ax) for ax in "ZYX"]
        return np.ascontiguousarray(np.transpose(out, order))
    raise ValueError(f"{source}: cannot interpret axes {axes_now!r} as ZYX for shape {out.shape}")


def read_volume(path: str, cfg: RunConfig) -> np.ndarray:
    with tifffile.TiffFile(path) as tf:
        if cfg.series_index >= len(tf.series):
            raise ValueError(f"{path}: --series-index {cfg.series_index} but file has only {len(tf.series)} series")
        series = tf.series[cfg.series_index]
        arr = series.asarray()
        axes = getattr(series, "axes", "") or ""
        if arr.ndim == 2 and axes == "YX" and _can_stack_tiff_pages_as_z(tf):
            if cfg.channel_index is not None:
                raise ValueError(f"{path}: plain page-stacked single-channel TIFF; remove --channel-index")
            arr = _read_tiff_pages_as_zyx(tf, path)
            axes = "ZYX"
    return _normalize_axes_to_zyx(arr, axes, channel_index=cfg.channel_index, source=path)


def probe_tiff_shape(path: str, cfg: RunConfig) -> tuple[tuple[int, int, int], np.dtype]:
    with tifffile.TiffFile(path) as tf:
        if cfg.series_index >= len(tf.series):
            raise ValueError(f"{path}: --series-index {cfg.series_index} but file has only {len(tf.series)} series")
        series = tf.series[cfg.series_index]
        axes = getattr(series, "axes", "") or ""
        if len(series.shape) == 2 and axes == "YX" and _can_stack_tiff_pages_as_z(tf):
            y, x = tf.pages[0].shape
            return (len(tf.pages), int(y), int(x)), np.dtype(tf.pages[0].dtype)
    arr = read_volume(path, cfg)
    return tuple(int(v) for v in arr.shape), arr.dtype


def pad_volume(arr: np.ndarray, cfg: RunConfig) -> np.ndarray:
    if cfg.padding_z == 0 and cfg.padding_y == 0 and cfg.padding_x == 0:
        return arr
    return np.pad(
        arr,
        ((cfg.padding_z, cfg.padding_z), (cfg.padding_y, cfg.padding_y), (cfg.padding_x, cfg.padding_x)),
        mode="constant",
    )


def downsample_volume(arr: np.ndarray, cfg: RunConfig) -> np.ndarray:
    if cfg.downsample_z <= 1 and cfg.downsample_xy <= 1:
        return arr.astype(np.float32, copy=False)
    return downscale_local_mean(arr, (cfg.downsample_z, cfg.downsample_xy, cfg.downsample_xy)).astype(np.float32, copy=False)


def robust_rescale(arr: np.ndarray, cfg: RunConfig) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    vmin, vmax = np.percentile(arr, (cfg.clip_low, cfg.clip_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return arr
    return exposure.rescale_intensity(arr, in_range=(vmin, vmax), out_range=np.float32).astype(np.float32, copy=False)


def foreground_weighted_center(vol: np.ndarray, threshold_percentile: float) -> tuple[float, float, float]:
    """Memory-safe intensity-weighted center on a thresholded volume, returned as (z,y,x)."""
    arr = vol.astype(np.float32, copy=False)
    thresh = np.percentile(arr, threshold_percentile)
    w = arr - thresh
    w[w < 0] = 0
    total = float(w.sum(dtype=np.float64))
    if total <= 0 or not np.isfinite(total):
        # Fallback to geometric center of the downsampled image.
        return tuple((np.array(arr.shape, dtype=float) - 1.0) / 2.0)  # type: ignore[return-value]

    z_profile = w.sum(axis=(1, 2), dtype=np.float64)
    y_profile = w.sum(axis=(0, 2), dtype=np.float64)
    x_profile = w.sum(axis=(0, 1), dtype=np.float64)
    cz = float(np.dot(np.arange(arr.shape[0], dtype=np.float64), z_profile) / total)
    cy = float(np.dot(np.arange(arr.shape[1], dtype=np.float64), y_profile) / total)
    cx = float(np.dot(np.arange(arr.shape[2], dtype=np.float64), x_profile) / total)
    return cz, cy, cx


def mip_phase_refine(ref: np.ndarray, mov: np.ndarray, max_refine_ds: float, upsample: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Fast MIP-based phase refinement, returned in downsampled ZYX pixels."""
    # XY estimates Y/X
    ref_xy = ref.max(axis=0)
    mov_xy = mov.max(axis=0)
    shift_yx, error_xy, _ = phase_cross_correlation(ref_xy, mov_xy, upsample_factor=upsample, normalization=None)

    # ZY estimates Z/Y; ZX estimates Z/X. Average the Z estimates.
    ref_zy = ref.max(axis=2)
    mov_zy = mov.max(axis=2)
    shift_zy, error_zy, _ = phase_cross_correlation(ref_zy, mov_zy, upsample_factor=upsample, normalization=None)

    ref_zx = ref.max(axis=1)
    mov_zx = mov.max(axis=1)
    shift_zx, error_zx, _ = phase_cross_correlation(ref_zx, mov_zx, upsample_factor=upsample, normalization=None)

    dz = float((shift_zy[0] + shift_zx[0]) / 2.0)
    dy = float((shift_yx[0] + shift_zy[1]) / 2.0)
    dx = float((shift_yx[1] + shift_zx[1]) / 2.0)
    shift = np.asarray([dz, dy, dx], dtype=np.float64)

    # Phase on biological data can occasionally jump to a wrong periodic solution.
    # Clamp this refinement; the center-of-signal shift remains the safe coarse estimate.
    norm = float(np.linalg.norm(shift))
    if max_refine_ds is not None and norm > max_refine_ds:
        shift = shift * (max_refine_ds / norm)

    details = {
        "mip_phase_shift_zyx_ds": shift.tolist(),
        "phase_error_xy": float(error_xy),
        "phase_error_zy": float(error_zy),
        "phase_error_zx": float(error_zx),
    }
    return shift, details


def estimate_shift_ds(ref_small: np.ndarray, mov_small: np.ndarray, cfg: RunConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate shift to apply to moving image, in downsampled ZYX pixels."""
    ref_reg = robust_rescale(ref_small, cfg)
    mov_reg = robust_rescale(mov_small, cfg)

    ref_center = np.asarray(foreground_weighted_center(ref_reg, cfg.center_threshold_percentile), dtype=np.float64)
    mov_center = np.asarray(foreground_weighted_center(mov_reg, cfg.center_threshold_percentile), dtype=np.float64)
    center_shift = ref_center - mov_center

    details: dict[str, Any] = {
        "reference_center_zyx_ds": ref_center.tolist(),
        "moving_center_zyx_ds": mov_center.tolist(),
        "center_shift_zyx_ds": center_shift.tolist(),
    }

    if cfg.estimate_method == "center":
        return center_shift, details

    if cfg.estimate_method == "phase_mip":
        phase_shift, phase_details = mip_phase_refine(ref_reg, mov_reg, max_refine_ds=1e9, upsample=cfg.phase_upsample)
        details.update(phase_details)
        return phase_shift, details

    if cfg.estimate_method == "center_phase_mip":
        # Move the downsampled moving volume approximately by center, then use MIP phase only for small residuals.
        rough = ndi.shift(mov_reg, shift=center_shift, order=1, mode="constant", cval=0.0, prefilter=False)
        refine, phase_details = mip_phase_refine(ref_reg, rough, max_refine_ds=8.0, upsample=cfg.phase_upsample)
        total = center_shift + refine
        details.update(phase_details)
        details["phase_refine_after_center_zyx_ds"] = refine.tolist()
        return total, details

    raise ValueError(f"Unknown estimate method: {cfg.estimate_method}")


def clamp_shift_full(shift: np.ndarray, cfg: RunConfig) -> np.ndarray:
    out = shift.astype(float).copy()
    if cfg.max_shift_z is not None:
        out[0] = float(np.clip(out[0], -cfg.max_shift_z, cfg.max_shift_z))
    if cfg.max_shift_yx is not None:
        out[1] = float(np.clip(out[1], -cfg.max_shift_yx, cfg.max_shift_yx))
        out[2] = float(np.clip(out[2], -cfg.max_shift_yx, cfg.max_shift_yx))
    return out


def integer_translate_zyx(arr: np.ndarray, shift_zyx: tuple[int, int, int]) -> np.ndarray:
    """Fast integer translation with constant-zero fill, preserving shape and dtype."""
    dz, dy, dx = shift_zyx
    out = np.zeros_like(arr)
    z, y, x = arr.shape

    src_z0 = max(0, -dz)
    src_z1 = min(z, z - dz) if dz >= 0 else z
    dst_z0 = max(0, dz)
    dst_z1 = dst_z0 + max(0, src_z1 - src_z0)

    src_y0 = max(0, -dy)
    src_y1 = min(y, y - dy) if dy >= 0 else y
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)

    src_x0 = max(0, -dx)
    src_x1 = min(x, x - dx) if dx >= 0 else x
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)

    if src_z1 > src_z0 and src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
    return out


def cast_to_dtype(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.rint(np.clip(arr, info.min, info.max)).astype(dtype, copy=False)
    return arr.astype(dtype, copy=False)


def apply_shift_fullres(arr: np.ndarray, shift_full_zyx: np.ndarray, cfg: RunConfig) -> tuple[np.ndarray, list[float]]:
    if cfg.apply_mode == "integer":
        shift_int = tuple(int(round(v)) for v in shift_full_zyx)
        return integer_translate_zyx(arr, shift_int), [float(v) for v in shift_int]

    if cfg.apply_mode == "subpixel":
        original_dtype = arr.dtype
        shifted = ndi.shift(
            arr.astype(np.float32, copy=False),
            shift=tuple(float(v) for v in shift_full_zyx),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        return cast_to_dtype(shifted, original_dtype), [float(v) for v in shift_full_zyx]

    raise ValueError(f"Unknown apply mode: {cfg.apply_mode}")


def save_tiff(path: Path, arr: np.ndarray, cfg: RunConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.save_dtype != "input":
        arr = cast_to_dtype(arr, np.dtype(cfg.save_dtype))
    tifffile.imwrite(path, arr, bigtiff=True, compression=cfg.compression, metadata=None)


def shift_to_um(shift_full_zyx: np.ndarray, cfg: RunConfig) -> list[float]:
    # Returned as z,y,x microns.
    return [
        float(shift_full_zyx[0] * cfg.spacing_z),
        float(shift_full_zyx[1] * cfg.spacing_y),
        float(shift_full_zyx[2] * cfg.spacing_x),
    ]


def process_position(position: str, files: list[Path], output_root: Path, cfg: RunConfig) -> list[dict[str, Any]]:
    out_dir = output_root / position
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    reference_path = files[0]
    print(f"[{position}] reading reference: {reference_path.name}", flush=True)
    t0 = time.time()
    ref_full = pad_volume(read_volume(str(reference_path), cfg), cfg)
    ref_small = downsample_volume(ref_full, cfg)
    ref_out = out_dir / reference_path.name
    if cfg.overwrite or not ref_out.exists():
        print(f"[{position}] writing padded reference: {ref_out}", flush=True)
        save_tiff(ref_out, ref_full, cfg)
    ref_shape = tuple(int(v) for v in ref_full.shape)
    ref_dtype = str(ref_full.dtype)

    ref_meta = {
        "position": position,
        "timepoint": reference_path.stem,
        "input": str(reference_path.resolve()),
        "output": str(ref_out),
        "status": "reference",
        "shape_zyx": list(ref_shape),
        "dtype": ref_dtype,
        "spacing_xyz": [cfg.spacing_x, cfg.spacing_y, cfg.spacing_z],
        "padding_zyx": [cfg.padding_z, cfg.padding_y, cfg.padding_x],
        "downsample_zyx": [cfg.downsample_z, cfg.downsample_xy, cfg.downsample_xy],
        "elapsed_sec": time.time() - t0,
    }
    (out_dir / f"{reference_path.stem}.json").write_text(json.dumps(ref_meta, indent=2), encoding="utf-8")
    rows.append({
        "position": position,
        "timepoint": reference_path.stem,
        "input": str(reference_path),
        "output": str(ref_out),
        "status": "reference",
        "shift_z_px": 0,
        "shift_y_px": 0,
        "shift_x_px": 0,
        "shift_z_um": 0,
        "shift_y_um": 0,
        "shift_x_um": 0,
        "elapsed_sec": ref_meta["elapsed_sec"],
    })

    for idx, moving_path in enumerate(files[1:], start=2):
        out_path = out_dir / moving_path.name
        meta_path = out_dir / f"{moving_path.stem}.json"
        if out_path.exists() and not cfg.overwrite:
            print(f"[{position}] {idx}/{len(files)} skip existing: {moving_path.name}", flush=True)
            rows.append({"position": position, "timepoint": moving_path.stem, "input": str(moving_path), "output": str(out_path), "status": "skipped_exists"})
            continue

        print(f"[{position}] {idx}/{len(files)} reading moving: {moving_path.name}", flush=True)
        t_start = time.time()
        mov_full = pad_volume(read_volume(str(moving_path), cfg), cfg)
        print(f"[{position}] {moving_path.name}: downsampling for shift estimate", flush=True)
        mov_small = downsample_volume(mov_full, cfg)

        print(f"[{position}] {moving_path.name}: estimating shift ({cfg.estimate_method})", flush=True)
        shift_ds, details = estimate_shift_ds(ref_small, mov_small, cfg)
        del mov_small
        shift_full = np.asarray(
            [shift_ds[0] * cfg.downsample_z, shift_ds[1] * cfg.downsample_xy, shift_ds[2] * cfg.downsample_xy],
            dtype=np.float64,
        )
        unclamped_shift_full = shift_full.copy()
        shift_full = clamp_shift_full(shift_full, cfg)

        print(
            f"[{position}] {moving_path.name}: applying {cfg.apply_mode} shift ZYX px "
            f"{shift_full[0]:.2f}, {shift_full[1]:.2f}, {shift_full[2]:.2f}",
            flush=True,
        )
        registered, applied_shift = apply_shift_fullres(mov_full, shift_full, cfg)
        del mov_full

        print(f"[{position}] {moving_path.name}: writing output", flush=True)
        save_tiff(out_path, registered, cfg)
        del registered

        elapsed = time.time() - t_start
        shift_um = shift_to_um(np.asarray(applied_shift, dtype=float), cfg)
        meta = {
            "position": position,
            "timepoint": moving_path.stem,
            "input": str(moving_path.resolve()),
            "reference": str(reference_path.resolve()),
            "output": str(out_path),
            "status": "ok",
            "method": cfg.estimate_method,
            "apply_mode": cfg.apply_mode,
            "shape_zyx": list(ref_shape),
            "dtype": ref_dtype,
            "spacing_xyz": [cfg.spacing_x, cfg.spacing_y, cfg.spacing_z],
            "padding_zyx": [cfg.padding_z, cfg.padding_y, cfg.padding_x],
            "downsample_zyx": [cfg.downsample_z, cfg.downsample_xy, cfg.downsample_xy],
            "estimated_shift_zyx_ds": [float(v) for v in shift_ds],
            "unclamped_shift_zyx_full_px": [float(v) for v in unclamped_shift_full],
            "applied_shift_zyx_full_px": [float(v) for v in applied_shift],
            "applied_shift_zyx_um": shift_um,
            "details": details,
            "elapsed_sec": elapsed,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append({
            "position": position,
            "timepoint": moving_path.stem,
            "input": str(moving_path),
            "output": str(out_path),
            "status": "ok",
            "shift_z_px": applied_shift[0],
            "shift_y_px": applied_shift[1],
            "shift_x_px": applied_shift[2],
            "shift_z_um": shift_um[0],
            "shift_y_um": shift_um[1],
            "shift_x_um": shift_um[2],
            "elapsed_sec": elapsed,
        })
        print(f"[{position}] {moving_path.name}: done in {elapsed/60:.1f} min", flush=True)

    del ref_full, ref_small
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", required=True, help="Folder containing position folders, or TIFFs for one organoid.")
    p.add_argument("--output-root", required=True)
    p.add_argument("--glob-pattern", default="*.tif*", help="TIFF pattern; '*.tif*' ignores OME XML and AVI files.")
    p.add_argument("--positions", nargs="*", default=None, help="Optional subset of position folder names to process.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--ome-xml", default=None)
    p.add_argument("--spacing-x", type=float, default=None)
    p.add_argument("--spacing-y", type=float, default=None)
    p.add_argument("--spacing-z", type=float, default=None)
    p.add_argument("--series-index", type=int, default=0)
    p.add_argument("--channel-index", type=int, default=None)

    p.add_argument("--padding-z", type=int, default=8)
    p.add_argument("--padding-y", type=int, default=64)
    p.add_argument("--padding-x", type=int, default=64)
    p.add_argument("--downsample-xy", type=int, default=8, help="Bigger = faster, less precise. Try 8 first; 4 for final.")
    p.add_argument("--downsample-z", type=int, default=2, help="Try 2 for speed; use 1 if Z drift is important.")
    p.add_argument("--clip-low", type=float, default=1.0)
    p.add_argument("--clip-high", type=float, default=99.0)
    p.add_argument("--center-threshold-percentile", type=float, default=75.0)
    p.add_argument(
        "--estimate-method",
        choices=["center", "phase_mip", "center_phase_mip"],
        default="center_phase_mip",
        help="center is fastest; center_phase_mip is still fast and usually more accurate; phase_mip can be less stable.",
    )
    p.add_argument("--phase-upsample", type=int, default=1, help="1 is fastest; 2-4 gives subpixel MIP refinement.")
    p.add_argument("--apply-mode", choices=["integer", "subpixel"], default="integer", help="integer is much faster and memory-safe.")
    p.add_argument("--max-shift-z", type=float, default=None, help="Optional clamp of full-res Z shift in pixels.")
    p.add_argument("--max-shift-yx", type=float, default=None, help="Optional clamp of full-res Y/X shift in pixels.")
    p.add_argument("--compression", default="none", help="none is fastest; zlib/lzw smaller but much slower.")
    p.add_argument("--save-dtype", default="input", help="input, uint16, uint8, float32, etc.")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    sx, sy, sz = args.spacing_x, args.spacing_y, args.spacing_z
    if args.ome_xml and (sx is None or sy is None or sz is None):
        parsed = parse_ome_spacing(args.ome_xml)
        if parsed is not None:
            px, py, pz = parsed
            sx = px if sx is None else sx
            sy = py if sy is None else sy
            sz = pz if sz is None else sz
            print(f"Using voxel spacing from OME: x={sx}, y={sy}, z={sz}")
        else:
            print(f"Warning: could not parse OME spacing from {args.ome_xml}; using voxel-unit spacing for missing values.")
    sx = 1.0 if sx is None else float(sx)
    sy = 1.0 if sy is None else float(sy)
    sz = 1.0 if sz is None else float(sz)
    compression = None if str(args.compression).lower() in {"none", "no", "false", "0"} else args.compression
    return RunConfig(
        spacing_x=sx,
        spacing_y=sy,
        spacing_z=sz,
        padding_z=max(0, args.padding_z),
        padding_y=max(0, args.padding_y),
        padding_x=max(0, args.padding_x),
        downsample_xy=max(1, args.downsample_xy),
        downsample_z=max(1, args.downsample_z),
        clip_low=float(args.clip_low),
        clip_high=float(args.clip_high),
        center_threshold_percentile=float(args.center_threshold_percentile),
        estimate_method=args.estimate_method,
        apply_mode=args.apply_mode,
        phase_upsample=max(1, args.phase_upsample),
        series_index=max(0, args.series_index),
        channel_index=args.channel_index,
        overwrite=args.overwrite,
        compression=compression,
        save_dtype=args.save_dtype,
        max_shift_z=args.max_shift_z,
        max_shift_yx=args.max_shift_yx,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    positions = discover_positions(input_root, args.glob_pattern)
    if args.positions:
        wanted = set(args.positions)
        positions = {k: v for k, v in positions.items() if k in wanted}
    if not positions:
        raise SystemExit(f"No TIFF files found under {input_root} with pattern {args.glob_pattern!r}")

    print(f"Discovered {len(positions)} position/organoid folder(s):")
    for pos, files in positions.items():
        shape, dtype = probe_tiff_shape(str(files[0]), cfg)
        print(f"  {pos}: {len(files)} TIFF timepoint(s), reference={files[0].name}, shape_zyx={shape}, dtype={dtype}")
    print("Different positions/organoids are processed independently; no global cross-organoid registration is performed.")
    print(f"Fast mode: estimate={cfg.estimate_method}, apply={cfg.apply_mode}, downsample_zyx=({cfg.downsample_z},{cfg.downsample_xy},{cfg.downsample_xy}), compression={cfg.compression}")

    if args.dry_run:
        for pos, files in positions.items():
            print(f"[{pos}] reference: {files[0].name}")
            for f in files[1:]:
                print(f"[{pos}] moving:   {f.name}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    t_all = time.time()
    for pos, files in positions.items():
        all_rows.extend(process_position(pos, files, output_root, cfg))

    summary_path = output_root / "fast_registration_summary.csv"
    if all_rows:
        fieldnames = sorted({k for row in all_rows for k in row.keys()})
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote summary CSV: {summary_path}")

    manifest = {
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "glob_pattern": args.glob_pattern,
        "positions": {p: [str(f.resolve()) for f in files] for p, files in positions.items()},
        "config": {
            "spacing_xyz": [cfg.spacing_x, cfg.spacing_y, cfg.spacing_z],
            "padding_zyx": [cfg.padding_z, cfg.padding_y, cfg.padding_x],
            "downsample_zyx": [cfg.downsample_z, cfg.downsample_xy, cfg.downsample_xy],
            "estimate_method": cfg.estimate_method,
            "apply_mode": cfg.apply_mode,
            "compression": cfg.compression,
            "global_registration": False,
        },
        "summary_csv": str(summary_path),
        "elapsed_sec": time.time() - t_all,
    }
    manifest_path = output_root / "registration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    print(f"All done in {(time.time() - t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
