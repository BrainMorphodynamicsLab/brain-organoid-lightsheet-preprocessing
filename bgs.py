#!/usr/bin/env python3
"""
Batch 3D background subtraction for registered light-sheet TIFF Z-stacks.

Data model
----------
- One folder = one organoid / position / replicate.
- One TIFF = one timepoint = one full 3D Z-stack with array order (Z, Y, X).
- Background subtraction is applied to every Z slice in every TIFF.
- Different organoids are processed independently.

Main method: zmin_smooth
------------------------
For each 3D TIFF, estimate a smooth XY background image from the minimum/low
percentile along Z, then subtract that same smooth background from every Z
slice. This follows the LSTree-style idea that a Z-minimum projection can
estimate background when each XY pixel sees background in at least one Z plane.

Size reduction
--------------
Optionally computes one consistent foreground crop per organoid from all
registered timepoints, then writes only that crop for every timepoint. This can
make files much smaller while keeping all timepoints in the same coordinate
frame for that organoid.

Typical use after registration
------------------------------
python3 script/bgs.py \
  --input-root result/registered/position_1 \
  --output-root result/bgsub \
  --glob-pattern "*.tif*" \
  --method zmin_smooth \
  --crop-to-foreground \
  --crop-padding-xy 80 \
  --crop-padding-z 8 \
  --bg-sigma 80 \
  --bg-alpha 1.0 \
  --auto-offset \
  --compression zlib \
  --overwrite

python3 script/bgs.py \
  --input-root result/registered/position_1 \
  --output-root result/bgsub_alpha105 \
  --glob-pattern "*.tif*" \
  --method zmin_smooth \
  --crop-to-foreground \
  --crop-padding-xy 80 \
  --crop-padding-z 8 \
  --bg-sigma 80 \
  --bg-alpha 1.05 \
  --auto-offset \
  --compression zlib \
  --overwrite
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
from typing import Any, Iterator

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi

_NATURAL_SORT_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class BBoxZYX:
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    def as_slices(self) -> tuple[slice, slice, slice]:
        return (slice(self.z0, self.z1), slice(self.y0, self.y1), slice(self.x0, self.x1))

    def shape(self) -> tuple[int, int, int]:
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)

    def to_dict(self) -> dict[str, int]:
        return {"z0": self.z0, "z1": self.z1, "y0": self.y0, "y1": self.y1, "x0": self.x0, "x1": self.x1}

    @staticmethod
    def full(shape_zyx: tuple[int, int, int]) -> "BBoxZYX":
        z, y, x = shape_zyx
        return BBoxZYX(0, z, 0, y, 0, x)


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    return tuple(int(tok) if tok.isdigit() else tok.lower() for tok in _NATURAL_SORT_RE.split(path.name))


def parse_ome_spacing(ome_path: str | None) -> tuple[float, float, float] | None:
    if not ome_path:
        return None
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
    subdirs = sorted((d for d in input_root.iterdir() if d.is_dir() and not d.name.startswith(".")), key=lambda p: p.name)
    positions: dict[str, list[Path]] = {}
    for subdir in subdirs:
        files = sorted((p for p in subdir.glob(glob_pattern) if p.is_file()), key=natural_sort_key)
        if files:
            positions[subdir.name] = files
    if positions:
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
    return all(tuple(p.shape) == first_shape and np.dtype(p.dtype) == first_dtype for p in tf.pages)


def tiff_shape_dtype(path: str | Path) -> tuple[tuple[int, int, int], np.dtype, str]:
    path = Path(path)
    with tifffile.TiffFile(str(path)) as tf:
        if _can_stack_tiff_pages_as_z(tf):
            y, x = tf.pages[0].shape
            return (len(tf.pages), int(y), int(x)), np.dtype(tf.pages[0].dtype), "pages"
        series = tf.series[0]
        arr_shape = tuple(int(v) for v in series.shape)
        axes = getattr(series, "axes", "") or ""
        dtype = np.dtype(series.dtype)
        # Drop singleton non-spatial axes.
        shape = list(arr_shape)
        ax_list = list(axes)
        for i in range(len(ax_list) - 1, -1, -1):
            if ax_list[i] in {"Z", "Y", "X"}:
                continue
            if shape[i] == 1:
                del shape[i]
                del ax_list[i]
        axes_now = "".join(ax_list)
        if len(shape) == 3 and axes_now == "ZYX":
            return (shape[0], shape[1], shape[2]), dtype, "series"
        if len(shape) == 3 and set(axes_now) == {"Z", "Y", "X"}:
            return tuple(shape[axes_now.index(ax)] for ax in "ZYX"), dtype, "series"
        if len(shape) == 3 and axes_now[-2:] == "YX":
            return (shape[0], shape[1], shape[2]), dtype, "series"
    arr = tifffile.imread(str(path))
    arr = normalize_to_zyx(arr, source=str(path))
    return tuple(int(v) for v in arr.shape), arr.dtype, "array"


def normalize_to_zyx(arr: np.ndarray, source: str = "array") -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim == 3:
        return np.ascontiguousarray(arr)
    if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
        arr = arr[..., 0]
        if arr.ndim == 3:
            return np.ascontiguousarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 1:
        return np.ascontiguousarray(arr[:, 0])
    raise ValueError(f"{source}: expected one 3D stack; got shape {arr.shape}")


def iter_slices_zyx(path: str | Path) -> Iterator[np.ndarray]:
    """Yield 2D slices in Z order without reading whole plain page-stacked TIFFs."""
    path = Path(path)
    with tifffile.TiffFile(str(path)) as tf:
        if _can_stack_tiff_pages_as_z(tf):
            for page in tf.pages:
                yield page.asarray()
            return
    arr = normalize_to_zyx(tifffile.imread(str(path)), source=str(path))
    for z in range(arr.shape[0]):
        yield arr[z]


def resize_area(frame: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    if factor <= 1:
        return frame.astype(np.float32, copy=False)
    h, w = frame.shape
    return cv2.resize(frame.astype(np.float32), (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)


def make_small_volume(path: Path, ds_z: int, ds_xy: int) -> np.ndarray:
    ds_z = max(1, int(ds_z))
    frames: list[np.ndarray] = []
    for z, sl in enumerate(iter_slices_zyx(path)):
        if z % ds_z != 0:
            continue
        frames.append(resize_area(sl, ds_xy))
    if not frames:
        raise ValueError(f"No slices read from {path}")
    return np.stack(frames, axis=0).astype(np.float32)


def largest_component_mask(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == int(np.argmax(counts))


def bbox_from_mask(mask: np.ndarray) -> BBoxZYX | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    return BBoxZYX(int(z0), int(z1), int(y0), int(y1), int(x0), int(x1))


def clamp_bbox(b: BBoxZYX, shape: tuple[int, int, int]) -> BBoxZYX:
    z, y, x = shape
    return BBoxZYX(
        max(0, min(z, b.z0)), max(0, min(z, b.z1)),
        max(0, min(y, b.y0)), max(0, min(y, b.y1)),
        max(0, min(x, b.x0)), max(0, min(x, b.x1)),
    )


def expand_bbox(b: BBoxZYX, pad_z: int, pad_y: int, pad_x: int, shape: tuple[int, int, int]) -> BBoxZYX:
    return clamp_bbox(BBoxZYX(b.z0 - pad_z, b.z1 + pad_z, b.y0 - pad_y, b.y1 + pad_y, b.x0 - pad_x, b.x1 + pad_x), shape)


def union_bbox(boxes: list[BBoxZYX], shape: tuple[int, int, int]) -> BBoxZYX:
    if not boxes:
        return BBoxZYX.full(shape)
    return clamp_bbox(BBoxZYX(
        min(b.z0 for b in boxes), max(b.z1 for b in boxes),
        min(b.y0 for b in boxes), max(b.y1 for b in boxes),
        min(b.x0 for b in boxes), max(b.x1 for b in boxes),
    ), shape)


def estimate_foreground_bbox(path: Path, shape: tuple[int, int, int], args: argparse.Namespace) -> BBoxZYX:
    small = make_small_volume(path, args.bbox_downsample_z, args.bbox_downsample_xy)
    pos = small[small > 0]
    if pos.size < 100:
        return BBoxZYX.full(shape)
    thr = float(np.percentile(pos, args.mask_percentile))
    mask = small >= thr
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=1)
    mask = largest_component_mask(mask)
    if args.bbox_dilate > 0:
        mask = ndi.binary_dilation(mask, iterations=int(args.bbox_dilate))
    b_small = bbox_from_mask(mask)
    if b_small is None:
        return BBoxZYX.full(shape)
    b_full = BBoxZYX(
        b_small.z0 * args.bbox_downsample_z,
        min(shape[0], b_small.z1 * args.bbox_downsample_z),
        b_small.y0 * args.bbox_downsample_xy,
        min(shape[1], b_small.y1 * args.bbox_downsample_xy),
        b_small.x0 * args.bbox_downsample_xy,
        min(shape[2], b_small.x1 * args.bbox_downsample_xy),
    )
    return expand_bbox(b_full, args.crop_padding_z, args.crop_padding_xy, args.crop_padding_xy, shape)


def crop_slice(sl: np.ndarray, bbox: BBoxZYX, z: int) -> np.ndarray | None:
    if z < bbox.z0 or z >= bbox.z1:
        return None
    return sl[bbox.y0:bbox.y1, bbox.x0:bbox.x1]


def smooth_2d_background(bg: np.ndarray, sigma: float, downsample: int) -> np.ndarray:
    bg = bg.astype(np.float32, copy=False)
    downsample = max(1, int(downsample))
    if downsample > 1:
        h, w = bg.shape
        small = cv2.resize(bg, (max(16, w // downsample), max(16, h // downsample)), interpolation=cv2.INTER_AREA)
        sig = max(0.5, float(sigma) / downsample)
        small = cv2.GaussianBlur(small, (0, 0), sig)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return cv2.GaussianBlur(bg, (0, 0), float(sigma)).astype(np.float32)


def estimate_z_background(path: Path, bbox: BBoxZYX, args: argparse.Namespace) -> np.ndarray:
    """Estimate smooth XY background from Z-min or low-Z percentile inside the bbox."""
    method = args.z_projection
    crop_shape = bbox.shape()
    if method == "min":
        bg = None
        for z, sl in enumerate(iter_slices_zyx(path)):
            cr = crop_slice(sl, bbox, z)
            if cr is None:
                continue
            cr_f = cr.astype(np.float32, copy=False)
            bg = cr_f.copy() if bg is None else np.minimum(bg, cr_f)
        if bg is None:
            raise ValueError(f"{path}: empty crop {bbox}")
    else:
        # Percentile needs a cropped stack; this is still much smaller when crop-to-foreground is used.
        stack = np.empty(crop_shape, dtype=np.float32)
        zi = 0
        for z, sl in enumerate(iter_slices_zyx(path)):
            cr = crop_slice(sl, bbox, z)
            if cr is None:
                continue
            stack[zi] = cr.astype(np.float32, copy=False)
            zi += 1
        if zi != crop_shape[0]:
            stack = stack[:zi]
        q = float(args.z_bg_percentile)
        bg = np.percentile(stack, q, axis=0).astype(np.float32)
        del stack
    return smooth_2d_background(bg, args.bg_sigma, args.bg_downsample)


def output_dtype_info(dtype_name: str, input_dtype: np.dtype) -> np.dtype:
    if dtype_name == "same":
        return np.dtype(input_dtype)
    return np.dtype(dtype_name)


def cast_output(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.rint(np.clip(arr, info.min, info.max)).astype(dtype)
    return arr.astype(dtype)


def estimate_offset_from_background(path: Path, bbox: BBoxZYX, bg_xy: np.ndarray, args: argparse.Namespace) -> float:
    if not args.auto_offset:
        return float(args.bg_offset)
    vals: list[np.ndarray] = []
    max_slices = max(1, int(args.offset_sample_slices))
    step = max(1, bbox.shape()[0] // max_slices)
    for z, sl in enumerate(iter_slices_zyx(path)):
        if z < bbox.z0 or z >= bbox.z1 or ((z - bbox.z0) % step != 0):
            continue
        cr = sl[bbox.y0:bbox.y1, bbox.x0:bbox.x1].astype(np.float32) - float(args.bg_alpha) * bg_xy
        # Estimate residual baseline from low-intensity pixels after subtraction.
        flat = cr.ravel()
        if flat.size:
            vals.append(np.percentile(flat, [5, 10, 20, 30]).astype(np.float32))
    if not vals:
        return 0.0
    sampled = np.concatenate(vals)
    off = float(np.percentile(sampled, args.offset_percentile))
    return max(0.0, off)


def write_bgsub_tiff(input_path: Path, output_path: Path, bbox: BBoxZYX, args: argparse.Namespace) -> dict[str, Any]:
    shape, in_dtype, _ = tiff_shape_dtype(input_path)
    out_dtype = output_dtype_info(args.output_dtype, in_dtype)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    bg_xy = estimate_z_background(input_path, bbox, args)
    offset = estimate_offset_from_background(input_path, bbox, bg_xy, args)

    with tifffile.TiffWriter(str(output_path), bigtiff=True) as tw:
        written = 0
        for z, sl in enumerate(iter_slices_zyx(input_path)):
            cr = crop_slice(sl, bbox, z)
            if cr is None:
                continue
            corr = cr.astype(np.float32, copy=False) - float(args.bg_alpha) * bg_xy - offset
            corr[corr < 0] = 0
            out = cast_output(corr, out_dtype)
            tw.write(out, compression=args.compression, metadata=None, contiguous=False)
            written += 1

    size_in = input_path.stat().st_size if input_path.exists() else 0
    size_out = output_path.stat().st_size if output_path.exists() else 0
    return {
        "input": str(input_path),
        "output": str(output_path),
        "status": "ok",
        "input_shape_zyx": list(shape),
        "output_shape_zyx": list(bbox.shape()),
        "input_dtype": str(in_dtype),
        "output_dtype": str(out_dtype),
        "bbox_zyx": bbox.to_dict(),
        "method": args.method,
        "z_projection": args.z_projection,
        "z_bg_percentile": args.z_bg_percentile if args.z_projection == "percentile" else None,
        "bg_sigma": args.bg_sigma,
        "bg_alpha": args.bg_alpha,
        "auto_offset": bool(args.auto_offset),
        "offset_subtracted": offset,
        "compression": args.compression,
        "input_bytes": int(size_in),
        "output_bytes": int(size_out),
        "size_ratio_output_vs_input": float(size_out / size_in) if size_in else None,
        "elapsed_sec": time.time() - t0,
    }


def process_position(position: str, files: list[Path], output_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    if not files:
        return []
    print(f"[{position}] {len(files)} TIFF(s)")
    shapes = [tiff_shape_dtype(f)[0] for f in files]
    base_shape = shapes[0]
    if any(s != base_shape for s in shapes):
        raise ValueError(f"[{position}] input shapes differ: {shapes}. Register/crop to common frame before BGS.")

    if args.crop_to_foreground:
        boxes = []
        for i, f in enumerate(files, 1):
            print(f"[{position}] estimating foreground crop {i}/{len(files)}: {f.name}")
            boxes.append(estimate_foreground_bbox(f, base_shape, args))
        bbox = union_bbox(boxes, base_shape)
    else:
        bbox = BBoxZYX.full(base_shape)

    print(f"[{position}] using common bbox z={bbox.z0}:{bbox.z1}, y={bbox.y0}:{bbox.y1}, x={bbox.x0}:{bbox.x1}; output shape={bbox.shape()}")
    out_dir = output_root / position
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for i, f in enumerate(files, 1):
        out = out_dir / f.name
        meta = out_dir / f"{f.stem}.bgsub.json"
        if out.exists() and not args.overwrite:
            print(f"[{position}] skip exists {i}/{len(files)}: {f.name}")
            results.append({"input": str(f), "output": str(out), "status": "skipped_exists", "bbox_zyx": bbox.to_dict()})
            continue
        print(f"[{position}] background subtract {i}/{len(files)}: {f.name}")
        r = write_bgsub_tiff(f, out, bbox, args)
        r["position"] = position
        r["timepoint"] = f.stem
        results.append(r)
        with meta.open("w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=2)
        print(f"[{position}] wrote {out.name}; shape={r['output_shape_zyx']}; size_ratio={r['size_ratio_output_vs_input']:.3f}; elapsed={r['elapsed_sec']:.1f}s")
    return results


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", required=True, help="Registered TIFF root. Can contain position folders or TIFFs for one position.")
    p.add_argument("--output-root", required=True, help="Output root for background-subtracted TIFFs.")
    p.add_argument("--glob-pattern", default="*.tif*", help="Only matching TIFF files are processed.")
    p.add_argument("--positions", nargs="*", default=None)
    p.add_argument("--ome-xml", default=None, help="Optional OME XML for reporting voxel spacing only.")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--method", default="zmin_smooth", choices=["zmin_smooth"], help="3D stack BGS method.")
    p.add_argument("--z-projection", default="min", choices=["min", "percentile"], help="Background XY projection over Z.")
    p.add_argument("--z-bg-percentile", type=float, default=2.0, help="Used only with --z-projection percentile.")
    p.add_argument("--bg-sigma", type=float, default=80.0, help="Gaussian smoothing sigma for the XY background image, in output pixels.")
    p.add_argument("--bg-downsample", type=int, default=8, help="Downsample factor used only for smoothing the background image.")
    p.add_argument("--bg-alpha", type=float, default=1.0, help="Background subtraction strength.")
    p.add_argument("--auto-offset", action="store_true", help="Subtract a residual positive baseline estimated from low-intensity pixels.")
    p.add_argument("--offset-percentile", type=float, default=50.0)
    p.add_argument("--offset-sample-slices", type=int, default=24)
    p.add_argument("--bg-offset", type=float, default=0.0)

    p.add_argument("--crop-to-foreground", action="store_true", help="Use one common foreground bbox per position to reduce output size.")
    p.add_argument("--crop-padding-xy", type=int, default=80)
    p.add_argument("--crop-padding-z", type=int, default=8)
    p.add_argument("--bbox-downsample-xy", type=int, default=8)
    p.add_argument("--bbox-downsample-z", type=int, default=2)
    p.add_argument("--bbox-dilate", type=int, default=2, help="3D dilation iterations on small foreground mask before bbox.")
    p.add_argument("--mask-percentile", type=float, default=70.0, help="Foreground threshold percentile on positive pixels for crop detection.")

    p.add_argument("--output-dtype", default="same", choices=["same", "uint16", "uint8", "float32"], help="Use uint16/same for analysis; uint8 only for previews.")
    p.add_argument("--compression", default="zlib", help="TIFF compression: zlib, lzw, none. zlib reduces size but is slower.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.compression).lower() in {"none", "no", "false", "0"}:
        args.compression = None

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    spacing = parse_ome_spacing(args.ome_xml)
    if spacing:
        print(f"Using voxel spacing from OME for metadata only: x={spacing[0]}, y={spacing[1]}, z={spacing[2]}")

    positions = discover_positions(input_root, args.glob_pattern)
    if args.positions:
        keep = set(args.positions)
        positions = {k: v for k, v in positions.items() if k in keep}
    if not positions:
        raise SystemExit(f"No TIFF files found under {input_root} with pattern {args.glob_pattern!r}")

    all_results: list[dict[str, Any]] = []
    t0 = time.time()
    for position, files in positions.items():
        all_results.extend(process_position(position, files, output_root, args))

    summary_csv = output_root / "background_subtraction_summary.csv"
    write_summary_csv(summary_csv, all_results)

    manifest = {
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "glob_pattern": args.glob_pattern,
        "positions": {p: [str(f.resolve()) for f in fs] for p, fs in positions.items()},
        "settings": {
            "method": args.method,
            "z_projection": args.z_projection,
            "z_bg_percentile": args.z_bg_percentile,
            "bg_sigma": args.bg_sigma,
            "bg_downsample": args.bg_downsample,
            "bg_alpha": args.bg_alpha,
            "auto_offset": args.auto_offset,
            "crop_to_foreground": args.crop_to_foreground,
            "crop_padding_xy": args.crop_padding_xy,
            "crop_padding_z": args.crop_padding_z,
            "output_dtype": args.output_dtype,
            "compression": args.compression,
            "spacing_xyz": list(spacing) if spacing else None,
        },
        "summary_csv": str(summary_csv),
        "results": all_results,
        "elapsed_sec": time.time() - t0,
    }
    with (output_root / "background_subtraction_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Saved summary: {summary_csv}")
    print(f"Saved manifest: {output_root / 'background_subtraction_manifest.json'}")


if __name__ == "__main__":
    main()
