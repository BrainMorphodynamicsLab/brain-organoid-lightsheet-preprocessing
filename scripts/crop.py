#!/usr/bin/env python3
"""
crop.py

Object-aware 3D cropping for large light-sheet TIFF Z-stacks.

Designed for:
  input_root/
    position_1/
      t0001_561_Fused.tif   # one full 3D Z-stack
      t0002_561_Fused.tif
      ...

Key differences from the earlier crop_3d_tiff.py:
- Detects the organoid mainly from an XY projection/MIP, not from all 3D thresholded voxels.
- Keeps the largest 2D object and can remove components touching the image border.
- This avoids letting scattered background speckles or edge artifacts force the crop to the full image.
- Applies one 3D crop box per organoid/position to all timepoints, so coordinates remain consistent.

Output:
  output_root/position_1/*.tif
  output_root/position_1/crop_common_box.json
  output_root/crop_manifest.json
  output_root/crop_summary.csv
  output_root/crop_qc/position_1/*.png

python3 script/crop.py \
  --input-root result/bgsub \
  --output-root result/cropped_object_xy \
  --glob-pattern "*.tif*" \
  --analysis-downsample-xy 4 \
  --analysis-downsample-z 1 \
  --xy-projection percentile \
  --xy-projection-percentile 95 \
  --xy-threshold-method otsu \
  --xy-threshold-floor-percentile 65 \
  --crop-padding-xy 80 \
  --crop-padding-z 8 \
  --no-crop-z \
  --compression zlib \
  --overwrite
  
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None


@dataclass
class CropBox:
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    def shape(self) -> Tuple[int, int, int]:
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)

    def as_slices(self):
        return (slice(self.z0, self.z1), slice(self.y0, self.y1), slice(self.x0, self.x1))

    def union(self, other: "CropBox") -> "CropBox":
        return CropBox(
            min(self.z0, other.z0), max(self.z1, other.z1),
            min(self.y0, other.y0), max(self.y1, other.y1),
            min(self.x0, other.x0), max(self.x1, other.x1),
        )


_TIME_RE = re.compile(r"(?:^|[^A-Za-z0-9])t(\d+)", re.IGNORECASE)


def natural_key(path: Path):
    m = _TIME_RE.search(path.name)
    if m:
        return (0, int(m.group(1)), path.name)
    parts = re.split(r"(\d+)", path.name)
    return (1, [int(p) if p.isdigit() else p.lower() for p in parts], path.name)


def discover_positions(input_root: Path, glob_pattern: str, positions: Optional[List[str]] = None) -> Dict[str, List[Path]]:
    input_root = input_root.resolve()
    direct_files = sorted([p for p in input_root.glob(glob_pattern) if p.is_file()], key=natural_key)
    out: Dict[str, List[Path]] = {}
    if direct_files:
        out[input_root.name] = direct_files
    else:
        for d in sorted([p for p in input_root.iterdir() if p.is_dir()]):
            if positions and d.name not in positions:
                continue
            files = sorted([p for p in d.glob(glob_pattern) if p.is_file()], key=natural_key)
            if files:
                out[d.name] = files
    if not out:
        raise FileNotFoundError(f"No TIFF files found under {input_root} using pattern {glob_pattern!r}")
    return out


def _normalize_to_zyx(arr: np.ndarray, source: Path) -> np.ndarray:
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 4:
        # Common single-channel or small-channel cases.
        if arr.shape[0] <= 4 and arr.shape[1] > 4:
            arr = arr[0]
        elif arr.shape[-1] <= 4 and arr.shape[0] > 4:
            arr = arr[..., 0]
        elif arr.shape[1] <= 4 and arr.shape[0] > 4:
            arr = arr[:, 0, :, :]
        else:
            raise ValueError(f"{source}: cannot infer one 3D ZYX stack from shape {arr.shape}")
    else:
        raise ValueError(f"{source}: expected 3D stack after cleanup, got shape {arr.shape}")
    if arr.ndim != 3:
        raise ValueError(f"{source}: expected 3D ZYX, got shape {arr.shape}")
    return arr


def read_volume(path: Path) -> np.ndarray:
    return _normalize_to_zyx(tifffile.imread(str(path)), path)


def downsample_view(vol: np.ndarray, dz: int, dy: int, dx: int) -> np.ndarray:
    return np.asarray(vol[::max(1, dz), ::max(1, dy), ::max(1, dx)])


def otsu_threshold(x: np.ndarray, nbins: int = 256) -> float:
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    lo = float(np.percentile(x, 0.5))
    hi = float(np.percentile(x, 99.5))
    if hi <= lo:
        return float(np.median(x))
    hist, edges = np.histogram(x, bins=nbins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1e-12)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1e-12))[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if variance12.size == 0 or not np.isfinite(variance12).any():
        return float(np.median(x))
    idx = int(np.nanargmax(variance12))
    return float(centers[idx])


def remove_border_labels(lab: np.ndarray) -> np.ndarray:
    border_vals = set()
    if lab.size == 0:
        return lab
    border_vals.update(np.unique(lab[0, :]).tolist())
    border_vals.update(np.unique(lab[-1, :]).tolist())
    border_vals.update(np.unique(lab[:, 0]).tolist())
    border_vals.update(np.unique(lab[:, -1]).tolist())
    border_vals.discard(0)
    if not border_vals:
        return lab
    out = lab.copy()
    for v in border_vals:
        out[out == v] = 0
    return out


def largest_component_2d(mask: np.ndarray, remove_border: bool, min_area: int) -> np.ndarray:
    mask = mask.astype(bool)
    if ndi is None:
        return mask
    # Fill ring/lumen holes only for bbox estimation. This does not change image intensities.
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5), bool), iterations=1)
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3), bool), iterations=1)
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    if remove_border:
        lab = remove_border_labels(lab)
    sizes = np.bincount(lab.ravel()) if lab.max() > 0 else np.array([0])
    if sizes.size <= 1:
        # If removing border labels removed everything, fall back to largest component without border removal.
        lab, n = ndi.label(mask)
        sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    if sizes.max() < min_area:
        keep = int(np.argmax(sizes))
    else:
        keep = int(np.argmax(sizes))
    return lab == keep


def bbox2d(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return None
    return int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1


def projection_xy(vol_ds: np.ndarray, mode: str, percentile: float) -> np.ndarray:
    x = vol_ds.astype(np.float32, copy=False)
    if mode == "max":
        return np.max(x, axis=0)
    if mode == "mean":
        return np.mean(x, axis=0)
    # percentile projection is less sensitive to isolated hot pixels than max projection.
    return np.percentile(x, percentile, axis=0).astype(np.float32)


def estimate_xy_bbox_from_mip(vol_ds: np.ndarray, args, source: Path) -> Tuple[Tuple[int, int, int, int], float, dict]:
    mip = projection_xy(vol_ds, args.xy_projection, args.xy_projection_percentile)
    mip_smooth = mip
    if ndi is not None and args.xy_smooth_sigma > 0:
        mip_smooth = ndi.gaussian_filter(mip_smooth, sigma=float(args.xy_smooth_sigma))

    positive = mip_smooth[mip_smooth > 0]
    if positive.size < 100:
        vals = mip_smooth[np.isfinite(mip_smooth)]
        thr = float(np.percentile(vals, 99.0)) if vals.size else 0.0
    elif args.xy_threshold_method == "percentile":
        thr = float(np.percentile(positive, args.xy_threshold_percentile))
    else:
        # Otsu on positive pixels is stable for bright organoid on low background.
        thr = otsu_threshold(positive)
        # Avoid an overly low Otsu threshold that includes broad haze.
        floor_thr = float(np.percentile(positive, args.xy_threshold_floor_percentile))
        thr = max(thr, floor_thr)

    mask = mip_smooth > thr
    mask = largest_component_2d(
        mask,
        remove_border=not args.keep_border_components,
        min_area=int(args.min_area_pixels),
    )
    bb = bbox2d(mask)
    if bb is None:
        raise RuntimeError(f"{source}: XY foreground detection failed. Try lower --xy-threshold-floor-percentile or disable --keep-border-components fallback.")
    return bb, thr, {
        "xy_projection_mode": args.xy_projection,
        "xy_projection_percentile": args.xy_projection_percentile,
        "xy_threshold": thr,
        "xy_mask_area_px_downsampled": int(mask.sum()),
    }


def estimate_z_range(vol_ds: np.ndarray, xy_box_ds: Tuple[int, int, int, int], args) -> Tuple[int, int, dict]:
    zdim = vol_ds.shape[0]
    y0, y1, x0, x1 = xy_box_ds
    sub = vol_ds[:, max(0, y0):min(vol_ds.shape[1], y1), max(0, x0):min(vol_ds.shape[2], x1)]
    if sub.size == 0:
        return 0, zdim, {"z_method": "fallback_full"}
    # Use high percentile in the XY crop per Z plane; robust to background zeros.
    profile = np.percentile(sub.astype(np.float32, copy=False), args.z_profile_percentile, axis=(1, 2))
    if ndi is not None and args.z_smooth_sigma > 0:
        profile = ndi.gaussian_filter1d(profile, sigma=float(args.z_smooth_sigma))
    pmax = float(np.max(profile)) if profile.size else 0.0
    pmin = float(np.min(profile)) if profile.size else 0.0
    if pmax <= pmin:
        return 0, zdim, {"z_method": "flat_full", "z_profile_min": pmin, "z_profile_max": pmax}
    thr = pmin + float(args.z_profile_fraction) * (pmax - pmin)
    inds = np.where(profile > thr)[0]
    if inds.size == 0:
        return 0, zdim, {"z_method": "empty_full", "z_threshold": thr, "z_profile_min": pmin, "z_profile_max": pmax}
    return int(inds.min()), int(inds.max()) + 1, {
        "z_method": "profile",
        "z_threshold": float(thr),
        "z_profile_min": pmin,
        "z_profile_max": pmax,
        "z_profile_percentile": args.z_profile_percentile,
    }


def ds_box_to_full(z0: int, z1: int, y0: int, y1: int, x0: int, x1: int, ds: Tuple[int, int, int], shape: Tuple[int, int, int]) -> CropBox:
    dz, dy, dx = ds
    return CropBox(
        max(0, z0 * dz), min(shape[0], z1 * dz),
        max(0, y0 * dy), min(shape[1], y1 * dy),
        max(0, x0 * dx), min(shape[2], x1 * dx),
    )


def expand_box(b: CropBox, shape: Tuple[int, int, int], pad_z: int, pad_xy: int, square_xy: bool) -> CropBox:
    z, y, x = shape
    z0 = max(0, b.z0 - pad_z); z1 = min(z, b.z1 + pad_z)
    y0 = max(0, b.y0 - pad_xy); y1 = min(y, b.y1 + pad_xy)
    x0 = max(0, b.x0 - pad_xy); x1 = min(x, b.x1 + pad_xy)

    if square_xy:
        cy = (y0 + y1) / 2.0
        cx = (x0 + x1) / 2.0
        side = int(math.ceil(max(y1 - y0, x1 - x0)))
        if side % 2:
            side += 1
        y0 = int(round(cy - side / 2)); y1 = y0 + side
        x0 = int(round(cx - side / 2)); x1 = x0 + side
        if y0 < 0:
            y1 -= y0; y0 = 0
        if x0 < 0:
            x1 -= x0; x0 = 0
        if y1 > y:
            y0 -= (y1 - y); y1 = y
        if x1 > x:
            x0 -= (x1 - x); x1 = x
        y0 = max(0, y0); x0 = max(0, x0)

    # Even dimensions are convenient for downstream viewers/codecs.
    if (z1 - z0) % 2 and z1 < z: z1 += 1
    if (y1 - y0) % 2 and y1 < y: y1 += 1
    if (x1 - x0) % 2 and x1 < x: x1 += 1
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        raise RuntimeError(f"Invalid crop after expansion: {(z0,z1,y0,y1,x0,x1)}")
    return CropBox(int(z0), int(z1), int(y0), int(y1), int(x0), int(x1))


def estimate_file_crop(path: Path, args) -> Tuple[CropBox, dict, Tuple[int, int, int], str]:
    vol = read_volume(path)
    shape = tuple(int(v) for v in vol.shape)
    dtype = str(vol.dtype)
    ds = (args.analysis_downsample_z, args.analysis_downsample_xy, args.analysis_downsample_xy)
    vol_ds = downsample_view(vol, *ds)

    xy_bb_ds, xy_thr, extra_xy = estimate_xy_bbox_from_mip(vol_ds, args, path)
    y0_ds, y1_ds, x0_ds, x1_ds = xy_bb_ds
    if args.no_crop_z:
        z0_ds, z1_ds, extra_z = 0, vol_ds.shape[0], {"z_method": "disabled_full"}
    else:
        z0_ds, z1_ds, extra_z = estimate_z_range(vol_ds, xy_bb_ds, args)

    raw_full = ds_box_to_full(z0_ds, z1_ds, y0_ds, y1_ds, x0_ds, x1_ds, ds, shape)
    expanded = expand_box(raw_full, shape, args.crop_padding_z, args.crop_padding_xy, args.square_xy)

    rec = {
        "file": str(path),
        "shape_zyx": list(shape),
        "dtype": dtype,
        "bbox_raw_full_zyx": asdict(raw_full),
        "bbox_expanded_full_zyx": asdict(expanded),
        "bbox_xy_downsampled_y0y1x0x1": [int(y0_ds), int(y1_ds), int(x0_ds), int(x1_ds)],
        "z_range_downsampled_z0z1": [int(z0_ds), int(z1_ds)],
        **extra_xy,
        **extra_z,
    }
    del vol, vol_ds
    return expanded, rec, shape, dtype


def estimate_position_crop(files: List[Path], args, position: str) -> Tuple[CropBox, List[dict], Tuple[int, int, int], str]:
    union: Optional[CropBox] = None
    records: List[dict] = []
    first_shape = None
    first_dtype = None
    for i, p in enumerate(files):
        print(f"[{position}] estimating object-aware crop {i+1}/{len(files)}: {p.name}", flush=True)
        b, rec, shape, dtype = estimate_file_crop(p, args)
        records.append(rec)
        if first_shape is None:
            first_shape = shape
            first_dtype = dtype
        if shape != first_shape:
            print(f"WARNING [{position}] shape differs for {p.name}: {shape} vs first {first_shape}", flush=True)
        union = b if union is None else union.union(b)
    if union is None or first_shape is None:
        raise RuntimeError(f"No crop estimated for {position}")
    final = expand_box(union, first_shape, pad_z=0, pad_xy=0, square_xy=args.square_xy)
    return final, records, first_shape, first_dtype or "unknown"


def imshow_norm(im: np.ndarray) -> np.ndarray:
    im = np.asarray(im, dtype=np.float32)
    pos = im[np.isfinite(im) & (im > 0)]
    if pos.size == 0:
        return np.zeros_like(im)
    hi = float(np.percentile(pos, 99.5))
    return np.clip(im / max(hi, 1.0), 0, 1)


def make_mips(vol: np.ndarray) -> Dict[str, np.ndarray]:
    return {"XY": np.max(vol, axis=0), "ZY": np.max(vol, axis=2), "ZX": np.max(vol, axis=1)}


def save_crop_preview(before: np.ndarray, after: np.ndarray, crop: CropBox, outpath: Path, title: str):
    if plt is None:
        return
    bm = make_mips(before)
    am = make_mips(after)
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    for r, plane in enumerate(["XY", "ZY", "ZX"]):
        ax = axes[r, 0]
        ax.imshow(imshow_norm(bm[plane]), cmap="gray")
        ax.set_title(f"Before {plane} MIP + crop box")
        ax.axis("off")
        if plane == "XY":
            rect = plt.Rectangle((crop.x0, crop.y0), crop.x1 - crop.x0, crop.y1 - crop.y0, fill=False, linewidth=1.5)
        elif plane == "ZY":
            rect = plt.Rectangle((crop.y0, crop.z0), crop.y1 - crop.y0, crop.z1 - crop.z0, fill=False, linewidth=1.5)
        else:
            rect = plt.Rectangle((crop.x0, crop.z0), crop.x1 - crop.x0, crop.z1 - crop.z0, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax = axes[r, 1]
        ax.imshow(imshow_norm(am[plane]), cmap="gray")
        ax.set_title(f"After crop {plane} MIP")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def tiff_compression_arg(name: str):
    if str(name).lower() in ("none", "no", "false", "0"):
        return None
    return name


def write_tiff(path: Path, vol: np.ndarray, compression: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), vol, bigtiff=True, photometric="minisblack", compression=tiff_compression_arg(compression), metadata={"axes": "ZYX"})


def run(args):
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    positions = discover_positions(input_root, args.glob_pattern, args.positions)
    print("Discovered positions/organoids:")
    for pos, files in positions.items():
        print(f"  {pos}: {len(files)} TIFF file(s)")

    manifest = {
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "glob_pattern": args.glob_pattern,
        "settings": vars(args),
        "positions": {},
        "results": [],
    }
    rows = []
    t0_all = time.time()

    for pos, files in positions.items():
        pos_out = output_root / pos
        pos_out.mkdir(parents=True, exist_ok=True)
        qc_out = output_root / "crop_qc" / pos
        qc_out.mkdir(parents=True, exist_ok=True)
        pos_t0 = time.time()

        crop, detection_records, first_shape, first_dtype = estimate_position_crop(files, args, pos)
        crop_shape = crop.shape()
        voxel_ratio = float(np.prod(crop_shape) / np.prod(first_shape))
        print(f"[{pos}] common object-aware crop: {asdict(crop)} -> shape={crop_shape}, voxel_ratio={voxel_ratio:.4f}", flush=True)

        pos_manifest = {
            "files": [str(p) for p in files],
            "input_shape_zyx_first": list(first_shape),
            "input_dtype_first": first_dtype,
            "common_crop_zyx": asdict(crop),
            "output_shape_zyx_expected": list(crop_shape),
            "voxel_ratio_output_vs_input_first": voxel_ratio,
            "detection_records": detection_records,
            "elapsed_sec_estimation": time.time() - pos_t0,
        }
        manifest["positions"][pos] = pos_manifest
        (pos_out / "crop_common_box.json").write_text(json.dumps(pos_manifest, indent=2), encoding="utf-8")

        for idx, src in enumerate(files):
            out = pos_out / src.name
            meta = pos_out / f"{src.stem}.crop.json"
            if out.exists() and not args.overwrite:
                print(json.dumps({"position": pos, "file": str(src), "status": "skipped", "output": str(out)}), flush=True)
                continue
            t0 = time.time()
            try:
                vol = read_volume(src)
                z, y, x = vol.shape
                cb = CropBox(max(0, min(crop.z0, z)), max(0, min(crop.z1, z)), max(0, min(crop.y0, y)), max(0, min(crop.y1, y)), max(0, min(crop.x0, x)), max(0, min(crop.x1, x)))
                cropped = vol[cb.as_slices()].copy()
                if not args.dry_run:
                    write_tiff(out, cropped, args.compression)
                    out_bytes = out.stat().st_size
                    status = "ok"
                else:
                    out_bytes = None
                    status = "dry_run"
                in_bytes = src.stat().st_size
                rec = {
                    "position": pos,
                    "input": str(src),
                    "output": str(out),
                    "status": status,
                    "input_shape_zyx": list(vol.shape),
                    "output_shape_zyx": list(cropped.shape),
                    "input_dtype": str(vol.dtype),
                    "output_dtype": str(cropped.dtype),
                    "common_crop_zyx": asdict(crop),
                    "applied_crop_zyx": asdict(cb),
                    "compression": args.compression,
                    "input_bytes": int(in_bytes),
                    "output_bytes": int(out_bytes) if out_bytes is not None else None,
                    "file_size_ratio_output_vs_input": float(out_bytes / in_bytes) if out_bytes is not None and in_bytes else None,
                    "voxel_ratio_output_vs_input": float(np.prod(cropped.shape) / np.prod(vol.shape)),
                    "elapsed_sec": time.time() - t0,
                }
                meta.write_text(json.dumps(rec, indent=2), encoding="utf-8")
                manifest["results"].append(rec)
                rows.append(rec)
                print(json.dumps({"position": pos, "file": str(src), "status": status, "output": str(out), "shape": list(cropped.shape)}), flush=True)
                if idx in {0, len(files)//2, len(files)-1} and not args.no_qc_png:
                    save_crop_preview(vol, cropped, cb, qc_out / f"{src.stem}_crop_preview.png", f"{pos} {src.name}: object-aware crop")
                del vol, cropped
            except Exception as exc:
                rec = {"position": pos, "input": str(src), "status": "error", "error": str(exc)}
                manifest["results"].append(rec)
                rows.append(rec)
                print(json.dumps(rec), flush=True)
                if not args.continue_on_error:
                    raise

    manifest["elapsed_sec"] = time.time() - t0_all
    (output_root / "crop_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if rows:
        keys = sorted({k for r in rows for k in r.keys() if k not in {"common_crop_zyx", "applied_crop_zyx"}})
        with (output_root / "crop_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
    print(f"Wrote manifest: {output_root / 'crop_manifest.json'}")
    print(f"Wrote summary:  {output_root / 'crop_summary.csv'}")


def build_parser():
    p = argparse.ArgumentParser(description="Object-aware 3D cropping for large light-sheet TIFF Z-stacks")
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--glob-pattern", default="*.tif*")
    p.add_argument("--positions", nargs="*", default=None)

    p.add_argument("--analysis-downsample-xy", type=int, default=4)
    p.add_argument("--analysis-downsample-z", type=int, default=1)

    p.add_argument("--xy-projection", choices=["percentile", "max", "mean"], default="percentile", help="Projection used to find XY object crop.")
    p.add_argument("--xy-projection-percentile", type=float, default=95.0, help="Used when --xy-projection percentile.")
    p.add_argument("--xy-smooth-sigma", type=float, default=2.0, help="Gaussian smoothing on downsampled XY projection before thresholding.")
    p.add_argument("--xy-threshold-method", choices=["otsu", "percentile"], default="otsu")
    p.add_argument("--xy-threshold-percentile", type=float, default=70.0, help="Used only when --xy-threshold-method percentile.")
    p.add_argument("--xy-threshold-floor-percentile", type=float, default=65.0, help="Otsu threshold is not allowed below this positive-pixel percentile.")
    p.add_argument("--min-area-pixels", type=int, default=500, help="Minimum 2D component area in downsampled XY pixels.")
    p.add_argument("--keep-border-components", action="store_true", help="Do not remove objects touching the image border. Usually leave this off.")

    p.add_argument("--no-crop-z", action="store_true", help="Crop XY only and keep all Z slices.")
    p.add_argument("--z-profile-percentile", type=float, default=99.0)
    p.add_argument("--z-profile-fraction", type=float, default=0.04, help="Lower is safer; higher crops Z more aggressively.")
    p.add_argument("--z-smooth-sigma", type=float, default=1.0)

    p.add_argument("--crop-padding-xy", type=int, default=80)
    p.add_argument("--crop-padding-z", type=int, default=8)
    p.add_argument("--square-xy", action="store_true", help="Make XY crop square around object.")

    p.add_argument("--compression", default="zlib")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--no-qc-png", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
