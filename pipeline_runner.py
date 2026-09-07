#!/usr/bin/env python3
"""Dataset discovery and orchestration for the light-sheet end-to-end pipeline.

This module deliberately keeps image algorithms in fuse_views.py, registration.py,
bgs.py, denoise.py, and crop.py.  It scans names and TIFF headers only, launches each stage with
``sys.executable`` and argument lists, validates outputs, supports resumable stage
manifests, and writes combined manifests/size reports.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    import tifffile
except Exception:  # Validation will report the dependency cleanly.
    tifffile = None

SCRIPT_DIR = Path(__file__).resolve().parent
STAGES = ("fusion", "registration", "background_subtraction", "denoise", "crop")
ORDER_BGS_CROP = "Background subtraction → Crop"
ORDER_CROP_BGS = "Crop → Background subtraction"
INPUT_RAW = "Raw View1/View2 TIFFs"
INPUT_FUSED = "Already fused TIFFs"
RUN_FULL = "Complete workflow"
RUN_SINGLE = "Single stage"

DEFAULT_CONFIG: dict[str, Any] = {
    "run_mode": RUN_FULL,
    "single_stage": "registration",
    "starting_data_type": INPUT_RAW,
    "input_dataset_folder": "",
    "output_root": "result",
    "position_folder_pattern": r"^(?P<position>.+)_(?P<settings>[^/]+)$",
    "settings_pattern": "Settings 1",
    "view1_token": "View1",
    "view2_token": "View2",
    "fused_token": "Fused",
    "tiff_glob_pattern": "*.tif*",
    "ome_companion_glob_pattern": "*.ome",
    "timepoint_regex": r"(?i)(?:^|[_-])t(?P<timepoint>\d+)",
    "channel_regex": r"(?i)^t\d+_(?P<channel>.+?)_(?:{view1}|{view2})\.tif{1,2}$",
    "fused_channel_regex": r"(?i)^t\d+_(?P<channel>.+?)_{fused}\.tif{1,2}$",
    "processed_channel_regex": r"(?i)^t\d+_(?P<channel>.+?)(?:_{fused})?\.tif{1,2}$",
    "selected_positions": "ALL",
    "selected_channels": "ALL",
    "first_timepoint": 1,
    "last_timepoint": 10000,
    "reference_channel": "",
    "validation_mode": "quick",
    "quick_validation_samples_per_position_channel": 2,
    "processing_order": ORDER_BGS_CROP,
    "coarse_crop_during_bgs": False,
    "enable_denoising": False,
    "dry_run": False,
    "resume": False,
    "overwrite": True,
    "start_stage": "fusion",
    "stop_after_stage": "final",
    "continue_on_error": False,
    "fusion_crop_um": 950,
    "fusion_save_as_crop": False,
    "registration_glob_pattern": "*.tif*",
    "registration_downsample_xy": 4,
    "registration_downsample_z": 1,
    "registration_estimate_method": "center_phase_mip",
    "registration_apply_mode": "integer",
    "registration_compression": "none",
    "registration_save_dtype": "input",
    "registration_padding_z": 8,
    "registration_padding_y": 64,
    "registration_padding_x": 64,
    "registration_phase_upsample": 1,
    "registration_series_index": 0,
    "registration_channel_index": "",
    "registration_clip_low": 1.0,
    "registration_clip_high": 99.0,
    "registration_center_threshold_percentile": 75.0,
    "registration_max_shift_z": "",
    "registration_max_shift_yx": "",
    "bgs_glob_pattern": "*.tif*",
    "bgs_method": "zmin_smooth",
    "bgs_z_projection": "min",
    "bgs_z_bg_percentile": 2.0,
    "bgs_bg_sigma": 80.0,
    "bgs_bg_downsample": 8,
    "bgs_bg_alpha": 1.0,
    "bgs_auto_offset": True,
    "bgs_offset_percentile": 50.0,
    "bgs_offset_sample_slices": 24,
    "bgs_bg_offset": 0.0,
    "bgs_crop_padding_xy": 80,
    "bgs_crop_padding_z": 8,
    "bgs_bbox_downsample_xy": 8,
    "bgs_bbox_downsample_z": 2,
    "bgs_bbox_dilate": 2,
    "bgs_mask_percentile": 70.0,
    "bgs_output_dtype": "same",
    "bgs_compression": "zlib",
    "denoise_glob_pattern": "*.tif*",
    "denoise_method": "median_gaussian2d",
    "denoise_median_size": 3,
    "denoise_sigma_xy": 0.6,
    "denoise_sigma_z": 0.0,
    "denoise_intensity_floor": 5.0,
    "denoise_output_dtype": "same",
    "denoise_compression": "zlib",
    "denoise_crop_to_foreground": False,
    "denoise_crop_padding_xy": 40,
    "denoise_crop_padding_z": 4,
    "denoise_mask_percentile": 70.0,
    "denoise_min_size_voxels": 10000,
    "crop_glob_pattern": "*.tif*",
    "crop_analysis_downsample_xy": 4,
    "crop_analysis_downsample_z": 1,
    "crop_xy_projection": "percentile",
    "crop_xy_projection_percentile": 95.0,
    "crop_xy_smooth_sigma": 2.0,
    "crop_xy_threshold_method": "otsu",
    "crop_xy_threshold_percentile": 70.0,
    "crop_xy_threshold_floor_percentile": 65.0,
    "crop_min_area_pixels": 500,
    "crop_keep_border_components": False,
    "crop_no_crop_z": True,
    "crop_z_profile_percentile": 99.0,
    "crop_z_profile_fraction": 0.04,
    "crop_z_smooth_sigma": 1.0,
    "crop_padding_xy": 80,
    "crop_padding_z": 8,
    "crop_square_xy": False,
    "crop_compression": "zlib",
    "crop_no_qc_png": False,
}


@dataclass
class DiscoveryRow:
    position: str
    settings: str
    timepoint: str
    channel: str
    view1_path: str = ""
    view2_path: str = ""
    fused_path: str = ""
    input_path: str = ""
    ome_path: str = ""
    pairing_status: str = "ok"
    source_folder: str = ""


@dataclass
class ScanResult:
    rows: list[DiscoveryRow] = field(default_factory=list)
    issues: list[dict[str, str]] = field(default_factory=list)
    positions: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    settings_by_position: dict[str, str] = field(default_factory=dict)
    ome_by_position: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [asdict(r) for r in self.rows],
            "issues": self.issues,
            "positions": self.positions,
            "channels": self.channels,
            "settings_by_position": self.settings_by_position,
            "ome_by_position": self.ome_by_position,
        }


class PipelineFailure(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def merged_config(values: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if values:
        cfg.update(values)
        # Backward compatibility for settings written before Run mode existed.
        if "run_mode" not in values:
            start = str(values.get("start_stage", DEFAULT_CONFIG["start_stage"]))
            stop = str(values.get("stop_after_stage", DEFAULT_CONFIG["stop_after_stage"]))
            if stop == start and start in STAGES:
                cfg["run_mode"] = RUN_SINGLE
                cfg["single_stage"] = start
    return cfg


def expand_regex_template(pattern: str, config: dict[str, Any]) -> str:
    replacements = {
        "{view1}": re.escape(str(config["view1_token"])),
        "{view2}": re.escape(str(config["view2_token"])),
        "{fused}": re.escape(str(config["fused_token"])),
    }
    for key, value in replacements.items():
        pattern = pattern.replace(key, value)
    return pattern


def _named_or_first(match: re.Match[str], name: str) -> str | None:
    if name in match.re.groupindex:
        return match.group(name)
    if match.groups():
        return match.group(1)
    return None


def _view_kind(filename: str, view1: str, view2: str) -> tuple[str | None, str | None]:
    stem = Path(filename).stem
    has1 = view1 in stem
    has2 = view2 in stem
    if has1 and has2:
        return None, "unsupported filename: contains both view tokens"
    if has1:
        return "view1", None
    if has2:
        return "view2", None
    return None, "unsupported filename: no configured view token"


def _selection(value: Any) -> list[str] | None:
    text = str(value or "").strip()
    if not text or text.upper() == "ALL":
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def _time_sort(value: str) -> tuple[int, Any]:
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def _iter_directories_bounded(root: Path, max_depth: int = 4) -> Iterable[Path]:
    """Yield directories without traversing below max_depth.

    Unlike ``Path.rglob``, this prunes the walk before descending into deeper
    server folders. This matters when the selected input root is on SMB/NFS.
    """
    yield root
    for current, dirnames, _filenames in os.walk(root, topdown=True):
        current_path = Path(current)
        try:
            depth = len(current_path.relative_to(root).parts)
        except ValueError:
            dirnames[:] = []
            continue
        if depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames.sort()
        for name in dirnames:
            yield current_path / name



def _uses_raw_pair_discovery(config: dict[str, Any]) -> bool:
    cfg = merged_config(config)
    return cfg.get("run_mode") == RUN_SINGLE and cfg.get("single_stage") == "fusion" or (
        cfg.get("run_mode") != RUN_SINGLE and cfg.get("starting_data_type") == INPUT_RAW
    )


def _uses_generic_stage_input(config: dict[str, Any]) -> bool:
    cfg = merged_config(config)
    return cfg.get("run_mode") == RUN_SINGLE and cfg.get("single_stage") != "fusion"


def _scan_stage_ready_dataset(config: dict[str, Any]) -> ScanResult:
    """Discover generic stage-ready TIFF stacks for one standalone stage.

    The input may directly contain TIFFs for one organoid, or may contain one
    TIFF folder per organoid/position.  No previous pipeline manifest is needed.
    """
    cfg = merged_config(config)
    root = Path(cfg["input_dataset_folder"]).expanduser().resolve()
    result = ScanResult()
    if not root.is_dir():
        result.issues.append({"type": "configuration", "path": str(root), "message": "Input dataset folder does not exist."})
        return result

    try:
        time_re = re.compile(expand_regex_template(str(cfg["timepoint_regex"]), cfg))
        channel_re = re.compile(expand_regex_template(str(cfg["processed_channel_regex"]), cfg))
    except re.error as exc:
        result.issues.append({"type": "configuration", "path": "", "message": f"Invalid regular expression: {exc}"})
        return result

    selected_positions = _selection(cfg.get("selected_positions"))
    selected_channels = _selection(cfg.get("selected_channels"))
    first_tp = int(cfg["first_timepoint"])
    last_tp = int(cfg["last_timepoint"])

    def matching_tiffs(directory: Path) -> list[Path]:
        try:
            return sorted(
                (p for p in directory.glob(str(cfg["tiff_glob_pattern"])) if p.is_file()),
                key=lambda p: p.name,
            )
        except OSError as exc:
            result.issues.append({"type": "configuration", "path": str(directory), "message": f"Could not list directory: {exc}"})
            return []

    direct_files = matching_tiffs(root)
    candidate_dirs: list[Path]
    if direct_files:
        # A folder containing TIFFs directly is one position. This avoids also
        # discovering unrelated result/QC subfolders below it.
        candidate_dirs = [root]
    else:
        candidate_dirs = []
        for directory in _iter_directories_bounded(root, max_depth=4):
            if directory == root or directory.name.startswith(".") or directory.name in {"crop_qc", "logs"}:
                continue
            files = matching_tiffs(directory)
            if files:
                candidate_dirs.append(directory)

    if not candidate_dirs:
        result.issues.append({
            "type": "missing input TIFF",
            "path": str(root),
            "message": f"No TIFF files matched {cfg['tiff_glob_pattern']!r}.",
        })
        return result

    seen_positions: dict[str, Path] = {}
    for directory in sorted(candidate_dirs, key=lambda p: str(p)):
        position = directory.name
        previous = seen_positions.get(position)
        if previous is not None and previous != directory:
            result.issues.append({
                "type": "ambiguous position folder",
                "path": f"{previous}; {directory}",
                "message": f"Two input folders have the same exact position name {position!r}.",
            })
            continue
        seen_positions[position] = directory
        if selected_positions is not None and position not in selected_positions:
            continue

        omes = sorted((p for p in directory.glob(str(cfg["ome_companion_glob_pattern"])) if p.is_file()), key=lambda p: p.name)
        ome_path = str(omes[0].resolve()) if len(omes) == 1 else ""
        if len(omes) > 1:
            result.issues.append({
                "type": "duplicate OME XML",
                "path": "; ".join(str(p.resolve()) for p in omes),
                "message": f"Position {position!r} has multiple OME companions.",
            })
        elif len(omes) == 0 and cfg.get("single_stage") == "registration":
            # Registration can still run in voxel units, but physical shift
            # reporting will not be available without spacing metadata.
            result.issues.append({
                "type": "warning missing OME XML",
                "path": str(directory),
                "message": "Registration will use voxel-unit spacing because no OME companion was found.",
            })

        records: dict[tuple[str, str], list[Path]] = {}
        for path in matching_tiffs(directory):
            tm = time_re.search(path.name)
            cm = channel_re.search(path.name)
            timepoint = _named_or_first(tm, "timepoint") if tm else None
            channel = _named_or_first(cm, "channel") if cm else None
            if timepoint is None or channel is None:
                reasons = []
                if tm is None:
                    reasons.append("unsupported filename: timepoint not parsed")
                if cm is None:
                    reasons.append("unsupported filename: channel not parsed")
                message = "; ".join(reasons)
                result.issues.append({"type": "unsupported filename", "path": str(path.resolve()), "message": message})
                result.rows.append(DiscoveryRow(
                    position=position,
                    settings="",
                    timepoint=timepoint or "",
                    channel=channel or "",
                    input_path=str(path.resolve()),
                    ome_path=ome_path,
                    pairing_status=message,
                    source_folder=str(directory.resolve()),
                ))
                continue
            if timepoint.isdigit() and not first_tp <= int(timepoint) <= last_tp:
                continue
            if selected_channels is not None and channel not in selected_channels:
                continue
            records.setdefault((timepoint, channel), []).append(path)

        if not records:
            result.issues.append({
                "type": "empty position",
                "path": str(directory),
                "message": "No TIFFs remained after applying timepoint/channel selections.",
            })

        for (timepoint, channel), paths in sorted(records.items(), key=lambda item: (_time_sort(item[0][0]), item[0][1])):
            status = "ok" if len(paths) == 1 else "duplicate input TIFF"
            if len(paths) > 1:
                result.issues.append({
                    "type": "duplicate input TIFF",
                    "path": str(directory),
                    "message": f"position={position}, timepoint={timepoint}, channel={channel}",
                })
            path_text = "; ".join(str(p.resolve()) for p in paths)
            result.rows.append(DiscoveryRow(
                position=position,
                settings="",
                timepoint=timepoint,
                channel=channel,
                input_path=path_text,
                fused_path=path_text,
                ome_path=ome_path,
                pairing_status=status,
                source_folder=str(directory.resolve()),
            ))
        result.settings_by_position[position] = ""
        if ome_path:
            result.ome_by_position[position] = ome_path

    result.positions = sorted({r.position for r in result.rows if r.position})
    result.channels = sorted({r.channel for r in result.rows if r.channel})
    if selected_positions is not None:
        for position in sorted(set(selected_positions) - set(result.positions)):
            result.issues.append({"type": "missing selected position", "path": str(root), "message": position})
    if selected_channels is not None:
        for channel in sorted(set(selected_channels) - set(result.channels)):
            result.issues.append({"type": "missing selected channel", "path": str(root), "message": channel})
    return result


def _scan_fused_dataset(config: dict[str, Any]) -> ScanResult:
    """Discover already-fused TIFF stacks without requiring View1/View2 pairs.

    The selected input may be either one folder containing fused TIFFs for one
    position, or a parent folder containing one fused-TIFF folder per position.
    Position names are the exact source-folder names so downstream standalone
    scripts discover and write the same names without copying image data.
    """
    cfg = merged_config(config)
    root = Path(cfg["input_dataset_folder"]).expanduser().resolve()
    result = ScanResult()
    if not root.is_dir():
        result.issues.append({"type": "configuration", "path": str(root), "message": "Input dataset folder does not exist."})
        return result

    try:
        folder_re = re.compile(str(cfg["position_folder_pattern"]))
        time_re = re.compile(expand_regex_template(str(cfg["timepoint_regex"]), cfg))
        channel_re = re.compile(expand_regex_template(str(cfg["fused_channel_regex"]), cfg))
    except re.error as exc:
        result.issues.append({"type": "configuration", "path": "", "message": f"Invalid regular expression: {exc}"})
        return result

    selected_positions = _selection(cfg.get("selected_positions"))
    selected_channels = _selection(cfg.get("selected_channels"))
    first_tp = int(cfg["first_timepoint"])
    last_tp = int(cfg["last_timepoint"])
    fused_token = str(cfg["fused_token"])

    candidate_dirs: list[Path] = []
    for directory in _iter_directories_bounded(root, max_depth=4):
        try:
            files = [p for p in directory.glob(str(cfg["tiff_glob_pattern"])) if p.is_file()]
        except OSError as exc:
            result.issues.append({"type": "configuration", "path": str(directory), "message": f"Could not list directory: {exc}"})
            continue
        if any(fused_token in p.stem for p in files):
            candidate_dirs.append(directory)

    if not candidate_dirs:
        result.issues.append({
            "type": "missing fused TIFF",
            "path": str(root),
            "message": f"No TIFF filenames containing fused token {fused_token!r} were found.",
        })
        return result

    seen_positions: dict[str, Path] = {}
    for directory in sorted(candidate_dirs, key=lambda p: str(p)):
        # Exact folder names are retained as positions. This matches the
        # standalone registration/BGS/crop scripts' own discovery behavior.
        position = directory.name
        settings = ""
        try:
            rel = directory.relative_to(root).as_posix()
        except ValueError:
            rel = directory.name
        match = folder_re.fullmatch(directory.name) or folder_re.fullmatch(rel)
        if match:
            settings = match.groupdict().get("settings", "") or ""

        previous = seen_positions.get(position)
        if previous is not None and previous != directory:
            result.issues.append({
                "type": "ambiguous position folder",
                "path": f"{previous}; {directory}",
                "message": f"Two fused-data folders have the same exact folder name {position!r}.",
            })
            continue
        seen_positions[position] = directory
        if selected_positions is not None and position not in selected_positions:
            continue

        tiffs = sorted(
            (p for p in directory.glob(str(cfg["tiff_glob_pattern"])) if p.is_file() and fused_token in p.stem),
            key=lambda p: p.name,
        )
        omes = sorted((p for p in directory.glob(str(cfg["ome_companion_glob_pattern"])) if p.is_file()), key=lambda p: p.name)
        if len(omes) == 0:
            ome_status = "missing OME XML"
            ome_path = ""
            result.issues.append({"type": ome_status, "path": str(directory), "message": "No OME companion matched the configured glob."})
        elif len(omes) > 1:
            ome_status = "duplicate OME XML"
            ome_path = "; ".join(str(p.resolve()) for p in omes)
            result.issues.append({"type": ome_status, "path": str(directory), "message": ome_path})
        else:
            ome_status = "ok"
            ome_path = str(omes[0].resolve())

        records: dict[tuple[str, str], list[Path]] = {}
        for path in tiffs:
            tm = time_re.search(path.name)
            cm = channel_re.search(path.name)
            timepoint = _named_or_first(tm, "timepoint") if tm else None
            channel = _named_or_first(cm, "channel") if cm else None
            if timepoint is None or channel is None:
                reasons = []
                if tm is None:
                    reasons.append("unsupported filename: timepoint not parsed")
                if cm is None:
                    reasons.append("unsupported filename: fused channel not parsed")
                message = "; ".join(reasons)
                result.issues.append({"type": "unsupported filename", "path": str(path.resolve()), "message": message})
                result.rows.append(DiscoveryRow(
                    position=position,
                    settings=settings,
                    timepoint=timepoint or "",
                    channel=channel or "",
                    fused_path=str(path.resolve()),
                    ome_path=ome_path,
                    pairing_status=message,
                    source_folder=str(directory.resolve()),
                ))
                continue
            if timepoint.isdigit() and not first_tp <= int(timepoint) <= last_tp:
                continue
            if selected_channels is not None and channel not in selected_channels:
                continue
            records.setdefault((timepoint, channel), []).append(path)

        if not records:
            result.issues.append({
                "type": "empty position",
                "path": str(directory),
                "message": "No fused TIFFs remained after applying timepoint/channel selections.",
            })

        for (timepoint, channel), paths in sorted(records.items(), key=lambda item: (_time_sort(item[0][0]), item[0][1])):
            statuses: list[str] = []
            if len(paths) > 1:
                statuses.append("duplicate fused TIFF")
            if ome_status != "ok":
                statuses.append(ome_status)
            status = "; ".join(statuses) if statuses else "ok"
            if statuses:
                for issue_type in sorted(set(statuses)):
                    result.issues.append({
                        "type": issue_type,
                        "path": str(directory),
                        "message": f"position={position}, timepoint={timepoint}, channel={channel}",
                    })
            result.rows.append(DiscoveryRow(
                position=position,
                settings=settings,
                timepoint=timepoint,
                channel=channel,
                fused_path="; ".join(str(p.resolve()) for p in paths),
                ome_path=ome_path,
                pairing_status=status,
                source_folder=str(directory.resolve()),
            ))
        result.settings_by_position[position] = settings
        if ome_status == "ok":
            result.ome_by_position[position] = ome_path

    result.positions = sorted({r.position for r in result.rows if r.position})
    result.channels = sorted({r.channel for r in result.rows if r.channel})
    if selected_positions is not None:
        for position in sorted(set(selected_positions) - set(result.positions)):
            result.issues.append({"type": "missing selected position", "path": str(root), "message": position})
    if selected_channels is not None:
        for channel in sorted(set(selected_channels) - set(result.channels)):
            result.issues.append({"type": "missing selected channel", "path": str(root), "message": channel})
    return result


def scan_dataset(config: dict[str, Any]) -> ScanResult:
    """Scan names without loading complete TIFF volumes."""
    cfg = merged_config(config)
    if _uses_generic_stage_input(cfg):
        return _scan_stage_ready_dataset(cfg)
    if cfg.get("starting_data_type") == INPUT_FUSED:
        return _scan_fused_dataset(cfg)
    root = Path(cfg["input_dataset_folder"]).expanduser().resolve()
    result = ScanResult()
    if not root.is_dir():
        result.issues.append({"type": "configuration", "path": str(root), "message": "Input dataset folder does not exist."})
        return result

    try:
        folder_re = re.compile(str(cfg["position_folder_pattern"]))
        time_re = re.compile(expand_regex_template(str(cfg["timepoint_regex"]), cfg))
        channel_re = re.compile(expand_regex_template(str(cfg["channel_regex"]), cfg))
    except re.error as exc:
        result.issues.append({"type": "configuration", "path": "", "message": f"Invalid regular expression: {exc}"})
        return result

    settings_pattern = str(cfg["settings_pattern"])
    candidates: list[tuple[Path, str, str]] = []
    for directory in _iter_directories_bounded(root, max_depth=4):
        try:
            rel = directory.relative_to(root).as_posix()
        except ValueError:
            continue
        match = folder_re.fullmatch(directory.name) or folder_re.fullmatch(rel)
        if not match:
            continue
        position = match.groupdict().get("position")
        settings = match.groupdict().get("settings", "")
        if not position:
            result.issues.append({"type": "unsupported filename", "path": str(directory), "message": "Position-folder pattern must provide a named 'position' group."})
            continue
        if settings_pattern and not fnmatch.fnmatchcase(settings, settings_pattern):
            continue
        candidates.append((directory, position, settings))

    # Detect ambiguous settings folders before pairing files.
    dirs_by_position: dict[str, list[tuple[Path, str]]] = {}
    for directory, position, settings in candidates:
        dirs_by_position.setdefault(position, []).append((directory, settings))
    ambiguous = {p for p, dirs in dirs_by_position.items() if len(dirs) > 1}
    for position in sorted(ambiguous):
        dirs = dirs_by_position[position]
        result.issues.append({
            "type": "ambiguous settings",
            "path": "; ".join(str(d[0]) for d in dirs),
            "message": f"Position {position!r} has multiple folders matching settings pattern {settings_pattern!r}; use a more specific pattern.",
        })

    selected_positions = _selection(cfg.get("selected_positions"))
    selected_channels = _selection(cfg.get("selected_channels"))
    first_tp = int(cfg["first_timepoint"])
    last_tp = int(cfg["last_timepoint"])

    for directory, position, settings in candidates:
        if position in ambiguous:
            continue
        if selected_positions is not None and position not in selected_positions:
            continue
        tiffs = sorted((p for p in directory.glob(str(cfg["tiff_glob_pattern"])) if p.is_file()), key=lambda p: p.name)
        omes = sorted((p for p in directory.glob(str(cfg["ome_companion_glob_pattern"])) if p.is_file()), key=lambda p: p.name)
        if not tiffs:
            result.issues.append({"type": "empty position", "path": str(directory), "message": f"No TIFFs matched {cfg['tiff_glob_pattern']!r}."})
            continue
        if len(omes) == 0:
            ome_status = "missing OME XML"
            ome_path = ""
            result.issues.append({"type": ome_status, "path": str(directory), "message": "No OME companion matched the configured glob."})
        elif len(omes) > 1:
            ome_status = "duplicate OME XML"
            ome_path = "; ".join(str(p.resolve()) for p in omes)
            result.issues.append({"type": ome_status, "path": str(directory), "message": ome_path})
        else:
            ome_status = "ok"
            ome_path = str(omes[0].resolve())

        pairs: dict[tuple[str, str], dict[str, list[str]]] = {}
        view_key_sets: dict[str, set[tuple[str, str]]] = {"view1": set(), "view2": set()}
        for path in tiffs:
            view, view_error = _view_kind(path.name, str(cfg["view1_token"]), str(cfg["view2_token"]))
            tm = time_re.search(path.name)
            cm = channel_re.search(path.name)
            timepoint = _named_or_first(tm, "timepoint") if tm else None
            channel = _named_or_first(cm, "channel") if cm else None
            if view_error or timepoint is None or channel is None:
                reasons = [x for x in [view_error, None if tm else "unsupported filename: timepoint not parsed", None if cm else "unsupported filename: channel not parsed"] if x]
                message = "; ".join(reasons)
                result.issues.append({"type": "unsupported filename", "path": str(path.resolve()), "message": message})
                result.rows.append(DiscoveryRow(position, settings, timepoint or "", channel or "", pairing_status=message, ome_path=ome_path, source_folder=str(directory.resolve())))
                continue
            if timepoint.isdigit() and not first_tp <= int(timepoint) <= last_tp:
                continue
            if selected_channels is not None and channel not in selected_channels:
                continue
            key = (timepoint, channel)
            pairs.setdefault(key, {"view1": [], "view2": []})[view].append(str(path.resolve()))
            view_key_sets[view].add(key)

        # Report deterministic key-set mismatches in addition to per-pair missing views.
        only1 = view_key_sets["view1"] - view_key_sets["view2"]
        only2 = view_key_sets["view2"] - view_key_sets["view1"]
        for tp, ch in sorted(only1 | only2, key=lambda k: (_time_sort(k[0]), k[1])):
            same_channel_other_tp = any(k[1] == ch and k[0] != tp for k in (only1 | only2))
            same_tp_other_channel = any(k[0] == tp and k[1] != ch for k in (only1 | only2))
            if same_channel_other_tp:
                result.issues.append({"type": "mismatched timepoint", "path": str(directory), "message": f"Unpaired view key timepoint={tp}, channel={ch}."})
            if same_tp_other_channel:
                result.issues.append({"type": "mismatched channel", "path": str(directory), "message": f"Unpaired view key timepoint={tp}, channel={ch}."})

        for (timepoint, channel), views in sorted(pairs.items(), key=lambda item: (_time_sort(item[0][0]), item[0][1])):
            statuses: list[str] = []
            if len(views["view1"]) == 0:
                statuses.append("missing View 1")
            elif len(views["view1"]) > 1:
                statuses.append("duplicate view")
            if len(views["view2"]) == 0:
                statuses.append("missing View 2")
            elif len(views["view2"]) > 1:
                statuses.append("duplicate view")
            if ome_status != "ok":
                statuses.append(ome_status)
            status = "; ".join(statuses) if statuses else "ok"
            if statuses:
                for issue_type in sorted(set(statuses)):
                    result.issues.append({"type": issue_type, "path": str(directory), "message": f"position={position}, timepoint={timepoint}, channel={channel}"})
            result.rows.append(DiscoveryRow(
                position=position,
                settings=settings,
                timepoint=timepoint,
                channel=channel,
                view1_path="; ".join(views["view1"]),
                view2_path="; ".join(views["view2"]),
                ome_path=ome_path,
                pairing_status=status,
                source_folder=str(directory.resolve()),
            ))
        result.settings_by_position[position] = settings
        if ome_status == "ok":
            result.ome_by_position[position] = ome_path

    result.positions = sorted({r.position for r in result.rows if r.position})
    result.channels = sorted({r.channel for r in result.rows if r.channel})
    if selected_positions is not None:
        missing = sorted(set(selected_positions) - set(result.positions))
        for position in missing:
            result.issues.append({"type": "missing selected position", "path": str(root), "message": position})
    if selected_channels is not None:
        missing = sorted(set(selected_channels) - set(result.channels))
        for channel in missing:
            result.issues.append({"type": "missing selected channel", "path": str(root), "message": channel})
    return result


def discovery_table_text(scan: ScanResult) -> str:
    generic_mode = any(r.input_path for r in scan.rows)
    fused_mode = any(r.fused_path for r in scan.rows)
    if generic_mode:
        headers = ["Position", "Timepoint", "Channel", "Input TIFF path", "OME path", "Status"]
        rows = [[r.position, r.timepoint, r.channel, r.input_path, r.ome_path, r.pairing_status] for r in scan.rows]
    elif fused_mode:
        headers = ["Position", "Settings", "Timepoint", "Channel", "Fused TIFF path", "OME path", "Status"]
        rows = [[r.position, r.settings, r.timepoint, r.channel, r.fused_path, r.ome_path, r.pairing_status] for r in scan.rows]
    else:
        headers = ["Position", "Settings", "Timepoint", "Channel", "View 1 path", "View 2 path", "OME path", "Pairing status"]
        rows = [[r.position, r.settings, r.timepoint, r.channel, r.view1_path, r.view2_path, r.ome_path, r.pairing_status] for r in scan.rows]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = min(60, max(widths[i], len(str(value))))
    def fmt(row: Iterable[Any]) -> str:
        vals = []
        for i, value in enumerate(row):
            text = str(value)
            if len(text) > widths[i]:
                text = text[: max(0, widths[i] - 1)] + "…"
            vals.append(text.ljust(widths[i]))
        return " | ".join(vals)
    return "\n".join([fmt(headers), "-+-".join("-" * w for w in widths), *(fmt(row) for row in rows)])


def _header_validation_rows(valid_rows: list[DiscoveryRow], mode: str, samples_per_group: int) -> list[DiscoveryRow]:
    if mode == "names_only":
        return []
    if mode == "full":
        return list(valid_rows)
    if mode != "quick":
        raise ValueError(f"Unknown validation mode: {mode!r}")

    selected: list[DiscoveryRow] = []
    groups: dict[tuple[str, str], list[DiscoveryRow]] = {}
    for row in valid_rows:
        groups.setdefault((row.position, row.channel), []).append(row)
    sample_count = max(1, int(samples_per_group))
    for rows in groups.values():
        ordered = sorted(rows, key=lambda r: _time_sort(r.timepoint))
        if len(ordered) <= sample_count:
            selected.extend(ordered)
            continue
        if sample_count == 1:
            indices = [0]
        else:
            indices = sorted({round(i * (len(ordered) - 1) / (sample_count - 1)) for i in range(sample_count)})
        selected.extend(ordered[i] for i in indices)
    return selected


def resolved_full_stage_sequence(config: dict[str, Any]) -> list[str]:
    """Return the complete workflow sequence for the selected starting data."""
    cfg = merged_config(config)
    if cfg["processing_order"] == ORDER_BGS_CROP:
        sequence = ["fusion", "registration", "background_subtraction"]
        if cfg.get("enable_denoising", False):
            sequence.append("denoise")
        sequence.append("crop")
    else:
        sequence = ["fusion", "registration", "crop", "background_subtraction"]
        if cfg.get("enable_denoising", False):
            sequence.append("denoise")
    if cfg.get("starting_data_type") == INPUT_FUSED:
        sequence = sequence[sequence.index("registration"):]
    return sequence


def resolved_stage_sequence(config: dict[str, Any]) -> list[str]:
    """Return the exact stages that will execute in this run."""
    cfg = merged_config(config)
    if cfg.get("run_mode") == RUN_SINGLE:
        return [str(cfg.get("single_stage", "registration"))]
    return resolved_full_stage_sequence(cfg)


def validate_configuration(
    config: dict[str, Any],
    scan: ScanResult | None = None,
    validation_mode: str | None = None,
) -> list[str]:
    """Validate naming and, optionally, TIFF headers.

    ``names_only`` never opens TIFFs. ``quick`` opens sampled TIFFs per
    position/channel. ``full`` opens every selected input TIFF header.
    """
    cfg = merged_config(config)
    mode = str(validation_mode or cfg.get("validation_mode", "quick")).strip().lower()
    errors: list[str] = []
    run_mode = str(cfg.get("run_mode", RUN_FULL))
    single_stage = str(cfg.get("single_stage", "registration"))
    raw_source = _uses_raw_pair_discovery(cfg)
    generic_source = _uses_generic_stage_input(cfg)
    fused_source = not raw_source and not generic_source

    if run_mode not in {RUN_FULL, RUN_SINGLE}:
        errors.append(f"Run mode is not recognized: {run_mode!r}.")
    if single_stage not in STAGES:
        errors.append(f"Single stage is not recognized: {single_stage!r}.")
    if run_mode == RUN_FULL and cfg.get("starting_data_type") not in {INPUT_RAW, INPUT_FUSED}:
        errors.append(f"Starting data type is not recognized: {cfg.get('starting_data_type')!r}.")
    if mode not in {"names_only", "quick", "full"}:
        errors.append("Validation mode must be 'quick' or 'full'.")
        mode = "quick"
    if not Path(cfg["input_dataset_folder"]).expanduser().is_dir():
        errors.append("Input dataset folder does not exist.")
    if int(cfg["first_timepoint"]) > int(cfg["last_timepoint"]):
        errors.append("First timepoint cannot be greater than last timepoint.")
    if cfg["processing_order"] not in {ORDER_BGS_CROP, ORDER_CROP_BGS}:
        errors.append("Processing order is not recognized.")
    if raw_source and str(cfg["view1_token"]) == str(cfg["view2_token"]):
        errors.append("View 1 and View 2 tokens must differ.")
    if str(cfg["bgs_output_dtype"]) not in {"same", "uint16", "uint8", "float32"}:
        errors.append("Unsupported BGS output dtype.")
    if str(cfg["denoise_output_dtype"]) not in {"same", "uint16", "uint8", "float32"}:
        errors.append("Unsupported denoising output dtype.")
    if str(cfg["denoise_method"]) not in {"none", "median2d", "gaussian2d", "median_gaussian2d", "gaussian3d"}:
        errors.append("Unsupported denoising method.")
    if int(cfg["denoise_median_size"]) < 1:
        errors.append("Denoising median size must be at least 1.")
    if float(cfg["denoise_sigma_xy"]) < 0 or float(cfg["denoise_sigma_z"]) < 0:
        errors.append("Denoising sigma values cannot be negative.")
    if float(cfg["denoise_intensity_floor"]) < 0:
        errors.append("Denoising intensity floor cannot be negative.")

    scan = scan or scan_dataset(cfg)
    valid_rows = [r for r in scan.rows if r.pairing_status == "ok"]
    if not valid_rows:
        if raw_source:
            errors.append("No complete View 1/View 2 TIFF pairs were discovered.")
        elif generic_source:
            errors.append("No valid stage-ready TIFF files were discovered.")
        else:
            errors.append("No valid fused TIFF files were discovered.")

    blocking_types = {
        "configuration", "ambiguous settings", "ambiguous position folder",
        "unsupported filename", "mismatched channel", "mismatched timepoint",
        "missing selected position", "missing selected channel", "empty position",
    }
    if raw_source:
        blocking_types.update({"missing View 1", "missing View 2", "duplicate view", "missing OME XML", "duplicate OME XML"})
    elif generic_source:
        blocking_types.update({"missing input TIFF", "duplicate input TIFF", "duplicate OME XML"})
    else:
        blocking_types.update({"missing fused TIFF", "duplicate fused TIFF", "missing OME XML", "duplicate OME XML"})
    for issue in scan.issues:
        if issue.get("type") in blocking_types:
            errors.append(f"{issue.get('type')}: {issue.get('message')} ({issue.get('path')})")

    channels = sorted({r.channel for r in valid_rows})
    ref = str(cfg.get("reference_channel", "")).strip()
    if raw_source:
        if ref:
            if ref not in channels:
                errors.append(f"Reference channel {ref!r} was not discovered among selected channels {channels}.")
            for position in sorted({r.position for r in valid_rows}):
                by_tp = {r.timepoint for r in valid_rows if r.position == position}
                ref_tp = {r.timepoint for r in valid_rows if r.position == position and r.channel == ref}
                missing = sorted(by_tp - ref_tp, key=_time_sort)
                if missing:
                    errors.append(f"Position {position!r} lacks reference channel {ref!r} at timepoints {missing}.")
        elif len(channels) != 1:
            errors.append(f"Reference channel is required because multiple channels were discovered: {channels}.")

    # Every selected file is checked for existence and nonzero size. This is
    # cheap compared with loading complete TIFF stacks over network storage.
    for row in valid_rows:
        if raw_source:
            path_items = (("View 1", row.view1_path), ("View 2", row.view2_path))
        else:
            path_items = (("Input", row.input_path or row.fused_path),)
        for label, raw_path in path_items:
            path = Path(raw_path)
            try:
                if not path.is_file():
                    errors.append(f"{label} TIFF is missing: {path}")
                elif path.stat().st_size <= 0:
                    errors.append(f"{label} TIFF is empty: {path}")
            except OSError as exc:
                errors.append(f"Could not stat {label} TIFF {path}: {exc}")

    header_rows = _header_validation_rows(
        valid_rows,
        mode,
        int(cfg.get("quick_validation_samples_per_position_channel", 2)),
    )
    shapes_by_position: dict[str, set[tuple[int, int, int]]] = {}
    verify_all_pages = mode == "full"
    for row in header_rows:
        try:
            if raw_source:
                info1 = _probe_tiff(Path(row.view1_path), verify_all_pages=verify_all_pages)
                info2 = _probe_tiff(Path(row.view2_path), verify_all_pages=verify_all_pages)
                shapes_by_position.setdefault(row.position, set()).add(tuple(info1["shape_zyx"]))
                if info1["shape_zyx"] != info2["shape_zyx"]:
                    errors.append(f"View shape mismatch at position={row.position}, timepoint={row.timepoint}, channel={row.channel}: {info1['shape_zyx']} vs {info2['shape_zyx']}.")
                if info1["dtype"] != info2["dtype"]:
                    errors.append(f"View dtype mismatch at position={row.position}, timepoint={row.timepoint}, channel={row.channel}: {info1['dtype']} vs {info2['dtype']}.")
                if info1["dtype"] != "uint16":
                    errors.append(f"Fusion input {row.view1_path} has dtype {info1['dtype']}; the preserved legacy fusion writer outputs uint16. Convert explicitly before the pipeline rather than allowing a silent dtype change.")
            else:
                source = Path(row.input_path or row.fused_path)
                info1 = _probe_tiff(source, verify_all_pages=verify_all_pages)
                shapes_by_position.setdefault(row.position, set()).add(tuple(info1["shape_zyx"]))
        except Exception as exc:
            errors.append(f"Could not validate TIFF headers for position={row.position}, timepoint={row.timepoint}, channel={row.channel}: {exc}")
    for position, shapes in shapes_by_position.items():
        if len(shapes) > 1:
            qualifier = "sampled " if mode == "quick" else ""
            errors.append(f"Position {position!r} has differing input shapes across {qualifier}selected timepoints/channels: {sorted(shapes)}. A stage run requires a common frame per position.")
    return list(dict.fromkeys(errors))

def _replace_view_token(name: str, view_token: str, fused_token: str) -> str:
    path = Path(name)
    stem = path.stem
    marker = "_" + view_token
    if stem.endswith(marker):
        stem = stem[: -len(marker)] + "_" + fused_token
    else:
        index = stem.rfind(view_token)
        if index < 0:
            raise ValueError(f"Cannot replace view token {view_token!r} in {name!r}")
        stem = stem[:index] + fused_token + stem[index + len(view_token):]
    return stem + path.suffix


def make_fusion_jobs(config: dict[str, Any], scan: ScanResult) -> dict[str, Any]:
    cfg = merged_config(config)
    positions: dict[str, dict[str, Any]] = {}
    for row in scan.rows:
        if row.pairing_status != "ok":
            continue
        positions.setdefault(row.position, {"ome_path": row.ome_path, "settings": row.settings, "records": []})["records"].append({
            "timepoint": row.timepoint,
            "channel": row.channel,
            "view1": row.view1_path,
            "view2": row.view2_path,
            "output_name": _replace_view_token(Path(row.view1_path).name, str(cfg["view1_token"]), str(cfg["fused_token"])),
        })
    return {"positions": positions}


def make_imported_fused_jobs(config: dict[str, Any], scan: ScanResult) -> dict[str, Any]:
    """Build downstream jobs from already-fused TIFFs without copying them."""
    positions: dict[str, dict[str, Any]] = {}
    for row in scan.rows:
        if row.pairing_status != "ok":
            continue
        pdata = positions.setdefault(row.position, {
            "ome_path": row.ome_path,
            "settings": row.settings,
            "source_folder": row.source_folder,
            "records": [],
        })
        pdata["records"].append({
            "timepoint": row.timepoint,
            "channel": row.channel,
            "fused_path": row.fused_path,
            "output_name": Path(row.fused_path).name,
        })
    return {"positions": positions}


def make_stage_ready_jobs(config: dict[str, Any], scan: ScanResult) -> dict[str, Any]:
    """Build one-stage jobs from generic TIFF folders without copying inputs."""
    positions: dict[str, dict[str, Any]] = {}
    for row in scan.rows:
        if row.pairing_status != "ok":
            continue
        source_path = row.input_path or row.fused_path
        pdata = positions.setdefault(row.position, {
            "ome_path": row.ome_path,
            "settings": row.settings,
            "source_folder": row.source_folder,
            "records": [],
        })
        pdata["records"].append({
            "timepoint": row.timepoint,
            "channel": row.channel,
            "input_path": source_path,
            "output_name": Path(source_path).name,
        })
    return {"positions": positions}


def make_pipeline_jobs(config: dict[str, Any], scan: ScanResult) -> dict[str, Any]:
    cfg = merged_config(config)
    if _uses_generic_stage_input(cfg):
        return make_stage_ready_jobs(cfg, scan)
    if cfg.get("starting_data_type") == INPUT_FUSED:
        return make_imported_fused_jobs(cfg, scan)
    return make_fusion_jobs(cfg, scan)


def software_versions() -> dict[str, str]:
    packages = ["numpy", "scipy", "tifffile", "opencv-python", "scikit-image", "matplotlib", "Pillow"]
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _quote_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def _probe_tiff(path: Path, verify_all_pages: bool = False) -> dict[str, Any]:
    if tifffile is None:
        raise RuntimeError("tifffile is required for TIFF validation")
    with tifffile.TiffFile(str(path)) as tf:
        compression = "unknown"
        if tf.pages:
            comp = getattr(tf.pages[0], "compression", None)
            compression = getattr(comp, "name", str(comp)) if comp is not None else "none"
        if len(tf.pages) >= 2 and len(tf.pages[0].shape) == 2:
            first_shape = tuple(tf.pages[0].shape)
            first_dtype = str(tf.pages[0].dtype)
            if verify_all_pages:
                indices = range(len(tf.pages))
            else:
                indices = sorted({0, len(tf.pages) // 2, len(tf.pages) - 1})
            if all(tuple(tf.pages[i].shape) == first_shape and str(tf.pages[i].dtype) == first_dtype for i in indices):
                y, x = first_shape
                shape = (len(tf.pages), int(y), int(x))
                dtype = first_dtype
            else:
                series = tf.series[0]
                raw_shape = tuple(int(v) for v in series.shape)
                axes = getattr(series, "axes", "") or ""
                shape, dtype = _series_shape_dtype(path, series, raw_shape, axes)
        else:
            series = tf.series[0]
            raw_shape = tuple(int(v) for v in series.shape)
            axes = getattr(series, "axes", "") or ""
            shape, dtype = _series_shape_dtype(path, series, raw_shape, axes)
    voxels = int(shape[0] * shape[1] * shape[2])
    size = path.stat().st_size
    return {"shape_zyx": list(shape), "dtype": dtype, "compression": compression, "bytes": size, "voxel_count": voxels, "bytes_per_voxel": (size / voxels if voxels else None)}


def _series_shape_dtype(path: Path, series: Any, raw_shape: tuple[int, ...], axes: str) -> tuple[tuple[int, int, int], str]:
    shape_list = list(raw_shape)
    axes_list = list(axes)
    for i in range(len(axes_list) - 1, -1, -1):
        if axes_list[i] not in {"Z", "Y", "X"} and shape_list[i] == 1:
            del axes_list[i]
            del shape_list[i]
    axes_now = "".join(axes_list)
    if len(shape_list) == 3 and set(axes_now) == {"Z", "Y", "X"}:
        shape = tuple(shape_list[axes_now.index(ax)] for ax in "ZYX")
    elif len(shape_list) == 3:
        shape = tuple(shape_list)
    elif len(shape_list) == 2:
        shape = (1, shape_list[0], shape_list[1])
    else:
        raise ValueError(f"Cannot determine ZYX shape from {path}: shape={raw_shape}, axes={axes}")
    return (int(shape[0]), int(shape[1]), int(shape[2])), str(series.dtype)

def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _metadata_record(stage: str, output_path: Path) -> dict[str, Any] | None:
    suffix = {"background_subtraction": ".bgsub.json", "denoise": ".denoise.json", "crop": ".crop.json"}.get(stage)
    if suffix:
        return _read_json(output_path.with_name(output_path.stem + suffix))
    if stage == "registration":
        return _read_json(output_path.with_name(output_path.stem + ".json"))
    return None


def _likely_size_driver(stage: str, voxel_ratio: float | None, file_ratio: float | None,
                        in_dtype: str, out_dtype: str, in_comp: str, out_comp: str,
                        bpv_before: float | None, bpv_after: float | None) -> str:
    if in_dtype != out_dtype:
        return "dtype conversion"
    if voxel_ratio is not None and voxel_ratio < 0.95:
        return "cropping and reduced voxel dimensions"
    if stage == "background_subtraction" and file_ratio is not None and file_ratio < 0.95:
        return "background subtraction and/or improved compression"
    if stage == "denoise" and file_ratio is not None and file_ratio < 0.95:
        return "denoising, intensity-floor sparsification, and/or improved compression"
    if in_comp != out_comp:
        return "different compression or TIFF-writing layout"
    if bpv_before and bpv_after and abs(bpv_after / bpv_before - 1.0) > 0.1:
        return "different TIFF-writing layout or compressibility"
    return "little size change or mixed effects"


def build_size_summary(output_root: Path, stage_transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transition in stage_transitions:
        stage = transition["stage"]
        for item in transition["files"]:
            inp = Path(item["input"])
            out = Path(item["output"])
            if not inp.is_file() or not out.is_file() or inp.stat().st_size == 0 or out.stat().st_size == 0:
                continue
            meta = _metadata_record(stage, out) or {}
            in_info = _probe_tiff(inp)
            out_info = _probe_tiff(out)
            input_shape = meta.get("input_shape_zyx", in_info["shape_zyx"])
            output_shape = meta.get("output_shape_zyx", meta.get("shape_zyx", out_info["shape_zyx"]))
            input_voxels = int(input_shape[0] * input_shape[1] * input_shape[2])
            output_voxels = int(output_shape[0] * output_shape[1] * output_shape[2])
            input_bytes = int(meta.get("input_bytes", in_info["bytes"]))
            output_bytes = int(meta.get("output_bytes", out_info["bytes"]))
            in_dtype = str(meta.get("input_dtype", in_info["dtype"]))
            out_dtype = str(meta.get("output_dtype", meta.get("dtype", out_info["dtype"])))
            voxel_ratio = output_voxels / input_voxels if input_voxels else None
            file_ratio = output_bytes / input_bytes if input_bytes else None
            bpv_before = input_bytes / input_voxels if input_voxels else None
            bpv_after = output_bytes / output_voxels if output_voxels else None
            in_comp = in_info["compression"]
            out_comp = str(meta.get("compression", out_info["compression"]))
            rows.append({
                "stage": stage,
                "position": item["position"],
                "file": out.name,
                "input_path": str(inp),
                "output_path": str(out),
                "input_shape_zyx": "x".join(map(str, input_shape)),
                "output_shape_zyx": "x".join(map(str, output_shape)),
                "input_voxel_count": input_voxels,
                "output_voxel_count": output_voxels,
                "voxel_ratio": voxel_ratio,
                "voxel_ratio_output_vs_input": voxel_ratio,
                "input_file_size": input_bytes,
                "output_file_size": output_bytes,
                "file_size_ratio": file_ratio,
                "input_file_size_bytes": input_bytes,
                "output_file_size_bytes": output_bytes,
                "file_size_ratio_output_vs_input": file_ratio,
                "bytes_per_voxel_before": bpv_before,
                "bytes_per_voxel_after": bpv_after,
                "compression": out_comp,
                "dtype": out_dtype,
                "input_compression": in_comp,
                "output_compression": out_comp,
                "input_dtype": in_dtype,
                "output_dtype": out_dtype,
                "likely_size_driver": _likely_size_driver(stage, voxel_ratio, file_ratio, in_dtype, out_dtype, in_comp, out_comp, bpv_before, bpv_after),
                "note": "Compressed black regions still occupy strips/pages, TIFF directories, offsets, and metadata; cropping removes actual voxels and compressed blocks.",
            })
    path = output_root / "pipeline_size_summary.csv"
    fields = list(rows[0].keys()) if rows else [
        "stage", "position", "file", "input_path", "output_path", "input_shape_zyx", "output_shape_zyx",
        "input_voxel_count", "output_voxel_count", "voxel_ratio", "voxel_ratio_output_vs_input",
        "input_file_size", "output_file_size", "file_size_ratio", "input_file_size_bytes",
        "output_file_size_bytes", "file_size_ratio_output_vs_input", "bytes_per_voxel_before", "bytes_per_voxel_after",
        "compression", "dtype", "input_compression", "output_compression", "input_dtype", "output_dtype", "likely_size_driver", "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


class PipelineRunner:
    def __init__(self, config: dict[str, Any], log_callback: Callable[[str], None] | None = None,
                 event_callback: Callable[[dict[str, Any]], None] | None = None,
                 prevalidated_scan: ScanResult | None = None,
                 skip_initial_validation: bool = False):
        self.config = merged_config(config)
        self.prevalidated_scan = prevalidated_scan
        self.skip_initial_validation = bool(skip_initial_validation)
        self.output_root = Path(self.config["output_root"]).expanduser().resolve()
        self.logs_root = self.output_root / "logs"
        self.stage_manifest_root = self.logs_root / "stage_manifests"
        self.log_callback = log_callback
        self.event_callback = event_callback
        self.stop_event = threading.Event()
        self.current_process: subprocess.Popen[str] | None = None
        self.commands: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self.stage_records: dict[str, Any] = {}
        self.completed_files = 0
        self.failed_files = 0
        self.current_timepoint = ""
        self._log_file = None

    def emit(self, message: str) -> None:
        text = message.rstrip("\n")
        print(text, flush=True)
        if self._log_file:
            self._log_file.write(text + "\n")
            self._log_file.flush()
        if self.log_callback:
            self.log_callback(text)

    def event(self, **payload: Any) -> None:
        if self.event_callback:
            self.event_callback(payload)
        print("PIPELINE_EVENT " + json.dumps(payload, default=str), flush=True)

    def request_stop(self) -> None:
        self.stop_event.set()
        self.emit("Stop requested; terminating the active child process safely.")
        self._terminate_current_process()

    def _terminate_current_process(self) -> None:
        proc = self.current_process
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=8)
        except Exception:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass

    def _timepoint_from_text(self, text: str) -> str:
        try:
            pattern = re.compile(expand_regex_template(str(self.config["timepoint_regex"]), self.config))
            match = pattern.search(text)
            if match:
                return _named_or_first(match, "timepoint") or ""
        except re.error:
            pass
        match = re.search(r"(?i)(?:^|[^A-Za-z0-9])(?:t\s*=?|timepoint\s+)(\d+)", text)
        return match.group(1) if match else ""

    def _position_from_text(self, text: str) -> str:
        match = re.search(r"^\[([^]\r\n]+)\]", text.strip())
        return match.group(1).strip() if match else ""

    def _run_command(self, command: list[str], stage: str, position: str = "") -> int:
        if self.stop_event.is_set():
            raise PipelineFailure("Pipeline stopped by user.")
        command_text = _quote_command(command)
        self.emit(f"COMMAND: {command_text}")
        record = {"stage": stage, "position": position, "command": command, "command_text": command_text, "start_time": utc_now()}
        self.commands.append(record)
        if self.config["dry_run"]:
            record.update({"end_time": utc_now(), "returncode": 0, "status": "dry_run"})
            return 0
        kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
            "cwd": str(SCRIPT_DIR.parent),
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True
        elif os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(command, **kwargs)
        self.current_process = proc
        try:
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, ""):
                if line:
                    self.emit(line)
                    timepoint = self._timepoint_from_text(line)
                    detected_position = position or self._position_from_text(line)
                    if timepoint and timepoint != self.current_timepoint:
                        self.current_timepoint = timepoint
                        self.event(
                            current_stage=stage,
                            current_position=detected_position,
                            current_timepoint=timepoint,
                            completed_file_count=self.completed_files,
                            failed_file_count=self.failed_files,
                        )
                    elif detected_position and not position:
                        self.event(
                            current_stage=stage,
                            current_position=detected_position,
                            current_timepoint=self.current_timepoint,
                            completed_file_count=self.completed_files,
                            failed_file_count=self.failed_files,
                        )
                if self.stop_event.is_set():
                    self._terminate_current_process()
                    break
            returncode = proc.wait()
        finally:
            self.current_process = None
        record.update({"end_time": utc_now(), "returncode": returncode, "status": "success" if returncode == 0 else "failed"})
        return returncode

    def _stage_sequence(self) -> list[str]:
        return resolved_stage_sequence(self.config)

    def _expected_names(self, jobs: dict[str, Any]) -> dict[str, list[str]]:
        return {position: [r["output_name"] for r in pdata["records"]] for position, pdata in jobs["positions"].items()}

    def _stage_root(self, stage: str) -> Path:
        single_crop = self.config.get("run_mode") == RUN_SINGLE and self.config.get("single_stage") == "crop"
        crop_folder = "04_cropped" if single_crop else ("05_cropped" if self.config.get("enable_denoising", False) else "04_cropped")
        return {
            "fusion": self.output_root / "01_fused",
            "registration": self.output_root / "02_registered",
            "background_subtraction": self.output_root / "03_background_subtracted",
            "denoise": self.output_root / "04_denoised",
            "crop": self.output_root / crop_folder,
        }[stage]

    def _stage_manifest_path(self, stage: str) -> Path:
        return self.stage_manifest_root / f"{stage}.json"

    def _stage_signature(self, stage: str, inputs: list[Path], expected: dict[str, list[str]], command_config: Any) -> str:
        sig = {
            "stage": stage,
            "inputs": [_file_signature(p) for p in sorted(inputs, key=lambda p: str(p)) if p.exists()],
            "expected": expected,
            "command_config": command_config,
        }
        return _json_hash(sig)

    def _validate_outputs(self, stage: str, expected: dict[str, list[str]], signature: str | None = None,
                          require_manifest: bool = True) -> tuple[bool, list[str]]:
        errors: list[str] = []
        root = self._stage_root(stage)
        total = 0
        for position, names in expected.items():
            pos_dir = root / position
            for name in names:
                total += 1
                path = pos_dir / name
                if not path.is_file():
                    errors.append(f"Missing output: {path}")
                elif path.stat().st_size <= 0:
                    errors.append(f"Zero-byte output: {path}")
        if total == 0:
            errors.append("No expected TIFF outputs were defined.")
        manifest = _read_json(self._stage_manifest_path(stage)) if require_manifest else None
        if require_manifest:
            if not manifest:
                errors.append(f"Missing or unreadable stage manifest: {self._stage_manifest_path(stage)}")
            else:
                if manifest.get("status") != "success":
                    errors.append(f"Stage manifest status is {manifest.get('status')!r}, not success.")
                if signature is not None and manifest.get("signature") != signature:
                    errors.append("Stage manifest signature does not match current inputs/configuration.")
                if manifest.get("expected_outputs") != expected:
                    errors.append("Stage manifest expected-output list does not match the current selection.")
        return not errors, errors

    def _write_stage_manifest(self, stage: str, status: str, signature: str, expected: dict[str, list[str]],
                              started: str, elapsed: float, failures: list[dict[str, Any]]) -> None:
        payload = {
            "stage": stage,
            "status": status,
            "signature": signature,
            "expected_outputs": expected,
            "output_root": str(self._stage_root(stage)),
            "start_time": started,
            "end_time": utc_now(),
            "duration_sec": elapsed,
            "failures": failures,
        }
        self._stage_manifest_path(stage).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.stage_records[stage] = payload

    def _copy_child_manifest(self, stage: str, position: str, source_name: str) -> None:
        source = self._stage_root(stage) / source_name
        if source.exists():
            destination = self.logs_root / "child_manifests" / stage / f"{position}_{source_name}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())

    def _command_for_position(self, stage: str, position: str, input_root: Path, output_root: Path, ome: str) -> list[str]:
        cfg = self.config
        overwrite = ["--overwrite"] if cfg["overwrite"] else []
        if stage == "registration":
            cmd = [sys.executable, str(SCRIPT_DIR / "registration.py"),
                   "--input-root", str(input_root), "--output-root", str(output_root),
                   "--glob-pattern", str(cfg["registration_glob_pattern"]), "--positions", position,
                   "--downsample-xy", str(cfg["registration_downsample_xy"]),
                   "--downsample-z", str(cfg["registration_downsample_z"]),
                   "--estimate-method", str(cfg["registration_estimate_method"]),
                   "--apply-mode", str(cfg["registration_apply_mode"]),
                   "--compression", str(cfg["registration_compression"]),
                   "--save-dtype", str(cfg["registration_save_dtype"]),
                   "--padding-z", str(cfg["registration_padding_z"]),
                   "--padding-y", str(cfg["registration_padding_y"]),
                   "--padding-x", str(cfg["registration_padding_x"]),
                   "--phase-upsample", str(cfg["registration_phase_upsample"]),
                   "--series-index", str(cfg["registration_series_index"]),
                   "--clip-low", str(cfg["registration_clip_low"]),
                   "--clip-high", str(cfg["registration_clip_high"]),
                   "--center-threshold-percentile", str(cfg["registration_center_threshold_percentile"])]
            if ome:
                cmd += ["--ome-xml", ome]
            for key, flag in [("registration_channel_index", "--channel-index"), ("registration_max_shift_z", "--max-shift-z"), ("registration_max_shift_yx", "--max-shift-yx")]:
                if str(cfg[key]).strip() != "":
                    cmd += [flag, str(cfg[key])]
            return cmd + overwrite
        if stage == "background_subtraction":
            cmd = [sys.executable, str(SCRIPT_DIR / "bgs.py"),
                   "--input-root", str(input_root), "--output-root", str(output_root),
                   "--glob-pattern", str(cfg["bgs_glob_pattern"]), "--positions", position,
                   "--method", str(cfg["bgs_method"]), "--z-projection", str(cfg["bgs_z_projection"]),
                   "--z-bg-percentile", str(cfg["bgs_z_bg_percentile"]), "--bg-sigma", str(cfg["bgs_bg_sigma"]),
                   "--bg-downsample", str(cfg["bgs_bg_downsample"]), "--bg-alpha", str(cfg["bgs_bg_alpha"]),
                   "--offset-percentile", str(cfg["bgs_offset_percentile"]),
                   "--offset-sample-slices", str(cfg["bgs_offset_sample_slices"]),
                   "--bg-offset", str(cfg["bgs_bg_offset"]),
                   "--bbox-downsample-xy", str(cfg["bgs_bbox_downsample_xy"]),
                   "--bbox-downsample-z", str(cfg["bgs_bbox_downsample_z"]),
                   "--bbox-dilate", str(cfg["bgs_bbox_dilate"]),
                   "--mask-percentile", str(cfg["bgs_mask_percentile"]),
                   "--output-dtype", str(cfg["bgs_output_dtype"]),
                   "--compression", str(cfg["bgs_compression"])]
            if ome:
                cmd += ["--ome-xml", ome]
            if cfg["bgs_auto_offset"]:
                cmd.append("--auto-offset")
            if cfg["coarse_crop_during_bgs"]:
                cmd += ["--crop-to-foreground", "--crop-padding-xy", "80", "--crop-padding-z", "8"]
            return cmd + overwrite
        if stage == "denoise":
            cmd = [sys.executable, str(SCRIPT_DIR / "denoise.py"),
                   "--input-root", str(input_root), "--output-root", str(output_root),
                   "--glob-pattern", str(cfg["denoise_glob_pattern"]),
                   "--method", str(cfg["denoise_method"]),
                   "--median-size", str(cfg["denoise_median_size"]),
                   "--sigma-xy", str(cfg["denoise_sigma_xy"]),
                   "--sigma-z", str(cfg["denoise_sigma_z"]),
                   "--intensity-floor", str(cfg["denoise_intensity_floor"]),
                   "--output-dtype", str(cfg["denoise_output_dtype"]),
                   "--compression", str(cfg["denoise_compression"]),
                   "--crop-padding-xy", str(cfg["denoise_crop_padding_xy"]),
                   "--crop-padding-z", str(cfg["denoise_crop_padding_z"]),
                   "--mask-percentile", str(cfg["denoise_mask_percentile"]),
                   "--min-size-voxels", str(cfg["denoise_min_size_voxels"])]
            if cfg["denoise_crop_to_foreground"]:
                cmd.append("--crop-to-foreground")
            return cmd + overwrite
        if stage == "crop":
            cmd = [sys.executable, str(SCRIPT_DIR / "crop.py"),
                   "--input-root", str(input_root), "--output-root", str(output_root),
                   "--glob-pattern", str(cfg["crop_glob_pattern"]), "--positions", position,
                   "--analysis-downsample-xy", str(cfg["crop_analysis_downsample_xy"]),
                   "--analysis-downsample-z", str(cfg["crop_analysis_downsample_z"]),
                   "--xy-projection", str(cfg["crop_xy_projection"]),
                   "--xy-projection-percentile", str(cfg["crop_xy_projection_percentile"]),
                   "--xy-smooth-sigma", str(cfg["crop_xy_smooth_sigma"]),
                   "--xy-threshold-method", str(cfg["crop_xy_threshold_method"]),
                   "--xy-threshold-percentile", str(cfg["crop_xy_threshold_percentile"]),
                   "--xy-threshold-floor-percentile", str(cfg["crop_xy_threshold_floor_percentile"]),
                   "--min-area-pixels", str(cfg["crop_min_area_pixels"]),
                   "--z-profile-percentile", str(cfg["crop_z_profile_percentile"]),
                   "--z-profile-fraction", str(cfg["crop_z_profile_fraction"]),
                   "--z-smooth-sigma", str(cfg["crop_z_smooth_sigma"]),
                   "--crop-padding-xy", str(cfg["crop_padding_xy"]),
                   "--crop-padding-z", str(cfg["crop_padding_z"]),
                   "--compression", str(cfg["crop_compression"])]
            if cfg["crop_keep_border_components"]:
                cmd.append("--keep-border-components")
            if cfg["crop_no_crop_z"]:
                cmd.append("--no-crop-z")
            if cfg["crop_square_xy"]:
                cmd.append("--square-xy")
            if cfg["crop_no_qc_png"]:
                cmd.append("--no-qc-png")
            if cfg["continue_on_error"]:
                cmd.append("--continue-on-error")
            return cmd + overwrite
        raise ValueError(stage)

    def _stage_inputs(self, stage: str, jobs: dict[str, Any]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
        positions = jobs["positions"]
        transitions: list[dict[str, Any]] = []
        roots: dict[str, Path] = {}
        sequence = self._stage_sequence()
        if stage not in sequence or stage == "fusion":
            raise ValueError(stage)

        standalone = self.config.get("run_mode") == RUN_SINGLE
        if standalone:
            for position, pdata in positions.items():
                roots[position] = Path(pdata["source_folder"])
                for record in pdata["records"]:
                    source = record.get("input_path") or record.get("fused_path")
                    transitions.append({
                        "position": position,
                        "input": str(source),
                        "output": str(self._stage_root(stage) / position / record["output_name"]),
                    })
            return roots, transitions

        stage_index = sequence.index(stage)
        imported_fused_registration = (
            stage_index == 0
            and stage == "registration"
            and self.config.get("starting_data_type") == INPUT_FUSED
        )
        previous = None if imported_fused_registration else sequence[stage_index - 1]
        for position, pdata in positions.items():
            if imported_fused_registration:
                roots[position] = Path(pdata["source_folder"])
                for record in pdata["records"]:
                    transitions.append({
                        "position": position,
                        "input": record["fused_path"],
                        "output": str(self._stage_root(stage) / position / record["output_name"]),
                    })
            else:
                assert previous is not None
                roots[position] = self._stage_root(previous) / position
                for record in pdata["records"]:
                    transitions.append({
                        "position": position,
                        "input": str(roots[position] / record["output_name"]),
                        "output": str(self._stage_root(stage) / position / record["output_name"]),
                    })
        return roots, transitions

    def _run_fusion(self, jobs: dict[str, Any], expected: dict[str, list[str]]) -> list[dict[str, Any]]:
        jobs_path = self.logs_root / "fusion_jobs.json"
        jobs_path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
        cmd = [sys.executable, str(SCRIPT_DIR / "fuse_views.py"),
               "--pipeline-jobs", str(jobs_path), "--output-root", str(self._stage_root("fusion")),
               "--reference-channel", str(self.config["reference_channel"]),
               "--crop", str(self.config["fusion_crop_um"]),
               "--fused-token", str(self.config["fused_token"]),
               "--max-projection-root", str(self.logs_root / "fusion_max")]
        if self.config["fusion_save_as_crop"]:
            cmd.append("--save-as-crop")
        if self.config["overwrite"]:
            cmd.append("--overwrite")
        rc = self._run_command(cmd, "fusion")
        if rc != 0:
            raise PipelineFailure(f"Fusion failed with exit status {rc}.")
        transitions = []
        for position, pdata in jobs["positions"].items():
            for record in pdata["records"]:
                transitions.append({"position": position, "input": record["view1"], "output": str(self._stage_root("fusion") / position / record["output_name"])})
        return transitions

    def _run_position_stage(self, stage: str, jobs: dict[str, Any]) -> list[dict[str, Any]]:
        roots, transitions = self._stage_inputs(stage, jobs)
        stage_failures = []
        for index, (position, pdata) in enumerate(jobs["positions"].items(), 1):
            self.current_timepoint = ""
            self.event(current_stage=stage, current_position=position, current_timepoint="", completed_file_count=self.completed_files, failed_file_count=self.failed_files)
            self.emit(f"[{stage}] position {position} ({index}/{len(jobs['positions'])})")
            fused_ome = self._stage_root("fusion") / position / "ome-tiff.companion.ome"
            ome = str(fused_ome) if fused_ome.is_file() else pdata.get("ome_path", "")
            cmd = self._command_for_position(stage, position, roots[position], self._stage_root(stage), ome)
            rc = self._run_command(cmd, stage, position)
            if rc != 0:
                failure = {"stage": stage, "position": position, "returncode": rc, "message": "Child process failed."}
                self.failures.append(failure)
                stage_failures.append(failure)
                self.failed_files += len(pdata["records"])
                if not self.config["continue_on_error"]:
                    raise PipelineFailure(f"{stage} failed for position {position} with exit status {rc}.")
            else:
                self.completed_files += len(pdata["records"])
                child_manifest = {
                    "registration": "registration_manifest.json",
                    "background_subtraction": "background_subtraction_manifest.json",
                    "denoise": "denoising_manifest.json",
                    "crop": "crop_manifest.json",
                }[stage]
                self._copy_child_manifest(stage, position, child_manifest)
            self.current_timepoint = ""
            self.event(current_stage=stage, current_position=position, current_timepoint="", completed_file_count=self.completed_files, failed_file_count=self.failed_files)
        if stage_failures:
            raise PipelineFailure(f"{stage} had failures; dependent stages are blocked until every selected position succeeds.")
        return transitions

    def run(self) -> int:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.stage_manifest_root.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_root / "pipeline.log"
        self._log_file = log_path.open("a", encoding="utf-8")
        started = utc_now()
        t0 = time.time()
        scan = self.prevalidated_scan or scan_dataset(self.config)
        self.emit("Dataset discovery table:\n" + discovery_table_text(scan))
        errors = [] if self.skip_initial_validation else validate_configuration(self.config, scan)
        if errors:
            for error in errors:
                self.emit("VALIDATION ERROR: " + error)
                self.failures.append({"stage": "validation", "message": error})
            self._write_errors_csv()
            self._write_pipeline_manifest(scan, started, time.time() - t0, "validation_failed", errors, [])
            if self._log_file:
                self._log_file.close(); self._log_file = None
            return 2
        jobs = make_pipeline_jobs(self.config, scan)
        expected = self._expected_names(jobs)
        sequence = self._stage_sequence()
        if not sequence:
            message = "No processing stage was selected."
            self.emit("VALIDATION ERROR: " + message)
            self.failures.append({"stage": "validation", "message": message})
            self._write_errors_csv()
            self._write_pipeline_manifest(scan, started, time.time() - t0, "validation_failed", [message], [])
            if self._log_file:
                self._log_file.close(); self._log_file = None
            return 2

        transitions_all: list[dict[str, Any]] = []
        status = "success"
        try:
            for stage in sequence:
                if self.stop_event.is_set():
                    raise PipelineFailure("Pipeline stopped by user.")
                stage_started = utc_now()
                stage_t0 = time.time()
                self.current_timepoint = ""
                self.event(
                    current_stage=stage,
                    current_position="",
                    current_timepoint="",
                    completed_file_count=self.completed_files,
                    failed_file_count=self.failed_files,
                )
                if stage == "fusion":
                    input_files = [Path(r["view1"]) for p in jobs["positions"].values() for r in p["records"]] + [Path(r["view2"]) for p in jobs["positions"].values() for r in p["records"]]
                    command_cfg = {k: self.config[k] for k in ["reference_channel", "fusion_crop_um", "fusion_save_as_crop", "fused_token", "overwrite"]}
                else:
                    _roots, transition_preview = self._stage_inputs(stage, jobs)
                    input_files = [Path(t["input"]) for t in transition_preview]
                    prefix = {
                        "registration": "registration_",
                        "background_subtraction": "bgs_",
                        "denoise": "denoise_",
                        "crop": "crop_",
                    }[stage]
                    command_cfg = {
                        k: v for k, v in self.config.items()
                        if k.startswith(prefix) or k in {
                            "overwrite", "coarse_crop_during_bgs", "continue_on_error", "enable_denoising",
                            "run_mode", "single_stage",
                        }
                    }
                signature = self._stage_signature(stage, input_files, expected, command_cfg)
                if self.config["resume"]:
                    valid, resume_errors = self._validate_outputs(stage, expected, signature=signature, require_manifest=True)
                    if valid:
                        self.emit(f"[{stage}] resume: valid stage manifest and all expected nonzero TIFFs found; skipping stage.")
                        self.stage_records[stage] = _read_json(self._stage_manifest_path(stage))
                        continue
                    self.emit(f"[{stage}] resume validation failed; rerunning: {'; '.join(resume_errors)}")
                try:
                    if stage == "fusion":
                        transitions = self._run_fusion(jobs, expected)
                        self.completed_files += sum(len(v) for v in expected.values())
                    else:
                        transitions = self._run_position_stage(stage, jobs)
                except Exception as stage_exc:
                    stage_failure = {"stage": stage, "message": str(stage_exc)}
                    self._write_stage_manifest(stage, "failed", signature, expected, stage_started, time.time() - stage_t0, [stage_failure])
                    raise
                transitions_all.append({"stage": stage, "files": transitions})
                if self.config["dry_run"]:
                    self._write_stage_manifest(stage, "dry_run", signature, expected, stage_started, time.time() - stage_t0, [])
                    continue
                valid, output_errors = self._validate_outputs(stage, expected, require_manifest=False)
                stage_status = "success" if valid else "failed"
                self._write_stage_manifest(stage, stage_status, signature, expected, stage_started, time.time() - stage_t0, [] if valid else [{"message": e} for e in output_errors])
                if not valid:
                    raise PipelineFailure(f"Output validation failed after {stage}: {'; '.join(output_errors)}")
            if not self.config["dry_run"]:
                report_transitions: list[dict[str, Any]] = []
                for report_stage in sequence:
                    if report_stage == "fusion":
                        files = [{"position": position, "input": record["view1"], "output": str(self._stage_root("fusion") / position / record["output_name"])} for position, pdata in jobs["positions"].items() for record in pdata["records"]]
                    else:
                        _roots, files = self._stage_inputs(report_stage, jobs)
                    report_transitions.append({"stage": report_stage, "files": files})
                build_size_summary(self.output_root, report_transitions)
                self.emit("Wrote combined stage-size report: " + str(self.output_root / "pipeline_size_summary.csv"))
                self.emit("File-size interpretation: compressed black regions are not zero bytes; TIFF pages/strips, directories, offsets, and metadata remain. Cropping removes actual voxels and often compressed blocks.")
        except PipelineFailure as exc:
            status = "stopped" if self.stop_event.is_set() else "failed"
            self.failures.append({"stage": self.stage_records.keys() and list(self.stage_records)[-1] or "pipeline", "message": str(exc)})
            self.emit(f"PIPELINE {status.upper()}: {exc}")
        except Exception as exc:
            status = "failed"
            self.failures.append({"stage": "pipeline", "message": str(exc), "traceback": traceback.format_exc()})
            self.emit("UNEXPECTED PIPELINE ERROR: " + traceback.format_exc())
        finally:
            self._write_errors_csv()
            self._write_pipeline_manifest(scan, started, time.time() - t0, status, [], transitions_all)
            self.event(current_stage="stopped" if status != "success" else "completed", current_position="", current_timepoint="", completed_file_count=self.completed_files, failed_file_count=self.failed_files)
            if self._log_file:
                self._log_file.close()
                self._log_file = None
        return 0 if status == "success" else (130 if status == "stopped" else 1)

    def _write_errors_csv(self) -> None:
        path = self.output_root / "pipeline_errors.csv"
        fields = ["stage", "position", "returncode", "message", "traceback"]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for failure in self.failures:
                writer.writerow({k: failure.get(k, "") for k in fields})

    def _write_pipeline_manifest(self, scan: ScanResult, started: str, elapsed: float, status: str,
                                 validation_errors: list[str], transitions: list[dict[str, Any]]) -> None:
        final_stage = self._stage_sequence()[-1]
        payload = {
            "status": status,
            "resolved_input_path": str(Path(self.config["input_dataset_folder"]).expanduser().resolve()),
            "resolved_output_path": str(self.output_root),
            "selected_positions": scan.positions,
            "selected_channels": scan.channels,
            "naming_rules": {k: self.config[k] for k in [
                "starting_data_type", "position_folder_pattern", "settings_pattern", "view1_token", "view2_token", "fused_token",
                "tiff_glob_pattern", "ome_companion_glob_pattern", "timepoint_regex", "channel_regex", "fused_channel_regex", "processed_channel_regex"]},
            "run_mode": self.config["run_mode"],
            "single_stage": self.config["single_stage"],
            "processing_order": self.config["processing_order"],
            "stage_sequence": self._stage_sequence(),
            "legacy_start_stage": self.config.get("start_stage"),
            "legacy_stop_after_stage": self.config.get("stop_after_stage"),
            "final_output_root": str(self._stage_root(self._stage_sequence()[-1])),
            "all_stage_parameters": self.config,
            "commands": self.commands,
            "start_time": started,
            "end_time": utc_now(),
            "duration_sec": elapsed,
            "stage_statuses": self.stage_records,
            "failures": self.failures,
            "validation_errors": validation_errors,
            "software_versions": software_versions(),
            "discovery": scan.to_dict(),
            "stage_transitions": transitions,
            "size_summary_csv": str(self.output_root / "pipeline_size_summary.csv"),
            "errors_csv": str(self.output_root / "pipeline_errors.csv"),
        }
        (self.output_root / "pipeline_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_config(path: str | Path) -> dict[str, Any]:
    return merged_config(json.loads(Path(path).read_text(encoding="utf-8")))


def save_default_config(path: str | Path) -> None:
    Path(path).write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="JSON settings file. Recommended for reproducible non-GUI runs.")
    p.add_argument("--write-default-config", metavar="PATH", help="Write a complete editable JSON configuration and exit.")
    p.add_argument("--scan", action="store_true", help="Scan and validate only; do not run image-processing stages.")
    p.add_argument("--run-mode", choices=[RUN_FULL, RUN_SINGLE])
    p.add_argument("--single-stage", choices=STAGES)
    p.add_argument("--starting-data-type", choices=[INPUT_RAW, INPUT_FUSED])
    p.add_argument("--input-dataset-folder")
    p.add_argument("--output-root")
    p.add_argument("--selected-positions")
    p.add_argument("--selected-channels")
    p.add_argument("--reference-channel")
    p.add_argument("--processing-order", choices=[ORDER_BGS_CROP, ORDER_CROP_BGS])
    p.add_argument("--coarse-crop-during-bgs", action="store_true", default=None)
    p.add_argument("--dry-run", action="store_true", default=None)
    p.add_argument("--resume", action="store_true", default=None)
    p.add_argument("--overwrite", action="store_true", default=None)
    p.add_argument("--no-overwrite", action="store_true")
    p.add_argument("--start-stage", choices=STAGES)
    p.add_argument("--stop-after-stage", choices=[*STAGES, "final"])
    p.add_argument("--continue-on-error", action="store_true", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_default_config:
        save_default_config(args.write_default_config)
        print(f"Wrote default configuration: {args.write_default_config}")
        return 0
    config = load_config(args.config) if args.config else merged_config()
    mapping = {
        "run_mode": args.run_mode,
        "single_stage": args.single_stage,
        "starting_data_type": args.starting_data_type,
        "input_dataset_folder": args.input_dataset_folder,
        "output_root": args.output_root,
        "selected_positions": args.selected_positions,
        "selected_channels": args.selected_channels,
        "reference_channel": args.reference_channel,
        "processing_order": args.processing_order,
        "coarse_crop_during_bgs": args.coarse_crop_during_bgs,
        "dry_run": args.dry_run,
        "resume": args.resume,
        "overwrite": args.overwrite,
        "start_stage": args.start_stage,
        "stop_after_stage": args.stop_after_stage,
        "continue_on_error": args.continue_on_error,
    }
    for key, value in mapping.items():
        if value is not None:
            config[key] = value
    if args.start_stage is not None and args.stop_after_stage == args.start_stage:
        config["run_mode"] = RUN_SINGLE
        config["single_stage"] = args.start_stage
    elif args.start_stage is not None or args.stop_after_stage is not None:
        raise SystemExit("--start-stage/--stop-after-stage now support only the same stage for a one-stage run. Use --run-mode and --single-stage.")
    if args.no_overwrite:
        config["overwrite"] = False
    if args.scan:
        scan = scan_dataset(config)
        print(discovery_table_text(scan))
        errors = validate_configuration(config, scan)
        if errors:
            print("\nValidation errors:")
            for error in errors:
                print("- " + error)
            return 2
        print("\nConfiguration is valid.")
        return 0
    runner = PipelineRunner(config)

    def handle_signal(signum: int, _frame: Any) -> None:
        runner.emit(f"Received signal {signum}.")
        runner.request_stop()

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        handled_signals.append(signal.SIGBREAK)
    for sig in handled_signals:
        signal.signal(sig, handle_signal)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
