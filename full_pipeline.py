#!/usr/bin/env python3
"""GUI-facing Script adapter for the configurable light-sheet processing pipeline.

Preferred launch:
    python3 gui/script_gui.py full_pipeline

The generic Script metadata remains available for package compatibility, while
script_gui.py routes this workflow to gui/pipeline_gui.py because the pipeline
needs a scan table, progress fields, dynamic warnings, and cooperative process
cancellation that the static generic form cannot safely provide.
"""
from __future__ import annotations

import os
import sys
from typing import Any

from pipeline_runner import DEFAULT_CONFIG, INPUT_RAW, INPUT_FUSED, RUN_FULL, RUN_SINGLE, ORDER_BGS_CROP, ORDER_CROP_BGS, PipelineRunner


class Script:
    def __init__(self) -> None:
        self.parameters = [
            "run_mode", "single_stage", "starting_data_type", "input_dataset_folder", "output_root", "position_folder_pattern", "settings_pattern",
            "view1_token", "view2_token", "fused_token", "tiff_glob_pattern",
            "ome_companion_glob_pattern", "timepoint_regex", "channel_regex", "fused_channel_regex", "processed_channel_regex",
            "selected_positions", "selected_channels", "first_timepoint", "last_timepoint", "reference_channel",
            "processing_order", "coarse_crop_during_bgs", "enable_denoising", "dry_run", "resume", "overwrite",
            "continue_on_error",
            "registration_downsample_xy", "registration_downsample_z", "registration_estimate_method",
            "registration_apply_mode", "registration_compression",
            "bgs_method", "bgs_bg_sigma", "bgs_bg_alpha", "bgs_auto_offset", "bgs_compression", "bgs_output_dtype",
            "denoise_method", "denoise_median_size", "denoise_sigma_xy", "denoise_sigma_z",
            "denoise_intensity_floor", "denoise_output_dtype", "denoise_compression",
            "denoise_crop_to_foreground", "denoise_crop_padding_xy", "denoise_crop_padding_z",
            "denoise_mask_percentile", "denoise_min_size_voxels",
            "crop_analysis_downsample_xy", "crop_analysis_downsample_z", "crop_xy_projection",
            "crop_xy_projection_percentile", "crop_xy_threshold_method", "crop_xy_threshold_floor_percentile",
            "crop_padding_xy", "crop_padding_z", "crop_no_crop_z", "crop_compression",
        ]
        self.parameter_label = {
            "run_mode": "What do you want to run?",
            "single_stage": "Single processing stage",
            "starting_data_type": "Starting data type",
            "input_dataset_folder": "Input folder", "output_root": "Output root",
            "position_folder_pattern": "Position-folder pattern", "settings_pattern": "Settings name or pattern",
            "view1_token": "View 1 token", "view2_token": "View 2 token", "fused_token": "Fused token",
            "tiff_glob_pattern": "TIFF glob pattern", "ome_companion_glob_pattern": "OME companion glob pattern",
            "timepoint_regex": "Timepoint regex or template", "channel_regex": "Raw-view channel regex or template",
            "fused_channel_regex": "Fused-TIFF channel regex or template",
            "processed_channel_regex": "Stage-ready TIFF channel regex",
            "selected_positions": "Selected positions", "selected_channels": "Selected channels",
            "first_timepoint": "First timepoint", "last_timepoint": "Last timepoint",
            "reference_channel": "Reference channel", "processing_order": "Processing order",
            "coarse_crop_during_bgs": "Coarse crop during background subtraction",
            "enable_denoising": "Enable denoising",
            "dry_run": "Dry run", "resume": "Resume", "overwrite": "Overwrite",
            "continue_on_error": "Continue on error",
            "registration_downsample_xy": "Registration downsample XY",
            "registration_downsample_z": "Registration downsample Z",
            "registration_estimate_method": "Registration estimate method",
            "registration_apply_mode": "Registration apply mode",
            "registration_compression": "Registration compression",
            "bgs_method": "BGS method", "bgs_bg_sigma": "BGS sigma", "bgs_bg_alpha": "BGS alpha",
            "bgs_auto_offset": "BGS auto offset", "bgs_compression": "BGS compression",
            "bgs_output_dtype": "BGS output dtype",
            "denoise_method": "Denoising method",
            "denoise_median_size": "Denoising median size",
            "denoise_sigma_xy": "Denoising sigma XY",
            "denoise_sigma_z": "Denoising sigma Z",
            "denoise_intensity_floor": "Denoising intensity floor",
            "denoise_output_dtype": "Denoising output dtype",
            "denoise_compression": "Denoising compression",
            "denoise_crop_to_foreground": "Extra foreground crop during denoising",
            "denoise_crop_padding_xy": "Denoising crop padding XY",
            "denoise_crop_padding_z": "Denoising crop padding Z",
            "denoise_mask_percentile": "Denoising crop mask percentile",
            "denoise_min_size_voxels": "Denoising crop minimum voxels",
            "crop_analysis_downsample_xy": "Crop analysis downsample XY",
            "crop_analysis_downsample_z": "Crop analysis downsample Z",
            "crop_xy_projection": "Crop XY projection",
            "crop_xy_projection_percentile": "Crop projection percentile",
            "crop_xy_threshold_method": "Crop threshold method",
            "crop_xy_threshold_floor_percentile": "Crop threshold floor percentile",
            "crop_padding_xy": "Crop padding XY", "crop_padding_z": "Crop padding Z",
            "crop_no_crop_z": "Keep all Z", "crop_compression": "Crop compression",
        }
        self.parameter_default_value = {key: DEFAULT_CONFIG[key] for key in self.parameters}
        self.parameter_type = {key: str for key in self.parameters}
        for key in [
            "first_timepoint", "last_timepoint", "registration_downsample_xy", "registration_downsample_z",
            "denoise_median_size", "denoise_crop_padding_xy", "denoise_crop_padding_z", "denoise_min_size_voxels",
            "crop_analysis_downsample_xy", "crop_analysis_downsample_z", "crop_padding_xy", "crop_padding_z",
        ]:
            self.parameter_type[key] = int
        for key in [
            "bgs_bg_sigma", "bgs_bg_alpha", "denoise_sigma_xy", "denoise_sigma_z",
            "denoise_intensity_floor", "denoise_mask_percentile",
            "crop_xy_projection_percentile", "crop_xy_threshold_floor_percentile",
        ]:
            self.parameter_type[key] = float
        for key in [
            "coarse_crop_during_bgs", "enable_denoising", "dry_run", "resume", "overwrite",
            "continue_on_error", "bgs_auto_offset", "denoise_crop_to_foreground", "crop_no_crop_z",
        ]:
            self.parameter_type[key] = bool
        self.parameter_control = {
            "run_mode": ("combobox", [RUN_FULL, RUN_SINGLE], "readonly"),
            "single_stage": ("combobox", ["fusion", "registration", "background_subtraction", "denoise", "crop"], "readonly"),
            "starting_data_type": ("combobox", [INPUT_RAW, INPUT_FUSED], "readonly"),
            "input_dataset_folder": "folder", "output_root": "folder",
            "processing_order": ("combobox", [ORDER_BGS_CROP, ORDER_CROP_BGS], "readonly"),
            "registration_estimate_method": ("combobox", ["center", "phase_mip", "center_phase_mip"], "readonly"),
            "registration_apply_mode": ("combobox", ["integer", "subpixel"], "readonly"),
            "crop_xy_projection": ("combobox", ["percentile", "max", "mean"], "readonly"),
            "crop_xy_threshold_method": ("combobox", ["otsu", "percentile"], "readonly"),
            "bgs_output_dtype": ("combobox", ["same", "uint16", "uint8", "float32"], "readonly"),
            "denoise_method": ("combobox", ["none", "median2d", "gaussian2d", "median_gaussian2d", "gaussian3d"], "readonly"),
            "denoise_output_dtype": ("combobox", ["same", "uint16", "uint8", "float32"], "readonly"),
        }
        for key in [
            "coarse_crop_during_bgs", "enable_denoising", "dry_run", "resume", "overwrite",
            "continue_on_error", "bgs_auto_offset", "denoise_crop_to_foreground", "crop_no_crop_z",
        ]:
            self.parameter_control[key] = "checkbox"
        self.parameter_help = {
            "run_mode": "Complete workflow runs the full configured sequence. Single stage runs exactly one selected operation with no prerequisite manifest requirement.",
            "single_stage": "Choose fusion, registration, BGS, denoising, or crop. For stages after fusion, point Input folder directly at stage-ready TIFFs.",
            "starting_data_type": "Choose raw View1/View2 pairs for the complete workflow, or Already fused TIFFs to begin directly at registration without a previous fusion manifest.",
            "input_dataset_folder": "For already-fused input, this may directly contain t0001_561_Fused.tif-style files plus one OME companion, or contain one such folder per position.",
            "fused_channel_regex": "Used only for already-fused input. It must extract the channel using a named 'channel' group.",
            "processed_channel_regex": "Used for a standalone registration, BGS, denoising, or crop run. It must extract the channel.",
            "processing_order": "Default is Background subtraction → Crop. Crop → Background subtraction is faster on smaller arrays but leaves the estimator less surrounding background and makes crop detection more sensitive to raw haze.",
            "coarse_crop_during_bgs": "Optional large-data optimization. Disabled by default. When enabled, BGS receives --crop-to-foreground --crop-padding-xy 80 --crop-padding-z 8; final crop still performs a tighter object-aware crop.",
            "position_folder_pattern": "Python regular expression with named groups 'position' and 'settings'. The default splits a folder at its final underscore.",
            "timepoint_regex": "Python regular expression with a named 'timepoint' group (or one capture group).",
            "channel_regex": "Python regular expression with a named 'channel' group. {view1}, {view2}, and {fused} placeholders are expanded safely.",
            "bgs_output_dtype": "Default 'same' preserves the input dtype. Select another dtype only intentionally.",
            "enable_denoising": "Optional. When enabled, denoising runs immediately after background subtraction. It is disabled by default to preserve the original workflow.",
            "denoise_method": "Conservative default is median_gaussian2d: a 3x3 median despeckle followed by mild XY Gaussian smoothing.",
            "denoise_intensity_floor": "Values below this threshold are set to zero after smoothing. Increase cautiously because weak signal can be removed.",
            "denoise_output_dtype": "Default 'same' preserves quantitative dtype. Avoid uint8 for analysis.",
            "denoise_crop_to_foreground": "Usually leave disabled because the pipeline already has a final object-aware crop.",
        }
        self.window_title = "Light-sheet full processing pipeline"
        self._runner: PipelineRunner | None = None

    def run(self, parameter_values: dict[str, Any], update_figures_callback: Any) -> None:
        config = dict(DEFAULT_CONFIG)
        config.update(parameter_values)
        self._runner = PipelineRunner(config)
        status = self._runner.run()
        if status:
            raise RuntimeError(f"Pipeline exited with status {status}")

    def stop(self) -> None:
        if self._runner is not None:
            self._runner.request_stop()

    def get_figure_count(self, parameter_values: dict[str, Any]) -> int:
        return 0


if __name__ == "__main__":
    python_interpreter = sys.executable
    script_gui = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "gui", "script_gui.py"))
    os.spawnl(os.P_WAIT, python_interpreter, python_interpreter, script_gui, "full_pipeline")
