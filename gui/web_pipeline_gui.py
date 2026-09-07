#!/usr/bin/env python3
"""Local browser interface for the configurable light-sheet pipeline.

This interface intentionally uses only Python's standard library for the web
server and browser integration. It is intended for systems where the packaged
Tk/Tcl runtime cannot start. Image processing remains in the existing scripts
and is orchestrated by scripts/pipeline_runner.py.

Launch from the project root:
    python3 gui/script_gui.py full_pipeline

Direct launch:
    python3 gui/web_pipeline_gui.py
"""
from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
import subprocess
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = PROJECT_ROOT / "scripts"
import sys

sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_runner import (  # noqa: E402
    DEFAULT_CONFIG,
    INPUT_RAW,
    INPUT_FUSED,
    RUN_FULL,
    RUN_SINGLE,
    ORDER_BGS_CROP,
    ORDER_CROP_BGS,
    PipelineRunner,
    merged_config,
    scan_dataset,
    validate_configuration,
)

HOST = "127.0.0.1"
DEFAULT_PORT = 8765

LABEL_OVERRIDES = {
    "run_mode": "What do you want to run?",
    "single_stage": "Single processing stage",
    "starting_data_type": "Starting data type",
    "input_dataset_folder": "Input folder",
    "output_root": "Output root",
    "position_folder_pattern": "Position-folder pattern",
    "settings_pattern": "Settings name or pattern",
    "view1_token": "View 1 token",
    "view2_token": "View 2 token",
    "fused_token": "Fused token",
    "tiff_glob_pattern": "TIFF glob pattern",
    "ome_companion_glob_pattern": "OME companion glob pattern",
    "timepoint_regex": "Timepoint regex or template",
    "channel_regex": "Raw-view channel regex or template",
    "fused_channel_regex": "Fused-TIFF channel regex or template",
    "processed_channel_regex": "Stage-ready TIFF channel regex",
    "selected_positions": "Selected positions",
    "selected_channels": "Selected channels",
    "first_timepoint": "First timepoint",
    "last_timepoint": "Last timepoint",
    "reference_channel": "Reference channel",
    "validation_mode": "Validation mode",
    "quick_validation_samples_per_position_channel": "Quick-validation samples per position/channel",
    "processing_order": "Processing order",
    "coarse_crop_during_bgs": "Coarse crop during background subtraction",
    "dry_run": "Dry run",
    "resume": "Resume",
    "overwrite": "Overwrite",
    "start_stage": "Start stage",
    "stop_after_stage": "Stop after stage",
    "continue_on_error": "Continue on error",
    "fusion_crop_um": "Fusion analysis crop width (µm)",
    "fusion_save_as_crop": "Save a fixed central crop during fusion",
    "registration_glob_pattern": "Registration input TIFF pattern",
    "registration_downsample_xy": "Registration downsample XY",
    "registration_downsample_z": "Registration downsample Z",
    "registration_phase_upsample": "Registration phase upsample factor",
    "registration_series_index": "TIFF series index",
    "registration_channel_index": "Internal TIFF channel index",
    "registration_clip_low": "Registration contrast low percentile",
    "registration_clip_high": "Registration contrast high percentile",
    "registration_center_threshold_percentile": "Registration foreground-centre percentile",
    "registration_max_shift_z": "Maximum allowed Z shift",
    "registration_max_shift_yx": "Maximum allowed Y/X shift",
    "bgs_z_projection": "BGS Z background projection",
    "bgs_z_bg_percentile": "BGS low-Z percentile",
    "bgs_bg_sigma": "BGS background smoothing sigma",
    "bgs_bg_downsample": "BGS background smoothing downsample",
    "bgs_bg_alpha": "BGS subtraction strength (alpha)",
    "bgs_auto_offset": "Automatically remove residual baseline",
    "bgs_bg_offset": "Manual residual baseline offset",
    "bgs_bbox_downsample_xy": "Coarse-crop detection downsample XY",
    "bgs_bbox_downsample_z": "Coarse-crop detection downsample Z",
    "bgs_bbox_dilate": "Coarse-crop mask expansion",
    "bgs_mask_percentile": "Coarse-crop foreground percentile",
    "crop_no_crop_z": "Keep all Z slices (disable Z cropping)",
    "crop_keep_border_components": "Allow foreground touching the image border",
    "crop_square_xy": "Make final XY crop square",
    "crop_no_qc_png": "Skip crop QC PNG previews",
}

HELP_TEXT = {
    "run_mode": "Choose the complete workflow or run exactly one selected stage.",
    "single_stage": "In Single stage mode, only this stage runs; no previous-stage manifest is required.",
    "starting_data_type": "Choose raw View1/View2 data for the complete workflow, or already-fused TIFFs to begin at registration without rerunning fusion.",
    "input_dataset_folder": "Click Browse… to choose an existing local or mounted-server folder in Finder, or paste an absolute path.",
    "output_root": "Click Browse… to choose the output folder. Intermediate folders are never deleted automatically.",
    "selected_positions": "Use ALL or exact comma-separated position names from the discovery table.",
    "selected_channels": "Use ALL or exact comma-separated channel names from the discovery table.",
    "reference_channel": "Required when more than one selected channel is discovered.",
    "validation_mode": "Quick opens only sampled first/last TIFF pairs per position/channel. Full opens every selected TIFF pair.",
    "quick_validation_samples_per_position_channel": "Number of TIFF pairs sampled per position/channel in Quick mode. Two checks the first and last selected timepoints.",
    "position_folder_pattern": "Python regular expression with named groups 'position' and 'settings'.",
    "settings_pattern": "Shell-style pattern. The default preserves the legacy 'Settings 1' convention.",
    "timepoint_regex": "Use a named group 'timepoint' or one capture group.",
    "channel_regex": "Use a named group 'channel'. {view1} and {view2} are expanded safely.",
    "fused_channel_regex": "Used only for already-fused input. Use a named group 'channel'; {fused} is expanded safely.",
    "stop_after_stage": "The value 'final' follows the selected processing order.",
    "coarse_crop_during_bgs": "Disabled by default. When enabled, BGS receives --crop-to-foreground with XY padding 80 and Z padding 8.",
    "registration_save_dtype": "Keep 'input' unless an explicit dtype conversion is intended.",
    "bgs_output_dtype": "Keep 'same' unless an explicit dtype conversion is intended.",
}

SETTING_HELP: dict[str, dict[str, str]] = {'input_dataset_folder': {'summary': 'The source folder used by the selected Starting data type.',
                          'effect': 'For raw input, discovery searches for View1/View2 pairs. For already-fused input, it searches directly for files such as t0001_561_Fused.tif. The pipeline never modifies files in this folder.',
                          'recommended': 'Choose the narrowest experiment folder. It may directly contain one position’s fused TIFFs, or contain one source folder per position.',
                          'units': 'Absolute folder path',
                          'caution': 'Do not select a very broad server root; network directory traversal can be slow.'},
 'output_root': {'summary': 'The destination where fused, registered, background-subtracted, cropped, log, and '
                            'manifest files are written.',
                 'effect': 'Changing it sends all generated outputs and intermediate stages to a different folder.',
                 'recommended': 'Use a separate empty or dedicated result folder, preferably on fast local or HPC '
                                'scratch storage.',
                 'units': 'Absolute folder path',
                 'caution': 'Do not use the original input folder as the output root.'},
 'position_folder_pattern': {'summary': 'A regular expression that extracts the exact position name and settings name '
                                        'from each acquisition folder.',
                             'effect': 'Changing it changes how folders are grouped into organoids/positions. '
                                       'Incorrect patterns can make valid folders invisible or merge names '
                                       'incorrectly.',
                             'recommended': 'Leave unchanged when folders look like “Position 1_Settings 1”. Edit only '
                                            'when your folder naming convention differs.',
                             'units': 'Python regular expression',
                             'caution': 'Use named groups (?P<position>...) and (?P<settings>...). Ambiguous matches '
                                        'are rejected.'},
 'settings_pattern': {'summary': 'Selects which settings/acquisition folder is used when a position has more than one '
                                 'settings folder.',
                      'effect': 'A broader pattern includes more folders; if multiple interpretations remain, '
                                'validation stops instead of guessing.',
                      'recommended': 'Use the exact setting name when possible, such as “Settings 1”.',
                      'units': 'Shell-style glob pattern',
                      'caution': 'Using * can create ambiguity when multiple settings folders exist.'},
 'view1_token': {'summary': 'The exact filename token identifying camera View 1.',
                 'effect': 'Files containing this token are assigned to the first camera when pairs are discovered.',
                 'recommended': 'Keep “View1” unless your files use another token such as CamLeft.',
                 'units': 'Filename text token',
                 'caution': 'It must differ from the View 2 token.'},
 'view2_token': {'summary': 'The exact filename token identifying camera View 2.',
                 'effect': 'Files containing this token are assigned to the second camera when pairs are discovered.',
                 'recommended': 'Keep “View2” unless your files use another token such as CamRight.',
                 'units': 'Filename text token',
                 'caution': 'It must differ from the View 1 token.'},
 'fused_token': {'summary': 'The token inserted into filenames produced by camera fusion.',
                 'effect': 'Changing it changes generated fused TIFF names and the rewritten OME companion entries.',
                 'recommended': 'Keep “Fused” for compatibility with the existing workflow.',
                 'units': 'Filename text token'},
 'tiff_glob_pattern': {'summary': 'Controls which source files are considered TIFF candidates during discovery.',
                       'effect': 'A narrower pattern speeds scanning and excludes unrelated files; a broader pattern '
                                 'may include unwanted TIFFs.',
                       'recommended': 'Use “*.tif*” for .tif and .tiff files, or a narrower pattern such as “t*.tif” '
                                      'when appropriate.',
                       'units': 'Shell-style file glob'},
 'ome_companion_glob_pattern': {'summary': 'Controls how the OME companion XML file is located in each selected '
                                           'acquisition folder.',
                                'effect': 'The OME companion supplies voxel spacing and fusion metadata. Exactly one '
                                          'matching file is required per position.',
                                'recommended': 'Use a specific pattern if the folder contains more than one .ome file.',
                                'units': 'Shell-style file glob',
                                'caution': 'Missing or duplicate matches stop validation.'},
 'timepoint_regex': {'summary': 'Extracts the timepoint number from each source TIFF filename.',
                     'effect': 'Changing it changes how files are grouped over time and how First/Last timepoint '
                               'filtering is applied.',
                     'recommended': 'Keep the default for names such as t0001_488_View1.tif.',
                     'units': 'Python regular expression',
                     'caution': 'Use a named group (?P<timepoint>\\d+) or one capture group.'},
 'channel_regex': {'summary': 'Extracts the channel name from each source TIFF filename.',
                   'effect': 'Changing it changes channel grouping, selected-channel filtering, and reference-channel '
                             'validation.',
                   'recommended': 'Keep the default for names such as t0001_488_View1.tif.',
                   'units': 'Python regular expression',
                   'caution': 'Use a named group (?P<channel>...). Tokens {view1}, {view2}, and {fused} are expanded '
                              'automatically.'},
 'selected_positions': {'summary': 'Chooses which detected organoids/positions are processed.',
                        'effect': 'ALL processes every detected position. A comma-separated list restricts scanning, '
                                  'validation, and processing to exact discovered names.',
                        'recommended': 'Use one position for the first test, then switch to ALL after checking the '
                                       'results.',
                        'units': 'ALL or comma-separated exact names',
                        'caution': 'Enter “Position 1”, not only “1”, when that is the discovered name.'},
 'selected_channels': {'summary': 'Chooses which detected image channels are fused and processed.',
                       'effect': 'ALL includes every detected channel. A comma-separated list limits processing and '
                                 'can substantially reduce runtime and output size.',
                       'recommended': 'Start with the main fluorescence channel, for example 488. Add Transmitted only '
                                      'when it is needed.',
                       'units': 'ALL or comma-separated exact names'},
 'first_timepoint': {'summary': 'The earliest numeric timepoint included in the run.',
                     'effect': 'Increasing it skips earlier TIFF pairs. Registration uses the first selected timepoint '
                               'as the reference for each position.',
                     'recommended': 'Use 1 for a complete experiment; use a later value only when intentionally '
                                    'processing a subset.',
                     'units': 'Timepoint number',
                     'caution': 'Changing this can change the registration reference timepoint.'},
 'last_timepoint': {'summary': 'The latest numeric timepoint included in the run.',
                    'effect': 'Lowering it creates a smaller test subset and makes scanning, validation, and '
                              'processing faster.',
                    'recommended': 'Use 2 or 3 for the first validation run, then expand to the full experiment.',
                    'units': 'Timepoint number'},
 'reference_channel': {'summary': 'The channel used to calculate the camera-fusion switching plane at each timepoint.',
                       'effect': 'The switching plane calculated from this channel is reused for the other selected '
                                 'channels at that timepoint.',
                       'recommended': 'Choose a bright, consistently present structural fluorescence channel such as '
                                      '488 or 561.',
                       'units': 'Exact discovered channel name',
                       'caution': 'It must be present for every selected position and timepoint. It is required when '
                                  'multiple channels are selected.'},
 'validation_mode': {'summary': 'Controls how many TIFF headers are opened before processing.',
                     'effect': 'Quick validates filenames and all file sizes but samples TIFF headers. Full opens '
                               'every selected View 1/View 2 header pair.',
                     'recommended': 'Use Full for a small first test or a new export format; use Quick for routine '
                                    'full datasets, especially on server storage.',
                     'units': 'quick or full',
                     'caution': 'Quick validation may not detect a damaged TIFF in the middle of a long time series '
                                'until processing reaches it.'},
 'quick_validation_samples_per_position_channel': {'summary': 'Number of timepoints sampled for TIFF-header checks in '
                                                              'Quick validation mode.',
                                                   'effect': 'Higher values increase confidence but require more '
                                                             'network/file operations. Two samples the first and last '
                                                             'selected timepoints.',
                                                   'recommended': 'Keep 2 for routine use. Increase to 3–5 when files '
                                                                  'may vary within an experiment.',
                                                   'units': 'TIFF pairs per position/channel'},
 'processing_order': {'summary': 'Selects whether background subtraction or object-aware cropping runs first after '
                                 'registration.',
                      'effect': 'Background subtraction → Crop preserves full background context. Crop → Background '
                                'subtraction processes fewer voxels but changes the context seen by the background '
                                'estimator.',
                      'recommended': 'Keep Background subtraction → Crop as the standard and validation default.',
                      'units': 'Workflow order',
                      'caution': 'Crop-first can be faster but crop detection sees raw haze and BGS sees less '
                                 'surrounding background.'},
 'coarse_crop_during_bgs': {'summary': 'Optionally lets the BGS stage make a broad foreground crop while correcting '
                                       'intensity.',
                            'effect': 'When enabled, BGS receives --crop-to-foreground with XY padding 80 and Z '
                                      'padding 8. A separate tighter final crop still runs afterward.',
                            'recommended': 'Keep disabled by default. Enable only after a representative small test '
                                           'proves that weak peripheral signal remains safely inside the coarse box.',
                            'units': 'On/off',
                            'caution': 'It can reduce runtime and disk use, but may remove weak edge signal and gives '
                                       'BGS less background context.'},
 'dry_run': {'summary': 'Builds and prints all commands without executing image processing.',
             'effect': 'No TIFF outputs are created. Discovery, validation, resolved folders, and command construction '
                       'are still checked.',
             'recommended': 'Enable before the first real run or after changing naming rules and processing order.',
             'units': 'On/off'},
 'resume': {'summary': 'Reuses completed stages only when manifests and every expected output TIFF validate.',
            'effect': 'Valid completed stages are skipped; incomplete, changed, or mismatched stages are rerun.',
            'recommended': 'Enable after an interrupted run or when continuing a validated output root.',
            'units': 'On/off',
            'caution': 'A nonempty folder alone is never accepted as a completed stage.'},
 'overwrite': {'summary': 'Allows a rerun stage to replace output TIFFs that already exist.',
               'effect': 'When disabled, existing outputs are protected but partial stages may fail or be skipped by '
                         'child scripts.',
               'recommended': 'Keep enabled with Resume: valid stages are skipped, while invalid partial stages can be '
                              'rebuilt.',
               'units': 'On/off'},
 'start_stage': {'summary': 'Selects the first stage to execute.',
                 'effect': 'Starting later skips earlier processing only when their output folders and stage manifests '
                           'validate.',
                 'recommended': 'Use fusion for a complete run. Use a later stage only when continuing from verified '
                                'intermediate outputs.',
                 'units': 'Stage name'},
 'stop_after_stage': {'summary': 'Stops cleanly after the selected stage.',
                      'effect': 'Useful for testing fusion or registration before committing to the full workflow. '
                                '“final” follows the selected processing order.',
                      'recommended': 'Use final for complete processing; use fusion or registration for small '
                                     'validation runs.',
                      'units': 'Stage name'},
 'continue_on_error': {'summary': 'Controls whether the runner attempts other positions after one position fails.',
                       'effect': 'When enabled, remaining positions in the same stage are attempted, but dependent '
                                 'stages do not proceed if the stage is incomplete.',
                       'recommended': 'Keep disabled for first runs so the first error is immediately visible.',
                       'units': 'On/off'},
 'fusion_crop_um': {'summary': 'Physical XY width used by the original fusion switching-plane analysis and optional '
                               'fixed central save crop.',
                    'effect': 'The value is converted to pixels using OME PhysicalSizeX. It affects the region used by '
                              'the existing fusion calculation.',
                    'recommended': 'Keep 950 µm unless a different value has already been validated for your '
                                   'acquisition geometry.',
                    'units': 'Micrometres (µm)',
                    'caution': 'This is not the final object-aware crop.'},
 'fusion_save_as_crop': {'summary': 'Writes only a fixed central fusion crop instead of the full fused XY frame.',
                         'effect': 'Can reduce fusion output size early, but assumes the useful object is centred and '
                                   'uses Fusion crop µm rather than object detection.',
                         'recommended': 'Keep disabled for initial runs; rely on the final object-aware crop.',
                         'units': 'On/off',
                         'caution': 'A fixed central crop can clip off-centre biological signal.'},
 'registration_glob_pattern': {'summary': 'Selects fused TIFF files processed by registration.',
                               'effect': 'A narrower pattern excludes files; a broader pattern may include unrelated '
                                         'TIFFs in the stage folder.',
                               'recommended': 'Keep “*.tif*”.',
                               'units': 'Shell-style file glob'},
 'registration_downsample_xy': {'summary': 'XY reduction factor used only when estimating registration shifts.',
                                'effect': 'Higher values are faster and use less memory but reduce shift-estimation '
                                          'detail. Full-resolution TIFFs are still shifted and written.',
                                'recommended': 'Keep 4. Try 2 for difficult subtle drift; use 1 only for small data.',
                                'units': 'Integer reduction factor'},
 'registration_downsample_z': {'summary': 'Z reduction factor used only when estimating registration shifts.',
                               'effect': 'Higher values speed estimation but reduce sensitivity to Z drift.',
                               'recommended': 'Keep 1 unless Z stacks are extremely deep and Z accuracy can be '
                                              'relaxed.',
                               'units': 'Integer reduction factor'},
 'registration_estimate_method': {'summary': 'Algorithm used to estimate 3D translation between timepoints.',
                                  'effect': 'center is fastest; phase_mip uses projection phase correlation; '
                                            'center_phase_mip combines a robust centre estimate with a smaller phase '
                                            'refinement.',
                                  'recommended': 'Keep center_phase_mip for the safest general-purpose default.',
                                  'units': 'Method choice'},
 'registration_apply_mode': {'summary': 'Controls whether estimated shifts are rounded to whole pixels or applied with '
                                        'interpolation.',
                             'effect': 'integer is faster and preserves original voxel values. subpixel is more '
                                       'precise but interpolates intensities and uses more memory.',
                             'recommended': 'Keep integer for initial large-data processing.',
                             'units': 'integer or subpixel'},
 'registration_compression': {'summary': 'Compression used when writing registered TIFFs.',
                              'effect': 'Compression reduces disk size but increases writing time and may require '
                                        'imagecodecs. It does not change voxel dimensions or dtype.',
                              'recommended': 'Keep none for speed and maximum compatibility; use zlib only when '
                                             'imagecodecs is installed and storage is limited.',
                              'units': 'TIFF compression codec'},
 'registration_save_dtype': {'summary': 'Output dtype for registered TIFFs.',
                             'effect': 'input preserves the source dtype. Other choices explicitly convert voxel '
                                       'values and can change precision, dynamic range, and file size.',
                             'recommended': 'Keep input unless conversion is intentional and validated.',
                             'units': 'TIFF pixel dtype',
                             'caution': 'uint8 substantially reduces intensity precision.'},
 'registration_padding_z': {'summary': 'Zero-valued Z margin added before applying registration shifts.',
                            'effect': 'Larger padding protects against clipping during Z translation but increases '
                                      'output voxel count and file size.',
                            'recommended': 'Keep 8 unless observed Z drift approaches the boundary.',
                            'units': 'Z slices'},
 'registration_padding_y': {'summary': 'Zero-valued Y margin added before applying registration shifts.',
                            'effect': 'Larger padding protects against clipping during Y translation but increases '
                                      'every registered volume.',
                            'recommended': 'Keep 64 unless measured drift requires more or storage pressure requires a '
                                           'validated smaller value.',
                            'units': 'Pixels'},
 'registration_padding_x': {'summary': 'Zero-valued X margin added before applying registration shifts.',
                            'effect': 'Larger padding protects against clipping during X translation but increases '
                                      'every registered volume.',
                            'recommended': 'Keep 64 unless measured drift requires more or storage pressure requires a '
                                           'validated smaller value.',
                            'units': 'Pixels'},
 'registration_phase_upsample': {'summary': 'Upsampling factor used by phase-correlation refinement.',
                                 'effect': 'Higher values may estimate finer fractional shifts but cost more '
                                           'processing time and are most relevant with subpixel application.',
                                 'recommended': 'Keep 1 for integer registration; test 2–4 only when subpixel '
                                                'precision is required.',
                                 'units': 'Integer factor'},
 'registration_series_index': {'summary': 'Selects which TIFF series is read when a file contains multiple series.',
                               'effect': 'Changing it selects a different internal image series.',
                               'recommended': 'Keep 0 for normal single-series Z-stack TIFFs.',
                               'units': 'Zero-based TIFF series index'},
 'registration_channel_index': {'summary': 'Selects one channel axis inside a multichannel TIFF.',
                                'effect': 'When blank, the script expects each TIFF to contain one channel. A number '
                                          'selects that internal channel index.',
                                'recommended': 'Leave blank because this pipeline normally stores one channel per '
                                               'TIFF.',
                                'units': 'Blank or zero-based channel index'},
 'registration_clip_low': {'summary': 'Low percentile used to rescale the downsampled image for shift estimation.',
                           'effect': 'Increasing it suppresses more dim background; excessive values may discard weak '
                                     'useful structures.',
                           'recommended': 'Keep 1.0.',
                           'units': 'Percentile'},
 'registration_clip_high': {'summary': 'High percentile used to rescale the downsampled image for shift estimation.',
                            'effect': 'Lowering it reduces the influence of very bright outliers; excessive lowering '
                                      'can flatten strong structures.',
                            'recommended': 'Keep 99.0.',
                            'units': 'Percentile'},
 'registration_center_threshold_percentile': {'summary': 'Brightness percentile used to weight the foreground centre '
                                                         'estimate.',
                                              'effect': 'Higher values focus on brighter structures; lower values '
                                                        'include more diffuse signal and background.',
                                              'recommended': 'Keep 75 unless centre estimates are visibly unstable.',
                                              'units': 'Percentile'},
 'registration_max_shift_z': {'summary': 'Optional safety limit for the applied Z shift.',
                              'effect': 'When set, larger estimated Z shifts are clipped to this magnitude.',
                              'recommended': 'Leave blank unless you know the physically plausible maximum drift.',
                              'units': 'Full-resolution Z pixels',
                              'caution': 'A limit that is too small prevents correction of real drift.'},
 'registration_max_shift_yx': {'summary': 'Optional shared safety limit for applied Y and X shifts.',
                               'effect': 'When set, larger Y/X estimates are clipped to this magnitude.',
                               'recommended': 'Leave blank unless you know the physically plausible maximum drift.',
                               'units': 'Full-resolution pixels',
                               'caution': 'A limit that is too small prevents correction of real drift.'},
 'bgs_glob_pattern': {'summary': 'Selects TIFF files processed by background subtraction.',
                      'effect': 'A narrower pattern excludes files; a broader pattern may include unrelated TIFFs.',
                      'recommended': 'Keep “*.tif*”.',
                      'units': 'Shell-style file glob'},
 'bgs_method': {'summary': 'Background-estimation algorithm.',
                'effect': 'zmin_smooth estimates a smooth XY background from the minimum or low percentile through Z '
                          'and subtracts it from every Z slice.',
                'recommended': 'Keep zmin_smooth to preserve the validated algorithm.',
                'units': 'Method choice'},
 'bgs_z_projection': {'summary': 'Statistic used through Z to estimate the XY background image.',
                      'effect': 'min is memory-efficient and follows the original assumption that each XY location '
                                'sees background in at least one Z plane. percentile is more robust to unusually dark '
                                'pixels but requires more memory.',
                      'recommended': 'Keep min for large data unless percentile has been specifically validated.',
                      'units': 'min or percentile'},
 'bgs_z_bg_percentile': {'summary': 'Low Z percentile used when BGS Z projection is set to percentile.',
                         'effect': 'Higher values estimate a brighter background and subtract more signal; lower '
                                   'values approach a minimum projection.',
                         'recommended': 'Keep 2.0 when percentile mode is used.',
                         'units': 'Percentile',
                         'caution': 'Ignored when Z projection is min.'},
 'bgs_bg_sigma': {'summary': 'Gaussian smoothing scale for the estimated XY background.',
                  'effect': 'Higher values model broader illumination gradients. Lower values follow smaller '
                            'structures and may subtract biological features.',
                  'recommended': 'Keep 80 for the validated default, then inspect weak structures before changing it.',
                  'units': 'Pixels at full resolution'},
 'bgs_bg_downsample': {'summary': 'Temporary XY reduction factor used while smoothing the background image.',
                       'effect': 'Higher values improve speed but make the background model less spatially detailed. '
                                 'Final output remains full resolution.',
                       'recommended': 'Keep 8.',
                       'units': 'Integer reduction factor'},
 'bgs_bg_alpha': {'summary': 'Multiplier controlling background-subtraction strength.',
                  'effect': 'The correction is input − alpha × background. Values above 1 subtract more; values below '
                            '1 subtract less.',
                  'recommended': 'Keep 1.0 for initial validation. Test 1.05 only when a residual background is '
                                 'consistently visible.',
                  'units': 'Dimensionless multiplier',
                  'caution': 'High values can remove weak biological signal.'},
 'bgs_auto_offset': {'summary': 'Automatically estimates and removes a remaining positive baseline after '
                                'smooth-background subtraction.',
                     'effect': 'When disabled, the fixed BGS background offset value is used instead.',
                     'recommended': 'Keep enabled for the validated default behavior.',
                     'units': 'On/off'},
 'bgs_offset_percentile': {'summary': 'Percentile of sampled residual baseline values used by automatic offset '
                                      'removal.',
                           'effect': 'Higher values remove a larger residual offset; lower values are more '
                                     'conservative.',
                           'recommended': 'Keep 50.',
                           'units': 'Percentile',
                           'caution': 'Used only when automatic offset is enabled.'},
 'bgs_offset_sample_slices': {'summary': 'Approximate maximum number of Z slices sampled to estimate the residual '
                                         'offset.',
                              'effect': 'More samples can improve representativeness but require more reads.',
                              'recommended': 'Keep 24.',
                              'units': 'Z slices'},
 'bgs_bg_offset': {'summary': 'Fixed residual baseline subtracted when automatic offset is disabled.',
                   'effect': 'Increasing it subtracts the same additional intensity from every corrected voxel.',
                   'recommended': 'Keep 0 unless a manual offset has been measured.',
                   'units': 'Intensity units',
                   'caution': 'Ignored when automatic offset is enabled.'},
 'bgs_crop_padding_xy': {'summary': 'XY safety margin retained around the coarse BGS foreground box.',
                         'effect': 'Larger values reduce clipping risk but preserve more voxels and reduce the speed '
                                   'benefit.',
                         'recommended': 'Keep 80 when coarse BGS cropping is enabled.',
                         'units': 'Pixels',
                         'caution': 'Used only when Coarse crop during background subtraction is enabled.'},
 'bgs_crop_padding_z': {'summary': 'Z safety margin retained around the coarse BGS foreground box.',
                        'effect': 'Larger values reduce clipping risk but preserve more Z slices.',
                        'recommended': 'Keep 8 when coarse BGS cropping is enabled.',
                        'units': 'Z slices',
                        'caution': 'Used only when Coarse crop during background subtraction is enabled.'},
 'bgs_bbox_downsample_xy': {'summary': 'XY reduction factor used to detect the coarse BGS foreground box.',
                            'effect': 'Higher values speed detection but make the bounding box less precise.',
                            'recommended': 'Keep 8.',
                            'units': 'Integer reduction factor',
                            'caution': 'Used only when coarse BGS cropping is enabled.'},
 'bgs_bbox_downsample_z': {'summary': 'Z reduction factor used to detect the coarse BGS foreground box.',
                           'effect': 'Higher values speed detection but reduce Z-boundary precision.',
                           'recommended': 'Keep 2.',
                           'units': 'Integer reduction factor',
                           'caution': 'Used only when coarse BGS cropping is enabled.'},
 'bgs_bbox_dilate': {'summary': 'Expands the downsampled coarse foreground mask before building its bounding box.',
                     'effect': 'Higher values create a safer, larger coarse crop and reduce clipping risk.',
                     'recommended': 'Keep 2.',
                     'units': 'Morphological dilation iterations',
                     'caution': 'Used only when coarse BGS cropping is enabled.'},
 'bgs_mask_percentile': {'summary': 'Brightness percentile used to define foreground during coarse BGS crop detection.',
                         'effect': 'Higher values produce a tighter mask focused on bright signal; lower values '
                                   'include weaker signal and make the box larger.',
                         'recommended': 'Keep 70 and validate weak peripheral structures before increasing it.',
                         'units': 'Percentile',
                         'caution': 'Used only when coarse BGS cropping is enabled.'},
 'bgs_output_dtype': {'summary': 'Output dtype for background-subtracted TIFFs.',
                      'effect': 'same preserves the registered dtype. Other choices change precision, dynamic range, '
                                'and file size.',
                      'recommended': 'Keep same for quantitative analysis.',
                      'units': 'TIFF pixel dtype',
                      'caution': 'uint8 substantially reduces intensity precision.'},
 'bgs_compression': {'summary': 'Compression used when writing background-subtracted TIFFs.',
                     'effect': 'zlib usually compresses zero-rich corrected images well, but increases CPU time and '
                               'may require imagecodecs.',
                     'recommended': 'Use zlib when imagecodecs is installed; otherwise use none.',
                     'units': 'TIFF compression codec'},
 'crop_glob_pattern': {'summary': 'Selects TIFF files processed by the final object-aware crop stage.',
                       'effect': 'A narrower pattern excludes files; a broader pattern may include unrelated TIFFs.',
                       'recommended': 'Keep “*.tif*”.',
                       'units': 'Shell-style file glob'},
 'crop_analysis_downsample_xy': {'summary': 'XY reduction factor used only for object detection.',
                                 'effect': 'Higher values make crop estimation faster but less precise. The output '
                                           'crop is still taken from full-resolution TIFFs.',
                                 'recommended': 'Keep 4.',
                                 'units': 'Integer reduction factor'},
 'crop_analysis_downsample_z': {'summary': 'Z reduction factor used only for crop detection.',
                                'effect': 'Higher values reduce analysis time but make Z-boundary detection less '
                                          'precise.',
                                'recommended': 'Keep 1, especially while Z cropping is enabled.',
                                'units': 'Integer reduction factor'},
 'crop_xy_projection': {'summary': 'Projection through Z used to detect the organoid in XY.',
                        'effect': 'percentile is robust to isolated hot pixels; max is sensitive to any bright '
                                  'structure; mean can dilute thin structures.',
                        'recommended': 'Keep percentile.',
                        'units': 'Projection method'},
 'crop_xy_projection_percentile': {'summary': 'Percentile through Z used when XY projection is percentile.',
                                   'effect': 'Higher values behave more like a maximum projection; lower values '
                                             'suppress sparse bright structures more strongly.',
                                   'recommended': 'Keep 95.',
                                   'units': 'Percentile',
                                   'caution': 'Ignored for max or mean projection.'},
 'crop_xy_smooth_sigma': {'summary': 'Gaussian smoothing applied to the downsampled XY projection before thresholding.',
                          'effect': 'Higher values suppress more small speckles but can blur small real protrusions.',
                          'recommended': 'Keep 2.',
                          'units': 'Downsampled pixels'},
 'crop_xy_threshold_method': {'summary': 'Method used to separate foreground organoid from background in the XY '
                                         'projection.',
                              'effect': 'otsu chooses a histogram threshold with a floor; percentile uses a fixed '
                                        'positive-pixel percentile.',
                              'recommended': 'Keep otsu for general use.',
                              'units': 'Threshold method'},
 'crop_xy_threshold_percentile': {'summary': 'Foreground percentile used when XY threshold method is percentile.',
                                  'effect': 'Higher values make a tighter brighter mask; lower values include weaker '
                                            'signal.',
                                  'recommended': 'Keep 70 when percentile thresholding is used.',
                                  'units': 'Percentile',
                                  'caution': 'Ignored when threshold method is otsu.'},
 'crop_xy_threshold_floor_percentile': {'summary': 'Minimum allowed threshold percentile when Otsu would select a very '
                                                   'low threshold.',
                                        'effect': 'Higher values reject more broad haze and make the crop tighter; '
                                                  'excessive values can exclude weak object edges.',
                                        'recommended': 'Keep 65.',
                                        'units': 'Percentile'},
 'crop_min_area_pixels': {'summary': 'Expected minimum connected-component area in the downsampled XY mask.',
                          'effect': 'Helps distinguish the main organoid from very small detections.',
                          'recommended': 'Keep 500 unless your downsampled organoid is genuinely smaller.',
                          'units': 'Downsampled pixels'},
 'crop_keep_border_components': {'summary': 'Allows foreground components touching the XY image border to remain '
                                            'eligible as the main object.',
                                 'effect': 'Enabling it can preserve a real organoid touching an edge, but can also '
                                           'retain bright border artifacts.',
                                 'recommended': 'Keep disabled unless the organoid legitimately touches the frame '
                                                'boundary.',
                                 'units': 'On/off'},
 'crop_no_crop_z': {'summary': 'Keeps the complete Z range while cropping only XY.',
                    'effect': 'Enabled means no Z slices are removed. Disabled enables Z-profile-based cropping and '
                              'can reduce data further.',
                    'recommended': 'Keep enabled for the safest default. Validate Z detection carefully before '
                                   'disabling it.',
                    'units': 'On/off',
                    'caution': 'The label means “do not crop Z”; enabling it preserves all Z slices.'},
 'crop_z_profile_percentile': {'summary': 'High XY percentile calculated for each Z plane when Z cropping is enabled.',
                               'effect': 'Higher values focus the Z profile on the brightest structures.',
                               'recommended': 'Keep 99.',
                               'units': 'Percentile',
                               'caution': 'Ignored while Crop no crop Z is enabled.'},
 'crop_z_profile_fraction': {'summary': 'Fraction of the Z-profile range used as the active-Z threshold.',
                             'effect': 'Higher values crop Z more aggressively; lower values preserve more slices.',
                             'recommended': 'Keep 0.04 and inspect Z boundaries before increasing it.',
                             'units': 'Fraction from 0 to 1',
                             'caution': 'Ignored while Crop no crop Z is enabled.'},
 'crop_z_smooth_sigma': {'summary': 'Smoothing applied along the Z activity profile before estimating Z boundaries.',
                         'effect': 'Higher values suppress plane-to-plane fluctuations but can blur short real Z '
                                   'features.',
                         'recommended': 'Keep 1.',
                         'units': 'Z-profile samples',
                         'caution': 'Ignored while Crop no crop Z is enabled.'},
 'crop_padding_xy': {'summary': 'Full-resolution XY safety margin added around the common detected organoid crop.',
                     'effect': 'Larger values preserve more context and reduce clipping risk but increase output '
                               'dimensions.',
                     'recommended': 'Keep 80 for initial validation; decrease only after checking QC previews across '
                                    'positions.',
                     'units': 'Pixels'},
 'crop_padding_z': {'summary': 'Z safety margin added around the detected Z range.',
                    'effect': 'Larger values preserve more Z context and reduce clipping risk.',
                    'recommended': 'Keep 8.',
                    'units': 'Z slices',
                    'caution': 'Ignored while Crop no crop Z is enabled.'},
 'crop_square_xy': {'summary': 'Expands the final XY crop to a square around the detected object.',
                    'effect': 'Useful for downstream tools requiring square frames, but usually retains more voxels '
                              'than a rectangular crop.',
                    'recommended': 'Keep disabled unless square input is required downstream.',
                    'units': 'On/off'},
 'crop_compression': {'summary': 'Compression used when writing final cropped TIFFs.',
                      'effect': 'zlib reduces storage but increases CPU time and may require imagecodecs. Cropping '
                                'itself reduces actual voxel dimensions regardless of compression.',
                      'recommended': 'Use zlib when imagecodecs is installed; otherwise use none.',
                      'units': 'TIFF compression codec'},
 'crop_no_qc_png': {'summary': 'Disables generation of crop quality-control PNG previews.',
                    'effect': 'Enabling it saves a little processing time and storage but removes the easiest visual '
                              'check of crop boundaries.',
                    'recommended': 'Keep disabled for initial and validation runs so QC PNGs are generated.',
                    'units': 'On/off',
                    'caution': 'The label means “do not create QC PNGs”; enabling it suppresses previews.'}}


LABEL_OVERRIDES.update({
    "enable_denoising": "Enable optional denoising",
    "denoise_glob_pattern": "Denoising input TIFF pattern",
    "denoise_method": "Denoising method",
    "denoise_median_size": "Median filter size",
    "denoise_sigma_xy": "Gaussian smoothing sigma XY",
    "denoise_sigma_z": "Gaussian smoothing sigma Z",
    "denoise_intensity_floor": "Intensity floor after denoising",
    "denoise_output_dtype": "Denoising output dtype",
    "denoise_compression": "Denoising compression",
    "denoise_crop_to_foreground": "Extra foreground crop during denoising",
    "denoise_crop_padding_xy": "Denoising crop padding XY",
    "denoise_crop_padding_z": "Denoising crop padding Z",
    "denoise_mask_percentile": "Denoising crop mask percentile",
    "denoise_min_size_voxels": "Denoising crop minimum foreground voxels",
})

SETTING_HELP.update({
    "enable_denoising": {
        "summary": "Adds an optional denoising stage immediately after background subtraction.",
        "effect": "When enabled, every 3D TIFF is denoised before the next stage. In the standard Background subtraction → Crop order, the sequence becomes BGS → Denoise → Final crop. In Crop → BGS order, denoising runs last.",
        "recommended": "Keep disabled for the first baseline run, then compare a representative small subset with denoising enabled. Enable it when camera noise or isolated bright speckles interfere with analysis.",
        "units": "On/off",
        "caution": "Denoising changes voxel intensities. Strong settings can blur fine structures or remove weak fluorescence.",
    },
    "denoise_glob_pattern": {
        "summary": "Selects TIFF files entering the optional denoising stage.",
        "effect": "A narrower pattern excludes files; an incorrect pattern can leave the stage with no inputs.",
        "recommended": "Keep *.tif* so the stage processes the preceding pipeline outputs.",
        "units": "Shell-style file glob",
    },
    "denoise_method": {
        "summary": "Chooses the spatial denoising algorithm applied to each 3D timepoint.",
        "effect": "none only applies the intensity floor; median2d removes isolated speckles slice-by-slice; gaussian2d smooths each XY slice; median_gaussian2d combines mild despeckling and smoothing; gaussian3d also smooths across Z.",
        "recommended": "Use median_gaussian2d for a conservative first test. Use gaussian3d only when Z spacing and structures justify cross-slice smoothing.",
        "units": "Algorithm choice",
        "caution": "Gaussian smoothing can blur small structures. 3D smoothing can mix information between Z planes.",
    },
    "denoise_median_size": {
        "summary": "Sets the 2D median-filter kernel used by median2d and median_gaussian2d.",
        "effect": "Larger values remove larger speckles but increasingly alter edges and small objects. Even values are rounded up by the denoising script.",
        "recommended": "Keep 3 for mild despeckling. Use 1 to disable the median component.",
        "units": "Pixels; odd integer",
        "caution": "Values above 3 should be validated carefully on fine biological structures.",
    },
    "denoise_sigma_xy": {
        "summary": "Controls Gaussian smoothing strength within each XY slice.",
        "effect": "Higher values produce smoother images and stronger noise suppression, but broaden edges and reduce fine detail.",
        "recommended": "Start at 0.3–0.6 pixels. The package default is 0.6.",
        "units": "XY pixels",
        "caution": "Large sigma values can merge nearby structures or reduce peak intensity.",
    },
    "denoise_sigma_z": {
        "summary": "Controls smoothing between adjacent Z planes when method is gaussian3d.",
        "effect": "Increasing it mixes signal across more Z planes. It has no effect for the 2D methods.",
        "recommended": "Keep 0 for anisotropic light-sheet data unless cross-Z smoothing has been specifically validated.",
        "units": "Z slices",
        "caution": "Z spacing is often much larger than XY spacing, so the same numeric sigma is not physically isotropic.",
    },
    "denoise_intensity_floor": {
        "summary": "Sets denoised values below a threshold to zero.",
        "effect": "Higher values remove more residual low-level noise and may improve compression, but can also erase weak fluorescence.",
        "recommended": "Start with 3–5 for uint16 data only after checking weak-signal regions. Set 0 to disable thresholding.",
        "units": "Input intensity units",
        "caution": "This is an absolute threshold, so its meaning depends on acquisition gain, offset, and dtype.",
    },
    "denoise_output_dtype": {
        "summary": "Selects the TIFF dtype written after denoising.",
        "effect": "same preserves the incoming dtype. uint16, uint8, or float32 explicitly convert intensities and change file size and quantitative interpretation.",
        "recommended": "Keep same for quantitative analysis.",
        "units": "TIFF dtype",
        "caution": "uint8 rescales each volume and loses quantitative precision; use it only for non-quantitative previews.",
    },
    "denoise_compression": {
        "summary": "Selects TIFF compression for denoised outputs.",
        "effect": "zlib/lzw reduce disk usage but require suitable codecs and more CPU; none writes larger files with fewer codec dependencies.",
        "recommended": "Use zlib when imagecodecs works. Use none when the interface reports missing imagecodecs.",
        "units": "Compression codec",
    },
    "denoise_crop_to_foreground": {
        "summary": "Lets denoising calculate an additional shared foreground crop across all timepoints in a position.",
        "effect": "When enabled, the denoising stage reduces dimensions before writing its outputs, using the padding and mask settings below.",
        "recommended": "Keep disabled because the pipeline already provides a dedicated final object-aware crop.",
        "units": "On/off",
        "caution": "This is a third possible crop in addition to coarse BGS crop and final crop. It can remove weak or moving peripheral signal.",
    },
    "denoise_crop_padding_xy": {
        "summary": "Adds XY margin around the optional denoising foreground crop.",
        "effect": "Higher values retain more surrounding pixels and reduce clipping risk, but produce larger outputs.",
        "recommended": "Keep 40 if the extra denoising crop is intentionally enabled.",
        "units": "Full-resolution pixels",
    },
    "denoise_crop_padding_z": {
        "summary": "Adds Z margin around the optional denoising foreground crop.",
        "effect": "Higher values retain more Z slices around detected foreground.",
        "recommended": "Keep 4 if the extra denoising crop is intentionally enabled.",
        "units": "Z slices",
    },
    "denoise_mask_percentile": {
        "summary": "Sets the foreground threshold percentile for the optional denoising crop.",
        "effect": "Higher values focus on brighter voxels and may tighten the crop; lower values include weaker signal and enlarge it.",
        "recommended": "Keep 70 and inspect representative weak-signal timepoints before changing it.",
        "units": "Percentile of positive voxels",
    },
    "denoise_min_size_voxels": {
        "summary": "Requires at least this many foreground voxels before a timepoint contributes to the optional denoising crop.",
        "effect": "Higher values ignore smaller detections; lower values make the crop more sensitive to small components and noise.",
        "recommended": "Keep 10000 unless your true object is substantially smaller.",
        "units": "Voxels",
    },
})

SETTING_HELP.update({
    "run_mode": {
        "summary": "Chooses between the complete end-to-end workflow and exactly one processing stage.",
        "effect": "Complete workflow follows the configured processing order. Single stage runs only the selected operation and reads TIFFs directly from the chosen input folder.",
        "recommended": "Use Complete workflow for normal processing. Use Single stage to rerun registration, BGS, denoising, crop, or fusion independently.",
        "units": "Run type",
    },
    "single_stage": {
        "summary": "The one and only image-processing stage executed in Single stage mode.",
        "effect": "Registration, BGS, denoising, and crop accept a folder containing TIFFs for one position or a parent containing one TIFF folder per position. Fusion expects raw View1/View2 pairs.",
        "recommended": "Select the exact operation you want to run. The output is written to that stage’s normal numbered folder.",
        "units": "Stage name",
        "caution": "The input must already be appropriate for the selected stage; the pipeline does not silently run prerequisite stages.",
    },
    "processed_channel_regex": {
        "summary": "Extracts the channel from TIFF filenames used for a standalone registration, BGS, denoising, or crop run.",
        "effect": "Changing it changes which stage-ready TIFF files and channels are discovered.",
        "recommended": "Keep the default for names such as t0001_561_Fused.tif or t0001_561.tif.",
        "units": "Python regular expression",
        "caution": "Use a named group (?P<channel>...) or one capture group.",
    },
    "starting_data_type": {
        "summary": "Selects whether the input contains raw View1/View2 camera pairs or TIFFs that have already been fused.",
        "effect": "Raw input runs the complete workflow. Already fused input omits fusion and uses the selected folder directly as registration input.",
        "recommended": "Choose Already fused TIFFs for folders containing files such as t0001_561_Fused.tif plus one OME companion.",
        "units": "Workflow source type",
        "caution": "Already fused mode does not require a previous fusion manifest, but it still validates filenames, OME metadata, file sizes, and sampled TIFF headers.",
    },
    "fused_channel_regex": {
        "summary": "Extracts the channel name from an already-fused TIFF filename.",
        "effect": "Changing it changes which fused files and channels are discovered when Starting data type is Already fused TIFFs.",
        "recommended": "Keep the default for names such as t0001_561_Fused.tif.",
        "units": "Python regular expression",
        "caution": "The expression must contain a named group (?P<channel>...) or one capture group.",
    },
    "start_stage": {
        "summary": "Selects the first processing stage executed in this run.",
        "effect": "With raw input, choosing a later stage requires valid pipeline-created outputs and manifests for earlier stages. With already-fused input, registration can start directly from the selected input folder without a fusion manifest.",
        "recommended": "Use fusion for raw View1/View2 input. Use registration for Already fused TIFFs.",
        "units": "Stage name",
        "caution": "Starting after registration still requires valid outputs and manifests for the skipped downstream stages.",
    },
    "reference_channel": {
        "summary": "The raw camera-fusion reference channel used to calculate the switching plane at each timepoint.",
        "effect": "It affects fusion only. It is automatically disabled and ignored when the starting data are already fused.",
        "recommended": "For raw multi-channel input, choose the fluorescence channel with reliable structure at every selected timepoint.",
        "units": "Exact discovered channel name",
    },
})

RECOMMENDED_KEYS = {
    'run_mode', 'starting_data_type', 'processing_order', 'coarse_crop_during_bgs', 'enable_denoising', 'validation_mode',
    'registration_estimate_method', 'registration_apply_mode',
    'registration_save_dtype', 'bgs_output_dtype', 'denoise_output_dtype', 'crop_no_crop_z',
}

PATH_FIELDS = {"input_dataset_folder", "output_root"}


CHOICES = {
    "run_mode": [RUN_FULL, RUN_SINGLE],
    "single_stage": ["fusion", "registration", "background_subtraction", "denoise", "crop"],
    "starting_data_type": [INPUT_RAW, INPUT_FUSED],
    "validation_mode": ["quick", "full"],
    "processing_order": [ORDER_BGS_CROP, ORDER_CROP_BGS],
    "start_stage": ["fusion", "registration", "background_subtraction", "denoise", "crop"],
    "stop_after_stage": ["fusion", "registration", "background_subtraction", "denoise", "crop", "final"],
    "registration_estimate_method": ["center", "phase_mip", "center_phase_mip"],
    "registration_apply_mode": ["integer", "subpixel"],
    "registration_compression": ["none", "zlib", "lzw"],
    "registration_save_dtype": ["input", "uint16", "uint8", "float32"],
    "bgs_method": ["zmin_smooth"],
    "bgs_z_projection": ["min", "percentile"],
    "bgs_output_dtype": ["same", "uint16", "uint8", "float32"],
    "bgs_compression": ["none", "zlib", "lzw"],
    "denoise_method": ["none", "median2d", "gaussian2d", "median_gaussian2d", "gaussian3d"],
    "denoise_output_dtype": ["same", "uint16", "uint8", "float32"],
    "denoise_compression": ["none", "zlib", "lzw", "zstd"],
    "crop_xy_projection": ["percentile", "max", "mean"],
    "crop_xy_threshold_method": ["otsu", "percentile"],
    "crop_compression": ["none", "zlib", "lzw"],
}

GROUPS = [
    (
        "1. Choose data and selection",
        False,
        "beginner",
        [
            "run_mode", "single_stage", "starting_data_type", "input_dataset_folder", "output_root",
            "selected_positions", "selected_channels", "first_timepoint", "last_timepoint",
            "reference_channel", "validation_mode", "quick_validation_samples_per_position_channel",
        ],
    ),
    (
        "2. Workflow essentials",
        False,
        "beginner",
        ["processing_order", "coarse_crop_during_bgs", "enable_denoising", "dry_run", "resume"],
    ),
    (
        "Naming and automatic discovery",
        True,
        "expert",
        [
            "position_folder_pattern", "settings_pattern", "view1_token", "view2_token", "fused_token",
            "tiff_glob_pattern", "ome_companion_glob_pattern", "timepoint_regex", "channel_regex", "fused_channel_regex", "processed_channel_regex",
        ],
    ),
    (
        "More workflow controls",
        True,
        "expert",
        ["overwrite", "continue_on_error", "fusion_crop_um", "fusion_save_as_crop"],
    ),
    (
        "Advanced registration",
        True,
        "expert",
        [key for key in DEFAULT_CONFIG if key.startswith("registration_")],
    ),
    (
        "Advanced background subtraction",
        True,
        "expert",
        [key for key in DEFAULT_CONFIG if key.startswith("bgs_")],
    ),
    (
        "3. Optional denoising settings",
        True,
        "beginner",
        [
            "denoise_method", "denoise_median_size", "denoise_sigma_xy",
            "denoise_intensity_floor", "denoise_output_dtype", "denoise_compression",
        ],
    ),
    (
        "Advanced denoising",
        True,
        "expert",
        [
            "denoise_glob_pattern", "denoise_sigma_z", "denoise_crop_to_foreground",
            "denoise_crop_padding_xy", "denoise_crop_padding_z",
            "denoise_mask_percentile", "denoise_min_size_voxels",
        ],
    ),
    (
        "Advanced final crop",
        True,
        "expert",
        [key for key in DEFAULT_CONFIG if key.startswith("crop_")],
    ),
]


def _label(key: str) -> str:
    if key in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[key]
    replacements = {"Bgs": "BGS", "Xy": "XY", "Z": "Z", "Tiff": "TIFF", "Dtype": "dtype", "Qc": "QC"}
    label = key.replace("_", " ").title()
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def _field_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _render_field(key: str) -> str:
    default = DEFAULT_CONFIG[key]
    value_type = _field_type(default)
    label = html.escape(_label(key))
    help_detail = SETTING_HELP[key]
    help_text = html.escape(help_detail["summary"])
    choices = CHOICES.get(key)
    attrs = f'data-config-key="{html.escape(key)}" data-value-type="{value_type}" id="field-{html.escape(key)}"'
    if value_type == "bool":
        control = f'<label class="switch-line"><input type="checkbox" {attrs}> <span>Enable</span></label>'
    elif choices:
        options = "".join(f'<option value="{html.escape(str(choice))}">{html.escape(str(choice))}</option>' for choice in choices)
        control = f'<select {attrs}>{options}</select>'
    elif value_type in {"int", "float"}:
        step = "1" if value_type == "int" else "any"
        control = f'<input type="number" step="{step}" {attrs}>'
    else:
        text_input = f'<input type="text" {attrs} autocomplete="off">'
        if key in PATH_FIELDS:
            prompt = "Choose the input dataset folder" if key == "input_dataset_folder" else "Choose the output root folder"
            control = (
                f'<div class="path-control">{text_input}'
                f'<button type="button" class="browse-button" data-target-key="{html.escape(key)}" '
                f'data-picker-prompt="{html.escape(prompt)}">Browse…</button></div>'
            )
        else:
            control = text_input
    recommended = '<span class="recommended-badge">Recommended default</span>' if key in RECOMMENDED_KEYS else ""
    unit = html.escape(help_detail.get("units", ""))
    unit_text = f'<span class="unit-text">{unit}</span>' if unit else ""
    label_html = (
        f'<div class="field-label"><label for="field-{html.escape(key)}">{label}</label>'
        f'<button type="button" class="help-button" data-help-key="{html.escape(key)}" '
        f'aria-label="Explain {label}" title="Explain this setting">?</button></div>'
    )
    helper = f'<div class="help">{help_text}</div><div class="field-meta">{recommended}{unit_text}</div>'
    return f'<div class="field" data-setting-key="{html.escape(key)}">{label_html}<div>{control}{helper}</div></div>'


def _render_groups() -> str:
    rendered: list[str] = []
    for title, collapsible, tier, keys in GROUPS:
        body = "".join(_render_field(key) for key in keys)
        class_name = f'setting-group {tier}-group'
        if collapsible:
            rendered.append(
                f'<details class="{class_name}"><summary>{html.escape(title)}</summary><div class="fields">{body}</div></details>'
            )
        else:
            rendered.append(
                f'<section class="{class_name}"><h2>{html.escape(title)}</h2><div class="fields">{body}</div></section>'
            )
    return "".join(rendered)


class AppState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.config = merged_config()
        self.logs: list[str] = []
        self.max_logs = 10000
        self.runner: PipelineRunner | None = None
        self.worker: threading.Thread | None = None
        self.returncode: int | None = None
        self.status: dict[str, Any] = {
            "running": False,
            "current_stage": "idle",
            "current_position": "",
            "current_timepoint": "",
            "completed_file_count": 0,
            "failed_file_count": 0,
            "message": "Ready.",
        }

    def append_log(self, line: str) -> None:
        with self.lock:
            self.logs.append(str(line))
            if len(self.logs) > self.max_logs:
                del self.logs[: len(self.logs) - self.max_logs]

    def update_event(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.status.update(payload)

    def start(self, config: dict[str, Any], scan: Any | None = None) -> tuple[bool, str]:
        with self.lock:
            if self.worker and self.worker.is_alive():
                return False, "A pipeline run is already active."
            self.config = merged_config(config)
            self.logs.clear()
            self.returncode = None
            self.status = {
                "running": True,
                "current_stage": "starting",
                "current_position": "",
                "current_timepoint": "",
                "completed_file_count": 0,
                "failed_file_count": 0,
                "message": "Pipeline is starting.",
            }
            self.runner = PipelineRunner(
                self.config,
                log_callback=self.append_log,
                event_callback=self.update_event,
                prevalidated_scan=scan,
                skip_initial_validation=scan is not None,
            )
            self.worker = threading.Thread(target=self._run_worker, name="PipelineWebWorker", daemon=True)
            self.worker.start()
            return True, "Pipeline started."

    def _run_worker(self) -> None:
        try:
            assert self.runner is not None
            rc = self.runner.run()
            with self.lock:
                self.returncode = rc
                self.status["running"] = False
                self.status["message"] = "Completed successfully." if rc == 0 else ("Stopped." if rc == 130 else "Failed.")
        except Exception:
            text = traceback.format_exc()
            self.append_log(text)
            with self.lock:
                self.returncode = 1
                self.status.update({"running": False, "current_stage": "failed", "message": "Unexpected web-runner error."})

    def stop(self) -> tuple[bool, str]:
        with self.lock:
            runner = self.runner
            active = bool(self.worker and self.worker.is_alive())
        if not active or runner is None:
            return False, "No pipeline run is active."
        runner.request_stop()
        return True, "Stop requested."

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        with self.lock:
            since = max(0, min(int(since), len(self.logs)))
            return {
                **self.status,
                "returncode": self.returncode,
                "logs": self.logs[since:],
                "next_log_index": len(self.logs),
            }


STATE = AppState()


def _resolve_settings_path(raw: str) -> Path:
    value = str(raw or "pipeline_settings.json").strip()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _existing_picker_folder(raw: str) -> Path:
    """Return an existing directory suitable as a native chooser start point."""
    text = str(raw or "").strip()
    candidate = Path(text).expanduser() if text else Path.home()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    while not candidate.is_dir() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.is_dir() else Path.home()


def _choose_folder_macos(current_path: str, prompt: str) -> tuple[str | None, bool]:
    """Open the macOS folder chooser without importing Tkinter.

    Returns ``(path, cancelled)``. The chooser runs on the same Mac as this
    Python web server, so mounted network volumes are available under /Volumes.
    """
    if sys.platform != "darwin":
        raise RuntimeError(
            "The native Browse button is currently available only on macOS. "
            "You can still paste an absolute folder path."
        )
    osascript = Path("/usr/bin/osascript")
    if not osascript.exists():
        raise RuntimeError(
            "macOS folder chooser is unavailable because /usr/bin/osascript was not found."
        )
    start_dir = _existing_picker_folder(current_path)
    safe_prompt = str(prompt or "Choose a folder").strip()[:200] or "Choose a folder"
    script = (
        "on run argv\n"
        "    set promptText to item 1 of argv\n"
        "    set defaultPath to item 2 of argv\n"
        "    try\n"
        "        set defaultFolder to POSIX file defaultPath as alias\n"
        "        set chosenFolder to choose folder with prompt promptText default location defaultFolder\n"
        "        return POSIX path of chosenFolder\n"
        "    on error number -128\n"
        "        return \"__PIPELINE_PICKER_CANCELLED__\"\n"
        "    end try\n"
        "end run\n"
    )
    completed = subprocess.run(
        [str(osascript), "-e", script, safe_prompt, str(start_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=600,
    )
    output = completed.stdout.strip()
    if output == "__PIPELINE_PICKER_CANCELLED__":
        return None, True
    if completed.returncode != 0:
        detail = (
            completed.stderr.strip()
            or output
            or f"osascript exited with status {completed.returncode}"
        )
        raise RuntimeError(f"Folder chooser failed: {detail}")
    if not output:
        raise RuntimeError("Folder chooser returned no path.")
    chosen = Path(output).expanduser().resolve()
    if not chosen.is_dir():
        raise RuntimeError(f"Selected path is not a directory: {chosen}")
    return str(chosen), False


def _dependency_info() -> dict[str, Any]:
    packages = ["numpy", "scipy", "tifffile", "skimage", "cv2", "matplotlib", "PIL", "imagecodecs"]
    installed = {name: importlib.util.find_spec(name) is not None for name in packages}
    return {
        "installed": installed,
        "imagecodecs_warning": not installed["imagecodecs"],
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "native_folder_picker": sys.platform == "darwin" and Path("/usr/bin/osascript").exists(),
    }


PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Light-sheet full pipeline</title>
<style>
:root {{ color-scheme: light; --accent:#245f9e; --border:#cfd7df; --muted:#5f6b76; --bg:#f4f7fa; --bad:#a51d2d; --good:#166534; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:#17212b; }}
header {{ background:#fff; border-bottom:1px solid var(--border); padding:18px 24px; position:sticky; top:0; z-index:10; }}
header h1 {{ margin:0 0 4px; font-size:22px; }}
header p {{ margin:0; color:var(--muted); }}
main {{ max-width:1500px; margin:0 auto; padding:18px; display:grid; grid-template-columns:minmax(480px, 0.9fr) minmax(520px, 1.1fr); gap:18px; }}
.panel {{ background:#fff; border:1px solid var(--border); border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,.04); overflow:hidden; }}
.panel-body {{ padding:16px; }}
section, details {{ border-bottom:1px solid #e7ebef; padding:12px 0; }}
section:last-child, details:last-child {{ border-bottom:0; }}
h2, summary {{ font-size:16px; font-weight:650; margin:0 0 12px; }}
summary {{ cursor:pointer; }}
.fields {{ display:grid; gap:9px; }}
.field {{ display:grid; grid-template-columns:minmax(190px, 0.8fr) minmax(260px, 1.2fr); gap:12px; align-items:start; }}
.field-label {{ padding-top:5px; display:flex; align-items:flex-start; gap:7px; font-weight:550; }}
.field-label label {{ line-height:1.35; }}
.help-button {{ width:23px; height:23px; min-width:23px; border-radius:50%; padding:0; font-weight:750; color:var(--accent); border-color:#8eb9e6; background:#eef6ff; line-height:21px; }}
.help-button:hover {{ background:#dceeff; }}
.path-control {{ display:flex; gap:8px; align-items:center; }}
.path-control input {{ flex:1 1 auto; min-width:0; }}
.path-control .browse-button {{ flex:0 0 auto; white-space:nowrap; }}
input[type=text], input[type=number], select {{ width:100%; border:1px solid #aeb9c4; border-radius:5px; padding:7px 8px; font:inherit; background:white; }}
input:focus, select:focus {{ outline:2px solid #8eb9e6; border-color:var(--accent); }}
.help {{ color:var(--muted); font-size:12px; margin-top:5px; line-height:1.4; }}
.field-meta {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:5px; align-items:center; }}
.recommended-badge {{ font-size:11px; color:#166534; background:#effaf2; border:1px solid #9fceb0; border-radius:999px; padding:2px 7px; }}
.unit-text {{ font-size:11px; color:#5f6b76; background:#f3f5f7; border-radius:999px; padding:2px 7px; }}
.switch-line {{ display:flex; align-items:center; gap:7px; min-height:34px; }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:8px; padding:12px 16px; border-bottom:1px solid var(--border); background:#fbfcfd; align-items:center; }}
button {{ border:1px solid #8d9aa7; background:#fff; border-radius:5px; padding:8px 12px; font:inherit; cursor:pointer; }}
button.primary {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
button.danger {{ color:var(--bad); border-color:#d8a3aa; }}
button:disabled {{ opacity:.5; cursor:not-allowed; }}
.settings-path {{ flex:1 1 300px; min-width:230px; }}
.notice {{ border:1px solid #e2c56e; background:#fff8dc; color:#6d5100; border-radius:6px; padding:10px; margin-bottom:10px; line-height:1.4; }}
.notice.info {{ border-color:#9bc0e6; background:#eef6ff; color:#204a70; }}
.notice.good {{ border-color:#8dc69d; background:#effaf2; color:var(--good); }}
.status-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-bottom:12px; }}
.status-card {{ border:1px solid var(--border); border-radius:6px; padding:9px; background:#fbfcfd; }}
.status-card b {{ display:block; font-size:12px; color:var(--muted); margin-bottom:4px; }}
.status-card span {{ overflow-wrap:anywhere; }}
#validation {{ white-space:pre-wrap; border:1px solid var(--border); background:#fbfcfd; padding:10px; border-radius:6px; min-height:48px; max-height:220px; overflow:auto; }}
#log {{ background:#101418; color:#e8edf2; padding:12px; border-radius:6px; height:330px; overflow:auto; white-space:pre-wrap; word-break:break-word; font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }}
.table-wrap {{ overflow:auto; max-height:440px; border:1px solid var(--border); border-radius:6px; margin-top:12px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }}
th, td {{ border-bottom:1px solid #dde3e8; border-right:1px solid #edf0f2; padding:6px; text-align:left; vertical-align:top; max-width:280px; overflow-wrap:anywhere; }}
th {{ position:sticky; top:0; background:#eef2f6; z-index:1; }}
.ok {{ color:var(--good); font-weight:650; }}
.bad {{ color:var(--bad); font-weight:650; }}
.warning-box {{ display:none; }}
.mode-bar {{ display:flex; gap:8px; align-items:center; margin-bottom:12px; padding:10px; border:1px solid var(--border); border-radius:8px; background:#fbfcfd; }}
.mode-bar strong {{ margin-right:auto; }}
.mode-button.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
body.beginner-mode .expert-group {{ display:none; }}
.quick-start {{ border:1px solid #9bc0e6; background:#eef6ff; border-radius:8px; padding:12px; margin-bottom:12px; }}
.quick-start h2 {{ margin:0 0 8px; }}
.quick-steps {{ display:grid; grid-template-columns:repeat(4,1fr); gap:7px; }}
.quick-step {{ background:#fff; border:1px solid #c9dced; border-radius:7px; padding:8px; font-size:12px; line-height:1.35; }}
.quick-step b {{ display:block; color:var(--accent); margin-bottom:2px; }}
.pipeline-summary {{ border:1px solid var(--border); border-radius:8px; padding:10px; background:#fbfcfd; margin-bottom:12px; line-height:1.45; }}
.pipeline-summary h3 {{ margin:0 0 6px; font-size:14px; }}
.summary-row {{ display:flex; gap:8px; margin-top:4px; }}
.summary-row b {{ min-width:130px; }}
.safe-text {{ color:var(--good); font-weight:650; }}
.caution-text {{ color:#8a5a00; font-weight:650; }}
.modal-backdrop {{ position:fixed; inset:0; background:rgba(12,20,28,.48); display:flex; align-items:center; justify-content:center; z-index:100; padding:18px; }}
.modal-backdrop[hidden] {{ display:none; }}
.help-modal {{ width:min(680px,100%); max-height:90vh; overflow:auto; background:#fff; border-radius:12px; box-shadow:0 16px 60px rgba(0,0,0,.3); }}
.help-modal-header {{ display:flex; align-items:center; gap:10px; padding:16px 18px; border-bottom:1px solid var(--border); }}
.help-modal-header h2 {{ margin:0; flex:1; }}
.help-modal-body {{ padding:18px; }}
.help-detail {{ margin-bottom:14px; }}
.help-detail b {{ display:block; color:#334155; margin-bottom:3px; }}
.help-current {{ padding:9px; border-radius:6px; background:#f3f6f8; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; overflow-wrap:anywhere; }}
.help-caution {{ border-left:4px solid #d79b25; background:#fff8dc; padding:9px 10px; }}
.field.denoise-disabled {{ opacity:0.48; }}
.field.denoise-disabled .help-button {{ opacity:1; }}
.field.denoise-disabled .help {{ font-style:italic; }}
@media (max-width:1050px) {{ main {{ grid-template-columns:1fr; }} .quick-steps {{ grid-template-columns:1fr 1fr; }} }}
@media (max-width:650px) {{ .field {{ grid-template-columns:1fr; gap:2px; }} .status-grid {{ grid-template-columns:1fr 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>Light-sheet end-to-end pipeline</h1>
  <p>Local browser interface. Data stays on the machine running Python and is not uploaded.</p>
</header>
<main>
  <div class="panel">
    <div class="toolbar">
      <input class="settings-path" id="settings-path" type="text" value="pipeline_settings.json" aria-label="Settings JSON path">
      <button id="save-button">Save settings</button>
      <button id="load-button">Load settings</button>
    </div>
    <div class="panel-body">
      <div class="quick-start">
        <h2>Quick start</h2>
        <div class="quick-steps">
          <div class="quick-step"><b>1. Choose folders</b>Select the experiment parent folder and a separate output folder.</div>
          <div class="quick-step"><b>2. Select data</b>Start with one position, one channel, and 2–3 timepoints.</div>
          <div class="quick-step"><b>3. Check setup</b>Scan filenames, then validate. Quick mode is suitable for routine full datasets.</div>
          <div class="quick-step"><b>4. Process</b>Use Dry run first, inspect commands, then start the real run.</div>
        </div>
      </div>
      <div class="mode-bar">
        <strong>Interface detail</strong>
        <button type="button" id="beginner-mode-button" class="mode-button active">Beginner</button>
        <button type="button" id="expert-mode-button" class="mode-button">Expert</button>
      </div>
      <div class="notice good">Recommended baseline: Background subtraction → Crop, with both Coarse BGS crop and Optional denoising <b>disabled</b>. Enable denoising only after comparing a representative small subset.</div>
      <div id="dependency-warning" class="notice warning-box"></div>
      <div id="starting-data-warning" class="notice info warning-box"></div>
      <div id="order-warning" class="notice warning-box"></div>
      <div id="coarse-warning" class="notice info warning-box"></div>
      <div id="denoise-warning" class="notice info warning-box"></div>
      {_render_groups()}
    </div>
  </div>
  <div class="panel">
    <div class="toolbar">
      <button id="scan-button">Scan dataset</button>
      <button id="validate-button">Validate configuration</button>
      <button class="primary" id="start-button">Start</button>
      <button class="danger" id="stop-button">Request stop</button>
      <button id="no-compression-button">Set output compression to none</button>
    </div>
    <div class="panel-body">
      <div id="pipeline-summary" class="pipeline-summary"></div>
      <div class="status-grid">
        <div class="status-card"><b>Current stage</b><span id="current-stage">idle</span></div>
        <div class="status-card"><b>Current position</b><span id="current-position">—</span></div>
        <div class="status-card"><b>Current timepoint</b><span id="current-timepoint">—</span></div>
        <div class="status-card"><b>Completed files</b><span id="completed-count">0</span></div>
        <div class="status-card"><b>Failed files</b><span id="failed-count">0</span></div>
      </div>
      <h2>Validation</h2>
      <div id="validation">Not yet validated.</div>
      <div class="table-wrap"><table id="discovery-table"><thead></thead><tbody></tbody></table></div>
      <h2 style="margin-top:16px">Live output log</h2>
      <pre id="log"></pre>
    </div>
  </div>
</main>

<div id="help-backdrop" class="modal-backdrop" hidden role="dialog" aria-modal="true" aria-labelledby="help-title">
  <div class="help-modal">
    <div class="help-modal-header">
      <h2 id="help-title">Setting help</h2>
      <button type="button" id="help-close-button" aria-label="Close help">Close</button>
    </div>
    <div id="help-modal-body" class="help-modal-body"></div>
  </div>
</div>
<script>
const defaults = {json.dumps(DEFAULT_CONFIG, ensure_ascii=False)};
const settingHelp = {json.dumps(SETTING_HELP, ensure_ascii=False)};
let logIndex = 0;
let isRunning = false;

function qs(id) {{ return document.getElementById(id); }}
function escapeHtml(value) {{
  return String(value ?? '').replace(/[&<>'"]/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}}[ch]));
}}
function setConfig(config) {{
  document.querySelectorAll('[data-config-key]').forEach(el => {{
    const key = el.dataset.configKey;
    const value = Object.prototype.hasOwnProperty.call(config, key) ? config[key] : defaults[key];
    if (el.dataset.valueType === 'bool') el.checked = Boolean(value);
    else el.value = value === null || value === undefined ? '' : String(value);
  }});
  updateWarnings();
}}
function getConfig() {{
  const config = {{}};
  document.querySelectorAll('[data-config-key]').forEach(el => {{
    const key = el.dataset.configKey;
    const type = el.dataset.valueType;
    if (type === 'bool') config[key] = el.checked;
    else if (type === 'int') config[key] = Number.parseInt(el.value || '0', 10);
    else if (type === 'float') config[key] = Number.parseFloat(el.value || '0');
    else config[key] = el.value;
  }});
  return config;
}}
async function postJson(url, body) {{
  const response = await fetch(url, {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(body)}});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${{response.status}}`);
  return payload;
}}
async function chooseFolder(key, prompt) {{
  const field = qs(`field-${{key}}`);
  const button = document.querySelector(`.browse-button[data-target-key="${{key}}"]`);
  if (!field) return;
  if (button) button.disabled = true;
  showValidation('Opening the macOS folder chooser…');
  try {{
    const result = await postJson('/api/choose-folder', {{
      current_path: field.value,
      prompt: prompt || 'Choose a folder'
    }});
    if (!result.cancelled && result.path) {{
      field.value = result.path;
      showValidation(`Selected folder: ${{result.path}}`, true);
    }} else {{
      showValidation('Folder selection was cancelled.', true);
    }}
  }} catch (err) {{
    showValidation(err.message);
  }} finally {{
    if (button) button.disabled = false;
  }}
}}
function showValidation(lines, good=false) {{
  const el = qs('validation');
  el.textContent = Array.isArray(lines) ? lines.join('\\n') : String(lines);
  el.style.color = good ? '#166534' : '#a51d2d';
}}
function renderScan(scan) {{
  const genericMode = (scan.rows || []).some(row => Boolean(row.input_path));
  const fusedMode = (scan.rows || []).some(row => Boolean(row.fused_path));
  const headers = genericMode
    ? ['Position','Timepoint','Channel','Input TIFF path','OME path','Status']
    : (fusedMode
      ? ['Position','Settings','Timepoint','Channel','Fused TIFF path','OME path','Status']
      : ['Position','Settings','Timepoint','Channel','View 1 path','View 2 path','OME path','Pairing status']);
  const keys = genericMode
    ? ['position','timepoint','channel','input_path','ome_path','pairing_status']
    : (fusedMode
      ? ['position','settings','timepoint','channel','fused_path','ome_path','pairing_status']
      : ['position','settings','timepoint','channel','view1_path','view2_path','ome_path','pairing_status']);
  qs('discovery-table').querySelector('thead').innerHTML = '<tr>' + headers.map(h => `<th>${{escapeHtml(h)}}</th>`).join('') + '</tr>';
  qs('discovery-table').querySelector('tbody').innerHTML = (scan.rows || []).map(row => '<tr>' + keys.map(key => {{
    const cls = key === 'pairing_status' ? (row[key] === 'ok' ? 'ok' : 'bad') : '';
    return `<td class="${{cls}}">${{escapeHtml(row[key] || '')}}</td>`;
  }}).join('') + '</tr>').join('');
}}
function formatValue(value) {{
  if (typeof value === 'boolean') return value ? 'Enabled' : 'Disabled';
  if (value === '' || value === null || value === undefined) return '(blank)';
  return String(value);
}}
function openSettingHelp(key) {{
  const detail = settingHelp[key];
  if (!detail) return;
  const field = qs(`field-${{key}}`);
  const current = field ? (field.dataset.valueType === 'bool' ? field.checked : field.value) : '';
  qs('help-title').textContent = document.querySelector(`[data-setting-key="${{key}}"] label`)?.textContent || key;
  const sections = [
    ['What it does', detail.summary],
    ['What changes when you edit it', detail.effect],
    ['Recommended use', detail.recommended],
    ['Units / value type', detail.units || 'Not specified'],
  ];
  const body = qs('help-modal-body');
  body.innerHTML = '';
  sections.forEach(([title, value]) => {{
    const div = document.createElement('div');
    div.className = 'help-detail';
    const b = document.createElement('b'); b.textContent = title;
    const p = document.createElement('div'); p.textContent = value || '';
    div.appendChild(b); div.appendChild(p); body.appendChild(div);
  }});
  const currentBox = document.createElement('div');
  currentBox.className = 'help-detail';
  const currentTitle = document.createElement('b'); currentTitle.textContent = 'Current and default values';
  const currentValue = document.createElement('div'); currentValue.className = 'help-current';
  currentValue.textContent = `Current: ${{formatValue(current)}}  |  Package default: ${{formatValue(defaults[key])}}`;
  currentBox.appendChild(currentTitle); currentBox.appendChild(currentValue); body.appendChild(currentBox);
  if (detail.caution) {{
    const caution = document.createElement('div'); caution.className = 'help-caution';
    caution.textContent = `Caution: ${{detail.caution}}`; body.appendChild(caution);
  }}
  qs('help-backdrop').hidden = false;
  qs('help-close-button').focus();
}}
function closeSettingHelp() {{ qs('help-backdrop').hidden = true; }}
function setInterfaceMode(mode) {{
  const beginner = mode !== 'expert';
  document.body.classList.toggle('beginner-mode', beginner);
  qs('beginner-mode-button').classList.toggle('active', beginner);
  qs('expert-mode-button').classList.toggle('active', !beginner);
  if (!beginner) document.querySelectorAll('.expert-group').forEach(group => {{ if (group.tagName === 'DETAILS') group.open = false; }});
  window.localStorage.setItem('lightsheet-interface-mode', beginner ? 'beginner' : 'expert');
}}
function stageLabel(stage) {{
  return {{
    fusion: 'Fusion',
    registration: 'Registration',
    background_subtraction: 'Background subtraction',
    denoise: 'Denoising',
    crop: 'Crop'
  }}[stage] || stage;
}}
function updatePipelineSummary() {{
  const cfg = getConfig();
  const single = cfg.run_mode === 'Single stage';
  let sequence;
  let starting;
  if (single) {{
    sequence = `${{stageLabel(cfg.single_stage)}} only`;
    starting = cfg.single_stage === 'fusion'
      ? 'Raw View1/View2 camera pairs'
      : `Stage-ready TIFF stacks for ${{stageLabel(cfg.single_stage)}}`;
  }} else {{
    const bgsFirst = cfg.processing_order === 'Background subtraction → Crop';
    const denoiseStep = cfg.enable_denoising ? ' → Denoise' : '';
    const prefix = cfg.starting_data_type === 'Already fused TIFFs' ? 'Imported fused TIFFs → Registration' : 'Fusion → Registration';
    sequence = bgsFirst
      ? `${{prefix}} → Background subtraction${{denoiseStep}} → Final crop`
      : `${{prefix}} → Crop → Background subtraction${{denoiseStep}}`;
    starting = cfg.starting_data_type;
  }}
  const runsBgs = !single || cfg.single_stage === 'background_subtraction';
  const coarse = !runsBgs
    ? '<span>Not used in this run</span>'
    : (cfg.coarse_crop_during_bgs
      ? '<span class="caution-text">Enabled — faster/smaller, but validate weak edge signal</span>'
      : '<span class="safe-text">Disabled — safer recommended default</span>');
  const runsDenoise = single ? cfg.single_stage === 'denoise' : cfg.enable_denoising;
  const denoise = runsDenoise
    ? `<span class="caution-text">Enabled — ${{escapeHtml(cfg.denoise_method)}}; compare fine and weak structures</span>`
    : '<span class="safe-text">Not run</span>';
  const dtypePreserved = single
    ? (cfg.single_stage === 'registration' ? cfg.registration_save_dtype === 'input'
      : cfg.single_stage === 'background_subtraction' ? cfg.bgs_output_dtype === 'same'
      : cfg.single_stage === 'denoise' ? cfg.denoise_output_dtype === 'same'
      : true)
    : (cfg.registration_save_dtype === 'input'
      && cfg.bgs_output_dtype === 'same'
      && (!cfg.enable_denoising || cfg.denoise_output_dtype === 'same'));
  const dtype = dtypePreserved
    ? '<span class="safe-text">Preserved</span>'
    : '<span class="caution-text">Explicit conversion selected</span>';
  qs('pipeline-summary').innerHTML = `
    <h3>Current run summary</h3>
    <div class="summary-row"><b>Run mode</b><span>${{escapeHtml(cfg.run_mode)}}</span></div>
    <div class="summary-row"><b>Starting data</b><span>${{escapeHtml(starting)}}</span></div>
    <div class="summary-row"><b>Stages executed</b><span>${{escapeHtml(sequence)}}</span></div>
    <div class="summary-row"><b>Coarse BGS crop</b><span>${{coarse}}</span></div>
    <div class="summary-row"><b>Denoising</b><span>${{denoise}}</span></div>
    <div class="summary-row"><b>Validation</b><span>${{escapeHtml(cfg.validation_mode)}} (${{escapeHtml(cfg.quick_validation_samples_per_position_channel)}} sample pair(s) in Quick mode)</span></div>
    <div class="summary-row"><b>Output dtype</b><span>${{dtype}}</span></div>`;
}}
function setSettingEnabled(key, enabled) {{
  const control = qs(`field-${{key}}`);
  const wrapper = document.querySelector(`[data-setting-key="${{key}}"]`);
  if (control) control.disabled = !enabled;
  if (wrapper) wrapper.classList.toggle('denoise-disabled', !enabled);
}}
function updateDenoiseControls() {{
  const single = qs('field-run_mode').value === 'Single stage';
  const enabled = single
    ? qs('field-single_stage').value === 'denoise'
    : qs('field-enable_denoising').checked;
  document.querySelectorAll('[data-setting-key^="denoise_"]').forEach(wrapper => {{
    const control = wrapper.querySelector('[data-config-key]');
    if (control) control.disabled = !enabled;
    wrapper.classList.toggle('denoise-disabled', !enabled);
  }});
}}
function updateRunControls() {{
  const single = qs('field-run_mode').value === 'Single stage';
  const stage = qs('field-single_stage').value;
  const fused = qs('field-starting_data_type').value === 'Already fused TIFFs';
  setSettingEnabled('single_stage', single);
  setSettingEnabled('starting_data_type', !single);
  setSettingEnabled('processing_order', !single);
  setSettingEnabled('enable_denoising', !single);

  const needsFusion = single ? stage === 'fusion' : !fused;
  setSettingEnabled('reference_channel', needsFusion);

  const inputLabel = document.querySelector('[data-setting-key="input_dataset_folder"] .field-label label');
  const inputHelp = document.querySelector('[data-setting-key="input_dataset_folder"] .help');
  let label = 'Input folder';
  let help = 'Choose the input folder.';
  if (single) {{
    const labels = {{
      fusion: 'Raw View1/View2 input folder',
      registration: 'Fused TIFF input folder',
      background_subtraction: 'Registered or stage-ready TIFF input folder',
      denoise: 'Background-subtracted or stage-ready TIFF input folder',
      crop: 'Stage-ready TIFF input folder'
    }};
    label = labels[stage] || 'Stage-ready TIFF input folder';
    help = stage === 'fusion'
      ? 'Choose the parent experiment folder containing raw View1/View2 acquisition folders.'
      : 'Choose one TIFF folder for one organoid, or a parent containing one TIFF folder per organoid. No previous pipeline manifest is required.';
  }} else if (fused) {{
    label = 'Fused TIFF input folder';
    help = 'Choose a folder containing t0001_561_Fused.tif-style files and one OME companion, or a parent containing one such folder per position.';
  }} else {{
    label = 'Raw dataset input folder';
    help = 'Choose the parent experiment folder containing View1/View2 acquisition folders.';
  }}
  if (inputLabel) inputLabel.textContent = label;
  if (inputHelp) inputHelp.textContent = help;

  const sourceBox = qs('starting-data-warning');
  if (single) {{
    sourceBox.style.display = 'block';
    sourceBox.textContent = `Single stage selected: only ${{stageLabel(stage)}} will run. Prerequisite stages and their manifests are not required. The input folder may contain one organoid directly or many organoid/position folders.`;
  }} else if (fused) {{
    sourceBox.style.display = 'block';
    sourceBox.textContent = 'Already fused input selected. Fusion will not run and no previous fusion manifest is required. Registration reads the fused TIFFs directly from this folder.';
  }} else {{
    sourceBox.style.display = 'none';
    sourceBox.textContent = '';
  }}
  updateDenoiseControls();
}}

function updateWarnings() {{
  updateRunControls();
  const single = qs('field-run_mode').value === 'Single stage';
  const stage = qs('field-single_stage').value;
  const order = qs('field-processing_order').value;
  const coarse = qs('field-coarse_crop_during_bgs').checked;
  const orderBox = qs('order-warning');
  orderBox.style.display = !single && order === {json.dumps(ORDER_CROP_BGS)} ? 'block' : 'none';
  orderBox.textContent = 'Warning: cropping before background subtraction may reduce processing time, but gives the background estimator less surrounding background and can make crop detection more sensitive to raw haze.';
  const runsBgs = !single || stage === 'background_subtraction';
  const coarseBox = qs('coarse-warning');
  coarseBox.style.display = runsBgs && coarse ? 'block' : 'none';
  coarseBox.textContent = single
    ? 'Coarse BGS crop is enabled for this background-subtraction-only run.'
    : 'Coarse BGS crop is enabled. Background subtraction will use --crop-to-foreground --crop-padding-xy 80 --crop-padding-z 8, followed by the final tighter object-aware crop.';
  const denoiseEnabled = single ? stage === 'denoise' : qs('field-enable_denoising').checked;
  const denoiseCrop = qs('field-denoise_crop_to_foreground').checked;
  const denoiseBox = qs('denoise-warning');
  denoiseBox.style.display = denoiseEnabled ? 'block' : 'none';
  denoiseBox.textContent = denoiseCrop
    ? 'Denoising includes an extra foreground crop. This changes intensities and dimensions; compare weak peripheral signal carefully.'
    : 'Denoising changes voxel intensities. Compare a small representative subset before a full dataset run.';
  updatePipelineSummary();
}}
async function scanDataset() {{
  showValidation('Scanning folder names and pairing files…');
  try {{
    const result = await postJson('/api/scan', {{config:getConfig()}});
    renderScan(result.scan);
    showValidation(result.errors.length ? result.errors.map(x => '• ' + x) : `Configuration is valid. Detected positions: ${{result.scan.positions.join(', ') || 'none'}}; channels: ${{result.scan.channels.join(', ') || 'none'}}.`, result.errors.length === 0);
  }} catch (err) {{ showValidation(err.message); }}
}}
async function validateConfig() {{
  showValidation(`Validating in ${{qs('field-validation_mode').value}} mode…`);
  try {{
    const result = await postJson('/api/validate', {{config:getConfig()}});
    renderScan(result.scan);
    showValidation(result.errors.length ? result.errors.map(x => '• ' + x) : 'Configuration is valid.', result.errors.length === 0);
  }} catch (err) {{ showValidation(err.message); }}
}}
async function startPipeline() {{
  showValidation(`Starting: scanning and ${{qs('field-validation_mode').value}}-validating once…`);
  try {{
    const result = await postJson('/api/start', {{config:getConfig()}});
    logIndex = 0;
    qs('log').textContent = result.message + '\\n';
  }} catch (err) {{ showValidation(err.message); }}
}}
async function stopPipeline() {{
  try {{ const result = await postJson('/api/stop', {{}}); showValidation(result.message, result.ok); }}
  catch (err) {{ showValidation(err.message); }}
}}
async function saveSettings() {{
  try {{
    const result = await postJson('/api/save-settings', {{path:qs('settings-path').value, config:getConfig()}});
    showValidation(result.message, true);
  }} catch (err) {{ showValidation(err.message); }}
}}
async function loadSettings() {{
  try {{
    const result = await postJson('/api/load-settings', {{path:qs('settings-path').value}});
    setConfig(result.config); showValidation(result.message, true);
  }} catch (err) {{ showValidation(err.message); }}
}}
async function pollStatus() {{
  try {{
    const response = await fetch(`/api/status?since=${{logIndex}}`);
    const result = await response.json();
    isRunning = Boolean(result.running);
    qs('current-stage').textContent = result.current_stage || 'idle';
    qs('current-position').textContent = result.current_position || '—';
    qs('current-timepoint').textContent = result.current_timepoint ? `t${{String(result.current_timepoint).padStart(4, '0')}}` : '—';
    qs('completed-count').textContent = result.completed_file_count ?? 0;
    qs('failed-count').textContent = result.failed_file_count ?? 0;
    qs('start-button').disabled = isRunning;
    qs('stop-button').disabled = !isRunning;
    if (result.logs && result.logs.length) {{
      const log = qs('log');
      log.textContent += result.logs.join('\\n') + '\\n';
      log.scrollTop = log.scrollHeight;
    }}
    logIndex = result.next_log_index || logIndex;
  }} catch (_) {{}}
  window.setTimeout(pollStatus, 800);
}}
async function loadDefaultsAndDependencies() {{
  setConfig(defaults);
  try {{
    const response = await fetch('/api/info');
    const info = await response.json();
    if (info.imagecodecs_warning) {{
      const box = qs('dependency-warning');
      box.style.display = 'block';
      box.textContent = `imagecodecs is not installed for ${{info.executable}}. Compressed TIFF reading/writing may fail. For a no-download run, use the button “Set output compression to none” and first verify that one source TIFF can be read.`;
    }}
    if (!info.native_folder_picker) {{
      document.querySelectorAll('.browse-button').forEach(button => {{
        button.disabled = true;
        button.title = 'Native Browse is available when this web server runs on macOS. Paste an absolute path on other systems.';
      }});
    }}
  }} catch (_) {{}}
}}
document.querySelectorAll('.help-button').forEach(button => {{
  button.addEventListener('click', () => openSettingHelp(button.dataset.helpKey));
}});
qs('help-close-button').addEventListener('click', closeSettingHelp);
qs('help-backdrop').addEventListener('click', event => {{ if (event.target === qs('help-backdrop')) closeSettingHelp(); }});
document.addEventListener('keydown', event => {{ if (event.key === 'Escape' && !qs('help-backdrop').hidden) closeSettingHelp(); }});
qs('beginner-mode-button').addEventListener('click', () => setInterfaceMode('beginner'));
qs('expert-mode-button').addEventListener('click', () => setInterfaceMode('expert'));
document.querySelectorAll('[data-config-key]').forEach(field => {{
  field.addEventListener('change', updatePipelineSummary);
  field.addEventListener('input', updatePipelineSummary);
}});
qs('scan-button').addEventListener('click', scanDataset);
qs('validate-button').addEventListener('click', validateConfig);
qs('start-button').addEventListener('click', startPipeline);
qs('stop-button').addEventListener('click', stopPipeline);
qs('save-button').addEventListener('click', saveSettings);
qs('load-button').addEventListener('click', loadSettings);
document.querySelectorAll('.browse-button').forEach(button => {{
  button.addEventListener('click', () => chooseFolder(button.dataset.targetKey, button.dataset.pickerPrompt));
}});
qs('no-compression-button').addEventListener('click', () => {{
  qs('field-registration_compression').value = 'none';
  qs('field-bgs_compression').value = 'none';
  qs('field-denoise_compression').value = 'none';
  qs('field-crop_compression').value = 'none';
  showValidation('Registration, BGS, denoising, and crop output compression were set to none. This does not change dtype, but output files will be larger.', true);
}});
qs('field-run_mode').addEventListener('change', updateWarnings);
qs('field-single_stage').addEventListener('change', updateWarnings);
qs('field-starting_data_type').addEventListener('change', updateWarnings);
qs('field-processing_order').addEventListener('change', updateWarnings);
qs('field-coarse_crop_during_bgs').addEventListener('change', updateWarnings);
qs('field-enable_denoising').addEventListener('change', updateWarnings);
qs('field-denoise_crop_to_foreground').addEventListener('change', updateWarnings);
loadDefaultsAndDependencies();
setInterfaceMode(window.localStorage.getItem('lightsheet-interface-mode') || 'beginner');
updateWarnings();
updatePipelineSummary();
pollStatus();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "LightSheetLocalGUI/1.0"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send_bytes(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"), "application/json; charset=utf-8", status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length > 10_000_000:
            raise ValueError("Request is too large.")
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON request body must be an object.")
        return value

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                since = int(query.get("since", ["0"])[0])
            except ValueError:
                since = 0
            self._json(STATE.snapshot(since))
            return
        if parsed.path == "/api/info":
            self._json(_dependency_info())
            return
        self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._read_json()
            if self.path == "/api/choose-folder":
                path, cancelled = _choose_folder_macos(
                    str(body.get("current_path", "")),
                    str(body.get("prompt", "Choose a folder")),
                )
                self._json({"path": path, "cancelled": cancelled})
                return
            if self.path == "/api/scan":
                config = merged_config(body.get("config") or {})
                scan = scan_dataset(config)
                errors = validate_configuration(config, scan, validation_mode="names_only")
                self._json({"scan": scan.to_dict(), "errors": errors})
                return
            if self.path == "/api/validate":
                config = merged_config(body.get("config") or {})
                scan = scan_dataset(config)
                errors = validate_configuration(config, scan)
                self._json({"scan": scan.to_dict(), "errors": errors})
                return
            if self.path == "/api/start":
                config = merged_config(body.get("config") or {})
                scan = scan_dataset(config)
                errors = validate_configuration(config, scan)
                if errors:
                    self._json({"error": "Configuration validation failed. Use Scan dataset or Validate configuration to review the details.", "errors": errors}, HTTPStatus.BAD_REQUEST)
                    return
                ok, message = STATE.start(config, scan)
                self._json({"ok": ok, "message": message}, HTTPStatus.OK if ok else HTTPStatus.CONFLICT)
                return
            if self.path == "/api/stop":
                ok, message = STATE.stop()
                self._json({"ok": ok, "message": message}, HTTPStatus.OK if ok else HTTPStatus.CONFLICT)
                return
            if self.path == "/api/save-settings":
                path = _resolve_settings_path(str(body.get("path", "")))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(merged_config(body.get("config") or {}), indent=2, ensure_ascii=False), encoding="utf-8")
                self._json({"message": f"Saved settings to {path}"})
                return
            if self.path == "/api/load-settings":
                path = _resolve_settings_path(str(body.get("path", "")))
                config = merged_config(json.loads(path.read_text(encoding="utf-8")))
                self._json({"config": config, "message": f"Loaded settings from {path}"})
                return
            self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
        except FileNotFoundError as exc:
            self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            self._json({"error": traceback.format_exc()}, HTTPStatus.INTERNAL_SERVER_ERROR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=HOST, help="Bind address. Keep 127.0.0.1 unless local-network access is deliberately required.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the default browser automatically.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{server.server_port}/"
    print("Light-sheet local browser GUI")
    print("Python:", sys.executable)
    print("Open:", url)
    print("Data stays on the machine running this Python process; keep this Terminal window open.")
    print("Press Control+C to close the GUI server.")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.4)
    except KeyboardInterrupt:
        print("\nStopping local GUI server.")
    finally:
        if STATE.runner is not None and STATE.snapshot().get("running"):
            STATE.runner.request_stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
