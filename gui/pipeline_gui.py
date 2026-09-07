#!/usr/bin/env python3
"""Dedicated Tkinter GUI for the full light-sheet processing pipeline."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import signal
import sys
import tempfile
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(1, str(ROOT))
sys.path.insert(2, str(SCRIPT_DIR))

from pipeline_runner import (  # noqa: E402
    DEFAULT_CONFIG,
    ORDER_BGS_CROP,
    ORDER_CROP_BGS,
    discovery_table_text,
    scan_dataset,
    validate_configuration,
)


class PipelineGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Light-sheet end-to-end image-processing pipeline")
        self.root.geometry("1450x930")
        self.root.configure(background="white")
        self.vars: dict[str, tk.Variable] = {}
        self.widgets: dict[str, tk.Widget] = {}
        self.scan_result = None
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.pipeline_process: subprocess.Popen[str] | None = None
        self.pipeline_reader: threading.Thread | None = None
        self.temp_config_path: str | None = None
        self._build()
        self._load_values(DEFAULT_CONFIG)
        self._poll_queue()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        title = tk.Label(self.root, text="Configurable light-sheet processing pipeline", font=("TkDefaultFont", 15, "bold"), bg="white")
        title.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 5))

        paned = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        paned.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        upper = tk.Frame(paned, bg="white")
        lower = tk.Frame(paned, bg="white")
        paned.add(upper, weight=3)
        paned.add(lower, weight=2)
        upper.columnconfigure(0, weight=1)
        upper.rowconfigure(0, weight=1)
        lower.columnconfigure(0, weight=1)
        lower.rowconfigure(1, weight=1)

        notebook = ttk.Notebook(upper)
        notebook.grid(row=0, column=0, sticky="nsew")
        self.dataset_tab = self._scrollable_tab(notebook, "Dataset and naming")
        self.workflow_tab = self._scrollable_tab(notebook, "Workflow")
        self.registration_tab = self._scrollable_tab(notebook, "Advanced registration")
        self.bgs_tab = self._scrollable_tab(notebook, "Advanced background subtraction")
        self.crop_tab = self._scrollable_tab(notebook, "Advanced crop")
        self._build_dataset_tab()
        self._build_workflow_tab()
        self._build_registration_tab()
        self._build_bgs_tab()
        self._build_crop_tab()

        status_frame = tk.Frame(lower, bg="white")
        status_frame.grid(row=0, column=0, sticky="ew", pady=(5, 2))
        for col in range(8):
            status_frame.columnconfigure(col, weight=1 if col in {1, 3, 5, 7} else 0)
        tk.Label(status_frame, text="Current stage:", bg="white").grid(row=0, column=0, sticky="w")
        self.current_stage = tk.StringVar(value="Idle")
        tk.Label(status_frame, textvariable=self.current_stage, bg="white", fg="blue").grid(row=0, column=1, sticky="w")
        tk.Label(status_frame, text="Current position:", bg="white").grid(row=0, column=2, sticky="w")
        self.current_position = tk.StringVar(value="")
        tk.Label(status_frame, textvariable=self.current_position, bg="white", fg="blue").grid(row=0, column=3, sticky="w")
        tk.Label(status_frame, text="Completed files:", bg="white").grid(row=0, column=4, sticky="w")
        self.completed_count = tk.StringVar(value="0")
        tk.Label(status_frame, textvariable=self.completed_count, bg="white", fg="green").grid(row=0, column=5, sticky="w")
        tk.Label(status_frame, text="Failed files:", bg="white").grid(row=0, column=6, sticky="w")
        self.failed_count = tk.StringVar(value="0")
        tk.Label(status_frame, textvariable=self.failed_count, bg="white", fg="red").grid(row=0, column=7, sticky="w")

        output_notebook = ttk.Notebook(lower)
        output_notebook.grid(row=1, column=0, sticky="nsew")
        discovery_frame = tk.Frame(output_notebook, bg="white")
        log_frame = tk.Frame(output_notebook, bg="white")
        output_notebook.add(discovery_frame, text="Dataset discovery")
        output_notebook.add(log_frame, text="Live output log")
        discovery_frame.columnconfigure(0, weight=1)
        discovery_frame.rowconfigure(0, weight=1)
        columns = ("position", "settings", "timepoint", "channel", "view1", "view2", "ome", "status")
        self.tree = ttk.Treeview(discovery_frame, columns=columns, show="headings")
        headings = ["Position", "Settings", "Timepoint", "Channel", "View 1 path", "View 2 path", "OME path", "Pairing status"]
        widths = [120, 120, 85, 100, 260, 260, 220, 180]
        for col, heading, width in zip(columns, headings, widths):
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, minwidth=60, stretch=col in {"view1", "view2", "ome", "status"})
        ybar = ttk.Scrollbar(discovery_frame, orient="vertical", command=self.tree.yview)
        xbar = ttk.Scrollbar(discovery_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")

        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, bg="black", fg="white", insertbackground="white", wrap="none")
        log_y = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        log_x = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=log_y.set, xscrollcommand=log_x.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        log_y.grid(row=0, column=1, sticky="ns")
        log_x.grid(row=1, column=0, sticky="ew")

        buttons = tk.Frame(self.root, bg="white")
        buttons.grid(row=2, column=0, sticky="ew", padx=10, pady=(3, 10))
        labels_commands = [
            ("Scan dataset", self.scan), ("Validate configuration", self.validate), ("Start", self.start),
            ("Request stop", self.request_stop), ("Save settings", self.save_settings), ("Load settings", self.load_settings),
        ]
        for i, (text, command) in enumerate(labels_commands):
            button = tk.Button(buttons, text=text, command=command, bg="white", relief="solid", borderwidth=1, padx=9, pady=5)
            button.grid(row=0, column=i, padx=3, sticky="w")
            self.widgets[f"button_{text}"] = button
        buttons.columnconfigure(len(labels_commands), weight=1)
        self.validation_text = tk.StringVar(value="Not yet validated.")
        self.validation_label = tk.Label(buttons, textvariable=self.validation_text, bg="white", fg="red", justify="left", wraplength=600)
        self.validation_label.grid(row=0, column=len(labels_commands), sticky="e")

    def _scrollable_tab(self, notebook: ttk.Notebook, title: str) -> tk.Frame:
        outer = tk.Frame(notebook, bg="white")
        canvas = tk.Canvas(outer, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg="white")
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        notebook.add(outer, text=title)
        inner.columnconfigure(1, weight=1)
        return inner

    def _var_for(self, key: str) -> tk.Variable:
        default = DEFAULT_CONFIG[key]
        if isinstance(default, bool):
            var: tk.Variable = tk.BooleanVar()
        else:
            var = tk.StringVar()
        self.vars[key] = var
        return var

    def _field(self, parent: tk.Frame, row: int, key: str, label: str, kind: str = "entry", choices: list[str] | None = None, help_text: str = "") -> int:
        tk.Label(parent, text=label + ":", bg="white", anchor="w").grid(row=row, column=0, sticky="nw", padx=(10, 6), pady=3)
        var = self.vars.get(key) or self._var_for(key)
        if kind == "folder":
            frame = tk.Frame(parent, bg="white")
            frame.grid(row=row, column=1, sticky="ew", padx=(0, 10), pady=3)
            frame.columnconfigure(0, weight=1)
            widget = tk.Entry(frame, textvariable=var)
            widget.grid(row=0, column=0, sticky="ew")
            tk.Button(frame, text="Browse…", command=lambda k=key: self._browse_folder(k), bg="white").grid(row=0, column=1, padx=(4, 0))
        elif kind == "checkbox":
            widget = tk.Checkbutton(parent, text="Enable", variable=var, bg="white", selectcolor="white", command=self._update_warnings)
            widget.grid(row=row, column=1, sticky="w", padx=(0, 10), pady=3)
        elif kind == "combobox":
            widget = ttk.Combobox(parent, textvariable=var, values=choices or [], state="readonly")
            widget.grid(row=row, column=1, sticky="ew", padx=(0, 10), pady=3)
            widget.bind("<<ComboboxSelected>>", lambda _e: self._update_warnings())
        else:
            widget = tk.Entry(parent, textvariable=var)
            widget.grid(row=row, column=1, sticky="ew", padx=(0, 10), pady=3)
        self.widgets[key] = widget
        if help_text:
            tk.Label(parent, text=help_text, bg="white", fg="#555555", justify="left", wraplength=650).grid(row=row + 1, column=1, sticky="w", padx=(0, 10), pady=(0, 4))
            return row + 2
        return row + 1

    def _section(self, parent: tk.Frame, row: int, title: str) -> int:
        tk.Label(parent, text=title, font=("TkDefaultFont", 11, "bold"), bg="white").grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(10, 4))
        ttk.Separator(parent, orient="horizontal").grid(row=row + 1, column=0, columnspan=2, sticky="ew", padx=8)
        return row + 2

    def _build_dataset_tab(self) -> None:
        p = self.dataset_tab
        r = self._section(p, 0, "Dataset roots and selections")
        r = self._field(p, r, "input_dataset_folder", "Input dataset folder", "folder")
        r = self._field(p, r, "output_root", "Output root", "folder")
        r = self._field(p, r, "selected_positions", "Selected positions", help_text="Use ALL or exact comma-separated names from the discovery table.")
        r = self._field(p, r, "selected_channels", "Selected channels", help_text="Use ALL or exact comma-separated detected channel names.")
        r = self._field(p, r, "first_timepoint", "First timepoint")
        r = self._field(p, r, "last_timepoint", "Last timepoint")
        r = self._field(p, r, "reference_channel", "Reference channel", help_text="Required when multiple channels are selected; no ambiguous automatic choice is made.")
        r = self._section(p, r, "Naming rules")
        r = self._field(p, r, "position_folder_pattern", "Position-folder pattern", help_text="Python regex with named groups position and settings. Default splits at the final underscore.")
        r = self._field(p, r, "settings_pattern", "Settings name or pattern", help_text="Shell-style pattern; Settings 1 is the legacy default.")
        r = self._field(p, r, "view1_token", "View 1 token")
        r = self._field(p, r, "view2_token", "View 2 token")
        r = self._field(p, r, "fused_token", "Fused token")
        r = self._field(p, r, "tiff_glob_pattern", "TIFF glob pattern")
        r = self._field(p, r, "ome_companion_glob_pattern", "OME companion glob pattern")
        r = self._field(p, r, "timepoint_regex", "Timepoint regex or template", help_text="Provide named group timepoint or one capture group.")
        self._field(p, r, "channel_regex", "Channel regex or template", help_text="Provide named group channel. {view1}, {view2}, and {fused} are escaped and expanded.")

    def _build_workflow_tab(self) -> None:
        p = self.workflow_tab
        r = self._section(p, 0, "Stage orchestration")
        r = self._field(p, r, "processing_order", "Processing order", "combobox", [ORDER_BGS_CROP, ORDER_CROP_BGS])
        self.order_warning = tk.Label(p, text="", bg="#fff4cc", fg="#7a4d00", justify="left", wraplength=720, padx=8, pady=6)
        self.order_warning.grid(row=r, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        r += 1
        r = self._field(p, r, "coarse_crop_during_bgs", "Coarse crop during background subtraction", "checkbox")
        self.coarse_warning = tk.Label(p, text="", bg="#eef6ff", fg="#204a70", justify="left", wraplength=720, padx=8, pady=6)
        self.coarse_warning.grid(row=r, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        r += 1
        r = self._field(p, r, "start_stage", "Start stage", "combobox", ["fusion", "registration", "background_subtraction", "crop"])
        r = self._field(p, r, "stop_after_stage", "Stop after stage", "combobox", ["fusion", "registration", "background_subtraction", "crop", "final"], help_text="final follows the selected processing order: Crop for the default order, Background subtraction for Crop → Background subtraction.")
        r = self._field(p, r, "dry_run", "Dry run", "checkbox")
        r = self._field(p, r, "resume", "Resume", "checkbox")
        r = self._field(p, r, "overwrite", "Overwrite", "checkbox")
        r = self._field(p, r, "continue_on_error", "Continue on error", "checkbox", help_text="Continues independent positions inside a stage, but dependent stages remain blocked if any selected position fails validation.")
        r = self._section(p, r, "Fusion")
        r = self._field(p, r, "fusion_crop_um", "Fusion analysis crop [µm]")
        self._field(p, r, "fusion_save_as_crop", "Save fusion as center crop", "checkbox", help_text="Disabled by default for the end-to-end pipeline; final object-aware crop remains separate.")
        self._update_warnings()

    def _build_registration_tab(self) -> None:
        p = self.registration_tab
        r = self._section(p, 0, "Registration parameters")
        fields = [
            ("registration_glob_pattern", "Glob pattern", "entry", None),
            ("registration_downsample_xy", "Downsample XY", "entry", None),
            ("registration_downsample_z", "Downsample Z", "entry", None),
            ("registration_estimate_method", "Estimate method", "combobox", ["center", "phase_mip", "center_phase_mip"]),
            ("registration_apply_mode", "Apply mode", "combobox", ["integer", "subpixel"]),
            ("registration_compression", "Compression", "combobox", ["none", "zlib", "lzw"]),
            ("registration_save_dtype", "Save dtype", "entry", None),
            ("registration_padding_z", "Padding Z", "entry", None),
            ("registration_padding_y", "Padding Y", "entry", None),
            ("registration_padding_x", "Padding X", "entry", None),
            ("registration_phase_upsample", "Phase upsample", "entry", None),
            ("registration_series_index", "Series index", "entry", None),
            ("registration_channel_index", "Channel index (blank = none)", "entry", None),
            ("registration_clip_low", "Clip low percentile", "entry", None),
            ("registration_clip_high", "Clip high percentile", "entry", None),
            ("registration_center_threshold_percentile", "Center threshold percentile", "entry", None),
            ("registration_max_shift_z", "Maximum Z shift (blank = none)", "entry", None),
            ("registration_max_shift_yx", "Maximum Y/X shift (blank = none)", "entry", None),
        ]
        for key, label, kind, choices in fields:
            r = self._field(p, r, key, label, kind, choices)
        tk.Label(p, text="Each selected position is registered independently. Its own OME companion is passed only to obtain voxel spacing; registration does not use XML to estimate the transform.", bg="#eef6ff", fg="#204a70", justify="left", wraplength=720, padx=8, pady=6).grid(row=r, column=0, columnspan=2, sticky="ew", padx=10, pady=8)

    def _build_bgs_tab(self) -> None:
        p = self.bgs_tab
        r = self._section(p, 0, "Background-subtraction parameters")
        fields = [
            ("bgs_glob_pattern", "Glob pattern", "entry", None),
            ("bgs_method", "Method", "combobox", ["zmin_smooth"]),
            ("bgs_z_projection", "Z projection", "combobox", ["min", "percentile"]),
            ("bgs_z_bg_percentile", "Z background percentile", "entry", None),
            ("bgs_bg_sigma", "Background sigma", "entry", None),
            ("bgs_bg_downsample", "Background smoothing downsample", "entry", None),
            ("bgs_bg_alpha", "Background alpha", "entry", None),
            ("bgs_auto_offset", "Automatic offset", "checkbox", None),
            ("bgs_offset_percentile", "Offset percentile", "entry", None),
            ("bgs_offset_sample_slices", "Offset sample slices", "entry", None),
            ("bgs_bg_offset", "Manual background offset", "entry", None),
            ("bgs_bbox_downsample_xy", "Coarse-crop bbox downsample XY", "entry", None),
            ("bgs_bbox_downsample_z", "Coarse-crop bbox downsample Z", "entry", None),
            ("bgs_bbox_dilate", "Coarse-crop bbox dilation", "entry", None),
            ("bgs_mask_percentile", "Coarse-crop mask percentile", "entry", None),
            ("bgs_output_dtype", "Output dtype", "combobox", ["same", "uint16", "uint8", "float32"]),
            ("bgs_compression", "Compression", "combobox", ["zlib", "lzw", "none"]),
        ]
        for key, label, kind, choices in fields:
            r = self._field(p, r, key, label, kind, choices)
        tk.Label(p, text="Output dtype defaults to same. The pipeline never changes dtype unless this field is explicitly changed.", bg="#fff4cc", fg="#7a4d00", justify="left", wraplength=720, padx=8, pady=6).grid(row=r, column=0, columnspan=2, sticky="ew", padx=10, pady=8)

    def _build_crop_tab(self) -> None:
        p = self.crop_tab
        r = self._section(p, 0, "Object-aware crop parameters")
        fields = [
            ("crop_glob_pattern", "Glob pattern", "entry", None),
            ("crop_analysis_downsample_xy", "Analysis downsample XY", "entry", None),
            ("crop_analysis_downsample_z", "Analysis downsample Z", "entry", None),
            ("crop_xy_projection", "XY projection", "combobox", ["percentile", "max", "mean"]),
            ("crop_xy_projection_percentile", "XY projection percentile", "entry", None),
            ("crop_xy_smooth_sigma", "XY smooth sigma", "entry", None),
            ("crop_xy_threshold_method", "XY threshold method", "combobox", ["otsu", "percentile"]),
            ("crop_xy_threshold_percentile", "XY threshold percentile", "entry", None),
            ("crop_xy_threshold_floor_percentile", "XY threshold floor percentile", "entry", None),
            ("crop_min_area_pixels", "Minimum area pixels", "entry", None),
            ("crop_keep_border_components", "Keep border components", "checkbox", None),
            ("crop_no_crop_z", "Keep all Z (--no-crop-z)", "checkbox", None),
            ("crop_z_profile_percentile", "Z profile percentile", "entry", None),
            ("crop_z_profile_fraction", "Z profile fraction", "entry", None),
            ("crop_z_smooth_sigma", "Z smooth sigma", "entry", None),
            ("crop_padding_xy", "Crop padding XY", "entry", None),
            ("crop_padding_z", "Crop padding Z", "entry", None),
            ("crop_square_xy", "Square XY crop", "checkbox", None),
            ("crop_compression", "Compression", "combobox", ["zlib", "lzw", "none"]),
            ("crop_no_qc_png", "Disable QC PNGs", "checkbox", None),
        ]
        for key, label, kind, choices in fields:
            r = self._field(p, r, key, label, kind, choices)
        tk.Label(p, text="The existing crop implementation calculates one common crop for all timepoints in each position. This behavior is preserved.", bg="#eef6ff", fg="#204a70", justify="left", wraplength=720, padx=8, pady=6).grid(row=r, column=0, columnspan=2, sticky="ew", padx=10, pady=8)

    def _browse_folder(self, key: str) -> None:
        path = filedialog.askdirectory(initialdir=str(self.vars[key].get() or Path.home()))
        if path:
            self.vars[key].set(path)

    def _update_warnings(self) -> None:
        order = str(self.vars.get("processing_order", tk.StringVar(value=ORDER_BGS_CROP)).get())
        if order == ORDER_CROP_BGS:
            self.order_warning.configure(text="Warning: cropping before background subtraction may reduce processing time, but gives the background estimator less surrounding background and may make crop detection more sensitive to raw haze.")
        else:
            self.order_warning.configure(text="Default: full-frame background subtraction is followed by final object-aware cropping, keeping intensity correction and dimensional reduction conceptually separate.")
        coarse = bool(self.vars.get("coarse_crop_during_bgs", tk.BooleanVar(value=False)).get())
        if coarse:
            self.coarse_warning.configure(text="Enabled optimization: BGS will receive --crop-to-foreground --crop-padding-xy 80 --crop-padding-z 8, followed by the final tighter object-aware crop.")
        else:
            self.coarse_warning.configure(text="Disabled by default: BGS remains full-frame and the final object-aware crop is a separate stage.")

    def _collect_config(self) -> dict[str, Any]:
        config = dict(DEFAULT_CONFIG)
        errors = []
        for key, var in self.vars.items():
            raw = var.get()
            default = DEFAULT_CONFIG[key]
            try:
                if isinstance(default, bool):
                    value = bool(raw)
                elif isinstance(default, int) and not isinstance(default, bool):
                    value = int(raw)
                elif isinstance(default, float):
                    value = float(raw)
                else:
                    value = str(raw)
                config[key] = value
            except (TypeError, ValueError):
                errors.append(f"{key} has an invalid value: {raw!r}")
        if errors:
            raise ValueError("\n".join(errors))
        return config

    def _load_values(self, values: dict[str, Any]) -> None:
        for key in DEFAULT_CONFIG:
            if key not in self.vars:
                continue
            self.vars[key].set(values.get(key, DEFAULT_CONFIG[key]))
        self._update_warnings()

    def _append_log(self, text: str) -> None:
        self.log.insert("end", text.rstrip("\n") + "\n")
        self.log.see("end")

    def scan(self) -> None:
        if self.pipeline_process and self.pipeline_process.poll() is None:
            messagebox.showinfo("Pipeline running", "Stop the active pipeline before scanning again.")
            return
        try:
            config = self._collect_config()
        except ValueError as exc:
            messagebox.showerror("Invalid fields", str(exc))
            return
        self.validation_text.set("Scanning dataset…")
        self.validation_label.configure(fg="blue")
        threading.Thread(target=self._scan_worker, args=(config,), daemon=True).start()

    def _scan_worker(self, config: dict[str, Any]) -> None:
        try:
            result = scan_dataset(config)
            self.worker_queue.put(("scan_result", (config, result)))
        except Exception as exc:
            self.worker_queue.put(("error", f"Dataset scan failed: {exc}"))

    def _display_scan(self, result: Any) -> None:
        self.scan_result = result
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in result.rows:
            self.tree.insert("", "end", values=(row.position, row.settings, row.timepoint, row.channel, row.view1_path, row.view2_path, row.ome_path, row.pairing_status))
        self._append_log("Dataset discovery table:\n" + discovery_table_text(result))
        if result.issues:
            self._append_log("Discovery issues:")
            for issue in result.issues:
                self._append_log(f"- {issue.get('type')}: {issue.get('message')} ({issue.get('path')})")

    def validate(self) -> bool:
        try:
            config = self._collect_config()
        except ValueError as exc:
            self.validation_text.set(str(exc))
            self.validation_label.configure(fg="red")
            return False
        result = scan_dataset(config)
        self._display_scan(result)
        errors = validate_configuration(config, result)
        if errors:
            self.validation_text.set(f"Validation failed: {len(errors)} issue(s). See log.")
            self.validation_label.configure(fg="red")
            self._append_log("Validation errors:")
            for error in errors:
                self._append_log("- " + error)
            return False
        self.validation_text.set("Configuration is valid.")
        self.validation_label.configure(fg="green")
        self._append_log("Configuration is valid.")
        return True

    def start(self) -> None:
        if self.pipeline_process and self.pipeline_process.poll() is None:
            messagebox.showinfo("Pipeline running", "A pipeline process is already running.")
            return
        if not self.validate():
            return
        config = self._collect_config()
        fd, path = tempfile.mkstemp(prefix="lightsheet_pipeline_", suffix=".json")
        os.close(fd)
        Path(path).write_text(json.dumps(config, indent=2), encoding="utf-8")
        self.temp_config_path = path
        command = [sys.executable, str(SCRIPT_DIR / "pipeline_runner.py"), "--config", path]
        self._append_log("Launching: " + subprocess.list2cmdline(command))
        self.current_stage.set("Starting")
        self.current_position.set("")
        self.completed_count.set("0")
        self.failed_count.set("0")
        popen_kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
            "cwd": str(ROOT),
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.pipeline_process = subprocess.Popen(command, **popen_kwargs)
        self.pipeline_reader = threading.Thread(target=self._read_pipeline_output, daemon=True)
        self.pipeline_reader.start()

    def _read_pipeline_output(self) -> None:
        assert self.pipeline_process is not None and self.pipeline_process.stdout is not None
        for line in iter(self.pipeline_process.stdout.readline, ""):
            if not line:
                break
            self.worker_queue.put(("pipeline_line", line))
        rc = self.pipeline_process.wait()
        self.worker_queue.put(("pipeline_done", rc))

    def request_stop(self) -> None:
        proc = self.pipeline_process
        if proc is None or proc.poll() is not None:
            self._append_log("No active pipeline process.")
            return
        self._append_log("Requesting cooperative stop…")
        if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()

    def save_settings(self) -> None:
        try:
            config = self._collect_config()
        except ValueError as exc:
            messagebox.showerror("Invalid fields", str(exc))
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if path:
            Path(path).write_text(json.dumps(config, indent=2), encoding="utf-8")
            self._append_log(f"Saved settings: {path}")

    def load_settings(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not path:
            return
        try:
            values = json.loads(Path(path).read_text(encoding="utf-8"))
            merged = dict(DEFAULT_CONFIG)
            merged.update(values)
            self._load_values(merged)
            self.scan_result = None
            self.validation_text.set("Settings loaded; scan and validate again.")
            self.validation_label.configure(fg="blue")
            self._append_log(f"Loaded settings: {path}")
        except Exception as exc:
            messagebox.showerror("Load settings", str(exc))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "scan_result":
                    _config, result = payload
                    self._display_scan(result)
                    self.validation_text.set(f"Scan complete: {len(result.rows)} discovery row(s), {len(result.issues)} issue(s).")
                    self.validation_label.configure(fg="green" if not result.issues else "red")
                elif kind == "pipeline_line":
                    line = payload.rstrip("\n")
                    if line.startswith("PIPELINE_EVENT "):
                        try:
                            event = json.loads(line[len("PIPELINE_EVENT "):])
                            if "current_stage" in event:
                                self.current_stage.set(str(event["current_stage"]))
                            if "current_position" in event:
                                self.current_position.set(str(event["current_position"]))
                            if "completed_file_count" in event:
                                self.completed_count.set(str(event["completed_file_count"]))
                            if "failed_file_count" in event:
                                self.failed_count.set(str(event["failed_file_count"]))
                        except json.JSONDecodeError:
                            pass
                    self._append_log(line)
                elif kind == "pipeline_done":
                    rc = int(payload)
                    self.current_stage.set("Completed" if rc == 0 else "Stopped/failed")
                    self.validation_text.set(f"Pipeline exited with status {rc}.")
                    self.validation_label.configure(fg="green" if rc == 0 else "red")
                    self._append_log(f"Pipeline process exited with status {rc}.")
                    if self.temp_config_path:
                        try:
                            Path(self.temp_config_path).unlink(missing_ok=True)
                        except Exception:
                            pass
                        self.temp_config_path = None
                elif kind == "error":
                    self.validation_text.set(str(payload))
                    self.validation_label.configure(fg="red")
                    self._append_log(str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def close(self) -> None:
        proc = self.pipeline_process
        if proc and proc.poll() is None:
            if not messagebox.askyesno("Pipeline running", "Request stop and close the GUI?"):
                return
            if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
