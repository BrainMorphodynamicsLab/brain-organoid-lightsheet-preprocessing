# brain-organoid-lightsheet-preprocessing
End-to-end processing of dual-camera light-sheet time-lapse data of organoids, from raw camera views to registered, background-corrected, cropped 3D volumes ready for analysis.

The pipeline takes the two opposing camera views produced by a light-sheet microscope and runs them through five stages:

**camera fusion → 3D registration → background subtraction → optional denoising → object-aware cropping**

Each stage is a standalone Python script. An orchestration layer discovers datasets on disk, validates that every timepoint has a complete view pair, runs the stages in order, records every command and parameter in a manifest, and supports resuming an interrupted run. The whole thing is operated through a local browser interface — no command-line knowledge required after installation.

Developed for [ADD: organoid type / biological application], as described in [ADD: manuscript citation once available].

---

## What it does

| Stage | Purpose |
|---|---|
| **Fusion** | Combines View 1 and View 2 into one volume using a DCT-based image-quality score to find the switching plane. The plane is computed once per timepoint from a reference channel and reused for the other channels. |
| **Registration** | Corrects sample drift over time. Shifts are estimated on downsampled stacks and applied at full resolution, independently per position, using the first selected timepoint as reference. |
| **Background subtraction** | Estimates a smooth XY background from the Z-minimum (or a low Z percentile) and subtracts it from every slice, with optional automatic removal of a residual baseline. |
| **Denoising** *(optional, off by default)* | Median and/or Gaussian filtering with an optional intensity floor. Changes voxel intensities — validate against your measurement before enabling. |
| **Cropping** | Detects the organoid and computes one common crop box across all timepoints of a position, preserving a consistent coordinate frame. Crops XY only by default. |

Registration, background subtraction, denoising, and cropping preserve the input dtype by default. All intermediate stages are written to disk and retained.

## Expected data layout

One folder per position/replicate; one TIFF per timepoint, channel, and camera view, each containing a complete ZYX stack:

```
Experiment/
└── Position 1_Settings 1/
    ├── t0001_488_View1.tif
    ├── t0001_488_View2.tif
    ├── t0002_488_View1.tif
    ├── t0002_488_View2.tif
    └── ome-tiff.companion.ome
```

Folder and filename patterns are configurable via regular expressions, so other naming conventions work too — see the manual, section 7.4.

## Requirements

- Python 3.9–3.11
- Disk space for all intermediate stages, not just the final output
- A modern browser (the GUI runs locally at `127.0.0.1`; no data is uploaded anywhere)

---

## Installation

Download or clone this repository, then set up a Python environment. **Install the packages and launch the GUI with the same Python interpreter** — mixing them is the single most common source of "module not found" errors.

### macOS — with conda (recommended if you have Anaconda or Miniconda)

```bash
cd /path/to/this/repository
conda env create -f environment.yml
conda activate lightsheet
```

### macOS — without conda

```bash
cd /path/to/this/repository
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-pipeline.txt
```

### Windows

Install Python from [python.org](https://www.python.org/downloads/), ticking **"Add Python to PATH"** during installation. Then, in Command Prompt:

```
cd /d "C:\path\to\this\repository"
py -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements-pipeline.txt
```


## Contact

Joana Schlag, joana.schlag@mls.uzh.ch
