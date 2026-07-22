# LabGrYMace

**LabGrYMace** (LabGym + *grimace*) quantifies **mouse facial expressions of pain**.

- **Objective and reproducible** — facial grimace is a recognized sign of pain in mice, but scoring it by eye is slow and subjective; LabGrYMace measures it automatically from [LabGym](https://github.com/umyelab/LabGym) tracking.
- **One calibrated score** — it merges ear, eye, and nose movement into a single **pain score** anchored to 0 (baseline) and 100 (a 1 mg/kg CNO reference), so results are comparable across animals and experiments.
- **No coding needed** — a point-and-click GUI outputs summary spreadsheets, the pain-score analysis, and a video with the score overlaid on every frame.

This repository contains two parts: **[`LabGrYMace/`](LabGrYMace)**, the pain-quantification
tool, and **[`LabGym/`](LabGym)**, our modified LabGym 2.9.0 that produces the tracking
output it reads.

---

## The outputs of LabGrYMace include:

**1. Summary spreadsheets** — `ear_summary.xlsx`, `eye_summary.xlsx`, `nose_summary.xlsx`,
one set per recording, holding the per-frame intensity table for each facial region.
Frames removed by the mirror-reflection filter are marked **orange**; the merged column
feeding the pain score is **yellow**.

![Summary spreadsheet](docs/images/output-summary.png)

**2. Pain score analysis** — `pain_scores.xlsx` (one score per recording),
`pain_scores_per_frame.xlsx` (the full time course), and the charts
`pain_score_chart.png` / `overall_pain_score_chart.png`. Optionally a 9×9 correlation
matrix across the intensity parameters, with a heatmap.

![Pain score](docs/images/output-painscore.png)

**3. Video with the pain score overlaid** — an `.mp4` copy of the recording with the
running pain score burned into each frame, so the score can be watched against the
animal's behavior.

![Overlay video](docs/images/output-overlay.gif)

---

## Installation

Requires **Python 3.10** (3.9 and 3.11 also work). Follow the block for your system.

> **Install into an isolated environment — never into conda `base`.** These packages pin
> specific versions of TensorFlow and NumPy, so installing them into `base` (or any shared
> environment) downgrades those libraries and can break other tools *and this one*. Always
> create a dedicated environment first and make sure it is **active** (your prompt shows
> its name, e.g. `(.venv)` or `(labgrymace)`) before running `pip install`. If you use
> Miniconda/Anaconda, a named conda environment is the most reliable choice:
>
> ```bash
> conda create -n labgrymace python=3.10 -y
> conda activate labgrymace
> ```
>
> Then run the two `pip install` lines below inside it (skip the `venv` step).

### macOS

```bash
git clone https://github.com/devindwj0304/LabGym-LabGrYMace.git
cd LabGym-LabGrYMace
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
pip install ./LabGym
```

This installs two commands: **`LabGrYMace`** (the pain tool) and **`LabGym_LabGrYMace`**
(our LabGym build). On Apple Silicon (M1–M4), use a native **arm64** Python from
<https://www.python.org/downloads/macos/> — one running under Rosetta crashes some wheels.

### Windows

Install Python from <https://www.python.org/downloads/windows/> and tick
**"Add python.exe to PATH"** during setup. Then, in **PowerShell**:

```powershell
git clone https://github.com/devindwj0304/LabGym-LabGrYMace.git
cd LabGym-LabGrYMace
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install .
pip install ./LabGym
```

This installs two commands: **`LabGrYMace`** (the pain tool) and **`LabGym_LabGrYMace`**
(our LabGym build). If activation is blocked, run once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

> **The `LabGym_LabGrYMace` build is required — not optional.** LabGrYMace reads output
> that only this 2.9.0 build produces; the latest upstream `LabGym` (3.0.1) will not work
> with it. Because they have different names, both can be installed at once — see
> [Two LabGyms, side by side](docs/LABGYM_CHANGES.md#two-labgyms-side-by-side).

**Models are not included.** The trained detectors and categorizers are ~2 GB and exceed
GitHub's file-size limit.

> **Model download:** _to be added._

Place a detector under `LabGym/LabGym_LabGrYMace/detectors/` and a categorizer under
`LabGym/LabGym_LabGrYMace/models/`, then select it in the GUI.

---

## Usage

### Launching

Activate your virtual environment first. Two GUIs are installed — **`LabGrYMace`** is the
pain-scoring tool, and **`LabGym_LabGrYMace`** is our LabGym build (run it to produce the
tracking output that LabGrYMace reads):

```bash
LabGrYMace
LabGym_LabGrYMace
```

If a command is *not found* — a PATH issue, common when the virtual environment sits
inside a conda base — launch the same window through Python instead. This always works:

```bash
python -m LabGrYMace
python -m LabGym_LabGrYMace
```

> **These are two independent windows.** `LabGrYMace` opens immediately.
> **`LabGym_LabGrYMace` takes about 10–30 seconds to open** the first time — it loads
> TensorFlow and detectron2 before the window appears, and the window can open *behind*
> other windows. Wait for it, and check your Dock/taskbar; it did not fail to launch.

![LabGrYMace GUI](docs/images/gui.png)

The workflow (in the **`LabGrYMace`** window) is two steps:

1. **Generate summary files** — point it at a folder of LabGym output. This writes the
   three `*_summary.xlsx` files into each recording's folder. Tick the correlation box to
   also export the 9×9 matrix and heatmap.
2. **Compute pain scores** — point it at those summary folders. This writes
   `pain_scores.xlsx` and the charts, and can generate the overlay video.

The mirror-reflection filter is **off by default** and is a checkbox in step 2. The
published figures were produced with it on; the reproduction scripts set it explicitly.

Also available from the command line and from Python:

```bash
python -m LabGrYMace.loaddata "/path/to/folder/of/LabGym/datasets"
```

```python
import numpy as np
from LabGrYMace.gui_main import load_raw_data, compute_per_frame_pain_scores

data  = load_raw_data("/path/to/one/animal/folder")
score = float(np.nanmean(compute_per_frame_pain_scores(data, lookback=60)))
print("overall pain score:", round(score, 2))
```

---

## Documentation

- **[How our LabGym differs from upstream](docs/LABGYM_CHANGES.md)** — what we added and
  which changes alter results
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**

---

## License

**GPL-3.0** ([`LICENSE`](LICENSE)). Built on **LabGym** (Ye Lab, University of Michigan),
also GPL-3.0, whose attribution and license are preserved in
[`LabGym/LICENSE.txt`](LabGym/LICENSE.txt). If you use LabGrYMace in research, please
cite LabGym as well.
