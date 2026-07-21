# LabGrYMace

**LabGrYMace** (LabGym + *grimace*) quantifies **mouse facial expressions of pain**.
It reads the tracking output of [LabGym](https://github.com/umyelab/LabGym), merges the
per-frame intensity signals of the ear, eye, and nose regions, and produces a calibrated
**pain score** (0 = baseline, 100 = 1 mg/kg CNO reference) through a point-and-click GUI.

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

Requires **Python 3.9–3.11** (3.10 recommended). Runs on macOS and Windows.

```bash
git clone https://github.com/devindwj0304/LabGym-LabGrYMace.git
cd LabGym-LabGrYMace

python3 -m venv .venv           # Windows: python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install .                   # LabGrYMace         -> command: LabGrYMace
pip install ./LabGym            # our modified LabGym -> command: LabGym_LabGrYMace
```

On Apple Silicon, use a native arm64 Python — running under Rosetta crashes some
scientific wheels. On Windows, tick **"Add python.exe to PATH"** in the installer.

> **LabGrYMace requires our LabGym build — this is not optional.** It is installed as a
> separate package, `LabGym_LabGrYMace` (version 2.9.0), and launches with the command
> **`LabGym_LabGrYMace`**. Because it has its own name, it lives happily next to the
> upstream `LabGym` (3.0.1): you can keep both in the same environment and never uninstall
> or switch. The latest upstream LabGym alone will **not** work with LabGrYMace — it does
> not produce the output LabGrYMace reads. See
> [Two LabGyms, side by side](docs/LABGYM_CHANGES.md#two-labgyms-side-by-side).

**Models are not included.** The trained detectors and categorizers are ~2 GB and exceed
GitHub's file-size limit.

> **Model download:** _to be added._

Place a detector under `LabGym/LabGym_LabGrYMace/detectors/` and a categorizer under
`LabGym/LabGym_LabGrYMace/models/`, then select it in the GUI.

---

## Usage

```bash
LabGrYMace
```

![LabGrYMace GUI](docs/images/gui.png)

The workflow is two steps:

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
