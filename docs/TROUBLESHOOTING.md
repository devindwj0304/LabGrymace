# Troubleshooting

Find the line that matches **what you see on screen**. Each entry says why it happens and
what to do.

Before anything else, check these two — they explain most problems:

1. **Is your environment active?** Your prompt must show its name, e.g. `(labgrymace)`.
   If it shows `(base)` or nothing, run `conda activate labgrymace` (or
   `source .venv/bin/activate` on macOS/Linux, `.venv\Scripts\activate` on Windows).
2. **Which command are you running?** `LabGrymace` is the pain-scoring tool.
   `LabGym_LabGrymace` is the LabGym build that produces its input. They are two separate
   programs.

---

## Installing

### `command not found: LabGrymace` / `'LabGrymace' is not recognized`

**Why:** the environment is not active, or the install did not finish.

**Fix:** activate the environment, then re-run `pip install .` from the repository folder.
If it still fails, `python -m LabGrymace` always works — it does not depend on the command
being on your PATH.

### `No module named LabGrymace`

**Why:** `pip install .` was run from the wrong folder.

**Fix:** run it from the folder that contains `pyproject.toml`, with the environment active.

### pip spends a long time building wxPython, then fails

**Why:** pip did not find a ready-made wheel and tried to compile it from source.

**Fix:** run `pip install --upgrade pip` first, then install again. On Linux, download a
matching wheel from <https://extras.wxpython.org/wxPython4/extras/linux/>.

### `Package 'labgrymace' requires a different Python`

**Why:** your Python is outside the supported range.

**Fix:** use Python 3.9, 3.10 or 3.11. Check with `python --version`; 3.10 is recommended.

### PowerShell refuses to activate the environment

**Why:** Windows blocks scripts by default.

**Fix:** run once, then try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### It worked yesterday, and today it crashes with `numpy.dtype size changed`

**Why:** you installed into conda's `base` environment (the prompt showed `(base)`).
`pip install ./LabGym` then downgraded NumPy and TensorFlow *for everything on your
machine*, which breaks LabGrymace and other tools with it.

**Fix:** install into an environment of its own instead:

```bash
conda create -n labgrymace python=3.10 -y
conda activate labgrymace
cd LabGym-LabGrymace
pip install .
pip install ./LabGym
```

To clean up `base`, run `pip uninstall -y LabGrymace LabGym_LabGrymace` there.

### `Illegal instruction`, or TensorFlow crashes on a Mac with Apple Silicon

**Why:** your Python is the Intel build running under Rosetta.

**Fix:** install a native **arm64** Python from
<https://www.python.org/downloads/macos/> and create the environment again.

---

## Launching

### `LabGym_LabGrymace` does nothing when I run it (but `LabGrymace` opens fine)

**Why:** it is starting — it loads TensorFlow and detectron2 before drawing its window.

**Fix:** wait **10–30 seconds** the first time, and look **behind your other windows**
(check the Dock on macOS, the taskbar on Windows). Running `LabGrymace` does not open the
LabGym window; each program needs its own command.

---

## Running LabGym

### The detector or categorizer list is empty

**Why:** the trained models are not included in this repository — they are about 2 GB, over
GitHub's file-size limit.

**Fix:** download them separately (see **Models** in the main README), then put each model
in its own folder under `LabGym/LabGym_LabGrymace/detectors/` or
`LabGym/LabGym_LabGrymace/models/`, and re-open the selection dialog.

### "This Categorizer uses a behavior mode that this build does not support"

**Why:** the categorizer's `model_parameters.txt` declares a behavior mode this build
cannot run. It is refused up front rather than failing halfway through an analysis.

**Fix:** use or retrain a categorizer in a supported mode.

### It runs on the CPU even though the machine has a GPU

**Why:** `CUDA_HOME` (or `CUDA_PATH`) does not point at your CUDA toolkit.

**Fix:** set it before launching; on an HPC cluster, load the CUDA module first. Note that
categorizer inference is pinned to the CPU **on purpose**, so that it cannot conflict with
detectron2 over the GPU — seeing the categorizer on CPU is expected, not a fault.

---

## Reading the results

These are not errors. They come up often enough to be worth explaining.

### Step 2 says it found no folders

**Why:** step 2 looks for folders that already contain `ear_summary.xlsx`,
`eye_summary.xlsx` and `nose_summary.xlsx`.

**Fix:** point it at the **output folder from step 1**, not at the raw LabGym folder.

### The first rows of `pain_scores_per_frame.xlsx` have no pain score

**Why:** each frame's score is built from the **preceding 2 seconds** (60 frames at
30 fps), so the first 59 frames have no history to average over.

**Fix:** nothing — this is by design, and those frames are excluded from every average.

### Some cells are empty in the middle of the spreadsheet

**Why:** that facial region was not visible in that frame. An empty cell means "not
measured", which is different from zero.

**Fix:** nothing. The pain score is still computed from whichever regions *were* visible,
with their weights rescaled to match — so a frame showing only the ears still gets a score.
It is normal for the eye and nose to be missing in over half the frames of a freely moving
mouse.

### The pain score itself is empty for some frames

**Why:** all three regions — ear, eye and nose — were missing at once.

**Fix:** nothing. Only those frames are skipped; window and overall scores ignore them
rather than treating them as zero.

### A pain score is negative

**Why:** the scale is anchored at 0 for uninjected baseline mice and 100 for 1 mg/kg CNO.
A negative value simply means the animal scored below the average baseline animal.

**Fix:** nothing — it does not mean "negative pain".
