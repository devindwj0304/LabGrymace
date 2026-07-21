# How our LabGym differs from upstream v2.9.0

[`LabGym/`](../LabGym) is **LabGym v2.9.0** with the upstream v2.9.1 logging/CLI update
backported, plus the changes below. It is included in this repository because LabGrYMace
consumes its output, and several of these changes are what make that output usable.

Upstream: <https://github.com/umyelab/LabGym> (Ye Lab, University of Michigan), GPL-3.0.

---

## Switching between this LabGym and the latest upstream LabGym

Both packages are named `LabGym`, so one Python environment can hold only **one** of
them at a time. Upstream keeps releasing newer versions (e.g. 3.0.1, published 2026),
while LabGrYMace needs **this** 2.9.0 build — it reads output fields and a model format
(`*.keras`) that stock LabGym does not produce or consume the same way. You can switch
back and forth freely.

**Check which one is currently installed:**

```bash
python -c "import LabGym; print(LabGym.__version__)"
```

`2.9.0` is this build; `3.0.1` (or higher) is upstream.

**Switch to the latest upstream LabGym** (for other projects):

```bash
pip uninstall -y LabGym
pip install --upgrade LabGym
```

**Switch back to this build** (required before running LabGrYMace):

```bash
pip uninstall -y LabGym
pip install ./LabGym          # run from the repository root
```

**Recommended — keep two environments** so you never reinstall:

```bash
python3 -m venv .venv-labgrymace   # this build + LabGrYMace
source .venv-labgrymace/bin/activate
pip install ./LabGym .

python3 -m venv .venv-upstream     # latest upstream LabGym
source .venv-upstream/bin/activate
pip install --upgrade LabGym
```

Activate whichever environment you need; nothing to uninstall.

---

## Added capabilities

- **Individual-specific behavior filtering** — each animal in a recording can be scored
  against its own set of allowed behaviors; a prediction belonging to a different
  animal's repertoire is suppressed to `NA`.
  *(`analyzebehavior_dt.py`, `gui_analyzer.py`)*

- **Video decode compatibility** — videos OpenCV cannot read (e.g. AV1) are detected and
  transcoded to H.264 automatically, with a progress dialog, instead of failing silently.
  *(`gui_preprocessor.py`)*

- **Per-frame ROI export** — optionally writes every frame of each animation as its own
  JPEG next to the pattern image, so you can inspect exactly what the model sees at each
  timestep. *(`gui_categorizer.py`, `tools.py`)*

- **Diagnostic confusion matrices** — training and testing write
  `diagnostic_confusion_matrix.png`, with per-class F1 in the diagonal cells.
  *(`categorizer.py`)*

- **TensorFlow 2.17 / Keras 3 compatibility** — loader wrappers let legacy SavedModel
  directories load under Keras 3. Categorizer inference in `analyzebehavior.py` is pinned
  to CPU to avoid CUDA context conflicts with detectron2; `analyzebehavior_dt.py`
  deliberately does not, so the two paths differ in device placement.
  *(`categorizer.py`, `analyzebehavior.py`, `__init__.py`)*

- **Larger training batches** — `train_combnet_onfly` uses a batch size of 128.
  *(`categorizer.py`)*

- **Robustness fixes** — guards against empty contour sets, zero-area contours, `None`
  frames, and out-of-bounds writes that previously raised.
  *(`tools.py`, `analyzebehavior.py`, `analyzebehavior_dt.py`)*

---

## Differences that change results

Three changes alter output rather than only preventing crashes. They are listed
separately so anyone comparing against stock LabGym knows where numbers may diverge.

| Change | Stock LabGym v2.9.0 | This build |
|---|---|---|
| Low-confidence frames with `uncertain == 0` | left as `NA` | **always** assigned the most likely behavior |
| Trained model format | SavedModel directory | `best_model.keras` / `final_model.keras` inside the model directory — **not loadable by stock v2.9.0** |
| Zero-mask frames | contribute a `nan` to `area_diffs` | contribute no entry, slightly changing `magnitude_area` and `vigor_area` |

---

## Portability note

Upstream-of-us, this fork hardcoded `CUDA_HOME` and `XLA_FLAGS` to
`/sw/pkgs/arc/cuda/12.6.3`, a path that exists only on the U-M Great Lakes cluster.
These are now read from the environment (`CUDA_HOME` / `CUDA_PATH`), falling back to
`/usr/local/cuda` or `/opt/cuda`, and left unset when no CUDA toolkit is present so
TensorFlow runs on CPU. *(`tools.py`)*

## Vendored dependency

`LabGym/LabGym/detectron2/` is a byte-identical copy of the detectron2 tree that upstream
LabGym v2.9.1 vendors, including its pre-built Windows C extensions
(`_C.cp39-win_amd64.pyd`, `_C.cp310-win_amd64.pyd`). It is required — `detector.py`
imports it relatively — and is not our code. detectron2 is Apache-2.0 licensed
(Meta Platforms, Inc.).
