# How our LabGym differs from upstream v2.9.0

[`LabGym/`](../LabGym) is **LabGym v2.9.0** with the upstream v2.9.1 logging/CLI update
backported, plus the changes below. It is included in this repository because LabGrYMace
consumes its output, and several of these changes are what make that output usable.

Upstream: <https://github.com/umyelab/LabGym> (Ye Lab, University of Michigan), GPL-3.0.

---

## Two LabGyms, side by side

There are two different things called "LabGym", and it matters which one you run:

| | **Upstream LabGym** | **This build (for LabGrYMace)** |
|---|---|---|
| What it is | the public LabGym, actively updated | LabGym 2.9.0 with our changes |
| Latest version | 3.0.1 (2026) and rising | pinned at 2.9.0 |
| pip package name | `LabGym` | `LabGym_LabGrYMace` |
| Install with | `pip install LabGym` | `pip install ./LabGym` (from the repo root) |
| Launch command | `LabGym` | `LabGym_LabGrYMace` |
| Use it for | general LabGym work, newest features | producing the tracking output LabGrYMace reads |

Because they now have **different package names and different commands, both can be
installed in the same environment at once** — no uninstalling, no switching:

```bash
pip install LabGym
pip install ./LabGym
```

The first installs upstream LabGym (launch with `LabGym`); the second installs this build
(launch with `LabGym_LabGrYMace`).

**How to run each one** (this is how you "activate" the LabGrYMace build):

```bash
LabGym
LabGym_LabGrYMace
```

`LabGym` opens upstream LabGym 3.0.1; `LabGym_LabGrYMace` opens the 2.9.0 build that
LabGrYMace needs.

**Check what is installed:**

```bash
pip show LabGym LabGym_LabGrYMace
python -c "import LabGym_LabGrYMace; print(LabGym_LabGrYMace.__version__)"
```

The last line prints `2.9.0`.

> **The 2.9.0 build is required for LabGrYMace — it is not an optional preference.**
> LabGrYMace reads output fields and a categorizer format (`*.keras`) that this build
> writes and upstream LabGym does not. Running LabGrYMace against output from upstream
> LabGym (3.0.1) will not work. Always generate LabGrYMace's input with
> `LabGym_LabGrYMace`.

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

`LabGym/LabGym_LabGrYMace/detectron2/` is the detectron2 tree that upstream LabGym v2.9.1
vendors, including its pre-built Windows C extensions (`_C.cp39-win_amd64.pyd`,
`_C.cp310-win_amd64.pyd`). It is required — `detector.py` imports it — and is not our
code. The only change from the upstream copy is the import namespace: references were
rewritten from `LabGym.detectron2` to `LabGym_LabGrYMace.detectron2` when the package was
renamed (see [Two LabGyms, side by side](#two-labgyms-side-by-side)); the logic is
untouched. detectron2 is Apache-2.0 licensed (Meta Platforms, Inc.).

## Package name

This build installs as `LabGym_LabGrYMace` rather than `LabGym`, and its launch command is
`LabGym_LabGrYMace`. The rename is what lets it sit alongside the upstream `LabGym` package
in one environment. The import namespace changed to match (`import LabGym_LabGrYMace`);
model and detector folders are unaffected, since they are located relative to the package,
not by name.
