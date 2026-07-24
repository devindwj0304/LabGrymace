# How our LabGym differs from upstream

[`LabGym/`](../LabGym) is the **upstream LabGym v2.9.1 source tree**, with the changes
below. It reports `__version__ = '2.9.0'`, the version LabGrymace was calibrated against.
It is included in this repository because LabGrymace consumes its output, and several of
these changes are what make that output usable.

Diffed file by file against upstream v2.9.1: of the package's 18 Python modules (excluding
the vendored detectron2 tree), **8 are byte-identical to upstream and 10 differ**. Every
differing block carries a `!New Update from Wenjin` comment, so the modifications can be
found by searching the source for that tag.

| Modified | What changed |
|---|---|
| `__init__.py` | GPU memory setup before any submodule import |
| `__main__.py` | startup PyPI version check removed |
| `gui_main.py` | credits and links for this build |
| `gui_analyzer.py` | unsupported-categorizer guard, per-animal behavior filtering |
| `gui_categorizer.py` | per-frame ROI export option |
| `gui_preprocessor.py` | automatic AV1 -> H.264 recovery |
| `tools.py` | CUDA path from the environment, per-frame pattern images, crash guards |
| `categorizer.py` | TF 2.17 / Keras 3 loading, `.keras` output, confusion matrices |
| `analyzebehavior.py` | CPU-pinned inference, trajectory and encoding fixes |
| `analyzebehavior_dt.py` | per-animal behavior filtering, low-confidence handling |

Upstream: <https://github.com/umyelab/LabGym> (Ye Lab, University of Michigan), GPL-3.0.

---

## Two LabGyms, side by side

There are two different things called "LabGym", and it matters which one you run:

| | **Upstream LabGym** | **This build (for LabGrymace)** |
|---|---|---|
| What it is | the public LabGym, actively updated | LabGym 2.9.0 with our changes |
| Latest version | 3.0.1 (2026) and rising | pinned at 2.9.0 |
| pip package name | `LabGym` | `LabGym_LabGrymace` |
| Install with | `pip install LabGym` | `pip install ./LabGym` (from the repo root) |
| Launch command | `LabGym` | `LabGym_LabGrymace` |
| Use it for | general LabGym work, newest features | producing the tracking output LabGrymace reads |

Because the package names and commands differ, **both can be installed in the same
environment at once** — nothing to uninstall, nothing to switch. Run `LabGym` for
upstream, `LabGym_LabGrymace` for this build. To check what you have:

```bash
pip show LabGym LabGym_LabGrymace
```

> **The 2.9.0 build is required for LabGrymace — not an optional preference.** LabGrymace
> reads output fields and a categorizer format (`*.keras`) that only this build writes, so
> always generate its input with `LabGym_LabGrymace`.

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
  frames, an empty `cdist` call during tracking, and `math.sqrt` on a non-positive animal
  area — all of which previously raised.
  *(`tools.py`, `analyzebehavior.py`, `analyzebehavior_dt.py`)*

- **Shared GPU between TensorFlow and PyTorch** — `__init__.py` sets TensorFlow memory
  growth and `PYTORCH_CUDA_ALLOC_CONF` before any submodule is imported, so detectron2 and
  the categorizer can share one GPU instead of the first framework claiming all of it.
  *(`__init__.py`)*

- **Trajectory rendering fix** — trajectories are drawn onto a copy of the background, so
  `Trajectory.jpg` is no longer progressively overwritten, and the same trajectories are
  also drawn on the annotated video frames. *(`analyzebehavior.py`)*

- **Unsupported categorizers are refused** — a categorizer whose `model_parameters.txt`
  declares behavior mode 4 is rejected with a clear message instead of failing mid-run.
  *(`gui_analyzer.py`, `analyzebehavior.py`)*

- **UTF-8 analysis log** — `Analysis log.txt` is written as UTF-8, fixing the
  `UnicodeEncodeError` on non-ASCII paths. *(`analyzebehavior.py`)*

- **No startup version check** — upstream contacts pypi.org on every launch and prompts
  users to upgrade; that call is removed, since it stalls behind restricted networks and
  the newer release cannot be used with LabGrymace. *(`__main__.py`)*

---

## Differences that change results

Four changes alter output rather than only preventing crashes. They are listed
separately so anyone comparing against stock LabGym knows where numbers may diverge.

| Change | Upstream v2.9.1 | This build | Where |
|---|---|---|---|
| Low-confidence frames with `uncertain == 0` | left as `NA` | **always** assigned the most likely behavior | `analyzebehavior_dt.py` |
| A prediction outside an animal's own behavior list | kept | suppressed to `NA` | `analyzebehavior_dt.py` |
| Trained model format | SavedModel directory | `best_model.keras` / `final_model.keras` inside the model directory — **not loadable by upstream v2.9.1** | `categorizer.py` |
| Zero-mask frames | contribute a `nan` to `area_diffs` | contribute no entry, slightly changing `magnitude_area` and `vigor_area` | `analyzebehavior_dt.py` |

---

## Portability note

Upstream-of-us, this fork hardcoded `CUDA_HOME` and `XLA_FLAGS` to
`/sw/pkgs/arc/cuda/12.6.3`, a path that exists only on the U-M Great Lakes cluster.
These are now read from the environment (`CUDA_HOME` / `CUDA_PATH`), falling back to
`/usr/local/cuda` or `/opt/cuda`, and left unset when no CUDA toolkit is present so
TensorFlow runs on CPU. *(`tools.py`)*

## Vendored dependency

`LabGym/LabGym_LabGrymace/detectron2/` is the detectron2 tree that upstream LabGym v2.9.1
vendors, including its pre-built Windows C extensions (`_C.cp39-win_amd64.pyd`,
`_C.cp310-win_amd64.pyd`). It is required — `detector.py` imports it — and is not our
code. The only change from the upstream copy is the import namespace, rewritten from
`LabGym.detectron2` to `LabGym_LabGrymace.detectron2` so this build can sit alongside the
upstream `LabGym` package; the logic is untouched. detectron2 is Apache-2.0 licensed
(Meta Platforms, Inc.).
