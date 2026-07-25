# How our LabGym differs from upstream

[`LabGym/`](../LabGym) is the upstream **LabGym v2.9.1** source tree with the changes below.
It reports version `2.9.0` — the build LabGrymace was calibrated against — and installs as
`LabGym_LabGrymace` so it can sit next to the public `LabGym`. Every change carries a
`!New Update from Wenjin` comment in the source; the other 8 Python modules are
byte-identical to upstream.

Upstream: <https://github.com/umyelab/LabGym> (Ye Lab, University of Michigan), GPL-3.0.

## Two LabGyms, side by side

| | **Upstream LabGym** | **This build** |
|---|---|---|
| pip package | `LabGym` | `LabGym_LabGrymace` |
| install | `pip install LabGym` | `pip install ./LabGym` |
| launch | `LabGym` | `LabGym_LabGrymace` |
| use for | general LabGym work | producing LabGrymace's input |

Different names, so both can be installed at once — nothing to uninstall or switch.

> **The 2.9.0 build is required for LabGrymace, not optional** — only it writes the output
> fields and `.keras` categorizer format LabGrymace reads.

## What changed

Rows marked **\*** change the output numbers; the rest prevent crashes or add convenience.

| File | Change |
|---|---|
| `analyzebehavior_dt.py` | **\*** per-animal behavior filtering; low-confidence frames assigned instead of `NA`; zero-mask frames dropped from `area_diffs` |
| `categorizer.py` | **\*** trained models saved as `.keras` (not loadable by upstream); TF 2.17 / Keras 3 loading; confusion matrices |
| `analyzebehavior.py` | categorizer inference pinned to CPU (avoids detectron2 CUDA clash); trajectory and UTF-8 log fixes |
| `tools.py` | `CUDA_HOME` read from the environment (not a hardcoded cluster path); per-frame pattern images; guards for `None` frames, empty contours, zero area |
| `gui_preprocessor.py` | unreadable videos (e.g. AV1) auto-transcoded to H.264 |
| `gui_categorizer.py` | optional per-frame ROI export |
| `gui_analyzer.py` | rejects categorizers in an unsupported behavior mode |
| `__init__.py` | TensorFlow and PyTorch share one GPU |
| `__main__.py` | startup PyPI version check removed |
| `gui_main.py` | credits and repo links for this build |

## Vendored detectron2

`LabGym_LabGrymace/detectron2/` is upstream's vendored detectron2 (Apache-2.0, Meta),
unchanged except the import namespace `LabGym.detectron2` → `LabGym_LabGrymace.detectron2`.
