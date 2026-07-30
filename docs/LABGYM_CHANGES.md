# How our LabGym differs from upstream

[`LabGym_LabGrymace/`](../LabGym_LabGrymace) is the upstream **LabGym v2.9.1** source tree with the changes below.
It reports version `2.9.0` — the build LabGrymace was calibrated against — and installs as
`LabGym_LabGrymace` so it can sit next to the public `LabGym`. Every change carries a
`!New Update from Wenjin` comment in the source; the remaining Python modules are
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
| `analyzebehavior_dt.py` | **\*** Each facial region is scored only against its own behaviors, so an ear is never given an eye behavior. Frames in which the detector finds nothing no longer affect the area measurements. |
| `categorizer.py` | Trained models are saved in the newer `.keras` format, which upstream LabGym cannot open, and load under TensorFlow 2.17 / Keras 3. Testing a categorizer also exports a confusion matrix. |
| `analyzebehavior.py` | The categorizer runs on the CPU so it does not compete with the detector for the GPU. Includes trajectory-drawing and UTF-8 log fixes. |
| `tools.py` | The CUDA path is read from the environment instead of a hardcoded cluster path. Adds a per-frame pattern-image option and guards against empty frames, empty contours, and zero-area contours that previously crashed. |
| `gui_preprocessor.py` | A video that cannot be read (for example AV1) is transcoded to H.264 automatically instead of failing. |
| `gui_categorizer.py` | Adds an option to export the detected region of every frame. |
| `gui_analyzer.py` | A categorizer trained in an unsupported behavior mode is refused with a clear message instead of failing partway through the analysis. |
| `__init__.py` | TensorFlow and PyTorch are configured to share one GPU, instead of the first one to load claiming all of its memory. |
| `__main__.py` | Removes the startup check that contacted PyPI on every launch and told users to upgrade to a release this build cannot read. |

## Vendored detectron2

`LabGym_LabGrymace/detectron2/` is upstream's vendored detectron2 (Apache-2.0, Meta),
unchanged except the import namespace `LabGym.detectron2` → `LabGym_LabGrymace.detectron2`.
