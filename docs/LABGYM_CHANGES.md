# How our LabGym differs from upstream

This is a **modified copy of [LabGym](https://github.com/umyelab/LabGym)** (Ye Lab,
University of Michigan; GPL-3.0), carrying the changes listed below. LabGrymace depends on
these changes, so the public LabGym cannot be used in its place.

The folder is `LabGym_LabGrymace/`; the import package inside it is
`LabGym_LabGrymace_copy/`. It installs as the pip package **`LabGym_LabGrymace`** and runs
with the command **`LabGym_LabGrymace`**. These names differ from the public `LabGym`, so
both can be installed on the same computer.

It reports version `2.9.0`. This number is fixed on purpose: 2.9.0 is the
exact build that LabGrymace was calibrated and tested against.

## Two LabGyms, side by side

| | **Upstream LabGym** | **This build** |
|---|---|---|
| pip package | `LabGym` | `LabGym_LabGrymace` |
| install | `pip install LabGym` | `pip install ./LabGym_LabGrymace` |
| launch | `LabGym` | `LabGym_LabGrymace` |
| use for | general LabGym work | producing LabGrymace's input |


> **LabGym_LabGrymace is required for LabGrymace, not optional.** Only it writes the output fields
> and the `.keras` categorizer format that LabGrymace reads.

## What changed

A row marked **\*** changes the numbers in the output. Every other row only prevents a
crash or adds a convenience, and does not change results.

| File | Change |
|---|---|
| `analyzebehavior_dt.py` | **\*** Each facial region is scored only against its own behaviors, so an ear is never labeled with an eye behavior. Frames in which the detector finds nothing no longer affect the area measurements. |
| `categorizer.py` | Trained models are saved in the newer `.keras` format, which upstream LabGym cannot open, and load under TensorFlow 2.17 / Keras 3. Testing a categorizer also exports a confusion matrix. |
| `analyzebehavior.py` | The categorizer runs on the CPU so it does not compete with the detector for the GPU. Also fixes trajectory drawing and UTF-8 log writing. |
| `tools.py` | The CUDA path is read from the environment instead of a hardcoded cluster path. Adds a per-frame pattern-image option, and guards against empty frames, empty contours, and zero-area contours that previously crashed. |
| `gui_preprocessor.py` | A video that OpenCV cannot read (for example AV1) is transcoded to H.264 automatically instead of failing. |
| `gui_categorizer.py` | Adds an option to export the detected region of every frame. |
| `gui_analyzer.py` | A categorizer trained in an unsupported behavior mode is refused with a clear message instead of failing partway through the analysis. |
| `__init__.py` | TensorFlow and PyTorch are configured to share one GPU, instead of whichever loads first taking all of its memory. |
| `__main__.py` | Removes the startup check that contacted PyPI on every launch and told users to upgrade to a release this build cannot read. |

## Vendored detectron2

`LabGym_LabGrymace_copy/detectron2/` is upstream's bundled detectron2 (Apache-2.0, Meta).
It is unchanged except for the import namespace, which was renamed from upstream's
`LabGym.detectron2` to `LabGym_LabGrymace_copy.detectron2`.
