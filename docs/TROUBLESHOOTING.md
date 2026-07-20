# Troubleshooting

**`command not found: LabGrYMace` (macOS) / `'LabGrYMace' is not recognized` (Windows)**
The virtual environment isn't active, or the install didn't finish. Re-activate it
(`source .venv/bin/activate` / `.venv\Scripts\activate`) and re-run `pip install .`.
`python -m LabGrYMace` always works as a fallback.

**pip tries to build wxPython from source and fails**
Run `pip install --upgrade pip` first so it finds the pre-built wheel. On Linux, install a
matching wheel from <https://extras.wxpython.org/wxPython4/extras/linux/>.

**"Illegal instruction" or a TensorFlow/AVX crash on Apple Silicon**
You are running an x86-64 Python under Rosetta. Reinstall a native arm64 Python from
<https://www.python.org/downloads/macos/> and recreate the virtual environment.

**`No module named LabGrYMace`**
Run `pip install .` from the folder containing `pyproject.toml`, with the venv active.

**PowerShell blocks virtual-environment activation**
Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

**The GUI doesn't appear**
Check behind other windows — the Dock (macOS) or taskbar (Windows). The first launch
takes 10–30 s while TensorFlow and detectron2 load.

**LabGym can't find a detector or categorizer**
Models are not shipped in this repository (see the Models section of the main README).
Place a detector under `LabGym/LabGym/detectors/` and a categorizer under
`LabGym/LabGym/models/`, one folder per model, then re-open the selection dialog.

**A categorizer is rejected as "Unsupported"**
Its `model_parameters.txt` declares a behavior mode this build does not support. Train or
select a categorizer in one of the supported modes.

**LabGym runs on CPU when a GPU is present**
`CUDA_HOME` (or `CUDA_PATH`) must point at your CUDA toolkit before launching. On an HPC
cluster, load the CUDA module first. Note that `analyzebehavior.py` pins categorizer
inference to CPU by design, to avoid CUDA context conflicts with detectron2.
