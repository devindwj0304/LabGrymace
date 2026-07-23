# Troubleshooting

**LabGrymace worked yesterday but breaks today, after installing LabGym**
You installed into conda `base` (the prompt showed `(base)`), so `pip install ./LabGym`
downgraded NumPy/TensorFlow there and broke the compiled libraries LabGrymace uses —
typically `numpy.dtype size changed`. Reinstall in a dedicated environment:

```bash
conda create -n labgrymace python=3.10 -y
conda activate labgrymace
cd LabGym-LabGrymace
pip install .
pip install ./LabGym
```

To repair `base`, run `pip uninstall -y LabGrymace LabGym_LabGrymace` there.

**`command not found: LabGrymace` (macOS) / `'LabGrymace' is not recognized` (Windows)**
The environment isn't active, or the install didn't finish. Re-activate it
(`conda activate labgrymace`, or `source .venv/bin/activate` / `.venv\Scripts\activate`)
and re-run `pip install .`. `python -m LabGrymace` always works as a fallback.

**pip tries to build wxPython from source and fails**
Run `pip install --upgrade pip` first so it finds the pre-built wheel. On Linux, install a
matching wheel from <https://extras.wxpython.org/wxPython4/extras/linux/>.

**"Illegal instruction" or a TensorFlow/AVX crash on Apple Silicon**
You are running an x86-64 Python under Rosetta. Reinstall a native arm64 Python from
<https://www.python.org/downloads/macos/> and recreate the virtual environment.

**`No module named LabGrymace`**
Run `pip install .` from the folder containing `pyproject.toml`, with the venv active.

**PowerShell blocks virtual-environment activation**
Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

**`LabGym_LabGrymace` seems not to launch (but `LabGrymace` does)**
It is launching — it just takes **10–30 seconds** to load TensorFlow and detectron2, and
can open behind other windows, so check your Dock/taskbar. They are separate programs:
each needs its own command (`LabGym_LabGrymace` for our build, plain `LabGym` for upstream
if you also installed it).

**LabGym can't find a detector or categorizer**
Models are not shipped in this repository (see the Models section of the main README).
Place a detector under `LabGym/LabGym_LabGrymace/detectors/` and a categorizer under
`LabGym/LabGym_LabGrymace/models/`, one folder per model, then re-open the selection dialog.

**A categorizer is rejected as "Unsupported"**
Its `model_parameters.txt` declares a behavior mode this build does not support. Train or
select a categorizer in one of the supported modes.

**LabGym runs on CPU when a GPU is present**
`CUDA_HOME` (or `CUDA_PATH`) must point at your CUDA toolkit before launching. On an HPC
cluster, load the CUDA module first. Note that `analyzebehavior.py` pins categorizer
inference to CPU by design, to avoid CUDA context conflicts with detectron2.
