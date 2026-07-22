# Troubleshooting

**LabGrYMace worked yesterday but breaks today, after installing LabGym**
You almost certainly installed into conda `base` (the prompt showed `(base)`, not your
environment name), and `pip install ./LabGym` downgraded NumPy/TensorFlow in `base`. That
breaks the compiled libraries LabGrYMace relies on (pandas, OpenCV, matplotlib), so it now
fails with errors like `numpy.dtype size changed`. Fix it by using a **dedicated
environment** instead of `base`:

```bash
conda create -n labgrymace python=3.10 -y
conda activate labgrymace
cd LabGym-LabGrYMace
pip install .
pip install ./LabGym
```

To repair the polluted `base` afterwards: `pip uninstall -y LabGrYMace LabGym_LabGrYMace`,
then reinstall whatever `base` needed (or `conda install -n base --force-reinstall numpy`).
The simplest habit is to never `pip install` into `base`.

**`command not found: LabGrYMace` (macOS) / `'LabGrYMace' is not recognized` (Windows)**
The environment isn't active, or the install didn't finish. Re-activate it
(`conda activate labgrymace`, or `source .venv/bin/activate` / `.venv\Scripts\activate`)
and re-run `pip install .`. `python -m LabGrYMace` always works as a fallback.

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

**`LabGym_LabGrYMace` seems not to launch (but `LabGrYMace` does)**
It is launching — it just takes about **10–30 seconds** to open, because it loads
TensorFlow and detectron2 first, while `LabGrYMace` opens instantly. Wait for it, and look
**behind other windows** (the Dock on macOS, the taskbar on Windows). The two are separate
programs: running `LabGrYMace` does not open the LabGym window, and vice versa — start each
one with its own command. If the command is reported as not found, use
`python -m LabGym_LabGrYMace`.

**Which command launches our LabGym?**
Our build installs as the package `LabGym_LabGrYMace` and launches with the command
`LabGym_LabGrYMace` (the plain `LabGym` command belongs to upstream LabGym, if you also
have it installed). See [Two LabGyms, side by side](LABGYM_CHANGES.md#two-labgyms-side-by-side).

**LabGym can't find a detector or categorizer**
Models are not shipped in this repository (see the Models section of the main README).
Place a detector under `LabGym/LabGym_LabGrYMace/detectors/` and a categorizer under
`LabGym/LabGym_LabGrYMace/models/`, one folder per model, then re-open the selection dialog.

**A categorizer is rejected as "Unsupported"**
Its `model_parameters.txt` declares a behavior mode this build does not support. Train or
select a categorizer in one of the supported modes.

**LabGym runs on CPU when a GPU is present**
`CUDA_HOME` (or `CUDA_PATH`) must point at your CUDA toolkit before launching. On an HPC
cluster, load the CUDA module first. Note that `analyzebehavior.py` pins categorizer
inference to CPU by design, to avoid CUDA context conflicts with detectron2.
