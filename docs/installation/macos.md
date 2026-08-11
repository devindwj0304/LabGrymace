# macOS

Download the **macOS 64-bit universal2** Python 3.10 installer from <https://www.python.org/downloads/macos/> and run it. The universal2 build runs natively on Apple Silicon, so its TensorFlow does not require AVX.

```bash
git clone https://github.com/devindwj0304/LabGrymace.git
cd LabGrymace
python3.10 -m pip install --upgrade pip wheel setuptools
python3.10 -m pip install .
python3.10 -m pip install ./LabGym_LabGrymace
```

```{note}
Do the following only if `LabGym_LabGrymace` aborts during launch with a TensorFlow AVX error. If it launches successfully, skip this step.
```

This error means Python is running under Rosetta rather than natively. Reinstall the universal2 Python 3.10 above and confirm that it reports `arm64`:

```bash
python3.10 -c "import platform; print(platform.machine())"   # must print arm64
```
