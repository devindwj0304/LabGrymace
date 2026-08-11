# Usage

## Launching

Two GUI applications are installed:

- **`LabGrymace`**: computes pain scores from LabGym output.
- **`LabGym_LabGrymace`**: our modified LabGym build that generates the tracking output used by LabGrymace.

Run either command from a terminal:

```bash
LabGrymace
LabGym_LabGrymace
```

If either command is not found, the Python 3.10 scripts folder is not on your PATH. Launch through Python instead. On Windows, use `py -3.10`:

```powershell
py -3.10 -m LabGrymace
py -3.10 -m LabGym_LabGrymace_copy
```

On macOS and Linux, use `python3.10`:

```bash
python3.10 -m LabGrymace
python3.10 -m LabGym_LabGrymace_copy
```

You can also start either application from a Python session. This follows the upstream LabGym method, but uses this build's package names, `LabGrymace` and `LabGym_LabGrymace`, rather than the upstream `LabGym`:

```python
from LabGrymace import __main__; __main__.main()          # the pain tool
from LabGym_LabGrymace_copy import __main__; __main__.main()    # the modified LabGym build
```

```{note}
The two applications run independently. `LabGrymace` opens immediately. `LabGym_LabGrymace` typically takes 10-30 seconds to start the first time because it loads TensorFlow and Detectron2 before the GUI appears.
```
