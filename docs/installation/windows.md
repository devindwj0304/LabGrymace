# Windows

Download the **Windows 64-bit** Python 3.10 installer from <https://www.python.org/downloads/windows/>, tick **"Add python.exe to PATH"** and **"Disable path length limit"**, then use PowerShell.

```powershell
git clone https://github.com/devindwj0304/LabGrymace.git
cd LabGrymace
py -3.10 -m pip install --upgrade pip wheel setuptools
py -3.10 -m pip install .
py -3.10 -m pip install ./LabGym_LabGrymace
```
