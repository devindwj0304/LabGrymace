# Linux

Install the system packages, including the GTK libraries that wxPython needs:

```bash
sudo apt update
sudo apt install build-essential git python3.10 python3-pip
sudo apt install libgtk-3-dev libsdl2-2.0-0 libsdl2-image-2.0-0 libsdl2-mixer-2.0-0
```

Then install LabGrymace:

```bash
git clone https://github.com/devindwj0304/LabGrymace.git
cd LabGrymace
python3.10 -m pip install --upgrade pip wheel setuptools
python3.10 -m pip install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04 wxPython
python3.10 -m pip install .
python3.10 -m pip install ./LabGym_LabGrymace
```
