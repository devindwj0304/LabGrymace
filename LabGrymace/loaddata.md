# loaddata.py

Builds the **summary files** that the pain-score / analysis stage consumes, from
raw LabGym output.

## Main classes
- `FacialDataIntegrator(base_path)` — merges raw LabGym per-region tracking output
  into integrated data files. `.process_all_datasets()` runs it over every dataset
  subfolder under `base_path`.
- `LoadDataAdvanced(base_path)` — builds the concise `ear_summary.xlsx`,
  `eye_summary.xlsx`, `nose_summary.xlsx` for each dataset (`.generate_all()`).

## Command-line use
```bash
python -m LabGrymace.loaddata "/path/to/folder/of/LabGym/datasets"
```
`base_path` is a positional argument (no longer hardcoded). It must point to a
folder whose subfolders are individual LabGym result datasets.

## Output (per dataset folder)
`ear_summary.xlsx`, `eye_summary.xlsx`, `nose_summary.xlsx` — per-frame intensity /
event columns. These are **complete / unfiltered**; the reflection (mirror) filter
is applied later, at load time in `gui_main.load_raw_data`, not here.

## Notes
- The previous private `__main__` (hardcoded OneDrive path + a `dataset_config`
  diagnostics import) was replaced with the argparse CLI above for portability.
