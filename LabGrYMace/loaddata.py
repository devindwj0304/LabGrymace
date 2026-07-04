# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np

class FacialDataIntegrator:
    def __init__(self, base_path):
        self.base_path = base_path
        self.output_base = os.path.join(base_path, "Summary of facial expression")
        
        # Define all datasets and their paths (expanded to include new datasets)
        self.datasets = {
            # Original datasets
            'Baseline_4116': os.path.join(base_path, "4116 Baseline Individual"),
            'Low_CNO_KI': os.path.join(base_path, "KI low CNO Individual"),
            # Previously added datasets
            'Baseline_4111': os.path.join(base_path, "4111KI baseline1 individual"),
            'Low_CNO_4112': os.path.join(base_path, "4112KI lowCNO individual"),
            'High_CNO_4113': os.path.join(base_path, "4113KI HighCNO individual"),
            # Datasets from 11.28 Updated data (0.5mg and 1mg CNO doses)
            'M12_0.5': os.path.join(base_path, "11.28 Updated data/M12_0_0-600_processed"),
            'M12_1mg_CNO': os.path.join(base_path, "11.28 Updated data/M12_1mgkgCNOip_0-600_processed"),
            'M13_0.5mg_CNO': os.path.join(base_path, "11.28 Updated data/M13_05mgkgCNO_0-600_processed"),
            'M13_1mg_CNO': os.path.join(base_path, "11.28 Updated data/M13_1mgkgCNOip_0-600_processed"),
            'M2_0.5': os.path.join(base_path, "11.28 Updated data/M2_0_0-600_processed Baseline individual"),
            'M2_1mg_CNO': os.path.join(base_path, "11.28 Updated data/M2_1mgkgCNO_0-600_processed"),
            'M4_0.5': os.path.join(base_path, "11.28 Updated data/M4_0_0-600_processed"),
            'M4_1mg_CNO': os.path.join(base_path, "11.28 Updated data/M4_1mgkgCNOip_0-600_processed"),
 # datasets from 11.29 Update (updated trimmed versions)
            'Low_CNO_4022': os.path.join(base_path, "11.29 Update/4022_lowCNO_trimmed_0-600_processed"),
            'High_CNO_4026_Updated': os.path.join(base_path, "11.29 Update/4026KI_highCNO_trimmed_0-600_processed"),
            'High_CNO_4111_Updated': os.path.join(base_path, "11.29 Update/4111KI_highCNO_trimmed _0-600_processed"),
            'High_CNO_4112_Updated': os.path.join(base_path, "11.29 Update/4112KI_highCNO_trimmed _0-600_processed"),
            'Low_CNO_4113_Updated': os.path.join(base_path, "11.29 Update/4113KI_lowCNO_TRIMMED_0-600_processed"),
            'Low_CNO_4116_Updated': os.path.join(base_path, "11.29 Update/4116KI_lowCNO_trimmed_0-600_processed"),
            # PBSip control group datasets (SEPARATE from Baseline!)
            'M12_PBSip': os.path.join(base_path, "11.29 Update/M12_PBSip_0-600_processed"),
            'M13_PBSip': os.path.join(base_path, "11.29 Update/M13_PBSip_0-600_processed"),
            'M2_PBSip': os.path.join(base_path, "11.29 Update/M2_PBSip_0-600_processed"),
            'M4_PBSip': os.path.join(base_path, "11.29 Update/M4_PBSip_0-600_processed"),
 # datasets from 12.16 updated data
            '4M1_1mgCNO_clip1': os.path.join(base_path, "12.16  updated data/4M1_1mgkgCNO_clip1_0-600_processed"),
            '4M1_1mgCNO_clip2': os.path.join(base_path, "12.16  updated data/4M1_1mgkgCNO_clip2_0-600_processed"),
            '5F1_1mgCNO': os.path.join(base_path, "12.16  updated data/5F1_1mgkgCNO_0-600_processed"),
            '5F2_1mgCNO': os.path.join(base_path, "12.16  updated data/5F2_1mgkgCNO_0-600_processed"),
            'Female1_MgSO4': os.path.join(base_path, "12.16  updated data/Female1_125mgkgMgSO4_0-600_processed"),
            'Female2_MgSO4': os.path.join(base_path, "12.16  updated data/Female2_MgSO4_125mgkg_0-600_processed"),
            'Male1_MgSO4': os.path.join(base_path, "12.16  updated data/Male1_MgSO4_125mgkg_0-600_processed"),
 # datasets from 12.23 Update MgSO4
            'Female1_125mgkgMgSO4_new': os.path.join(base_path, "12.23 Update MgSO4/Female1_125mgkgMgSO4_0-600_processed"),
            'Female1_250mgkgMgSO4': os.path.join(base_path, "12.23 Update MgSO4/Female1_250mgkgMgSO4_0-600_processed"),
            'Female2_250mgkgMgSO4': os.path.join(base_path, "12.23 Update MgSO4/Female2_250mgkgMgSO4_600-1200_processed"),
            'Female2_MgSO4_125mgkg_new': os.path.join(base_path, "12.23 Update MgSO4/Female2_MgSO4_125mgkg_0-600_processed"),
            'Female3_125mgSO4': os.path.join(base_path, "12.23 Update MgSO4/Female3_125mgSO4_0-600_processed"),
            'Female3_62': os.path.join(base_path, "12.23 Update MgSO4/Female3_62_0-600_processed"),
            'Male1_250mgkgMgSO4': os.path.join(base_path, "12.23 Update MgSO4/Male1_250mgkgMgSO4_0-600_processed"),
            'Male1_MgSO4_125mgkg_new': os.path.join(base_path, "12.23 Update MgSO4/Male1_MgSO4_125mgkg_0-600_processed"),
            'Male2_125mgkgMgSO4': os.path.join(base_path, "12.23 Update MgSO4/Male2_125mgkgMgSO4_0-600_processed"),
            'Male2_62': os.path.join(base_path, "12.23 Update MgSO4/Male2_62_0-600_processed"),
            'Male3_62': os.path.join(base_path, "12.23 Update MgSO4/Male3_62_0-300_ 600-900_processed"),
            'Male3_125mgkgMgSO4': os.path.join(base_path, "12.23 Update MgSO4/Male3_125mgkgMgSO4 (1)_600-1200_processed"),

            # Capsaicin intraplantar injection datasets (1.8 capsaicin result)
            # Female mouse 2-12F
            'Capsaicin_2-12F_125ugmL': os.path.join(base_path, "1.8 capsaicin result/2-12F-intraplantar_125ugmLcap_0-600_processed"),
            'Capsaicin_2-12F_vehicle': os.path.join(base_path, "1.8 capsaicin result/2-12F_intraplantar_vehicle_0-600_processed"),
            # Female mouse 2-24F
            'Capsaicin_2-24F_125ugmL': os.path.join(base_path, "1.8 capsaicin result/2-24F-intraplantar_125ugmL_cap_0-600_processed"),
            'Capsaicin_2-24F_vehicle': os.path.join(base_path, "1.8 capsaicin result/2-24F_intraplantar_vehicle_0-600_processed"),
            # Male mouse 2_34M
            'Capsaicin_2-34M_vehicle': os.path.join(base_path, "1.8 capsaicin result/2_34M_intraplantar_vehicle_0-600_processed"),

 # capsaicin datasets from 2.3 update capsaicin Video Analysis
            # Female mouse 2-0F
            'Capsaicin_2-0F_125ugmL': os.path.join(base_path, "2.3 update capsaicin Video Analysis/2-0F-intraplantar_125ugmLcap_0-600_processed"),
            'Capsaicin_2-0F_vehicle': os.path.join(base_path, "2.3 update capsaicin Video Analysis/2-0F_intraplantar_vehicle_0-600_processed"),
            # Male mouse 2-0M
            'Capsaicin_2-0M_125ugmL': os.path.join(base_path, "2.3 update capsaicin Video Analysis/2-0M_intraplantar_125ugmLcap_0-600_processed"),
            'Capsaicin_2-0M_vehicle': os.path.join(base_path, "2.3 update capsaicin Video Analysis/2_0M_intraplantar_vehicle_0-600_processed"),
            # Male mouse 2-13M
            'Capsaicin_2-13M_125ugmL': os.path.join(base_path, "2.3 update capsaicin Video Analysis/2-13M_intraplantar_125ugmL_cap_0-600_processed"),
            'Capsaicin_2-13M_vehicle': os.path.join(base_path, "2.3 update capsaicin Video Analysis/2_13M_intraplantar_vehicle_0-600_processed"),
            # Male mouse 2-34M (capsaicin treatment)
            'Capsaicin_2-34M_125ugmL': os.path.join(base_path, "2.3 update capsaicin Video Analysis/2-34M_intraplantar_125ugmL_cap_0-600_processed"),

 # true baseline datasets from 2.26 (Analyze videos true baseline 2.26)
            '9F1_true_baseline': os.path.join(base_path, "../Analyze videos true baseline 2.26/9F1_true-baseline_0-600_processed"),
            '9F2_true_baseline_escape': os.path.join(base_path, "../Analyze videos true baseline 2.26/9F2_true-baseline-escape_0-600_processed"),
            # 9M1 has an extra nested folder layer
            '9M1_true_baseline': os.path.join(base_path, "../Analyze videos true baseline 2.26/9M1_true-baseline_0-600_processed/9M1_true-baseline_0-600_processed"),
            '9M3_true_baseline': os.path.join(base_path, "../Analyze videos true baseline 2.26/9M3_true-baseline_0-600_processed"),
        }
        
        # Eye behavior folders
        self.eye_behaviors = ['fEyeBL', 'fEyeOT', 'sEyeBL', 'sEyeOT']
        
        # Ear behavior folders
        self.ear_behaviors = ['fEarBL', 'fEarPB', 'sEarBL', 'sEarPB']
        
        # Nose behavior folders
        self.nose_behaviors = ['noseBL', 'noseBul']
        
        # Parameter files for each facial feature (excluding summary)
        self.eye_parameters = [
            'eyes_acceleration.xlsx',
            'eyes_intensity_area.xlsx',
            'eyes_intensity_length.xlsx',
            'eyes_magnitude_area.xlsx',
            'eyes_magnitude_length.xlsx',
            'eyes_probability.xlsx',
            'eyes_speed.xlsx',
            'eyes_velocity.xlsx',
            'eyes_vigor_area.xlsx',
            'eyes_vigor_length.xlsx'
        ]
        
        self.ear_parameters = [
            'ears_acceleration.xlsx',
            'ears_intensity_area.xlsx',
            'ears_intensity_length.xlsx',
            'ears_magnitude_area.xlsx',
            'ears_magnitude_length.xlsx',
            'ears_probability.xlsx',
            'ears_speed.xlsx',
            'ears_velocity.xlsx',
            'ears_vigor_area.xlsx',
            'ears_vigor_length.xlsx'
        ]
        
        self.nose_parameters = [
            'nose_acceleration.xlsx',
            'nose_intensity_area.xlsx',
            'nose_intensity_length.xlsx',
            'nose_magnitude_area.xlsx',
            'nose_magnitude_length.xlsx',
            'nose_probability.xlsx',
            'nose_speed.xlsx',
            'nose_velocity.xlsx',
            'nose_vigor_area.xlsx',
            'nose_vigor_length.xlsx'
        ]
    
    def load_behavior_data(self, dataset_path, behavior_name, feature_type, param_files):
        print(f"\n=== Loading {behavior_name} {feature_type} data ===")
        
        behavior_path = os.path.join(dataset_path, behavior_name)
        behavior_data = {}
        
        for param_file in param_files:
            file_path = os.path.join(behavior_path, param_file)
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_excel(file_path)
                    param_name = param_file.replace(f'{feature_type}_', '').replace('.xlsx', '')
                    behavior_data[param_name] = df
                    print(f"  Loaded {param_name}: {df.shape}")
                except Exception as e:
                    print(f"  Error loading {param_file}: {e}")
            else:
                print(f"  File not found: {param_file}")
        
        return behavior_data
    
    def check_time_alignment(self, all_data):
        print("\n=== Checking time alignment ===")
        
        # Check if all data has consistent time length
        for behavior in all_data:
            print(f"\n{behavior}:")
            for param in all_data[behavior]:
                df = all_data[behavior][param]
                if df is not None and not df.empty:
                    print(f"  {param}: {df.shape[0]} time points")
    
    def integrate_feature_data(self, dataset_name, feature_type):
        print(f"Starting {dataset_name} {feature_type} data integration...")
        
        dataset_path = self.datasets[dataset_name]
        all_data = {}
        
        # Select appropriate behaviors and parameters
        if feature_type == 'eyes':
            behaviors = self.eye_behaviors
            param_files = self.eye_parameters
        elif feature_type == 'ears':
            behaviors = self.ear_behaviors
            param_files = self.ear_parameters
        elif feature_type == 'nose':
            behaviors = self.nose_behaviors
            param_files = self.nose_parameters
        else:
            print(f"Unknown feature type: {feature_type}")
            return {}
        
        # Load data for each behavior
        for behavior in behaviors:
            all_data[behavior] = self.load_behavior_data(dataset_path, behavior, feature_type, param_files)
        
        # Check time alignment
        self.check_time_alignment(all_data)
        
        return all_data
    
    def create_integrated_excel(self, all_data, dataset_name, feature_type):
        print(f"\n=== Creating integrated {dataset_name} {feature_type} Excel file ===")
        
        # Create dataset-specific output directory
        dataset_output_path = os.path.join(self.output_base, dataset_name)
        os.makedirs(dataset_output_path, exist_ok=True)
        
        # Create Excel writer with output path
        output_file = os.path.join(dataset_output_path, f'integrated_{feature_type}_data.xlsx')
        
        # Check if we have any valid data before creating Excel file
        has_valid_data = False
        for behavior in all_data:
            if all_data[behavior] and any(df is not None and not df.empty for df in all_data[behavior].values()):
                has_valid_data = True
                break
        
        if not has_valid_data:
            print(f"No valid data found for {dataset_name} {feature_type}. Skipping Excel creation.")
            return None
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Create a worksheet for each behavior
            for behavior in all_data:
                print(f"Creating sheet for {behavior}...")
                
                # Get all parameter data for this behavior
                behavior_data = all_data[behavior]
                
                if behavior_data and any(df is not None and not df.empty for df in behavior_data.values()):
                    # Find first valid dataframe to get time length
                    first_valid_df = None
                    for param_name, df in behavior_data.items():
                        if df is not None and not df.empty:
                            first_valid_df = df
                            break
                    
                    if first_valid_df is not None:
                        # Create integrated dataframe
                        integrated_df = pd.DataFrame()
                        
                        # Keep time column if first column is time
                        if len(first_valid_df.columns) > 0:
                            time_col = first_valid_df.iloc[:, 0]
                            integrated_df['Time'] = time_col
                        
                        # Add data for each parameter
                        for param_name, df in behavior_data.items():
                            if df is not None and not df.empty:
                                # Assume columns from index 1 onwards are data columns
                                for col_idx in range(1, len(df.columns)):
                                    col_name = f"{param_name}_{df.columns[col_idx]}"
                                    integrated_df[col_name] = df.iloc[:, col_idx]
                        
                        # Write to Excel worksheet
                        integrated_df.to_excel(writer, sheet_name=behavior, index=False)
                        print(f"  {behavior} sheet created with {integrated_df.shape[0]} rows, {integrated_df.shape[1]} columns")
        
        print(f"\nIntegrated {dataset_name} {feature_type} Excel file saved as: {output_file}")
        return output_file
    
    def check_integrated_files_exist(self, dataset_name):
        """Check if all integrated data files already exist for a dataset."""
        output_dir = os.path.join(self.output_base, dataset_name)

        # Check for all three integrated files
        files_to_check = [
            os.path.join(output_dir, "integrated_eyes_data.xlsx"),
            os.path.join(output_dir, "integrated_ears_data.xlsx"),
            os.path.join(output_dir, "integrated_nose_data.xlsx")
        ]

        return all(os.path.exists(f) for f in files_to_check)

    def process_all_datasets(self):
        print("Processing all datasets and facial features...")

        results = {}

        for dataset_name in self.datasets.keys():
            print(f"\n{'='*50}")
            print(f"Processing {dataset_name} dataset")
            print(f"{'='*50}")

            # Check if all integrated files already exist
            if self.check_integrated_files_exist(dataset_name):
                print(f" All integrated files already exist for {dataset_name}, skipping...")
                results[dataset_name] = {
                    'eyes': os.path.join(self.output_base, dataset_name, "integrated_eyes_data.xlsx"),
                    'ears': os.path.join(self.output_base, dataset_name, "integrated_ears_data.xlsx"),
                    'nose': os.path.join(self.output_base, dataset_name, "integrated_nose_data.xlsx")
                }
                continue

            results[dataset_name] = {}

            # Process each facial feature for this dataset
            for feature_type in ['eyes', 'ears', 'nose']:
                print(f"\n--- Processing {feature_type} for {dataset_name} ---")

                # Integrate feature data
                feature_data = self.integrate_feature_data(dataset_name, feature_type)

                # Create Excel file
                if feature_data and any(data for data in feature_data.values()):
                    output_file = self.create_integrated_excel(feature_data, dataset_name, feature_type)
                    results[dataset_name][feature_type] = output_file
                else:
                    print(f"No {feature_type} data found for {dataset_name}")
                    results[dataset_name][feature_type] = None

        return results


class LoadDataAdvanced:
    """Build concise eye/ear/nose summaries next to existing integrated files.

    Output per dataset (under Summary of facial expression/<Dataset_Key>):
      - eye_summary.xlsx (Time, Eye0Pos, Eye1Pos, Eye0Event, Eye1Event, Acceleration 0/1, plus other LabGym params)
      - ear_summary.xlsx  (Time, Ear0Pos, Ear1Pos, Ear0Event, Ear1Event, Acceleration 0/1, plus other LabGym params)
      - nose_summary.xlsx (Time, NoseEvent, Acceleration, plus other LabGym params)

    Labeling rules (strict, no defaults):
      - At each time, detect which integrated sheet has any parameter value:
          eyes: sEyeOT/sEyeBL/fEyeOT/fEyeBL
          ears: sEarPB/sEarBL/fEarPB/fEarBL
          nose: noseBul/noseBL
      - If exactly one sheet is active: derive Pos/Event directly from the sheet name (f/s, PB/BL, OT/BL, Bul/BL).
      - If none are active: leave Pos/Event blank.
      - If more than one are active: leave Pos/Event blank and record an error row in Behavior_Conflicts.

    Parameters: copied from integrated sheets (side/front kept by precedence rules when fusing numeric values),
    but Pos/Event labels are only sheet-driven as above (no inference, no fill).
    """

    def __init__(self, base_path):
        self.base_path = base_path
        self.summary_root = os.path.join(base_path, "Summary of facial expression")
        # Additional LabGym parameters to include besides acceleration
        self.param_keys = [
            'intensity_area', 'intensity_length',
            'magnitude_area', 'magnitude_length',
            'probability', 'speed', 'velocity',
            'vigor_area', 'vigor_length'
        ]

    # ---------- filesystem helpers ----------
    def _resolve_folder(self, base, name):
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
        try:
            for n in os.listdir(base):
                if n.lower() == name.lower():
                    alt = os.path.join(base, n)
                    print(f"Resolved folder '{name}' -> '{n}' in {base}")
                    return alt
        except Exception:
            pass
        print(f"Folder not found: {path}")
        return None

    # ---------- helpers ----------
    def _standardize_summary_columns(self, df, feature):
        if df is None or df.empty:
            return df
        import re
        col_map = {}

        # Normalizer: lowercase + strip non-alphanum
        def norm(s):
            return re.sub(r'[^a-z0-9]', '', str(s).lower())

        cols_norm = {norm(c): c for c in df.columns}

        # Time (more tolerant)
        for key in ['time', 'timestamp', 'frame', 'frames', 'times', 'timeS']:
            k = norm(key)
            if k in cols_norm:
                col_map[cols_norm[k]] = 'Time'
                break

        # Feature-specific tokens
        f = feature.lower()
        tokens = []
        if f == 'eyes':
            tokens = [('eye0pos', 'Eye0Pos'), ('eye1pos', 'Eye1Pos'),
                      ('eye0event', 'Eye0Event'), ('eye1event', 'Eye1Event')]
        elif f == 'ears':
            tokens = [('ear0pos', 'Ear0Pos'), ('ear1pos', 'Ear1Pos'),
                      ('ear0event', 'Ear0Event'), ('ear1event', 'Ear1Event')]
        else:
            tokens = [('nosepos', 'NosePos'), ('noseevent', 'NoseEvent')]

        for key, target in tokens:
            if key in cols_norm:
                col_map[cols_norm[key]] = target
        # Partial-match fallback (handles headers like 'Eye0PosforS')
        if feature.lower() == 'eyes':
            for nk, orig in cols_norm.items():
                if 'eye0pos' in nk and 'event' not in nk:
                    col_map.setdefault(orig, 'Eye0Pos')
                if 'eye1pos' in nk and 'event' not in nk:
                    col_map.setdefault(orig, 'Eye1Pos')
                if 'eye0event' in nk:
                    col_map.setdefault(orig, 'Eye0Event')
                if 'eye1event' in nk:
                    col_map.setdefault(orig, 'Eye1Event')
        elif feature.lower() == 'ears':
            for nk, orig in cols_norm.items():
                if 'ear0pos' in nk and 'event' not in nk:
                    col_map.setdefault(orig, 'Ear0Pos')
                if 'ear1pos' in nk and 'event' not in nk:
                    col_map.setdefault(orig, 'Ear1Pos')
                if 'ear0event' in nk:
                    col_map.setdefault(orig, 'Ear0Event')
                if 'ear1event' in nk:
                    col_map.setdefault(orig, 'Ear1Event')
        else:
            for nk, orig in cols_norm.items():
                if 'nosepos' in nk and 'event' not in nk:
                    col_map.setdefault(orig, 'NosePos')
                if 'noseevent' in nk:
                    col_map.setdefault(orig, 'NoseEvent')

        if col_map:
            try:
                df = df.rename(columns=col_map)
            except Exception:
                pass

        # Ensure a Time column exists; if not, try first column fallback
        if 'Time' not in df.columns and len(df.columns) > 0:
            first = df.columns[0]
            try:
                # If numeric monotonic, treat as time
                s = pd.to_numeric(df[first], errors='coerce')
                if s.notna().sum() > 0:
                    df = df.rename(columns={first: 'Time'})
            except Exception:
                pass
        return df

    def _load_summary(self, feature, folder):
        if not folder or not os.path.exists(folder):
            print(f"Summary folder missing: {folder}")
            return None
        # Be robust to filename variations; search case-insensitively
        target = f"{feature}_all_summary.xlsx".lower()
        cand = None
        try:
            for fn in os.listdir(folder):
                if fn.lower() == target:
                    cand = os.path.join(folder, fn)
                    break
            if cand is None:
                # fallback: contains all and summary keywords
                for fn in os.listdir(folder):
                    l = fn.lower()
                    if l.endswith('.xlsx') and feature in l and 'summary' in l and 'all' in l:
                        cand = os.path.join(folder, fn)
                        break
        except Exception:
            cand = None
        if cand is None:
            print(f"Summary file missing in folder: {folder} (looking for {target})")
            return None
        try:
            df = pd.read_excel(cand)
            df = self._standardize_summary_columns(df, feature)
            if df is not None:
                print(f"Loaded summary: {cand} | cols: {list(df.columns)}")
            return df
        except Exception as e1:
            try:
                df = pd.read_excel(cand, engine='openpyxl')
                df = self._standardize_summary_columns(df, feature)
                if df is not None:
                    print(f"Loaded summary (openpyxl): {cand} | cols: {list(df.columns)}")
                return df
            except Exception as e2:
                print(f"Failed to load {cand}: {e1 or e2}")
                return None

    def _merge_time(self, *dfs):
        times = None
        for df in dfs:
            if df is not None and 'Time' in df.columns:
                t = pd.Series(df['Time']).dropna().unique()
                times = np.union1d(times, t) if times is not None else t
        if times is None:
            return None
        return pd.Index(sorted(times))

    def _time_index_from_accel(self, *acc_dfs):
        times = None
        for df in acc_dfs:
            if df is not None and 'Time' in df.columns:
                t = pd.Series(df['Time']).dropna().unique()
                times = np.union1d(times, t) if times is not None else t
        return None if times is None else pd.Index(sorted(times))

    def _coalesce(self, idx, preferred, fallback):
        s = pd.Series(index=idx, dtype=float)
        if preferred is not None:
            preferred = preferred.reindex(idx)
        if fallback is not None:
            fallback = fallback.reindex(idx)
        for t in idx:
            pv = preferred.loc[t] if preferred is not None else np.nan
            fv = fallback.loc[t] if fallback is not None else np.nan
            s.loc[t] = pv if pd.notna(pv) else fv
        return s

    def _coalesce_with_conflicts(self, idx, preferred, fallback, *,
                                 dataset_key, feature, field, rule,
                                 tol=1e-9):
        s = pd.Series(index=idx, dtype=float)
        conflicts = []
        if preferred is not None:
            preferred = preferred.reindex(idx)
        if fallback is not None:
            fallback = fallback.reindex(idx)
        for t in idx:
            pv = preferred.loc[t] if preferred is not None else np.nan
            fv = fallback.loc[t] if fallback is not None else np.nan
            if pd.notna(pv):
                s.loc[t] = pv
                if pd.notna(fv):
                    try:
                        dp = float(pv); df = float(fv)
                        if abs(dp - df) > tol:
                            conflicts.append({'Dataset': dataset_key, 'Feature': feature, 'Field': field,
                                              'Time': t, 'Preferred': dp, 'Fallback': df, 'Diff': dp-df, 'Rule': rule})
                    except Exception:
                        if str(pv) != str(fv):
                            conflicts.append({'Dataset': dataset_key, 'Feature': feature, 'Field': field,
                                              'Time': t, 'Preferred': pv, 'Fallback': fv, 'Diff': None, 'Rule': rule})
            elif pd.notna(fv):
                s.loc[t] = fv
            else:
                s.loc[t] = np.nan
        return s, conflicts

    def _load_accel(self, feature, folder):
        if not folder or not os.path.exists(folder):
            return None
        fpath = os.path.join(folder, f"{feature}_acceleration.xlsx")
        if not os.path.exists(fpath):
            return None
        try:
            df = pd.read_excel(fpath, engine='openpyxl')
        except Exception:
            return None
        # Identify time + two numeric columns for channels 0/1
        time_col = None
        for c in df.columns:
            if str(c).lower() == 'time':
                time_col = c; break
        if time_col is None:
            time_col = df.columns[0]
        num_cols = [c for c in df.columns if c != time_col]
        if len(num_cols) == 0:
            return None
        # Try to pick _0/_1 columns; else first two if available
        c0 = next((c for c in num_cols if str(c).endswith('_0') or str(c).endswith('0')), num_cols[0])
        c1 = next((c for c in num_cols if str(c).endswith('_1') or str(c).endswith('1') and c != c0), (num_cols[1] if len(num_cols)>1 else None))
        res = {'Time': df[time_col]}
        res['ch0'] = df[c0]
        if c1 is not None:
            res['ch1'] = df[c1]
        return pd.DataFrame(res)

    # Read acceleration from integrated_*_data.xlsx sheet
    def _load_accel_from_integrated(self, dataset_key, feature, behavior):
        feat = feature.lower()
        integ_path = os.path.join(self.summary_root, dataset_key, f'integrated_{feat}_data.xlsx')
        if not os.path.exists(integ_path):
            print(f"Integrated file missing: {integ_path}")
            return None
        try:
            xls = pd.ExcelFile(integ_path)
            if behavior not in xls.sheet_names:
                print(f"Integrated sheet missing: {integ_path} :: {behavior}. Sheets={xls.sheet_names}")
                return None
            df = pd.read_excel(integ_path, sheet_name=behavior)
        except Exception:
            return None

        # Find Time column (prefer exact match)
        time_col = None
        for c in df.columns:
            if str(c).strip().lower() == 'time':
                time_col = c; break
        if time_col is None:
            time_col = df.columns[0]

        # Heuristic: pick columns that contain 'accel' and end with _0/_1 or have 0/1 token
        cols_lower = {c: str(c).lower() for c in df.columns}
        acc_cols = [c for c in df.columns if 'accel' in cols_lower[c]]
        if not acc_cols:
            # try exact names like 'acceleration_0'
            acc_cols = [c for c in df.columns if str(c).lower().startswith('acceleration')]
        if not acc_cols:
            return None

        def pick_channel(columns, ch):
            # look for *_<ch>
            for c in columns:
                lc = str(c).lower()
                if lc.endswith(f'_{ch}') or lc.endswith(str(ch)):
                    return c
            # fallback: if only one column, return it for ch0
            if ch == 0 and columns:
                return columns[0]
            return None

        c0 = pick_channel(acc_cols, 0)
        c1 = pick_channel(acc_cols, 1)
        out = {'Time': df[time_col]}
        if c0 is not None:
            out['ch0'] = df[c0]
        if c1 is not None:
            out['ch1'] = df[c1]
        return pd.DataFrame(out)

    def _load_param_from_integrated(self, dataset_key, feature, behavior, param_key):
        """Load a generic dual-organ parameter from integrated_{feature}_data.xlsx sheet.
        Returns a DataFrame with columns: Time, ch0 (optional), ch1 (optional)."""
        feat = feature.lower()
        integ_path = os.path.join(self.summary_root, dataset_key, f'integrated_{feat}_data.xlsx')
        if not os.path.exists(integ_path):
            return None
        try:
            xls = pd.ExcelFile(integ_path)
            if behavior not in xls.sheet_names:
                return None
            df = pd.read_excel(integ_path, sheet_name=behavior)
        except Exception:
            return None

        # find time
        time_col = None
        for c in df.columns:
            if str(c).strip().lower() == 'time':
                time_col = c; break
        if time_col is None:
            time_col = df.columns[0]

        # pick parameter columns by prefix match like '<param_key>_0', '<param_key>_1'
        cols_lower = {c: str(c).lower() for c in df.columns}
        candidates = [c for c in df.columns if cols_lower[c].startswith(f"{param_key.lower()}_")]
        if not candidates:
            # fallback: contains param key
            candidates = [c for c in df.columns if param_key.lower() in cols_lower[c]]
        if not candidates:
            return None

        def pick_channel(columns, ch):
            for c in columns:
                lc = str(c).lower()
                if lc.endswith(f'_{ch}') or lc.endswith(str(ch)):
                    return c
            if ch == 0 and columns:
                return columns[0]
            return None

        c0 = pick_channel(candidates, 0)
        c1 = pick_channel(candidates, 1)
        out = {'Time': df[time_col]}
        if c0 is not None:
            out['ch0'] = df[c0]
        if c1 is not None:
            out['ch1'] = df[c1]
        return pd.DataFrame(out)

    def _ffill_label_series(self, s):
        """Forward-fill string labels: treat '' as NaN, ffill, then replace remaining NaN with ''."""
        if s is None:
            return s
        ss = s.copy()
        ss = ss.replace('', np.nan)
        ss = ss.ffill()
        ss = ss.fillna('')
        return ss

    def _ffill_bfill_labels(self, s):
        if s is None:
            return s
        ss = s.copy()
        ss = ss.replace('', np.nan)
        ss = ss.ffill().bfill()
        ss = ss.fillna('')
        return ss

    def _presence_from_integrated_sheet(self, dataset_key, feature, behavior):
        """Return boolean Series (indexed by Time) indicating the row has any
        meaningful parameter value in that sheet.
        Rules:
          - Numeric columns: count as present only if value != 0 (NaN -> False)
          - Non-numeric columns: non-empty string counts as present
          - Time column is excluded
        """
        feat = feature.lower()
        integ_path = os.path.join(self.summary_root, dataset_key, f'integrated_{feat}_data.xlsx')
        if not os.path.exists(integ_path):
            return None
        try:
            xls = pd.ExcelFile(integ_path)
            if behavior not in xls.sheet_names:
                return None
            df = pd.read_excel(integ_path, sheet_name=behavior)
        except Exception:
            return None
        if df is None or df.empty:
            return None
        # Time column
        time_col = None
        for c in df.columns:
            if str(c).strip().lower() == 'time':
                time_col = c; break
        if time_col is None:
            time_col = df.columns[0]
        # Exclude Time + any 'probability' columns from presence decision
        data_cols = [c for c in df.columns if c != time_col and 'probability' not in str(c).lower()]
        if not data_cols:
            return None
        # Build presence by scanning columns with type-aware rules
        presence = pd.Series(False, index=df.index)
        for c in data_cols:
            s = df[c]
            # try numeric evaluation
            is_num = pd.api.types.is_numeric_dtype(s)
            if is_num:
                presence = presence | (s.fillna(0) != 0)
            else:
                presence = presence | (s.astype(str).str.strip() != '') & s.notna()
        presence.index = df[time_col]
        return presence.astype(bool)

    # _sheet_presence_scores removed (no longer needed)

    def _presence_by_channel(self, dataset_key, feature, behavior):
        """Return (ch0_presence, ch1_presence) boolean Series (indexed by Time)
        indicating per-channel activity for a given integrated sheet.
        Probability columns are excluded. Numeric zeros are treated as no value.
        """
        feat = feature.lower()
        integ_path = os.path.join(self.summary_root, dataset_key, f'integrated_{feat}_data.xlsx')
        if not os.path.exists(integ_path):
            return None, None
        try:
            xls = pd.ExcelFile(integ_path)
            if behavior not in xls.sheet_names:
                return None, None
            df = pd.read_excel(integ_path, sheet_name=behavior)
        except Exception:
            return None, None
        if df is None or df.empty:
            return None, None

        # Time and data cols
        time_col = None
        for c in df.columns:
            if str(c).strip().lower() == 'time':
                time_col = c; break
        if time_col is None:
            time_col = df.columns[0]
        data_cols = [c for c in df.columns if c != time_col and 'probability' not in str(c).lower()]
        if not data_cols:
            return pd.Series(False, index=df[time_col]), pd.Series(False, index=df[time_col])

        ch0 = pd.Series(False, index=df.index)
        ch1 = pd.Series(False, index=df.index)

        # Robust channel detector: supports patterns like
        #  _0, _1, 0_mean, 1_mean, ch0/ch1, left/right, l/r tokens
        import re
        def channel_of(col_name):
            n = str(col_name).lower().strip()
            # obvious numeric suffixes
            if re.search(r'(^|[_\s-])0($|[_\s-])', n) or re.search(r'(^|[_\s-])0(mean|max|min)?($|[_\s-])', n):
                return 0
            if re.search(r'(^|[_\s-])1($|[_\s-])', n) or re.search(r'(^|[_\s-])1(mean|max|min)?($|[_\s-])', n):
                return 1
            # ch0/ch1
            if re.search(r'(^|[_\s-])ch?0($|[_\s-])', n):
                return 0
            if re.search(r'(^|[_\s-])ch?1($|[_\s-])', n):
                return 1
            # left/right tokens (avoid matching words like length)
            if re.search(r'(^|[_\s-])(left|l)([_\s-]|$)', n) and 'length' not in n:
                return 0
            if re.search(r'(^|[_\s-])(right|r)([_\s-]|$)', n):
                return 1
            return None

        for c in data_cols:
            s = df[c]
            # value present for this column
            if pd.api.types.is_numeric_dtype(s):
                present = (s.fillna(0) != 0)
            else:
                present = (s.astype(str).str.strip() != '') & s.notna()

            ch = channel_of(c)
            if ch == 0:
                ch0 = ch0 | present
            elif ch == 1:
                ch1 = ch1 | present
            else:
                ch0 = ch0 | present
                ch1 = ch1 | present
        # Map to Time index
        ch0.index = df[time_col]
        ch1.index = df[time_col]
        return ch0.astype(bool), ch1.astype(bool)

    def _sheet_nonzero_counts(self, dataset_key, feature, behavior):
        """Return a Series (indexed by Time) with count of nonzero/nonempty
        parameter values in that sheet row. Probability columns are excluded.
        """
        feat = feature.lower()
        integ_path = os.path.join(self.summary_root, dataset_key, f'integrated_{feat}_data.xlsx')
        if not os.path.exists(integ_path):
            return None
        try:
            xls = pd.ExcelFile(integ_path)
            if behavior not in xls.sheet_names:
                return None
            df = pd.read_excel(integ_path, sheet_name=behavior)
        except Exception:
            return None
        if df is None or df.empty:
            return None
        time_col = None
        for c in df.columns:
            if str(c).strip().lower() == 'time':
                time_col = c; break
        if time_col is None:
            time_col = df.columns[0]
        data_cols = [c for c in df.columns if c != time_col and 'probability' not in str(c).lower()]
        if not data_cols:
            s = pd.Series(0, index=df[time_col])
            s.index = df[time_col]
            return s
        mask = pd.DataFrame(False, index=df.index, columns=data_cols)
        for c in data_cols:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                mask[c] = s.fillna(0) != 0
            else:
                mask[c] = (s.astype(str).str.strip() != '') & s.notna()
        scores = mask.sum(axis=1)
        scores.index = df[time_col]
        return scores

    def _prefer_df(self, *candidates):
        """Return the first non-None DataFrame candidate (non-empty preferred)."""
        for d in candidates:
            if d is not None:
                return d
        return None

    # ---------- diagnostics ----------
    def debug_one(self, dataset_key):
        print("\n==== DEBUG DATASET ====\n", dataset_key)
        # Resolve raw path from DatasetConfig
        from dataset_config import DatasetConfig
        cfg = DatasetConfig(self.base_path)
        ds_map, _ = cfg.get_working_datasets()
        if dataset_key not in ds_map:
            print(f"Dataset key not found in config. Available: {list(ds_map.keys())}")
            return
        raw_path = ds_map[dataset_key]
        print(f"Raw path: {raw_path}")

        def pcols(df):
            return list(df.columns) if df is not None else None

        # Eyes
        print("\n-- EYES --")
        eb = ['sEyeBL', 'fEyeBL', 'sEyeOT', 'fEyeOT']
        eyes_summaries = []
        for b in eb:
            folder = self._resolve_folder(raw_path, b)
            df = self._load_summary('eyes', folder)
            eyes_summaries.append(df)
            print(f"Behavior {b}: folder={folder}")
            print(f"  summary cols: {pcols(df)}")
            acc = self._load_accel_from_integrated(dataset_key, 'eyes', b)
            print(f"  accel cols (integrated): {pcols(acc)}")
        idx = self._merge_time(*eyes_summaries)
        print(f"Eyes time index length: {0 if idx is None else len(idx)}")

        # Ears
        print("\n-- EARS --")
        erb = ['sEarBL', 'fEarBL', 'sEarPB', 'fEarPB']
        ears_summaries = []
        for b in erb:
            folder = self._resolve_folder(raw_path, b)
            df = self._load_summary('ears', folder)
            ears_summaries.append(df)
            print(f"Behavior {b}: folder={folder}")
            print(f"  summary cols: {pcols(df)}")
            acc = self._load_accel_from_integrated(dataset_key, 'ears', b)
            print(f"  accel cols (integrated): {pcols(acc)}")
        idx = self._merge_time(*ears_summaries)
        print(f"Ears time index length: {0 if idx is None else len(idx)}")

        # Nose
        print("\n-- NOSE --")
        nb = ['noseBL', 'noseBul']
        nose_summaries = []
        for b in nb:
            folder = self._resolve_folder(raw_path, b)
            df = self._load_summary('nose', folder)
            nose_summaries.append(df)
            print(f"Behavior {b}: folder={folder}")
            print(f"  summary cols: {pcols(df)}")
            acc = self._load_accel_from_integrated(dataset_key, 'nose', b)
            print(f"  accel cols (integrated): {pcols(acc)}")
        idx = self._merge_time(*nose_summaries)
        print(f"Nose time index length: {0 if idx is None else len(idx)}")

    # ---------- builders ----------
    def build_eye_summary(self, dataset_key, dataset_raw_path):
        out_dir = os.path.join(self.summary_root, dataset_key)
        os.makedirs(out_dir, exist_ok=True)

        # folders
        def resolve_folder(base, name):
            path = os.path.join(base, name)
            if os.path.exists(path):
                return path
            # try case-insensitive match
            try:
                names = os.listdir(base)
                for n in names:
                    if n.lower() == name.lower():
                        alt = os.path.join(base, n)
                        print(f"Resolved folder '{name}' -> '{n}' in {base}")
                        return alt
            except Exception:
                pass
            print(f"Folder not found: {path}")
            return None

        s_bl = resolve_folder(dataset_raw_path, 'sEyeBL')
        f_bl = resolve_folder(dataset_raw_path, 'fEyeBL')
        s_ot = resolve_folder(dataset_raw_path, 'sEyeOT')
        f_ot = resolve_folder(dataset_raw_path, 'fEyeOT')

        s_bl_df = self._load_summary('eyes', s_bl)
        f_bl_df = self._load_summary('eyes', f_bl)
        s_ot_df = self._load_summary('eyes', s_ot)
        f_ot_df = self._load_summary('eyes', f_ot)

        # Build primary time index from integrated acceleration (frame-level),
        # then union with summary times to ensure full coverage
        acc_s_ot_idx_src = self._load_accel_from_integrated(dataset_key, 'eyes', 'sEyeOT')
        acc_s_bl_idx_src = self._load_accel_from_integrated(dataset_key, 'eyes', 'sEyeBL')
        acc_f_ot_idx_src = self._load_accel_from_integrated(dataset_key, 'eyes', 'fEyeOT')
        acc_f_bl_idx_src = self._load_accel_from_integrated(dataset_key, 'eyes', 'fEyeBL')
        idx_acc = self._time_index_from_accel(acc_s_ot_idx_src, acc_s_bl_idx_src, acc_f_ot_idx_src, acc_f_bl_idx_src)
        idx_sum = self._merge_time(s_bl_df, f_bl_df, s_ot_df, f_ot_df)
        if idx_acc is not None and idx_sum is not None:
            idx = pd.Index(sorted(np.union1d(idx_acc.values, idx_sum.values)))
        else:
            idx = idx_acc or idx_sum
        if idx is None:
            print(f"No eyes summary/accel timebase found for {dataset_key}")
            return None

        def ser(df, name):
            if df is None or name not in df.columns:
                return None
            return df.set_index('Time')[name]

        # Positions: prefer side over front; for side/front choose OT first then BL just to maximize fill
        def first_non_none(*vals):
            for v in vals:
                if v is not None:
                    return v
            return None

        conflicts = []
        pos0_side = first_non_none(ser(s_ot_df, 'Eye0Pos'), ser(s_bl_df, 'Eye0Pos'))
        pos0_front = first_non_none(ser(f_ot_df, 'Eye0Pos'), ser(f_bl_df, 'Eye0Pos'))
        pos1_side = first_non_none(ser(s_ot_df, 'Eye1Pos'), ser(s_bl_df, 'Eye1Pos'))
        pos1_front = first_non_none(ser(f_ot_df, 'Eye1Pos'), ser(f_bl_df, 'Eye1Pos'))

        eye0pos, conf = self._coalesce_with_conflicts(idx, pos0_side, pos0_front,
                                                      dataset_key=dataset_key, feature='eyes', field='Eye0Pos', rule='side>front')
        conflicts.extend(conf)
        eye1pos, conf = self._coalesce_with_conflicts(idx, pos1_side, pos1_front,
                                                      dataset_key=dataset_key, feature='eyes', field='Eye1Pos', rule='side>front')
        conflicts.extend(conf)

        # Events: OT > BL, within each side > front
        def event(which):
            side_ot = ser(s_ot_df, which); front_ot = ser(f_ot_df, which)
            side_bl = ser(s_bl_df, which); front_bl = ser(f_bl_df, which)
            ot, c1 = self._coalesce_with_conflicts(idx, side_ot, front_ot,
                                                   dataset_key=dataset_key, feature='eyes', field=which, rule='OT: side>front')
            bl, c2 = self._coalesce_with_conflicts(idx, side_bl, front_bl,
                                                   dataset_key=dataset_key, feature='eyes', field=which, rule='BL: side>front')
            fused, c3 = self._coalesce_with_conflicts(idx, ot, bl,
                                                      dataset_key=dataset_key, feature='eyes', field=which, rule='OT>BL')
            return fused, (c1 + c2 + c3)

        e0, c = event('Eye0Event'); conflicts.extend(c); eye0evt = e0
        e1, c = event('Eye1Event'); conflicts.extend(c); eye1evt = e1

        # Acceleration: side > front; choose OT/BL whichever available
        # Prefer integrated for acceleration; fallback to raw param if needed
        acc_s_ot = self._prefer_df(
            self._load_accel_from_integrated(dataset_key, 'eyes', 'sEyeOT'),
            self._load_accel('eyes', s_ot)
        )
        acc_s_bl = self._prefer_df(
            self._load_accel_from_integrated(dataset_key, 'eyes', 'sEyeBL'),
            self._load_accel('eyes', s_bl)
        )
        acc_f_ot = self._prefer_df(
            self._load_accel_from_integrated(dataset_key, 'eyes', 'fEyeOT'),
            self._load_accel('eyes', f_ot)
        )
        acc_f_bl = self._prefer_df(
            self._load_accel_from_integrated(dataset_key, 'eyes', 'fEyeBL'),
            self._load_accel('eyes', f_bl)
        )

        def acc_serie(df, ch):
            if df is None: return None
            if ch == 0 and 'ch0' in df:
                return df.set_index('Time')['ch0']
            if ch == 1 and 'ch1' in df:
                return df.set_index('Time')['ch1']
            return None

        # precedence: side(OT) > side(BL) > front(OT) > front(BL)
        side0 = self._coalesce(idx, acc_serie(acc_s_ot, 0), acc_serie(acc_s_bl, 0))
        front0 = self._coalesce(idx, acc_serie(acc_f_ot, 0), acc_serie(acc_f_bl, 0))
        acc0, c = self._coalesce_with_conflicts(idx, side0, front0,
                                                dataset_key=dataset_key, feature='eyes', field='Acceleration 0', rule='side>front')
        conflicts.extend(c)
        side1 = self._coalesce(idx, acc_serie(acc_s_ot, 1), acc_serie(acc_s_bl, 1))
        front1 = self._coalesce(idx, acc_serie(acc_f_ot, 1), acc_serie(acc_f_bl, 1))
        acc1, c = self._coalesce_with_conflicts(idx, side1, front1,
                                                dataset_key=dataset_key, feature='eyes', field='Acceleration 1', rule='side>front')
        conflicts.extend(c)

        # (No defaults, no inference) — labels derived strictly from sheet presence below

        # Finally, derive labels strictly from sheet groups (no default fill)
        s_ot_presence = self._presence_from_integrated_sheet(dataset_key, 'eyes', 'sEyeOT')
        s_bl_presence = self._presence_from_integrated_sheet(dataset_key, 'eyes', 'sEyeBL')
        f_ot_presence = self._presence_from_integrated_sheet(dataset_key, 'eyes', 'fEyeOT')
        f_bl_presence = self._presence_from_integrated_sheet(dataset_key, 'eyes', 'fEyeBL')

        s_ot = (s_ot_presence.reindex(idx).fillna(False).astype(bool) if s_ot_presence is not None else pd.Series(False, index=idx))
        s_bl = (s_bl_presence.reindex(idx).fillna(False).astype(bool) if s_bl_presence is not None else pd.Series(False, index=idx))
        f_ot = (f_ot_presence.reindex(idx).fillna(False).astype(bool) if f_ot_presence is not None else pd.Series(False, index=idx))
        f_bl = (f_bl_presence.reindex(idx).fillna(False).astype(bool) if f_bl_presence is not None else pd.Series(False, index=idx))

        s_any = s_ot | s_bl
        f_any = f_ot | f_bl

        # Conflicts: more than one behavior active at same time
        for t in idx:
            active_behaviors = []
            if s_ot.loc[t]: active_behaviors.append('sEyeOT')
            if s_bl.loc[t]: active_behaviors.append('sEyeBL')
            if f_ot.loc[t]: active_behaviors.append('fEyeOT')
            if f_bl.loc[t]: active_behaviors.append('fEyeBL')
            if len(active_behaviors) > 1:
                conflicts.append({'Dataset': dataset_key, 'Feature': 'eyes', 'Field': 'Behavior',
                                  'Time': t, 'Preferred': ','.join(active_behaviors), 'Fallback': '', 'Diff': 0,
                                  'Rule': 'Multiple behaviors active'})

        # Pos
        eye0pos_label = pd.Series(index=idx, dtype=object)
        eye1pos_label = pd.Series(index=idx, dtype=object)
        eye0pos_label[s_any & ~f_any] = 's'
        eye0pos_label[f_any & ~s_any] = 'f'
        # Events
        e0_ot_pres = s_ot | f_ot
        e0_bl_pres = s_bl | f_bl
        e1_ot_pres = e0_ot_pres.copy()
        e1_bl_pres = e0_bl_pres.copy()
        eye0evt_label = pd.Series(index=idx, dtype=object)
        eye1evt_label = pd.Series(index=idx, dtype=object)
        eye0evt_label[e0_ot_pres & ~e0_bl_pres] = 'OT'
        eye0evt_label[e0_bl_pres & ~e0_ot_pres] = 'BL'
        eye1evt_label = eye0evt_label.copy()

        # Strict labels are set below by sheet presence; no additional fallbacks here

        # Do not default-fill; leave empties as-is

        # IMPORTANT: Do NOT assign any Pos/Event here.
        # Leave them blank; labels will be strictly written later by enforce_sheet_labels()
        blanks = [''] * len(idx)
        df_out = pd.DataFrame({
            'Time': idx.values,
            'Eye0Pos': blanks,
            'Eye1Pos': blanks,
            'Eye0Event': blanks,
            'Eye1Event': blanks,
            'Acceleration 0': acc0.values,
            'Acceleration 1': acc1.values
        })

        # Add other LabGym parameters (dual-organ channels 0/1)
        def disp(name):
            return ' '.join([w.capitalize() for w in name.split('_')])
        for p in self.param_keys:
            # side sources
            if p == 'probability':
                # sometimes probability could be single-channel; still try dual
                pass
            s_ot = self._load_param_from_integrated(dataset_key, 'eyes', 'sEyeOT', p)
            s_bl = self._load_param_from_integrated(dataset_key, 'eyes', 'sEyeBL', p)
            f_ot = self._load_param_from_integrated(dataset_key, 'eyes', 'fEyeOT', p)
            f_bl = self._load_param_from_integrated(dataset_key, 'eyes', 'fEyeBL', p)

            def serie(df, ch):
                if df is None: return None
                col = 'ch0' if ch == 0 else 'ch1'
                return df.set_index('Time')[col] if col in df.columns else None

            side0 = self._coalesce(idx, serie(s_ot, 0), serie(s_bl, 0))
            front0 = self._coalesce(idx, serie(f_ot, 0), serie(f_bl, 0))
            fused0, _ = self._coalesce_with_conflicts(idx, side0, front0,
                                                      dataset_key=dataset_key, feature='eyes', field=f'{p} 0', rule='side>front')
            side1 = self._coalesce(idx, serie(s_ot, 1), serie(s_bl, 1))
            front1 = self._coalesce(idx, serie(f_ot, 1), serie(f_bl, 1))
            fused1, _ = self._coalesce_with_conflicts(idx, side1, front1,
                                                      dataset_key=dataset_key, feature='eyes', field=f'{p} 1', rule='side>front')
            df_out[f'{disp(p)} 0'] = fused0.values
            df_out[f'{disp(p)} 1'] = fused1.values

        out_path = os.path.join(out_dir, 'eye_summary.xlsx')
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            df_out.to_excel(writer, index=False, sheet_name='Summary')
            # no conflict sheet needed
        print(f"eye_summary written: {out_path}")
        return out_path

    def build_ear_summary(self, dataset_key, dataset_raw_path):
        out_dir = os.path.join(self.summary_root, dataset_key)
        os.makedirs(out_dir, exist_ok=True)

        def resolve_folder(base, name):
            path = os.path.join(base, name)
            if os.path.exists(path):
                return path
            try:
                for n in os.listdir(base):
                    if n.lower() == name.lower():
                        alt = os.path.join(base, n)
                        print(f"Resolved folder '{name}' -> '{n}' in {base}")
                        return alt
            except Exception:
                pass
            print(f"Folder not found: {path}")
            return None

        s_bl = resolve_folder(dataset_raw_path, 'sEarBL')
        f_bl = resolve_folder(dataset_raw_path, 'fEarBL')
        s_pb = resolve_folder(dataset_raw_path, 'sEarPB')
        f_pb = resolve_folder(dataset_raw_path, 'fEarPB')

        s_bl_df = self._load_summary('ears', s_bl)
        f_bl_df = self._load_summary('ears', f_bl)
        s_pb_df = self._load_summary('ears', s_pb)
        f_pb_df = self._load_summary('ears', f_pb)

        acc_s_pb_idx_src = self._load_accel_from_integrated(dataset_key, 'ears', 'sEarPB')
        acc_s_bl_idx_src = self._load_accel_from_integrated(dataset_key, 'ears', 'sEarBL')
        acc_f_pb_idx_src = self._load_accel_from_integrated(dataset_key, 'ears', 'fEarPB')
        acc_f_bl_idx_src = self._load_accel_from_integrated(dataset_key, 'ears', 'fEarBL')
        idx_acc = self._time_index_from_accel(acc_s_pb_idx_src, acc_s_bl_idx_src, acc_f_pb_idx_src, acc_f_bl_idx_src)
        idx_sum = self._merge_time(s_bl_df, f_bl_df, s_pb_df, f_pb_df)
        if idx_acc is not None and idx_sum is not None:
            idx = pd.Index(sorted(np.union1d(idx_acc.values, idx_sum.values)))
        else:
            idx = idx_acc or idx_sum
        if idx is None:
            print(f"No ears summary/accel timebase found for {dataset_key}")
            return None

        def ser(df, name):
            if df is None or name not in df.columns:
                return None
            return df.set_index('Time')[name]

        # helper for this function
        def first_non_none(*vals):
            for v in vals:
                if v is not None:
                    return v
            return None

        conflicts = []
        # Pos: side > front; prefer PB first then BL to maximize fill
        pos0_side = first_non_none(ser(s_pb_df, 'Ear0Pos'), ser(s_bl_df, 'Ear0Pos'))
        pos0_front = first_non_none(ser(f_pb_df, 'Ear0Pos'), ser(f_bl_df, 'Ear0Pos'))
        pos1_side = first_non_none(ser(s_pb_df, 'Ear1Pos'), ser(s_bl_df, 'Ear1Pos'))
        pos1_front = first_non_none(ser(f_pb_df, 'Ear1Pos'), ser(f_bl_df, 'Ear1Pos'))
        ear0pos, c = self._coalesce_with_conflicts(idx, pos0_side, pos0_front,
                                                   dataset_key=dataset_key, feature='ears', field='Ear0Pos', rule='side>front')
        conflicts.extend(c)
        ear1pos, c = self._coalesce_with_conflicts(idx, pos1_side, pos1_front,
                                                   dataset_key=dataset_key, feature='ears', field='Ear1Pos', rule='side>front')
        conflicts.extend(c)

        # Event: PB > BL within side>front
        def event(which):
            side_pb = ser(s_pb_df, which); front_pb = ser(f_pb_df, which)
            side_bl = ser(s_bl_df, which); front_bl = ser(f_bl_df, which)
            pb, c1 = self._coalesce_with_conflicts(idx, side_pb, front_pb,
                                                   dataset_key=dataset_key, feature='ears', field=which, rule='PB: side>front')
            bl, c2 = self._coalesce_with_conflicts(idx, side_bl, front_bl,
                                                   dataset_key=dataset_key, feature='ears', field=which, rule='BL: side>front')
            fused, c3 = self._coalesce_with_conflicts(idx, pb, bl,
                                                      dataset_key=dataset_key, feature='ears', field=which, rule='PB>BL')
            return fused, (c1 + c2 + c3)

        e0, c = event('Ear0Event'); conflicts.extend(c); ear0evt = e0
        e1, c = event('Ear1Event'); conflicts.extend(c); ear1evt = e1

        # Acceleration
        acc_s_pb = self._prefer_df(
            self._load_accel_from_integrated(dataset_key, 'ears', 'sEarPB'),
            self._load_accel('ears', s_pb)
        )
        acc_s_bl = self._prefer_df(
            self._load_accel_from_integrated(dataset_key, 'ears', 'sEarBL'),
            self._load_accel('ears', s_bl)
        )
        acc_f_pb = self._prefer_df(
            self._load_accel_from_integrated(dataset_key, 'ears', 'fEarPB'),
            self._load_accel('ears', f_pb)
        )
        acc_f_bl = self._prefer_df(
            self._load_accel_from_integrated(dataset_key, 'ears', 'fEarBL'),
            self._load_accel('ears', f_bl)
        )

        def acc_serie(df, ch):
            if df is None: return None
            if ch == 0 and 'ch0' in df:
                return df.set_index('Time')['ch0']
            if ch == 1 and 'ch1' in df:
                return df.set_index('Time')['ch1']
            return None

        side0 = self._coalesce(idx, acc_serie(acc_s_pb, 0), acc_serie(acc_s_bl, 0))
        front0 = self._coalesce(idx, acc_serie(acc_f_pb, 0), acc_serie(acc_f_bl, 0))
        acc0, c = self._coalesce_with_conflicts(idx, side0, front0,
                                                dataset_key=dataset_key, feature='ears', field='Acceleration 0', rule='side>front')
        conflicts.extend(c)
        side1 = self._coalesce(idx, acc_serie(acc_s_pb, 1), acc_serie(acc_s_bl, 1))
        front1 = self._coalesce(idx, acc_serie(acc_f_pb, 1), acc_serie(acc_f_bl, 1))
        acc1, c = self._coalesce_with_conflicts(idx, side1, front1,
                                                dataset_key=dataset_key, feature='ears', field='Acceleration 1', rule='side>front')
        conflicts.extend(c)

        # Build Pos labels (f/s) from which source provided acceleration
        # Also use whole-sheet presence for robust labeling
        s_pb_presence = self._presence_from_integrated_sheet(dataset_key, 'ears', 'sEarPB')
        s_bl_presence = self._presence_from_integrated_sheet(dataset_key, 'ears', 'sEarBL')
        f_pb_presence = self._presence_from_integrated_sheet(dataset_key, 'ears', 'fEarPB')
        f_bl_presence = self._presence_from_integrated_sheet(dataset_key, 'ears', 'fEarBL')
        sheet_side_any = ((s_pb_presence.reindex(idx).fillna(False) if s_pb_presence is not None else pd.Series(False, index=idx)) |
                          (s_bl_presence.reindex(idx).fillna(False) if s_bl_presence is not None else pd.Series(False, index=idx)))
        sheet_front_any = ((f_pb_presence.reindex(idx).fillna(False) if f_pb_presence is not None else pd.Series(False, index=idx)) |
                           (f_bl_presence.reindex(idx).fillna(False) if f_bl_presence is not None else pd.Series(False, index=idx)))

        def _pres(series):
            return (series.reindex(idx).notna()) if series is not None else pd.Series(False, index=idx)

        side0_any = _pres(side0) | sheet_side_any
        front0_any = _pres(front0) | sheet_front_any
        side1_any = _pres(side1) | sheet_side_any
        front1_any = _pres(front1) | sheet_front_any

        ear0pos_label = pd.Series(index=idx, dtype=object)
        ear1pos_label = pd.Series(index=idx, dtype=object)
        for t in idx:
            ear0pos_label.loc[t] = 's' if side0_any.loc[t] else ('f' if front0_any.loc[t] else '')
            ear1pos_label.loc[t] = 's' if side1_any.loc[t] else ('f' if front1_any.loc[t] else '')

        # Build Event labels (PB/BL) by presence from event series (preferred) or ANY parameter sources (fallback)
        def presence(series):
            return (series.reindex(idx).notna()) if series is not None else pd.Series(False, index=idx)

        e0_pb_series = ser(s_pb_df, 'Ear0Event') if s_pb_df is not None and 'Ear0Event' in s_pb_df.columns else None
        e0_bl_series = ser(s_bl_df, 'Ear0Event') if s_bl_df is not None and 'Ear0Event' in s_bl_df.columns else None
        e1_pb_series = ser(s_pb_df, 'Ear1Event') if s_pb_df is not None and 'Ear1Event' in s_pb_df.columns else None
        e1_bl_series = ser(s_bl_df, 'Ear1Event') if s_bl_df is not None and 'Ear1Event' in s_bl_df.columns else None

        # Start with acceleration presence
        side0_any = presence(side0)
        front0_any = presence(front0)
        side1_any = presence(side1)
        front1_any = presence(front1)
        e0_pb_pres = presence(e0_pb_series) | presence(acc_serie(acc_s_pb, 0)) | presence(acc_serie(acc_f_pb, 0))
        e0_bl_pres = presence(e0_bl_series) | presence(acc_serie(acc_s_bl, 0)) | presence(acc_serie(acc_f_bl, 0))
        e1_pb_pres = presence(e1_pb_series) | presence(acc_serie(acc_s_pb, 1)) | presence(acc_serie(acc_f_pb, 1))
        e1_bl_pres = presence(e1_bl_series) | presence(acc_serie(acc_s_bl, 1)) | presence(acc_serie(acc_f_bl, 1))

        # Enrich with other parameters from integrated sheets (so any parameter implies pos/event)
        for p in self.param_keys:
            s_pb_p = self._load_param_from_integrated(dataset_key, 'ears', 'sEarPB', p)
            s_bl_p = self._load_param_from_integrated(dataset_key, 'ears', 'sEarBL', p)
            f_pb_p = self._load_param_from_integrated(dataset_key, 'ears', 'fEarPB', p)
            f_bl_p = self._load_param_from_integrated(dataset_key, 'ears', 'fEarBL', p)
            def serie(df, ch):
                if df is None: return None
                col = 'ch0' if ch == 0 else 'ch1'
                return df.set_index('Time')[col] if col in df.columns else None
            side0_any |= presence(serie(s_pb_p, 0)) | presence(serie(s_bl_p, 0))
            front0_any |= presence(serie(f_pb_p, 0)) | presence(serie(f_bl_p, 0))
            side1_any |= presence(serie(s_pb_p, 1)) | presence(serie(s_bl_p, 1))
            front1_any |= presence(serie(f_pb_p, 1)) | presence(serie(f_bl_p, 1))
            e0_pb_pres |= presence(serie(s_pb_p, 0)) | presence(serie(f_pb_p, 0))
            e0_bl_pres |= presence(serie(s_bl_p, 0)) | presence(serie(f_bl_p, 0))
            e1_pb_pres |= presence(serie(s_pb_p, 1)) | presence(serie(f_pb_p, 1))
            e1_bl_pres |= presence(serie(s_bl_p, 1)) | presence(serie(f_bl_p, 1))

        for t in idx:
            ear0pos_label.loc[t] = 's' if side0_any.loc[t] else ('f' if front0_any.loc[t] else '')
            ear1pos_label.loc[t] = 's' if side1_any.loc[t] else ('f' if front1_any.loc[t] else '')

        ear0evt_label = pd.Series(index=idx, dtype=object)
        ear1evt_label = pd.Series(index=idx, dtype=object)
        for t in idx:
            ear0evt_label.loc[t] = 'PB' if e0_pb_pres.loc[t] else ('BL' if e0_bl_pres.loc[t] else '')
            ear1evt_label.loc[t] = 'PB' if e1_pb_pres.loc[t] else ('BL' if e1_bl_pres.loc[t] else '')

        # No default fill for ears; leave empty when no single active sheet

        df_out = pd.DataFrame({
            'Time': idx.values,
            'Ear0Pos': ear0pos_label.values,
            'Ear1Pos': ear1pos_label.values,
            'Ear0Event': ear0evt_label.values,
            'Ear1Event': ear1evt_label.values,
            'Acceleration 0': acc0.values,
            'Acceleration 1': acc1.values
        })

        # Add other LabGym parameters (dual-organ channels 0/1)
        def disp(name):
            return ' '.join([w.capitalize() for w in name.split('_')])
        for p in self.param_keys:
            s_pb = self._load_param_from_integrated(dataset_key, 'ears', 'sEarPB', p)
            s_bl = self._load_param_from_integrated(dataset_key, 'ears', 'sEarBL', p)
            f_pb = self._load_param_from_integrated(dataset_key, 'ears', 'fEarPB', p)
            f_bl = self._load_param_from_integrated(dataset_key, 'ears', 'fEarBL', p)

            def serie(df, ch):
                if df is None: return None
                col = 'ch0' if ch == 0 else 'ch1'
                return df.set_index('Time')[col] if col in df.columns else None

            side0 = self._coalesce(idx, serie(s_pb, 0), serie(s_bl, 0))
            front0 = self._coalesce(idx, serie(f_pb, 0), serie(f_bl, 0))
            fused0, _ = self._coalesce_with_conflicts(idx, side0, front0,
                                                      dataset_key=dataset_key, feature='ears', field=f'{p} 0', rule='side>front')
            side1 = self._coalesce(idx, serie(s_pb, 1), serie(s_bl, 1))
            front1 = self._coalesce(idx, serie(f_pb, 1), serie(f_bl, 1))
            fused1, _ = self._coalesce_with_conflicts(idx, side1, front1,
                                                      dataset_key=dataset_key, feature='ears', field=f'{p} 1', rule='side>front')
            df_out[f'{disp(p)} 0'] = fused0.values
            df_out[f'{disp(p)} 1'] = fused1.values

        out_path = os.path.join(out_dir, 'ear_summary.xlsx')
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            df_out.to_excel(writer, index=False, sheet_name='Summary')
            # no conflict sheet needed
        print(f"ear_summary written: {out_path}")
        return out_path

    def build_nose_summary(self, dataset_key, dataset_raw_path):
        out_dir = os.path.join(self.summary_root, dataset_key)
        os.makedirs(out_dir, exist_ok=True)

        def resolve_folder(base, name):
            path = os.path.join(base, name)
            if os.path.exists(path):
                return path
            try:
                for n in os.listdir(base):
                    if n.lower() == name.lower():
                        alt = os.path.join(base, n)
                        print(f"Resolved folder '{name}' -> '{n}' in {base}")
                        return alt
            except Exception:
                pass
            print(f"Folder not found: {path}")
            return None

        nose_bl = resolve_folder(dataset_raw_path, 'noseBL')
        nose_bul = resolve_folder(dataset_raw_path, 'noseBul')

        bl_df = self._load_summary('nose', nose_bl)
        bul_df = self._load_summary('nose', nose_bul)

        acc_bul_idx_src = self._load_accel_from_integrated(dataset_key, 'nose', 'noseBul')
        acc_bl_idx_src = self._load_accel_from_integrated(dataset_key, 'nose', 'noseBL')
        idx_acc = self._time_index_from_accel(acc_bul_idx_src, acc_bl_idx_src)
        idx_sum = self._merge_time(bl_df, bul_df)
        if idx_acc is not None and idx_sum is not None:
            idx = pd.Index(sorted(np.union1d(idx_acc.values, idx_sum.values)))
        else:
            idx = idx_acc or idx_sum
        if idx is None:
            print(f"No nose summary/accel timebase found for {dataset_key}")
            return None

        def ser(df, name):
            if df is None or name not in df.columns:
                return None
            return df.set_index('Time')[name]

        # Bul > BL
        conflicts = []
        nosepos, c = self._coalesce_with_conflicts(idx, ser(bul_df, 'NosePos'), ser(bl_df, 'NosePos'),
                                                   dataset_key=dataset_key, feature='nose', field='NosePos', rule='Bul>BL')
        conflicts.extend(c)
        noseevt, c = self._coalesce_with_conflicts(idx, ser(bul_df, 'NoseEvent'), ser(bl_df, 'NoseEvent'),
                                                   dataset_key=dataset_key, feature='nose', field='NoseEvent', rule='Bul>BL')
        conflicts.extend(c)

        # Acceleration
        acc_bul = self._prefer_df(
            self._load_accel_from_integrated(dataset_key, 'nose', 'noseBul'),
            self._load_accel('nose', nose_bul)
        )
        acc_bl = self._prefer_df(
            self._load_accel_from_integrated(dataset_key, 'nose', 'noseBL'),
            self._load_accel('nose', nose_bl)
        )
        def acc_serie(df):
            if df is None: return None
            # nose single organ -> use ch0 if present, else first numeric column
            col = 'ch0' if 'ch0' in df else next((c for c in df.columns if c != 'Time'), None)
            if col is None: return None
            return df.set_index('Time')[col]
        acc = self._coalesce(idx, acc_serie(acc_bul), acc_serie(acc_bl))

        blanks = [''] * len(idx)
        df_out = pd.DataFrame({
            'Time': idx.values,
            'NosePos': blanks,
            'NoseEvent': blanks,
            'Acceleration': acc.values
        })

        # Add other LabGym parameters (single organ)
        def disp(name):
            return ' '.join([w.capitalize() for w in name.split('_')])
        for p in self.param_keys:
            bul = self._load_param_from_integrated(dataset_key, 'nose', 'noseBul', p)
            bl = self._load_param_from_integrated(dataset_key, 'nose', 'noseBL', p)
            def serie(df):
                if df is None: return None
                if 'ch0' in df.columns:
                    return df.set_index('Time')['ch0']
                # pick any numeric column other than Time
                for c in df.columns:
                    if c != 'Time':
                        return df.set_index('Time')[c]
                return None
            fused, _ = self._coalesce_with_conflicts(idx, serie(bul), serie(bl),
                                                     dataset_key=dataset_key, feature='nose', field=p, rule='Bul>BL')
            df_out[disp(p)] = fused.values

        out_path = os.path.join(out_dir, 'nose_summary.xlsx')
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            df_out.to_excel(writer, index=False, sheet_name='Summary')
        print(f"nose_summary written: {out_path}")
        return out_path

    def check_summary_files_exist(self, dataset_key):
        """Check if all summary files already exist for a dataset."""
        output_dir = os.path.join(self.summary_root, dataset_key)

        # Check for all three summary files
        files_to_check = [
            os.path.join(output_dir, "eye_summary.xlsx"),
            os.path.join(output_dir, "ear_summary.xlsx"),
            os.path.join(output_dir, "nose_summary.xlsx")
        ]

        return all(os.path.exists(f) for f in files_to_check)

    def generate_all(self):
        # Use unified dataset configuration to locate raw folders
        from dataset_config import DatasetConfig
        cfg = DatasetConfig(self.base_path)
        datasets, _ = cfg.get_working_datasets()
        results = {}
        for key, raw_path in datasets.items():
            print(f"\nGenerating summaries for {key}")

            # Check if all summary files already exist
            if self.check_summary_files_exist(key):
                print(f" All summary files already exist for {key}, skipping...")
                results[key] = {
                    'eye_summary': os.path.join(self.summary_root, key, "eye_summary.xlsx"),
                    'ear_summary': os.path.join(self.summary_root, key, "ear_summary.xlsx"),
                    'nose_summary': os.path.join(self.summary_root, key, "nose_summary.xlsx")
                }
                continue

            results[key] = {
                'eye_summary': self.build_eye_summary(key, raw_path),
                'ear_summary': self.build_ear_summary(key, raw_path),
                'nose_summary': self.build_nose_summary(key, raw_path),
            }
            # After building, run consistency checks for eyes/ears
            try:
                self.enforce_sheet_labels(key)
                self.run_checks(key)
            except Exception as e:
                print(f"Post-process/check failed for {key}: {e}")
        return results

    # --------------------
    # Consistency checking
    # --------------------
    def _presence_bool(self, series, idx):
        if series is None:
            return pd.Series(False, index=idx)
        return series.reindex(idx).fillna(False).astype(bool)

    def _behaviors_for_feature(self, feature):
        if feature == 'eyes':
            return ['sEyeOT', 'sEyeBL', 'fEyeOT', 'fEyeBL']
        if feature == 'ears':
            return ['sEarPB', 'sEarBL', 'fEarPB', 'fEarBL']
        return []

    def _presence_map(self, dataset_key, feature, idx):
        mp = {}
        for b in self._behaviors_for_feature(feature):
            mp[b] = self._presence_bool(
                self._presence_from_integrated_sheet(dataset_key, feature, b), idx
            )
        return mp

    def _expected_labels(self, feature, presence_map, idx):
        expected_pos = pd.Series('', index=idx, dtype=object)
        expected_evt = pd.Series('', index=idx, dtype=object)
        conflicts = []
        # priority order to break ties: side>front then OT/PB > BL
        order = (['sEyeOT','sEyeBL','fEyeOT','fEyeBL'] if feature=='eyes' else
                 ['sEarPB','sEarBL','fEarPB','fEarBL'])
        for t in idx:
            active = [b for b,s in presence_map.items() if bool(s.loc[t])]
            if len(active) == 1:
                b = active[0]
                expected_pos.loc[t] = 's' if b.startswith('s') else 'f'
                if feature=='eyes':
                    expected_evt.loc[t] = 'OT' if b.endswith('OT') else 'BL'
                else:
                    expected_evt.loc[t] = 'PB' if b.endswith('PB') else 'BL'
            elif len(active) > 1:
                # choose best by order to report (but keep expected empty)
                chosen = max(order, key=lambda k: (1 if k in active else 0))
                conflicts.append({'Time': t, 'Active': ','.join(active), 'Chosen': chosen})
        return expected_pos, expected_evt, conflicts

    def _append_check_sheet(self, summary_path, feature, idx, expected_pos, expected_evt, found_df, conflicts):
        if summary_path is None or not os.path.exists(summary_path):
            return
        issues = []
        pos_cols = (['Eye0Pos','Eye1Pos'] if feature=='eyes' else ['Ear0Pos','Ear1Pos'])
        evt_cols = (['Eye0Event','Eye1Event'] if feature=='eyes' else ['Ear0Event','Ear1Event'])

        # normalize found_df columns
        for c in pos_cols+evt_cols:
            if c not in found_df.columns:
                found_df[c] = ''

        for t in idx:
            exp_p = expected_pos.loc[t]
            exp_e = expected_evt.loc[t]
            f_p0 = str(found_df.loc[found_df['Time']==t, pos_cols[0]].iloc[0]) if not found_df.loc[found_df['Time']==t].empty else ''
            f_p1 = str(found_df.loc[found_df['Time']==t, pos_cols[1]].iloc[0]) if not found_df.loc[found_df['Time']==t].empty else ''
            f_e0 = str(found_df.loc[found_df['Time']==t, evt_cols[0]].iloc[0]) if not found_df.loc[found_df['Time']==t].empty else ''
            f_e1 = str(found_df.loc[found_df['Time']==t, evt_cols[1]].iloc[0]) if not found_df.loc[found_df['Time']==t].empty else ''

            # rule: if expected empty -> labels should be empty
            if exp_p == '' and (f_p0 or f_p1 or f_e0 or f_e1):
                issues.append({'Time': t, 'ExpectedPos': exp_p, 'ExpectedEvent': exp_e,
                               'FoundPos0': f_p0, 'FoundPos1': f_p1, 'FoundEvent0': f_e0, 'FoundEvent1': f_e1,
                               'Issue': 'Labels set but no active behavior'})
            # if expected non-empty -> both channels should match expected
            if exp_p != '':
                if f_p0 != exp_p or f_p1 != exp_p:
                    issues.append({'Time': t, 'ExpectedPos': exp_p, 'ExpectedEvent': exp_e,
                                   'FoundPos0': f_p0, 'FoundPos1': f_p1, 'FoundEvent0': f_e0, 'FoundEvent1': f_e1,
                                   'Issue': 'Pos mismatch'})
                if f_e0 != exp_e or f_e1 != exp_e:
                    issues.append({'Time': t, 'ExpectedPos': exp_p, 'ExpectedEvent': exp_e,
                                   'FoundPos0': f_p0, 'FoundPos1': f_p1, 'FoundEvent0': f_e0, 'FoundEvent1': f_e1,
                                   'Issue': 'Event mismatch'})

        # Append sheet
        try:
            with pd.ExcelWriter(summary_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                if issues:
                    pd.DataFrame(issues).to_excel(writer, sheet_name='Check', index=False)
                else:
                    pd.DataFrame([{'Message':'No issues found'}]).to_excel(writer, sheet_name='Check', index=False)
                if conflicts:
                    pd.DataFrame(conflicts).to_excel(writer, sheet_name='Behavior_Conflicts', index=False)
        except Exception as e:
            print(f"Failed to append check sheet to {summary_path}: {e}")

    def run_checks(self, dataset_key):
        # Build a common time index from integrated eyes to align checks
        feat_to_file = {
            'eyes': os.path.join(self.summary_root, dataset_key, 'eye_summary.xlsx'),
            'ears': os.path.join(self.summary_root, dataset_key, 'ear_summary.xlsx')
        }
        for feature in ['eyes','ears']:
            # Load summary
            summary_path = feat_to_file[feature]
            if not os.path.exists(summary_path):
                continue
            try:
                df = pd.read_excel(summary_path, sheet_name='Summary')
            except Exception:
                continue
            if 'Time' not in df.columns:
                continue
            idx = pd.Index(df['Time'].dropna().unique())
            presence_map = self._presence_map(dataset_key, feature, idx)
            exp_pos, exp_evt, conflicts = self._expected_labels(feature, presence_map, idx)
            self._append_check_sheet(summary_path, feature, idx, exp_pos, exp_evt, df, conflicts)

    def enforce_sheet_labels(self, dataset_key):
        """After summaries are built, strictly override Pos/Event based on sheet presence only.
        If exactly one behavior is active → set labels from sheet name; if none or multiple → blank and log conflict.
        Applies to eyes and ears.
        """
        for feature, file_name in [('eyes','eye_summary.xlsx'), ('ears','ear_summary.xlsx'), ('nose','nose_summary.xlsx')]:
            summary_path = os.path.join(self.summary_root, dataset_key, file_name)
            if not os.path.exists(summary_path):
                continue
            try:
                df = pd.read_excel(summary_path, sheet_name='Summary')
            except Exception:
                continue
            if 'Time' not in df.columns:
                continue
            idx = pd.Index(df['Time'].dropna().unique())
            # Apply strict labels per channel (0 / 1) using per-channel presence
            time_to_rows = {t: df.index[df['Time']==t].tolist() for t in idx}
            if feature in ['eyes','ears']:
                behaviors = self._behaviors_for_feature(feature)
                pres0 = {}
                pres1 = {}
                for b in behaviors:
                    p0, p1 = self._presence_by_channel(dataset_key, feature, b)
                    pres0[b] = self._presence_bool(p0, idx)
                    pres1[b] = self._presence_bool(p1, idx)

                def label_from_behavior(b, which_feature):
                    pos = ('s' if b.startswith('s') else 'f') if which_feature in ['eyes','ears'] else ''
                    if which_feature == 'eyes':
                        evt = 'OT' if b.endswith('OT') else 'BL'
                    elif which_feature == 'ears':
                        evt = 'PB' if b.endswith('PB') else 'BL'
                    else:
                        evt = ''
                    return pos, evt

                # Column mapping
                if feature == 'eyes':
                    pos_cols = ['Eye0Pos','Eye1Pos']; evt_cols = ['Eye0Event','Eye1Event']
                else:
                    pos_cols = ['Ear0Pos','Ear1Pos']; evt_cols = ['Ear0Event','Ear1Event']

                for t in idx:
                    rows = time_to_rows.get(t, [])
                    if not rows:
                        continue
                    r = rows[0]
                    active0 = [b for b in behaviors if pres0[b].loc[t]]
                    active1 = [b for b in behaviors if pres1[b].loc[t]]
                    # channel 0
                    if len(active0) == 1:
                        p,e = label_from_behavior(active0[0], feature)
                        df.at[r, pos_cols[0]] = p
                        df.at[r, evt_cols[0]] = e
                    else:
                        df.at[r, pos_cols[0]] = ''
                        df.at[r, evt_cols[0]] = ''
                    # channel 1
                    if len(active1) == 1:
                        p,e = label_from_behavior(active1[0], feature)
                        df.at[r, pos_cols[1]] = p
                        df.at[r, evt_cols[1]] = e
                    else:
                        df.at[r, pos_cols[1]] = ''
                        df.at[r, evt_cols[1]] = ''
            else:
                # nose: simple presence by sheet (probability excluded); take sheet suffix directly
                bul_pres = self._presence_bool(
                    self._presence_from_integrated_sheet(dataset_key, 'nose', 'noseBul'), idx)
                bl_pres = self._presence_bool(
                    self._presence_from_integrated_sheet(dataset_key, 'nose', 'noseBL'), idx)
                for t in idx:
                    rows = time_to_rows.get(t, [])
                    if not rows:
                        continue
                    r = rows[0]
                    if bul_pres.loc[t] and not bl_pres.loc[t]:
                        df.at[r, 'NoseEvent'] = 'Bul'
                    elif bl_pres.loc[t] and not bul_pres.loc[t]:
                        df.at[r, 'NoseEvent'] = 'BL'
                    elif bul_pres.loc[t] and bl_pres.loc[t]:
                        # If both present, prefer Bul per your instruction to take sheet suffix directly
                        df.at[r, 'NoseEvent'] = 'Bul'
                    else:
                        df.at[r, 'NoseEvent'] = ''

            # Write back (replace Summary)
            with pd.ExcelWriter(summary_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name='Summary', index=False)

    # --------------------
    # Simple preview (no write, no conflict handling) as you requested
    # --------------------
    def preview_labels_simple(self, dataset_key, feature='ears', head=30):
        """Print a simple preview of labels per time based strictly on sheet names.
        - For each time t: find which sheet(s) have any parameter value in that row
        - If exactly one sheet is active, derive Pos/Event directly from the sheet name
        - If none or multiple are active, leave Pos/Event blank
        This does NOT write files; it just prints the first `head` rows.
        """
        if feature not in ['eyes','ears','nose']:
            print(f"Unsupported feature: {feature}")
            return
        integ_path = os.path.join(self.summary_root, dataset_key, f"integrated_{feature}_data.xlsx")
        if not os.path.exists(integ_path):
            print(f"Integrated file missing: {integ_path}")
            return

        # collect presence for expected behaviors
        behaviors = self._behaviors_for_feature(feature)
        # build global index from any present sheet
        all_times = None
        sheet_presence = {}
        for b in behaviors:
            s = self._presence_from_integrated_sheet(dataset_key, feature, b)
            if s is not None:
                sheet_presence[b] = s
                all_times = np.union1d(all_times, s.index.values) if all_times is not None else s.index.values
        if all_times is None:
            print("No data in any behavior sheets.")
            return
        idx = pd.Index(sorted(all_times))

        def to_bool(s):
            if s is None:
                return pd.Series(False, index=idx)
            return s.reindex(idx).fillna(False).astype(bool)

        # normalize presence per behavior
        pres = {b: to_bool(sheet_presence.get(b)) for b in behaviors}

        # print header
        print(f"\n=== PREVIEW ({dataset_key} / {feature}) ===")
        print("Time\tActive\tPos\tEvent")

        count = 0
        for t in idx:
            active = [b for b in behaviors if pres[b].loc[t]]
            pos = evt = ''
            if len(active) == 1:
                b = active[0]
                if feature in ['eyes','ears']:
                    pos = 's' if b.startswith('s') else 'f'
                if feature == 'eyes':
                    evt = 'OT' if b.endswith('OT') else 'BL'
                elif feature == 'ears':
                    evt = 'PB' if b.endswith('PB') else 'BL'
                else:
                    evt = 'Bul' if b.endswith('Bul') else 'BL'
            print(f"{t}\t{','.join(active) if active else '-'}\t{pos}\t{evt}")
            count += 1
            if count >= head:
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=("Generate integrated data + ear/eye/nose summary files from a "
                     "folder of LabGym output datasets. Point base_path at the folder "
                     "that contains your per-animal LabGym result subfolders."))
    parser.add_argument("base_path",
                        help="Path to the folder containing your LabGym dataset subfolders.")
    base_path = parser.parse_args().base_path

    # Step 1: Generate integrated data files for all datasets
    print(f"\n{'='*60}")
    print("STEP 1: Generating integrated data files...")
    print(f"{'='*60}")
    integrator = FacialDataIntegrator(base_path)
    integrator_results = integrator.process_all_datasets()

    print(f"\n{'='*60}")
    print("STEP 1 COMPLETED!")
    print(f"{'='*60}")

    # Step 2: Build advanced concise summaries based on integrated files
    print(f"\n{'='*60}")
    print("STEP 2: Generating summary files...")
    print(f"{'='*60}")
    advanced = LoadDataAdvanced(base_path)
    adv_results = advanced.generate_all()

    print(f"\n{'='*60}")
    print("ALL PROCESSING COMPLETED!")
    print(f"{'='*60}")

    # Display results summary
    for dataset_name, files in adv_results.items():
        print(f"\n{dataset_name}:")
        print(f"  eye_summary: {files['eye_summary']}")
        print(f"  ear_summary: {files['ear_summary']}")
        print(f"  nose_summary: {files['nose_summary']}")
