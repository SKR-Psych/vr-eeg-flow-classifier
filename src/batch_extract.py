import os
import glob
import argparse
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from src.loader import load_bids_data
from src.preprocessing import preprocess_raw
from src.features import extract_epoch_features
from src.classifier import parse_desc

def process_single_session(sub: str, ses: str, bids_root: str, window_len: float, step: float) -> pd.DataFrame:
    """
    Worker function to process a single subject session in parallel.
    """
    ses_dir = os.path.join(bids_root, f"sub-{sub}", f"ses-{ses}")
    if not os.path.exists(ses_dir):
        return None
        
    try:
        # 1. Load raw data
        raw = load_bids_data(sub, ses, bids_root)
        annotations = raw.annotations
        
        # 2. Extract and pair trials by matching trial_nr and condition
        spawned_events = {}
        touched_events = {}
        
        for ann in annotations:
            onset = ann['onset']
            desc = ann['description']
            event_type, params = parse_desc(desc)
            
            trial_nr = params.get('trial_nr')
            condition = params.get('condition')
            
            if not trial_nr or not condition:
                continue
                
            key = (trial_nr, condition)
            
            if event_type == 'box:spawned':
                spawned_events[key] = {
                    'start_time': onset,
                    'normal_or_conflict': params.get('normal_or_conflict')
                }
            elif event_type == 'box:touched':
                touched_events[key] = onset

        # Pair them up
        paired_trials = []
        for key, spawned in spawned_events.items():
            if key in touched_events:
                start_time = spawned['start_time']
                end_time = touched_events[key]
                normal_or_conflict = spawned['normal_or_conflict']
                
                if normal_or_conflict not in ('normal', 'conflict'):
                    continue
                    
                label = 1 if normal_or_conflict == 'normal' else 0
                paired_trials.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'label': label
                })

        if not paired_trials:
            print(f"[!] No valid paired trials found for sub-{sub} ses-{ses}. Skipping.")
            return None
            
        # 3. Preprocess
        print(f"[*] Preprocessing raw data for sub-{sub} ses-{ses}...")
        raw_clean = preprocess_raw(raw)
        
        # 4. Extract features
        print(f"[*] Extracting features for sub-{sub} ses-{ses}...")
        df_features = extract_epoch_features(raw_clean, window_len=window_len, step=step)
        
        # 5. Align with trials
        labels = []
        midpoint_offset = window_len / 2.0
        for idx, row in df_features.iterrows():
            t_mid = row['timestamp'] + midpoint_offset
            assigned_label = -1
            
            for trial in paired_trials:
                if trial['start_time'] <= t_mid <= trial['end_time']:
                    assigned_label = trial['label']
                    break
            labels.append(assigned_label)
            
        df_features['label'] = labels
        
        # Filter out ISI
        df_clean = df_features[df_features['label'] != -1].copy()
        
        # Add subject and session identifiers
        df_clean.insert(0, 'subject_id', f"sub-{sub}")
        df_clean.insert(1, 'session_id', f"ses-{ses}")
        
        print(f"[+] Successfully processed sub-{sub} ses-{ses}: {df_clean.shape[0]} aligned windows.")
        return df_clean
        
    except Exception as e:
        print(f"[!] Failed processing sub-{sub} ses-{ses}: {e}")
        return None

def batch_extract(bids_root: str = "data/ds003846", window_len: float = 2.0, step: float = 0.2, n_jobs: int = -2):
    # Scan for all subjects
    sub_dirs = glob.glob(os.path.join(bids_root, "sub-*"))
    subjects = [os.path.basename(d).replace("sub-", "") for d in sub_dirs if os.path.isdir(d)]
    subjects.sort()
    
    print(f"[*] Found {len(subjects)} subjects: {subjects}")
    
    sessions = ["EMS", "Vibro", "Visual"]
    
    # Build tasks list
    tasks = []
    for sub in subjects:
        for ses in sessions:
            tasks.append((sub, ses))
            
    print(f"[*] Dispatching {len(tasks)} sessions in parallel (n_jobs={n_jobs})...")
    
    # Execute loop in parallel using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_session)(sub, ses, bids_root, window_len, step)
        for sub, ses in tasks
    )
    
    # Filter out empty results
    master_dfs = [r for r in results if r is not None]
    
    if master_dfs:
        print("\n[*] Concatenating master feature datasets...")
        df_master = pd.concat(master_dfs, ignore_index=True)
        
        # Save output
        out_dir = os.path.join(bids_root, "derivatives")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, "master_features.csv")
        df_master.to_csv(out_csv, index=False)
        
        print("\n" + "="*50)
        print("=== Master Feature Extraction Complete ===")
        print(f"Total Windows: {df_master.shape[0]}")
        print(f"Unique Subjects: {df_master['subject_id'].nunique()}")
        print(f"Unique Sessions: {df_master['session_id'].nunique()}")
        print(f"Label Balance:   Flow={np.sum(df_master['label']==1)}, Disrupted={np.sum(df_master['label']==0)}")
        print(f"File Saved:      {out_csv}")
        print("="*50)
    else:
        print("[!] Error: No feature datasets compiled.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features in batch across all subjects and sessions.")
    parser.add_argument("--root", type=str, default="data/ds003846", help="BIDS root directory")
    parser.add_argument("--window", type=float, default=2.0, help="Window size in seconds")
    parser.add_argument("--step", type=float, default=0.2, help="Step size in seconds")
    parser.add_argument("--jobs", type=int, default=-2, help="Number of parallel jobs (-1 for all, -2 for all but 1)")
    args = parser.parse_args()
    
    batch_extract(args.root, args.window, args.step, args.jobs)
