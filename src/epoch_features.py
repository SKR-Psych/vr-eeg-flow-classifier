import argparse
import os
import sys
import mne
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import load_bids_data
from src.preprocessing import preprocess_raw
from src.classifier import parse_desc

def extract_event_epochs(raw: mne.io.Raw, event_target: str = 'box:touched', tmin: float = -0.1, tmax: float = 0.6, baseline: tuple = (-0.1, 0.0)) -> tuple:
    """
    Extracts event-locked EEG epochs centered on target event markers (e.g. 'box:touched').
    
    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed clean MNE Raw data.
    event_target : str, optional
        Target event annotation string prefix. Default is 'box:touched'.
    tmin : float, optional
        Start time before event in seconds. Default is -0.1 (-100 ms).
    tmax : float, optional
        End time after event in seconds. Default is 0.6 (+600 ms).
    baseline : tuple, optional
        Baseline correction time window. Default is (-0.1, 0.0).

    Returns
    -------
    epochs : mne.Epochs
        MNE Epochs object containing the segmented trials.
    labels : np.ndarray
        1D numpy array of binary ground truth labels (1 = Normal/Flow, 0 = Conflict/Disrupted).
    metadata : pd.DataFrame
        DataFrame with trial details (trial_nr, condition, onset, label).
    """
    # Standardize sfreq to 500.0 Hz if needed for consistent epoch length across sessions
    if raw.info['sfreq'] != 500.0:
        raw = raw.copy().resample(500.0, verbose=False)

    annotations = raw.annotations
    sfreq = raw.info['sfreq']
    
    event_samples = []
    labels = []
    metadata_rows = []
    
    for ann in annotations:
        onset = ann['onset']
        desc = ann['description']
        event_type, params = parse_desc(desc)
        
        if event_type == event_target:
            normal_or_conflict = params.get('normal_or_conflict')
            if normal_or_conflict not in ('normal', 'conflict'):
                continue
                
            sample = int(round(onset * sfreq))
            label = 1 if normal_or_conflict == 'normal' else 0
            
            event_samples.append([sample, 0, label])
            labels.append(label)
            metadata_rows.append({
                'onset': onset,
                'sample': sample,
                'trial_nr': params.get('trial_nr'),
                'condition': params.get('condition'),
                'normal_or_conflict': normal_or_conflict,
                'label': label
            })
            
    if not event_samples:
        print(f"[!] Warning: No '{event_target}' events found with valid labels.")
        return None, np.array([]), pd.DataFrame()
        
    events_arr = np.array(event_samples, dtype=int)
    event_id = {'Conflict': 0, 'Normal': 1}
    
    # Pick EEG channels only
    raw_eeg = raw.copy().pick(picks='eeg')
    
    # Construct MNE Epochs
    epochs = mne.Epochs(
        raw_eeg,
        events=events_arr,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        event_repeated='drop',
        preload=True,
        verbose=False
    )
    
    labels_arr = epochs.events[:, 2]
    metadata_df = pd.DataFrame(metadata_rows)
    
    print(f"[*] Extracted {len(epochs)} event-locked epochs around '{event_target}' ({tmin}s to {tmax}s).")
    print(f"    Label balance: Normal={np.sum(labels_arr == 1)}, Conflict={np.sum(labels_arr == 0)}")
    
    return epochs, labels_arr, metadata_df

def extract_epoch_features(epochs: mne.Epochs) -> pd.DataFrame:
    """
    Extracts time-domain ERP features (amplitude peaks and windows) and 
    spectral band powers from event-locked epochs for baseline comparison.
    """
    data = epochs.get_data(units='uV')  # (n_epochs, n_channels, n_times)
    ch_names = epochs.ch_names
    times = epochs.times
    
    feature_list = []
    
    # Define channel indices of interest
    fz_idx = ch_names.index('Fz') if 'Fz' in ch_names else 0
    cz_idx = ch_names.index('Cz') if 'Cz' in ch_names else 0
    pz_idx = ch_names.index('Pz') if 'Pz' in ch_names else 0
    
    # Define time windows for Prediction Error Negativity (PEN/N200) and P300
    pen_mask = (times >= 0.15) & (times <= 0.35)
    p300_mask = (times >= 0.30) & (times <= 0.50)
    post_mask = (times >= 0.20) & (times <= 0.60)
    
    for i in range(len(epochs)):
        feat = {}
        trial_data = data[i]  # (n_channels, n_times)
        
        # 1. Frontal/Central/Parietal ERP Window Means (uV)
        feat['pen_fz_mean'] = float(np.mean(trial_data[fz_idx, pen_mask]))
        feat['pen_cz_mean'] = float(np.mean(trial_data[cz_idx, pen_mask]))
        feat['p300_pz_mean'] = float(np.mean(trial_data[pz_idx, p300_mask]))
        
        # 2. Peak-to-trough amplitudes
        feat['pen_fz_min'] = float(np.min(trial_data[fz_idx, pen_mask]))
        feat['p300_cz_max'] = float(np.max(trial_data[cz_idx, p300_mask]))
        
        # 3. Global Field Power (GFP) over time window
        gfp = np.std(trial_data, axis=0)  # across channels
        feat['gfp_pen_mean'] = float(np.mean(gfp[pen_mask]))
        feat['gfp_post_mean'] = float(np.mean(gfp[post_mask]))
        
        feature_list.append(feat)
        
    return pd.DataFrame(feature_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract event-locked epochs from BIDS dataset.")
    parser.add_argument("--subject", type=str, default="02", help="Subject ID (e.g. 02)")
    parser.add_argument("--session", type=str, default="EMS", help="Session ID (e.g. EMS, Vibro, Visual)")
    parser.add_argument("--root", type=str, default="data/ds003846", help="BIDS root directory")
    args = parser.parse_args()
    
    try:
        raw = load_bids_data(args.subject, args.session, args.root)
        raw_clean = preprocess_raw(raw)
        epochs, labels, meta = extract_event_epochs(raw_clean)
        
        if epochs is not None:
            df_feat = extract_epoch_features(epochs)
            print(f"[+] Feature Extraction Complete. Shape: {df_feat.shape}")
            print(df_feat.describe())
    except Exception as e:
        print(f"[!] Epoch extraction failed: {e}")
        import traceback
        traceback.print_exc()
