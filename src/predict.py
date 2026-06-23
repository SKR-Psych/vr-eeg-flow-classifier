import argparse
import os
import time
import mne
import joblib
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from mne.time_frequency import psd_array_welch
from mne.preprocessing import ICA
from src.loader import load_bids_data
from src.features import calculate_shannon_entropy, calculate_plv, calculate_relative_power
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Not setting position.*")

def calibrate_ica(raw: mne.io.Raw, calibration_duration: float = 60.0) -> ICA:
    """
    Simulates a calibration phase by fitting ICA on a baseline chunk (e.g., first 60 seconds of raw data).
    Detects and excludes eye/muscle artifact components.
    """
    print("\n" + "="*50)
    print("=== STEP 2: Calibration (ICA Fitting & Artifact Detection) ===")
    print("="*50)
    
    # 1. Ensure montage is standard 10-20 (required for mne-icalabel)
    print("[*] Setting standard 10-20 montage...")
    raw.set_montage('standard_1020', on_missing='ignore')
    
    # 2. Extract baseline/calibration chunk
    baseline_tmax = min(calibration_duration, raw.times[-1])
    print(f"[*] Extracting first {baseline_tmax:.1f} seconds of raw data for baseline calibration...")
    raw_calib = raw.copy().crop(tmin=0.0, tmax=baseline_tmax, include_tmax=True)
    
    # 3. Filter copy at 1.0–45.0 Hz for ICA fitting
    print("[*] Creating 1.0 Hz high-passed copy for ICA fitting...")
    raw_calib_fit = raw_calib.copy()
    raw_calib_fit.filter(
        l_freq=1.0, 
        h_freq=45.0, 
        method='iir', 
        iir_params=dict(order=4, ftype='butter'), 
        phase='zero', 
        verbose=False
    )
    raw_calib_fit.pick(picks='eeg', exclude='bads')
    
    # 4. Set up and fit ICA
    print("[*] Initializing baseline ICA (extended Infomax)...")
    ica = ICA(
        n_components=0.99, 
        method='infomax', 
        fit_params=dict(extended=True), 
        random_state=97
    )
    ica.fit(raw_calib_fit, verbose=False)
    print(f"[*] Baseline ICA completed. Extracted {ica.n_components_} components.")
    
    # 5. Detect eye components via EOG correlation (Fp2 channel)
    eog_channel = 'Fp2'
    eog_exclude = []
    if eog_channel in raw.ch_names:
        print(f"[*] Correlating ICA components with EOG channel '{eog_channel}'...")
        eog_inds, _ = ica.find_bads_eog(raw_calib, ch_name=eog_channel, verbose=False)
        eog_exclude = list(eog_inds)
        print(f"[*] EOG correlation identified component indices: {eog_exclude}")
    else:
        print(f"[!] Warning: EOG channel '{eog_channel}' not found in raw data. Skipping EOG correlation.")
        
    # 6. Detect eye and muscle components via mne-icalabel
    icalabel_exclude = []
    try:
        from mne_icalabel import label_components
        print("[*] Running mne-icalabel for automatic artifact classification...")
        labels_dict = label_components(raw_calib_fit, ica, method='iclabel')
        labels = labels_dict['labels']
        for idx, lbl in enumerate(labels):
            if 'eye' in lbl or 'muscle' in lbl:
                icalabel_exclude.append(idx)
        print(f"[*] mne-icalabel flagged component indices: {icalabel_exclude}")
    except Exception as e:
        print(f"[!] mne-icalabel classification skipped/failed: {e}")
        
    # Combine exclusions
    exclude_components = list(set(eog_exclude + icalabel_exclude))
    exclude_components.sort()
    ica.exclude = exclude_components
    print(f"[+] Calibration complete. Final component indices excluded: {ica.exclude}\n")
    return ica

def extract_features_single_window(raw_slice: mne.io.Raw, window_len: float = 2.0, pad_len: float = 1.0) -> dict:
    """
    Extracts the 14 EEG flow features from the preprocessed raw slice.
    Uses padding to prevent edge transients for PSD and Hilbert phase.
    """
    sfreq = raw_slice.info['sfreq']
    ch_names = raw_slice.ch_names
    total_samples = raw_slice.n_times
    window_samples = int(window_len * sfreq)
    pad_samples = int(pad_len * sfreq)
    
    # 1. Define channel groupings robustly (matching features.py)
    fmt_ch_names = ['Fz', 'FCz', 'Cz']
    fmt_indices = [ch_names.index(ch) for ch in fmt_ch_names if ch in ch_names]
    
    smr_ch_names = ['C3', 'C4', 'CP3', 'CP4']
    smr_indices = [ch_names.index(ch) for ch in smr_ch_names if ch in ch_names]
    
    left_asym_channels = ['F3', 'F7', 'C3', 'P3']
    right_asym_channels = ['F4', 'F8', 'C4', 'P4']
    left_asym_indices = [ch_names.index(ch) for ch in left_asym_channels if ch in ch_names]
    right_asym_indices = [ch_names.index(ch) for ch in right_asym_channels if ch in ch_names]
    
    entropy_channels = ['AF7', 'AF8']
    entropy_indices = [ch_names.index(ch) for ch in entropy_channels if ch in ch_names]
    
    frontal_plv_channels = ['F3', 'F4', 'Fz']
    parietal_plv_channels = ['P3', 'P4', 'Pz']
    frontal_plv_indices = [ch_names.index(ch) for ch in frontal_plv_channels if ch in ch_names]
    parietal_plv_indices = [ch_names.index(ch) for ch in parietal_plv_channels if ch in ch_names]
    
    # 2. Extract Phase arrays on the entire padded slice to avoid edge transients on the cropped 2.0s
    raw_theta = raw_slice.copy().filter(
        l_freq=4.0, h_freq=8.0, 
        method='iir', iir_params=dict(order=4, ftype='butter'), 
        phase='zero', verbose=False
    )
    analytic_theta = hilbert(raw_theta.get_data())
    phase_theta = np.angle(analytic_theta)
    
    raw_alpha = raw_slice.copy().filter(
        l_freq=8.0, h_freq=12.0, 
        method='iir', iir_params=dict(order=4, ftype='butter'), 
        phase='zero', verbose=False
    )
    analytic_alpha = hilbert(raw_alpha.get_data())
    phase_alpha = np.angle(analytic_alpha)
    
    # 3. Crop signals and phase arrays to the core window (last 2.0s)
    # The raw_slice is window_len + pad_len. We take the last window_len.
    start_sample = total_samples - window_samples
    if start_sample < 0:
        start_sample = 0
        
    window_data = raw_slice.get_data()[:, start_sample:]
    cropped_phase_theta = phase_theta[:, start_sample:]
    cropped_phase_alpha = phase_alpha[:, start_sample:]
    
    # 4. Estimate PSD on window using Welch
    psds, freqs = psd_array_welch(window_data, sfreq=sfreq, fmin=0.5, fmax=45.0, verbose=False)
    
    features = {}
    
    # Feature 1: Frontal Midline Theta (Fm Theta)
    if fmt_indices:
        theta_powers = calculate_relative_power(psds, freqs, 4.0, 8.0)
        features['fm_theta'] = float(np.mean(theta_powers[fmt_indices]))
        
    # Feature 2: SMR Alpha and Beta over Motor Cortex
    for idx in smr_indices:
        ch_name = ch_names[idx].lower()
        ch_psd = psds[idx:idx+1, :]
        features[f'smr_alpha_{ch_name}'] = float(calculate_relative_power(ch_psd, freqs, 8.0, 12.0)[0])
        features[f'smr_beta_{ch_name}'] = float(calculate_relative_power(ch_psd, freqs, 12.0, 30.0)[0])
        
    # Feature 3: Hemispheric Beta Power Asymmetry
    if left_asym_indices and right_asym_indices:
        left_beta = np.mean(calculate_relative_power(psds, freqs, 12.0, 30.0)[left_asym_indices])
        right_beta = np.mean(calculate_relative_power(psds, freqs, 12.0, 30.0)[right_asym_indices])
        denom = left_beta + right_beta
        features['beta_asymmetry'] = float((left_beta - right_beta) / denom if denom > 0 else 0.0)
        
    # Feature 4: Prefrontal Shannon Entropy
    for idx in entropy_indices:
        ch_name = ch_names[idx].lower()
        features[f'entropy_{ch_name}'] = calculate_shannon_entropy(window_data[idx])
        
    # Feature 5: Frontal-Parietal Connectivity (PLV)
    if frontal_plv_indices and parietal_plv_indices:
        # Theta PLV
        theta_plvs = []
        for f_idx in frontal_plv_indices:
            for p_idx in parietal_plv_indices:
                val = calculate_plv(cropped_phase_theta[f_idx], cropped_phase_theta[p_idx])
                theta_plvs.append(val)
        features['plv_theta'] = float(np.mean(theta_plvs))
        
        # Alpha PLV
        alpha_plvs = []
        for f_idx in frontal_plv_indices:
            for p_idx in parietal_plv_indices:
                val = calculate_plv(cropped_phase_alpha[f_idx], cropped_phase_alpha[p_idx])
                alpha_plvs.append(val)
        features['plv_alpha'] = float(np.mean(alpha_plvs))
        
    return features

def simulate_realtime_prediction(subject: str, session: str, bids_root: str, model_path: str, window_len: float = 2.0, pad_len: float = 1.0, step: float = 0.2, max_steps: int = None):
    """
    Runs the real-time simulation loop. Slices EEG data, preprocesses, extracts features, and classifies.
    """
    # 1. Load trained model
    print("=== STEP 1: Loading Classifier ===")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path '{model_path}' does not exist.")
    clf = joblib.load(model_path)
    print(f"[+] Loaded classifier model: {type(clf)}")
    
    # Identify model features expected
    # The feature list must match this order exactly for the scikit-learn model
    feature_cols = [
        'fm_theta', 'smr_alpha_c3', 'smr_beta_c3', 'smr_alpha_c4', 'smr_beta_c4',
        'smr_alpha_cp3', 'smr_beta_cp3', 'smr_alpha_cp4', 'smr_beta_cp4',
        'beta_asymmetry', 'entropy_af7', 'entropy_af8', 'plv_theta', 'plv_alpha'
    ]
    print(f"[*] Expected feature columns: {feature_cols}")
    
    # 2. Load continuous EEG data
    print("\n=== Loading EEG Raw Data ===")
    raw = load_bids_data(subject, session, bids_root)
    sfreq = raw.info['sfreq']
    
    # 3. Baseline ICA Calibration
    ica = calibrate_ica(raw, calibration_duration=60.0)
    
    # 4. Simulation time steps setup
    duration = raw.times[-1]
    # We need a window of size (window_len + pad_len) to start predicting.
    start_time = window_len + pad_len
    
    time_steps = np.arange(start_time, duration, step)
    if max_steps is not None:
        time_steps = time_steps[:max_steps]
        
    print("=== STEP 3: Real-Time Stream Simulation ===")
    print(f"[*] Simulating real-time EEG stream (window={window_len}s, pad={pad_len}s, step={step}s)")
    print(f"[*] Total steps to run: {len(time_steps)}")
    print(f"{'Time (s)':<10} | {'Predicted State':<15} | {'Flow Prob':<10} | {'Latency (ms)':<12}")
    print("-"*60)
    
    # Get raw data numpy arrays and info template to construct RawArray extremely fast
    # Extract channel coordinates and bad channel information
    info_template = raw.info.copy()
    raw_data = raw.get_data()
    
    latencies = []
    
    # Real-time simulation loop
    for t in time_steps:
        loop_start = time.perf_counter()
        
        # 1. Acquire raw window of length (window_len + pad_len)
        start_sec = t - window_len - pad_len
        end_sec = t
        start_samp = int(start_sec * sfreq)
        end_samp = int(end_sec * sfreq)
        
        # Extract EEG data chunk
        slice_data = raw_data[:, start_samp:end_samp]
        
        # Create virtual raw slice in-memory using RawArray (extremely fast, <1ms)
        raw_slice = mne.io.RawArray(slice_data, info_template, verbose=False)
        raw_slice.set_montage('standard_1020', on_missing='ignore', verbose=False)
        
        # 2. Preprocess slice
        # Filter raw slice 0.5 - 45.0 Hz
        raw_slice.filter(
            l_freq=0.5, 
            h_freq=45.0, 
            method='iir', 
            iir_params=dict(order=4, ftype='butter'), 
            phase='zero', 
            verbose=False
        )
        
        # Apply pre-calibrated ICA filter
        ica.apply(raw_slice, verbose=False)
        
        # Interpolate bad channels in slice
        if len(raw_slice.info['bads']) > 0:
            raw_slice.interpolate_bads(reset_bads=True, verbose=False)
            
        # 3. Extract features
        features_dict = extract_features_single_window(raw_slice, window_len, pad_len)
        
        # Format feature vector
        df_features = pd.DataFrame([features_dict], columns=feature_cols)
        
        # 4. Predict
        label_pred = clf.predict(df_features)[0]
        
        # Handle probability extraction depending on model support
        if hasattr(clf, 'predict_proba'):
            prob = clf.predict_proba(df_features)[0]
            flow_prob = prob[1]
        elif hasattr(clf, 'decision_function'):
            # Convert decision function to pseudo-probability via sigmoid
            df_val = clf.decision_function(df_features)[0]
            flow_prob = 1.0 / (1.0 + np.exp(-df_val))
        else:
            flow_prob = 1.0 if label_pred == 1 else 0.0
            
        state_label = "Flow" if label_pred == 1 else "Disrupted"
        
        # 5. Measure latency
        latency_ms = (time.perf_counter() - loop_start) * 1000.0
        latencies.append(latency_ms)
        
        print(f"{t:<10.1f} | {state_label:<15} | {flow_prob:<10.4f} | {latency_ms:<12.2f}")
        
        # Sleep to simulate real-time speed, adjusting for processing latency
        sleep_sec = max(0.0, step - (latency_ms / 1000.0))
        time.sleep(sleep_sec)
        
    print("-"*60)
    print("=== Real-Time Simulation Completed ===")
    print(f"Average latency: {np.mean(latencies):.2f} ms")
    print(f"Max latency:     {np.max(latencies):.2f} ms")
    print(f"95th percentile: {np.percentile(latencies, 95):.2f} ms")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate real-time EEG flow prediction pipeline.")
    parser.add_argument("--model", type=str, required=True, help="Path to the calibrated classifier (.joblib)")
    parser.add_argument("--subject", type=str, default="02", help="Subject ID (e.g. 02)")
    parser.add_argument("--session", type=str, default="EMS", help="Session ID (e.g. EMS, Vibro, Visual)")
    parser.add_argument("--root", type=str, default="data/ds003846", help="BIDS root directory")
    parser.add_argument("--step", type=float, default=0.2, help="Real-time simulated stepping interval in seconds")
    parser.add_argument("--window", type=float, default=2.0, help="Feature extraction window size in seconds")
    parser.add_argument("--pad", type=float, default=1.0, help="Buffer padding size in seconds for edge artifact reduction")
    parser.add_argument("--max-steps", type=int, default=50, help="Limit number of simulation steps (default: 50)")
    args = parser.parse_args()
    
    try:
        simulate_realtime_prediction(
            subject=args.subject,
            session=args.session,
            bids_root=args.root,
            model_path=args.model,
            window_len=args.window,
            pad_len=args.pad,
            step=args.step,
            max_steps=args.max_steps
        )
    except Exception as e:
        print(f"[!] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
