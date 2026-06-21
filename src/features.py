import argparse
import os
import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from mne.time_frequency import psd_array_welch
from src.preprocessing import preprocess_raw
from src.loader import load_bids_data

def calculate_shannon_entropy(signal: np.ndarray, num_bins: int = 50) -> float:
    """
    Calculates the time-domain Shannon entropy of a 1D signal.
    Applies Z-score standardization first, and bins amplitudes in range (-3, 3).
    """
    std_val = np.std(signal)
    if std_val == 0:
        return 0.0
    z_signal = (signal - np.mean(signal)) / std_val
    
    # Calculate probability distribution
    counts, _ = np.histogram(z_signal, bins=num_bins, range=(-3, 3))
    probs = counts / len(signal)
    probs = probs[probs > 0]  # Filter out zero probabilities to avoid log2(0)
    
    return float(-np.sum(probs * np.log2(probs)))

def calculate_plv(phase1: np.ndarray, phase2: np.ndarray) -> float:
    """
    Computes the Phase Locking Value (PLV) from pre-calculated phase arrays.
    """
    phase_diff = phase1 - phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return float(plv)

def calculate_relative_power(psds: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """
    Calculates the relative band power from PSD estimation.
    Relative power = (power in band) / (total power from 0.5 to 45 Hz).
    """
    band_idx = (freqs >= fmin) & (freqs <= fmax)
    band_power = np.sum(psds[:, band_idx], axis=1)
    total_power = np.sum(psds, axis=1)
    # Avoid division by zero
    total_power[total_power == 0] = 1.0
    return band_power / total_power

def extract_epoch_features(raw: mne.io.Raw, window_len: float = 2.0, step: float = 2.0) -> pd.DataFrame:
    """
    Segments raw EEG data into sliding windows and extracts flow-related biomarkers.
    Optimizes phase calculations by pre-computing Hilbert phases on continuous data.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed clean MNE Raw data.
    window_len : float, optional
        Length of the window in seconds. Default is 2.0.
    step : float, optional
        Step size between windows in seconds. Default is 2.0.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing timestamps and extracted feature columns.
    """
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    total_samples = raw.n_times
    window_samples = int(window_len * sfreq)
    step_samples = int(step * sfreq)
    
    print(f"[*] Extracting features: window={window_len}s ({window_samples} samples), step={step}s ({step_samples} samples)")

    # Define channel groupings robustly
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

    # Precompute instantaneous phases on the entire continuous raw data to protect against edge transients
    print("[*] Pre-computing Hilbert phases for Theta (4-8 Hz)...")
    raw_theta = raw.copy().filter(
        l_freq=4.0, h_freq=8.0, 
        method='iir', iir_params=dict(order=4, ftype='butter'), 
        phase='zero', verbose=False
    )
    # hilbert operates along the last axis by default (samples)
    analytic_theta = hilbert(raw_theta.get_data())
    phase_theta = np.angle(analytic_theta)
    del raw_theta, analytic_theta  # Free memory
    
    print("[*] Pre-computing Hilbert phases for Alpha (8-12 Hz)...")
    raw_alpha = raw.copy().filter(
        l_freq=8.0, h_freq=12.0, 
        method='iir', iir_params=dict(order=4, ftype='butter'), 
        phase='zero', verbose=False
    )
    analytic_alpha = hilbert(raw_alpha.get_data())
    phase_alpha = np.angle(analytic_alpha)
    del raw_alpha, analytic_alpha  # Free memory

    data_list = []
    
    # Slide window
    for start in range(0, total_samples - window_samples + 1, step_samples):
        end = start + window_samples
        timestamp = start / sfreq
        
        # Get raw data slice
        window_data, _ = raw[:, start:end]
        
        # Estimate PSD on window using Welch
        psds, freqs = psd_array_welch(window_data, sfreq=sfreq, fmin=0.5, fmax=45.0, verbose=False)
        
        features = {'timestamp': timestamp}
        
        # 1. Frontal Midline Theta (Fm Theta)
        if fmt_indices:
            theta_powers = calculate_relative_power(psds, freqs, 4.0, 8.0)
            features['fm_theta'] = float(np.mean(theta_powers[fmt_indices]))
            
        # 2. SMR Alpha and Beta over Motor Cortex
        for idx in smr_indices:
            ch_name = ch_names[idx].lower()
            ch_psd = psds[idx:idx+1, :]
            features[f'smr_alpha_{ch_name}'] = float(calculate_relative_power(ch_psd, freqs, 8.0, 12.0)[0])
            features[f'smr_beta_{ch_name}'] = float(calculate_relative_power(ch_psd, freqs, 12.0, 30.0)[0])
            
        # 3. Hemispheric Beta Power Asymmetry
        if left_asym_indices and right_asym_indices:
            left_beta = np.mean(calculate_relative_power(psds, freqs, 12.0, 30.0)[left_asym_indices])
            right_beta = np.mean(calculate_relative_power(psds, freqs, 12.0, 30.0)[right_asym_indices])
            denom = left_beta + right_beta
            features['beta_asymmetry'] = float((left_beta - right_beta) / denom if denom > 0 else 0.0)
            
        # 4. Prefrontal Shannon Entropy
        for idx in entropy_indices:
            ch_name = ch_names[idx].lower()
            features[f'entropy_{ch_name}'] = calculate_shannon_entropy(window_data[idx])
            
        # 5. Frontal-Parietal Connectivity (PLV)
        if frontal_plv_indices and parietal_plv_indices:
            # Theta PLV
            theta_plvs = []
            for f_idx in frontal_plv_indices:
                for p_idx in parietal_plv_indices:
                    val = calculate_plv(phase_theta[f_idx, start:end], phase_theta[p_idx, start:end])
                    theta_plvs.append(val)
            features['plv_theta'] = float(np.mean(theta_plvs))
            
            # Alpha PLV
            alpha_plvs = []
            for f_idx in frontal_plv_indices:
                for p_idx in parietal_plv_indices:
                    val = calculate_plv(phase_alpha[f_idx, start:end], phase_alpha[p_idx, start:end])
                    alpha_plvs.append(val)
            features['plv_alpha'] = float(np.mean(alpha_plvs))
            
        data_list.append(features)
        
    return pd.DataFrame(data_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract EEG biomarkers from BIDS dataset.")
    parser.add_argument("--subject", type=str, default="02", help="Subject ID (e.g. 02)")
    parser.add_argument("--session", type=str, default="EMS", help="Session ID (e.g. EMS, Vibro, Visual)")
    parser.add_argument("--root", type=str, default="data/ds003846", help="BIDS root directory")
    parser.add_argument("--window", type=float, default=2.0, help="Sliding window size in seconds")
    parser.add_argument("--step", type=float, default=2.0, help="Sliding step size in seconds")
    args = parser.parse_args()
    
    try:
        # Load raw BIDS data
        raw = load_bids_data(args.subject, args.session, args.root)
        
        # Preprocess the data using rank-safe pipeline
        print("[*] Running preprocessing clean pipeline...")
        raw_clean = preprocess_raw(raw)
        
        # Extract features
        df_features = extract_epoch_features(raw_clean, window_len=args.window, step=args.step)
        
        print("\n=== Feature Extraction Verification ===")
        print(f"Extracted {df_features.shape[0]} windows with {df_features.shape[1]} features.")
        print(f"Features DataFrame shape: {df_features.shape}")
        print("\nFeature statistics (mean):")
        for col in df_features.columns:
            if col != 'timestamp':
                print(f"  - {col:<20} : {df_features[col].mean():.5f}")
                
        # Save to CSV in data directory
        os.makedirs(os.path.join(args.root, "derivatives"), exist_ok=True)
        out_csv = os.path.join(args.root, "derivatives", f"sub-{args.subject}_ses-{args.session}_features.csv")
        df_features.to_csv(out_csv, index=False)
        print(f"\n[*] Extracted features successfully saved to: {out_csv}")
        
    except Exception as e:
        print(f"[!] Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
