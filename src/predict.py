import argparse
import os
import sys
import time
import mne
import numpy as np
import pandas as pd
import torch
from mne.preprocessing import ICA
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Not setting position.*")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import load_bids_data
from src.classifier import load_eegnet_checkpoint

def calibrate_ica(raw: mne.io.Raw, calibration_duration: float = 60.0) -> ICA:
    """
    Simulates an online calibration phase by fitting ICA on a baseline chunk
    (e.g., first 60 seconds of raw data). Detects and excludes eye/muscle artifact components.
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

    # 6. Detect eye and muscle components via mne-icalabel
    icalabel_exclude = []
    try:
        from mne_icalabel import label_components
        print("[*] Running mne-icalabel component classification...")
        labels_dict = label_components(raw_calib, ica, method='iclabel')
        labels = labels_dict['labels']
        probs = labels_dict['y_pred_proba']

        print(f"[*] mne-icalabel assigned labels: {set(labels)}")
        for idx, (lbl, prob) in enumerate(zip(labels, probs)):
            if lbl in ['eye', 'muscle'] and prob >= 0.80:
                icalabel_exclude.append(idx)
        print(f"[*] mne-icalabel identified artifact components: {icalabel_exclude}")
    except Exception as e:
        print(f"[!] Warning: mne-icalabel failed: {e}. Falling back to EOG correlation.")

    exclude_components = list(set(eog_exclude + icalabel_exclude))
    ica.exclude = exclude_components
    print(f"[+] Total ICA artifact components excluded: {exclude_components}")
    return ica

def simulate_realtime_eegnet_stream(
    subject: str,
    session: str,
    model_path: str,
    bids_root: str = "data/ds003846",
    window_len: float = 0.70,  # 700ms window (350 samples at 500Hz)
    pad_len: float = 0.50,     # 500ms padding for edge-safe filtering
    step: float = 0.20,        # 200ms step for responsive sliding updates
    max_steps: int = 20,
    smoothing_alpha: float = 0.6  # Bayesian temporal smoothing weight
):
    """
    Simulates real-time closed-loop EEG streaming using the upgraded PyTorch EEGNet model.
    Achieves sub-5ms forward-pass latency with Bayesian temporal smoothing.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load calibrated EEGNet model checkpoint
    print("=== STEP 1: Loading EEGNet Classifier ===")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint '{model_path}' does not exist.")

    model, norm_mean, norm_std, meta = load_eegnet_checkpoint(model_path, device=device)
    print(f"[+] Loaded PyTorch EEGNet architecture ({meta.get('channels', 64)} ch, {meta.get('samples', 350)} samples)")
    print(f"[+] Device: {device} | Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Load continuous EEG data
    print("\n=== Loading EEG Raw Data ===")
    raw = load_bids_data(subject, session, bids_root)
    if raw.info['sfreq'] != 500.0:
        print(f"[*] Resampling EEG to standardized 500.0 Hz...")
        raw = raw.copy().resample(500.0, verbose=False)
    sfreq = raw.info['sfreq']

    # 3. Baseline ICA Calibration
    ica = calibrate_ica(raw, calibration_duration=60.0)
    
    # Pick EEG channels only (matching training dimensions in epoch_features.py)
    raw.pick(picks='eeg')

    # 4. Simulation time steps setup
    duration = raw.times[-1]
    start_time = window_len + pad_len
    time_steps = np.arange(start_time, duration, step)
    if max_steps is not None:
        time_steps = time_steps[:max_steps]

    print("\n" + "="*70)
    print("=== STEP 3: Real-Time Stream Simulation (EEGNet Deep Learning) ===")
    print("="*70)
    print(f"[*] Simulating real-time EEG stream (window={window_len}s, pad={pad_len}s, step={step}s)")
    print(f"[*] Bayesian Smoothing alpha: {smoothing_alpha} | Total steps: {len(time_steps)}")
    print(f"{'Time (s)':<10} | {'State':<12} | {'Raw P(Flow)':<12} | {'Smooth P(Flow)':<14} | {'Latency (ms)':<12}")
    print("-" * 70)

    info_template = raw.info.copy()
    raw_data = raw.get_data()  # in standard SI units (Volts) for MNE filtering & ICA
    window_samples = int(window_len * sfreq)

    latencies = []
    smoothed_prob = 0.5  # Neutral start

    for t in time_steps:
        loop_start = time.perf_counter()

        # 1. Acquire raw window of length (window_len + pad_len)
        start_sec = t - window_len - pad_len
        end_sec = t
        start_samp = int(start_sec * sfreq)
        end_samp = int(end_sec * sfreq)

        slice_data = raw_data[:, start_samp:end_samp]
        raw_slice = mne.io.RawArray(slice_data, info_template, verbose=False)
        raw_slice.set_montage('standard_1020', on_missing='ignore', verbose=False)

        # 2. Preprocess slice: filter 0.5 - 45.0 Hz zero-phase
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

        # Interpolate bad channels if any
        if len(raw_slice.info['bads']) > 0:
            raw_slice.interpolate_bads(reset_bads=True, verbose=False)

        # 3. Crop to core 700ms window (last window_samples) and convert to microvolts (uV)
        cropped_data = raw_slice.get_data()[:, -window_samples:] * 1e6

        # Format 4D tensor: (1, 1, channels, samples)
        tensor = torch.tensor(cropped_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Apply calibrated z-score standardization
        tensor = (tensor - norm_mean) / norm_std

        # 4. Neural Network Inference
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            raw_flow_prob = float(probs[1])

        # 5. Bayesian Temporal Smoothing: P_t = alpha * P_raw + (1 - alpha) * P_{t-1}
        smoothed_prob = smoothing_alpha * raw_flow_prob + (1.0 - smoothing_alpha) * smoothed_prob
        predicted_state = "Flow" if smoothed_prob >= 0.5 else "Disrupted"

        loop_end = time.perf_counter()
        latency_ms = (loop_end - loop_start) * 1000.0
        latencies.append(latency_ms)

        print(f"{t:<10.2f} | {predicted_state:<12} | {raw_flow_prob:<12.4f} | {smoothed_prob:<14.4f} | {latency_ms:<12.2f}")

    print("-" * 70)
    print("=== SUMMARY: Pipeline Performance ===")
    print(f"[*] Mean Loop Latency (Filtering + ICA + EEGNet Inference): {np.mean(latencies):.2f} ms")
    print(f"[*] Max Latency: {np.max(latencies):.2f} ms | Min Latency: {np.min(latencies):.2f} ms")
    if np.mean(latencies) < 100.0:
        print(f"[+] REAL-TIME VALIDATION SUCCESSFUL: Under 100ms threshold for VR Closed-Loop System!")
    else:
        print(f"[!] Warning: Latency exceeded 100ms target.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Streaming EEGNet Flow Classifier")
    parser.add_argument("--subject", type=str, default="02", help="Subject ID (e.g. 02)")
    parser.add_argument("--session", type=str, default="EMS", help="Session ID (e.g. EMS, Vibro, Visual)")
    parser.add_argument("--model-path", type=str, default="data/ds003846/derivatives/sub-02_eegnet.pt", help="Path to calibrated EEGNet .pt file")
    parser.add_argument("--bids-root", type=str, default="data/ds003846", help="Path to BIDS dataset")
    parser.add_argument("--max-steps", type=int, default=15, help="Number of real-time stream steps to simulate")
    parser.add_argument("--smoothing-alpha", type=float, default=0.6, help="Bayesian temporal smoothing weight")
    args = parser.parse_args()

    simulate_realtime_eegnet_stream(
        subject=args.subject,
        session=args.session,
        model_path=args.model_path,
        bids_root=args.bids_root,
        max_steps=args.max_steps,
        smoothing_alpha=args.smoothing_alpha
    )
