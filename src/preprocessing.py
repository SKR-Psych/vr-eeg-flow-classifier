import argparse
import os
import mne
import numpy as np
from mne.preprocessing import ICA
from src.loader import load_bids_data

def preprocess_raw(raw: mne.io.Raw, ica_n_components: float = 0.99) -> mne.io.Raw:
    """
    Preprocesses raw EEG data following rank-deficiency and dimension-safe best practices:
    1. Identifies bad channels but does NOT interpolate them yet.
    2. Filters a copy of the data at 1.0 - 45.0 Hz and drops bad channels & EOG to fit ICA on clean independent channels.
    3. Filters main raw data at 0.5 - 45.0 Hz.
    4. Fits ICA on 1.0 Hz high-passed copy using PCA-based component count (0.99 variance).
    5. Detects eye/muscle components via EOG correlation and mne-icalabel.
    6. Excludes noise components and applies ICA to main 0.5 Hz filtered raw data.
    7. Interpolates bad channels as the final step.

    Parameters
    ----------
    raw : mne.io.Raw
        The loaded MNE Raw object containing EEG and EOG channels.
    ica_n_components : float or int, optional
        The number of components for ICA. Default is 0.99 (explanations of 99% PCA variance).

    Returns
    -------
    mne.io.Raw
        The preprocessed, clean, and interpolated EEG data.
    """
    # Set standard 10-20 montage to provide electrode coordinates (required by mne-icalabel)
    print("[*] Setting standard 10-20 montage for channel coordinates...")
    raw.set_montage('standard_1020', on_missing='ignore')

    # 1. Identify bad channels
    bads = raw.info['bads']
    print(f"[*] Bad channels identified in dataset: {bads}")

    # 2. Filter copy at 1.0–45.0 Hz for ICA fitting
    print("[*] Creating 1.0 Hz high-passed copy for ICA fitting...")
    raw_ica_fit = raw.copy()
    raw_ica_fit.filter(
        l_freq=1.0, 
        h_freq=45.0, 
        method='iir', 
        iir_params=dict(order=4, ftype='butter'), 
        phase='zero', 
        verbose=False
    )

    # Pick only EEG channels, excluding bads (to maintain correct rank for ICA)
    print("[*] Exclude EOG and bad channels for ICA fit to avoid rank-deficiency...")
    raw_ica_fit.pick(picks='eeg', exclude='bads')

    # 3. Filter main data at 0.5–45.0 Hz
    print("[*] Filtering main raw data at 0.5–45.0 Hz...")
    raw.filter(
        l_freq=0.5, 
        h_freq=45.0, 
        method='iir', 
        iir_params=dict(order=4, ftype='butter'), 
        phase='zero', 
        verbose=False
    )

    # 4. Set up and fit ICA
    print(f"[*] Initializing ICA (n_components={ica_n_components})...")
    # Using 'infomax' with extended=True (more robust for eye and muscle artifacts)
    ica = ICA(
        n_components=ica_n_components, 
        method='infomax', 
        fit_params=dict(extended=True), 
        random_state=97
    )
    
    print("[*] Fitting ICA (this might take a few seconds)...")
    ica.fit(raw_ica_fit, verbose=False)
    print(f"[*] ICA fit completed. Extracted {ica.n_components_} components.")

    # 5. Detect eye components via EOG correlation
    # In this dataset, 'Fp2' is marked as an EOG channel
    eog_channel = 'Fp2'
    eog_exclude = []
    if eog_channel in raw.ch_names:
        print(f"[*] Correlating ICA components with EOG channel '{eog_channel}'...")
        eog_inds, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channel, verbose=False)
        eog_exclude = list(eog_inds)
        print(f"[*] EOG correlation identified component indices: {eog_exclude}")
    else:
        print(f"[!] Warning: EOG channel '{eog_channel}' not found in raw data. Skipping EOG correlation.")

    # 6. Detect eye and muscle components via mne-icalabel
    icalabel_exclude = []
    try:
        from mne_icalabel import label_components
        print("[*] Running mne-icalabel for automatic artifact classification...")
        labels_dict = label_components(raw_ica_fit, ica, method='iclabel')
        labels = labels_dict['labels']
        probabilities = labels_dict['y_pred_proba']
        
        print("\nComponent classification results:")
        for idx, (lbl, prob) in enumerate(zip(labels, probabilities)):
            max_prob = prob[np.argmax(prob)]
            print(f"  - IC {idx:02d}: {lbl:<10} (p={max_prob:.3f})")
            
        # Target 'eye' and 'muscle' components for exclusion
        for idx, lbl in enumerate(labels):
            if lbl in ('eye', 'muscle'):
                icalabel_exclude.append(idx)
        print(f"[*] mne-icalabel flagged component indices: {icalabel_exclude}")
    except Exception as e:
        print(f"[!] mne-icalabel classification skipped/failed: {e}")

    # Combine exclusions
    exclude_components = list(set(eog_exclude + icalabel_exclude))
    exclude_components.sort()
    print(f"[*] Final component indices excluded: {exclude_components}")
    ica.exclude = exclude_components

    # 7. Apply ICA weights to main 0.5-45 Hz data
    print("[*] Applying ICA exclusions to main raw data...")
    ica.apply(raw, verbose=False)

    # 8. Interpolate bad channels at the very end
    if len(bads) > 0:
        print(f"[*] Interpolating bad channels: {bads}...")
        raw.interpolate_bads(reset_bads=True, verbose=False)
    else:
        print("[*] No bad channels to interpolate.")

    print("[*] Preprocessing completed successfully.")
    return raw

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess EEG raw data.")
    parser.add_argument("--subject", type=str, default="02", help="Subject ID (e.g. 02)")
    parser.add_argument("--session", type=str, default="EMS", help="Session ID (e.g. EMS, Vibro, Visual)")
    parser.add_argument("--root", type=str, default="data/ds003846", help="BIDS root directory")
    args = parser.parse_args()

    try:
        # Load raw BIDS data
        raw = load_bids_data(args.subject, args.session, args.root)
        
        # Calculate raw signal variance (std dev) for comparison
        print("[*] Calculating initial signal statistics...")
        eeg_channels = [ch for ch in raw.ch_names if raw.get_channel_types()[raw.ch_names.index(ch)] == 'eeg']
        raw_eeg_data = raw.copy().pick(picks='eeg')
        initial_std = np.std(raw_eeg_data.get_data())
        
        # Run preprocessing
        clean_raw = preprocess_raw(raw, ica_n_components=0.99)
        
        # Calculate clean signal variance (std dev)
        clean_eeg_data = clean_raw.copy().pick(picks='eeg')
        final_std = np.std(clean_eeg_data.get_data())
        
        # Calculate change in standard deviation (should be lower due to noise removal)
        reduction = (initial_std - final_std) / initial_std * 100
        
        print("\n=== Preprocessing Verification Summary ===")
        print(f"Subject / Session:   sub-{args.subject} / ses-{args.session}")
        print(f"Initial EEG Std Dev: {initial_std:.3e}")
        print(f"Clean EEG Std Dev:   {final_std:.3e}")
        print(f"EEG Amplitude Reduc: {reduction:.2f}%")
        
        # Check that bad channels list is now empty in clean raw
        print(f"Post-clean raw.info['bads']: {clean_raw.info['bads']}")
        
    except Exception as e:
        print(f"[!] Failed preprocessing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
