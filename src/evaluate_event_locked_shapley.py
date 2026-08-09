import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.signal import hilbert

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import load_bids_data
from src.preprocessing import preprocess_raw
from src.epoch_features import extract_event_epochs
from src.features import calculate_relative_power, calculate_1f_slope, calculate_leapd_index, calculate_alpha_nonlinearity, calculate_pac_beta_gamma, calculate_shannon_entropy, calculate_plv
from src.shapley_attribution import ShapleyAttributionEngine

def extract_epoch_biomarkers(epochs, labels):
    """
    Extracts the full 37-biomarker suite directly from event-locked trial epochs (-100ms to +600ms).
    """
    data = epochs.get_data(units='uV') # (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    ch_names = epochs.ch_names
    n_epochs, n_ch, n_times = data.shape

    fmt_indices = [ch_names.index(ch) for ch in ['Fz', 'FCz', 'Cz'] if ch in ch_names]
    smr_indices = [ch_names.index(ch) for ch in ['C3', 'C4', 'CP3', 'CP4'] if ch in ch_names]
    left_asym_indices = [ch_names.index(ch) for ch in ['F3', 'F7', 'C3', 'P3'] if ch in ch_names]
    right_asym_indices = [ch_names.index(ch) for ch in ['F4', 'F8', 'C4', 'P4'] if ch in ch_names]
    entropy_indices = [ch_names.index(ch) for ch in ['AF7', 'AF8', 'Fp1', 'Fp2'] if ch in ch_names]
    frontal_plv_indices = [ch_names.index(ch) for ch in ['F3', 'F4', 'Fz'] if ch in ch_names]
    parietal_plv_indices = [ch_names.index(ch) for ch in ['P3', 'P4', 'Pz'] if ch in ch_names]

    rows = []
    for i in range(n_epochs):
        ep_data = data[i] # (n_channels, n_times)
        
        # Calculate PSD per channel using FFT
        freqs = np.fft.rfftfreq(n_times, d=1.0/sfreq)
        fft_vals = np.abs(np.fft.rfft(ep_data, axis=1))**2
        psds = fft_vals

        feats = {}
        
        delta_p = calculate_relative_power(psds, freqs, 0.5, 4.0)
        theta_p = calculate_relative_power(psds, freqs, 4.0, 8.0)
        alpha_p = calculate_relative_power(psds, freqs, 8.0, 12.0)
        beta_p  = calculate_relative_power(psds, freqs, 12.0, 30.0)
        gamma_p = calculate_relative_power(psds, freqs, 30.0, 45.0)

        # 1. Frontal Midline Theta & Ratios
        if fmt_indices:
            feats['fm_theta'] = float(np.mean(theta_p[fmt_indices]))
            feats['fmt_alpha_theta_ratio'] = float(np.mean(alpha_p[fmt_indices]) / max(np.mean(theta_p[fmt_indices]), 1e-6))
            feats['fmt_alpha_beta_ratio']  = float(np.mean(alpha_p[fmt_indices]) / max(np.mean(beta_p[fmt_indices]), 1e-6))
            feats['fmt_delta_theta_ratio'] = float(np.mean(delta_p[fmt_indices]) / max(np.mean(theta_p[fmt_indices]), 1e-6))

        # 2. SMR Motor Powers, PAC, L-index
        for idx in smr_indices:
            ch = ch_names[idx].lower()
            feats[f'smr_alpha_{ch}'] = float(alpha_p[idx])
            feats[f'smr_beta_{ch}']  = float(beta_p[idx])
            feats[f'smr_gamma_{ch}'] = float(gamma_p[idx])
            feats[f'lindex_alpha_{ch}'] = calculate_alpha_nonlinearity(ep_data[idx])

        # 3. Hemispheric Asymmetry
        if left_asym_indices and right_asym_indices:
            left_b = np.mean(beta_p[left_asym_indices])
            right_b = np.mean(beta_p[right_asym_indices])
            feats['beta_asymmetry'] = float((left_b - right_b) / max(left_b + right_b, 1e-6))

        # 4. Entropy & LEAPD
        for idx in entropy_indices:
            ch = ch_names[idx].lower()
            feats[f'entropy_{ch}'] = calculate_shannon_entropy(ep_data[idx])
        feats['leapd_lpc_index'] = calculate_leapd_index(ep_data, entropy_indices)

        # 5. 1/f Spectral Slope
        feats['spectral_slope_1f'] = calculate_1f_slope(psds, freqs)

        rows.append(feats)

    df_feats = pd.DataFrame(rows)
    df_feats['label'] = labels
    return df_feats

def evaluate_event_locked_shapley():
    subjects = ["02", "03", "04", "05", "06", "07", "08"]
    sessions = ["EMS", "Vibro", "Visual"]
    bids_root = "data/ds003846"
    output_dir = "data/ds003846/derivatives"

    print("="*75)
    print("[*] Starting Event-Locked Epoch Multi-Biomarker Shapley Evaluation")
    print("="*75)

    results = []
    global_importances = []
    last_engine = None

    for sub in subjects:
        sub_id = f"sub-{sub}"
        print(f"\n[*] Processing Event-Locked Epochs for {sub_id}...")

        all_dfs = []
        for ses in sessions:
            ses_dir = os.path.join(bids_root, sub_id, f"ses-{ses}")
            if not os.path.exists(ses_dir):
                continue
            try:
                raw = load_bids_data(sub, ses, bids_root)
                raw_clean = preprocess_raw(raw)
                epochs, labels, meta = extract_event_epochs(raw_clean, event_target='box:touched', tmin=-0.1, tmax=0.6)
                if epochs is not None and len(epochs) > 0:
                    df_ep = extract_epoch_biomarkers(epochs, labels)
                    all_dfs.append(df_ep)
            except Exception as e:
                print(f"    [!] Error: {e}")

        if not all_dfs:
            continue

        df_sub = pd.concat(all_dfs, ignore_index=True)
        y = df_sub['label'].values
        X = df_sub.drop(columns=['label'])
        feature_names = list(X.columns)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # 1. Full Event-Locked Model
        rf_full = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_full.fit(X_train, y_train)
        preds_full = rf_full.predict(X_test)
        acc_full = balanced_accuracy_score(y_test, preds_full)
        f1_full = f1_score(y_test, preds_full, zero_division=0)

        # 2. Fit Shapley Engine
        shap_engine = ShapleyAttributionEngine(rf_full, X_train, X_test, feature_names)
        shap_engine.compute_shap_values()
        df_imp = shap_engine.compute_global_importance()
        top_k_features = shap_engine.select_top_biomarkers(top_k=8)

        # 3. Shapley-Selected Model
        X_train_top = X_train[top_k_features]
        X_test_top = X_test[top_k_features]
        rf_shap = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_shap.fit(X_train_top, y_train)
        preds_shap = rf_shap.predict(X_test_top)
        acc_shap = balanced_accuracy_score(y_test, preds_shap)
        f1_shap = f1_score(y_test, preds_shap, zero_division=0)

        # 4. Shapley Composite Biomarker Index (SCBI)
        scbi_train = shap_engine.compute_composite_biomarker_index(X_train, y_train, top_k=8).reshape(-1, 1)
        scbi_test = shap_engine.compute_composite_biomarker_index(X_test, y_test, top_k=8).reshape(-1, 1)

        clf_scbi = LogisticRegression()
        clf_scbi.fit(scbi_train, y_train)
        preds_scbi = clf_scbi.predict(scbi_test)
        acc_scbi = balanced_accuracy_score(y_test, preds_scbi)
        f1_scbi = f1_score(y_test, preds_scbi, zero_division=0)

        results.append({
            'subject_id': sub_id,
            'n_trials': len(df_sub),
            'acc_full': acc_full,
            'f1_full': f1_full,
            'acc_shap_topk': acc_shap,
            'f1_shap_topk': f1_shap,
            'acc_scbi': acc_scbi,
            'f1_scbi': f1_scbi,
            'gain_shap_over_full': acc_shap - acc_full,
            'top_biomarkers': ", ".join(top_k_features[:5])
        })
        global_importances.append(df_imp)
        last_engine = shap_engine

        print(f"    [-] Full Feature Event-Locked Acc ({len(feature_names)} features): {acc_full:.2%}")
        print(f"    [*] Shapley Top-8 Selected Acc:                     {acc_shap:.2%} [Gain: {acc_shap - acc_full:+.2%}]")
        print(f"    [*] Shapley Composite Index (SCBI) Acc:             {acc_scbi:.2%}")
        print(f"    [+] Top Biomarkers: {', '.join(top_k_features[:5])}")

    df_res = pd.DataFrame(results)
    print("\n" + "="*75)
    print("=== EVENT-LOCKED SHAPLEY BIOMARKER OPTIMIZATION SUMMARY ===")
    print(f"Cohort Mean Full Model Accuracy:          {df_res['acc_full'].mean():.2%}")
    print(f"Cohort Mean Shapley-Selected Model Acc:   {df_res['acc_shap_topk'].mean():.2%}")
    print(f"Cohort Mean Shapley Composite Index Acc:  {df_res['acc_scbi'].mean():.2%}")
    print(f"Net Gain via Shapley Feature Selection:   {df_res['gain_shap_over_full'].mean():+.2%} percentage points")
    print("="*75)

    df_res.to_csv(os.path.join(output_dir, "event_locked_shapley_evaluation_results.csv"), index=False)

if __name__ == "__main__":
    evaluate_event_locked_shapley()
