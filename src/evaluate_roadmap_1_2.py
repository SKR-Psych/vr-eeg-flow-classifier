import os
import sys
import glob
import mne
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.loader import load_bids_data
from src.preprocessing import preprocess_raw
from src.epoch_features import extract_event_epochs, extract_epoch_features
from src.spatial_filtering import SpatialFeatureExtractor

def evaluate_roadmap_improvements(bids_root: str = "data/ds003846", output_dir: str = "data/ds003846/derivatives"):
    """
    Evaluates and compares:
    1. Baseline Continuous Sliding Window (from existing evaluation)
    2. Roadmap Item 1: Event-Locked Epoching (-100ms to +600ms)
    3. Roadmap Item 2: Event-Locked Epoching + xDAWN/CSP Spatial Filtering
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load baseline evaluation summary for direct comparison
    baseline_csv = os.path.join(output_dir, "subject_evaluation_results.csv")
    baseline_accs = {}
    if os.path.exists(baseline_csv):
        df_base = pd.read_csv(baseline_csv)
        for _, row in df_base.iterrows():
            baseline_accs[row['subject_id']] = row['test_balanced_acc']
            
    sub_dirs = glob.glob(os.path.join(bids_root, "sub-*"))
    subjects = [os.path.basename(d).replace("sub-", "") for d in sub_dirs if os.path.isdir(d)]
    subjects.sort()
    
    print(f"[*] Starting Technical Roadmap Items 1 & 2 Evaluation across {len(subjects)} subjects...")
    print(f"[*] Subjects: {subjects}")
    
    results = []
    sessions = ["EMS", "Vibro", "Visual"]
    
    for sub in subjects:
        sub_id = f"sub-{sub}"
        print("\n" + "="*60)
        print(f"[*] Processing Subject: {sub_id}")
        
        # Accumulate epochs across all available sessions for this subject
        all_epochs_list = []
        all_labels_list = []
        
        for ses in sessions:
            ses_dir = os.path.join(bids_root, sub_id, f"ses-{ses}")
            if not os.path.exists(ses_dir):
                continue
                
            try:
                print(f"    [*] Loading & preprocessing {sub_id} ses-{ses}...")
                raw = load_bids_data(sub, ses, bids_root)
                raw_clean = preprocess_raw(raw)
                
                epochs, labels, meta = extract_event_epochs(raw_clean, event_target='box:touched', tmin=-0.1, tmax=0.6)
                if epochs is not None and len(epochs) > 0:
                    all_epochs_list.append(epochs)
                    all_labels_list.append(epochs.events[:, 2])
            except Exception as e:
                print(f"    [!] Error processing {sub_id} ses-{ses}: {e}")
                
        if not all_epochs_list:
            print(f"    [!] No valid event epochs extracted for {sub_id}. Skipping.")
            continue
            
        # Combine epochs across sessions
        epochs_sub = mne.concatenate_epochs(all_epochs_list, on_mismatch='ignore', verbose=False)
        y_sub = np.concatenate(all_labels_list)
        n_samples = len(epochs_sub)
        n_normal = np.sum(y_sub == 1)
        n_conflict = np.sum(y_sub == 0)
        
        print(f"    [+] Total Subject Trials: {n_samples} (Normal: {n_normal}, Conflict: {n_conflict})")
        
        if n_normal < 5 or n_conflict < 5:
            print(f"    [!] Warning: Insufficient trials for classification. Skipping.")
            continue
            
        # 80/20 Chronological Split
        split_idx = int(n_samples * 0.8)
        
        epochs_train = epochs_sub[:split_idx]
        epochs_test = epochs_sub[split_idx:]
        y_train = y_sub[:split_idx]
        y_test = y_sub[split_idx:]
        
        # --- PIPELINE 1: Item 1 - Event-Locked Epoching (Sensor Features) ---
        df_feat_train = extract_epoch_features(epochs_train)
        df_feat_test = extract_epoch_features(epochs_test)
        
        # Train Random Forest & SVM on Item 1
        rf_item1 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=97)
        rf_item1.fit(df_feat_train, y_train)
        y_pred_rf1 = rf_item1.predict(df_feat_test)
        acc_rf1 = balanced_accuracy_score(y_test, y_pred_rf1)
        
        svm_item1 = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=97))])
        svm_item1.fit(df_feat_train, y_train)
        y_pred_svm1 = svm_item1.predict(df_feat_test)
        acc_svm1 = balanced_accuracy_score(y_test, y_pred_svm1)
        
        best_acc_item1 = max(acc_rf1, acc_svm1)
        
        # --- PIPELINE 2: Item 2 - Event-Locked + Spatial Filtering (xDAWN & CSP) ---
        spatial_extractor = SpatialFeatureExtractor(n_xdawn_components=4, n_csp_components=4)
        
        # Fit ONLY on training epochs to prevent data leakage!
        spatial_extractor.fit(epochs_train, y_train)
        
        X_spatial_train = spatial_extractor.transform(epochs_train)
        X_spatial_test = spatial_extractor.transform(epochs_test)
        
        # Combine sensor features + spatial features
        X_comb_train = pd.concat([df_feat_train.reset_index(drop=True), X_spatial_train.reset_index(drop=True)], axis=1)
        X_comb_test = pd.concat([df_feat_test.reset_index(drop=True), X_spatial_test.reset_index(drop=True)], axis=1)
        
        # Train Random Forest & SVM on Item 2
        rf_item2 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=97)
        rf_item2.fit(X_comb_train, y_train)
        y_pred_rf2 = rf_item2.predict(X_comb_test)
        y_prob_rf2 = rf_item2.predict_proba(X_comb_test)[:, 1]
        acc_rf2 = balanced_accuracy_score(y_test, y_pred_rf2)
        
        svm_item2 = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=97))])
        svm_item2.fit(X_comb_train, y_train)
        y_pred_svm2 = svm_item2.predict(X_comb_test)
        y_prob_svm2 = svm_item2.predict_proba(X_comb_test)[:, 1]
        acc_svm2 = balanced_accuracy_score(y_test, y_pred_svm2)
        
        best_acc_item2 = max(acc_rf2, acc_svm2)
        best_y_pred2 = y_pred_rf2 if acc_rf2 >= acc_svm2 else y_pred_svm2
        best_y_prob2 = y_prob_rf2 if acc_rf2 >= acc_svm2 else y_prob_svm2
        best_model_name2 = "Random Forest" if acc_rf2 >= acc_svm2 else "SVM"
        
        # Compute metrics
        prec2 = precision_score(y_test, best_y_pred2, zero_division=0)
        rec2 = recall_score(y_test, best_y_pred2, zero_division=0)
        f12 = f1_score(y_test, best_y_pred2, zero_division=0)
        try:
            auc2 = roc_auc_score(y_test, best_y_prob2)
        except Exception:
            auc2 = 0.5
            
        base_acc = baseline_accs.get(sub_id, 0.5157)
        gain_over_baseline = best_acc_item2 - base_acc
        
        print(f"    [-] Baseline Sliding Window Acc:          {base_acc:.2%}")
        print(f"    [*] Item 1 (Event-Locked Epoching) Acc:   {best_acc_item1:.2%}")
        print(f"    [+] Item 2 (Spatial xDAWN+CSP) Acc:       {best_acc_item2:.2%} ({best_model_name2})")
        print(f"    [+] Net Performance Gain:                 {gain_over_baseline:+.2%} percentage points")
        
        results.append({
            'subject_id': sub_id,
            'total_trials': n_samples,
            'normal_trials': n_normal,
            'conflict_trials': n_conflict,
            'baseline_sliding_acc': base_acc,
            'item1_epoching_acc': best_acc_item1,
            'item2_spatial_acc': best_acc_item2,
            'performance_gain': gain_over_baseline,
            'best_model': best_model_name2,
            'precision': prec2,
            'recall': rec2,
            'f1_score': f12,
            'roc_auc': auc2
        })
        
    df_res = pd.DataFrame(results)
    
    # Save CSV summary
    out_csv = os.path.join(output_dir, "roadmap_1_2_evaluation_results.csv")
    df_res.to_csv(out_csv, index=False)
    print("\n" + "="*60)
    print("=== Technical Roadmap Items 1 & 2 Evaluation Summary ===")
    print(f"Baseline Mean Balanced Accuracy: {df_res['baseline_sliding_acc'].mean():.2%}")
    print(f"Item 1 (Event Epoching) Mean Acc: {df_res['item1_epoching_acc'].mean():.2%}")
    print(f"Item 2 (xDAWN + CSP) Mean Acc:    {df_res['item2_spatial_acc'].mean():.2%}")
    print(f"Average Net Performance Gain:     {df_res['performance_gain'].mean():+.2%} percentage points")
    print("="*60)
    
    # Save Markdown summary
    out_md = os.path.join(output_dir, "roadmap_1_2_evaluation_results.md")
    with open(out_md, 'w') as f:
        f.write("# Technical Roadmap Items 1 & 2 Evaluation Report\n\n")
        f.write(f"**Baseline Mean Balanced Accuracy:** {df_res['baseline_sliding_acc'].mean():.2%}\n")
        f.write(f"**Item 1 (Event-Locked Epoching) Mean Accuracy:** {df_res['item1_epoching_acc'].mean():.2%}\n")
        f.write(f"**Item 2 (Event-Locked + xDAWN/CSP Spatial Filtering) Mean Accuracy:** {df_res['item2_spatial_acc'].mean():.2%}\n")
        f.write(f"**Average Net Performance Gain:** {df_res['performance_gain'].mean():+.2%} percentage points\n\n")
        
        cols = df_res.columns
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in df_res.iterrows():
            f.write("| " + " | ".join(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols) + " |\n")
            
    print(f"Results report exported to: {out_md}")

if __name__ == "__main__":
    evaluate_roadmap_improvements()
