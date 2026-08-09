import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import load_bids_data
from src.preprocessing import preprocess_raw
from src.features import extract_epoch_features
from src.classifier import parse_desc
from src.shapley_attribution import ShapleyAttributionEngine

def evaluate_subject_shapley(subject: str, bids_root: str = "data/ds003846"):
    """
    Loads raw sessions for a single subject, extracts features, performs trial alignment,
    and runs Shapley attribution and feature selection comparison.
    """
    sessions = ["EMS", "Vibro", "Visual"]
    sub_id = f"sub-{subject}"

    all_dfs = []
    
    for ses in sessions:
        ses_dir = os.path.join(bids_root, sub_id, f"ses-{ses}")
        if not os.path.exists(ses_dir):
            continue
        try:
            cache_csv = os.path.join(bids_root, "derivatives", f"{sub_id}_ses-{ses}_expanded_features.csv")
            raw = load_bids_data(subject, ses, bids_root)
            if os.path.exists(cache_csv):
                print(f"    [*] Loading cached expanded features from: {cache_csv}")
                df_feat = pd.read_csv(cache_csv)
            else:
                raw_clean = preprocess_raw(raw)
                df_feat = extract_epoch_features(raw_clean, window_len=2.0, step=2.0)
                os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
                df_feat.to_csv(cache_csv, index=False)
            
            # Align events
            annotations = raw.annotations
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
                    spawned_events[key] = {'start_time': onset, 'normal_or_conflict': params.get('normal_or_conflict')}
                elif event_type == 'box:touched':
                    touched_events[key] = onset

            paired_trials = []
            for key, spawned in spawned_events.items():
                if key in touched_events:
                    st = spawned['start_time']
                    et = touched_events[key]
                    nc = spawned['normal_or_conflict']
                    if nc in ('normal', 'conflict'):
                        paired_trials.append({'start_time': st, 'end_time': et, 'label': 1 if nc == 'normal' else 0})

            # Map timestamps to trial labels
            labels = []
            for _, row in df_feat.iterrows():
                t_mid = row['timestamp'] + 1.0
                lbl = -1
                for tr in paired_trials:
                    if tr['start_time'] <= t_mid <= tr['end_time']:
                        lbl = tr['label']
                        break
                labels.append(lbl)

            df_feat['label'] = labels
            df_clean = df_feat[df_feat['label'] != -1].copy()
            if len(df_clean) > 0:
                all_dfs.append(df_clean)
        except Exception as e:
            print(f"    [!] Error extracting subject {sub_id} ses-{ses}: {e}")

    if not all_dfs:
        return None

    df_sub = pd.concat(all_dfs, ignore_index=True)
    if len(df_sub) < 10:
        return None

    y = df_sub['label'].values
    X = df_sub.drop(columns=['timestamp', 'label'])
    feature_names = list(X.columns)

    # 80/20 Chronological Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    # 1. Full Feature Model (Random Forest & Extra Trees)
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_full.fit(X_train, y_train)
    preds_full = rf_full.predict(X_test)
    probs_full = rf_full.predict_proba(X_test)[:, 1]
    acc_full = balanced_accuracy_score(y_test, preds_full)
    f1_full = f1_score(y_test, preds_full, zero_division=0)
    try:
        auc_full = roc_auc_score(y_test, probs_full)
    except Exception:
        auc_full = 0.5

    # 2. Fit Shapley Attribution Engine
    shap_engine = ShapleyAttributionEngine(rf_full, X_train, X_test, feature_names)
    shap_values = shap_engine.compute_shap_values()
    df_imp = shap_engine.compute_global_importance()
    top_k_features = shap_engine.select_top_biomarkers(top_k=8)

    # 3. Shapley-Selected Model (Trained exclusively on top 8 Shapley features)
    X_train_top = X_train[top_k_features]
    X_test_top = X_test[top_k_features]

    rf_shap = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_shap.fit(X_train_top, y_train)
    preds_shap = rf_shap.predict(X_test_top)
    probs_shap = rf_shap.predict_proba(X_test_top)[:, 1]
    acc_shap = balanced_accuracy_score(y_test, preds_shap)
    f1_shap = f1_score(y_test, preds_shap, zero_division=0)
    try:
        auc_shap = roc_auc_score(y_test, probs_shap)
    except Exception:
        auc_shap = 0.5

    # 4. Shapley Composite Biomarker Index (SCBI) Model
    scbi_train = shap_engine.compute_composite_biomarker_index(X_train, y_train, top_k=8).reshape(-1, 1)
    scbi_test = shap_engine.compute_composite_biomarker_index(X_test, y_test, top_k=8).reshape(-1, 1)

    clf_scbi = LogisticRegression()
    clf_scbi.fit(scbi_train, y_train)
    preds_scbi = clf_scbi.predict(scbi_test)
    probs_scbi = clf_scbi.predict_proba(scbi_test)[:, 1]
    acc_scbi = balanced_accuracy_score(y_test, preds_scbi)
    f1_scbi = f1_score(y_test, preds_scbi, zero_division=0)
    try:
        auc_scbi = roc_auc_score(y_test, probs_scbi)
    except Exception:
        auc_scbi = 0.5

    return {
        'subject_id': sub_id,
        'n_samples': len(df_sub),
        'n_features_full': len(feature_names),
        'acc_full': acc_full,
        'f1_full': f1_full,
        'auc_full': auc_full,
        'acc_shap_topk': acc_shap,
        'f1_shap_topk': f1_shap,
        'auc_shap_topk': auc_shap,
        'acc_scbi': acc_scbi,
        'f1_scbi': f1_scbi,
        'auc_scbi': auc_scbi,
        'gain_shap_over_full': acc_shap - acc_full,
        'top_biomarkers': ", ".join(top_k_features[:5]),
        'df_imp': df_imp,
        'shap_engine': shap_engine
    }

def evaluate_shapley_pipeline(bids_root: str = "data/ds003846", output_dir: str = "data/ds003846/derivatives"):
    """
    Runs complete Shapley attribution and biomarker benchmark evaluation across participants.
    """
    os.makedirs(output_dir, exist_ok=True)
    subjects = ["02", "03", "04", "05", "06", "07", "08"]

    print("="*75)
    print(f"[*] Starting Shapley Value Feature Attribution & Multi-Biomarker Optimization Pipeline")
    print(f"[*] Benchmark Cohort: {subjects}")
    print("="*75)

    results = []
    global_importances = []
    last_engine = None

    for sub in subjects:
        print(f"\n[*] Processing Subject: sub-{sub}...")
        res = evaluate_subject_shapley(sub, bids_root)
        if res is not None:
            results.append(res)
            global_importances.append(res['df_imp'])
            last_engine = res['shap_engine']
            print(f"    [-] Full Feature Acc ({res['n_features_full']} features): {res['acc_full']:.2%}")
            print(f"    [*] Shapley Top-8 Selected Acc:         {res['acc_shap_topk']:.2%} [Gain: {res['gain_shap_over_full']:+.2%}]")
            print(f"    [*] Shapley Composite Biomarker Acc:     {res['acc_scbi']:.2%}")
            print(f"    [+] Top Attributed Biomarkers:           {res['top_biomarkers']}")

    if not results:
        print("[!] No results obtained.")
        return

    df_res = pd.DataFrame([{
        'subject_id': r['subject_id'],
        'n_samples': r['n_samples'],
        'acc_full': r['acc_full'],
        'f1_full': r['f1_full'],
        'auc_full': r['auc_full'],
        'acc_shap_topk': r['acc_shap_topk'],
        'f1_shap_topk': r['f1_shap_topk'],
        'auc_shap_topk': r['auc_shap_topk'],
        'acc_scbi': r['acc_scbi'],
        'f1_scbi': r['f1_scbi'],
        'auc_scbi': r['auc_scbi'],
        'gain_shap_over_full': r['gain_shap_over_full'],
        'top_biomarkers': r['top_biomarkers']
    } for r in results])

    # Save CSV
    csv_path = os.path.join(output_dir, "shapley_evaluation_results.csv")
    df_res.to_csv(csv_path, index=False)

    # Average global feature importance across cohort
    df_concat_imp = pd.concat(global_importances, ignore_index=True)
    df_mean_imp = df_concat_imp.groupby('biomarker')['mean_abs_shap'].mean().reset_index()
    df_mean_imp = df_mean_imp.sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
    df_mean_imp['importance_percentage'] = (df_mean_imp['mean_abs_shap'] / df_mean_imp['mean_abs_shap'].sum()) * 100.0

    # Save summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top10_imp = df_mean_imp.head(10)
    bars = ax.barh(top10_imp['biomarker'][::-1], top10_imp['mean_abs_shap'][::-1], color='#1f77b4', alpha=0.85)
    ax.set_xlabel(r'Mean Absolute Shapley Value |$\Phi$| across Benchmark Cohort', fontsize=12, fontweight='bold')
    ax.set_title('Cohort-Wide Biomarker Shapley Attribution Ranking', fontsize=14, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    for i, (val, pct) in enumerate(zip(top10_imp['mean_abs_shap'][::-1], top10_imp['importance_percentage'][::-1])):
        ax.text(val + 0.0005, i, f'{pct:.1f}%', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "shapley_biomarker_importance_cohort.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    if last_engine is not None:
        last_engine.plot_shap_summary(os.path.join(output_dir, "shapley_attributions_summary.png"))

    # Save Markdown report
    md_path = os.path.join(output_dir, "shapley_evaluation_report.md")
    with open(md_path, 'w') as f:
        f.write("# Shapley Feature Attribution & Multi-Biomarker Optimization Report\n\n")
        f.write(f"**Cohort Mean Full Model Accuracy:** {df_res['acc_full'].mean():.2%}\n")
        f.write(f"**Cohort Mean Shapley-Selected Model Accuracy (Top-8 Features):** {df_res['acc_shap_topk'].mean():.2%}\n")
        f.write(f"**Cohort Mean Shapley Composite Index (SCBI) Accuracy:** {df_res['acc_scbi'].mean():.2%}\n")
        f.write(f"**Average Net Accuracy Gain via Shapley Feature Selection:** {df_res['gain_shap_over_full'].mean():+.2%} percentage points\n\n")

        f.write("## Top 10 Cohort-Wide Biomarkers Ranked by Shapley Value:\n")
        for idx, row in top10_imp.iterrows():
            f.write(f"{idx+1}. **{row['biomarker']}**: Mean |" + r"$\Phi$" + f"| = {row['mean_abs_shap']:.4f} ({row['importance_percentage']:.1f}% total contribution)\n")

        f.write("\n## Subject Breakdown Table:\n\n")
        cols = ['subject_id', 'n_samples', 'acc_full', 'acc_shap_topk', 'acc_scbi', 'gain_shap_over_full', 'top_biomarkers']
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in df_res.iterrows():
            f.write("| " + " | ".join(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols) + " |\n")

    print("\n" + "="*75)
    print("=== SUMMARY OF SHAPLEY ATTRIBUTION & BIOMARKER OPTIMIZATION ===")
    print(f"Cohort Mean Full Model Accuracy ({df_res['n_samples'].sum()} total windows): {df_res['acc_full'].mean():.2%}")
    print(f"Cohort Mean Shapley-Selected Model Accuracy (Top-8 Biomarkers):         {df_res['acc_shap_topk'].mean():.2%}")
    print(f"Cohort Mean Shapley Composite Biomarker Index (SCBI) Accuracy:         {df_res['acc_scbi'].mean():.2%}")
    print(f"Average Accuracy Gain using Shapley Biomarker Selection:               {df_res['gain_shap_over_full'].mean():+.2%} percentage points")
    print("="*75)
    print(f"[+] Exported CSV to: {csv_path}")
    print(f"[+] Exported Report to: {md_path}")
    print(f"[+] Exported Figure to: {plot_path}")

if __name__ == "__main__":
    evaluate_shapley_pipeline()
