import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def evaluate_all_subjects(features_csv: str, output_dir: str):
    """
    Evaluates individual subject-calibrated classifiers for all subjects in the dataset.
    Saves a summary CSV and exports the best trained models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the master features dataset
    print(f"[*] Loading master features from: {features_csv}")
    df = pd.read_csv(features_csv)
    subjects = sorted(df['subject_id'].unique())
    print(f"[*] Found {len(subjects)} subjects: {subjects}")
    
    results = []
    
    # 2. Iterate through each subject and evaluate
    for sub in subjects:
        print(f"\n" + "="*50)
        print(f"[*] Processing Subject: {sub}")
        
        # Filter data for this subject
        sub_df = df[df['subject_id'] == sub].copy()
        
        # Drop identifiers to get features and labels
        y = sub_df['label'].values
        X = sub_df.drop(columns=['subject_id', 'session_id', 'timestamp', 'label'])
        
        n_samples = len(sub_df)
        n_flow = np.sum(y == 1)
        n_disrupted = np.sum(y == 0)
        
        print(f"    Total windows: {n_samples} (Flow: {n_flow}, Disrupted: {n_disrupted})")
        
        if n_flow < 5 or n_disrupted < 5:
            print(f"    [!] Warning: Too few samples for classification. Skipping.")
            continue
            
        # Chronological train/test split (80/20) to prevent time-series autocorrelation leakage
        split_idx = int(n_samples * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # --- Random Forest ---
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=97)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_bal_acc = balanced_accuracy_score(y_test, y_pred_rf)
        
        # --- SVM Pipeline ---
        svm_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=97))
        ])
        svm_pipe.fit(X_train, y_train)
        y_pred_svm = svm_pipe.predict(X_test)
        svm_bal_acc = balanced_accuracy_score(y_test, y_pred_svm)
        
        # Determine best classifier
        best_model_name = "Random Forest" if rf_bal_acc >= svm_bal_acc else "SVM"
        best_bal_acc = max(rf_bal_acc, svm_bal_acc)
        y_pred_best = y_pred_rf if rf_bal_acc >= svm_bal_acc else y_pred_svm
        
        # Compute detailed metrics for the best model
        prec = precision_score(y_test, y_pred_best, zero_division=0)
        rec = recall_score(y_test, y_pred_best, zero_division=0)
        f1 = f1_score(y_test, y_pred_best, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred_best)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        print(f"    Random Forest Balanced Acc: {rf_bal_acc:.2%}")
        print(f"    SVM Balanced Acc:           {svm_bal_acc:.2%}")
        print(f"    Selected Model:             {best_model_name} ({best_bal_acc:.2%})")
        print(f"    Confusion Matrix:           TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Record results
        results.append({
            'subject_id': sub,
            'total_samples': n_samples,
            'flow_samples': n_flow,
            'disrupted_samples': n_disrupted,
            'rf_balanced_acc': rf_bal_acc,
            'svm_balanced_acc': svm_bal_acc,
            'best_model': best_model_name,
            'test_balanced_acc': best_bal_acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        })
        
        # Train final production model on 100% of subject data
        print(f"    [*] Training final production {best_model_name} on 100% of subject data...")
        if best_model_name == "Random Forest":
            final_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=97)
        else:
            final_model = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=97))
            ])
            
        final_model.fit(X, y)
        
        # Save model
        model_path = os.path.join(output_dir, f"{sub}_calibrated_classifier.joblib")
        joblib.dump(final_model, model_path)
        print(f"    [+] Saved final model to: {model_path}")
        
    # 3. Create results DataFrame and save
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, "subject_evaluation_results.csv")
    results_df.to_csv(results_csv, index=False)
    
    print("\n" + "="*50)
    print("=== Subject-Calibrated Evaluations Complete ===")
    print(f"Results summary saved to: {results_csv}")
    print(f"Mean Balanced Accuracy across subjects: {results_df['test_balanced_acc'].mean():.2%}")
    print("="*50)
    
    # Save a markdown version for reports
    results_md = os.path.join(output_dir, "subject_evaluation_results.md")
    with open(results_md, 'w') as f:
        f.write("# Subject-Calibrated Classifier Performance Summary\n\n")
        f.write(f"**Mean Balanced Accuracy across subjects:** {results_df['test_balanced_acc'].mean():.2%}\n\n")
        try:
            f.write(results_df.to_markdown(index=False))
        except ImportError:
            # Fallback to manual markdown table creation if tabulate is not installed
            cols = results_df.columns
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
            for _, row in results_df.iterrows():
                f.write("| " + " | ".join(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols) + " |\n")
    print(f"Markdown report saved to: {results_md}")

if __name__ == "__main__":
    evaluate_all_subjects(
        features_csv="data/ds003846/derivatives/master_features.csv",
        output_dir="data/ds003846/derivatives"
    )
