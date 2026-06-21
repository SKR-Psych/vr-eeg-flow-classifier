import argparse
import os
import mne
import numpy as np
import pandas as pd
import joblib
from src.loader import load_bids_data

def parse_desc(desc: str):
    """
    Parses key-value parameters from BIDS task annotation strings.
    E.g. "box:spawned;condition:vibro;trial_nr:10;normal_or_conflict:normal"
    """
    parts = desc.split(';')
    event_type = parts[0]
    params = {}
    for part in parts:
        if ':' in part:
            subparts = part.split(':', 1)
            if len(subparts) == 2:
                params[subparts[0]] = subparts[1]
    return event_type, params

def load_labels_and_features(features_csv: str, subject: str, session: str, bids_root: str, window_len: float = 2.0) -> tuple:
    """
    Loads features and parses annotations to assign labels.
    """
    # 1. Load continuous raw data annotations
    raw = load_bids_data(subject, session, bids_root)
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
            
            # Skip invalid labels
            if normal_or_conflict not in ('normal', 'conflict'):
                continue
                
            label = 1 if normal_or_conflict == 'normal' else 0
            paired_trials.append({
                'start_time': start_time,
                'end_time': end_time,
                'label': label,
                'trial_nr': key[0],
                'condition': key[1]
            })

    print(f"[*] Parsed and paired {len(paired_trials)} valid experimental trials.")

    # 3. Load feature matrix
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Features file '{features_csv}' does not exist.")
        
    df_features = pd.read_csv(features_csv)
    
    # 4. Map window timestamps (midpoint) to trials
    labels = []
    midpoint_offset = window_len / 2.0
    
    for idx, row in df_features.iterrows():
        t_mid = row['timestamp'] + midpoint_offset
        assigned_label = -1  # Default to ignored (e.g. ISI)
        
        for trial in paired_trials:
            if trial['start_time'] <= t_mid <= trial['end_time']:
                assigned_label = trial['label']
                break
        labels.append(assigned_label)
        
    df_features['label'] = labels
    
    # Filter out rows outside trials to preserve data purity
    df_clean = df_features[df_features['label'] != -1].copy()
    print(f"[*] Aligned {df_features.shape[0]} feature windows.")
    print(f"[*] Ignored {df_features.shape[0] - df_clean.shape[0]} windows falling outside active trials (ISI).")
    print(f"[*] Maintained {df_clean.shape[0]} pure windows for training.")
    
    y = df_clean['label'].values
    X = df_clean.drop(columns=['timestamp', 'label'])
    
    return X, y

def train_classifier(X: pd.DataFrame, y: np.ndarray) -> tuple:
    """
    Trains and compares Random Forest and Support Vector Machine (SVM) models.
    Uses chronological split (first 80% train, last 20% test) to prevent temporal data leakage.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # 1. Chronological split (prevent autocorrelation data leakage)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"[*] Chronological split completed:")
    print(f"    Train size: {X_train.shape[0]} windows (label balance: {np.bincount(y_train)})")
    print(f"    Test size:  {X_test.shape[0]} windows (label balance: {np.bincount(y_test)})")

    # 2. Train Random Forest
    print("[*] Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=97)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    # 3. Train Support Vector Machine (SVM) with StandardScaler
    print("[*] Training SVM Classifier (with scaling pipeline)...")
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=97))
    ])
    svm_pipe.fit(X_train, y_train)
    y_pred_svm = svm_pipe.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)

    # 4. Report metrics
    print("\n" + "="*20 + " Classifier Evaluation " + "="*20)
    print(f"1. Random Forest Test Accuracy: {acc_rf:.4%}")
    print(classification_report(y_test, y_pred_rf, target_names=['Disrupted', 'Flow']))
    print("Random Forest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))

    print(f"\n2. SVM Test Accuracy:          {acc_svm:.4%}")
    print(classification_report(y_test, y_pred_svm, target_names=['Disrupted', 'Flow']))
    print("SVM Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm))
    print("="*63)

    # Choose best model
    if acc_rf >= acc_svm:
        print("\n[*] Random Forest performed best. Selecting Random Forest.")
        return rf, {'type': 'Random Forest', 'accuracy': acc_rf}
    else:
        print("\n[*] SVM Pipeline performed best. Selecting SVM Pipeline.")
        return svm_pipe, {'type': 'SVM Pipeline', 'accuracy': acc_svm}

def save_model(model, filepath: str):
    """
    Persists the trained model using joblib.
    """
    joblib.dump(model, filepath)
    print(f"[*] Model successfully saved to: {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align features and train EEG flow classifiers.")
    parser.add_argument("--features", type=str, required=True, help="Path to extracted features CSV file")
    parser.add_argument("--subject", type=str, default="02", help="Subject ID (e.g. 02)")
    parser.add_argument("--session", type=str, default="EMS", help="Session ID (e.g. EMS, Vibro, Visual)")
    parser.add_argument("--root", type=str, default="data/ds003846", help="BIDS root directory")
    args = parser.parse_args()

    try:
        # Load and align labels and features
        X, y = load_labels_and_features(args.features, args.subject, args.session, args.root)
        
        # Train classifiers
        best_model, meta = train_classifier(X, y)
        
        # Export model
        out_model_path = os.path.join(args.root, "derivatives", f"sub-{args.subject}_ses-{args.session}_classifier.joblib")
        save_model(best_model, out_model_path)
        
        print("\n=== Pipeline Verification Successful ===")
        print(f"Model selected: {meta['type']}")
        print(f"Test Accuracy:  {meta['accuracy']:.2%}")
        
    except Exception as e:
        print(f"[!] Pipeline training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
