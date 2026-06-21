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

def load_labels_and_features(features_csv: str, subject: str = None, session: str = None, bids_root: str = None, window_len: float = 2.0, is_global: bool = False) -> tuple:
    """
    Loads features and parses annotations to assign labels.
    """
    if is_global:
        if not os.path.exists(features_csv):
            raise FileNotFoundError(f"Master features file '{features_csv}' does not exist.")
        df = pd.read_csv(features_csv)
        print(f"[*] Loaded master features dataset from: {features_csv}")
        print(f"[*] Total master windows: {df.shape[0]}")
        
        y = df['label'].values
        groups = df['subject_id'].values
        X = df.drop(columns=['subject_id', 'session_id', 'timestamp', 'label'])
        return X, y, groups

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
    
    return X, y, None

def train_classifier(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray = None) -> tuple:
    """
    Trains and compares Random Forest and Support Vector Machine (SVM) models.
    Supports GroupKFold cross-validation if groups are provided.
    Otherwise uses chronological split (first 80% train, last 20% test).
    Applies class balancing and evaluates using Balanced Accuracy.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GroupKFold

    if groups is not None:
        # GroupKFold (Leave-One-Subject-Out simulation)
        # We group by subject_id
        gkf = GroupKFold(n_splits=5)
        
        rf_accs = []
        svm_accs = []
        
        print(f"[*] Starting 5-Fold GroupKFold Cross-Validation (Grouping by Subject)...")
        print(f"    Total samples: {X.shape[0]} windows, label balance: {np.bincount(y)}")
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=97)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            rf_accs.append(balanced_accuracy_score(y_test, y_pred_rf))
            
            # SVM Pipeline
            svm_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', class_weight='balanced', probability=False, random_state=97))
            ])
            svm_pipe.fit(X_train, y_train)
            y_pred_svm = svm_pipe.predict(X_test)
            svm_accs.append(balanced_accuracy_score(y_test, y_pred_svm))
            
        mean_rf_acc = np.mean(rf_accs)
        mean_svm_acc = np.mean(svm_accs)
        
        print("\n" + "="*20 + " GroupKFold CV Evaluation " + "="*20)
        print(f"1. Random Forest Mean Balanced Accuracy: {mean_rf_acc:.4%}")
        print(f"2. SVM Pipeline Mean Balanced Accuracy:  {mean_svm_acc:.4%}")
        print("="*66)
        
        # Train final models on 100% of data for global export
        print("\n[*] Training final models on 100% of data for global export...")
        rf_final = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=97)
        rf_final.fit(X, y)
        
        svm_final_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', class_weight='balanced', probability=False, random_state=97))
        ])
        svm_final_pipe.fit(X, y)
        
        if mean_rf_acc >= mean_svm_acc:
            print("[*] Selecting Random Forest as the final global model.")
            return rf_final, {'type': 'Random Forest', 'accuracy': mean_rf_acc}
        else:
            print("[*] Selecting SVM Pipeline as the final global model.")
            return svm_final_pipe, {'type': 'SVM Pipeline', 'accuracy': mean_svm_acc}
            
    else:
        # 1. Chronological split (prevent autocorrelation data leakage)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"[*] Chronological split completed:")
        print(f"    Train size: {X_train.shape[0]} windows (label balance: {np.bincount(y_train)})")
        print(f"    Test size:  {X_test.shape[0]} windows (label balance: {np.bincount(y_test)})")
    
        # 2. Train Random Forest (balanced class weights)
        print("[*] Training Random Forest Classifier (balanced weights)...")
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=97)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        acc_rf = balanced_accuracy_score(y_test, y_pred_rf)
    
        # 3. Train Support Vector Machine (SVM) with StandardScaler (balanced class weights)
        print("[*] Training SVM Classifier (with scaling pipeline and balanced weights)...")
        svm_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', class_weight='balanced', probability=False, random_state=97))
        ])
        svm_pipe.fit(X_train, y_train)
        y_pred_svm = svm_pipe.predict(X_test)
        acc_svm = balanced_accuracy_score(y_test, y_pred_svm)
    
        # 4. Report metrics
        print("\n" + "="*20 + " Classifier Evaluation " + "="*20)
        print(f"1. Random Forest Test Balanced Accuracy: {acc_rf:.4%}")
        print(classification_report(y_test, y_pred_rf, target_names=['Disrupted', 'Flow']))
        print("Random Forest Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_rf))
    
        print(f"\n2. SVM Test Balanced Accuracy:          {acc_svm:.4%}")
        print(classification_report(y_test, y_pred_svm, target_names=['Disrupted', 'Flow']))
        print("SVM Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_svm))
        print("="*63)
    
        # Choose best model based on Balanced Accuracy
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
    parser.add_argument("--global", dest="is_global", action="store_true", help="Train a generalized multi-subject classifier")
    parser.set_defaults(is_global=False)
    args = parser.parse_args()

    try:
        if args.is_global:
            # Load and align labels and features globally
            X, y, groups = load_labels_and_features(
                features_csv=args.features,
                is_global=True
            )
            
            # Train classifiers using GroupKFold
            best_model, meta = train_classifier(X, y, groups)
            
            # Export global model
            out_model_path = os.path.join(args.root, "derivatives", "global_flow_classifier.joblib")
            save_model(best_model, out_model_path)
            
            print("\n=== Global Pipeline Verification Successful ===")
            print(f"Model selected: {meta['type']}")
            print(f"CV Balanced Accuracy: {meta['accuracy']:.2%}")
        else:
            # Load and align labels and features for single subject
            X, y, _ = load_labels_and_features(
                features_csv=args.features, 
                subject=args.subject, 
                session=args.session, 
                bids_root=args.root,
                is_global=False
            )
            
            # Train classifiers
            best_model, meta = train_classifier(X, y)
            
            # Export model
            out_model_path = os.path.join(args.root, "derivatives", f"sub-{args.subject}_ses-{args.session}_classifier.joblib")
            save_model(best_model, out_model_path)
            
            print("\n=== Pipeline Verification Successful ===")
            print(f"Model selected: {meta['type']}")
            print(f"Test Balanced Accuracy:  {meta['accuracy']:.2%}")
            
    except Exception as e:
        print(f"[!] Pipeline training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
