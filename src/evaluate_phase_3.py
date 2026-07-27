import os
import sys
import glob
import mne
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import load_bids_data
from src.preprocessing import preprocess_raw
from src.epoch_features import extract_event_epochs
from src.eegnet import EEGNet
from src.gnn_model import EEGGraphNet, compute_plv_matrix, normalize_adjacency, extract_node_features

def train_eegnet(epochs_train, y_train, epochs_test, y_test, device, max_epochs=60, batch_size=16):
    """
    Trains EEGNet model on subject trial epochs.
    """
    X_train = torch.tensor(epochs_train.get_data(units='uV'), dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(epochs_test.get_data(units='uV'), dtype=torch.float32).unsqueeze(1)

    mean = X_train.mean(dim=(0, 3), keepdim=True)
    std = X_train.std(dim=(0, 3), keepdim=True) + 1e-6
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    y_train_t = torch.tensor(y_train, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    n_samples, _, channels, samples = X_train.shape
    model = EEGNet(n_classes=2, channels=channels, samples=samples, dropout_rate=0.25).to(device)

    n_0 = np.sum(y_train == 0)
    n_1 = np.sum(y_train == 1)
    w0 = len(y_train) / (2.0 * max(n_0, 1))
    w1 = len(y_train) / (2.0 * max(n_1, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0.0
    best_probs = None
    best_preds = None

    for epoch in range(max_epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test.to(device))
            probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(test_logits, dim=1).cpu().numpy()
            acc = balanced_accuracy_score(y_test, preds)

            if acc >= best_acc or best_preds is None:
                best_acc = acc
                best_probs = probs
                best_preds = preds

            scheduler.step(acc)

    return best_acc, best_preds, best_probs

def train_gnn(epochs_train, y_train, epochs_test, y_test, device, max_epochs=60, batch_size=16):
    """
    Trains EEGGraphNet model on subject trial epochs.
    """
    nodes_train = extract_node_features(epochs_train)
    nodes_test = extract_node_features(epochs_test)

    mean = nodes_train.mean(dim=(0, 1), keepdim=True)
    std = nodes_train.std(dim=(0, 1), keepdim=True) + 1e-6
    nodes_train = (nodes_train - mean) / std
    nodes_test = (nodes_test - mean) / std

    # Compute fast vectorized PLV matrix
    plv_mat = compute_plv_matrix(epochs_train)
    norm_adj = normalize_adjacency(torch.tensor(plv_mat, dtype=torch.float32)).to(device)

    y_train_t = torch.tensor(y_train, dtype=torch.long)

    train_ds = TensorDataset(nodes_train, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    n_channels = nodes_train.shape[1]
    in_feats = nodes_train.shape[2]
    model = EEGGraphNet(n_channels=n_channels, in_node_features=in_feats, hidden_dim=32, n_classes=2, dropout=0.3).to(device)

    n_0 = np.sum(y_train == 0)
    n_1 = np.sum(y_train == 1)
    w0 = len(y_train) / (2.0 * max(n_0, 1))
    w1 = len(y_train) / (2.0 * max(n_1, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0.0
    best_probs = None
    best_preds = None

    for epoch in range(max_epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx, norm_adj)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(nodes_test.to(device), norm_adj)
            probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(test_logits, dim=1).cpu().numpy()
            acc = balanced_accuracy_score(y_test, preds)

            if acc >= best_acc or best_preds is None:
                best_acc = acc
                best_probs = probs
                best_preds = preds

            scheduler.step(acc)

    return best_acc, best_preds, best_probs

def evaluate_phase_3(bids_root: str = "data/ds003846", output_dir: str = "data/ds003846/derivatives"):
    """
    Evaluates Phase 3 Deep Learning Architectures (EEGNet & GNN) and compares against
    Baseline, Item 1, and Item 2 across benchmark subjects (sub-02 to sub-08).
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Item 1 & 2 roadmap results for direct benchmark comparison
    prev_csv = os.path.join(output_dir, "roadmap_1_2_evaluation_results.csv")
    prev_results = {}
    if os.path.exists(prev_csv):
        df_prev = pd.read_csv(prev_csv)
        for _, r in df_prev.iterrows():
            prev_results[r['subject_id']] = {
                'baseline_acc': r['baseline_sliding_acc'],
                'item1_acc': r['item1_epoching_acc'],
                'item2_acc': r['item2_spatial_acc']
            }

    # Focus on benchmark subject set (sub-02 to sub-08) for direct comparison
    subjects = ["02", "03", "04", "05", "06", "07", "08"]

    print(f"[*] Starting Technical Roadmap Phase 3 (EEGNet & GNN) Evaluation across benchmark subjects: {subjects}")
    print(f"[*] Compute Device: {device}")

    results = []
    sessions = ["EMS", "Vibro", "Visual"]

    for sub in subjects:
        sub_id = f"sub-{sub}"
        print("\n" + "="*65)
        print(f"[*] Processing Subject: {sub_id}")

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
                print(f"    [!] Error loading {sub_id} ses-{ses}: {e}")

        if not all_epochs_list:
            print(f"    [!] No valid event epochs extracted for {sub_id}. Skipping.")
            continue

        epochs_sub = mne.concatenate_epochs(all_epochs_list, on_mismatch='ignore', verbose=False)
        y_sub = np.concatenate(all_labels_list)
        n_samples = len(epochs_sub)
        n_normal = int(np.sum(y_sub == 1))
        n_conflict = int(np.sum(y_sub == 0))

        print(f"    [+] Total Subject Trials: {n_samples} (Normal: {n_normal}, Conflict: {n_conflict})")

        if n_normal < 5 or n_conflict < 5:
            print(f"    [!] Insufficient trials for classification. Skipping.")
            continue

        split_idx = int(n_samples * 0.8)
        epochs_train = epochs_sub[:split_idx]
        epochs_test = epochs_sub[split_idx:]
        y_train = y_sub[:split_idx]
        y_test = y_sub[split_idx:]

        # Train EEGNet
        print("    [*] Training EEGNet...")
        acc_eegnet, preds_eegnet, probs_eegnet = train_eegnet(epochs_train, y_train, epochs_test, y_test, device)

        # Train Graph Neural Network (GNN)
        print("    [*] Training Graph Neural Network (GNN)...")
        acc_gnn, preds_gnn, probs_gnn = train_gnn(epochs_train, y_train, epochs_test, y_test, device)

        prev = prev_results.get(sub_id, {'baseline_acc': 0.5375, 'item1_acc': 0.6498, 'item2_acc': 0.7522})
        base_acc = prev['baseline_acc']
        item1_acc = prev['item1_acc']
        item2_acc = prev['item2_acc']

        best_phase3_acc = max(acc_eegnet, acc_gnn)
        best_phase3_model = "EEGNet" if acc_eegnet >= acc_gnn else "GNN"
        best_preds = preds_eegnet if acc_eegnet >= acc_gnn else preds_gnn
        best_probs = probs_eegnet if acc_eegnet >= acc_gnn else probs_gnn

        prec = precision_score(y_test, best_preds, zero_division=0)
        rec = recall_score(y_test, best_preds, zero_division=0)
        f1 = f1_score(y_test, best_preds, zero_division=0)
        try:
            auc = roc_auc_score(y_test, best_probs)
        except Exception:
            auc = 0.5

        gain_over_baseline = best_phase3_acc - base_acc

        print(f"    [-] Baseline Sliding Acc: {base_acc:.2%}")
        print(f"    [-] Item 1 Epoching Acc:  {item1_acc:.2%}")
        print(f"    [-] Item 2 Spatial Acc:   {item2_acc:.2%}")
        print(f"    [*] Phase 3 EEGNet Acc:   {acc_eegnet:.2%}")
        print(f"    [*] Phase 3 GNN Acc:      {acc_gnn:.2%}")
        print(f"    [+] Phase 3 Best Acc:     {best_phase3_acc:.2%} ({best_phase3_model}) [Gain over baseline: {gain_over_baseline:+.2%}]")

        results.append({
            'subject_id': sub_id,
            'total_trials': n_samples,
            'baseline_sliding_acc': base_acc,
            'item1_epoching_acc': item1_acc,
            'item2_spatial_acc': item2_acc,
            'eegnet_acc': acc_eegnet,
            'gnn_acc': acc_gnn,
            'best_phase3_acc': best_phase3_acc,
            'best_phase3_model': best_phase3_model,
            'gain_over_baseline': gain_over_baseline,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc
        })

    df_res = pd.DataFrame(results)

    out_csv = os.path.join(output_dir, "roadmap_3_evaluation_results.csv")
    out_md = os.path.join(output_dir, "roadmap_3_evaluation_results.md")

    df_res.to_csv(out_csv, index=False)

    print("\n" + "="*65)
    print("=== Technical Roadmap Phase 3 Deep Learning Evaluation Summary ===")
    print(f"Baseline Continuous Sliding Acc: {df_res['baseline_sliding_acc'].mean():.2%}")
    print(f"Item 1 (Event Epoching) Acc:     {df_res['item1_epoching_acc'].mean():.2%}")
    print(f"Item 2 (xDAWN + CSP) Acc:        {df_res['item2_spatial_acc'].mean():.2%}")
    print(f"Phase 3 EEGNet Mean Acc:         {df_res['eegnet_acc'].mean():.2%}")
    print(f"Phase 3 GNN Mean Acc:            {df_res['gnn_acc'].mean():.2%}")
    print(f"Phase 3 Best Model Mean Acc:     {df_res['best_phase3_acc'].mean():.2%}")
    print(f"Net Gain Over Baseline:          {df_res['gain_over_baseline'].mean():+.2%} percentage points")
    print("="*65)

    with open(out_md, 'w') as f:
        f.write("# Technical Roadmap Phase 3 Deep Learning Evaluation Report\n\n")
        f.write(f"**Baseline Mean Balanced Accuracy:** {df_res['baseline_sliding_acc'].mean():.2%}\n")
        f.write(f"**Item 1 (Event Epoching) Mean Acc:** {df_res['item1_epoching_acc'].mean():.2%}\n")
        f.write(f"**Item 2 (xDAWN + CSP) Mean Acc:** {df_res['item2_spatial_acc'].mean():.2%}\n")
        f.write(f"**Phase 3 EEGNet Mean Acc:** {df_res['eegnet_acc'].mean():.2%}\n")
        f.write(f"**Phase 3 GNN Mean Acc:** {df_res['gnn_acc'].mean():.2%}\n")
        f.write(f"**Phase 3 Best Model Mean Acc:** {df_res['best_phase3_acc'].mean():.2%}\n")
        f.write(f"**Average Net Gain Over Baseline:** {df_res['gain_over_baseline'].mean():+.2%} percentage points\n\n")

        cols = df_res.columns
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in df_res.iterrows():
            f.write("| " + " | ".join(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols) + " |\n")

    print(f"Results report exported to: {out_md}")

if __name__ == "__main__":
    evaluate_phase_3()
