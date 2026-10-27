import os
import sys
import time
import glob
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import load_bids_data
from src.preprocessing import preprocess_raw
from src.epoch_features import extract_event_epochs
from src.eegnet import EEGNet, Conv2dWithConstraint, LinearWithConstraint

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) channel-attention block (Hu et al., 2018).
    Recalibrates channel-wise feature responses by modeling interdependencies.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, bottleneck, bias=False),
            nn.ELU(),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights

class MS_SE_EEGNet(nn.Module):
    """
    Multi-Scale Squeeze-and-Excitation EEGNet (Altuwaijri et al., 2022; Riyad et al., 2020).
    Replaces Block 1 single temporal kernel with 3 parallel temporal kernels (Theta, Alpha, Beta/Gamma)
    and applies Squeeze-and-Excitation attention to suppress noise.
    """
    def __init__(
        self,
        n_classes: int = 2,
        channels: int = 64,
        samples: int = 350,
        dropout_rate: float = 0.25,
        temporal_kernels: tuple = (250, 64, 16),
        filters_per_branch: int = 4,
        D: int = 2,
        F2: int = 16
    ):
        super().__init__()
        self.n_classes = n_classes
        self.channels = channels
        self.samples = samples
        self.temporal_kernels = temporal_kernels
        self.filters_per_branch = filters_per_branch
        self.D = D
        self.F2 = F2

        total_F1 = filters_per_branch * len(temporal_kernels)  # 4 * 3 = 12

        # 1. Parallel Multi-Scale Temporal Convolutions
        self.temporal_branches = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=filters_per_branch,
                kernel_size=(1, k),
                padding='same',
                bias=False
            ) for k in temporal_kernels
        ])
        self.bn_temporal = nn.BatchNorm2d(total_F1)

        # 2. Depthwise Spatial Convolution (Max-Norm <= 1.0)
        self.depthwise_conv = Conv2dWithConstraint(
            in_channels=total_F1,
            out_channels=total_F1 * D,
            kernel_size=(channels, 1),
            groups=total_F1,
            bias=False,
            max_norm=1.0
        )
        self.bn_spatial = nn.BatchNorm2d(total_F1 * D)

        # 3. Squeeze-and-Excitation Channel Attention
        self.se = SqueezeExcitation(channels=total_F1 * D, reduction=4)
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # 4. Separable Convolution (Depthwise + Pointwise)
        self.separable_depthwise = nn.Conv2d(
            in_channels=total_F1 * D,
            out_channels=total_F1 * D,
            kernel_size=(1, 16),
            padding='same',
            groups=total_F1 * D,
            bias=False
        )
        self.separable_pointwise = nn.Conv2d(
            in_channels=total_F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn_separable = nn.BatchNorm2d(F2)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Classifier Head
        self.in_features = self._get_flatten_size()
        self.classifier = LinearWithConstraint(
            in_features=self.in_features,
            out_features=n_classes,
            max_norm=0.25
        )

    def _get_flatten_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.channels, self.samples)
            branch_outs = [branch(dummy) for branch in self.temporal_branches]
            x = torch.cat(branch_outs, dim=1)
            x = self.bn_temporal(x)
            x = self.depthwise_conv(x)
            x = self.bn_spatial(x)
            x = F.elu(x)
            x = self.se(x)
            x = self.pooling1(x)
            x = self.separable_depthwise(x)
            x = self.separable_pointwise(x)
            x = self.bn_separable(x)
            x = F.elu(x)
            x = self.pooling2(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        branch_outs = [branch(x) for branch in self.temporal_branches]
        x = torch.cat(branch_outs, dim=1)
        x = self.bn_temporal(x)

        x = self.depthwise_conv(x)
        x = self.bn_spatial(x)
        x = F.elu(x)
        x = self.se(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.bn_separable(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        x = x.flatten(start_dim=1)
        return self.classifier(x)

def measure_latency(model: nn.Module, channels: int, samples: int, n_runs: int = 100) -> float:
    """Measures single-epoch forward pass inference latency on CPU (ms)."""
    model.eval()
    dummy = torch.randn(1, 1, channels, samples)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
    return ((time.perf_counter() - t0) / n_runs) * 1000.0

def train_and_eval(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, 
                   X_val: torch.Tensor, y_val: np.ndarray, max_epochs: int = 40, batch_size: int = 32):
    device = torch.device('cpu')
    model = model.to(device)

    n_0 = (y_train == 0).sum().item()
    n_1 = (y_train == 1).sum().item()
    w0 = len(y_train) / (2.0 * max(n_0, 1))
    w1 = len(y_train) / (2.0 * max(n_1, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_acc = 0.0
    best_state = None
    best_preds = None
    best_probs = None

    for epoch in range(max_epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_val)
            probs = torch.softmax(test_logits, dim=1)[:, 1].numpy()
            preds = torch.argmax(test_logits, dim=1).numpy()
            val_acc = balanced_accuracy_score(y_val, preds)

            if val_acc >= best_val_acc or best_state is None:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_preds = preds
                best_probs = probs
            scheduler.step(val_acc)

    model.load_state_dict(best_state)
    precision = precision_score(y_val, best_preds, zero_division=0)
    recall = recall_score(y_val, best_preds, zero_division=0)
    f1 = f1_score(y_val, best_preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_val, best_probs)
    except Exception:
        roc_auc = 0.5

    return {
        'balanced_acc': best_val_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

def get_or_load_epochs(sub_id: str, bids_root: str, cache_dir: str):
    sub_num = sub_id.replace("sub-", "")
    cache_path = os.path.join(cache_dir, f"{sub_id}_benchmark_epochs_cache.pt")
    
    if os.path.exists(cache_path):
        print(f"[*] Loading cached preprocessed epochs for {sub_id}...")
        cache_data = torch.load(cache_path, weights_only=False)
        return cache_data['X_raw'], cache_data['labels']

    sub_path = os.path.join(bids_root, sub_id)
    ses_dirs = glob.glob(os.path.join(sub_path, "ses-*"))
    ses_list = [os.path.basename(d).replace("ses-", "") for d in ses_dirs if os.path.isdir(d)]
    ses_list.sort()

    all_epochs = []
    all_labels = []

    print(f"[*] Preprocessing {sub_id} sessions: {ses_list}...")
    for ses in ses_list:
        try:
            raw = load_bids_data(sub_num, ses, bids_root)
            raw_clean = preprocess_raw(raw)
            epochs, labels, _ = extract_event_epochs(raw_clean, event_target='box:touched', tmin=-0.1, tmax=0.6)
            if epochs is not None and len(epochs) > 0:
                all_epochs.append(epochs)
                all_labels.append(epochs.events[:, 2])
        except Exception as e:
            print(f"    [!] Error in {sub_id} ses-{ses}: {e}")

    if not all_epochs:
        return None, None

    concat_epochs = mne.concatenate_epochs(all_epochs, on_mismatch='ignore', verbose=False)
    concat_labels = np.concatenate(all_labels)
    X_raw = torch.tensor(concat_epochs.get_data(units='uV'), dtype=torch.float32).unsqueeze(1)

    os.makedirs(cache_dir, exist_ok=True)
    torch.save({'X_raw': X_raw, 'labels': concat_labels}, cache_path)
    print(f"[+] Cached preprocessed epochs to {cache_path}")
    return X_raw, concat_labels

def run_cohort_benchmark():
    bids_root = "data/ds003846"
    derivatives_dir = "data/ds003846/derivatives"
    out_csv = os.path.join(derivatives_dir, "cohort_ms_se_vs_eegnet_comparison.csv")

    sub_dirs = glob.glob(os.path.join(bids_root, "sub-*"))
    subjects = [os.path.basename(d) for d in sub_dirs if os.path.isdir(d)]
    subjects.sort()

    print("=" * 75)
    print(f"COHORT BENCHMARK: MS-SE-EEGNet vs. Vanilla EEGNet")
    print(f"Total Subjects: {len(subjects)} ({subjects})")
    print(f"Results Output: {out_csv}")
    print("=" * 75)

    results = []

    for sub_id in subjects:
        t0 = time.time()
        print(f"\n{'='*30} Processing {sub_id} {'='*30}")
        try:
            X_raw, labels = get_or_load_epochs(sub_id, bids_root, derivatives_dir)
            if X_raw is None or len(labels) < 40:
                print(f"[!] {sub_id} has insufficient trials. Skipping.")
                continue

            n_total = len(labels)
            n_flow = int(np.sum(labels == 1))
            n_conflict = int(np.sum(labels == 0))

            if n_flow < 5 or n_conflict < 5:
                print(f"[!] {sub_id} has degenerate class distribution ({n_flow} vs {n_conflict}). Skipping.")
                continue

            # 80/20 Chronological Split
            split_idx = int(n_total * 0.8)
            X_train_raw = X_raw[:split_idx]
            mean = X_train_raw.mean(dim=(0, 3), keepdim=True)
            std = X_train_raw.std(dim=(0, 3), keepdim=True) + 1e-6

            X_train = (X_train_raw - mean) / std
            y_train = torch.tensor(labels[:split_idx], dtype=torch.long)

            X_val = (X_raw[split_idx:] - mean) / std
            y_val = labels[split_idx:]

            _, _, channels, samples = X_train.shape

            print(f"[*] Trials: {n_total} (Train={len(X_train)}, Val={len(X_val)}) | Channels={channels}, Samples={samples}")

            # Fixed seed 42 for reproducible cohort comparison
            torch.manual_seed(42)
            np.random.seed(42)
            vanilla_model = EEGNet(n_classes=2, channels=channels, samples=samples, dropout_rate=0.25)
            res_v = train_and_eval(vanilla_model, X_train, y_train, X_val, y_val, max_epochs=40)
            lat_v = measure_latency(vanilla_model, channels, samples)

            torch.manual_seed(42)
            np.random.seed(42)
            ms_se_model = MS_SE_EEGNet(n_classes=2, channels=channels, samples=samples, dropout_rate=0.25)
            res_m = train_and_eval(ms_se_model, X_train, y_train, X_val, y_val, max_epochs=40)
            lat_m = measure_latency(ms_se_model, channels, samples)

            elapsed = time.time() - t0
            record = {
                'subject_id': sub_id,
                'total_trials': n_total,
                'val_trials': len(X_val),
                'channels': channels,
                'eegnet_balanced_acc': round(res_v['balanced_acc'], 4),
                'ms_se_balanced_acc': round(res_m['balanced_acc'], 4),
                'acc_gain': round(res_m['balanced_acc'] - res_v['balanced_acc'], 4),
                'eegnet_recall': round(res_v['recall'], 4),
                'ms_se_recall': round(res_m['recall'], 4),
                'recall_gain': round(res_m['recall'] - res_v['recall'], 4),
                'eegnet_f1': round(res_v['f1_score'], 4),
                'ms_se_f1': round(res_m['f1_score'], 4),
                'f1_gain': round(res_m['f1_score'] - res_v['f1_score'], 4),
                'eegnet_precision': round(res_v['precision'], 4),
                'ms_se_precision': round(res_m['precision'], 4),
                'eegnet_roc_auc': round(res_v['roc_auc'], 4),
                'ms_se_roc_auc': round(res_m['roc_auc'], 4),
                'eegnet_latency_ms': round(lat_v, 3),
                'ms_se_latency_ms': round(lat_m, 3),
                'processing_time_s': round(elapsed, 1)
            }
            results.append(record)

            print(f"[+] {sub_id} Results:")
            print(f"    Balanced Acc: EEGNet={record['eegnet_balanced_acc']:.2%} | MS-SE={record['ms_se_balanced_acc']:.2%} (Gain={record['acc_gain']:+.2%})")
            print(f"    Flow Recall:  EEGNet={record['eegnet_recall']:.2%} | MS-SE={record['ms_se_recall']:.2%} (Gain={record['recall_gain']:+.2%})")
            print(f"    F1-Score:     EEGNet={record['eegnet_f1']:.3f} | MS-SE={record['ms_se_f1']:.3f} (Gain={record['f1_gain']:+.3f})")
            print(f"    Time Elapsed: {elapsed:.1f}s")

            # Save incremental progress to CSV
            pd.DataFrame(results).to_csv(out_csv, index=False)

        except Exception as e:
            print(f"[!] Error processing {sub_id}: {e}")

    df_cohort = pd.DataFrame(results)
    df_cohort.to_csv(out_csv, index=False)

    print("\n" + "=" * 75)
    print("COHORT BENCHMARK COMPLETE")
    print(f"Evaluated Subjects: {len(df_cohort)}")
    print(f"Mean Balanced Accuracy: EEGNet={df_cohort['eegnet_balanced_acc'].mean():.2%} | MS-SE-EEGNet={df_cohort['ms_se_balanced_acc'].mean():.2%}")
    print(f"Mean Flow Recall:       EEGNet={df_cohort['eegnet_recall'].mean():.2%} | MS-SE-EEGNet={df_cohort['ms_se_recall'].mean():.2%}")
    print(f"Mean F1-Score:          EEGNet={df_cohort['eegnet_f1'].mean():.3f} | MS-SE-EEGNet={df_cohort['ms_se_f1'].mean():.3f}")
    print(f"Mean ROC-AUC:           EEGNet={df_cohort['eegnet_roc_auc'].mean():.3f} | MS-SE-EEGNet={df_cohort['ms_se_roc_auc'].mean():.3f}")
    print(f"Full results saved to: {out_csv}")
    print("=" * 75)

if __name__ == "__main__":
    run_cohort_benchmark()
