import argparse
import os
import sys
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import load_bids_data
from src.preprocessing import preprocess_raw
from src.epoch_features import extract_event_epochs
from src.eegnet import EEGNet

def parse_desc(desc: str):
    """
    Parses key-value parameters from BIDS task annotation strings.
    E.g. 'box:spawned;condition:vibro;trial_nr:10;normal_or_conflict:normal'
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

def train_eegnet_classifier(
    epochs_train: mne.Epochs,
    y_train: np.ndarray,
    epochs_val: mne.Epochs = None,
    y_val: np.ndarray = None,
    device: torch.device = None,
    max_epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    dropout_rate: float = 0.25
) -> tuple:
    """
    Trains and calibrates an EEGNet deep neural network classifier on event-locked EEG epochs.

    Parameters
    ----------
    epochs_train : mne.Epochs
        Training trial epochs of shape (n_trials, n_channels, n_samples).
    y_train : np.ndarray
        1D binary array of training ground truth labels (1 = Flow/Normal, 0 = Disrupted/Conflict).
    epochs_val : mne.Epochs, optional
        Validation trial epochs.
    y_val : np.ndarray, optional
        Validation labels.
    device : torch.device, optional
        Computation device (CUDA or CPU).
    max_epochs : int, optional
        Maximum training epochs (default: 60).
    batch_size : int, optional
        Batch size (default: 16).
    lr : float, optional
        Initial learning rate (default: 1e-3).
    dropout_rate : float, optional
        Dropout rate (default: 0.25).

    Returns
    -------
    model : EEGNet
        Calibrated PyTorch EEGNet model.
    norm_params : dict
        Channel-wise z-score normalization parameters ('mean', 'std').
    metrics : dict
        Validation balanced accuracy and performance metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert epochs to microvolts tensor: shape (n_trials, 1, n_channels, n_samples)
    X_train = torch.tensor(epochs_train.get_data(units='uV'), dtype=torch.float32).unsqueeze(1)
    
    # Compute channel-wise training normalization parameters (mean and std across trials & time)
    norm_mean = X_train.mean(dim=(0, 3), keepdim=True)
    norm_std = X_train.std(dim=(0, 3), keepdim=True) + 1e-6

    X_train = (X_train - norm_mean) / norm_std
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    n_samples, _, channels, samples = X_train.shape
    model = EEGNet(n_classes=2, channels=channels, samples=samples, dropout_rate=dropout_rate).to(device)

    # Balanced Class Weighting to combat class imbalance
    n_0 = np.sum(y_train == 0)
    n_1 = np.sum(y_train == 1)
    w0 = len(y_train) / (2.0 * max(n_0, 1))
    w1 = len(y_train) / (2.0 * max(n_1, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    has_val = epochs_val is not None and y_val is not None and len(y_val) > 0
    if has_val:
        X_val = torch.tensor(epochs_val.get_data(units='uV'), dtype=torch.float32).unsqueeze(1)
        X_val = (X_val - norm_mean) / norm_std

    best_val_acc = 0.0
    best_state_dict = None
    best_preds = None
    best_probs = None

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if has_val:
            model.eval()
            with torch.no_grad():
                test_logits = model(X_val.to(device))
                probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(test_logits, dim=1).cpu().numpy()
                val_acc = balanced_accuracy_score(y_val, preds)

                if val_acc >= best_val_acc or best_state_dict is None:
                    best_val_acc = val_acc
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_preds = preds
                    best_probs = probs

                scheduler.step(val_acc)
        else:
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    metrics = {'balanced_accuracy': best_val_acc}
    if has_val and best_preds is not None:
        metrics['precision'] = precision_score(y_val, best_preds, zero_division=0)
        metrics['recall'] = recall_score(y_val, best_preds, zero_division=0)
        metrics['f1_score'] = f1_score(y_val, best_preds, zero_division=0)
        try:
            metrics['roc_auc'] = roc_auc_score(y_val, best_probs)
        except Exception:
            metrics['roc_auc'] = 0.5
        cm = confusion_matrix(y_val, best_preds)
        metrics['confusion_matrix'] = cm

    norm_params = {
        'mean': norm_mean.cpu(),
        'std': norm_std.cpu(),
        'channels': channels,
        'samples': samples
    }

    return model, norm_params, metrics

def save_eegnet_checkpoint(model: EEGNet, norm_params: dict, save_path: str, extra_meta: dict = None):
    """
    Saves a self-contained calibrated EEGNet model checkpoint (.pt).
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'channels': norm_params['channels'],
        'samples': norm_params['samples'],
        'norm_mean': norm_params['mean'],
        'norm_std': norm_params['std'],
        'architecture': 'EEGNet',
        'sfreq': 500.0,
        'metadata': extra_meta or {}
    }
    torch.save(checkpoint, save_path)
    print(f"[+] Successfully exported calibrated EEGNet checkpoint to: {save_path}")

def load_eegnet_checkpoint(checkpoint_path: str, device: torch.device = None) -> tuple:
    """
    Loads an exported EEGNet checkpoint and reconstructs model and normalization tensors.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    channels = checkpoint['channels']
    samples = checkpoint['samples']
    
    model = EEGNet(n_classes=2, channels=channels, samples=samples).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    norm_mean = checkpoint['norm_mean'].to(device)
    norm_std = checkpoint['norm_std'].to(device)

    return model, norm_mean, norm_std, checkpoint.get('metadata', {})

def train_and_export_subject(
    subject: str,
    bids_root: str = "data/ds003846",
    output_dir: str = "data/ds003846/derivatives",
    sessions: list = None,
    max_epochs: int = 60
):
    """
    Loads all sessions for a subject, segments event-locked epochs, performs chronological
    split, trains EEGNet, and saves 'sub-<id>_eegnet.pt'.
    """
    if sessions is None:
        sessions = ["EMS", "Vibro", "Visual"]

    sub_id = f"sub-{subject}" if not subject.startswith("sub-") else subject
    sub_num = sub_id.replace("sub-", "")

    print("\n" + "="*60)
    print(f"[*] Training and Calibrating EEGNet for: {sub_id}")
    print("="*60)

    all_epochs_list = []
    all_labels_list = []

    for ses in sessions:
        ses_dir = os.path.join(bids_root, sub_id, f"ses-{ses}")
        if not os.path.exists(ses_dir):
            continue

        try:
            print(f"    [*] Loading & preprocessing {sub_id} ses-{ses}...")
            raw = load_bids_data(sub_num, ses, bids_root)
            raw_clean = preprocess_raw(raw)
            epochs, labels, meta = extract_event_epochs(raw_clean, event_target='box:touched', tmin=-0.1, tmax=0.6)
            if epochs is not None and len(epochs) > 0:
                all_epochs_list.append(epochs)
                all_labels_list.append(epochs.events[:, 2])
        except Exception as e:
            print(f"    [!] Error loading {sub_id} ses-{ses}: {e}")

    if not all_epochs_list:
        raise RuntimeError(f"No valid event epochs found for {sub_id}")

    epochs_sub = mne.concatenate_epochs(all_epochs_list, on_mismatch='ignore', verbose=False)
    y_sub = np.concatenate(all_labels_list)
    n_samples = len(epochs_sub)
    n_flow = np.sum(y_sub == 1)
    n_disrupted = np.sum(y_sub == 0)

    print(f"[+] Total Subject Epochs: {n_samples} (Flow/Normal: {n_flow}, Disrupted/Conflict: {n_disrupted})")

    # 80/20 Chronological Split
    split_idx = int(n_samples * 0.8)
    epochs_train = epochs_sub[:split_idx]
    epochs_val = epochs_sub[split_idx:]
    y_train = y_sub[:split_idx]
    y_val = y_sub[split_idx:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, norm_params, metrics = train_eegnet_classifier(
        epochs_train=epochs_train,
        y_train=y_train,
        epochs_val=epochs_val,
        y_val=y_val,
        device=device,
        max_epochs=max_epochs
    )

    print(f"[+] Validation Balanced Accuracy: {metrics['balanced_accuracy']:.2%}")
    print(f"    Precision: {metrics.get('precision', 0):.3f} | Recall: {metrics.get('recall', 0):.3f} | F1: {metrics.get('f1_score', 0):.3f}")

    save_path = os.path.join(output_dir, f"{sub_id}_eegnet.pt")
    save_eegnet_checkpoint(model, norm_params, save_path, extra_meta={
        'subject_id': sub_id,
        'val_balanced_accuracy': metrics['balanced_accuracy'],
        'n_train_trials': len(epochs_train),
        'n_val_trials': len(epochs_val)
    })
    return save_path, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & Calibrate EEGNet Flow Classifier")
    parser.add_argument("--subject", type=str, default="02", help="Subject ID (e.g. 02)")
    parser.add_argument("--bids-root", type=str, default="data/ds003846", help="Path to BIDS dataset")
    parser.add_argument("--output-dir", type=str, default="data/ds003846/derivatives", help="Output directory for .pt models")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    args = parser.parse_args()

    train_and_export_subject(
        subject=args.subject,
        bids_root=args.bids_root,
        output_dir=args.output_dir,
        max_epochs=args.epochs
    )
