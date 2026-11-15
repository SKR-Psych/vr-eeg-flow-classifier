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
from src.eegnet import (
    EEGNet,
    MS_SE_EEGNet,
    EEGMoEClassifier,
    compute_eeg_signal_quality_features
)
import torch.nn.functional as F


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
    dropout_rate: float = 0.25,
    model_type: str = 'eegnet'
) -> tuple:
    """
    Trains and calibrates an EEGNet or MS-SE-EEGNet deep neural network classifier on event-locked EEG epochs.

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

    if model_type.lower() in ['moe', 'eeg_moe', 'eegmoe']:
        return train_moe_classifier(
            epochs_train=epochs_train,
            y_train=y_train,
            epochs_val=epochs_val,
            y_val=y_val,
            device=device,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            dropout_rate=dropout_rate
        )


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
    if model_type.lower() in ['ms_se_eegnet', 'ms-se-eegnet', 'msseeegnet']:
        model = MS_SE_EEGNet(n_classes=2, channels=channels, samples=samples, dropout_rate=dropout_rate).to(device)
    else:
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


def generate_synthetic_eeg_artifacts(
    X_clean: torch.Tensor,
    sfreq: float = 500.0,
    artifact_mode: str = "mixed",
    severity: float = 1.0,
    rng: np.random.Generator = None
) -> tuple:
    """
    Synthesizes realistic, neuro-physically grounded VR and clinical movement artifacts on standardized
    EEG epoch tensors for training and validating the Mixture-of-Experts gating layer.

    Modes supported:
    - 'tremor' : 4-6 Hz Parkinsonian rest/postural tremor + 8-12 Hz action tremor harmonic bursts
                 coupled to motor and peripheral scalp channels.
    - 'jerk'   : Sudden VR HMD head-strap slip / mechanical jerk step-transients (high d^3x/dt^3)
                 and focal electrode pop spikes on perimeter channels.
    - 'emg'    : High-frequency (25-45 Hz) cranial/jaw muscle bursts on frontal/temporal/occipital leads.
    - 'mixed'  : Randomly assigns each trial in the batch to 'tremor', 'jerk', or 'emg' (or combination).

    Parameters
    ----------
    X_clean : torch.Tensor
        Standardized clean EEG tensor of shape (N, 1, C, T).
    sfreq : float, optional
        Sampling rate in Hz (default: 500.0).
    artifact_mode : str, optional
        Type of artifact ('tremor', 'jerk', 'emg', 'mixed').
    severity : float, optional
        Amplitude scaling factor relative to standardized channel unit variance (default: 1.0).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    X_noisy : torch.Tensor
        Artifact-perturbed EEG tensor of shape (N, 1, C, T).
    subtypes : list[str]
        List of artifact subtype labels per trial.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    device = X_clean.device
    dtype = X_clean.dtype
    X_np = X_clean.detach().cpu().numpy().copy()
    n_trials, _, n_ch, n_samples = X_np.shape
    t = np.arange(n_samples, dtype=np.float32) / float(sfreq)

    # Peripheral / strap-bearing channel subset (approx 35% of channels: frontal/temporal/occipital indices)
    n_periph = max(4, n_ch // 3)
    periph_pool = np.unique(np.concatenate([
        np.arange(0, min(8, n_ch)),
        np.arange(max(0, n_ch - n_periph), n_ch)
    ]))

    subtypes = []
    for i in range(n_trials):
        mode = artifact_mode
        if mode == "mixed":
            mode = rng.choice(["tremor", "jerk", "emg", "tremor_jerk"])
        subtypes.append(mode)

        # Select 25% to 50% of channels affected in this trial
        n_affected = int(rng.integers(max(3, n_ch // 5), max(6, n_ch // 2)))
        affected_chs = rng.choice(periph_pool if len(periph_pool) >= n_affected else np.arange(n_ch),
                                  size=n_affected, replace=False)

        if "tremor" in mode:
            # 4.5 - 6.5 Hz primary Parkinsonian tremor + 9 - 11 Hz harmonic
            f_tremor = float(rng.uniform(4.5, 6.5))
            f_harm = f_tremor * 2.0
            phase0 = float(rng.uniform(0, 2 * np.pi))
            phase1 = float(rng.uniform(0, 2 * np.pi))

            # Smooth burst envelope (intermittent within or across the 700ms window)
            burst_center = float(rng.uniform(0.15, 0.55))
            burst_width = float(rng.uniform(0.12, 0.30))
            envelope = np.exp(-0.5 * ((t - burst_center) / burst_width) ** 2)

            tremor_wave = (
                1.8 * severity * np.sin(2 * np.pi * f_tremor * t + phase0)
                + 0.9 * severity * np.sin(2 * np.pi * f_harm * t + phase1)
            ) * envelope

            ch_scales = rng.uniform(0.6, 1.5, size=(n_affected, 1)).astype(np.float32)
            X_np[i, 0, affected_chs, :] += ch_scales * tremor_wave[None, :]

        if "jerk" in mode:
            # Sudden mechanical HMD strap slip (high 3rd derivative / jerk) + transient electrode pop
            n_jerks = int(rng.integers(2, 5))
            for _ in range(n_jerks):
                t_idx = int(rng.integers(15, n_samples - 25))
                jerk_amp = float(rng.choice([-1.0, 1.0]) * rng.uniform(2.5, 4.5) * severity)
                # Sharp biphasic transient (high d^3x/dt^3) + exponential decay
                pulse_len = int(rng.integers(6, 16))
                pulse = np.zeros(n_samples, dtype=np.float32)
                pulse[t_idx] = jerk_amp
                pulse[t_idx + 1] = -0.85 * jerk_amp
                pulse[t_idx + 2:t_idx + pulse_len] = (
                    0.4 * jerk_amp * np.exp(-np.linspace(0, 3.0, pulse_len - 2, dtype=np.float32))
                )
                ch_sub = rng.choice(affected_chs, size=max(2, len(affected_chs) // 2), replace=False)
                ch_polarity = rng.choice([-1.0, 1.0], size=(len(ch_sub), 1)).astype(np.float32)
                X_np[i, 0, ch_sub, :] += ch_polarity * pulse[None, :]

        if "emg" in mode:
            # High-frequency cranial muscle artifact (25 - 45 Hz band-limited burst)
            white = rng.standard_normal((n_affected, n_samples)).astype(np.float32)
            freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
            spec = np.fft.rfft(white, axis=-1)
            emg_mask = (freqs >= 24.0) & (freqs <= 45.0)
            spec[:, ~emg_mask] *= 0.05
            emg_noise = np.fft.irfft(spec, n=n_samples, axis=-1).astype(np.float32)
            emg_noise /= (np.std(emg_noise, axis=-1, keepdims=True) + 1e-6)

            # Localize burst in time
            b_start = int(rng.integers(0, n_samples // 2))
            b_end = min(n_samples, b_start + int(rng.integers(n_samples // 3, n_samples)))
            win = np.zeros((1, n_samples), dtype=np.float32)
            win[0, b_start:b_end] = 1.0
            X_np[i, 0, affected_chs, :] += (1.7 * severity) * emg_noise * win

    return torch.tensor(X_np, dtype=dtype, device=device), subtypes


def train_moe_classifier(
    epochs_train: mne.Epochs,
    y_train: np.ndarray,
    epochs_val: mne.Epochs = None,
    y_val: np.ndarray = None,
    device: torch.device = None,
    max_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    dropout_rate: float = 0.25
) -> tuple:
    """
    Trains the Bidirectional Mixture-of-Experts (EEGMoEClassifier) model by training the gating network
    alongside Expert 0 (Vanilla EEGNet) and Expert 1 (MS-SE-EEGNet).

    Training stages:
    1. Expert Specialization & Joint Co-Training:
       - Clean epochs (regime=0) supervise Expert 0 (EEGNet) and clean task mixture.
       - Artifact-augmented epochs (regime=1: intermittent tremor, HMD jerk, EMG) supervise
         Expert 1 (MS-SE-EEGNet) and noisy task mixture.
       - Gate supervision loss trains EEGGatingNetwork to route clean -> Expert 0 and noisy -> Expert 1
         in either direction.
       - Probability alignment loss on clean data ensures switching models does not shift baseline P(Flow).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train_raw = torch.tensor(epochs_train.get_data(units='uV'), dtype=torch.float32).unsqueeze(1)

    norm_mean = X_train_raw.mean(dim=(0, 3), keepdim=True)
    norm_std = X_train_raw.std(dim=(0, 3), keepdim=True) + 1e-6

    X_train_clean = (X_train_raw - norm_mean) / norm_std
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    n_samples, _, channels, samples = X_train_clean.shape
    model = EEGMoEClassifier(
        n_classes=2,
        channels=channels,
        samples=samples,
        dropout_rate=dropout_rate
    ).to(device)

    # Pre-calibrate gate signal-quality feature standardization on clean + synthetic noisy training set
    rng = np.random.default_rng(42)
    with torch.no_grad():
        X_synth_init, _ = generate_synthetic_eeg_artifacts(X_train_clean, artifact_mode="mixed", severity=1.0, rng=rng)
        feats_clean = compute_eeg_signal_quality_features(X_train_clean.to(device))
        feats_noisy = compute_eeg_signal_quality_features(X_synth_init.to(device))
        all_gate_feats = torch.cat([feats_clean, feats_noisy], dim=0)
        model.gate_feat_mean.copy_(all_gate_feats.mean(dim=0, keepdim=True))
        model.gate_feat_std.copy_(all_gate_feats.std(dim=0, keepdim=True) + 1e-5)

    # Balanced class weights for task CrossEntropy
    n_0 = np.sum(y_train == 0)
    n_1 = np.sum(y_train == 1)
    w0 = len(y_train) / (2.0 * max(n_0, 1))
    w1 = len(y_train) / (2.0 * max(n_1, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)

    task_criterion = nn.CrossEntropyLoss(weight=class_weights)
    gate_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    train_ds = TensorDataset(X_train_clean, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    has_val = epochs_val is not None and y_val is not None and len(y_val) > 0
    if has_val:
        X_val_raw = torch.tensor(epochs_val.get_data(units='uV'), dtype=torch.float32).unsqueeze(1)
        X_val_clean = (X_val_raw - norm_mean) / norm_std
        X_val_noisy, _ = generate_synthetic_eeg_artifacts(
            X_val_clean, artifact_mode="mixed", severity=1.0, rng=np.random.default_rng(1337)
        )

    best_score = -1.0
    best_state_dict = None
    best_metrics = {}

    # Pre-generate paired artifact-augmented training dataset so epoch loop is pure PyTorch tensor math
    X_train_noisy_bank, _ = generate_synthetic_eeg_artifacts(
        X_train_clean, artifact_mode="mixed", severity=1.05, rng=rng
    )
    train_ds = TensorDataset(X_train_clean, X_train_noisy_bank, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(max_epochs):
        model.train()
        for bx_clean, bx_noisy_base, by in train_loader:
            bx_clean = bx_clean.to(device)
            bx_noisy_base = bx_noisy_base.to(device)
            by = by.to(device)
            bsz = bx_clean.size(0)

            # Fast on-device amplitude jitter of artifact perturbation
            scale = 0.80 + 0.45 * torch.rand(bsz, 1, 1, 1, device=device)
            bx_noisy = bx_clean + scale * (bx_noisy_base - bx_clean)


            optimizer.zero_grad()

            # Combine clean (regime=0) and artifact-contaminated (regime=1) batches into a single
            # forward pass so in-place max-norm weight constraints in Conv2dWithConstraint / LinearWithConstraint
            # execute once per backward pass.
            bx_combined = torch.cat([bx_clean, bx_noisy], dim=0)
            logits_all, info_all = model(bx_combined, return_gate_info=True)

            logits_clean = logits_all[:bsz]
            logits_noisy = logits_all[bsz:]

            # 1. Mixture task loss on both clean and noisy conditions
            loss_mix_clean = task_criterion(logits_clean, by)
            loss_mix_noisy = task_criterion(logits_noisy, by)

            # 2. Expert specialization losses:
            # Expert 0 (EEGNet) specializes on clean data; Expert 1 (MS-SE-EEGNet) trains on both clean & noisy
            loss_e0_clean = task_criterion(info_all["logits_eegnet"][:bsz], by)
            loss_e1_noisy = task_criterion(info_all["logits_msse"][bsz:], by)
            loss_e1_clean = task_criterion(info_all["logits_msse"][:bsz], by)

            # 3. Bidirectional gate routing loss (0 = EEGNet for clean, 1 = MS-SE-EEGNet for noisy)
            gate_targets = torch.cat([
                torch.zeros(bsz, dtype=torch.long, device=device),
                torch.ones(bsz, dtype=torch.long, device=device)
            ], dim=0)
            loss_gate = gate_criterion(info_all["gate_logits"], gate_targets)

            # 4. Calibration alignment loss: keep P_0(Flow) and P_1(Flow) aligned on clean trials
            # so switching models does not shift the measured cognitive state probability
            loss_align = F.mse_loss(info_all["probs_eegnet"][:bsz], info_all["probs_msse"][:bsz])


            total_loss = (
                0.5 * (loss_mix_clean + loss_mix_noisy)
                + 0.5 * (loss_e0_clean + 0.6 * loss_e1_noisy + 0.4 * loss_e1_clean)
                + 0.4 * loss_gate
                + 1.0 * loss_align
            )
            total_loss.backward()
            optimizer.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                val_logits_c, val_info_c = model(X_val_clean.to(device), return_gate_info=True)
                val_logits_n, val_info_n = model(X_val_noisy.to(device), return_gate_info=True)

                probs_c = val_info_c["combined_probs"][:, 1].cpu().numpy()
                preds_c = np.argmax(val_info_c["combined_probs"].cpu().numpy(), axis=1)
                acc_c = balanced_accuracy_score(y_val, preds_c)

                preds_n = np.argmax(val_info_n["combined_probs"].cpu().numpy(), axis=1)
                acc_n = balanced_accuracy_score(y_val, preds_n)

                # Gate partitioning accuracy (Clean -> 0, Noisy -> 1)
                route_c_acc = float((val_info_c["selected_expert"] == 0).float().mean().item())
                route_n_acc = float((val_info_n["selected_expert"] == 1).float().mean().item())
                gate_acc = 0.5 * (route_c_acc + route_n_acc)

                # Mean absolute probability discrepancy between experts on clean validation epochs
                prob_drift = float(torch.abs(
                    val_info_c["probs_eegnet"][:, 1] - val_info_c["probs_msse"][:, 1]
                ).mean().item())

                composite_score = 0.55 * acc_c + 0.30 * acc_n + 0.15 * gate_acc
                if composite_score >= best_score or best_state_dict is None:
                    best_score = composite_score
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_metrics = {
                        'balanced_accuracy': acc_c,
                        'noisy_balanced_accuracy': acc_n,
                        'gate_partition_accuracy': gate_acc,
                        'clean_route_eegnet_rate': route_c_acc,
                        'noisy_route_msse_rate': route_n_acc,
                        'inter_expert_prob_drift': prob_drift,
                        'precision': precision_score(y_val, preds_c, zero_division=0),
                        'recall': recall_score(y_val, preds_c, zero_division=0),
                        'f1_score': f1_score(y_val, preds_c, zero_division=0),
                        'roc_auc': roc_auc_score(y_val, probs_c) if len(np.unique(y_val)) > 1 else 0.5,
                        'confusion_matrix': confusion_matrix(y_val, preds_c)
                    }
                scheduler.step(composite_score)
        else:
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {'balanced_accuracy': 0.0}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    norm_params = {
        'mean': norm_mean.cpu(),
        'std': norm_std.cpu(),
        'channels': channels,
        'samples': samples
    }
    return model, norm_params, best_metrics


def save_eegnet_checkpoint(model: nn.Module, norm_params: dict, save_path: str, extra_meta: dict = None):
    """
    Saves a self-contained calibrated EEGNet, MS-SE-EEGNet, or EEGMoEClassifier model checkpoint (.pt).
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if isinstance(model, EEGMoEClassifier):
        arch_name = 'EEGMoEClassifier'
    elif isinstance(model, MS_SE_EEGNet):
        arch_name = 'MS_SE_EEGNet'
    else:
        arch_name = 'EEGNet'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'channels': norm_params['channels'],
        'samples': norm_params['samples'],
        'norm_mean': norm_params['mean'],
        'norm_std': norm_params['std'],
        'architecture': arch_name,
        'sfreq': 500.0,
        'metadata': extra_meta or {}
    }
    torch.save(checkpoint, save_path)
    print(f"[+] Successfully exported calibrated {arch_name} checkpoint to: {save_path}")

def load_eegnet_checkpoint(checkpoint_path: str, device: torch.device = None) -> tuple:
    """
    Loads an exported EEGNet, MS-SE-EEGNet, or EEGMoEClassifier checkpoint and reconstructs model and normalization tensors.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    channels = checkpoint['channels']
    samples = checkpoint['samples']
    architecture = checkpoint.get('architecture', 'EEGNet')

    if architecture == 'EEGMoEClassifier':
        model = EEGMoEClassifier(n_classes=2, channels=channels, samples=samples).to(device)
    elif architecture == 'MS_SE_EEGNet':
        model = MS_SE_EEGNet(n_classes=2, channels=channels, samples=samples).to(device)
    else:
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
    max_epochs: int = 60,
    model_type: str = "eegnet"
):
    """
    Loads all sessions for a subject, segments event-locked epochs, performs chronological
    split, trains EEGNet or MS-SE-EEGNet, and saves the calibrated model checkpoint.
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
        max_epochs=max_epochs,
        model_type=model_type
    )

    print(f"[+] Validation Balanced Accuracy: {metrics['balanced_accuracy']:.2%}")
    print(f"    Precision: {metrics.get('precision', 0):.3f} | Recall: {metrics.get('recall', 0):.3f} | F1: {metrics.get('f1_score', 0):.3f}")
    if 'gate_partition_accuracy' in metrics:
        print(f"    Noisy Balanced Accuracy: {metrics['noisy_balanced_accuracy']:.2%} | Gate Partition Acc: {metrics['gate_partition_accuracy']:.2%}")
        print(f"    Inter-Expert Clean Drift: {metrics['inter_expert_prob_drift']:.4f}")

    if model_type.lower() in ['moe', 'eeg_moe', 'eegmoe']:
        model_suffix = "moe_eegnet"
    elif model_type.lower() in ['ms_se_eegnet', 'ms-se-eegnet', 'msseeegnet']:
        model_suffix = "ms_se_eegnet"
    else:
        model_suffix = "eegnet"
    save_path = os.path.join(output_dir, f"{sub_id}_{model_suffix}.pt")
    save_eegnet_checkpoint(model, norm_params, save_path, extra_meta={
        'subject_id': sub_id,
        'val_balanced_accuracy': metrics['balanced_accuracy'],
        'model_type': model_type,
        'n_train_trials': len(epochs_train),
        'n_val_trials': len(epochs_val)
    })
    return save_path, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & Calibrate Flow Classifier (EEGNet / MS-SE-EEGNet / MoE)")
    parser.add_argument("--subject", type=str, default="02", help="Subject ID (e.g. 02)")
    parser.add_argument("--bids-root", type=str, default="data/ds003846", help="Path to BIDS dataset")
    parser.add_argument("--output-dir", type=str, default="data/ds003846/derivatives", help="Output directory for .pt models")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--model-type", type=str, default="eegnet", choices=["eegnet", "ms_se_eegnet", "moe"], help="Architecture to train (default: eegnet)")
    args = parser.parse_args()


    train_and_export_subject(
        subject=args.subject,
        bids_root=args.bids_root,
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        model_type=args.model_type
    )
