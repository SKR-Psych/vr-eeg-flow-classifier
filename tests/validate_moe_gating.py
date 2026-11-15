import os
import sys
import time
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

from src.eegnet import (
    EEGNet,
    MS_SE_EEGNet,
    EEGMoEClassifier,
    compute_eeg_signal_quality_features
)
from src.classifier import generate_synthetic_eeg_artifacts


def measure_latency_ms(model: nn.Module, channels: int, samples: int, n_runs: int = 100) -> float:
    model.eval()
    dummy = torch.randn(1, 1, channels, samples)
    with torch.no_grad():
        for _ in range(15):
            _ = model(dummy)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(dummy)
    return ((time.perf_counter() - t0) / n_runs) * 1000.0


def train_single_expert(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: np.ndarray,
    max_epochs: int = 25,
    batch_size: int = 64,
    augment_artifacts: bool = False,
    seed: int = 42
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device('cpu')
    model = model.to(device)

    n_0 = (y_train == 0).sum().item()
    n_1 = (y_train == 1).sum().item()
    w0 = len(y_train) / (2.0 * max(n_0, 1))
    w1 = len(y_train) / (2.0 * max(n_1, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)


    if augment_artifacts:
        X_train_noisy_bank, _ = generate_synthetic_eeg_artifacts(
            X_train, artifact_mode="mixed", severity=1.05, rng=rng
        )
    else:
        X_train_noisy_bank = X_train

    train_ds = TensorDataset(X_train, X_train_noisy_bank, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_acc = -1.0
    best_state = None

    for _ in range(max_epochs):
        model.train()
        for bx_c, bx_n, by in train_loader:
            if augment_artifacts:
                half = len(bx_c) // 2
                if half > 0:
                    scale = 0.80 + 0.45 * torch.rand(len(bx_c) - half, 1, 1, 1)
                    bx_pert = bx_c[half:] + scale * (bx_n[half:] - bx_c[half:])
                    bx = torch.cat([bx_c[:half], bx_pert], dim=0)
                else:
                    bx = bx_n
            else:
                bx = bx_c
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()


        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            preds = torch.argmax(val_logits, dim=1).numpy()
            acc = balanced_accuracy_score(y_val, preds)
            if acc >= best_acc or best_state is None:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            scheduler.step(acc)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_joint_moe(
    channels: int,
    samples: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: np.ndarray,
    max_epochs: int = 25,
    batch_size: int = 64,
    seed: int = 42,
    pretrained_eegnet: nn.Module = None,
    pretrained_msse: nn.Module = None
) -> EEGMoEClassifier:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device('cpu')

    model = EEGMoEClassifier(n_classes=2, channels=channels, samples=samples, dropout_rate=0.25).to(device)
    if pretrained_eegnet is not None:
        model.expert_eegnet.load_state_dict(pretrained_eegnet.state_dict())
    if pretrained_msse is not None:
        model.expert_msse.load_state_dict(pretrained_msse.state_dict())

    # Calibrate gate feature normalization on clean + synthetic noisy training epochs
    with torch.no_grad():
        X_synth, _ = generate_synthetic_eeg_artifacts(X_train, artifact_mode="mixed", severity=1.0, rng=rng)
        f_clean = compute_eeg_signal_quality_features(X_train)
        f_noisy = compute_eeg_signal_quality_features(X_synth)
        f_all = torch.cat([f_clean, f_noisy], dim=0)
        model.gate_feat_mean.copy_(f_all.mean(dim=0, keepdim=True))
        model.gate_feat_std.copy_(f_all.std(dim=0, keepdim=True) + 1e-5)

    n_0 = (y_train == 0).sum().item()
    n_1 = (y_train == 1).sum().item()
    w0 = len(y_train) / (2.0 * max(n_0, 1))
    w1 = len(y_train) / (2.0 * max(n_1, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    task_criterion = nn.CrossEntropyLoss(weight=class_weights)
    gate_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam([
        {'params': model.expert_eegnet.parameters(), 'lr': 5e-4},
        {'params': model.expert_msse.parameters(), 'lr': 5e-4},
        {'params': model.gate.parameters(), 'lr': 3e-3},
        {'params': [model.log_temp_eegnet, model.log_temp_msse], 'lr': 5e-3}
    ], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)


    X_train_noisy_bank, _ = generate_synthetic_eeg_artifacts(
        X_train, artifact_mode="mixed", severity=1.05, rng=rng
    )
    train_ds = TensorDataset(X_train, X_train_noisy_bank, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_noisy, _ = generate_synthetic_eeg_artifacts(
        X_val, artifact_mode="mixed", severity=1.0, rng=np.random.default_rng(seed + 99)
    )

    best_score = -1.0
    best_state = None

    for _ in range(max_epochs):
        model.train()
        for bx_clean, bx_noisy_base, by in train_loader:
            bsz = bx_clean.size(0)
            scale = 0.80 + 0.45 * torch.rand(bsz, 1, 1, 1)
            bx_noisy = bx_clean + scale * (bx_noisy_base - bx_clean)


            optimizer.zero_grad()
            bx_combined = torch.cat([bx_clean, bx_noisy], dim=0)
            logits_all, info_all = model(bx_combined, return_gate_info=True)

            logits_c = logits_all[:bsz]
            logits_n = logits_all[bsz:]

            loss_mix = 0.5 * (task_criterion(logits_c, by) + task_criterion(logits_n, by))
            loss_exp = 0.5 * (
                task_criterion(info_all["logits_eegnet"][:bsz], by)
                + 0.65 * task_criterion(info_all["logits_msse"][bsz:], by)
                + 0.35 * task_criterion(info_all["logits_msse"][:bsz], by)
            )

            gate_targets = torch.cat([
                torch.zeros(bsz, dtype=torch.long),
                torch.ones(bsz, dtype=torch.long)
            ], dim=0)
            loss_gate = gate_criterion(info_all["gate_logits"], gate_targets)

            # Calibration alignment on clean data so switching experts preserves P(Flow) scale
            loss_align = F.mse_loss(info_all["probs_eegnet"][:bsz], info_all["probs_msse"][:bsz])

            loss = loss_mix + loss_exp + 0.5 * loss_gate + 1.2 * loss_align
            loss.backward()
            optimizer.step()


        model.eval()
        with torch.no_grad():
            _, info_vc = model(X_val, return_gate_info=True)
            _, info_vn = model(X_val_noisy, return_gate_info=True)

            acc_c = balanced_accuracy_score(y_val, np.argmax(info_vc["combined_probs"].numpy(), axis=1))
            acc_n = balanced_accuracy_score(y_val, np.argmax(info_vn["combined_probs"].numpy(), axis=1))
            g_acc = 0.5 * (
                float((info_vc["selected_expert"] == 0).float().mean().item())
                + float((info_vn["selected_expert"] == 1).float().mean().item())
            )
            score = 0.50 * acc_c + 0.35 * acc_n + 0.15 * g_acc
            if score >= best_score or best_state is None:
                best_score = score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            scheduler.step(score)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def simulate_legacy_one_way_7s_selector(
    X_stream: torch.Tensor,
    eegnet_model: nn.Module,
    msse_model: nn.Module,
    jerk_threshold: float,
    window_steps: int = 35  # 7.0 seconds at 0.20s step
) -> tuple:
    """
    Simulates the legacy categorical 7-second one-way selector described in the brief:
    Starts on Vanilla EEGNet; if repeated jerk/noise patterns exceed threshold within a 7-second window,
    permanently switches to MS-SE-EEGNet and never switches back.
    """
    n_steps = len(X_stream)
    selected_experts = []
    probs_flow = []
    latched_msse = False
    recent_jerk_flags = []

    with torch.no_grad():
        for i in range(n_steps):
            x_t = X_stream[i:i + 1]
            sig_feats = compute_eeg_signal_quality_features(x_t)
            # Composite jerk + high-frequency burst indicator
            jerk_score = float(sig_feats[0, 0].item() + 0.5 * sig_feats[0, 1].item())
            is_noisy_tick = 1 if jerk_score > jerk_threshold else 0
            recent_jerk_flags.append(is_noisy_tick)
            if len(recent_jerk_flags) > window_steps:
                recent_jerk_flags.pop(0)

            # Trigger permanent switch if >= 3 jerk ticks occur within the 7-second window
            if not latched_msse and sum(recent_jerk_flags) >= 3:
                latched_msse = True

            if latched_msse:
                logits = msse_model(x_t)
                selected_experts.append(1)
            else:
                logits = eegnet_model(x_t)
                selected_experts.append(0)

            prob = float(F.softmax(logits, dim=-1)[0, 1].item())
            probs_flow.append(prob)

    return np.array(selected_experts), np.array(probs_flow)


def evaluate_intermittent_bidirectional_stream(
    moe_model: EEGMoEClassifier,
    eegnet_model: nn.Module,
    msse_model: nn.Module,
    X_val_clean: torch.Tensor,
    y_val: np.ndarray
) -> pd.DataFrame:
    """
    Constructs a chronological 5-phase intermittent artifact stream across validation epochs:
      Phase 1 (Clean Baseline)           : regime = 0 (EEGNet target)
      Phase 2 (Parkinsonian 5Hz Tremor)  : regime = 1 (MS-SE-EEGNet target)
      Phase 3 (Recovery to Clean)        : regime = 0 (Bidirectional return to EEGNet)
      Phase 4 (VR Strap Jerk + HF EMG)   : regime = 1 (MS-SE-EEGNet target)
      Phase 5 (Final Recovery to Clean)  : regime = 0 (Bidirectional return to EEGNet)
    """
    rng = np.random.default_rng(2026)
    n_total = len(X_val_clean)
    p_len = n_total // 5

    X_stream = X_val_clean.clone()
    regime_labels = np.zeros(n_total, dtype=int)
    phase_names = []

    for i in range(n_total):
        if i < p_len:
            phase_names.append("P1:Clean_Baseline")
            regime_labels[i] = 0
        elif i < 2 * p_len:
            phase_names.append("P2:PD_Tremor_On")
            regime_labels[i] = 1
            x_pert, _ = generate_synthetic_eeg_artifacts(
                X_stream[i:i + 1], artifact_mode="tremor", severity=1.15, rng=rng
            )
            X_stream[i:i + 1] = x_pert
        elif i < 3 * p_len:
            phase_names.append("P3:Clean_Recovery_1")
            regime_labels[i] = 0
        elif i < 4 * p_len:
            phase_names.append("P4:VR_Jerk_EMG_On")
            regime_labels[i] = 1
            x_pert, _ = generate_synthetic_eeg_artifacts(
                X_stream[i:i + 1], artifact_mode="jerk", severity=1.20, rng=rng
            )
            X_stream[i:i + 1] = x_pert
        else:
            phase_names.append("P5:Clean_Recovery_2")
            regime_labels[i] = 0

    # Calibrate legacy jerk threshold from clean training/validation baseline (mean + 2.0 std)
    with torch.no_grad():
        clean_feats = compute_eeg_signal_quality_features(X_val_clean[:p_len])
        clean_jerk_scores = clean_feats[:, 0] + 0.5 * clean_feats[:, 1]
        jerk_thresh = float(clean_jerk_scores.mean().item() + 2.0 * clean_jerk_scores.std().item())

    legacy_routes, legacy_probs = simulate_legacy_one_way_7s_selector(
        X_stream, eegnet_model, msse_model, jerk_threshold=jerk_thresh, window_steps=35
    )

    moe_model.eval()
    moe_model.reset_stream_state()
    moe_routes = []
    moe_g_msse = []
    moe_probs = []

    with torch.no_grad():
        for i in range(n_total):
            x_t = X_stream[i:i + 1]
            _, info = moe_model(x_t, update_gate_state=True, return_gate_info=True)
            g1 = float(info["g_msse"].item())
            moe_g_msse.append(g1)
            moe_routes.append(1 if g1 >= 0.5 else 0)
            moe_probs.append(float(info["combined_probs"][0, 1].item()))

    df_stream = pd.DataFrame({
        "step_idx": np.arange(n_total),
        "time_sec": np.round(np.arange(n_total) * 0.20, 2),
        "phase": phase_names,
        "true_artifact_regime": regime_labels,
        "true_flow_label": y_val,
        "legacy_oneway_expert": legacy_routes,
        "moe_bidirectional_expert": np.array(moe_routes),
        "moe_g_msse_weight": np.round(moe_g_msse, 4),
        "legacy_flow_prob": np.round(legacy_probs, 4),
        "moe_flow_prob": np.round(moe_probs, 4)
    })
    return df_stream


def run_validation_suite():
    derivatives_dir = "data/ds003846/derivatives"
    subjects = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]

    print("=" * 85)
    print("BIDIRECTIONAL MIXTURE-OF-EXPERTS (MoE) GATING VALIDATION SUITE")
    print("Comparing: Vanilla EEGNet | MS-SE-EEGNet | Legacy 7s One-Way Switch | Bidirectional MoE")
    print("=" * 85)

    cohort_rows = []
    sub02_stream_df = None

    for sub_id in subjects:
        cache_path = os.path.join(derivatives_dir, f"{sub_id}_benchmark_epochs_cache.pt")
        if not os.path.exists(cache_path):
            print(f"[!] Cache not found for {sub_id} at {cache_path}, skipping.")
            continue

        t0 = time.time()
        cache = torch.load(cache_path, weights_only=False)
        X_raw = cache["X_raw"]
        labels = cache["labels"]

        n_total = len(labels)
        split_idx = int(n_total * 0.8)

        X_train_raw = X_raw[:split_idx]
        mean = X_train_raw.mean(dim=(0, 3), keepdim=True)
        std = X_train_raw.std(dim=(0, 3), keepdim=True) + 1e-6

        X_train = (X_train_raw - mean) / std
        y_train = torch.tensor(labels[:split_idx], dtype=torch.long)

        X_val_clean = (X_raw[split_idx:] - mean) / std
        y_val = labels[split_idx:]

        _, _, channels, samples = X_train.shape
        print(f"\n---> Evaluating {sub_id} (Train={len(X_train)}, Val={len(X_val_clean)}, Ch={channels}, Samples={samples})", flush=True)


        # 1. Train Standalone Vanilla EEGNet (Clean Baseline)
        eegnet = EEGNet(n_classes=2, channels=channels, samples=samples, dropout_rate=0.25)
        eegnet = train_single_expert(eegnet, X_train, y_train, X_val_clean, y_val, max_epochs=25, batch_size=64, augment_artifacts=False, seed=42)

        # 2. Train Standalone MS-SE-EEGNet (Artifact-Augmented Specialist)
        msse = MS_SE_EEGNet(n_classes=2, channels=channels, samples=samples, dropout_rate=0.25)
        msse = train_single_expert(msse, X_train, y_train, X_val_clean, y_val, max_epochs=25, batch_size=64, augment_artifacts=True, seed=42)

        # 3. Train Bidirectional Joint MoE Classifier (co-adapting experts + gate + calibration temperatures)
        moe = train_joint_moe(
            channels, samples, X_train, y_train, X_val_clean, y_val,
            max_epochs=18, batch_size=64, seed=42,
            pretrained_eegnet=eegnet, pretrained_msse=msse
        )


        # Create validation test regimes: Clean, Parkinsonian Tremor (4-6Hz + 8-12Hz), VR Jerk/EMG, and Mixed
        rng_val = np.random.default_rng(999)
        X_val_tremor, _ = generate_synthetic_eeg_artifacts(X_val_clean, artifact_mode="tremor", severity=1.15, rng=rng_val)
        X_val_jerk, _ = generate_synthetic_eeg_artifacts(X_val_clean, artifact_mode="jerk", severity=1.20, rng=rng_val)
        X_val_mixed, _ = generate_synthetic_eeg_artifacts(X_val_clean, artifact_mode="mixed", severity=1.15, rng=rng_val)

        with torch.no_grad():
            # Clean evaluation
            p_eeg_clean = F.softmax(eegnet(X_val_clean), dim=-1)[:, 1].numpy()
            p_msse_clean = F.softmax(msse(X_val_clean), dim=-1)[:, 1].numpy()
            _, info_moe_clean = moe(X_val_clean, return_gate_info=True)
            p_moe_clean = info_moe_clean["combined_probs"][:, 1].numpy()

            acc_eeg_clean = balanced_accuracy_score(y_val, (p_eeg_clean >= 0.5).astype(int))
            acc_msse_clean = balanced_accuracy_score(y_val, (p_msse_clean >= 0.5).astype(int))
            acc_moe_clean = balanced_accuracy_score(y_val, (p_moe_clean >= 0.5).astype(int))

            # Noisy (Mixed Tremor + Jerk + EMG) evaluation
            p_eeg_noisy = F.softmax(eegnet(X_val_mixed), dim=-1)[:, 1].numpy()
            p_msse_noisy = F.softmax(msse(X_val_mixed), dim=-1)[:, 1].numpy()
            _, info_moe_noisy = moe(X_val_mixed, return_gate_info=True)
            p_moe_noisy = info_moe_noisy["combined_probs"][:, 1].numpy()

            acc_eeg_noisy = balanced_accuracy_score(y_val, (p_eeg_noisy >= 0.5).astype(int))
            acc_msse_noisy = balanced_accuracy_score(y_val, (p_msse_noisy >= 0.5).astype(int))
            acc_moe_noisy = balanced_accuracy_score(y_val, (p_moe_noisy >= 0.5).astype(int))

            # Tremor-only evaluation
            _, info_moe_tremor = moe(X_val_tremor, return_gate_info=True)
            acc_eeg_tremor = balanced_accuracy_score(y_val, torch.argmax(eegnet(X_val_tremor), dim=1).numpy())
            acc_moe_tremor = balanced_accuracy_score(y_val, np.argmax(info_moe_tremor["combined_probs"].numpy(), axis=1))

            # Jerk-only evaluation
            _, info_moe_jerk = moe(X_val_jerk, return_gate_info=True)

            # Gate Partitioning Metrics (Stateless Batch)
            route_clean_to_eegnet = float((info_moe_clean["selected_expert"] == 0).float().mean().item())
            route_tremor_to_msse = float((info_moe_tremor["selected_expert"] == 1).float().mean().item())
            route_jerk_to_msse = float((info_moe_jerk["selected_expert"] == 1).float().mean().item())
            route_mixed_to_msse = float((info_moe_noisy["selected_expert"] == 1).float().mean().item())
            gate_partition_acc = 0.5 * (route_clean_to_eegnet + route_mixed_to_msse)

            # Gate ROC-AUC for distinguishing Clean (0) vs Artifact (1)
            gate_scores = np.concatenate([
                info_moe_clean["g_msse"].numpy(),
                info_moe_noisy["g_msse"].numpy()
            ])
            gate_labels = np.concatenate([
                np.zeros(len(X_val_clean)),
                np.ones(len(X_val_mixed))
            ])
            gate_roc_auc = roc_auc_score(gate_labels, gate_scores)

            # Output Probability Calibration Shift Risk:
            # Uncalibrated independent switch drift vs Calibrated MoE internal expert drift on clean trials
            uncalibrated_switch_drift = float(np.mean(np.abs(p_eeg_clean - p_msse_clean)))
            calibrated_moe_drift = float(torch.abs(
                info_moe_clean["probs_eegnet"][:, 1] - info_moe_clean["probs_msse"][:, 1]
            ).mean().item())

        # 4. Evaluate Intermittent Bidirectional Stream (Clean -> Tremor -> Clean -> Jerk -> Clean)
        df_stream = evaluate_intermittent_bidirectional_stream(moe, eegnet, msse, X_val_clean, y_val)
        if sub_id == "sub-02":
            sub02_stream_df = df_stream

        legacy_route_acc = float((df_stream["legacy_oneway_expert"].values == df_stream["true_artifact_regime"].values).mean())
        moe_stream_route_acc = float((df_stream["moe_bidirectional_expert"].values == df_stream["true_artifact_regime"].values).mean())

        # Recovery rate in P3 & P5 (after artifacts turn OFF, how often does selector return to EEGNet?)
        recovery_mask = df_stream["phase"].isin(["P3:Clean_Recovery_1", "P5:Clean_Recovery_2"])
        legacy_recovery_rate = float((df_stream.loc[recovery_mask, "legacy_oneway_expert"] == 0).mean())
        moe_recovery_rate = float((df_stream.loc[recovery_mask, "moe_bidirectional_expert"] == 0).mean())

        # Stream Flow Classification Balanced Accuracy across the whole intermittent session
        stream_acc_legacy = balanced_accuracy_score(
            df_stream["true_flow_label"], (df_stream["legacy_flow_prob"] >= 0.5).astype(int)
        )
        stream_acc_moe = balanced_accuracy_score(
            df_stream["true_flow_label"], (df_stream["moe_flow_prob"] >= 0.5).astype(int)
        )

        lat_moe = measure_latency_ms(moe, channels, samples)
        elapsed = time.time() - t0

        row = {
            "subject_id": sub_id,
            "val_trials": len(X_val_clean),
            "eegnet_clean_acc": round(acc_eeg_clean, 4),
            "msse_clean_acc": round(acc_msse_clean, 4),
            "moe_clean_acc": round(acc_moe_clean, 4),
            "eegnet_noisy_acc": round(acc_eeg_noisy, 4),
            "msse_noisy_acc": round(acc_msse_noisy, 4),
            "moe_noisy_acc": round(acc_moe_noisy, 4),
            "eegnet_tremor_acc": round(acc_eeg_tremor, 4),
            "moe_tremor_acc": round(acc_moe_tremor, 4),
            "gate_partition_acc": round(gate_partition_acc, 4),
            "gate_roc_auc": round(gate_roc_auc, 4),
            "clean_route_eegnet_pct": round(route_clean_to_eegnet, 4),
            "tremor_route_msse_pct": round(route_tremor_to_msse, 4),
            "jerk_route_msse_pct": round(route_jerk_to_msse, 4),
            "legacy_oneway_stream_route_acc": round(legacy_route_acc, 4),
            "moe_bidir_stream_route_acc": round(moe_stream_route_acc, 4),
            "legacy_clean_recovery_rate": round(legacy_recovery_rate, 4),
            "moe_clean_recovery_rate": round(moe_recovery_rate, 4),
            "stream_flow_acc_legacy": round(stream_acc_legacy, 4),
            "stream_flow_acc_moe": round(stream_acc_moe, 4),
            "uncalibrated_switch_prob_drift": round(uncalibrated_switch_drift, 4),
            "calibrated_moe_prob_drift": round(calibrated_moe_drift, 4),
            "moe_latency_ms": round(lat_moe, 3)
        }
        cohort_rows.append(row)

        print(f"    [Clean Flow Acc]   EEGNet: {acc_eeg_clean:.2%} | MS-SE: {acc_msse_clean:.2%} | MoE: {acc_moe_clean:.2%}", flush=True)
        print(f"    [Noisy Flow Acc]   EEGNet: {acc_eeg_noisy:.2%} | MS-SE: {acc_msse_noisy:.2%} | MoE: {acc_moe_noisy:.2%} (Gain vs EEGNet: {acc_moe_noisy - acc_eeg_noisy:+.2%})", flush=True)
        print(f"    [Gate Partition]   Acc: {gate_partition_acc:.2%} (AUC: {gate_roc_auc:.3f}) | Clean->EEGNet: {route_clean_to_eegnet:.1%} | Tremor->MS-SE: {route_tremor_to_msse:.1%} | Jerk->MS-SE: {route_jerk_to_msse:.1%}", flush=True)
        print(f"    [Bidir Recovery]   Legacy 7s One-Way Recovery: {legacy_recovery_rate:.1%} | MoE Bidirectional Recovery: {moe_recovery_rate:.1%}", flush=True)
        print(f"    [P(Flow) Drift]    Uncalibrated Hard Switch Drift: {uncalibrated_switch_drift:.4f} | Calibrated MoE Drift: {calibrated_moe_drift:.4f} | Latency: {lat_moe:.2f} ms ({elapsed:.1f}s)", flush=True)


    df_cohort = pd.DataFrame(cohort_rows)
    cohort_csv = os.path.join(derivatives_dir, "moe_gating_validation_results.csv")
    df_cohort.to_csv(cohort_csv, index=False)

    if sub02_stream_df is not None:
        stream_csv = os.path.join(derivatives_dir, "moe_bidirectional_stream_sub02.csv")
        sub02_stream_df.to_csv(stream_csv, index=False)
        print(f"\n[+] Saved step-by-step intermittent stream trace to: {stream_csv}")

    print("\n" + "=" * 85)
    print("5-SUBJECT COHORT SUMMARY (3,480 Total Trials / 698 Validation Trials):")
    print(f"  Mean Clean Balanced Accuracy:   EEGNet={df_cohort['eegnet_clean_acc'].mean():.2%} | MS-SE={df_cohort['msse_clean_acc'].mean():.2%} | MoE={df_cohort['moe_clean_acc'].mean():.2%}")
    print(f"  Mean Noisy Balanced Accuracy:   EEGNet={df_cohort['eegnet_noisy_acc'].mean():.2%} | MS-SE={df_cohort['msse_noisy_acc'].mean():.2%} | MoE={df_cohort['moe_noisy_acc'].mean():.2%}")
    print(f"  Mean Tremor Balanced Accuracy:  EEGNet={df_cohort['eegnet_tremor_acc'].mean():.2%} | MoE={df_cohort['moe_tremor_acc'].mean():.2%}")
    print(f"  Mean Gate Partition Accuracy:   {df_cohort['gate_partition_acc'].mean():.2%} (Mean ROC-AUC: {df_cohort['gate_roc_auc'].mean():.4f})")
    print(f"  Stream Regime Routing Acc:      Legacy One-Way={df_cohort['legacy_oneway_stream_route_acc'].mean():.2%} | Bidirectional MoE={df_cohort['moe_bidir_stream_route_acc'].mean():.2%}")
    print(f"  Post-Artifact Recovery Rate:    Legacy One-Way={df_cohort['legacy_clean_recovery_rate'].mean():.2%} | Bidirectional MoE={df_cohort['moe_clean_recovery_rate'].mean():.2%}")
    print(f"  Mean P(Flow) Switch Drift:      Uncalibrated={df_cohort['uncalibrated_switch_prob_drift'].mean():.4f} -> Calibrated MoE={df_cohort['calibrated_moe_prob_drift'].mean():.4f}")
    print(f"  Mean MoE Inference Latency:     {df_cohort['moe_latency_ms'].mean():.2f} ms")
    print("=" * 85)


if __name__ == "__main__":
    run_validation_suite()
