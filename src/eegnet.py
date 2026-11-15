import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dWithConstraint(nn.Conv2d):
    """
    Conv2d layer with max-norm constraint on weights (commonly used in EEGNet).
    Constrains each individual filter's weight vector across kernel dimensions dim=(1, 2, 3).
    """
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None and self.training:
            with torch.no_grad():
                norm = self.weight.norm(2, dim=(1, 2, 3), keepdim=True)
                desired = torch.clamp(norm, max=self.max_norm)
                self.weight.copy_(self.weight * (desired / (1e-8 + norm)))
        return super().forward(x)

class LinearWithConstraint(nn.Linear):
    """
    Linear layer with max-norm constraint on weights.
    Constrains each output unit's incoming weight vector across input dimension dim=1.
    """
    def __init__(self, *args, max_norm: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None and self.training:
            with torch.no_grad():
                norm = self.weight.norm(2, dim=1, keepdim=True)
                desired = torch.clamp(norm, max=self.max_norm)
                self.weight.copy_(self.weight * (desired / (1e-8 + norm)))
        return super().forward(x)

class EEGNet(nn.Module):
    """
    EEGNet Architecture (Lawhern et al., 2018).
    Designed specifically for EEG signals and ERP classification.

    Parameters
    ----------
    n_classes : int, optional
        Number of output target classes (default: 2).
    channels : int, optional
        Number of EEG electrodes/channels (default: 64).
    samples : int, optional
        Number of time points per epoch (default: 350, corresponding to 700ms at 500Hz).
    dropout_rate : float, optional
        Dropout probability (default: 0.25).
    kernel_length : int, optional
        Length of temporal kernel in Block 1 (default: 250, corresponding to half sampling
        rate fs // 2 at 500 Hz to capture oscillations down to 2-4 Hz including frontal Theta).
    F1 : int, optional
        Number of temporal filters (default: 8).
    D : int, optional
        Depth multiplier for spatial filters (default: 2).
    F2 : int, optional
        Number of point-wise filters (default: 16).
    """
    def __init__(
        self,
        n_classes: int = 2,
        channels: int = 64,
        samples: int = 350,
        dropout_rate: float = 0.25,
        kernel_length: int = 250,
        F1: int = 8,
        D: int = 2,
        F2: int = 16
    ):
        super().__init__()
        self.n_classes = n_classes
        self.channels = channels
        self.samples = samples
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # --- BLOCK 1: Temporal Conv + Spatial Depthwise Conv ---
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(F1)

        self.depthwise_conv = Conv2dWithConstraint(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(channels, 1),
            groups=F1,
            bias=False,
            max_norm=1.0
        )
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # --- BLOCK 2: Separable Conv (Depthwise + Pointwise) ---
        self.separable_depthwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False
        )
        self.separable_pointwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Calculate flattened feature size dynamically
        self.in_features = self._get_flatten_size()

        # --- CLASSIFIER HEAD ---
        self.classifier = LinearWithConstraint(
            in_features=self.in_features,
            out_features=n_classes,
            max_norm=0.25
        )

    def _get_flatten_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.channels, self.samples)
            x = self.conv1(dummy)
            x = self.batchnorm1(x)
            x = self.depthwise_conv(x)
            x = self.batchnorm2(x)
            x = F.elu(x)
            x = self.pooling1(x)
            x = self.separable_depthwise(x)
            x = self.separable_pointwise(x)
            x = self.batchnorm3(x)
            x = F.elu(x)
            x = self.pooling2(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, channels, samples) or (batch_size, channels, samples).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch_size, 1, channels, samples)

        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        # Flatten & Classify
        x = x.flatten(start_dim=1)
        logits = self.classifier(x)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the flattened representation before the linear classification head.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        return x.flatten(start_dim=1)



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
    and applies Squeeze-and-Excitation attention to suppress noise (e.g. VR-strap movement artifacts).

    Parameters
    ----------
    n_classes : int, optional
        Number of output target classes (default: 2).
    channels : int, optional
        Number of EEG electrodes/channels (default: 64).
    samples : int, optional
        Number of time points per epoch (default: 350, corresponding to 700ms at 500Hz).
    dropout_rate : float, optional
        Dropout probability (default: 0.25).
    temporal_kernels : tuple, optional
        Lengths of temporal kernels in Block 1 (default: (250, 64, 16) corresponding to
        Theta 4-8Hz, Alpha/SMR 8-13Hz, and Beta/Gamma 13-45Hz at 500 Hz).
    filters_per_branch : int, optional
        Number of temporal filters per multi-scale branch (default: 4, giving 12 filters total).
    D : int, optional
        Depth multiplier for spatial filters (default: 2).
    F2 : int, optional
        Number of point-wise filters (default: 16).
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

        total_F1 = filters_per_branch * len(temporal_kernels)

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

        # Classifier Head (Max-Norm <= 0.25)
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

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the flattened representation before the linear classification head.
        """
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
        return x.flatten(start_dim=1)


def compute_eeg_signal_quality_features(x: torch.Tensor, sfreq: float = 500.0) -> torch.Tensor:
    """
    Computes an 8-dimensional differentiable EEG signal-quality and artifact descriptor vector
    directly from standardized EEG epoch tensors (EEG-only route).

    Designed to detect VR head-strap movement jerks, electrode pops, cranial muscle (EMG) bursts,
    and rhythmic 4-12 Hz Parkinsonian / action tremor bursts without relying on external IMU or game telemetry.

    Parameters
    ----------
    x : torch.Tensor
        EEG input tensor of shape (batch_size, 1, channels, samples) or (batch_size, channels, samples).
    sfreq : float, optional
        Sampling rate in Hz (default: 500.0).

    Returns
    -------
    features : torch.Tensor
        Tensor of shape (batch_size, 8) containing:
        [0] mean_rms_jerk       : Log-scaled RMS 3rd temporal derivative (d^3x/dt^3) across channels
        [1] max_channel_jerk    : Log-scaled max single-channel RMS jerk (captures focal electrode pop / strap slip)
        [2] hf_emg_burst_ratio  : Ratio of 1st-difference high-frequency energy (>20 Hz EMG/noise) to broadband energy
        [3] max_ptp_div_ten     : Max peak-to-peak channel excursion (scaled by 1/10)
        [4] channel_std_cv      : Spatial dispersion (std / mean of channel standard deviations; high when subset of leads is noisy)
        [5] excess_kurtosis     : Log-scaled mean positive excess kurtosis across channels (captures transient spikes)
        [6] tremor_4_12_ratio   : Relative FFT spectral power in the 4-12 Hz tremor/alpha-theta band
        [7] tremor_crest_factor : Peak-to-average spectral ratio inside 4-12 Hz (distinguishes narrowband tremor oscillation from 1/f background)
    """
    if x.dim() == 4:
        sig = x.squeeze(1)  # (B, C, T)
    elif x.dim() == 3:
        sig = x
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")

    eps = 1e-6
    b, c, t = sig.shape

    # 1. Temporal derivatives: velocity (d1), acceleration (d2), jerk (d3)
    d1 = sig[:, :, 1:] - sig[:, :, :-1]
    d2 = d1[:, :, 1:] - d1[:, :, :-1]
    d3 = d2[:, :, 1:] - d2[:, :, :-1]

    ch_rms_jerk = torch.sqrt(torch.mean(d3 ** 2, dim=-1) + eps)  # (B, C)
    mean_rms_jerk = torch.log1p(torch.mean(ch_rms_jerk, dim=-1))
    max_channel_jerk = torch.log1p(torch.max(ch_rms_jerk, dim=-1)[0])

    # 2. High-frequency EMG / line-length burst ratio
    var_sig = torch.var(sig, dim=-1, unbiased=False) + eps  # (B, C)
    var_d1 = torch.var(d1, dim=-1, unbiased=False)
    hf_emg_burst_ratio = torch.mean(var_d1 / (var_sig + eps), dim=-1)

    # 3. Peak-to-peak amplitude excursion & spatial variance dispersion
    ch_ptp = torch.max(sig, dim=-1)[0] - torch.min(sig, dim=-1)[0]  # (B, C)
    max_ptp_div_ten = torch.log1p(torch.max(ch_ptp, dim=-1)[0] / 10.0)

    ch_std = torch.sqrt(var_sig)  # (B, C)
    channel_std_cv = torch.std(ch_std, dim=-1, unbiased=False) / (torch.mean(ch_std, dim=-1) + eps)

    # 4. Excess Kurtosis (4th standardized moment - 3)
    ch_mean = torch.mean(sig, dim=-1, keepdim=True)
    z_sig = (sig - ch_mean) / (ch_std.unsqueeze(-1) + eps)
    kurt = torch.mean(z_sig ** 4, dim=-1) - 3.0  # (B, C)
    excess_kurtosis = torch.log1p(torch.mean(F.relu(kurt), dim=-1))

    # 5. Rhythmic Tremor Band (4 - 12 Hz) Power & Narrowband Crest Factor via rFFT
    fft_mag2 = torch.abs(torch.fft.rfft(sig, dim=-1)) ** 2  # (B, C, F_bins)
    freqs = torch.fft.rfftfreq(t, d=1.0 / sfreq, device=sig.device)
    valid_mask = (freqs >= 1.0) & (freqs <= 45.0)
    tremor_mask = (freqs >= 4.0) & (freqs <= 12.0)

    if valid_mask.any() and tremor_mask.any():
        total_power = torch.sum(fft_mag2[:, :, valid_mask], dim=-1) + eps  # (B, C)
        tremor_bins = fft_mag2[:, :, tremor_mask]  # (B, C, F_tremor)
        tremor_power = torch.sum(tremor_bins, dim=-1)  # (B, C)
        tremor_4_12_ratio = torch.mean(tremor_power / total_power, dim=-1)

        max_tremor_bin = torch.max(tremor_bins, dim=-1)[0]
        mean_tremor_bin = torch.mean(tremor_bins, dim=-1) + eps
        tremor_crest_factor = torch.log1p(torch.max(max_tremor_bin / mean_tremor_bin, dim=-1)[0])
    else:
        tremor_4_12_ratio = torch.zeros(b, device=sig.device, dtype=sig.dtype)
        tremor_crest_factor = torch.zeros(b, device=sig.device, dtype=sig.dtype)

    features = torch.stack([
        mean_rms_jerk,
        max_channel_jerk,
        hf_emg_burst_ratio,
        max_ptp_div_ten,
        channel_std_cv,
        excess_kurtosis,
        tremor_4_12_ratio,
        tremor_crest_factor
    ], dim=-1)
    return features


class EEGGatingNetwork(nn.Module):
    """
    Bidirectional Mixture-of-Experts Gating Network for EEG-only routing between
    Expert 0 (Vanilla EEGNet, clean/high-SNR specialist) and Expert 1 (MS-SE-EEGNet,
    multi-scale + SE channel-attention specialist for movement/tremor/headset artifacts).

    Supports both:
    - Stateless batch evaluation / training on individual epochs (with optional context features)
    - Stateful real-time streaming over a rolling 7-second window (e.g. 35 steps at 200ms step)
      that tracks sustained vs. intermittent artifacts and routes back to EEGNet as soon as
      conditions recover.
    """
    def __init__(
        self,
        signal_feature_dim: int = 8,
        expert_latent_dim: int = 0,
        hidden_dim: int = 16,
        context_window_seconds: float = 7.0,
        step_seconds: float = 0.20,
        ema_decay: float = 0.70
    ):
        super().__init__()
        self.signal_feature_dim = signal_feature_dim
        self.expert_latent_dim = expert_latent_dim
        self.hidden_dim = hidden_dim
        self.context_window_seconds = context_window_seconds
        self.step_seconds = step_seconds
        self.max_buffer_len = max(1, int(round(context_window_seconds / max(step_seconds, 1e-3))))
        self.ema_decay = ema_decay

        # Input to MLP: instantaneous signal features (8) + rolling 7s context signal features (8) + optional latent summary
        total_in_dim = signal_feature_dim * 2 + expert_latent_dim
        self.feature_norm = nn.LayerNorm(total_in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2)  # Logits for [Expert 0: EEGNet, Expert 1: MS-SE-EEGNet]
        )

        # Rolling buffer for stateful real-time streaming (stores recent signal_feature vectors)
        self._rolling_buffer = []
        self._ema_features = None

    def reset_stream_state(self):
        """Resets the 7-second rolling context buffer and EMA state for streaming."""
        self._rolling_buffer = []
        self._ema_features = None

    def forward(
        self,
        sig_features: torch.Tensor,
        context_features: torch.Tensor = None,
        latent_summary: torch.Tensor = None,
        update_state: bool = False,
        temperature: float = 1.0
    ) -> tuple:
        """
        Computes bidirectional routing weights [g_eegnet, g_msse] in [0, 1].

        Parameters
        ----------
        sig_features : torch.Tensor
            Instantaneous EEG signal quality features of shape (B, 8).
        context_features : torch.Tensor, optional
            Rolling 7-second context features of shape (B, 8). If None and update_state=True,
            uses and updates the internal 7-second ring buffer / bidirectional EMA.
            If None and update_state=False, defaults to sig_features.
        latent_summary : torch.Tensor, optional
            Optional detached latent feature summary from experts of shape (B, expert_latent_dim).
        update_state : bool, optional
            If True, updates internal streaming history buffer (used in online inference).
        temperature : float, optional
            Softmax temperature for gate sharpness (default: 1.0).

        Returns
        -------
        gate_weights : torch.Tensor
            Softmax routing probabilities of shape (B, 2) where column 0 = EEGNet, column 1 = MS-SE-EEGNet.
        gate_logits : torch.Tensor
            Raw 2-class gate logits of shape (B, 2).
        """
        if context_features is None:
            if update_state:
                # Update rolling 7-second buffer and bidirectional EMA per step
                det_feat = sig_features.detach()
                self._rolling_buffer.append(det_feat)
                if len(self._rolling_buffer) > self.max_buffer_len:
                    self._rolling_buffer.pop(0)

                if self._ema_features is None or self._ema_features.shape != det_feat.shape:
                    self._ema_features = det_feat.clone()
                else:
                    # Bidirectional EMA: rises with sustained jerk/tremor over 7s window, decays rapidly when clean
                    self._ema_features = self.ema_decay * det_feat + (1.0 - self.ema_decay) * self._ema_features

                # Combine recent 7s buffer mean with fast bidirectional EMA so recovery from tremor is immediate
                buf_mean = torch.stack(self._rolling_buffer, dim=0).mean(dim=0)
                context_features = 0.65 * self._ema_features + 0.35 * buf_mean
            else:
                context_features = sig_features

        inputs = [sig_features, context_features]
        if self.expert_latent_dim > 0:
            if latent_summary is None:
                latent_summary = torch.zeros(
                    sig_features.size(0), self.expert_latent_dim,
                    device=sig_features.device, dtype=sig_features.dtype
                )
            inputs.append(latent_summary)

        gate_in = torch.cat(inputs, dim=-1)
        gate_in = self.feature_norm(gate_in)
        gate_logits = self.mlp(gate_in)
        gate_weights = F.softmax(gate_logits / max(temperature, 1e-4), dim=-1)
        return gate_weights, gate_logits


class EEGMoEClassifier(nn.Module):
    """
    Mixture-of-Experts (MoE) EEG Flow Classifier with Bidirectional Gating.

    Combines:
    - Expert 0: Vanilla EEGNet (default high-SNR ERP specialist)
    - Expert 1: MS-SE-EEGNet (multi-scale temporal + Squeeze-and-Excitation channel-attention
      specialist for VR movement, strap strain, and intermittent tremor)
    - EEGGatingNetwork: Examines incoming EEG signal-quality descriptors and latent norms
      to dynamically route between Expert 0 and Expert 1 in either direction.
    - Per-expert temperature scaling (log_temp_eegnet, log_temp_msse) to ensure calibrated,
      comparable output probabilities so switching experts does not artificially shift P(Flow).
    """
    def __init__(
        self,
        n_classes: int = 2,
        channels: int = 64,
        samples: int = 350,
        dropout_rate: float = 0.25,
        sfreq: float = 500.0,
        routing_mode: str = "soft",
        context_window_seconds: float = 7.0,
        step_seconds: float = 0.20
    ):
        super().__init__()
        self.n_classes = n_classes
        self.channels = channels
        self.samples = samples
        self.sfreq = sfreq
        self.routing_mode = routing_mode  # 'soft' (probabilistic mixture) or 'hard' (argmax expert)

        self.expert_eegnet = EEGNet(
            n_classes=n_classes,
            channels=channels,
            samples=samples,
            dropout_rate=dropout_rate
        )
        self.expert_msse = MS_SE_EEGNet(
            n_classes=n_classes,
            channels=channels,
            samples=samples,
            dropout_rate=dropout_rate
        )

        # Summary of expert latent norms (2 scalars: L2 norm of EEGNet features & MS-SE features)
        self.gate = EEGGatingNetwork(
            signal_feature_dim=8,
            expert_latent_dim=2,
            hidden_dim=16,
            context_window_seconds=context_window_seconds,
            step_seconds=step_seconds
        )

        # Learnable log-temperatures per expert to align logit scales and prevent P(Flow) jumps on switch
        self.log_temp_eegnet = nn.Parameter(torch.zeros(1))
        self.log_temp_msse = nn.Parameter(torch.zeros(1))

        # Running standardization buffers for the 8 signal-quality features (calibrated during training)
        self.register_buffer("gate_feat_mean", torch.zeros(1, 8))
        self.register_buffer("gate_feat_std", torch.ones(1, 8))

    def reset_stream_state(self):
        """Resets the internal 7-second rolling context buffer of the gating network."""
        self.gate.reset_stream_state()

    def forward(
        self,
        x: torch.Tensor,
        context_x: torch.Tensor = None,
        update_gate_state: bool = False,
        return_gate_info: bool = False,
        routing_mode: str = None
    ):
        """
        Forward pass through both EEG experts and the bidirectional gating network.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG tensor of shape (B, 1, C, T) or (B, C, T).
        context_x : torch.Tensor, optional
            Optional 7-second context EEG tensor or precomputed context features.
        update_gate_state : bool, optional
            Whether to update the stateful 7-second rolling context buffer (set True during live streaming).
        return_gate_info : bool, optional
            If False (default), returns combined log-probabilities/logits of shape (B, n_classes)
            for 100% drop-in compatibility with EEGNet and MS_SE_EEGNet.
            If True, returns (combined_logits, gate_info_dict).
        routing_mode : str, optional
            Overrides self.routing_mode ('soft' or 'hard') for this forward pass.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        mode = (routing_mode or self.routing_mode).lower()

        # 1. Extract expert features & calibrated logits
        feat_0 = self.expert_eegnet.extract_features(x)
        feat_1 = self.expert_msse.extract_features(x)

        temp_0 = torch.exp(self.log_temp_eegnet).clamp(min=0.1, max=10.0)
        temp_1 = torch.exp(self.log_temp_msse).clamp(min=0.1, max=10.0)

        logits_0 = self.expert_eegnet.classifier(feat_0) / temp_0
        logits_1 = self.expert_msse.classifier(feat_1) / temp_1

        probs_0 = F.softmax(logits_0, dim=-1)
        probs_1 = F.softmax(logits_1, dim=-1)

        # 2. Compute EEG signal-quality features & standardize
        raw_sig_feats = compute_eeg_signal_quality_features(x, sfreq=self.sfreq)
        norm_sig_feats = (raw_sig_feats - self.gate_feat_mean) / (self.gate_feat_std + 1e-6)

        norm_ctx_feats = None
        if context_x is not None:
            if context_x.dim() == 2 and context_x.size(-1) == 8:
                norm_ctx_feats = (context_x - self.gate_feat_mean) / (self.gate_feat_std + 1e-6)
            else:
                raw_ctx = compute_eeg_signal_quality_features(context_x, sfreq=self.sfreq)
                norm_ctx_feats = (raw_ctx - self.gate_feat_mean) / (self.gate_feat_std + 1e-6)

        # Detached latent norms so gate routing gradients don't destabilize expert feature extractors
        latent_norms = torch.stack([
            feat_0.detach().norm(2, dim=-1) / (feat_0.size(-1) ** 0.5),
            feat_1.detach().norm(2, dim=-1) / (feat_1.size(-1) ** 0.5)
        ], dim=-1)

        # 3. Bidirectional Gate Routing
        gate_weights, gate_logits = self.gate(
            sig_features=norm_sig_feats,
            context_features=norm_ctx_feats,
            latent_summary=latent_norms,
            update_state=update_gate_state
        )

        g0 = gate_weights[:, 0:1]  # Weight for Expert 0 (Vanilla EEGNet)
        g1 = gate_weights[:, 1:2]  # Weight for Expert 1 (MS-SE-EEGNet)

        if mode == "hard":
            hard_g1 = (g1 >= 0.5).float()
            hard_g0 = 1.0 - hard_g1
            combined_probs = hard_g0 * probs_0 + hard_g1 * probs_1
        else:
            combined_probs = g0 * probs_0 + g1 * probs_1

        # Convert calibrated mixture probability back to log-space logits so F.softmax(out, dim=1)
        # and F.cross_entropy(out, y) work identically to standard EEGNet / MS_SE_EEGNet
        combined_logits = torch.log(combined_probs.clamp(min=1e-7, max=1.0 - 1e-7))

        if not return_gate_info:
            return combined_logits

        gate_info = {
            "gate_weights": gate_weights,           # (B, 2): [g_eegnet, g_msse]
            "gate_logits": gate_logits,             # (B, 2)
            "g_eegnet": g0.squeeze(-1),             # (B,)
            "g_msse": g1.squeeze(-1),               # (B,)
            "selected_expert": torch.argmax(gate_weights, dim=-1),  # 0 = EEGNet, 1 = MS_SE_EEGNet
            "probs_eegnet": probs_0,                # (B, 2)
            "probs_msse": probs_1,                  # (B, 2)
            "combined_probs": combined_probs,       # (B, 2)
            "logits_eegnet": logits_0,              # (B, 2)
            "logits_msse": logits_1,                # (B, 2)
            "signal_quality_features": raw_sig_feats  # (B, 8)
        }
        return combined_logits, gate_info



if __name__ == "__main__":
    print("[*] Testing EEGNet and MS-SE-EEGNet PyTorch Models...")

    # Test channels=64 and channels=32 with samples=350, verifying forward & backward pass in .train() mode
    for ModelClass in [EEGNet, MS_SE_EEGNet]:
        name = ModelClass.__name__
        for ch in [64, 32]:
            print(f"\n--- Testing {name} with {ch} channels ---")
            model = ModelClass(n_classes=2, channels=ch, samples=350)
            model.train()
            sample_input = torch.randn(8, 1, ch, 350, requires_grad=True)
            output = model(sample_input)
            loss = output.sum()
            loss.backward()

            print(f"[+] Model instantiated successfully.")
            print(f"    Input Shape:       {sample_input.shape}")
            print(f"    Output Shape:      {output.shape}")
            print(f"    Flatten Size:      {model.in_features}")
            print(f"    Backward Pass:     OK (grad norm: {sample_input.grad.norm().item():.4f})")

    # Verification of Max-Norm Constraints in .train() mode with large weights
    for ModelClass in [EEGNet, MS_SE_EEGNet]:
        name = ModelClass.__name__
        print(f"\n--- Verifying Max-Norm Constraints for {name} with Large Initial Weights ---")
        model = ModelClass(n_classes=2, channels=64, samples=350)
        model.train()
        with torch.no_grad():
            model.depthwise_conv.weight.fill_(10.0)
            model.classifier.weight.fill_(10.0)

        # Trigger forward pass to enforce constraints
        dummy_input = torch.randn(2, 1, 64, 350)
        _ = model(dummy_input)

        conv_norms = model.depthwise_conv.weight.norm(2, dim=(1, 2, 3))
        lin_norms = model.classifier.weight.norm(2, dim=1)

        max_conv_norm = conv_norms.max().item()
        max_lin_norm = lin_norms.max().item()
        print(f"    Max Conv2d Filter Norm:  {max_conv_norm:.4f} (Constraint: <= 1.0)")
        print(f"    Max Linear Class Norm:   {max_lin_norm:.4f} (Constraint: <= 0.25)")

        assert max_conv_norm <= 1.0 + 1e-5, f"{name} Conv2d constraint failed: {max_conv_norm}"
        assert max_lin_norm <= 0.25 + 1e-5, f"{name} Linear constraint failed: {max_lin_norm}"
    print("\n[+] All constraint and architectural verifications passed successfully!")

