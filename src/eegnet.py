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

