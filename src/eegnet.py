import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dWithConstraint(nn.Conv2d):
    """
    Conv2d layer with max-norm constraint on weights (commonly used in EEGNet).
    """
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None and self.training:
            with torch.no_grad():
                norm = self.weight.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=self.max_norm)
                self.weight.copy_(self.weight * (desired / (1e-8 + norm)))
        return super().forward(x)

class LinearWithConstraint(nn.Linear):
    """
    Linear layer with max-norm constraint on weights.
    """
    def __init__(self, *args, max_norm: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None and self.training:
            with torch.no_grad():
                norm = self.weight.norm(2, dim=0, keepdim=True)
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
        Length of temporal kernel in Block 1 (default: 64, half sampling rate).
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
        kernel_length: int = 64,
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

if __name__ == "__main__":
    print("[*] Testing EEGNet PyTorch Model...")
    model = EEGNet(n_classes=2, channels=64, samples=350)
    sample_input = torch.randn(8, 1, 64, 350)
    output = model(sample_input)
    print(f"[+] Model instantiated successfully.")
    print(f"    Input Shape:  {sample_input.shape}")
    print(f"    Output Shape: {output.shape}")
    print(f"    Flatten Size: {model.in_features}")
