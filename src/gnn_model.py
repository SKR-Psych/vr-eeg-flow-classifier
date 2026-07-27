import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
from scipy.signal import hilbert

def compute_plv_matrix(epochs: mne.Epochs, fmin: float = 4.0, fmax: float = 12.0) -> np.ndarray:
    """
    Computes Phase Locking Value (PLV) adjacency matrix across channels 
    in the specified frequency range (Theta + Alpha) using fast vectorized operations.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object.
    fmin : float
        Min frequency (Hz). Default: 4.0.
    fmax : float
        Max frequency (Hz). Default: 12.0.

    Returns
    -------
    plv_matrix : np.ndarray
        Symmetric 2D array of shape (n_channels, n_channels) with PLV values in [0, 1].
    """
    # Filter epochs to target band
    filtered = epochs.copy().filter(l_freq=fmin, h_freq=fmax, method='iir', verbose=False)
    data = filtered.get_data(units='uV')  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape

    # Analytic signal via Hilbert transform across time
    analytic = hilbert(data, axis=-1)
    phase = np.angle(analytic)  # (n_epochs, n_channels, n_times)

    # Vectorized PLV computation via complex matrix multiplication
    # Reshape phase matrix to (n_channels, n_epochs * n_times)
    phase_flat = phase.transpose(1, 0, 2).reshape(n_channels, -1)
    complex_phase = np.exp(1j * phase_flat)  # (n_channels, total_time_points)

    # Inner product across time points normalized by total time length
    plv = np.abs(complex_phase @ complex_phase.conj().T) / phase_flat.shape[1]
    np.fill_diagonal(plv, 1.0)

    return plv

def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Symmetrically normalizes adjacency matrix: D^(-1/2) * (A + I) * D^(-1/2)
    """
    adj_self = adj + torch.eye(adj.size(0), device=adj.device)
    deg = torch.sum(adj_self, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    deg_mat = torch.diag(deg_inv_sqrt)
    return deg_mat @ adj_self @ deg_mat

class GraphConvLayer(nn.Module):
    """
    Spectral Graph Convolution Layer: H' = ELU( D^(-1/2) A D^(-1/2) H W )
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (batch_size, n_nodes, in_features).
        norm_adj : torch.Tensor
            Normalized adjacency matrix of shape (n_nodes, n_nodes).
        """
        support = torch.matmul(x, self.weight)
        output = torch.matmul(norm_adj, support) + self.bias
        return F.elu(output)

class EEGGraphNet(nn.Module):
    """
    Graph Neural Network for EEG ERP classification.

    Parameters
    ----------
    n_channels : int
        Number of EEG electrodes / graph nodes (default: 64).
    in_node_features : int
        Number of feature dimensions per node (default: 8 summary features).
    hidden_dim : int
        Hidden representation size for graph convolution layers (default: 32).
    n_classes : int
        Number of output classes (default: 2).
    dropout : float
        Dropout probability (default: 0.3).
    """
    def __init__(
        self,
        n_channels: int = 64,
        in_node_features: int = 8,
        hidden_dim: int = 32,
        n_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.n_channels = n_channels

        self.gconv1 = GraphConvLayer(in_node_features, hidden_dim)
        self.gconv2 = GraphConvLayer(hidden_dim, hidden_dim * 2)

        self.dropout = nn.Dropout(dropout)

        # Global Readout (Mean + Max pooling over nodes)
        self.fc1 = nn.Linear(hidden_dim * 2 * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        h = self.gconv1(x, norm_adj)
        h = self.dropout(h)
        h = self.gconv2(h, norm_adj)

        h_mean = torch.mean(h, dim=1)
        h_max = torch.max(h, dim=1)[0]
        h_graph = torch.cat([h_mean, h_max], dim=1)

        out = F.elu(self.fc1(h_graph))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

def extract_node_features(epochs: mne.Epochs) -> torch.Tensor:
    """
    Extracts node-level spatio-temporal features for each channel.
    """
    data = epochs.get_data(units='uV')  # (n_epochs, n_channels, n_times)
    times = epochs.times
    n_epochs, n_channels, n_times = data.shape

    pen_mask = (times >= 0.15) & (times <= 0.35)
    p300_mask = (times >= 0.30) & (times <= 0.50)

    data_theta = epochs.copy().filter(4.0, 8.0, method='iir', verbose=False).get_data(units='uV')
    data_alpha = epochs.copy().filter(8.0, 12.0, method='iir', verbose=False).get_data(units='uV')
    data_beta = epochs.copy().filter(12.0, 30.0, method='iir', verbose=False).get_data(units='uV')

    nodes = np.zeros((n_epochs, n_channels, 8))

    for i in range(n_epochs):
        trial = data[i]
        nodes[i, :, 0] = np.mean(trial[:, pen_mask], axis=1)
        nodes[i, :, 1] = np.min(trial[:, pen_mask], axis=1)
        nodes[i, :, 2] = np.mean(trial[:, p300_mask], axis=1)
        nodes[i, :, 3] = np.max(trial[:, p300_mask], axis=1)
        nodes[i, :, 4] = np.std(trial, axis=1)
        nodes[i, :, 5] = np.var(data_theta[i], axis=1)
        nodes[i, :, 6] = np.var(data_alpha[i], axis=1)
        nodes[i, :, 7] = np.var(data_beta[i], axis=1)

    return torch.tensor(nodes, dtype=torch.float32)

if __name__ == "__main__":
    print("[*] Testing Vectorized EEGGraphNet Model...")
    model = EEGGraphNet(n_channels=64, in_node_features=8, hidden_dim=32, n_classes=2)
    sample_nodes = torch.randn(8, 64, 8)
    sample_adj = torch.rand(64, 64)
    sample_adj = (sample_adj + sample_adj.T) / 2.0
    norm_adj = normalize_adjacency(sample_adj)
    
    out = model(sample_nodes, norm_adj)
    print(f"[+] Vectorized EEGGraphNet instantiated successfully. Output shape: {out.shape}")
