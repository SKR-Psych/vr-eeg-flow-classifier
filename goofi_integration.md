# Goofi-Pipe Integration Documentation: Real-Time EEGNet Node

This document provides a blueprint and code template for packaging our calibrated **EEGNet** flow classifier as a custom high-speed node inside the **goofi-pipe** framework.

Goofi-pipe is a graphical user interface and programming framework for real-time biosignal processing. Custom nodes inherit from a base `Node` class and define input/output slots, configurable parameters, and processing logic.

---

## 1. Custom Node Architecture (EEGNet Deep Learning)

Our upgraded custom classification node, `EEGNetFlowClassifier`, is designed to:
1. **Ingest Multi-Channel EEG Buffers**: Subscribes directly to an EEG buffer array `(channels, samples)` spanning a $700\text{ ms}$ window ($350$ time samples at $500\text{ Hz}$).
2. **Dynamically Load PyTorch Model**: Reads a file path parameter pointing to a calibrated `.pt` checkpoint (containing model weights, channel count, and train-set normalization parameters `norm_mean` and `norm_std`).
3. **Sub-3ms Neural Network Inference**: Applies z-score standardization and performs a forward pass through PyTorch EEGNet to obtain softmax state probabilities.
4. **Bayesian Temporal Smoothing**: Smoothes raw probabilities across successive windows ($P_t = \alpha P_{\text{raw}} + (1-\alpha) P_{t-1}$) to prevent erratic VR adaptations caused by single noisy intervals.
5. **Stream Outputs**: Publishes the smoothed flow probability ($0.0$ to $1.0$) and discrete state label (`Flow` or `Disrupted`) to downstream game engines or Firebase synchronizers.

---

## 2. Python Code Template

Save this code inside `goofi/nodes/analysis/eegnet_flowclassifier.py` in your local goofi-pipe repository directory to register it.

```python
import os
import numpy as np
import torch
from goofi.data import DataType
from goofi.node import Node

# Import EEGNet architecture from repository
from src.eegnet import EEGNet
from src.classifier import load_eegnet_checkpoint

class EEGNetFlowClassifier(Node):
    @staticmethod
    def config_input_slots():
        # Ingests a 2D EEG buffer array of shape (channels, samples)
        return {
            "eeg_buffer": DataType.ARRAY
        }

    @staticmethod
    def config_output_slots():
        # Outputs flow probability array and discrete state string
        return {
            "flow_probability": DataType.ARRAY,
            "flow_state": DataType.STRING
        }

    @staticmethod
    def config_params():
        # Defines user-configurable GUI parameters in the editor
        return {
            "model": {
                "model_path": "data/ds003846/derivatives/sub-02_eegnet.pt",
                "smoothing_alpha": 0.6,
                "device": "cpu"
            }
        }

    def setup(self):
        """Called when the node is initialized in the pipeline."""
        self.model = None
        self.norm_mean = None
        self.norm_std = None
        self.loaded_model_path = None
        self.smoothed_prob = 0.5
        self.device = torch.device('cpu')

    def process(self, eeg_buffer):
        """
        Executes on every pipeline tick when buffer data is received.
        """
        if eeg_buffer is None or eeg_buffer.data is None:
            return None

        # 1. Retrieve GUI parameters
        model_path = self.params["model"]["model_path"].value
        alpha = float(self.params["model"]["smoothing_alpha"].value)
        device_str = self.params["model"]["device"].value
        device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')

        # 2. Dynamically load/reload PyTorch model if checkpoint path changed
        if self.model is None or self.loaded_model_path != model_path or self.device != device:
            try:
                self.model, self.norm_mean, self.norm_std, meta = load_eegnet_checkpoint(model_path, device=device)
                self.loaded_model_path = model_path
                self.device = device
                print(f"[EEGNetFlowClassifier] Dynamically loaded EEGNet model from: {model_path}")
            except Exception as e:
                print(f"[EEGNetFlowClassifier] Failed to load model from '{model_path}': {e}")
                return None

        # 3. Format input buffer into 4D tensor (1, 1, channels, samples)
        raw_arr = eeg_buffer.data
        if not isinstance(raw_arr, np.ndarray):
            raw_arr = np.array(raw_arr)

        # Ensure correct orientation (channels, samples)
        expected_channels = self.norm_mean.shape[2]
        expected_samples = self.norm_mean.shape[3]

        if raw_arr.shape[0] != expected_channels and raw_arr.shape[1] == expected_channels:
            raw_arr = raw_arr.T

        # Crop or pad to expected sample length (e.g. 350 samples = 700ms at 500Hz)
        if raw_arr.shape[1] > expected_samples:
            raw_arr = raw_arr[:, -expected_samples:]
        elif raw_arr.shape[1] < expected_samples:
            # Pad with zeros if buffer hasn't fully filled yet
            pad_width = expected_samples - raw_arr.shape[1]
            raw_arr = np.pad(raw_arr, ((0, 0), (pad_width, 0)), mode='edge')

        # 4. Neural Network Inference (< 3 ms)
        try:
            tensor = torch.tensor(raw_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            # Apply training z-score standardization
            tensor = (tensor - self.norm_mean) / self.norm_std

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                raw_flow_prob = float(probs[1])

            # 5. Bayesian Temporal Smoothing
            self.smoothed_prob = alpha * raw_flow_prob + (1.0 - alpha) * self.smoothed_prob
            state_label = "Flow" if self.smoothed_prob >= 0.5 else "Disrupted"

        except Exception as e:
            print(f"[EEGNetFlowClassifier] Inference error: {e}")
            return None

        # 6. Return values packaged for goofi-pipe output ports
        return {
            "flow_probability": (np.array([self.smoothed_prob]), {}),
            "flow_state": (state_label, {})
        }
```

---

## 3. Node Integration Guide

### Category Placement
Place the node script file in `goofi/nodes/analysis/` in your local `goofi-pipe` setup. Goofi-pipe automatically scans this directory and adds the class to the node menu.

### Wiring Inputs in the GUI
1. Open the goofi-pipe editor by running `goofi-pipe` in your terminal.
2. Open the node creation menu (press `Tab` or double-click the background).
3. Search for and create the `EEGNetFlowClassifier` node (under the `analysis` category).
4. Connect the output of a multi-channel LSL stream buffer (700 ms window) into the `eeg_buffer` input port.
5. Connect `flow_probability` and `flow_state` outputs to a downstream Google Firebase sync node or Unity VR socket sender.
