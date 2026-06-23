# Goofi-Pipe Integration Documentation

This document provides a blueprint and code template for packaging our calibrated EEG flow classifier as a custom node inside the **goofi-pipe** framework. 

Goofi-pipe is a graphical user interface and programming framework for real-time biosignal processing. Custom nodes inherit from a base `Node` class and define input/output slots, configurable parameters, and processing logic.

---

## 1. Custom Node Architecture

Our custom classification node, `FlowClassifier`, is designed to:
1. **Ingest Streaming Features**: Subscribes to a real-time `Table` (a dictionary of data arrays) containing the 14 computed EEG features.
2. **Dynamically Load Classifier**: Reads a file path parameter pointing to a calibrated `.joblib` model. If the path changes or the node starts up, it dynamically loads/reloads the model in memory.
3. **Classify Real-Time Windows**: Evaluates incoming feature matrices and handles probability calculations for both Random Forest (via `predict_proba`) and SVM models (via sigmoid-mapped `decision_function`).
4. **Stream Outputs**: Publishes the predicted flow probability (0.0 to 1.0) and the text label ("Flow" or "Disrupted") to downstream nodes (e.g., game engines, visualizers, or audio synthesizers).

---

## 2. Python Code Template

Save this code inside `goofi/nodes/analysis/flowclassifier.py` in your local goofi-pipe repository directory to register it.

```python
import joblib
import numpy as np
import pandas as pd
from goofi.data import DataType
from goofi.node import Node

class FlowClassifier(Node):
    @staticmethod
    def config_input_slots():
        # Ingests a table (dictionary of feature streams)
        return {
            "features": DataType.TABLE
        }

    @staticmethod
    def config_output_slots():
        # Outputs a float array for probability and a string for state label
        return {
            "flow_probability": DataType.ARRAY,
            "flow_state": DataType.STRING
        }

    @staticmethod
    def config_params():
        # Defines user-configurable GUI parameters in the editor
        return {
            "model": {
                "model_path": "data/ds003846/derivatives/sub-02_ses-EMS_classifier.joblib",
            }
        }

    def setup(self):
        """Called when the node is initialized in the pipeline."""
        self.clf = None
        self.loaded_model_path = None

    def process(self, features):
        """
        Executes on every pipeline tick when input data is received.
        """
        if features is None or features.data is None:
            return None

        # 1. Retrieve the model path parameter from GUI
        model_path = self.params["model"]["model_path"].value

        # 2. Dynamically load/reload model if needed
        if self.clf is None or self.loaded_model_path != model_path:
            try:
                self.clf = joblib.load(model_path)
                self.loaded_model_path = model_path
                print(f"[FlowClassifier] Dynamically loaded model from: {model_path}")
            except Exception as e:
                print(f"[FlowClassifier] Failed to load model from '{model_path}': {e}")
                return None

        # 3. List of features in the exact training/inference order
        feature_cols = [
            'fm_theta', 'smr_alpha_c3', 'smr_beta_c3', 'smr_alpha_c4', 'smr_beta_c4',
            'smr_alpha_cp3', 'smr_beta_cp3', 'smr_alpha_cp4', 'smr_beta_cp4',
            'beta_asymmetry', 'entropy_af7', 'entropy_af8', 'plv_theta', 'plv_alpha'
        ]

        # 4. Extract incoming features from goofi-pipe Table
        data_dict = {}
        for col in feature_cols:
            if col in features.data:
                # If feature is a streaming array, extract the latest value
                val_data = features.data[col].data
                if isinstance(val_data, np.ndarray):
                    val = val_data[-1] if val_data.size > 0 else 0.0
                else:
                    val = val_data
                data_dict[col] = float(val)
            else:
                # Fallback default value if feature is missing
                data_dict[col] = 0.0

        # Construct single-row pandas DataFrame
        df_features = pd.DataFrame([data_dict], columns=feature_cols)

        # 5. Run classification inference
        try:
            label_pred = self.clf.predict(df_features)[0]
            
            # Extract probability (supporting Random Forest and SVM decision boundaries)
            if hasattr(self.clf, 'predict_proba'):
                prob = self.clf.predict_proba(df_features)[0]
                flow_prob = prob[1]  # Index 1 corresponds to Flow (Class 1)
            elif hasattr(self.clf, 'decision_function'):
                # Map decision function score to [0, 1] range using Sigmoid function
                df_val = self.clf.decision_function(df_features)[0]
                flow_prob = 1.0 / (1.0 + np.exp(-df_val))
            else:
                flow_prob = 1.0 if label_pred == 1 else 0.0
                
            state_label = "Flow" if label_pred == 1 else "Disrupted"
            
        except Exception as e:
            print(f"[FlowClassifier] Inference error: {e}")
            return None

        # 6. Return values packaged for goofi-pipe output ports
        return {
            "flow_probability": (np.array([flow_prob]), {}),
            "flow_state": (state_label, {})
        }
```

---

## 3. Node Integration Guide

### Category Placement
Place the node script file in the `goofi/nodes/analysis/` directory. Goofi-pipe automatically scans directories under `goofi/nodes/` and adds classes that inherit from `Node` to the node selection menu.

### Wireing Inputs in the GUI
1. Open the goofi-pipe editor by running `goofi-pipe` in your terminal.
2. Open the node creation menu (press `Tab` or double-click the background).
3. Search for and create the `FlowClassifier` node (under the `analysis` category).
4. Connect the output of a streaming feature extraction node (which outputs a `Table` of relative powers, entropy, asymmetry, and PLV) into the `features` input port of the `FlowClassifier`.
5. Connect the `flow_probability` or `flow_state` outputs to a downstream data visualizer, OSC/LSL sender, or game engine connector.

### Model Recalibration Flow (Real-World Pipeline)
* During baseline calibration, the user runs the baseline training script which outputs a trained `.joblib` model (e.g. `user_classifier.joblib`).
* Update the `model_path` parameter in the `FlowClassifier` GUI node properties window. The node will immediately load the newly trained model in the background and continue real-time processing without requiring a pipeline restart.
