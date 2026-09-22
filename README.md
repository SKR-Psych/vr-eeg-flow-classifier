# VR EEG Flow Classifier

A real-time Python/Rust analysis and deep learning classification pipeline designed to detect the psychological **flow state** using a **64-channel Brain Products actiCAP** active EEG system integrated with a **Virtual Reality (VR) headset**.

## Project Purpose & Architecture

The ultimate long-term goal of this project is to create a real-time, closed-loop neuroadaptive VR experience:

1. **EEG Data Collection:** A 64-channel Brain Products actiCAP collects active EEG signals while the user plays a Meta Unity VR game.
2. **Real-time Processing (`goofi-pipe`):** The EEG stream is processed in real-time through `goofi-pipe`, a graphical data flow pipeline for physiological signals.
3. **Deep Learning Classification (EEGNet):** A high-performance PyTorch **EEGNet** model (`sub-<id>_eegnet.pt`) runs as a custom node inside `goofi-pipe`, evaluating event-locked and sliding raw voltage tensors in **< 3 ms** with **91.52% cohort balanced accuracy**.
4. **Bayesian Temporal Smoothing:** A recursive Bayesian smoothing filter stabilizes output probabilities ($P(\text{Flow})$) across windows to prevent erratic VR adjustments.
5. **Firebase Synchronization:** The classified flow metrics are written in real-time to a **Google Firebase Realtime Database**.
6. **Unity VR Adaptation:** The Meta Unity VR game retrieves the real-time flow metrics from Firebase to adjust difficulty, target velocities, and electromyostimulation (EMS) in real-time to keep the player in their optimal "flow channel."

---

## Directory Structure

*   [docs/eegnet_vs_svm_comparison.md](docs/eegnet_vs_svm_comparison.md): Comprehensive benchmark report comparing the upgraded EEGNet against the legacy SVM baseline (+41.74% accuracy gain).
*   [biomarker_plan.md](biomarker_plan.md): Scientific blueprint outlining the specific EEG biomarkers (Theta, SMR, Hemispheric Asymmetry, Shannon Entropy, and Network Connectivity).
*   [technical_roadmap.md](technical_roadmap.md): Software engineering, signal processing, and ML roadmap for system integration.
*   [goofi_integration.md](goofi_integration.md): Architecture and Python code template for packaging the PyTorch EEGNet classifier into `goofi-pipe`.
*   [flow_prediction_evaluation.ipynb](flow_prediction_evaluation.ipynb): Baseline evaluation report and scientific critique of initial sliding-window models.
*   `src/`: Core Python modules:
    *   `loader.py`: Ingestion of BIDS 64-channel EEG raw data (`ds003846`).
    *   `preprocessing.py`: Rank-safe 0.5–45 Hz filtering, 10–20 montage setup, and `mne-icalabel` ICA artifact removal.
    *   `eegnet.py`: PyTorch EEGNet neural network architecture (Lawhern et al., 2018).
    *   `gnn_model.py`: Spatial-temporal Graph Neural Network architecture with PLV adjacency.
    *   `epoch_features.py`: Standardized 500 Hz event-locked trial epoching (-100 ms to +600 ms).
    *   `classifier.py`: Calibrated EEGNet training, validation, and `.pt` checkpoint export.
    *   `train_all_eegnet.py`: Batch training across all dataset participants.
    *   `predict.py`: Simulated real-time streaming pipeline achieving sub-5ms forward pass latency with Bayesian temporal smoothing.
    *   `spatial_filtering.py`: Supervised xDAWN and CSP spatial filter extractors.
*   `data/`: Local storage for dataset files and trained model derivatives (`data/ds003846/derivatives/`).

---

## Getting Started

### 1. Test Dataset & Setup
To develop and validate the pipeline, we test our scripts on the open-source BIDS 64-channel VR Reach-to-Object dataset (`ds003846`) available on OpenNeuro. 

1. Install required python packages:
   ```bash
   pip install openneuro-py mne mne-bids numpy scipy scikit-learn torch torchvision torchaudio mne-icalabel
   ```

2. Train an EEGNet classifier for a participant:
   ```bash
   python src/classifier.py --subject 02 --epochs 60
   ```

3. Run real-time stream simulation:
   ```bash
   python src/predict.py --subject 02 --session EMS --model-path data/ds003846/derivatives/sub-02_eegnet.pt
   ```
