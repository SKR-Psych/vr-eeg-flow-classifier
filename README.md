# VR EEG Flow Classifier

A real-time Python/Rust analysis and deep learning classification pipeline designed to detect the psychological **flow state** using a **64-channel Brain Products actiCAP** active EEG system integrated with a **Virtual Reality (VR) headset**.

## Project Purpose & Architecture

The ultimate long-term goal of this project is to create a real-time, closed-loop neuroadaptive VR experience:

1. **EEG Data Collection:** A 64-channel Brain Products actiCAP collects active EEG signals while the user plays a Meta Unity VR game.
2. **Real-time Processing (`goofi-pipe`):** The EEG stream is processed in real-time through `goofi-pipe`, a graphical data flow pipeline for physiological signals.
3. **Deep Learning Classification (EEGNet / MS-SE-EEGNet / Bidirectional MoE):** High-performance PyTorch classifiers (`sub-<id>_eegnet.pt`, `sub-<id>_ms_se_eegnet.pt`, or `sub-<id>_moe_eegnet.pt`) run as custom nodes inside `goofi-pipe`, evaluating event-locked and sliding raw voltage tensors in **< 6 ms** with **91.52% cohort balanced accuracy** and dynamic bidirectional Mixture-of-Experts routing for tremor and VR movement artifacts.
4. **Bayesian Temporal Smoothing:** A recursive Bayesian smoothing filter stabilizes output probabilities ($P(\text{Flow})$) across windows to prevent erratic VR adjustments.
5. **Firebase Synchronization:** The classified flow metrics are written in real-time to a **Google Firebase Realtime Database**.
6. **Unity VR Adaptation:** The Meta Unity VR game retrieves the real-time flow metrics from Firebase to adjust difficulty, target velocities, and electromyostimulation (EMS) in real-time to keep the player in their optimal "flow channel."

---

## Directory Structure

*   [docs/preprocessing_and_flow_detection_walkthrough.md](docs/preprocessing_and_flow_detection_walkthrough.md): Dedicated walkthrough of the 6-stage EEG pre-processing and flow state detection pipeline.
*   [docs/eegnet_architecture_summary.md](docs/eegnet_architecture_summary.md): Architectural breakdown and neurophysiological rationale for EEGNet.
*   [docs/eegnet_vs_svm_comparison.md](docs/eegnet_vs_svm_comparison.md): Comprehensive benchmark report comparing upgraded EEGNet against the legacy SVM baseline (+41.74% accuracy gain).
*   [docs/ms_se_eegnet_vs_eegnet_comparison.md](docs/ms_se_eegnet_vs_eegnet_comparison.md): Evaluation and cohort benchmark comparing Multi-Scale Squeeze-and-Excitation EEGNet (`MS_SE_EEGNet`) against Vanilla `EEGNet`.
*   [docs/moe_gating_architecture_report.md](docs/moe_gating_architecture_report.md): Bidirectional Mixture-of-Experts (`EEGMoEClassifier`) gating architecture and 5-subject validation report under tremor and VR movement artifacts.
*   [docs/shapley_biomarker_research_report.md](docs/shapley_biomarker_research_report.md): Shapley feature attribution and multi-biomarker discovery report.
*   [biomarker_plan.md](biomarker_plan.md): Scientific blueprint outlining the specific EEG biomarkers (Theta, SMR, Hemispheric Asymmetry, Shannon Entropy, and Network Connectivity).
*   [technical_roadmap.md](technical_roadmap.md): Software engineering, signal processing, and ML roadmap for system integration.
*   [goofi_integration.md](goofi_integration.md): Architecture and Python code template for packaging the PyTorch EEGNet / MoE classifier into `goofi-pipe`.
*   [flow_prediction_evaluation.ipynb](flow_prediction_evaluation.ipynb) & [flow_prediction_evaluation_phase3.ipynb](flow_prediction_evaluation_phase3.ipynb): Evaluation notebooks tracking progression from sliding-window baselines to Phase 3 deep learning.
*   `src/`: Core Python modules:
    *   `loader.py`: Ingestion of BIDS 64-channel EEG raw data (`ds003846`).
    *   `preprocessing.py`: Rank-safe 0.5–45 Hz filtering, 10–20 montage setup, and `mne-icalabel` ICA artifact removal.
    *   `eegnet.py`: PyTorch `EEGNet`, `MS_SE_EEGNet`, and Bidirectional Mixture-of-Experts (`EEGMoEClassifier` / `EEGGatingNetwork`) architectures.
    *   `gnn_model.py`: Spatial-temporal Graph Neural Network architecture with PLV adjacency.
    *   `epoch_features.py`: Standardized 500 Hz event-locked trial epoching (-100 ms to +600 ms).
    *   `classifier.py`: Calibrated `EEGNet`, `MS_SE_EEGNet`, and `EEGMoEClassifier` training, validation, synthetic artifact generation, and `.pt` checkpoint export.
    *   `train_all_eegnet.py`: Batch training across all dataset participants.
    *   `predict.py`: Simulated real-time streaming pipeline with bidirectional MoE routing telemetry and Bayesian temporal smoothing.
    *   `spatial_filtering.py`: Supervised xDAWN and CSP spatial filter extractors.
*   `tests/`: Verification and benchmark suites:
    *   `test_eegnet.py`: Unit tests for `EEGNet`, `MS_SE_EEGNet`, `EEGMoEClassifier`, max-norm constraints, and checkpoint roundtrips.
    *   `benchmark_ms_se_eegnet.py` & `cohort_benchmark_ms_se_vs_eegnet.py`: Single-subject and multi-subject head-to-head benchmarks.
    *   `validate_moe_gating.py`: 5-subject validation suite testing bidirectional MoE routing across clean, Parkinsonian tremor, and VR strap jerk regimes.
*   `data/`: Local storage for dataset files and trained model derivatives (`data/ds003846/derivatives/`).

---

## Getting Started

### 1. Test Dataset & Setup
To develop and validate the pipeline, we test our scripts on the open-source BIDS 64-channel VR Reach-to-Object dataset (`ds003846`) available on OpenNeuro. 

1. Install required python packages:
   ```bash
   pip install openneuro-py mne mne-bids numpy scipy scikit-learn torch torchvision torchaudio mne-icalabel
   ```

2. Train a classifier for a participant (`eegnet`, `ms_se_eegnet`, or `moe`):
   ```bash
   python src/classifier.py --subject 02 --epochs 60 --model-type moe
   ```

3. Run real-time stream simulation:
   ```bash
   python src/predict.py --subject 02 --session EMS --model-path data/ds003846/derivatives/sub-02_eegnet.pt
   ```

