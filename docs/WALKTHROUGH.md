# EEGNet vs. Legacy SVM: Performance Walkthrough & Architectural Comparison

**Project:** VR-EEG Flow State Classifier  
**Dataset:** OpenNeuro `ds003846` (64-Channel actiCAP EEG, VR Reach-to-Touch)  
**Document Status:** Formal Validation & Upgrade Walkthrough  

---

## 1. Executive Summary

This walkthrough captures the comparative evaluation between the **Legacy Baseline Classifiers** (Support Vector Machines [SVM] & Random Forest with continuous 2.0s sliding-window features) and the upgraded **EEGNet Deep Learning Architecture** (event-locked spatio-temporal convolutions).

### Key Takeaways:
1. **Dramatic Accuracy Gain:** Moving from the legacy SVM to EEGNet elevated mean balanced accuracy across the participant cohort from **49.78% (chance level) to 91.52%**, representing an average net improvement of **+41.74 percentage points**.
2. **Elimination of Prediction Error Dilution:** The legacy SVM operated on wide 2.0-second sliding windows, which completely diluted the sub-second (~300 ms) Prediction Error Negativity (PEN/N200) into identical reaching background noise. EEGNet processes event-locked 700 ms trials ($-100\text{ ms}$ to $+600\text{ ms}$) centered directly on the contact event (`box:touched`).
3. **Continuous Phase Preservation:** While the legacy pipeline collapsed 64 channels into 14 scalar summary statistics, EEGNet applies 2D depthwise spatial and separable temporal convolutions directly on raw microvolt dynamics, preserving cross-electrode phase topologies.
4. **Latency Reduction for Closed-Loop VR:** Feature extraction for the legacy SVM required ~**82.1 ms** (due to Welch PSD, Hilbert transforms, and PLV calculations). A forward pass of EEGNet runs in **< 2.5 ms**, well within the 100 ms real-time budget for neuroadaptive VR mini-games.

---

## 2. Cohort-Wide Benchmark Comparison

The following table summarizes performance across all benchmarked participants using chronological 80/20 train/test splits:

| Metric | Legacy SVM Baseline | Legacy RF Baseline | Upgraded EEGNet | Net Gain (EEGNet vs. SVM) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Balanced Accuracy** | **49.78%** | **50.48%** | **91.52%** | **+41.74%** |
| **Mean Precision** | 0.812 | 0.806 | **0.970** | **+15.8%** |
| **Mean Recall (Sensitivity)**| 0.638 | 0.923 | **0.923** | **+28.5%** |
| **Mean F1-Score** | 0.709 | 0.854 | **0.945** | **+23.6%** |
| **Mean ROC-AUC** | ~0.501 | ~0.505 | **0.963** | **+46.2%** |
| **Feature Extraction / Prep** | 82.11 ms | 82.11 ms | **< 1.0 ms** | **~80 ms faster** |
| **Model Inference Latency** | 0.25 ms | 1.10 ms | **2.20 ms** | **Real-Time Validated** |
| **Total Pipeline Latency** | ~82.36 ms | ~83.21 ms | **< 3.2 ms** | **~25x speedup** |

---

## 3. Subject-by-Subject Performance Breakdown

Data sourced from [`data/ds003846/derivatives/eegnet_vs_svm_comparison.csv`](../data/ds003846/derivatives/eegnet_vs_svm_comparison.csv):

| Participant | Legacy SVM Balanced Acc | Legacy Best Baseline | Upgraded EEGNet Balanced Acc | EEGNet Precision | EEGNet F1-Score | EEGNet ROC-AUC | Net Gain over SVM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `sub-02` | 53.24% | 53.24% (SVM) | **85.56%** | 0.926 | 0.930 | 0.949 | **+32.32%** |
| `sub-03` | 57.50% | 58.35% (RF)  | **77.78%** | 0.949 | 0.881 | 0.824 | **+20.28%** |
| `sub-04` | 53.02% | 53.02% (SVM) | **93.19%** | 0.989 | 0.939 | 0.985 | **+40.17%** |
| `sub-05` | 52.39% | 52.39% (SVM) | **86.87%** | 0.948 | 0.911 | 0.911 | **+34.47%** |
| `sub-06` | 54.09% | 54.09% (SVM) | **92.96%** | 0.970 | 0.959 | 0.958 | **+38.88%** |
| `sub-07` | 48.06% | 51.66% (RF)  | **88.71%** | 0.947 | 0.937 | 0.936 | **+40.66%** |
| `sub-08` | 50.60% | 53.52% (RF)  | **96.64%** | 1.000 | 0.965 | 0.991 | **+46.04%** |
| `sub-10` | 42.74% | 48.29% (RF)  | **96.56%** | 0.980 | 0.985 | 0.979 | **+53.83%** |
| `sub-11` | 48.88% | 51.36% (RF)  | **98.18%** | 1.000 | 0.981 | 0.990 | **+49.30%** |
| `sub-12` | 44.13% | 49.80% (RF)  | **92.45%** | 0.969 | 0.954 | 0.970 | **+48.32%** |
| `sub-13` | 35.29% | 41.76% (RF)  | **97.01%** | 0.992 | 0.977 | 0.990 | **+61.72%** |
| `sub-14` | 53.86% | 53.86% (SVM) | **82.65%** | 0.917 | 0.910 | 0.906 | **+28.79%** |
| `sub-15` | 48.04% | 49.87% (RF)  | **85.85%** | 0.964 | 0.878 | 0.896 | **+37.82%** |
| `sub-16` | 49.99% | 50.92% (RF)  | **95.23%** | 0.985 | 0.966 | 0.978 | **+45.24%** |
| `sub-17` | 49.33% | 51.07% (RF)  | **97.57%** | 1.000 | 0.975 | 0.993 | **+48.24%** |
| `sub-18` | 50.28% | 50.28% (SVM) | **91.90%** | 0.953 | 0.967 | 0.939 | **+41.61%** |
| `sub-19` | 50.58% | 50.58% (SVM) | **90.71%** | 0.968 | 0.933 | 0.951 | **+40.13%** |
| `sub-20` | 54.11% | 54.11% (SVM) | **97.62%** | 0.990 | 0.986 | 0.986 | **+43.51%** |
| **Cohort Mean** | **49.78%** | **51.57%** | **91.52%** | **0.970** | **0.945** | **0.963** | **+41.74%** |

---

## 4. Why EEGNet Outperforms the Legacy SVM

```mermaid
graph TD
    subgraph Legacy SVM Paradigm [Legacy SVM Paradigm - 49.78% Accuracy]
        A1[Raw Continuous EEG] --> B1[Wide 2.0s Sliding Window]
        B1 --> C1[Dilutes 300ms Prediction Error Waveform]
        C1 --> D1[Compute 14 Scalar Summary Statistics]
        D1 --> E1[SVM RBF Kernel]
        E1 --> F1[Near-Chance Predictions]
    end

    subgraph Upgraded EEGNet Paradigm [Upgraded EEGNet Paradigm - 91.52% Accuracy]
        A2[Raw Continuous EEG] --> B2[Event-Locked Epoching: -100ms to +600ms]
        B2 --> C2[Isolates Clean PEN/N200 & P300 Dynamics]
        C2 --> D2[Direct 4D Voltage Tensor: N x 1 x 64 x 350]
        D2 --> E2[Block 1: Temporal Conv + Depthwise Spatial Conv]
        E2 --> F2[Block 2: Separable Spatial-Temporal Conv]
        F2 --> G2[High-Accuracy Flow Discrimination]
    end
```

### 1. Temporal Resolution & Signal Dilution
In the reach-to-touch VR task, reaching toward a normal box versus reaching toward a conflict/mismatch box produces indistinguishable cognitive and motor background states. The only neurophysiological discrepancy occurs in a transient **$150\text{ to }350\text{ ms}$** post-contact window (Prediction Error Negativity, PEN). 
* **Legacy SVM:** Averaging power spectra over a 2.0-second window buried this 200 ms transient in 1,800 ms of identical motor reaching noise.
* **EEGNet:** Time-locked segmentation centered on `box:touched` guarantees the receptive field focuses precisely on the evoked potential.

### 2. Spatio-Temporal Convolutions vs. Hand-Crafted Scalar Features
The legacy SVM discarded phase topography by computing summary statistics (Welch PSD, band power averages). EEGNet implements:
1. **Temporal Convolutions (`Conv2d`):** Learns frequency band filters directly from data at multiple frequency scales without phase distortion.
2. **Depthwise Spatial Convolutions (`Conv2dWithConstraint`):** Learns spatial filters mapping electrode topographies, mathematically equivalent to spatial beamforming or CSP.
3. **Separable Convolutions:** Decouples spatial and temporal learning, preventing parameter explosion (only $\sim 2,500$ parameters total).

### 3. Class Imbalance Handling
The dataset exhibits significant class imbalance (approx. 4:1 normal vs. conflict trials). The legacy SVM struggled to construct an optimal hyper-plane without biasing predictions toward the majority class. EEGNet integrates dynamic inverse class weights directly into cross-entropy loss:
$$w_c = \frac{N_{\text{total}}}{2 \cdot N_c}$$
combined with `ReduceLROnPlateau` scheduling, enabling balanced sensitivity across both states.

---

## 5. Conclusion & Operational Impact

The transition from SVM to EEGNet transforms the project from a prototype operating at chance level to a robust, state-of-the-art neuroadaptive system. All legacy SVM models and obsolete code have been retired from the active pipeline, with EEGNet establishing the primary classification engine for real-time closed-loop VR adaptation.
