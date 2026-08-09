# Shapley Feature Attribution & Multi-Biomarker Discovery for VR-EEG Flow State Classification

**Authors:** Research & Development Team  
**Dataset:** OpenNeuro `ds003846` (64-Channel actiCAP EEG during VR Reach-to-Touch)  
**Date:** August 2026  
**Document Status:** Academic Research Reference & Technical Documentation  

---

## Executive Summary

This research document investigates the application of **Shapley Value Feature Attribution (SHAP)** to an expanded suite of 37 neurophysiological EEG biomarkers for predicting flow states and prediction errors in virtual reality (VR). 

Key findings demonstrate that:
1. **Biomarker Ranking**: Linear Predictive Coding spectral envelope energy (**LEAPD LPC Index**, *Anjum et al., 2024*) was identified as the **single most predictive scalar biomarker** ($|\Phi| = 0.0202$, 4.9% cohort contribution), followed by sensorimotor alpha power (`C4`) and gamma power (`C3/CP3`).
2. **Noise Reduction in Classical Models**: Filtering feature sets down to top Shapley-attributed biomarkers boosted traditional machine learning performance from **58.42% to 60.40%** (+1.98 percentage point net accuracy gain).
3. **Architectural Comparison**: While Shapley-guided selection optimizes traditional scalar feature models, deep learning architectures (**EEGNet** and **S-T GNNs**) achieve superior performance (**91.87% accuracy**) because 2D spatial-temporal depthwise convolutions retain complete continuous phase topography across all 64 scalp electrodes.

---

## 1. Introduction & Scientific Background

Flow state and prediction error detection in immersive Virtual Reality (VR) require tracking subtle shifts in cortical synchrony while filtering out mechanical and electromagnetic interference (EMI). While prior studies relied on isolated frequency bands (e.g. Frontal Midline Theta), recent neurophysiological literature suggests multi-modal and non-linear EEG signatures provide richer insights into cognitive control and sensorimotor automation.

### Targeted Neurophysiological Biomarkers (Literature Review):
Based on recent literature across EEG cognitive and clinical studies:
- **LEAPD Index (Anjum et al., 2024)**: Geometric mean of Linear Predictive Coding (LPC) spectral envelope shape energies across ROI sensors, quantifying spectral fingerprint shifts.
- **Sensorimotor Rhythms ($\alpha, \beta, \gamma$) (Tan et al., 2024; Hodnik et al., 2024)**: Sensorimotor alpha/beta power and $\beta \to \gamma$ Phase-Amplitude Coupling (PAC) over C3/C4/CP3/CP4, capturing movement automation and suppression of peripheral noise.
- **Aperiodic 1/f Spectral Slope (Giménez-Aparisi et al., 2023)**: Log-log linear exponent ($\chi$) of Welch PSD across 1–40 Hz, indexing global excitation-inhibition ($E/I$) balance.
- **Sensorimotor $\alpha$-Band Non-Linearity (L-Index) (Özkurt, 2024)**: High-order moment asymmetry (skewness and kurtosis), capturing non-linear dynamic complexity.
- **Prefrontal Shannon Entropy (Rosso et al.)**: Time-domain signal uncertainty over prefrontal array (AF7, AF8, Fp1, Fp2).
- **Long-Range Coherence (Waninger et al., 2020)**: Inter-regional $\beta$ and $\gamma$ coherence between frontal and parietal electrode pairs (Fp2–P3, F3–P4).

---

## 2. Methodology

```mermaid
graph TD
    A[64-Channel actiCAP Raw EEG] --> B[Zero-Phase Bandpass Filter 0.5-45Hz]
    B --> C[Infomax ICA & ICLabel Artifact Rejection]
    C --> D[Event-Locked Epoching: -100ms to +600ms]
    
    D --> E[Multi-Biomarker Extraction: 37 Features]
    E --> F[LEAPD LPC & 1/f Spectral Slope]
    E --> G[Sensorimotor PAC & Alpha Non-Linearity]
    E --> H[Long-Range Coherence & Entropy]
    
    F --> I[Feature Matrix X]
    G --> I
    H --> I
    
    I --> J[Random Forest / XGBoost Model Training]
    J --> K[Shapley Attribution Engine: SHAP TreeExplainer]
    K --> L[Global Feature Importance Ranking |Phi_j|]
    L --> M[Shapley-Selected Top-8 Feature Model]
    L --> N[Shapley Composite Biomarker Index SCBI]
```

### 2.1 Feature Extraction Pipeline (`src/features.py`)
Features were extracted from continuous and event-locked (-100 ms to +600 ms centered on `box:touched`) EEG windows across 64 scalp electrodes.

### 2.2 Shapley Attribution Engine (`src/shapley_attribution.py`)
Using cooperative game theory, the Shapley value $\phi_j(x)$ measures feature $j$'s marginal contribution to the prediction across all possible feature sub-coalitions $S \subseteq F \setminus \{j\}$:
$$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]$$

Global biomarker importance $|\Phi_j|$ was computed as the cohort-wide mean absolute SHAP value:
$$|\Phi_j| = \frac{1}{N} \sum_{i=1}^N |\phi_{i,j}|$$

### 2.3 Synthesis of Shapley Composite Biomarker Index (SCBI)
To condense high-dimensional biomarkers into a single interpretable clinical index, the **Shapley Composite Biomarker Index (SCBI)** was formulated:
$$SCBI_i = \sum_{j \in \text{Top-K}} w_j \cdot z(X_{i,j})$$
where $w_j = \text{sign}(\text{corr}(X_j, y)) \times \frac{|\Phi_j|}{\sum |\Phi|}$, and $z(X_{i,j})$ is z-score standardized feature value.

---

## 3. Experimental Results & Discussion

### 3.1 Biomarker Importance Ranking (Cohort-Wide)

| Rank | Biomarker Symbol | Category | Mean SHAP $|\Phi|$ | % Contribution | Proposed Mechanism |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | `leapd_lpc_index` | LPC Envelope Energy | **0.0202** | **4.9%** | LPC geometric energy shifts as cortical rhythms synchronize |
| **2** | `smr_alpha_c4` | Motor Cortex Power | **0.0174** | **4.2%** | Right motor inhibition during automated reaching |
| **3** | `smr_gamma_cp3` | Sensorimotor Burst | **0.0155** | **3.8%** | Local gamma synchronization during movement execution |
| **4** | `smr_gamma_c3` | Sensorimotor Burst | **0.0150** | **3.6%** | Left motor cortex task-related gamma power |
| **5** | `entropy_fp2` | Signal Complexity | **0.0143** | **3.5%** | Prefrontal entropy drops during focused attention |
| **6** | `spectral_slope_1f` | Aperiodic Slope | **0.0137** | **3.3%** | Steeper $1/f$ slope reflects reduced background noise |
| **7** | `fmt_delta_theta_ratio`| Band Power Ratio | **0.0133** | **3.2%** | Frontal midline slowing under cognitive workload |
| **8** | `lindex_alpha_cp3` | Non-Linearity | **0.0130** | **3.1%** | Non-linear alpha wave asymmetry post-touch |
| **9** | `entropy_af8` | Signal Complexity | **0.0128** | **3.1%** | Prefrontal signal regularization during flow immersion |
| **10** | `longrange_gamma_coherence`| Connectivity | **0.0128** | **3.1%** | Frontal-parietal gamma network coupling |

### 3.2 Performance Comparison Across Modeling Paradigms

Evaluation was conducted using chronological 80/20 train/test splits across benchmark participants (`sub-02` through `sub-08`):

| Paradigm / Model Architecture | Mean Accuracy | Precision | F1-Score | Architectural Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **Full Feature Set (26 Scalar Biomarkers)** | 58.42% | 0.610 | 0.582 | Multi-feature summary baseline |
| **Shapley-Selected Model (Top-8 Features)** | **60.40%** | **0.638** | **0.601** | **+1.98% gain** via noise feature pruning |
| **Shapley Composite Biomarker Index (SCBI)** | 55.61% | 0.575 | 0.550 | 1D interpretable composite index |
| **xDAWN + CSP Spatial Filtered Classifier** | **75.22%** | 0.771 | 0.758 | Maximizes ERP SNR & separates spatial variance |
| **Phase 3 EEGNet (Convolutional NN)** | **91.52%** | 0.920 | 0.914 | 2D depthwise spatial + separable temporal convolutions |
| **Phase 3 Peak S-T GNN (Graph Neural Net)** | **91.87%** | **0.925** | **0.919** | Dynamic PLV graph connectivity topology |

---

## 4. Key Scientific Insights

1. **Why Shapley Biomarker Pruning Works**: Traditional feature models suffer when flooded with correlated or noisy band powers. By isolating top SHAP-attributed features (`leapd_lpc_index`, `smr_alpha_c4`, `entropy_fp2`, `spectral_slope_1f`), model accuracy consistently improved across participants.
2. **Why Deep Learning Outperforms Feature Engineering**: Scalar biomarkers compress 700 ms of multi-channel voltage waveforms into single summary statistics. **EEGNet and GNNs** achieve **91.87%** because 2D spatial-temporal convolutions preserve raw microsecond voltage dynamics and cross-electrode phase topologies.

---

## 5. Future Research & Development Roadmap

1. **Shapley-Guided Attention in Deep Neural Networks (Hybrid EEGNet-SHAP)**:
   Use global Shapley values as prior spatial attention weights inside EEGNet's depthwise spatial layer to enforce neurophysiologically-constrained neural network training.
2. **Real-Time VR Neurofeedback Loop**:
   Deploy `leapd_lpc_index` and `smr_alpha_c4` in real-time streaming engines (e.g. *Lab Streaming Layer / GOOFI*) to adapt VR task difficulty dynamically whenever flow state disruption is detected.

---

*Report saved to project repository: `docs/shapley_biomarker_research_report.md`*
