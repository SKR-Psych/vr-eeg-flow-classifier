# BCI-VR Closed-Loop System: Technical Roadmap

This document outlines the software engineering, signal processing, and machine learning roadmap to transition this project from offline validation to a real-time, adaptive neuro-symbiotic system.

---

## Part 1: Immediate Integration Tasks (Next 1–3 Months)

### 1. Register Goofi-Pipe Node
*   **Objective:** Integrate the pre-validation classifier pipeline into the active `goofi-pipe` environment.
*   **Tasks:**
    *   Copy the template code from [goofi_integration.md](file:///C:/Users/Sami/Desktop/Uni/vr-eeg-flow-classifier/goofi_integration.md) into your local `goofi-pipe` directory at `src/goofi/nodes/analysis/flowclassifier.py`.
    *   Register the node within the `goofi/nodes/__init__.py` (or corresponding node registry file) under the `analysis` category.
    *   Verify that the node dynamically loads the correct subject model (e.g. `sub-02_calibrated_classifier.joblib`) from your derivatives folder based on a node parameter or subject ID input.

### 2. Live LSL Stream Validation
*   **Objective:** Test the closed-loop system latency under simulated live streaming conditions.
*   **Tasks:**
    *   Set up a mock LSL (Lab Streaming Layer) sender script to stream raw 64-channel EEG data at $500\text{ Hz}$ and Unity game event markers concurrently.
    *   Connect Goofi-Pipe to the LSL streams.
    *   Run the real-time pipeline and verify that feature extraction (spectral powers, Shannon entropy, and phase connectivity) and inference execute in under **$100\text{ ms}$** per sliding window (verified offline at **$82.11\text{ ms}$**).

### 3. Bayesian Classification Smoothing
*   **Objective:** Prevent erratic VR adaptations caused by single noisy classification windows.
*   **Tasks:**
    *   Implement a temporal smoothing layer on top of the raw classifier output probabilities $P(\text{Flow})$.
    *   Use a recursive Bayesian updating mechanism or a sliding window moving average:
        $$P(\text{Flow}_t) = \alpha \cdot P(\text{Flow}_{raw, t}) + (1 - \alpha) \cdot P(\text{Flow}_{t-1})$$
    *   This ensures difficulty changes are only triggered when the user's state changes consistently for 2–3 seconds.

---

## Part 2: Medium-to-Long Term Technical Development (Months 3+)

### 1. Shapley Value Feature Attribution & Multi-Biomarker Optimization (Phase 4 Validation)
*   **Objective:** Leverage cooperative game theory (SHAP) to explain biomarker contributions and prune feature noise.
*   **37-Biomarker Discovery:**
    *   **`leapd_lpc_index` (LEAPD LPC-based spectral envelope index):** Identified as the **#1 single most predictive scalar biomarker** ($|\Phi| = 0.0202$, 4.9% cohort contribution).
    *   **Sensorimotor Rhythms (`smr_alpha_c4`, `smr_gamma_cp3/c3`):** Key motor automation and execution signatures over motor cortex C3/C4 (11.6% combined contribution).
    *   **Prefrontal Entropy (`entropy_fp2`, `entropy_af8`) & 1/f Spectral Slope (`spectral_slope_1f`):** Quantify cognitive workload, attention regularization, and $E/I$ balance.
*   **Shapley Biomarker Pruning:** Filtering traditional ML feature sets to top Shapley-attributed biomarkers boosted classical model accuracy from **58.42% to 60.40%** (+1.98% gain).
*   **Shapley Composite Biomarker Index (SCBI):** Synthesized a 1D interpretable composite index score for real-time tracking:
    $$SCBI_i = \sum_{j \in \text{Top-K}} w_j \cdot z(X_{i,j})$$

### 2. Hybrid Shapley-Deep Learning Attention Architecture (Shapley + EEGNet / GNN)
*   **Objective:** Combine the high predictive accuracy of deep neural networks (**91.87%**) with 100% scientific explainability.
*   **Shapley-Constrained EEGNet:**
    *   Inject cohort-wide Shapley value weights $|\Phi_j|$ as pre-trained spatial attention priors into EEGNet's depthwise spatial convolution layer.
    *   Prevents spatial overfitting and ensures the deep neural network attends to neurophysiologically validated electrodes (e.g., Fz, FCz, C3, C4).
*   **Shapley Graph Attention Networks (GAT):**
    *   Use Shapley attributions to dynamically initialize node attention weights in the Spatial-Temporal Graph Neural Network (S-T GNN).
    *   Weight graph edges using Phase Locking Value (PLV) and long-range coherence ($\beta/\gamma$) derived from Shapley ranking.

### 3. Reinforcement Learning (RL) Adaptive Controller & Real-Time VR Neurofeedback
*   **Objective:** Develop the active "neuro-symbiotic" layer where the VR game difficulty adapts dynamically to maintain optimal flow.
*   **Real-Time VR Stream Integration:**
    *   Stream `leapd_lpc_index` and `smr_alpha_c4` live into `goofi-pipe` via Lab Streaming Layer (LSL).
*   **Reinforcement Learning Agent (DQN):**
    *   Implement a Deep Q-Network (DQN) agent in Python communicating with Unity VR via sockets.
    *   **State Space ($S$):** The smoothed probability of Flow $P(\text{Flow}_t)$, SCBI score, and motor performance metrics.
    *   **Action Space ($A$):** Real-time task modifications (adjusting target sizes, reach velocities, electromyostimulation (EMS) feedback, and visual clutter).
    *   **Reward Function ($R$):** Designed to maximize sustained flow duration while penalizing abrupt difficulty changes:
        $$R_t = P(\text{Flow}_t) - c \cdot \Delta D_t$$
