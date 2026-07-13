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

### 1. Multi-Modal Neural Network Architecture (GNNs + CNNs)
*   **Objective:** Upgrade from classical machine learning (SVM/RF) to deep learning models that process the spatial and topological features of the brain.
*   **GNN Connectivity modeling:**
    *   Map the 64 EEG channels as nodes in a graph.
    *   Calculate dynamic functional connectivity matrices (e.g., Phase Locking Value in Theta/Alpha) to serve as weighted edges between nodes.
    *   Train a Graph Neural Network (GNN) to classify flow states based on the topological synchronization of the brain.
*   **CNN Spatial Modeling:**
    *   Convert power spectral densities into 2D topographical maps (topomaps).
    *   Train a Convolutional Neural Network (CNN) to extract spatial patterns of alpha blocking and frontal theta activity.

### 2. Reinforcement Learning (RL) Adaptive Controller
*   **Objective:** Develop the active "neuro-symbiotic" layer where the VR game difficulty adapts dynamically to maintain optimal flow.
*   **Tasks:**
    *   Implement a Reinforcement Learning agent (e.g., Q-learning or Deep Q-Networks) in Python that communicates with the Unity VR environment.
    *   **State Space ($S$):** The smoothed probability of Flow $P(\text{Flow}_t)$ and motor performance metrics (reaching speed, accuracy, jitter).
    *   **Action Space ($A$):** Modifications to game parameters (adjust target sizes, alter speed, modulate electromyostimulation (EMS) feedback strength, change visual spacing).
    *   **Reward Function ($R$):** Design the reward to maximize the duration of the flow state:
        $$R_t = P(\text{Flow}_t) - c \cdot \Delta D_t$$
        *(where $\Delta D_t$ penalizes large, jarring difficulty changes to maintain sensory immersion).*
