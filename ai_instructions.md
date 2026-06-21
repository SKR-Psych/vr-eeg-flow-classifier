# AI Brief & System Instructions: VR EEG Flow Classifier

Welcome, AI coding assistant. This repository is dedicated to building the signal processing and machine learning classifier codebase to detect the **psychological flow state** in real-time.

Read this document carefully to understand the architectural requirements, project context, and success criteria.

---

## 1. Project Context & The Big Picture

This repository is one module of a larger real-time closed-loop neuroadaptive VR ecosystem:

```
[64-Channel actiCAP EEG] 
         │
         ▼ (Real-time LSL Stream)
[goofi-pipe Processing] 
         │
         ▼ (This Repository's Python/Rust Node)
[Classified Flow State Output] ──► [Google Firebase Realtime DB] 
                                                  │
                                                  ▼ (Dynamic Settings Adjustments)
                                     [Meta Unity VR Mini-Games]
```

### The Pipeline Steps:
1. **EEG Data Source:** A 64-channel Brain Products actiCAP active electrode system integrated with a Meta VR headset.
2. **Streaming & GUI:** Real-time data is handled by `goofi-pipe` (a Python-based graphical data pipeline tool).
3. **Classifier Node (This Repo):** You will write the classification algorithm (Python for prototyping/ML, Rust for low-latency speed) that hooks into `goofi-pipe` as a custom node.
4. **Data Sync:** The classification output (e.g., a normalized "Flow Score" from 0 to 1, or state classifications like `[Boredom, Flow, Stress]`) is sent to **Google Firebase**.
5. **VR Game Adaptation:** The Unity VR game reads active flow scores from Firebase and uses them to adjust difficulty settings, challenges, or environments for various VR mini-games in real-time.

---

## 2. Technical Blueprint & Feature Engineering

You must implement the feature extraction steps defined in the [biomarker_plan.md](biomarker_plan.md) file in this repository:
*   **Preprocessing:** Bandpass filter (0.5–45 Hz), bad channel interpolation, and ICA-based rejection of blinks/VR head-strap muscle noise.
*   **Spectral Power:** Normalized relative band powers (Theta, Alpha, Beta, Gamma).
*   **Frontal Midline Theta ($Fm\theta$):** Extract from Fz/FCz/Cz. Look for the inverted-U curve to isolate flow from disengagement and stress.
*   **Sensorimotor Rhythms (SMR):** Extract from C3/C4 for motor evaluation during active VR interaction.
*   **Inter-Hemispheric Asymmetry:** Track Beta asymmetry (asymmetry near 0 indicates peak flow).
*   **Signal Complexity:** Shannon Entropy of prefrontal AF7/AF8 channels (lower entropy indicates focused coordination).
*   **Functional Connectivity:** Coherence metrics (PLV / PSI) between Frontal (ECN) and Parietal (DMN) regions.

---

## 3. Implementation Guidelines for AI Assistants

When asked to generate code or structure this project, adhere to these rules:
1. **Python Prototype First:** Build a modular Python pipeline utilizing `mne` (for BIDS dataset preprocessing and loading) and `scikit-learn` or `PyTorch` (for classifying flow states).
2. **Rust Porting:** Optimize low-latency feature extraction (e.g., FFT calculation, entropy, and asymmetry computation) in **Rust** (bind via `pyo3` or compile to a standalone binary) to ensure real-time performance inside `goofi-pipe`.
3. **Goofi-Pipe Integration:** Ensure the module exposes a clean Python interface suitable for wrapping as a custom `goofi-pipe` node.
4. **Firebase Sync:** Write a lightweight sync script to write the classifier outputs asynchronously to Firebase Realtime Database.

---

## 4. Success Criteria
*   **Accuracy:** Successfully discriminate the "flow state" from "boredom" and "stress/overload" on 64-channel test data (like the OpenNeuro `ds003846` dataset).
*   **Latency:** The feature extraction and classification loop must complete in **less than 100ms** to support real-time Unity updates.
*   **Clean Code:** Follow PEP 8 guidelines for Python and standard idiom conventions for Rust. Ensure all code blocks are well-documented.
