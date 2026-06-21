# VR EEG Flow Classifier

A real-time Python/Rust analysis and classification pipeline designed to detect the psychological **flow state** using a **64-channel Brain Products actiCAP** active EEG system integrated with a **Virtual Reality (VR) headset**.

## Project Purpose & Architecture

The ultimate long-term goal of this project is to create a real-time, closed-loop neuroadaptive VR experience. Once the classifiers are trained and validated, the pipeline will function as follows:

1. **EEG Data Collection:** A 64-channel Brain Products actiCAP collects active EEG signals while the user plays a Meta Unity VR game.
2. **Real-time Processing (`goofi-pipe`):** The EEG stream is processed in real-time through `goofi-pipe`, a graphical data flow pipeline for physiological signals.
3. **Flow State Classification:** The Python/Rust module developed in this repository will run as a custom node inside `goofi-pipe` to estimate the user's flow state (e.g., flow score from 0 to 1, or state classifications like Boredom, Flow, or Stress).
4. **Firebase Synchronization:** The classified flow metrics are written in real-time to a **Google Firebase Realtime Database**.
5. **Unity VR Adaptation:** The Meta Unity VR game retrieves the real-time flow metrics from Firebase. Inside Firebase, specific settings configuration parameters for different VR mini-games are adjusted dynamically (e.g., scaling difficulty, changing environment cues, adjusting task challenge levels) to keep the player in their optimal "flow channel."

---

## Directory Structure

*   [biomarker_plan.md](biomarker_plan.md): The scientific blueprint outlining the specific EEG biomarkers (Theta, SMR, Hemispheric Asymmetry, Shannon Entropy, and Network Connectivity), how they are calculated, and references to the source literature.
*   [ai_instructions.md](ai_instructions.md): A technical guide for AI coding assistants (like Cursor, Claude, or Gemini) to understand project constraints, success criteria, and code style.
*   `src/`: (To be created) Python prototyping scripts and Rust-optimized signal processing libraries.
*   `data/`: (To be created) Local directories to store test dataset folders (e.g., BIDS datasets from OpenNeuro).

---

## Getting Started

### 1. Test Dataset
To develop and validate the pipeline, we test our scripts on the open-source BIDS 64-channel VR Reach-to-Object dataset (`ds003846`) available on OpenNeuro. 
*   Cloned repository containing the dataset details: `https://github.com/SKR-Psych/ds003846`

### 2. Setup (Prototyping Environment)
Install the core Python dependencies for BIDS loading and signal processing:
```bash
pip install mne mne-bids numpy scipy scikit-learn pyfire
```

*(Further setup instructions and Rust build steps will be added as code development begins.)*
