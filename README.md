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

*   [biomarker_plan.md](biomarker_plan.md): Scientific blueprint outlining the specific EEG biomarkers (Theta, SMR, Hemispheric Asymmetry, Shannon Entropy, and Network Connectivity).
*   [technical_roadmap.md](technical_roadmap.md): Software engineering, signal processing, and ML roadmap for system integration.
*   [goofi_integration.md](goofi_integration.md): Architecture and Python code template for packaging the classifier into `goofi-pipe`.
*   [flow_prediction_evaluation.ipynb](flow_prediction_evaluation.ipynb): Baseline evaluation report and scientific critique of offline performance.
*   `src/`: Core Python modules:
    *   `loader.py`: Ingestion of BIDS 64-channel EEG raw data (`ds003846`).
    *   `preprocessing.py`: Rank-safe 0.5–45 Hz filtering, 10–20 montage setup, and `mne-icalabel` ICA artifact removal.
    *   `features.py`: Calculation of the 14 continuous EEG biomarkers across sliding windows.
    *   `classifier.py`: Classifier training (Random Forest & SVM) with chronological splitting and GroupKFold CV.
    *   `batch_extract.py`: Parallel feature extraction across all dataset subjects and sessions.
    *   `evaluate_all.py`: Multi-subject evaluation and export of calibrated `.joblib` models.
    *   `predict.py`: Simulated real-time streaming pipeline achieving sub-100ms latency (~82ms).
*   `data/`: Local storage for dataset files and trained model derivatives (`data/ds003846/derivatives/`).

---

## Getting Started

### 1. Test Dataset & Setup
To develop and validate the pipeline, we test our scripts on the open-source BIDS 64-channel VR Reach-to-Object dataset (`ds003846`) available on OpenNeuro. 

The raw dataset is extremely large (~11.4 GB) and is ignored by Git (`data/` is added to `.gitignore`). To set up the data locally:

1. Install the dataset download client and required python packages:
   ```bash
   pip install openneuro-py mne mne-bids numpy scipy scikit-learn pyfire
   ```

2. Download the dataset to the local `data/ds003846` directory:
   ```bash
   python -m openneuro download --dataset=ds003846 --tag=2.0.2 --target-dir=data/ds003846
   ```

*(Further setup instructions and Rust build steps will be added as code development begins.)*
