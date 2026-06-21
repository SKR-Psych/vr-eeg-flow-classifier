import argparse
import os
import mne
from mne_bids import BIDSPath, read_raw_bids

def load_bids_data(subject: str, session: str, bids_root: str = "data/ds003846") -> mne.io.Raw:
    """
    Loads raw EEG data for a given subject and session from a BIDS dataset.

    Parameters
    ----------
    subject : str
        The subject ID (e.g., '02').
        Note: The 'sub-' prefix is automatically handled.
    session : str
        The session name (e.g., 'EMS', 'Vibro', 'Visual').
        Note: The 'ses-' prefix is automatically handled.
    bids_root : str, optional
        The path to the root of the BIDS dataset, by default "data/ds003846".

    Returns
    -------
    mne.io.Raw
        The MNE Raw object containing the loaded EEG data, channel info, and annotations.
    """
    # Standardize subject/session names (strip prefix if user entered it)
    subject = subject.replace("sub-", "")
    session = session.replace("ses-", "")

    print(f"[*] Constructing BIDS path for subject '{subject}', session '{session}'...")
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task="PredictionError",
        datatype="eeg",
        root=bids_root
    )

    print(f"[*] Reading raw BIDS data from: {bids_path.basename}")
    # Read raw data using MNE-BIDS
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    
    print("[*] Loading data into memory...")
    raw.load_data(verbose=False)
    
    return raw

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and inspect EEG data in BIDS format.")
    parser.add_argument("--subject", type=str, default="02", help="Subject ID (e.g. 02)")
    parser.add_argument("--session", type=str, default="EMS", help="Session ID (e.g. EMS, Vibro, Visual)")
    parser.add_argument("--root", type=str, default="data/ds003846", help="BIDS root directory")
    args = parser.parse_args()

    # Verify root path
    if not os.path.exists(args.root):
        print(f"[!] Error: BIDS root directory '{args.root}' does not exist.")
        exit(1)

    try:
        raw = load_bids_data(args.subject, args.session, args.root)
        
        print("\n=== EEG Dataset Metadata Summary ===")
        print(f"Subject:     {args.subject}")
        print(f"Session:     {args.session}")
        print(f"Project:     {raw.info.get('project_id', 'N/A')}")
        print(f"Channels:    {len(raw.ch_names)}")
        print(f"Sample Rate: {raw.info['sfreq']} Hz")
        print(f"Duration:    {raw.times[-1]:.2f} seconds ({raw.n_times} samples)")
        
        # Get channel types
        ch_types = raw.get_channel_types()
        unique_types = set(ch_types)
        print("Channel Types:")
        for t in unique_types:
            count = ch_types.count(t)
            print(f"  - {t}: {count}")

        # Show some channel names
        print(f"First 10 Channels: {raw.ch_names[:10]}")
        
        # Get annotations/events
        annotations = raw.annotations
        print(f"Annotations/Events count: {len(annotations)}")
        if len(annotations) > 0:
            import numpy as np
            unique_descriptions, counts = np.unique(annotations.description, return_counts=True)
            print("Unique Event Descriptions:")
            for desc, count in zip(unique_descriptions, counts):
                print(f"  - '{desc}': {count}")
                
    except Exception as e:
        print(f"[!] Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
