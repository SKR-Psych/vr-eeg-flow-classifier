import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classifier import train_and_export_subject

def train_all_eegnet_models(
    bids_root: str = "data/ds003846",
    output_dir: str = "data/ds003846/derivatives",
    max_epochs: int = 60,
    subject_list: list = None
):
    """
    Batch trains and exports calibrated EEGNet models (.pt) across all subjects in the BIDS dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    if subject_list is None or len(subject_list) == 0:
        sub_dirs = glob.glob(os.path.join(bids_root, "sub-*"))
        subjects = [os.path.basename(d).replace("sub-", "") for d in sub_dirs if os.path.isdir(d)]
        subjects.sort()
    else:
        subjects = subject_list

    print("="*60)
    print(f"[*] Batch EEGNet Training & Model Export Pipeline")
    print(f"[*] Subjects to train: {subjects}")
    print(f"[*] Output directory: {output_dir}")
    print("="*60)

    summary_records = []

    for sub in subjects:
        try:
            pt_path, metrics = train_and_export_subject(
                subject=sub,
                bids_root=bids_root,
                output_dir=output_dir,
                max_epochs=max_epochs
            )
            summary_records.append({
                'subject_id': f"sub-{sub}",
                'model_file': os.path.basename(pt_path),
                'val_balanced_accuracy': metrics['balanced_accuracy'],
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1_score': metrics.get('f1_score', 0.0),
                'roc_auc': metrics.get('roc_auc', 0.0),
                'status': 'SUCCESS'
            })
        except Exception as e:
            print(f"[!] Error processing sub-{sub}: {e}")
            summary_records.append({
                'subject_id': f"sub-{sub}",
                'model_file': None,
                'val_balanced_accuracy': None,
                'precision': None,
                'recall': None,
                'f1_score': None,
                'roc_auc': None,
                'status': f"FAILED: {str(e)}"
            })

    df_summary = pd.DataFrame(summary_records)
    summary_csv = os.path.join(output_dir, "eegnet_batch_training_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print("\n" + "="*60)
    print(f"[+] Batch EEGNet Training Completed! Summary saved to: {summary_csv}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Train & Export EEGNet Models")
    parser.add_argument("--bids-root", type=str, default="data/ds003846", help="Dataset root")
    parser.add_argument("--output-dir", type=str, default="data/ds003846/derivatives", help="Derivatives directory")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs per subject")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subset of subjects (e.g. 02 03)")
    args = parser.parse_args()

    train_all_eegnet_models(
        bids_root=args.bids_root,
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        subject_list=args.subjects
    )
