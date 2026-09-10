"""
scripts/colab_mount.py
Mounts Google Drive and configures the environment for Colab sessions.
Run this at the start of every Colab notebook before any experiment.
Usage: exec(open('scripts/colab_mount.py').read())  — from a Colab cell
"""

import os
import sys

GDRIVE_BASE = "/content/drive/My Drive/fq-ssiuu-research"
REPO_PATH = "/content/fq-ssiuu"


def mount_drive():
    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
        print(f"Google Drive mounted. Base path: {GDRIVE_BASE}")
        # Verify expected folders exist
        for folder in [
            "checkpoints",
            "datasets",
            "results",
            "logs",
            "certificates",
            "merged_models",
        ]:
            path = os.path.join(GDRIVE_BASE, folder)
            if not os.path.exists(path):
                print(f"  WARNING: Expected folder not found: {path}")
                print(
                    f"  Create it manually in Google Drive at: My Drive/fq-ssiuu-research/{folder}"
                )
            else:
                print(f"  OK: {path}")
    except ImportError:
        print("Not in Colab — skipping Drive mount.")


def setup_python_path():
    src_path = os.path.join(REPO_PATH, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"Added to PYTHONPATH: {src_path}")


def verify_hf_token():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("WARNING: HF_TOKEN environment variable not set.")
        print("Set it with: import os; os.environ['HF_TOKEN'] = 'your_token_here'")
    else:
        print(f"HF_TOKEN detected (length: {len(token)})")


if __name__ == "__main__":
    mount_drive()
    setup_python_path()
    verify_hf_token()
    print("Colab environment configured.")
