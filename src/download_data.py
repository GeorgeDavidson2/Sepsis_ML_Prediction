"""
src/download_data.py
Downloads the PhysioNet/CinC 2019 dataset from Kaggle and organises
the raw .psv files into data/raw/training_setA/ and training_setB/.

Requires a Kaggle API token (~/.kaggle/kaggle.json).
Run from the project root: python src/download_data.py
"""

import os
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SET_A_DIR    = PROJECT_ROOT / "data" / "raw" / "training_setA"
SET_B_DIR    = PROJECT_ROOT / "data" / "raw" / "training_setB"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "_tmp_download"

KAGGLE_DATASET = "salikhussaini49/prediction-of-sepsis"


def check_kaggle_credentials():
    """Verify kaggle.json exists and kaggle package is importable."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("ERROR: kaggle.json not found.")
        print("  1. Go to https://www.kaggle.com/settings → API → Create New Token")
        print(f"  2. Place the downloaded kaggle.json at: {kaggle_json}")
        sys.exit(1)
    # Ensure correct permissions (Kaggle CLI requires 600)
    kaggle_json.chmod(0o600)

    try:
        import kaggle
    except ImportError:
        print("ERROR: kaggle package not found. Install it with:")
        print("  pip install kaggle")
        sys.exit(1)

    print(f"Kaggle credentials found: {kaggle_json}")
    print(f"Kaggle package version  : {kaggle.__version__}")


def check_already_downloaded():
    """Skip download if data already exists."""
    set_a_count = len(list(SET_A_DIR.glob("*.psv"))) if SET_A_DIR.exists() else 0
    set_b_count = len(list(SET_B_DIR.glob("*.psv"))) if SET_B_DIR.exists() else 0

    if set_a_count > 1000 and set_b_count > 1000:
        print(f"Data already present — Set A: {set_a_count:,} files, Set B: {set_b_count:,} files.")
        print("Delete data/raw/training_setA/ and training_setB/ to re-download.")
        sys.exit(0)


def download_from_kaggle():
    """Download and unzip dataset using the Kaggle Python API."""
    from kaggle import api

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset: {KAGGLE_DATASET}")
    print("This may take a few minutes (~42 MB)...")

    api.authenticate()
    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(DOWNLOAD_DIR),
        unzip=True,
        quiet=False
    )
    print("Download complete.")


def organise_files():
    """
    Move downloaded .psv files into the correct project folders.

    The Kaggle dataset contains two subfolders:
      training_setA/  → data/raw/training_setA/
      training_setB/  → data/raw/training_setB/
    """
    SET_A_DIR.mkdir(parents=True, exist_ok=True)
    SET_B_DIR.mkdir(parents=True, exist_ok=True)

    psv_files = list(DOWNLOAD_DIR.rglob("*.psv"))
    if not psv_files:
        print("ERROR: No .psv files found in downloaded archive.")
        print(f"  Check contents of: {DOWNLOAD_DIR}")
        sys.exit(1)

    print(f"Found {len(psv_files):,} .psv files. Organising into Set A and Set B...")

    moved_a = moved_b = 0
    for f in psv_files:
        parent = f.parent.name.lower()
        if "seta" in parent or "set_a" in parent or "training_seta" in parent or parent.endswith("a"):
            shutil.move(str(f), SET_A_DIR / f.name)
            moved_a += 1
        elif "setb" in parent or "set_b" in parent or "training_setb" in parent or parent.endswith("b"):
            shutil.move(str(f), SET_B_DIR / f.name)
            moved_b += 1
        else:
            # Fallback: split by filename — p0xxxxx → A, p1xxxxx → B
            first_digit = f.stem[1] if len(f.stem) > 1 and f.stem[1].isdigit() else '0'
            if int(first_digit) <= 4:
                shutil.move(str(f), SET_A_DIR / f.name)
                moved_a += 1
            else:
                shutil.move(str(f), SET_B_DIR / f.name)
                moved_b += 1

    return moved_a, moved_b


def cleanup():
    """Remove the temporary download directory."""
    if DOWNLOAD_DIR.exists():
        shutil.rmtree(DOWNLOAD_DIR)
        print("Cleaned up temporary download folder.")


def print_summary():
    set_a_count = len(list(SET_A_DIR.glob("*.psv")))
    set_b_count = len(list(SET_B_DIR.glob("*.psv")))

    print("\n" + "─" * 50)
    print("DOWNLOAD COMPLETE")
    print("─" * 50)
    print(f"  Set A : {set_a_count:>6,} files  →  data/raw/training_setA/")
    print(f"  Set B : {set_b_count:>6,} files  →  data/raw/training_setB/")
    print(f"  Total : {set_a_count + set_b_count:>6,} patient files")
    print("─" * 50)

    if set_a_count < 1000 or set_b_count < 1000:
        print("WARNING: File counts are lower than expected (~20,000 per set).")
        print("  The Kaggle mirror may be a subset. Consider downloading from:")
        print("  https://physionet.org/content/challenge-2019/1.0.0/")
    else:
        print("Next step: run notebooks/00_EDA.ipynb")


if __name__ == "__main__":
    print("=" * 50)
    print("PhysioNet/CinC 2019 Dataset Downloader")
    print("=" * 50)

    check_kaggle_credentials()
    check_already_downloaded()
    download_from_kaggle()
    organise_files()
    cleanup()
    print_summary()
