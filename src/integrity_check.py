"""
src/integrity_check.py
────────────────────────────────────────────────────────────────────────────────
Runs 5 integrity checks on the raw PhysioNet/CinC 2019 dataset and saves a
report to results/metrics/integrity_report.txt.

Run from the project root:
    python src/integrity_check.py

Expected results:
    - Total files  : ~40,336
    - Columns/file : 41 (40 features + SepsisLabel)
    - SepsisLabel  : only 0 and 1
    - ICULOS       : monotonically increasing in every file
    - Sepsis rate  : ~5.6%
"""

import random
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SET_A_DIR    = PROJECT_ROOT / "data" / "raw" / "training_setA"
SET_B_DIR    = PROJECT_ROOT / "data" / "raw" / "training_setB"
REPORT_PATH  = PROJECT_ROOT / "results" / "metrics" / "integrity_report.txt"

EXPECTED_COLS   = 41   # 40 features + SepsisLabel
EXPECTED_TOTAL  = 40_336
EXPECTED_SEPSIS = 0.056  # ~5.6%

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────

class Tee:
    """Write output to both stdout and a StringIO buffer."""
    def __init__(self):
        self.buf = StringIO()

    def write(self, text):
        sys.stdout.write(text)
        self.buf.write(text)

    def flush(self):
        sys.stdout.flush()

    def getvalue(self):
        return self.buf.getvalue()


def divider(tee, char="─", width=60):
    tee.write(char * width + "\n")


# ── Checks ────────────────────────────────────────────────────────────────────

def check_file_counts(tee, files_a, files_b):
    tee.write("\nCHECK 1 — File Counts\n")
    divider(tee)
    total = len(files_a) + len(files_b)
    tee.write(f"  Set A : {len(files_a):>6,} files  (expected ~20,336)\n")
    tee.write(f"  Set B : {len(files_b):>6,} files  (expected ~20,000)\n")
    tee.write(f"  Total : {total:>6,} files  (expected ~40,336)\n")

    if total < 30_000:
        tee.write("  WARNING: Total file count is much lower than expected.\n")
        tee.write("           You may have a partial download.\n")
    else:
        tee.write("  PASS\n")


def check_columns(tee, all_files):
    tee.write("\nCHECK 2 — Column Consistency\n")
    divider(tee)
    bad_files = []
    for f in tqdm(all_files, desc="  Checking columns", leave=False):
        df = pd.read_csv(f, sep="|", nrows=1)
        if len(df.columns) != EXPECTED_COLS:
            bad_files.append((f, len(df.columns)))

    if bad_files:
        tee.write(f"  FAIL — {len(bad_files)} file(s) have wrong column count:\n")
        for path, count in bad_files[:10]:
            tee.write(f"    {Path(path).name}  →  {count} columns\n")
    else:
        tee.write(f"  All {len(all_files):,} files have exactly {EXPECTED_COLS} columns.\n")
        tee.write("  PASS\n")

    return len(bad_files) == 0


def check_sepsis_label(tee, all_files, sample_size=1000):
    tee.write("\nCHECK 3 — SepsisLabel Values\n")
    divider(tee)
    sample = random.sample(all_files, min(sample_size, len(all_files)))
    unique_vals = set()
    for f in tqdm(sample, desc="  Checking SepsisLabel", leave=False):
        df = pd.read_csv(f, sep="|", usecols=["SepsisLabel"])
        unique_vals.update(df["SepsisLabel"].dropna().unique().tolist())

    allowed = {0, 1, 0.0, 1.0}
    unexpected = unique_vals - allowed
    tee.write(f"  Unique SepsisLabel values found : {sorted(unique_vals)}\n")
    if unexpected:
        tee.write(f"  FAIL — unexpected values: {unexpected}\n")
    else:
        tee.write("  PASS — only 0 and 1 present.\n")

    return len(unexpected) == 0


def check_iculos(tee, all_files, sample_size=500):
    tee.write("\nCHECK 4 — ICULOS Monotonically Increasing\n")
    divider(tee)
    sample = random.sample(all_files, min(sample_size, len(all_files)))
    bad = []
    for f in tqdm(sample, desc="  Checking ICULOS", leave=False):
        df = pd.read_csv(f, sep="|", usecols=["ICULOS"])
        if not df["ICULOS"].is_monotonic_increasing:
            bad.append(Path(f).name)

    if bad:
        tee.write(f"  FAIL — {len(bad)} file(s) have non-monotonic ICULOS:\n")
        for name in bad[:10]:
            tee.write(f"    {name}\n")
    else:
        tee.write(f"  All {len(sample):,} sampled files have monotonic ICULOS.\n")
        tee.write("  PASS\n")

    return len(bad) == 0


def check_quick_stats(tee, all_files):
    tee.write("\nCHECK 5 — Quick Stats (full scan)\n")
    divider(tee)
    sepsis_count = 0
    stay_lengths = []

    for f in tqdm(all_files, desc="  Scanning all files", leave=False):
        df = pd.read_csv(f, sep="|", usecols=["SepsisLabel"])
        stay_lengths.append(len(df))
        if df["SepsisLabel"].max() == 1:
            sepsis_count += 1

    total = len(all_files)
    stay_lengths.sort()
    median_stay = stay_lengths[len(stay_lengths) // 2]
    sepsis_pct  = 100 * sepsis_count / total

    tee.write(f"  Total patients   : {total:,}\n")
    tee.write(f"  Sepsis patients  : {sepsis_count:,}  ({sepsis_pct:.1f}%)\n")
    tee.write(f"  Stay length (h)  : min={stay_lengths[0]}, "
              f"median={median_stay}, max={stay_lengths[-1]}\n")

    if abs(sepsis_pct / 100 - EXPECTED_SEPSIS) > 0.02:
        tee.write(f"  WARNING: Sepsis prevalence {sepsis_pct:.1f}% deviates from expected ~5.6%\n")
    else:
        tee.write("  PASS — sepsis prevalence is within expected range.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tee = Tee()

    tee.write("=" * 60 + "\n")
    tee.write("PhysioNet/CinC 2019 — Dataset Integrity Report\n")
    tee.write("=" * 60 + "\n")

    # Collect files
    files_a = sorted(str(p) for p in SET_A_DIR.glob("*.psv")) if SET_A_DIR.exists() else []
    files_b = sorted(str(p) for p in SET_B_DIR.glob("*.psv")) if SET_B_DIR.exists() else []
    all_files = files_a + files_b

    if not all_files:
        tee.write("\nERROR: No .psv files found.\n")
        tee.write("  Run: python src/download_data.py\n")
        sys.exit(1)

    # Run checks
    check_file_counts(tee, files_a, files_b)
    check_columns(tee, all_files)
    check_sepsis_label(tee, all_files)
    check_iculos(tee, all_files)
    check_quick_stats(tee, all_files)

    tee.write("\n" + "=" * 60 + "\n")
    tee.write("Integrity check complete.\n")
    tee.write("=" * 60 + "\n")

    # Save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(tee.getvalue())
    print(f"\nReport saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
