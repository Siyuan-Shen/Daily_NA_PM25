#!/usr/bin/env python3
"""
Scan job_output/*.txt files and classify each job as:
  SUCCESS  - completed without errors (ends with "Total time taken")
  FAIL     - Python or C++ exception found in output (even if job reached the end)
  KILLED   - SLURM cancelled the job
  RUNNING  - job is currently active in the SLURM queue
  UNKNOWN  - none of the above

Writes job_status.csv with columns: job_id, status, last_line
"""

import os
import re
import csv
import glob
import subprocess
from collections import Counter

JOB_OUTPUT_DIR = "./job_output"
OUTPUT_CSV = "./job_output/job_status.csv"

# Patterns that indicate a crash anywhere in the file
FAIL_PATTERNS = re.compile(
    r"Traceback \(most recent call last\)"
    r"|terminate called"
    r"|Segmentation fault"
    r"|-- Process \d+ terminated with the following error"
)

CANCELLED_PATTERN = re.compile(r"CANCELLED (AT|BY)")


def is_running(job_id):
    try:
        result = subprocess.run(
            ["squeue", "--job", job_id, "--noheader"],
            capture_output=True, text=True, timeout=10
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def classify(filepath, job_id):
    with open(filepath, "r", errors="replace") as f:
        lines = f.readlines()

    full_text = "".join(lines)
    non_empty = [l.rstrip() for l in lines if l.strip()]
    last_line = non_empty[-1] if non_empty else ""

    # Priority 1: SLURM cancellation
    if CANCELLED_PATTERN.search(full_text):
        for l in reversed(non_empty):
            if "CANCELLED" in l:
                return "KILLED", l.strip()
        return "KILLED", last_line

    # Priority 2: Python / C++ crash anywhere in file
    if FAIL_PATTERNS.search(full_text):
        return "FAIL", last_line

    # Priority 3: clean completion
    if "Total time taken" in last_line:
        return "SUCCESS", last_line

    # Priority 4: still in SLURM queue
    if is_running(job_id):
        return "RUNNING", last_line

    return "UNKNOWN", last_line


def main():
    pattern = os.path.join(JOB_OUTPUT_DIR, "job-*-output.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No job output files found in {JOB_OUTPUT_DIR}")
        return

    rows = []
    for filepath in files:
        fname = os.path.basename(filepath)
        m = re.search(r"job-(\d+)-output", fname)
        job_id = m.group(1) if m else fname
        status, last_line = classify(filepath, job_id)
        rows.append((job_id, status, last_line))

    rows.sort(key=lambda r: int(r[0]) if r[0].isdigit() else 0)

    with open(OUTPUT_CSV, "w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["job_id", "status", "last_line"])
        writer.writerows(rows)

    counts = Counter(r[1] for r in rows)
    print(f"Wrote {len(rows)} entries to {OUTPUT_CSV}")
    for status in ["SUCCESS", "FAIL", "KILLED", "RUNNING", "UNKNOWN"]:
        if status in counts:
            print(f"  {status}: {counts[status]}")


if __name__ == "__main__":
    main()
