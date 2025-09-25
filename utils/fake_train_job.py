"""Lightweight fake training script for remote dry-run tests.

Behavior:
  * Prints a start message
  * Emits 3 fake 'epoch' lines with random accuracies
  * Sleeps a random 5-15 seconds to simulate work
  * Writes a JSON results file (default: train_results.json) containing:
      - val_loss (random float 2.0-5.0)
      - throughput (random int 1000-5000)
      - timestamp (epoch seconds)
      - run_id (random 8-char alphanumeric)
  * Prints a finished message

Usage (remote host):
  python fake_train_job.py --out train_results.json

You can extend this later to actually run a small model forward pass instead of sleeping.
"""
from __future__ import annotations

import argparse
import json
import random
import string
import time
from pathlib import Path


def run(out_path: str):
    print("Starting fake training...")
    for i in range(3):
        acc = random.random()
        print(f"epoch {i} acc={acc:.4f}")
    sleep_s = random.randint(5, 15)
    time.sleep(sleep_s)
    result = {
        "val_loss": round(random.uniform(2.0, 5.0), 4),
        "throughput": random.randint(1000, 5000),
        "timestamp": time.time(),
        "run_id": ''.join(random.choices(string.ascii_lowercase + string.digits, k=8)),
        "sleep_s": sleep_s,
    }
    Path(out_path).write_text(json.dumps(result))
    print("Finished fake training.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="train_results.json", help="Output JSON file name")
    args = parser.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
