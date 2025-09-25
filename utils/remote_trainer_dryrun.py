"""Dry-run test script for RemoteTrainer.

This submits lightweight placeholder 'training' commands to two different
instances so you can verify scheduling, polling, artifact collection, and
logging without running full training.

Hosts used:
  - 34.136.139.51
  - 34.69.195.101

WHAT THE PLACEHOLDER COMMAND DOES
---------------------------------
Each job:
  * Sleeps a random short duration (to emulate variable training time)
  * Writes a small JSON results file (train_results.json)
  * Prints progress lines to stdout

Adjust the SSH username / key path via CLI arguments.

Example:
  python utils/remote_trainer_dryrun.py --user xinting --key /home/xinting/.ssh/id_rsa \
      --jobs 4 --parallel 2 --log-level DEBUG

TODO: Integrate with real search loop later (feeding configs & capturing val_loss).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import string
import time
from pathlib import Path

from remote_trainer import RemoteTrainer, build_training_command, JobStatus  # type: ignore

HOSTS = ["34.136.139.51", "34.69.195.101"]

def make_placeholder_training(result_name: str = "train_results.json") -> str:
    """Return the command to invoke the standalone fake training script.

    Assumes `fake_train_job.py` is present on the remote host in the current
    working directory of the launched job (or on PYTHONPATH). If not, you can
    copy it beforehand or adjust this to use a fully qualified path.

    TODO: If repository isn't cloned on remote, add a sync step (e.g., rsync
    or `scp fake_train_job.py <host>:<dir>` before submission).
    """
    return f"python fake_train_job.py --out {result_name}"


def main():
    parser = argparse.ArgumentParser(description="Dry-run RemoteTrainer across two hosts")
    parser.add_argument("--user", required=True, help="SSH username")
    parser.add_argument("--key", required=True, help="Path to SSH private key")
    parser.add_argument("--jobs", type=int, default=4, help="Number of jobs to submit")
    parser.add_argument("--parallel", type=int, default=2, help="Max parallel (currently advisory)")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-job timeout seconds")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    trainer = RemoteTrainer(hosts=HOSTS, user=args.user, key_filename=args.key, max_parallel=args.parallel)

    jobs = []
    for i in range(args.jobs):
        # Build a placeholder command, not using build_training_command since we are not calling a real script.
        cmd = make_placeholder_training()
        job = trainer.submit(cmd, timeout_s=args.timeout)
        jobs.append(job)
        time.sleep(0.5)  # small stagger

    # Wait for all jobs
    trainer.wait_all()
    trainer.fetch_all()

    # Summarize
    summary = []
    for job in jobs:
        summary.append({
            'job_id': job.job_id,
            'host': job.host,
            'status': job.status,
            'result': job.result_payload,
            'error': job.error,
        })
    print("\nJob Summary:")
    print(json.dumps(summary, indent=2))

    # Simple stats
    ok = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)
    print(f"Completed {ok}/{len(jobs)} jobs")

if __name__ == "__main__":
    main()
