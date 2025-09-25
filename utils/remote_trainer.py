"""Remote training orchestration utilities using Fabric.

Goal:
  Submit one or more model training jobs to remote GCP (or other SSH reachable) instances,
  monitor progress, and fetch result artifacts back locally.

Design Overview:
  - RemoteTrainer: main facade for submitting and tracking jobs across hosts.
  - RemoteJob: dataclass capturing job metadata & lifecycle state.
  - Uses Fabric for SSH exec and file transfer (similar pattern to `fabric_trial.py`).
  - Jobs launched with a lightweight bash wrapper writing PID, stdout/stderr, and a completion marker.
  - Polling loop (pull-based) watches for a "done" sentinel file then pulls artifacts.

Key Assumptions (subject to change – see TODOs):
  1. There exists a local training script (e.g. `train.py`) that when run writes a JSON results file.
  2. The remote environment already has all dependencies installed & the correct Python available.
  3. Authentication via SSH key file (no passphrase prompt) similar to fabric_trial.
  4. We accept running N jobs concurrently (default sequential for safety; add parallel exec TODO).

You SHOULD edit/update the TODO sections with your project-specific details.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fabric import Connection  # type: ignore

# ---------------------------------------------------------------------------
# Configuration & Defaults
# ---------------------------------------------------------------------------
DEFAULT_REMOTE_WORKDIR = "/tmp/pyppa_runs"  # TODO: confirm target workdir (maybe project repo path?)
DEFAULT_RESULTS_SUBDIR = "results"          # stored below remote workdir/job_id
DEFAULT_POLL_INTERVAL_S = 10.0               # initial poll interval
MAX_POLL_INTERVAL_S = 120.0                  # exponential backoff cap
DEFAULT_TIMEOUT_S = 8 * 60 * 60              # 8h default timeout (TODO: adjust)
ARTIFACT_STDOUT = "stdout.log"
ARTIFACT_STDERR = "stderr.log"
ARTIFACT_META = "meta.json"                 # internal metadata we generate
ARTIFACT_RESULT_JSON = "train_results.json" # TODO: ensure your training script outputs this
DONE_SENTINEL = "DONE"                      # content of done marker file

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
class JobStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"  # (future TODO: implement cancellation)

@dataclass
class RemoteJob:
    host: str
    user: str
    key_filename: Optional[str]
    command: str                      # training command executed remotely
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    remote_workdir: str = DEFAULT_REMOTE_WORKDIR
    status: str = JobStatus.PENDING
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
    backoff_factor: float = 1.4
    local_artifact_dir: Path = field(default_factory=lambda: Path("remote_runs"))
    result_payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # internal fields not typically set by caller
    _pid: Optional[int] = None
    _thread: Optional[threading.Thread] = None

    # Derived paths
    def remote_job_dir(self) -> str:
        return f"{self.remote_workdir}/{self.job_id}"

    def remote_stdout_path(self) -> str:
        return f"{self.remote_job_dir()}/{ARTIFACT_STDOUT}"

    def remote_stderr_path(self) -> str:
        return f"{self.remote_job_dir()}/{ARTIFACT_STDERR}"

    def remote_done_path(self) -> str:
        return f"{self.remote_job_dir()}/done.marker"

    def remote_result_json(self) -> str:
        return f"{self.remote_job_dir()}/{ARTIFACT_RESULT_JSON}"

    def remote_meta_json(self) -> str:
        return f"{self.remote_job_dir()}/{ARTIFACT_META}"

    def local_job_dir(self) -> Path:
        return self.local_artifact_dir / self.job_id

# ---------------------------------------------------------------------------
# Remote Trainer
# ---------------------------------------------------------------------------
class RemoteTrainer:
    def __init__(
        self,
        hosts: List[str],
        user: str,
        key_filename: Optional[str] = None,
        max_parallel: int = 1,  # TODO: raise for parallel scheduling
        logger: Optional[logging.Logger] = None,
        connect_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.hosts = hosts
        self.user = user
        self.key_filename = key_filename
        self.max_parallel = max_parallel
        self.logger = logger or logging.getLogger(__name__)
        self.connect_kwargs = connect_kwargs or {}
        if key_filename:
            self.connect_kwargs.setdefault("key_filename", key_filename)
        # simple round-robin host selection
        self._host_index = 0
        self._jobs: Dict[str, RemoteJob] = {}
        self.logger.debug("RemoteTrainer initialized: hosts=%s max_parallel=%d", hosts, max_parallel)

    # ---------------------- Public API ----------------------
    def submit(self, command: str, *, host: Optional[str] = None, timeout_s: float = DEFAULT_TIMEOUT_S) -> RemoteJob:
        """Submit a training job.

        Parameters
        ----------
        command : str
            Full shell command to execute remotely. Should create the result JSON file.
            Example (placeholder):
                python train.py --config mycfg.json --out train_results.json
            TODO: inject environment activation, dataset path exports, etc.
        host : Optional[str]
            Specific host to target; if None uses round-robin.
        timeout_s : float
            Max wall-clock seconds before marking TIMEOUT.
        """
        chosen_host = host or self._next_host()
        job = RemoteJob(host=chosen_host, user=self.user, key_filename=self.key_filename, command=command, timeout_s=timeout_s)
        self._jobs[job.job_id] = job
        self.logger.info("Submitting job %s to %s", job.job_id, chosen_host)
        job._thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        job._thread.start()
        return job

    def list_jobs(self) -> List[RemoteJob]:
        return list(self._jobs.values())

    def fetch_all(self) -> None:
        """Force fetch artifacts for all completed jobs (idempotent)."""
        for job in self._jobs.values():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.TIMEOUT):
                self._fetch_artifacts(job)

    def wait(self, job: RemoteJob) -> RemoteJob:
        """Block until a job completes (or fails/timeout)."""
        self.logger.debug("Waiting on job %s", job.job_id)
        if job._thread:
            job._thread.join()
        return job

    def wait_all(self) -> List[RemoteJob]:
        for job in self._jobs.values():
            self.wait(job)
        return self.list_jobs()

    # ---------------------- Internals ----------------------
    def _next_host(self) -> str:
        host = self.hosts[self._host_index % len(self.hosts)]
        self._host_index += 1
        return host

    def _connection(self, host: str) -> Connection:
        return Connection(host=host, user=self.user, connect_kwargs=self.connect_kwargs)

    def _run_job(self, job: RemoteJob) -> None:
        try:
            with self._connection(job.host) as c:
                self.logger.debug("[%s] Preparing remote directories for job %s", job.host, job.job_id)
                # create working dir structure
                c.run(f"mkdir -p {job.remote_job_dir()}")
                # build wrapped command
                wrapped = self._wrap_command(job)
                self.logger.debug("[%s] Launch command: %s", job.host, wrapped)
                job.status = JobStatus.RUNNING
                job.started_at = time.time()
                # launch
                result = c.run(wrapped, hide=True, warn=True)
                if result.exited != 0:
                    self.logger.warning("[%s] Launch script exited code %s (this may still be ok if backgrounded)", job.host, result.exited)
                # polling loop
                self._poll_until_done(c, job)
        except Exception as e:  # broad catch to ensure status recorded
            job.status = JobStatus.FAILED
            job.error = repr(e)
            job.ended_at = time.time()
            self.logger.exception("Job %s failed: %s", job.job_id, e)
        finally:
            # attempt fetch on terminal states
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.TIMEOUT):
                try:
                    with self._connection(job.host) as c2:
                        self._fetch_artifacts(job, conn=c2)
                except Exception as fe:
                    self.logger.warning("Artifact fetch for job %s failed: %s", job.job_id, fe)

    def _wrap_command(self, job: RemoteJob) -> str:
        # We background the actual training process so that Fabric can return
        # quickly and we handle polling ourselves.
        # TODO: Insert environment activation (conda / venv) if needed, e.g. `source ~/env/bin/activate &&`.
        train_cmd = job.command
        remote_dir = job.remote_job_dir()
        stdout = job.remote_stdout_path()
        stderr = job.remote_stderr_path()
        done = job.remote_done_path()
        meta = job.remote_meta_json()
        result_json = job.remote_result_json()
        # Write meta file early
        meta_json_inline = json.dumps({
            "job_id": job.job_id,
            "submitted_at": job.submitted_at,
            "host": job.host,
            "command": train_cmd,
            # TODO: add git commit hash, search config, etc.
        })
        # Use bash heredoc to create meta + launch background process
        wrapped = (
            f"bash -lc 'set -euo pipefail; "
            f"mkdir -p {remote_dir}; "
            f"python - <<\"PYEOF\" > /dev/null 2>&1 || true\nimport json,sys\nopen(\"{meta}\",\"w\").write({meta_json_inline!r})\nPYEOF\n"
            f"( {{ {train_cmd}; }} > {stdout} 2> {stderr}; echo {DONE_SENTINEL!r} > {done} ) & echo $! > {remote_dir}/pid'"
        )
        return wrapped

    def _poll_until_done(self, c: Connection, job: RemoteJob) -> None:
        interval = job.poll_interval_s
        while True:
            # Check timeout
            if job.started_at and (time.time() - job.started_at) > job.timeout_s:
                job.status = JobStatus.TIMEOUT
                job.ended_at = time.time()
                self.logger.error("Job %s timed out after %.1fs", job.job_id, job.timeout_s)
                return
            try:
                # Check if done marker exists
                res = c.run(f"test -f {job.remote_done_path()} && echo 1 || echo 0", hide=True)
                done_flag = res.stdout.strip() == "1"
                if done_flag:
                    job.status = JobStatus.COMPLETED
                    job.ended_at = time.time()
                    self.logger.info("Job %s completed in %.1fs", job.job_id, job.ended_at - (job.started_at or job.submitted_at))
                    # Attempt to parse result JSON if present
                    try:
                        r = c.run(f"python - <<'PYEOF'\nimport json,sys,os\nf=\"{job.remote_result_json()}\"\n\nprint(open(f).read() if os.path.exists(f) else '')\nPYEOF", hide=True, warn=True)
                        txt = r.stdout.strip()
                        if txt:
                            job.result_payload = json.loads(txt)
                    except Exception as pe:
                        self.logger.debug("Job %s: result parse failed (%s)", job.job_id, pe)
                    return
                # Not done yet; optionally retrieve partial stdout tail for debug every few loops
                if int(time.time()) % 300 < interval:  # every ~5m boundary
                    try:
                        tail_cmd = f"bash -lc 'tail -n 20 {job.remote_stdout_path()} 2>/dev/null || true'"
                        tail = c.run(tail_cmd, hide=True, warn=True).stdout
                        self.logger.debug("[%s:%s] tail stdout:\n%s", job.host, job.job_id, tail)
                    except Exception:
                        pass
            except Exception as e:
                self.logger.warning("Polling error for job %s: %s", job.job_id, e)
            time.sleep(interval)
            interval = min(MAX_POLL_INTERVAL_S, interval * job.backoff_factor)

    def _fetch_artifacts(self, job: RemoteJob, conn: Optional[Connection] = None) -> None:
        local_dir = job.local_job_dir()
        local_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug("Fetching artifacts for job %s -> %s", job.job_id, local_dir)
        try:
            c = conn or self._connection(job.host)
            with (c if conn else c) as cc:  # consistent interface
                for fname in [ARTIFACT_STDOUT, ARTIFACT_STDERR, ARTIFACT_META, ARTIFACT_RESULT_JSON, "pid", "done.marker"]:
                    remote_path = f"{job.remote_job_dir()}/{fname}"
                    try:
                        cc.get(remote_path, str(local_dir / fname), preserve_mode=False)
                    except Exception:
                        # missing artifacts are expected for some names
                        continue
                # Write a local job status manifest
                manifest = {
                    "job_id": job.job_id,
                    "host": job.host,
                    "status": job.status,
                    "error": job.error,
                    "started_at": job.started_at,
                    "ended_at": job.ended_at,
                    "duration_s": (job.ended_at - job.started_at) if job.started_at and job.ended_at else None,
                    "command": job.command,
                    "result_payload": job.result_payload,
                }
                (local_dir / "local_manifest.json").write_text(json.dumps(manifest, indent=2))
        except Exception as e:
            self.logger.warning("Artifact fetch encountered error for job %s: %s", job.job_id, e)

# ---------------------------------------------------------------------------
# Helper / Convenience
# ---------------------------------------------------------------------------
def build_training_command(
    script: str = "train.py",
    config_path: Optional[str] = None,
    out_json: str = ARTIFACT_RESULT_JSON,
    extra_args: Optional[List[str]] = None,
    python_exec: str = "python",
) -> str:
    """Construct a training command.

    NOTE: This is a simple formatter; adapt as needed for your real training.

    TODOs / Questions:
      - Do you use distributed launch (torchrun / mpirun)?
      - Do you need environment variables (CUDA_VISIBLE_DEVICES, data paths)?
      - Should we pin number of threads / seeds?
    """
    parts = [python_exec, script]
    if config_path:
        parts += ["--config", config_path]
    parts += ["--out", out_json]
    if extra_args:
        parts += extra_args
    return " ".join(parts)


