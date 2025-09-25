"""
SSH-based training dispatcher for GCP internal IP workers.

Requirements:
  pip install paramiko

Assumptions:
  - Passwordless SSH (public key) from controller -> workers.
  - Python + your training environment present on the workers.
  - Each worker can run at least N jobs in parallel (controlled via max_parallel_per_worker).
  - Your training command writes a small JSON summary to result.json.

TODOs:
  - Replace TRAIN_CMD_TEMPLATE with your actual training entrypoint.
  - Optionally push your code/env via rsync or a container launch cmd.
  - If you prefer artifact storage on GCS, modify _remote_cmd to use gsutil cp.
"""

import asyncio
import concurrent.futures
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import paramiko

# ---------- Tunables ----------
REMOTE_BASE = "/tmp/training_jobs"   # remote job root
REMOTE_PYTHON = "python3"            # which python to use on worker
LOG_TAIL_KB = 256                    # when retrieving failure logs
SSH_CONNECT_TIMEOUT = 10
SSH_CMD_TIMEOUT = 3600               # for short commands (not the long training)
SFTP_CHUNK = 1 << 20                 # 1MB
# Replace with your actual command. It MUST:
#   - Read config from the first arg (a JSON path)
#   - Write its final JSON summary to the second arg (result path)
# Example demo just sleeps and writes a file.
TRAIN_CMD_TEMPLATE = (
    '{py} - <<\'PYEOF\' "{cfg}" "{out}"\n'
    "import json,sys,time,random; cfg=sys.argv[1]; out=sys.argv[2];\n"
    "time.sleep(3 + int(random.random()*3));\n"
    "json.dump({'val_loss': 3.14, 'throughput_tps': 512, 'energy_mJ_per_token': 7.2}, open(out,'w'))\n"
    "PYEOF\n"
)

# --------------------------------

@dataclass
class Worker:
    ip: str
    user: str = "ubuntu"             # change if needed
    port: int = 22
    key_filename: Optional[str] = None  # e.g., "~/.ssh/id_rsa"

@dataclass
class JobStatus:
    job_id: str
    state: str           # queued | running | succeeded | failed
    created_at: float
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    worker_ip: Optional[str] = None
    remote_dir: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SSHClientPool:
    """Lightweight pool; creates one client per usage (keeps code simple & robust)."""
    def __init__(self):
        self._lock = asyncio.Lock()

    def _connect(self, worker: Worker) -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            worker.ip,
            username=worker.user,
            port=worker.port,
            key_filename=os.path.expanduser(worker.key_filename) if worker.key_filename else None,
            timeout=SSH_CONNECT_TIMEOUT,
            allow_agent=True,
            look_for_keys=True,
        )
        return client

    def exec(self, worker: Worker, command: str, timeout: int = SSH_CMD_TIMEOUT) -> Tuple[int, str, str]:
        client = self._connect(worker)
        try:
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            rc = stdout.channel.recv_exit_status()
            out = stdout.read().decode(errors="ignore")
            err = stderr.read().decode(errors="ignore")
            return rc, out, err
        finally:
            client.close()

    def sftp_put(self, worker: Worker, local_path: str, remote_path: str):
        client = self._connect(worker)
        try:
            sftp = client.open_sftp()
            with sftp.file(remote_path, "wb") as rf, open(local_path, "rb") as lf:
                while True:
                    data = lf.read(SFTP_CHUNK)
                    if not data:
                        break
                    rf.write(data)
            sftp.close()
        finally:
            client.close()

    def sftp_get(self, worker: Worker, remote_path: str, local_path: str):
        client = self._connect(worker)
        try:
            sftp = client.open_sftp()
            with sftp.file(remote_path, "rb") as rf, open(local_path, "wb") as lf:
                while True:
                    data = rf.read(SFTP_CHUNK)
                    if not data:
                        break
                    lf.write(data)
            sftp.close()
        finally:
            client.close()

class SSHDispatcher:
    def __init__(
        self,
        workers: Sequence[Worker],
        *,
        max_parallel_per_worker: int = 1,
        poll_every_s: int = 15,
        job_timeout_s: int = 24*3600,
        local_results_dir: str = "results_ssh",
    ):
        if not workers:
            raise ValueError("No workers provided")

        self.workers = list(workers)
        self.max_parallel_per_worker = max_parallel_per_worker
        self.poll_every_s = poll_every_s
        self.job_timeout_s = job_timeout_s
        self.pool = SSHClientPool()
        self.local_results_dir = Path(local_results_dir)
        self.local_results_dir.mkdir(parents=True, exist_ok=True)

        # semaphores per worker to limit concurrency
        self._sems: Dict[str, asyncio.Semaphore] = {
            w.ip: asyncio.Semaphore(self.max_parallel_per_worker) for w in self.workers
        }

        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(8, len(workers)*2))

    async def _run_in_thread(self, fn, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._thread_pool, lambda: fn(*args, **kwargs))

    def _remote_safe_path(self, *parts) -> str:
        return "/".join(p.strip("/").replace("..", "__") for p in parts)

    def _build_remote_commands(self, job_dir: str, cfg_path: str, out_path: str, log_path: str) -> str:
        # Ensure dirs & launch detached with nohup; write PID file.
        pid_path = f"{job_dir}/pid"
        start_ts = f"{job_dir}/started_at"
        end_ts = f"{job_dir}/ended_at"
        train_cmd = TRAIN_CMD_TEMPLATE.format(py=REMOTE_PYTHON, cfg=cfg_path, out=out_path)

        # We use a small wrapper to echo timestamps and capture return code.
        # The training runs in background; stdout/stderr -> log.
        cmd = f"""
set -e
mkdir -p "{job_dir}"
date +%s > "{start_ts}" || true
( nohup bash -lc '{train_cmd}' > "{log_path}" 2>&1; echo $? > "{job_dir}/rc"; date +%s > "{end_ts}" ) &
echo $! > "{pid_path}"
"""
        return cmd

    async def _submit_one(self, worker: Worker, config: Dict[str, Any]) -> JobStatus:
        job_id = str(uuid.uuid4())
        job_dir = self._remote_safe_path(REMOTE_BASE, job_id)
        remote_cfg = f"{job_dir}/config.json"
        remote_out = f"{job_dir}/result.json"
        remote_log = f"{job_dir}/train.log"

        # 1) ensure base dir exists
        await self._run_in_thread(self.pool.exec, worker, f'mkdir -p "{REMOTE_BASE}" && echo ok', 30)

        # 2) upload config
        cfg_tmp = Path(f".tmp_cfg_{job_id}.json")
        cfg_tmp.write_text(json.dumps(config, indent=2))
        try:
            await self._run_in_thread(self.pool.sftp_put, worker, str(cfg_tmp), remote_cfg)
        finally:
            try:
                cfg_tmp.unlink()
            except FileNotFoundError:
                pass

        # 3) launch detached
        cmd = self._build_remote_commands(job_dir, remote_cfg, remote_out, remote_log)
        rc, out, err = await self._run_in_thread(self.pool.exec, worker, cmd, 30)
        if rc != 0:
            return JobStatus(job_id, "failed", time.time(), worker_ip=worker.ip, remote_dir=job_dir, error=f"submit rc={rc} err={err} out={out}")

        return JobStatus(job_id, "running", time.time(), started_at=time.time(), worker_ip=worker.ip, remote_dir=job_dir)

    async def _poll_until_done(self, job: JobStatus) -> JobStatus:
        assert job.remote_dir and job.worker_ip
        worker = next(w for w in self.workers if w.ip == job.worker_ip)
        start = time.time()

        remote_out = f"{job.remote_dir}/result.json"
        remote_rc  = f"{job.remote_dir}/rc"
        remote_end = f"{job.remote_dir}/ended_at"
        remote_log = f"{job.remote_dir}/train.log"

        while True:
            # check timeout
            if (time.time() - start) > self.job_timeout_s:
                job.state = "failed"
                job.ended_at = time.time()
                job.error = f"timeout after {self.job_timeout_s}s"
                return job

            # check result existence
            rc, out, err = await self._run_in_thread(self.pool.exec, worker, f'test -f "{remote_out}" && echo done || echo no', 20)
            if rc == 0 and "done" in out:
                # fetch result
                local_result = self.local_results_dir / f"{job.job_id}_result.json"
                await self._run_in_thread(self.pool.sftp_get, worker, remote_out, str(local_result))
                try:
                    job.result = json.loads(local_result.read_text())
                except Exception as e:
                    job.result = None
                    job.error = f"result.json parse error: {e}"
                job.state = "succeeded" if job.result is not None else "failed"
                # fetch log too
                local_log = self.local_results_dir / f"{job.job_id}_train.log"
                await self._run_in_thread(self.pool.sftp_get, worker, remote_log, str(local_log))
                # end timestamp
                rc2, out2, _ = await self._run_in_thread(self.pool.exec, worker, f'cat "{remote_end}" 2>/dev/null || date +%s', 10)
                try:
                    job.ended_at = float(out2.strip())
                except Exception:
                    job.ended_at = time.time()
                return job

            # if rc file exists with non-zero, mark failed & pull tail of logs
            rc2, out2, _ = await self._run_in_thread(self.pool.exec, worker, f'test -f "{remote_rc}" && cat "{remote_rc}" || echo NA', 10)
            if rc2 == 0 and out2.strip().isdigit() and int(out2.strip()) != 0:
                job.state = "failed"
                job.ended_at = time.time()
                # fetch last KB of logs for debugging
                local_log = self.local_results_dir / f"{job.job_id}_train.log"
                await self._run_in_thread(self.pool.sftp_get, worker, remote_log, str(local_log))
                tail = local_log.read_bytes()[-LOG_TAIL_KB*1024:].decode(errors="ignore")
                job.error = f"remote return code={out2.strip()}\n--- log tail ---\n{tail}"
                return job

            await asyncio.sleep(self.poll_every_s)

    async def _run_job(self, worker: Worker, cfg: Dict[str, Any]) -> JobStatus:
        sem = self._sems[worker.ip]
        async with sem:
            submitted = await self._submit_one(worker, cfg)
        if submitted.state == "failed":
            return submitted
        return await self._poll_until_done(submitted)

    async def dispatch(self, configs: Sequence[Dict[str, Any]]) -> List[JobStatus]:
        # simple round-robin
        assignments: List[Tuple[Worker, Dict[str, Any]]] = [
            (self.workers[i % len(self.workers)], cfg) for i, cfg in enumerate(configs)
        ]
        tasks = [asyncio.create_task(self._run_job(w, c)) for (w, c) in assignments]
        results: List[JobStatus] = []
        for t in asyncio.as_completed(tasks):
            try:
                res = await t
                results.append(res)
            except Exception as e:
                results.append(JobStatus(job_id=str(uuid.uuid4()), state="failed", created_at=time.time(), error=str(e)))
        return results
