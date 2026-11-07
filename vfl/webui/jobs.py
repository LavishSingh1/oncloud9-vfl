"""Background jobs for the unified UI: each job is a subprocess (usually `python -m vfl.cli ...`, so the UI
and the CLI run exactly the same code) with a live log tail, progress, cancel, rerun and an optional queue.

Jobs run in their own process group, so cancel/stop also reaches anything they started.
"""
from __future__ import annotations

import collections
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def env_with_root(extra: dict | None = None) -> dict:
    env = {**os.environ, **(extra or {})}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [ROOT, env.get("PYTHONPATH", "")]))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def cli_cmd(args: list[str]) -> list[str]:
    return [sys.executable, "-m", "vfl.cli"] + [str(a) for a in args]


class Job:
    def __init__(self, kind: str, cmd: list[str], cwd: str, label: str = "", meta: dict | None = None,
                 on_done=None, timeout: float | None = None):
        self.id = time.strftime("%H%M%S-") + uuid.uuid4().hex[:5]
        self.kind, self.cmd, self.cwd, self.label = kind, cmd, cwd, label or kind
        self.meta = meta or {}
        self.on_done = on_done
        self.timeout = timeout
        self.status = "queued"
        self.t_queued = time.time()
        self.t_start = self.t_end = None
        self.returncode = None
        self.lines = collections.deque(maxlen=4000)
        self.proc = None
        self.result = None
        self.progress = None                 # optional "n/m" parsed by a progress hook

    def view(self, tail: int = 200) -> dict:
        return {"id": self.id, "kind": self.kind, "label": self.label, "status": self.status,
                "queued": self.t_queued, "started": self.t_start, "ended": self.t_end,
                "returncode": self.returncode, "meta": self.meta, "progress": self.progress,
                "result": self.result, "log": "".join(list(self.lines)[-tail:]) if tail else "",
                "argv": self.cmd[2:] if self.cmd[:2] == [sys.executable, "-m"] else self.cmd}


class JobManager:
    """`serial=True` turns the manager into a queue (one job at a time, in order) - used for benchmarks."""

    def __init__(self, serial: bool = False, keep: int = 200):
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.lock = threading.Lock()
        self.serial = serial
        self.keep = keep
        self._queue: collections.deque[Job] = collections.deque()
        self._worker = None

    def submit(self, kind, cmd, cwd=ROOT, label="", meta=None, on_done=None, timeout=None, env=None,
               progress=None, ok_codes=(0,)) -> Job:
        j = Job(kind, cmd, cwd, label, meta, on_done, timeout)
        j.ok_codes = tuple(ok_codes)
        j.env = env_with_root(env)
        j.progress_hook = progress
        with self.lock:
            self.jobs[j.id] = j
            self.order.append(j.id)
            while len(self.order) > self.keep:
                self.jobs.pop(self.order.pop(0), None)
            if self.serial:
                self._queue.append(j)
                if not self._worker or not self._worker.is_alive():
                    self._worker = threading.Thread(target=self._drain, daemon=True, name="job-queue")
                    self._worker.start()
                return j
        threading.Thread(target=self._run, args=(j,), daemon=True, name=f"job-{j.id}").start()
        return j

    def _drain(self):
        while True:
            with self.lock:
                if not self._queue:
                    return
                j = self._queue.popleft()
            if j.status == "cancelled":
                continue
            self._run(j)

    def _run(self, j: Job):
        j.status, j.t_start = "running", time.time()
        try:
            j.proc = subprocess.Popen(j.cmd, cwd=j.cwd, env=j.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1, start_new_session=True)
            deadline = time.time() + j.timeout if j.timeout else None
            for line in j.proc.stdout:
                j.lines.append(line)
                if j.progress_hook:
                    try:
                        p = j.progress_hook(line)
                        if p is not None:
                            j.progress = p
                    except Exception:
                        pass
                if deadline and time.time() > deadline:
                    self._kill(j)
                    j.lines.append(f"\n[timed out after {j.timeout:.0f}s]\n")
                    break
            j.returncode = j.proc.wait()
            if j.status != "cancelled":
                j.status = "done" if j.returncode in j.ok_codes else "failed"
        except Exception as e:                                   # pragma: no cover
            j.lines.append(f"\n[{type(e).__name__}: {e}]\n")
            j.status = "failed"
        if j.on_done:
            try:
                j.result = j.on_done(j)
            except Exception as e:
                j.lines.append(f"\n[post-processing failed: {type(e).__name__}: {e}]\n")
        j.t_end = time.time()                                    # set last: wait() returns the result too

    @staticmethod
    def _kill(j: Job):
        if j.proc and j.proc.poll() is None:
            try:
                os.killpg(j.proc.pid, signal.SIGTERM)
                j.proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(j.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def cancel(self, jid: str) -> dict:
        j = self.jobs.get(jid)
        if not j:
            raise KeyError(f"unknown job {jid}")
        if j.status in ("queued", "running"):
            j.status = "cancelled"
            self._kill(j)
        return j.view()

    def rerun(self, jid: str) -> Job:
        j = self.jobs.get(jid)
        if not j:
            raise KeyError(f"unknown job {jid}")
        return self.submit(j.kind, j.cmd, j.cwd, j.label + " (rerun)", dict(j.meta), j.on_done, j.timeout,
                           progress=j.progress_hook, ok_codes=j.ok_codes)

    def get(self, jid: str) -> Job:
        j = self.jobs.get(jid)
        if not j:
            raise KeyError(f"unknown job {jid}")
        return j

    def list(self, kind: str | None = None, tail: int = 0) -> list[dict]:
        with self.lock:
            js = [self.jobs[i] for i in reversed(self.order) if i in self.jobs]
        return [j.view(tail) for j in js if kind is None or j.kind == kind]

    def wait(self, jid: str, timeout: float = 600) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.jobs[jid]
            if j.status in ("done", "failed", "cancelled") and j.t_end:
                return j.view(10000)
            time.sleep(0.05)
        raise TimeoutError(jid)
