"""Machine-readable telemetry (FR-10): per-batch / per-epoch / per-run JSONL records,
resource sampling (NFR-6) and human-readable event logs for debugging.

Logging rule (NFR-7): records hold timings, counts, sizes and aggregate metrics only.
Entity ids, feature values and labels are never written.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import platform
import socket
import sys
import threading
import time

import numpy as np
import psutil


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}"


class RingHandler(logging.Handler):
    """Keeps the last N formatted log lines for the dashboard."""

    def __init__(self, n: int = 400):
        super().__init__()
        self.lines = collections.deque(maxlen=n)

    def emit(self, record):
        try:
            self.lines.append({"t": record.created, "level": record.levelname, "msg": self.format(record)})
        except Exception:  # pragma: no cover
            pass


RING = RingHandler()


def setup_logging(node: str, log_dir: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter(f"%(asctime)s %(levelname)-7s [{node}] %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(log_dir, f"{node}.events.log"))
    fh.setFormatter(fmt)
    RING.setFormatter(logging.Formatter("%(message)s"))
    for h in (sh, fh, RING):
        root.addHandler(h)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    return logging.getLogger("vfl")


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "hostname": socket.gethostname(), "cpu_count": os.cpu_count(), "numpy": np.__version__}
    for mod in ("torch", "grpc", "google.protobuf", "sklearn"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    return info


class JsonlWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", buffering=1)

    def write(self, rec: dict):
        line = json.dumps(rec, default=_json_default)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self):
        with self._lock:
            self._fh.close()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else None


class ResourceMonitor:
    """Samples this process's CPU% and RSS every `interval` seconds."""

    def __init__(self, interval: float = 1.0, writer: JsonlWriter | None = None, tags: dict | None = None):
        self.interval = interval
        self.writer = writer
        self.tags = tags or {}
        self.proc = psutil.Process()
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thr = None

    def start(self):
        self.proc.cpu_percent(None)
        self._thr = threading.Thread(target=self._run, daemon=True, name="resource-monitor")
        self._thr.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            cpu = self.proc.cpu_percent(None)
            rss = self.proc.memory_info().rss / 2 ** 20
            t = time.time()
            self.samples.append((t, cpu, rss))
            if self.writer:
                self.writer.write({**self.tags, "t": t, "cpu_percent": cpu, "rss_mib": rss})

    def stop(self) -> dict:
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2)
        return self.summary()

    def summary(self) -> dict:
        if not self.samples:
            rss = self.proc.memory_info().rss / 2 ** 20
            return {"cpu_avg_percent": None, "cpu_max_percent": None, "rss_avg_mib": rss, "rss_max_mib": rss,
                    "n_samples": 0}
        cpu = [s[1] for s in self.samples]
        rss = [s[2] for s in self.samples]
        return {"cpu_avg_percent": float(np.mean(cpu)), "cpu_max_percent": float(np.max(cpu)),
                "rss_avg_mib": float(np.mean(rss)), "rss_max_mib": float(np.max(rss)),
                "n_samples": len(self.samples)}

    def latest(self) -> dict:
        if self.samples:
            t, c, r = self.samples[-1]
            return {"cpu_percent": c, "rss_mib": r}
        return {"cpu_percent": None, "rss_mib": self.proc.memory_info().rss / 2 ** 20}


class SessionLog:
    """Per-node, per-session telemetry directory: logs/<node>/<session>/"""

    def __init__(self, root: str, node: str, session_id: str):
        self.dir = os.path.join(root, node, session_id)
        os.makedirs(self.dir, exist_ok=True)
        self.batches = JsonlWriter(os.path.join(self.dir, "batches.jsonl"))
        self.epochs = JsonlWriter(os.path.join(self.dir, "epochs.jsonl"))
        self.resources = JsonlWriter(os.path.join(self.dir, "resources.jsonl"))
        self.recent = collections.deque(maxlen=2000)

    def batch(self, rec: dict):
        self.batches.write(rec)
        self.recent.append(rec)

    def write_json(self, name: str, obj: dict):
        path = os.path.join(self.dir, name)
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=2, default=_json_default)
        return path

    def close(self):
        for w in (self.batches, self.epochs, self.resources):
            w.close()


def latency_summary(batch_ms: list[float]) -> dict:
    return {"batch_ms_p50": percentile(batch_ms, 50), "batch_ms_p95": percentile(batch_ms, 95),
            "batch_ms_mean": float(np.mean(batch_ms)) if batch_ms else None, "n": len(batch_ms)}


# ------------------------------------------------------------------ security events (v2 item 12)
LAYERS = ("network", "tls", "auth", "rate", "protocol", "data", "inference", "enrollment", "integrity")
OUTCOMES = ("blocked", "detected", "succeeded")
EVIDENCE = ("server-observed", "attacker-reported", "gate-observed", "party-reported")


class SecurityLog:
    """Append-only security.jsonl per node + in-memory feed/counters for dashboards and Prometheus.

    One record per event: time, node, source IP, claimed identity, certificate identity, attack type,
    layer, outcome (blocked | detected | succeeded), defence that acted, evidence source
    (server-observed | attacker-reported | gate-observed), method, session, detail.
    No entity ids, feature values or labels are ever written here.
    """

    def __init__(self, node: str = "node", path: str | None = None):
        self.node = node
        self.path = path
        self.lock = threading.Lock()
        self.feed = collections.deque(maxlen=2000)
        self.counts = collections.Counter()
        self.sinks = []                       # callables(rec) e.g. ship to the coordinator
        self._fh = None
        if path:
            self.open(path)

    def open(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self._fh = open(path, "a", buffering=1)

    def event(self, attack_type: str, layer: str, outcome: str, defence: str = "",
              evidence: str = "server-observed", source_ip: str | None = None, claimed: str | None = None,
              cert_identity: str | None = None, method: str | None = None, session: str | None = None,
              detail: str = "", node: str | None = None, t: float | None = None, forward: bool = True) -> dict:
        from .ids import one_line             # peer-supplied strings: one line, bounded (no forged log lines)

        def f(v, n=128):
            return None if v is None else one_line(v, n)
        rec = {"t": t or time.time(), "time": now_iso(), "node": f(node) or self.node, "source_ip": f(source_ip),
               "claimed_identity": f(claimed), "cert_identity": f(cert_identity), "attack_type": f(attack_type, 64),
               "layer": layer if layer in LAYERS else "protocol", "outcome": outcome if outcome in OUTCOMES else "detected",
               "defence": f(defence, 200) or "", "evidence": f(evidence, 32), "method": f(method, 64),
               "session": f(session, 64), "detail": one_line(detail, 500)}
        self.ingest(rec, forward=forward)
        lvl = logging.WARNING if rec["outcome"] != "blocked" or rec["attack_type"] != "duplicate_or_replay" else logging.INFO
        logging.getLogger("vfl.security").log(lvl, "SECURITY %s/%s %s from %s claimed=%s cert=%s: %s",
                                              rec["layer"], rec["attack_type"], rec["outcome"], rec["source_ip"],
                                              rec["claimed_identity"], rec["cert_identity"], rec["detail"])
        return rec

    def ingest(self, rec: dict, forward: bool = False):
        with self.lock:
            self.feed.append(rec)
            self.counts[(rec["attack_type"], rec["outcome"])] += 1
            if self._fh:
                self._fh.write(json.dumps(rec, default=_json_default) + "\n")
        if forward:
            for s in list(self.sinks):
                try:
                    s(rec)
                except Exception:  # pragma: no cover
                    pass

    def recent(self, n: int = 200) -> list[dict]:
        with self.lock:
            return list(self.feed)[-n:]

    def summary(self) -> dict:
        with self.lock:
            by_type = collections.defaultdict(dict)
            for (a, o), c in self.counts.items():
                by_type[a][o] = c
            tot = collections.Counter()
            for (a, o), c in self.counts.items():
                tot[o] += c
            return {"by_type": dict(by_type), "totals": dict(tot), "events": int(sum(self.counts.values()))}


SECURITY = SecurityLog()


def setup_security_log(node: str, log_dir: str) -> SecurityLog:
    SECURITY.node = node
    SECURITY.open(os.path.join(log_dir, "security.jsonl"))
    return SECURITY


def load_security_events(path: str) -> list[dict]:
    out = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return out
