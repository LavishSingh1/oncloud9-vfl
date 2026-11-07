"""Helpers for benchmark/test scripts: launch a local deployment as separate processes,
or drive an existing (LAN / cloud) deployment through the coordinator's HTTP API."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def http(method, url, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode().replace("NaN", "null"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{e.code}: {e.read().decode()}") from None


class Coordinator:
    """Client for a running coordinator dashboard/API."""

    def __init__(self, url="http://localhost:8080"):
        self.url = url.rstrip("/")

    def state(self):
        return http("GET", f"{self.url}/api/state")

    def wait_parties(self, n_passive, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                ps = [p for p in self.state()["parties"] if p["alive"] and p["status"] in ("idle", "done", "failed")]
                if any(p["role"] == "active" for p in ps) and sum(p["role"] == "passive" for p in ps) >= n_passive:
                    return ps
            except Exception:
                pass
            time.sleep(0.5)
        raise TimeoutError("parties did not become ready")

    def start(self, config: dict, resume_from=None) -> dict:
        for i in range(40):
            try:
                return http("POST", f"{self.url}/api/sessions", {"config": config, "resume_from": resume_from})
            except RuntimeError as e:
                if "busy" not in str(e) and "still running" not in str(e) or i == 39:
                    raise
                time.sleep(0.5)

    def run(self, config: dict, resume_from=None, timeout=7200, label=None, quiet=False) -> dict:
        b = self.start(config, resume_from)
        sid = b["session_id"]
        if not quiet:
            print(f"  -> {sid} {label or ''} {json.dumps(config)[:140]}", flush=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            d = http("GET", f"{self.url}/api/sessions/{sid}")
            if d["state"] in ("COMPLETED", "FAILED", "STOPPED"):
                # give parties a moment to flush their final reports
                for _ in range(20):
                    if set(d["participants"] if isinstance(d["participants"][0], str)
                           else [p["party_id"] for p in d["participants"]]) <= set(d.get("finals", {})):
                        break
                    time.sleep(0.25)
                    d = http("GET", f"{self.url}/api/sessions/{sid}")
                if not quiet:
                    print(f"     {d['state']} in {time.time() - t0:.1f}s", flush=True)
                self.wait_idle()
                return d
            time.sleep(0.5)
        raise TimeoutError(f"session {sid} did not finish")

    def stop(self, sid):
        return http("POST", f"{self.url}/api/sessions/{sid}/stop", {})

    def wait_idle(self, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.state()
            cur = st.get("current") or {}
            if cur.get("state") not in ("STARTING", "RUNNING", "STOPPING") and \
                    all(p["status"] in ("idle", "done", "failed") for p in st["parties"] if p["alive"]):
                return
            time.sleep(0.3)
        raise TimeoutError("parties did not return to idle")


class LocalDeployment:
    """Start coordinator + active + passives as subprocesses on this machine.

    Used by the benchmarks (start() waits until every party is idle) and by the unified UI's Deployment panel
    (start(wait=False) + status()/log_tail()/restart()/stop()). Every node runs in its own process group, so
    stop() also ends anything a node started: no orphans.
    """

    def __init__(self, workdir, data_dir, n_passive=1, tls=True, base_port=52050, ui_base=18080,
                 active_advertise=None, session_config=None, env=None, certs_dir=None, active_id="active",
                 bind="127.0.0.1", ui_token=None, lab_mode=False, cmd_prefix=None, test_identities=False):
        self.workdir = os.path.abspath(workdir)
        self.data_dir = os.path.abspath(data_dir)
        self.n_passive = n_passive
        self.tls = tls
        self.ports = {"coord": base_port, "active": base_port + 1}
        self.ui = {"coord": ui_base, "active": ui_base + 1}
        self.active_advertise = active_advertise
        self.session_config = session_config or os.path.join(ROOT, "configs", "session.yaml")
        self.env = {**os.environ, **(env or {})}
        self.env["PYTHONPATH"] = os.pathsep.join(filter(None, [ROOT, self.env.get("PYTHONPATH", "")]))
        self.procs = {}
        self._certs = os.path.abspath(certs_dir) if certs_dir else None
        self.active_id = active_id
        self.bind = bind
        self.ui_token = ui_token
        if ui_token:
            # through the environment, never argv: command lines are world-readable (ps, /proc/<pid>/cmdline)
            self.env["VFL_UI_TOKEN"] = ui_token
        self.lab_mode = lab_mode
        self.test_identities = test_identities or lab_mode
        self.cmd_prefix = cmd_prefix or [PY, "-m", "vfl.cli"]       # tests swap in a fake node
        self.started_at = {}
        os.makedirs(self.workdir, exist_ok=True)

    @property
    def certs(self):
        return self._certs or os.path.join(self.workdir, "certs")

    @property
    def url(self):
        return f"http://127.0.0.1:{self.ui['coord']}"

    def _tls_args(self):
        return ["--tls-dir", self.certs] if self.tls else ["--insecure"]

    def party_file(self, name):
        """The party's data file: from its manifest (csv or npz), else <name>.csv."""
        m = os.path.join(self.data_dir, f"{name}.manifest.json")
        if os.path.exists(m):
            try:
                return os.path.join(self.data_dir, json.load(open(m))["file"])
            except Exception:
                pass
        return os.path.join(self.data_dir, f"{name}.csv")

    def plan(self):
        """Every node: name, role, argv (after `vfl.cli`), gRPC and dashboard ports."""
        common = self._tls_args() + ["--log-dir", "logs"]      # the UI token reaches the nodes as VFL_UI_TOKEN
        lab = ["--lab-mode"] if self.lab_mode else []
        coord = f"127.0.0.1:{self.ports['coord']}"
        adv = self.active_advertise or f"127.0.0.1:{self.ports['active']}"
        nodes = [{"name": "coordinator", "role": "coordinator", "grpc": self.ports["coord"], "ui": self.ui["coord"],
                  "args": ["coordinator", "--listen", f"{self.bind}:{self.ports['coord']}", "--ui-port",
                           str(self.ui["coord"]), "--config", self.session_config, "--run-dir", "runs"] + common + lab},
                 {"name": self.active_id, "role": "active", "grpc": self.ports["active"], "ui": self.ui["active"],
                  "args": ["active", "--id", self.active_id, "--data", self.party_file("active"), "--coordinator", coord,
                           "--listen", f"{self.bind}:{self.ports['active']}", "--advertise", adv,
                           "--ui-port", str(self.ui["active"]), "--ckpt-dir", "checkpoints"] + common}]
        for i in range(1, self.n_passive + 1):
            nodes.append({"name": f"passive-{i}", "role": "passive", "grpc": None, "ui": self.ui["active"] + i,
                          "args": ["passive", "--id", f"passive-{i}", "--data", self.party_file(f"passive-{i}"),
                                   "--coordinator", coord, "--ui-port", str(self.ui["active"] + i),
                                   "--ckpt-dir", "checkpoints"] + common})
        if self.lab_mode and self._certs:
            nodes[0]["args"] += ["--lab-certs", self.certs]
        return nodes

    def _spawn(self, name, args):
        out = open(os.path.join(self.workdir, f"{name}.out"), "a")
        p = subprocess.Popen(self.cmd_prefix + args, cwd=self.workdir, stdout=out, stderr=subprocess.STDOUT,
                             env=self.env, start_new_session=True)
        self.procs[name] = p
        self.started_at[name] = time.time()
        return p

    def ensure_certs(self):
        if self.tls and not os.path.exists(os.path.join(self.certs, "ca.crt")):
            # 'rogue-node' (a CA-signed non-participant for the security tests) only in lab/test certificate sets
            extra = ["--extra", "rogue-node"] if self.test_identities else []
            subprocess.check_call([PY, "-m", "vfl.cli", "certs", "--out", self.certs, "--passives",
                                   str(self.n_passive)] + extra,
                                  cwd=self.workdir, env=self.env, stdout=subprocess.DEVNULL)

    def start(self, wait=True):
        self.ensure_certs()
        nodes = self.plan()
        self._spawn(nodes[0]["name"], nodes[0]["args"])
        time.sleep(1.5 if wait else 0.5)
        for n in nodes[1:]:
            self._spawn(n["name"], n["args"])
        c = Coordinator(self.url)
        if wait:
            c.wait_parties(self.n_passive, timeout=90)
        return c

    # ---- UI support: per-node status, logs, restart
    def node_state(self, name, ui_port):
        p = self.procs.get(name)
        if p is None:
            return "stopped"
        if p.poll() is not None:
            return "failed" if p.returncode not in (0, -15) else "stopped"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{ui_port}/healthz", timeout=0.7) as r:
                if r.status == 200:
                    return "live"
        except Exception:
            pass
        return "starting"

    def status(self):
        out = []
        for n in self.plan():
            p = self.procs.get(n["name"])
            out.append({"name": n["name"], "role": n["role"], "pid": p.pid if p else None,
                        "state": self.node_state(n["name"], n["ui"]), "grpc_port": n["grpc"], "ui_port": n["ui"],
                        "dashboard": f"http://127.0.0.1:{n['ui']}", "returncode": p.poll() if p else None,
                        "uptime_s": time.time() - self.started_at[n["name"]] if n["name"] in self.started_at else None})
        return out

    def log_tail(self, name, n=200):
        path = os.path.join(self.workdir, f"{name}.out")
        if not os.path.exists(path):
            return ""
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 200 * n))
            return b"\n".join(fh.read().splitlines()[-n:]).decode(errors="replace")

    def stop_node(self, name):
        p = self.procs.pop(name, None)
        if not p:
            return
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            p.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait(timeout=5)

    def restart(self, name):
        node = next((n for n in self.plan() if n["name"] == name), None)
        if node is None:
            raise KeyError(f"unknown node {name}")
        self.stop_node(name)
        return self._spawn(name, node["args"])

    def spawn_netem(self, listen_port, target_port, rtt_ms, stats_path=None, capture_path=None):
        args = ["netem", "--listen", f"127.0.0.1:{listen_port}", "--target", f"127.0.0.1:{target_port}",
                "--rtt-ms", str(rtt_ms)]
        if stats_path:
            args += ["--stats", stats_path]
        if capture_path:
            args += ["--capture", capture_path]
        p = self._spawn("netem", args)
        wait_port("127.0.0.1", listen_port)
        return p

    def stop(self):
        for name, p in list(self.procs.items()):
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in self.procs.values():
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait(timeout=5)
        self.procs.clear()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.stop()


def wait_port(host, port, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"{host}:{port} did not open")


def ensure_data(out_dir, dataset="breast_cancer", n_passive=1, extra=None):
    if os.path.exists(os.path.join(out_dir, "active.csv")):
        return out_dir
    args = [PY, "-m", "vfl.cli", "prepare-data", "--dataset", dataset, "--passives", str(n_passive), "--out", out_dir]
    subprocess.check_call(args + (extra or []), cwd=ROOT, stdout=subprocess.DEVNULL,
                          env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH", "")]))})
    return out_dir
