"""Deployment panel (v2.1 item 4).

Local mode: one click starts coordinator + active + N passives on this machine with the chosen saved dataset,
certificates and dashboard ports, by reusing bench/harness.LocalDeployment (the same launcher the benchmarks
use, and the Python equivalent of scripts/run_local.sh). Each node runs in its own process group, so stopping a
deployment leaves no orphan processes.

Multi-machine mode: a zip bundle per machine with only what that machine needs - its own certificate and key
plus the CA certificate, its own data file and party manifest, and the exact command to run. The root CA key
(`ca.key`) never goes into any bundle, and no party gets another party's data or key. The panel then watches
the coordinator's party list as nodes register.

One active party per session (PRD FR-1): the coordinator refuses a second live active party. Several active
parties are supported the honest way: run several deployments side by side (each "slot" uses its own port
block), i.e. one coordinator, one session and one label owner each. A single coordinator that runs concurrent
sessions with different active parties is future work (its session state is one-current-session by design).
"""
from __future__ import annotations

import atexit
import json
import os
import re
import sys
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SLOT_PORTS = {"grpc": 50050, "ui": 8080, "stride": 100}          # slot 0 = the documented defaults
ACTIVE_NOTE = ("Exactly one active (label-owning) party per session: PRD FR-1, enforced by the coordinator. "
               "To run several active parties, start several deployments (slots); each gets its own coordinator, "
               "ports and session.")
HOST_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$|^\[?[0-9a-fA-F:]+\]?$")


def _harness():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from bench import harness
    return harness


def slot_ports(slot: int) -> dict:
    off = SLOT_PORTS["stride"] * int(slot)
    return {"grpc": SLOT_PORTS["grpc"] + off, "ui": SLOT_PORTS["ui"] + (0 if slot == 0 else 10000 + off)}


def dataset_parties(data_dir: str) -> dict:
    """{'active': [...], 'passive': [...]} from the party manifests (or v1 CSV names)."""
    act, pas = [], []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith(".manifest.json"):
            try:
                m = json.load(open(os.path.join(data_dir, f)))
            except Exception:
                continue
            (act if m.get("role") == "active" else pas).append(m["name"])
    if not act and os.path.exists(os.path.join(data_dir, "active.csv")):
        act = ["active"]
        pas = sorted(f[:-4] for f in os.listdir(data_dir) if re.match(r"passive-\d+\.csv$", f))
    pas.sort(key=lambda n: int(n.split("-")[-1]) if n.split("-")[-1].isdigit() else 0)
    return {"active": act, "passive": pas}


class DeploymentManager:
    def __init__(self, workspace, session_config: str | None = None):
        self.ws = workspace
        self.session_config = session_config or os.path.join(ROOT, "configs", "session.yaml")
        self.local: dict[str, object] = {}       # name -> LocalDeployment
        self.meta: dict[str, dict] = {}
        self._tokens: dict[str, str] = {}        # multi-machine coordinator tokens: kept server-side, never in views
        self.cmd_prefix = None                    # tests swap in a stand-in node program; None = python -m vfl.cli
        atexit.register(self.stop_all)

    # ------------------------------------------------------------ local
    def launch_local(self, name: str, dataset: str, n_passive: int, tls: str = "static", certs: str | None = None,
                     slot: int = 0, lab_mode: bool = False, ui_token: str | None = None,
                     session_config: str | None = None, cmd_prefix=None, rtt_relay_ms=None) -> dict:
        self.ws.check_name(name)
        self.ws.check_name(dataset)                       # v2.2 joined raw names into paths (RT-16)
        if certs:
            self.ws.check_name(certs)
        if name in self.local and any(p.poll() is None for p in self.local[name].procs.values()):
            raise ValueError(f"deployment {name!r} is already running; stop it first")
        n_passive = int(n_passive)
        if not 1 <= n_passive <= 8:
            raise ValueError("passive parties: 1-8")
        data_dir = self.ws.path("datasets", dataset)
        if not os.path.isdir(data_dir):
            raise ValueError(f"no saved dataset {dataset!r}; prepare one in Data preparation")
        parts = dataset_parties(data_dir)
        if len(parts["active"]) != 1:
            raise ValueError(f"dataset {dataset!r} must hold exactly one active party file (FR-1)")
        if n_passive > len(parts["passive"]):
            raise ValueError(f"dataset {dataset!r} has {len(parts['passive'])} passive part(ies); "
                             f"prepare it with more parties to run {n_passive}")
        if tls not in ("static", "insecure"):
            raise ValueError("transport: static mTLS certificates or insecure (local debugging only)")
        used = {m["slot"] for k, m in self.meta.items() if k != name and self._running(k)}
        if int(slot) in used:
            raise ValueError(f"slot {slot} is in use by another running deployment; pick another slot")
        ports = slot_ports(int(slot))
        certs_dir = self.ws.path("certs", certs) if certs else None
        if tls == "static" and certs_dir and not os.path.exists(os.path.join(certs_dir, "ca.crt")):
            raise ValueError(f"certificate set {certs!r} has no ca.crt")
        H = _harness()
        dep = H.LocalDeployment(self.ws.path("deployments", name), data_dir, n_passive, tls=(tls == "static"),
                                base_port=ports["grpc"], ui_base=ports["ui"], certs_dir=certs_dir,
                                session_config=session_config or self.session_config, ui_token=ui_token,
                                lab_mode=lab_mode, cmd_prefix=cmd_prefix or self.cmd_prefix)
        relay = None
        if rtt_relay_ms not in (None, "", "off"):
            rtt = float(rtt_relay_ms)
            if not 0 <= rtt <= 2000:
                raise ValueError("RTT relay: 0-2000 ms")
            relay = ports["grpc"] + 90                    # passives reach the active party through `vfl.cli netem`
            dep.active_advertise = f"127.0.0.1:{relay}"
            dep.spawn_netem(relay, dep.ports["active"], rtt, stats_path=os.path.join(dep.workdir, "netem.json"))
        dep.start(wait=False)
        self.local[name] = dep
        self.meta[name] = {"name": name, "mode": "local", "dataset": dataset, "n_passive": n_passive, "tls": tls,
                           "certs": certs or ("(generated)" if tls == "static" else None), "slot": int(slot),
                           "lab_mode": lab_mode, "started": time.time(), "coordinator_url": dep.url,
                           "coordinator": f"127.0.0.1:{dep.ports['coord']}",
                           "active": f"127.0.0.1:{dep.ports['active']}", "certs_dir": dep.certs,
                           "data_dir": data_dir, "hosts": ["127.0.0.1", "localhost"],
                           "rtt_relay_ms": float(rtt_relay_ms) if relay else None,
                           "relay": f"127.0.0.1:{relay}" if relay else None}
        self._write_plan(name)
        return self.view(name)

    def _write_plan(self, name):
        """deployment.json: what is running where. The standalone attack tool accepts it as its target list."""
        m = self.meta[name]
        plan = {"name": name, "coordinator": m.get("coordinator"), "active": m.get("active"),
                "coordinator_url": m.get("coordinator_url"), "hosts": m.get("hosts", []), "certs": m.get("certs_dir"),
                "data_dir": m.get("data_dir"), "log_dirs": [os.path.join(self.ws.path("deployments", name), "logs")]}
        d = self.ws.path("deployments", name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "deployment.json"), "w") as fh:
            json.dump(plan, fh, indent=2)

    def _running(self, name) -> bool:
        dep = self.local.get(name)
        return bool(dep) and any(p.poll() is None for p in dep.procs.values())

    def view(self, name: str) -> dict:
        m = dict(self.meta.get(name) or {})
        if not m:
            raise KeyError(f"unknown deployment {name}")
        dep = self.local.get(name)
        if dep is not None:
            m["nodes"] = dep.status()
            m["running"] = self._running(name)
            m["registered"] = self._registered(m.get("coordinator_url"))
        elif m.get("mode") == "multi-machine":
            m["registered"] = self._registered(m.get("coordinator_url"), self._tokens.get(name))
        return m

    def _registered(self, url, token=None):
        if not url:
            return None
        try:
            req = urllib.request.Request(url.rstrip("/") + "/api/state",
                                         headers={"Authorization": f"Bearer {token}"} if token else {})
            with urllib.request.urlopen(req, timeout=1.0) as r:
                st = json.loads(r.read().decode().replace("NaN", "null"))
            return [{"party_id": p["party_id"], "role": p["role"], "alive": p["alive"], "status": p["status"],
                     "hostname": p.get("hostname"), "ui_url": p.get("ui_url")} for p in st.get("parties", [])]
        except Exception:
            return None

    def list(self) -> list[dict]:
        out = []
        for n in list(self.meta):
            try:
                out.append(self.view(n))
            except KeyError:
                pass
        return out

    def stop(self, name: str) -> dict:
        dep = self.local.get(name)
        if not dep:
            raise KeyError(f"unknown deployment {name}")
        dep.stop()
        return self.view(name)

    def restart_node(self, name: str, node: str) -> dict:
        dep = self.local.get(name)
        if not dep:
            raise KeyError(f"unknown deployment {name}")
        dep.restart(node)
        return self.view(name)

    def stop_node(self, name: str, node: str) -> dict:
        dep = self.local.get(name)
        if not dep:
            raise KeyError(f"unknown deployment {name}")
        dep.stop_node(node)
        return self.view(name)

    def log(self, name: str, node: str, n: int = 200) -> str:
        dep = self.local.get(name)
        if not dep:
            raise KeyError(f"unknown deployment {name}")
        if node not in {x["name"] for x in dep.plan()}:
            raise KeyError(f"unknown node {node}")
        return dep.log_tail(node, n)

    def stop_all(self):
        for dep in list(self.local.values()):
            try:
                dep.stop()
            except Exception:
                pass

    # ------------------------------------------------------------ multi-machine
    def make_bundles(self, name: str, dataset: str, certs: str, n_passive: int, coordinator_host: str,
                     active_host: str, passive_hosts: list[str], tls: str = "static",
                     grpc_port: int = 50050, ui_port: int = 8080) -> dict:
        """One zip per machine, plus the exact run command for each."""
        self.ws.check_name(name)
        self.ws.check_name(dataset)                       # v2.2 joined raw names into paths (RT-16)
        self.ws.check_name(certs)
        if tls != "static":
            # machines on a LAN/the internet without mTLS have no identity at all: anyone can join (RT-20)
            raise ValueError("multi-machine bundles always use mTLS; insecure mode is for one machine only")
        n_passive = int(n_passive)
        hosts = [coordinator_host, active_host] + list(passive_hosts or [])
        if len(passive_hosts or []) != n_passive:
            raise ValueError(f"give one host per passive party ({n_passive})")
        for h in hosts:
            if not h or not HOST_RE.match(h):
                raise ValueError(f"bad host name or IP {h!r}")
        data_dir = self.ws.path("datasets", dataset)
        parts = dataset_parties(data_dir) if os.path.isdir(data_dir) else {"active": [], "passive": []}
        if len(parts["active"]) != 1 or len(parts["passive"]) < n_passive:
            raise ValueError(f"dataset {dataset!r} needs one active and at least {n_passive} passive party files")
        cdir = self.ws.path("certs", certs) if certs else None
        if tls == "static" and (not cdir or not os.path.exists(os.path.join(cdir, "ca.crt"))):
            raise ValueError("choose a certificate set made for these parties (Data preparation > Certificates)")
        pki = bool(cdir) and os.path.exists(os.path.join(cdir, "pki.json"))
        out_dir = self.ws.path("deployments", name, "bundles")
        tokens: dict[str, str] = {}
        os.makedirs(out_dir, exist_ok=True)
        tls_arg = "--tls-dir certs" if tls == "static" else "--insecure"
        coord = f"{coordinator_host}:{grpc_port}"
        # the coordinator dashboard listens on the LAN and can start/stop sessions: never without a token
        import secrets
        ui_token = secrets.token_urlsafe(18)
        H = _harness()
        dep = H.LocalDeployment(self.ws.path("deployments", name), data_dir, n_passive)
        nodes = [{"name": "coordinator", "host": coordinator_host, "ui": ui_port,
                  "cmd": f"python -m vfl.cli coordinator --listen 0.0.0.0:{grpc_port} --ui-port {ui_port} "
                         f"--ui-host 0.0.0.0 --config configs/session.yaml {tls_arg}",
                  "env": {"VFL_UI_TOKEN": ui_token},
                  "files": [("configs/session.yaml", self.session_config)]},
                 {"name": "active", "host": active_host, "ui": ui_port + 1,
                  "cmd": f"python -m vfl.cli active --id active --data data/{os.path.basename(dep.party_file('active'))} "
                         f"--manifest data/active.manifest.json --coordinator {coord} --listen 0.0.0.0:{grpc_port + 1} "
                         f"--advertise {active_host}:{grpc_port + 1} --ui-host 0.0.0.0 --ui-port {ui_port + 1} "
                         f"--public-ui-url http://{active_host}:{ui_port + 1} {tls_arg}",
                  "files": []}]
        nodes[1]["env"] = {"VFL_UI_TOKEN": secrets.token_urlsafe(18)}   # party dashboards are network-facing too
        for i, h in enumerate(passive_hosts, 1):
            nodes.append({"name": f"passive-{i}", "host": h, "ui": ui_port + 1 + i,
                          "env": {"VFL_UI_TOKEN": secrets.token_urlsafe(18)},
                          "cmd": f"python -m vfl.cli passive --id passive-{i} "
                                 f"--data data/{os.path.basename(dep.party_file(f'passive-{i}'))} "
                                 f"--manifest data/passive-{i}.manifest.json --coordinator {coord} --ui-host 0.0.0.0 "
                                 f"--ui-port {ui_port + 1 + i} --public-ui-url http://{h}:{ui_port + 1 + i} {tls_arg}",
                          "files": []})
        for n in nodes:
            files = list(n["files"])
            if n["name"] != "coordinator":
                files += [(f"data/{os.path.basename(dep.party_file(n['name']))}", dep.party_file(n["name"])),
                          (f"data/{n['name']}.manifest.json", os.path.join(data_dir, f"{n['name']}.manifest.json"))]
            if tls == "static":
                files.append(("certs/ca.crt", os.path.join(cdir, "ca.crt")))
                files += [(f"certs/{n['name']}.{ext}", os.path.join(cdir, f"{n['name']}.{ext}")) for ext in ("crt", "key")]
                if pki and n["name"] == "coordinator":       # the coordinator issues short-lived certs in pki mode
                    files += [(f"certs/{f}", os.path.join(cdir, f)) for f in
                              ("intermediate.crt", "intermediate.key", "pki.json", "issued_static.json")]
            files = [(arc, src) for arc, src in files if os.path.exists(src)]
            if any(os.path.basename(arc) == "ca.key" or "/root/" in arc for arc, _ in files):
                raise RuntimeError("refusing to bundle a CA private key")
            node_env = n.pop("env", None) or {}
            tokens[n["name"]] = node_env.get("VFL_UI_TOKEN")
            env = "".join(f"export {k}='{v}'\n" for k, v in node_env.items())
            run = ("#!/usr/bin/env bash\n# Run on " + n["host"] + " from the vfl-system folder after unzipping this "
                   "bundle into it.\n# The dashboard token is passed in the environment, never on the command line "
                   "(command lines are visible to every local user).\nset -euo pipefail\numask 077\n"
                   + env + n["cmd"] + "\n")
            readme = (f"{n['name']} bundle for {n['host']}\n\nContains only what this machine needs. "
                      "Never copy ca.key (the CA's private key) to any machine.\n\nRun:\n  " + n["cmd"] + "\n")
            zpath = os.path.join(out_dir, f"{n['name']}.zip")
            fd = os.open(zpath + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.close(fd)                                  # a bundle holds a private key: owner-only file (RT-11)
            with zipfile.ZipFile(zpath + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
                for arc, src in files:
                    z.write(src, arc)
                z.writestr(f"run-{n['name']}.sh", run)
                z.writestr(f"README-{n['name']}.txt", readme)
            os.chmod(zpath + ".tmp", 0o600)
            os.replace(zpath + ".tmp", zpath)
            n["bundle"] = os.path.relpath(zpath, self.ws.root)
            n["contents"] = sorted([a for a, _ in files] + [f"run-{n['name']}.sh", f"README-{n['name']}.txt"])
            n.pop("files")
        url = f"http://{coordinator_host}:{ui_port}"
        for n in nodes:                                   # views are readable without the token: never put it there
            assert all(t not in n["cmd"] for t in tokens.values() if t)
        self._tokens[name] = ui_token
        self.meta[name] = {"name": name, "mode": "multi-machine",
                           "dataset": dataset, "n_passive": n_passive,
                           "tls": tls, "certs": certs, "slot": None, "coordinator_url": url, "coordinator": coord,
                           "active": f"{active_host}:{grpc_port + 1}", "hosts": sorted(set(hosts)), "machines": nodes,
                           "started": time.time(), "certs_dir": cdir, "data_dir": data_dir}
        self._write_plan(name)
        return {**self.view(name), "ui_token": ui_token,                   # shown once, to the operator who asked
                "ui_tokens": tokens,
                "ui_token_note": "dashboard tokens, shown once: the coordinator's is the operator token (start/stop); "
                                 "each party's opens that party's dashboard (?token=...). Also inside each bundle."}
