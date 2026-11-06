"""Run the PRD evaluation plan (sec. 7) and collect everything `bench/analyze.py` needs.

Suites (comma-separated with --suites):
  e1    correctness & utility: B0/B1/local-VFL baselines + networked C1 over --seeds,
        plus a same-seed repeat for reproducibility (NFR-1, NFR-2)
  e2    network & efficiency: RTT x batch-size sweep (NFR-3..NFR-5)
  eff   efficiency variant: fp16 vs fp32 representations/gradients, paired seeds (NFR-6)
  def   inference defence on/off (and noise sweep), attack metric vs utility (FR-13, NFR-8)
  psi   DH-PSI alignment, any number of parties: correctness, time, bytes (FR-12, NFR-8)
  e5    faults: duplicate + 2 s delay (completes / times out + retries), stop + resume (NFR-10)
  sweep scaling: parties x partition x overlap, each in its own local deployment
  lopo  leave-one-party-out contribution (needs --passives >= 2)
  security  conformance suite against a dedicated deployment

Local mode (default) launches coordinator/active/passives as separate processes on this
machine; RTT is emulated with a relaying proxy that also measures wire bytes. Remote mode
(--url http://COORD:8080) drives an already running LAN/cloud deployment and emulates RTT
at the passive parties (inject_rtt_ms); baselines then need --data-dir with all CSVs.

    python bench/run_benchmarks.py --suites e1,e2,eff,def,psi,e5 --out results/bench
    python bench/analyze.py results/bench
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.harness import Coordinator, LocalDeployment, ensure_data, http  # noqa: E402
from vfl.config import load_session_config  # noqa: E402
from vfl.local_sim import run_baselines  # noqa: E402
from vfl.telemetry import environment_info  # noqa: E402

PROXY_PORT = 53099


class Runner:
    def __init__(self, a):
        self.a = a
        self.out = os.path.abspath(a.out)
        os.makedirs(os.path.join(self.out, "sessions"), exist_ok=True)
        self.manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "sessions": [],
                         "baselines": [], "env": environment_info()}
        self.index_path = os.path.join(self.out, "index.json")
        self.index = self._load_index()
        self.dep = None
        self.local = not a.url
        self.rtt_mode = a.rtt_mode or ("netem" if self.local else "app")
        if self.local:
            extra = ["--n", str(a.n), "--features", str(a.features)] if a.dataset == "synthetic" else []
            if a.partitioner:
                extra += ["--partitioner", a.partitioner]
            for flag, val in (("--partition", a.partition), ("--overlap", a.overlap), ("--binarize", a.binarize)):
                if val is not None:
                    extra += [flag, str(val)]
            self.data = ensure_data(a.data_dir or os.path.join(self.out, "data"), a.dataset, a.passives, extra or None)
            work = os.path.join(self.out, "deployment")
            self.dep = LocalDeployment(work, self.data, a.passives, tls=not a.insecure,
                                       active_advertise=f"127.0.0.1:{PROXY_PORT}" if self.rtt_mode == "netem" else None,
                                       session_config=a.config)
            self.coord = self.dep.start()
        else:
            self.data = a.data_dir
            self.coord = Coordinator(a.url)
            self.coord.wait_parties(a.passives)
        self.base_cfg = {"dataset_tag": a.dataset, "epochs": a.epochs}
        # the coordinator verifies every party holds this split (v2 item 1)
        for k in ("partition", "overlap", "binarize"):
            if getattr(a, k) is not None:
                self.base_cfg[k] = getattr(a, k)

    # ------------------------------------------------------------------ session
    def session(self, suite, label, cfg, resume_from=None, rtt=0.0, params=None):
        cfg = {**self.base_cfg, **cfg}
        cfg.setdefault("name", f"{suite}-{label}")
        stats = None
        if self.rtt_mode == "netem" and self.local:
            stats = os.path.join(self.out, "sessions", f"netem-{len(self.manifest['sessions'])}.json")
            self._restart_proxy(rtt, stats)
            cfg["inject_rtt_ms"] = 0.0
        else:
            cfg["inject_rtt_ms"] = float(rtt)
        d = self.coord.run(cfg, resume_from=resume_from, label=label)
        if stats:
            time.sleep(2.2)  # proxy flushes stats every 2 s
        sid = d["session_id"]
        d["netem"] = json.load(open(stats)) if stats and os.path.exists(stats) else None
        d["rtt_ms"] = float(rtt)
        d["rtt_mode"] = self.rtt_mode
        with open(os.path.join(self.out, "sessions", f"{sid}.json"), "w") as fh:
            json.dump(d, fh, default=str)
        if self.local:  # keep complete per-batch telemetry for the analysis
            src = os.path.join(self.dep.workdir, "runs", sid)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(self.out, "runs", sid), dirs_exist_ok=True)
        self.manifest["sessions"].append({"suite": suite, "label": label, "session_id": sid, "state": d["state"],
                                          "rtt_ms": float(rtt), "params": params or {}, "config": cfg,
                                          "resume_from": resume_from})
        last = (d.get("epoch_metrics") or [{}])[-1]
        self._index(suite, label, sid, cfg, d["state"], {**(params or {}), "rtt_ms": float(rtt)},
                    {"test_auroc": (last.get("test") or {}).get("auroc"), "completed_epochs": d.get("completed_epochs")})
        self._save()
        return d

    def _restart_proxy(self, rtt, stats):
        p = self.dep.procs.pop("netem", None)
        if p:
            p.terminate()
            p.wait(timeout=5)
        self.dep.spawn_netem(PROXY_PORT, self.dep.ports["active"], rtt, stats_path=stats)
        time.sleep(0.8)

    def _load_index(self):
        try:
            return json.load(open(self.index_path))
        except Exception:
            return []

    def _index(self, suite, label, sid, cfg, state, params, extra=None):
        import hashlib
        keys = {k: cfg.get(k) for k in ("dataset_tag", "seed", "epochs", "batch_size", "alignment", "codec",
                                        "bottom_model", "partition", "overlap", "inject_rtt_ms", "num_classes")}
        keys.update(params or {})
        chash = hashlib.sha256(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:12]
        row = {"run_id": sid, "suite": suite, "label": label, "state": state, "config_hash": chash,
               "seed": cfg.get("seed"), "config": keys, "created": time.time(),
               "software": {k: self.manifest["env"].get(k) for k in ("torch", "grpc", "numpy", "python")},
               "bench_dir": os.path.basename(self.out)}
        row.update(extra or {})
        self.index = [r for r in self.index if r["run_id"] != sid] + [row]
        with open(self.index_path, "w") as fh:
            json.dump(self.index, fh, indent=2, default=str)

    def _save(self):
        with open(os.path.join(self.out, "manifest.json"), "w") as fh:
            json.dump(self.manifest, fh, indent=2, default=str)

    # ------------------------------------------------------------------ suites
    def e1(self):
        print("== E1 correctness & utility")
        for seed in self.a.seeds:
            if self.data and os.path.exists(os.path.join(self.data, "active.csv")):
                cfg = load_session_config(self.a.config, extra={**self.base_cfg, "seed": seed})
                res = run_baselines(self.data, cfg, log=lambda *x: None)
                path = os.path.join(self.out, f"baselines_seed{seed}.json")
                with open(path, "w") as fh:
                    json.dump(res, fh, indent=2, default=str)
                self.manifest["baselines"].append({"seed": seed, "file": os.path.basename(path)})
                print(f"  baselines seed {seed}: B0 {res['b0_centralized']['final_test']['auroc']:.4f}  "
                      f"B1 {res['b1_active_only']['final_test']['auroc']:.4f}  "
                      f"local-VFL {res['local_vfl']['final_test']['auroc']:.4f}")
            self.session("e1", f"c1-seed{seed}", {"seed": seed}, params={"seed": seed})
        seed = self.a.seeds[0]
        self.session("e1", f"c1-seed{seed}-repeat", {"seed": seed}, params={"seed": seed, "repeat": True})

    def e2(self):
        print("== E2 RTT x batch-size sweep")
        for bs in self.a.batch_sizes:
            for rtt in self.a.rtts:
                for seed in self.a.sweep_seeds:
                    self.session("e2", f"bs{bs}-rtt{rtt:g}-seed{seed}",
                                 {"batch_size": bs, "seed": seed, "epochs": self.a.sweep_epochs}, rtt=rtt,
                                 params={"batch_size": bs, "rtt_ms": rtt, "seed": seed})

    def eff(self):
        print(f"== NFR-6 efficiency variants ({','.join(self.a.codecs)})")
        for seed in self.a.seeds:
            for codec in self.a.codecs:
                self.session("eff", f"{codec}-seed{seed}", {"codec": codec, "seed": seed},
                             params={"codec": codec, "seed": seed})

    def defence(self):
        print("== FR-13 defence on/off")
        seed = self.a.seeds[0]
        self.session("def", "off", {"seed": seed, "defense": {"type": "none"}}, params={"noise_multiplier": 0.0})
        for nm in self.a.noise:
            self.session("def", f"grad_noise-sigma{nm:g}",
                         {"seed": seed, "defense": {"type": "grad_noise", "clip": self.a.clip, "noise_multiplier": nm}},
                         params={"noise_multiplier": nm, "clip": self.a.clip})

    def psi(self):
        """DH-PSI vs experimental alignment for this deployment's party count: the two-party protocol with one
        passive, the multi-party protocol (v2.1) with two or more."""
        print(f"== FR-12 DH-PSI alignment ({self.a.passives + 1} parties)")
        self.session("psi", "dh_psi", {"alignment": "dh_psi", "epochs": 1}, params={"alignment": "dh_psi"})
        self.session("psi", "experimental", {"alignment": "experimental", "epochs": 1},
                     params={"alignment": "experimental"})

    def e5(self):
        print("== E5 faults")
        ep = max(3, self.a.fault_epochs)
        fault = {"duplicate": [[0, 2]], "delay": [[1, 3]], "delay_s": 2.0}
        self.session("e5", "dup+delay-deadline5s", {"epochs": ep, "fault": fault, "rpc_deadline_s": 5.0},
                     params={"rpc_deadline_s": 5.0})
        self.session("e5", "dup+delay-deadline1s", {"epochs": ep, "fault": fault, "rpc_deadline_s": 1.0},
                     params={"rpc_deadline_s": 1.0})
        # stop after at least one completed epoch, then resume from the checkpoint
        ep_i = ep + 5
        cfg = {**self.base_cfg, "epochs": ep_i, "name": "e5-to-interrupt", "inject_rtt_ms": 60.0}
        if self.rtt_mode == "netem" and self.local:
            self._restart_proxy(0, None)
        sid = self.coord.start(cfg)["session_id"]
        print(f"  -> {sid} (will be stopped after epoch 1)")
        while True:
            d = http("GET", f"{self.coord.url}/api/sessions/{sid}")
            if d.get("completed_epochs", 0) >= 1 or d["state"] not in ("STARTING", "RUNNING"):
                break
            time.sleep(0.2)
        self.coord.stop(sid)
        while http("GET", f"{self.coord.url}/api/sessions/{sid}")["state"] in ("STARTING", "RUNNING", "STOPPING"):
            time.sleep(0.3)
        self.coord.wait_idle()
        d = http("GET", f"{self.coord.url}/api/sessions/{sid}")
        with open(os.path.join(self.out, "sessions", f"{sid}.json"), "w") as fh:
            json.dump(d, fh, default=str)
        self.manifest["sessions"].append({"suite": "e5", "label": "interrupted", "session_id": sid,
                                          "state": d["state"], "rtt_ms": 0.0, "params": {}, "config": cfg})
        self.session("e5", "resumed", {"epochs": ep_i}, resume_from=sid,
                     params={"resume_from": sid})

    def lopo(self):
        """Leave-one-party-out (v2.1 item 5): for each passive party, a run without it, next to a run with
        everyone, paired by seed. Contribution = AUROC(all) - AUROC(without party). Needs >= 2 passives
        (with one passive, 'without it' is the B1 active-only baseline)."""
        print("== LOPO: leave-one-party-out contribution")
        pids = [f"passive-{i}" for i in range(1, self.a.passives + 1)]
        if len(pids) < 2:
            print("  skipped: needs --passives >= 2 (with one passive, compare B1 in E1)")
            return
        for seed in self.a.seeds:
            self.session("lopo", f"all-seed{seed}", {"seed": seed}, params={"seed": seed, "without": None})
            for pid in pids:
                keep = ["active"] + [p for p in pids if p != pid]
                self.session("lopo", f"without-{pid}-seed{seed}", {"seed": seed, "participants": keep},
                             params={"seed": seed, "without": pid})

    def sweep(self):
        """Scaling sweep (item 36): parties x partition x overlap, each in its own deployment.
        Reports utility vs B0/B1 and slot wait per party vs party count. Local mode only."""
        print("== sweep: parties x partition x overlap")
        if not self.local:
            print("  skipped: --suites sweep needs local mode (own deployments per party count)")
            return
        base = 56200
        for i, n_parties in enumerate(self.a.sweep_parties):
            for strat in self.a.sweep_partitions:
                for ov in self.a.sweep_overlaps:
                    tag = f"P{n_parties}-{strat}-ov{ov:g}"
                    work = os.path.join(self.out, "sweeps", tag)
                    data = os.path.join(work, "data")
                    extra = ["--partitioner", "v2", "--partition", strat, "--overlap", str(ov)]
                    if self.a.dataset == "synthetic":
                        extra += ["--n", str(self.a.n), "--features", str(max(self.a.features, 4 * n_parties))]
                    try:
                        ensure_data(data, self.a.dataset, n_parties - 1, extra)
                    except Exception as e:
                        print(f"  {tag}: data skipped ({e})")
                        continue
                    cfg = load_session_config(self.a.config, extra={**self.base_cfg, "epochs": self.a.sweep_epochs,
                                                                   "partition": strat, "overlap": ov})
                    bl = run_baselines(data, cfg, log=lambda *a: None)
                    port = base + i * 20
                    dep = LocalDeployment(work, data, n_parties - 1, tls=not self.a.insecure, base_port=port,
                                          ui_base=port + 1000, session_config=self.a.config,
                                          active_advertise=f"127.0.0.1:{PROXY_PORT}" if self.rtt_mode == "netem" else None)
                    try:
                        if self.rtt_mode == "netem":
                            dep.spawn_netem(PROXY_PORT, dep.ports["active"], float(self.a.sweep_rtt))
                        coord = dep.start()
                        d = coord.run({**self.base_cfg, "epochs": self.a.sweep_epochs, "partition": strat,
                                       "overlap": ov, "inject_rtt_ms": 0.0 if self.rtt_mode == "netem" else self.a.sweep_rtt,
                                       "name": tag}, label=tag)
                        sid = d["session_id"]
                        if os.path.isdir(os.path.join(dep.workdir, "runs", sid)):
                            import shutil
                            shutil.copytree(os.path.join(dep.workdir, "runs", sid),
                                            os.path.join(self.out, "runs", sid), dirs_exist_ok=True)
                        with open(os.path.join(self.out, "sessions", f"{sid}.json"), "w") as fh:
                            json.dump(d, fh, default=str)
                        slot = _slot_wait_by_party(d)
                        self.manifest["sessions"].append({"suite": "sweep", "label": tag, "session_id": sid,
                                                          "state": d["state"], "rtt_ms": float(self.a.sweep_rtt),
                                                          "params": {"n_parties": n_parties, "partition": strat,
                                                                     "overlap": ov}, "config": d["config"]})
                        last = (d.get("epoch_metrics") or [{}])[-1]
                        self._index("sweep", tag, sid, d["config"], d["state"],
                                    {"n_parties": n_parties, "partition": strat, "overlap": ov},
                                    {"test_auroc": (last.get("test") or {}).get("auroc"),
                                     "b0_auroc": (bl.get("b0_centralized") or {}).get("final_test", {}).get("auroc"),
                                     "b1_auroc": (bl.get("b1_active_only") or {}).get("final_test", {}).get("auroc"),
                                     "slot_wait_ms": slot})
                        self._save()
                        print(f"  {tag}: test AUROC {(last.get('test') or {}).get('auroc')}")
                    finally:
                        dep.stop()

    def security(self):
        """Run the security conformance suite against a dedicated deployment (item 37)."""
        print("== security conformance suite")
        from bench.security_test import run_conformance
        rep = run_conformance(os.path.join(self.out, "security"), self.a, not self.a.insecure)
        self.manifest["security"] = rep["summary"]
        self._save()
        print("  " + json.dumps(rep["summary"]))

    def close(self):
        self._save()
        if self.dep:
            self.dep.stop()


def _slot_wait_by_party(d):
    """The active party's mean slot wait (ms) for this run, plus the party count."""
    agg = d.get("finals", {}).get("active", {})
    return {"mean_wait_parties_ms": agg.get("wait_parties_ms_mean"), "n_parties": len(d.get("participants") or [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"results/bench-{time.strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--suites", default="e1,e2,eff,def,psi,e5")
    ap.add_argument("--url", help="coordinator URL of a running deployment (remote mode)")
    ap.add_argument("--data-dir", help="per-party CSVs (local deployment + baselines)")
    ap.add_argument("--dataset", default="breast_cancer",
                    help="breast_cancer | synthetic (v1); v2 also digits, adult, bank, credit_default, mnist, fashion_mnist")
    ap.add_argument("--partitioner", choices=["v1", "v2"], help="data generator (default v1)")
    ap.add_argument("--partition", help="split strategy passed to prepare-data and checked at session start")
    ap.add_argument("--overlap", type=float, help="entity overlap passed to prepare-data and checked")
    ap.add_argument("--binarize", help="label rule for image datasets (v2), e.g. odd_even")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--features", type=int, default=40)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                     "configs", "session.yaml"))
    ap.add_argument("--passives", type=int, default=1)
    ap.add_argument("--insecure", action="store_true")
    ap.add_argument("--rtt-mode", choices=["netem", "app"])
    ap.add_argument("--epochs", type=int, default=20, help="epochs for utility runs (E1, eff, def)")
    ap.add_argument("--sweep-epochs", type=int, default=3)
    ap.add_argument("--fault-epochs", type=int, default=3)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--sweep-seeds", default="1")
    ap.add_argument("--rtts", default="0,20,50,100")
    ap.add_argument("--batch-sizes", default="32,64,128")
    ap.add_argument("--noise", default="0.5,1,2")
    ap.add_argument("--sweep-parties", default="2,3,4")
    ap.add_argument("--sweep-partitions", default="random,dirichlet")
    ap.add_argument("--sweep-overlaps", default="1.0,0.8")
    ap.add_argument("--sweep-rtt", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=0.05)
    ap.add_argument("--codecs", default="fp32,fp16", help="eff suite: codecs to compare (fp32,fp16,int8,topk)")
    a = ap.parse_args()
    a.seeds = [int(x) for x in a.seeds.split(",")]
    a.sweep_seeds = [int(x) for x in a.sweep_seeds.split(",")]
    a.rtts = [float(x) for x in a.rtts.split(",")]
    a.batch_sizes = [int(x) for x in a.batch_sizes.split(",")]
    a.noise = [float(x) for x in a.noise.split(",")]
    a.sweep_parties = [int(x) for x in str(a.sweep_parties).split(",")]
    a.sweep_partitions = [x for x in a.sweep_partitions.split(",")]
    a.sweep_overlaps = [float(x) for x in str(a.sweep_overlaps).split(",")]
    a.codecs = [x.strip() for x in a.codecs.split(",") if x.strip()]
    r = Runner(a)
    t0 = time.time()
    try:
        for s in a.suites.split(","):
            {"e1": r.e1, "e2": r.e2, "eff": r.eff, "def": r.defence, "psi": r.psi, "e5": r.e5,
             "sweep": r.sweep, "security": r.security, "lopo": r.lopo}[s.strip()]()
    finally:
        r.manifest["wall_s"] = time.time() - t0
        r.close()
    print(f"done in {time.time() - t0:.0f}s -> {r.out}\nnext: python bench/analyze.py {r.out}")


if __name__ == "__main__":
    main()
