"""A synthetic benchmark folder with the same layout and field names bench/run_benchmarks.py and the nodes
write (manifest.json, index.json, sessions/*.json, runs/*/batches_*.jsonl, baselines_seed*.json,
security/conformance.json), so bench/analyze.py and the UI's analytics can be tested without running VFL.
The numbers are made up; only the structure matters."""
from __future__ import annotations

import json
import os

import numpy as np

from vfl.metrics_curves import curves


def _metrics(rng, auc_level, n=120, k=2):
    y = rng.integers(0, k, n)
    if k == 2:
        prob = np.clip(y * auc_level + rng.normal(0.5 - auc_level / 2, 0.22, n), 0.001, 0.999)
        pred = (prob >= 0.5).astype(int)
    else:
        prob = rng.dirichlet(np.ones(k), n)
        prob[np.arange(n), y] += auc_level
        prob /= prob.sum(1, keepdims=True)
        pred = prob.argmax(1)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    auc = float(roc_auc_score(y, prob)) if k == 2 else float(roc_auc_score(y, prob, multi_class="ovr"))
    return {"n": n, "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, average="binary" if k == 2 else "macro")),
            "loss": float(0.7 - 0.4 * auc_level), "auroc": auc, "curves": curves(y, prob)}


def make_bench(path: str, seeds=(1, 2, 3)) -> str:
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(path, "sessions"), exist_ok=True)
    os.makedirs(os.path.join(path, "security"), exist_ok=True)
    manifest = {"created": "2026-11-02 18:00:00", "args": {"data_dir": None}, "sessions": [], "baselines": [],
                "env": {"python": "3.12", "torch": "2.4", "grpc": "1.66"}, "wall_s": 123.0}
    index = []
    env = {"python": "3.12", "torch": "2.4", "grpc": "1.66", "cpu_count": 8, "platform": "Linux-6.1-x86_64",
           "hostname": "lab"}
    t_base = 1_790_000_000.0
    counter = [0]

    def session(suite, label, cfg, params, n_passive=1, epochs=4, auc=0.6, bs=64, rtt=0.0, extra_cfg=None,
                state="COMPLETED", start_epoch=0):
        counter[0] += 1
        sid = f"s20261102-1800{counter[0]:02d}-ab{counter[0]:02d}"
        pids = ["active"] + [f"passive-{i}" for i in range(1, n_passive + 1)]
        config = {"epochs": epochs, "batch_size": bs, "seed": params.get("seed", 1), "codec": "fp32",
                  "alignment": "experimental", "bottom_model": "mlp", "inject_rtt_ms": rtt,
                  "target_val_auroc": 0.75, "defense": {"type": "none", "noise_multiplier": 1.0, "clip": 0.05},
                  "straggler": {"policy": "wait"}, "fault": {"duplicate": [], "delay": []}, "start_epoch": start_epoch,
                  "rpc_deadline_s": 30.0, "partition": "random", "dataset_tag": "breast_cancer", **(extra_cfg or {}), **cfg}
        t0 = t_base + 1000 * counter[0]
        em, pe = [], {p: [] for p in pids}
        for ep in range(start_epoch, epochs):
            lv = auc * (0.75 + 0.25 * (ep + 1) / epochs)
            row = {"epoch": ep, "epoch_s": 1.5 + 0.01 * rtt, "train_loss": 0.6 - 0.08 * ep, "train_batches": 6,
                   "val": _metrics(rng, lv), "test": _metrics(rng, lv),
                   "leakage": {"norm_leak_auc": 0.9 - (0.3 if config["defense"]["type"] != "none" else 0),
                               "cosine_leak_auc": 0.85 - (0.3 if config["defense"]["type"] != "none" else 0)},
                   "resources": {"cpu_percent": 40 + 5 * ep, "rss_mib": 300 + 4 * ep, "t": t0 + 2 + 1.5 * ep}}
            em.append(row)
            pe["active"].append(row)
            for p in pids[1:]:
                pe[p].append({"epoch": ep, "epoch_s": 1.5, "resources": {"cpu_percent": 20 + ep, "rss_mib": 250 + ep,
                                                                          "t": t0 + 2 + 1.5 * ep}})
        n_train = 400
        mib_t = n_train * 16 * 4 * 2 / 2 ** 20 * (0.5 if config["codec"] == "fp16" else 1)
        finals = {"active": {"final_train_loss": em[-1]["train_loss"] if em else None, "optimizer_steps": 6 * epochs,
                             "optimizer_steps_this_session": 6 * (epochs - start_epoch), "scheduled_batches": 6 * (epochs - start_epoch),
                             "committed_batches": 6 * (epochs - start_epoch), "wait_parties_ms_mean": 2.0 + 3 * n_passive + rtt / 10,
                             "duplicates_suppressed": 1 if suite == "e5" else 0, "checkpoint_restore_s": 0.01,
                             "server_compute_ms_p50": 1.2, "env": env,
                             "resources": {"cpu_avg_percent": 40.0, "cpu_max_percent": 60.0, "rss_avg_mib": 300.0, "rss_max_mib": 320.0}}}
        for p in pids[1:]:
            finals[p] = {"epochs": [{"epoch": e} for e in range(start_epoch, epochs)], "epoch_s_median": 1.5 + rtt / 200 + bs / 1000,
                         "batch_ms_p50": 10 + rtt + bs / 20, "batch_ms_p95": 15 + rtt * 1.1 + bs / 15,
                         "throughput_batches_per_s": 1000 / (10 + rtt + bs / 20), "throughput_train_only": 50.0,
                         "mib_per_epoch_tensor": mib_t, "mib_per_epoch_app": mib_t * 1.04, "n_train_entities": n_train,
                         "counters": {"retries": 1 if config["rpc_deadline_s"] < 2 else 0,
                                      "rpc_timeouts": 1 if config["rpc_deadline_s"] < 2 else 0, "duplicate_acks": 1,
                                      "dropped_batches": 0},
                         "resources": {"cpu_avg_percent": 20.0, "cpu_max_percent": 30.0, "rss_avg_mib": 250.0, "rss_max_mib": 260.0},
                         "recovery_first_commit_s": 0.4, "optimizer_steps_this_session": 6 * (epochs - start_epoch),
                         "scheduled_batches": 6 * (epochs - start_epoch), "committed_batches": 6 * (epochs - start_epoch), "env": env}
        alignment = {p: {"mode": config["alignment"], "private": config["alignment"] == "dh_psi", "n_local": 500,
                         "n_aligned": 450, "align_ms": 900.0 if config["alignment"] == "dh_psi" else 20.0,
                         "app_bytes_up": 64000 if config["alignment"] == "dh_psi" else 9000, "app_bytes_down": 30000,
                         "order_digest": "abc123", "disclosure": "intersection size", "splits": {"train": n_train}} for p in pids}
        d = {"session_id": sid, "state": state, "config": config, "participants": [{"party_id": p, "role": "active" if p == "active" else "passive"} for p in pids],
             "epoch_metrics": em, "party_epochs": pe, "finals": finals, "alignment": alignment, "started": t0, "finished": t0 + 2 + 1.5 * epochs,
             "completed_epochs": epochs, "netem": {"bytes_total": int(mib_t * 1.2 * 2 ** 20 * epochs)}, "rtt_ms": rtt}
        json.dump(d, open(os.path.join(path, "sessions", f"{sid}.json"), "w"))
        rdir = os.path.join(path, "runs", sid)
        os.makedirs(rdir, exist_ok=True)
        with open(os.path.join(rdir, "batches_passive-1.jsonl"), "w") as fh:
            for ep in range(start_epoch, epochs):
                for b in range(6):
                    slow = 2000.0 if (suite == "e5" and [ep, b] in config["fault"]["delay"]) else 0.0
                    fh.write(json.dumps({"role": "passive", "phase": "train", "epoch": ep, "batch": b,
                                         "batch_ms": 10 + rtt + slow + rng.random(), "fwd_ms": 1.0, "encode_ms": 0.2,
                                         "injected_rtt_ms": rtt, "network_ms": 2.0, "server_wait_ms": 1.0,
                                         "server_compute_ms": 1.5, "server_delay_ms": slow, "decode_ms": 0.1,
                                         "bwd_ms": 0.8, "step_ms": 0.3, "tensor_bytes_up": bs * 64,
                                         "tensor_bytes_down": bs * 64, "expected_tensor_bytes": bs * 64}) + "\n")
        manifest["sessions"].append({"suite": suite, "label": label, "session_id": sid, "state": state, "rtt_ms": rtt,
                                     "params": params, "config": config})
        index.append({"run_id": sid, "suite": suite, "label": label, "state": state, "seed": params.get("seed"),
                      "config_hash": sid[-6:], "config": {k: config.get(k) for k in ("dataset_tag", "seed", "epochs", "batch_size",
                                                                                   "codec", "bottom_model", "partition", "inject_rtt_ms")},
                      "test_auroc": (em[-1]["test"]["auroc"] if em else None)})
        return sid

    for s in seeds:
        b = {"b0_centralized": {}, "b1_active_only": {}, "local_vfl": {}}
        for key, lvl in (("b0_centralized", 0.7), ("b1_active_only", 0.45)):
            eps = [{"epoch": e, "train_loss": 0.6 - 0.07 * e, "test": _metrics(rng, lvl * (0.8 + 0.05 * e))} for e in range(4)]
            b[key] = {"final_test": eps[-1]["test"], "epochs": eps, "params": 1000}
        b["local_vfl"] = {"final_test": _metrics(rng, 0.6), "final_train_loss": 0.4, "params": 900}
        fn = f"baselines_seed{s}.json"
        json.dump(b, open(os.path.join(path, fn), "w"))
        manifest["baselines"].append({"seed": s, "file": fn})
        session("e1", f"c1-seed{s}", {"seed": s}, {"seed": s})
    session("e1", "c1-repeat", {"seed": seeds[0]}, {"seed": seeds[0], "repeat": True})
    for bs in (32, 128):
        for rtt in (0.0, 50.0):
            session("e2", f"bs{bs}-rtt{rtt:g}", {"batch_size": bs}, {"batch_size": bs, "rtt_ms": rtt, "seed": 1}, bs=bs, rtt=rtt, epochs=2)
    for codec in ("fp32", "fp16", "int8"):
        session("eff", f"{codec}-seed1", {"codec": codec}, {"codec": codec, "seed": 1})
    session("def", "off", {}, {"noise_multiplier": 0.0})
    session("def", "sigma1", {"defense": {"type": "grad_noise", "noise_multiplier": 1.0, "clip": 0.05}}, {"noise_multiplier": 1.0, "clip": 0.05})
    session("def", "rr-eps2", {"defense": {"type": "none", "label_rr_epsilon": 2.0}}, {"noise_multiplier": 0.0})
    session("psi", "dh_psi", {"alignment": "dh_psi"}, {"alignment": "dh_psi"}, epochs=1)
    session("psi", "experimental", {}, {"alignment": "experimental"}, epochs=1)
    fault = {"duplicate": [[0, 2]], "delay": [[1, 3]], "delay_s": 2.0}
    session("e5", "dup+delay-deadline5s", {"fault": fault, "rpc_deadline_s": 5.0}, {"rpc_deadline_s": 5.0}, epochs=3)
    session("e5", "dup+delay-deadline1s", {"fault": fault, "rpc_deadline_s": 1.0}, {"rpc_deadline_s": 1.0}, epochs=3)
    session("e5", "resumed", {}, {"resume_from": "x"}, epochs=4, start_epoch=2)
    for n in (2, 3):
        for strat in ("random", "dirichlet"):
            session("sweep", f"P{n}-{strat}", {"partition": strat, "bottom_model": "mlp" if n == 2 else "linear"},
                    {"n_parties": n, "partition": strat, "overlap": 1.0}, n_passive=n - 1, epochs=2)
    for s in seeds[:2]:
        session("lopo", f"all-seed{s}", {"seed": s}, {"seed": s, "without": None}, n_passive=2, auc=0.62)
        for pid, a in (("passive-1", 0.55), ("passive-2", 0.6)):
            session("lopo", f"without-{pid}-seed{s}", {"seed": s}, {"seed": s, "without": pid}, n_passive=2, auc=a)
    json.dump({"table": [{"case": "no_cert", "expected": "blocked", "outcome": "blocked", "as_expected": True, "server_logged": True},
                         {"case": "membership", "expected": "reported", "outcome": "succeeded", "as_expected": True, "server_logged": False},
                         {"case": "free_rider", "expected": "detected", "outcome": "succeeded", "as_expected": False, "server_logged": False}],
               "summary": {"all_as_expected": False, "as_expected": 2, "cases": 3, "blocked": 1, "inference_succeeded": 1}},
              open(os.path.join(path, "security", "conformance.json"), "w"))
    json.dump(manifest, open(os.path.join(path, "manifest.json"), "w"))
    json.dump(index, open(os.path.join(path, "index.json"), "w"))
    return path
