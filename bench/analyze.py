"""Turn a benchmark directory into figures, tables and an NFR/requirement report (PRD 7.5).

    python bench/analyze.py results/bench [--security results/security/security_report.json]
                                          [--cost configs/cloud_cost.example.yaml]

Outputs in <bench>/report/:
  report.md                 human-readable summary with all tables
  nfr_table.csv / .md       measured value + pass / fail / not attempted per NFR
  requirements_matrix.md    FR -> experiment -> evidence
  summary.json              every computed number (machine-readable)
  fig_*.png                 utility vs epoch, latency/throughput vs RTT, bytes per entity,
                            time breakdown, privacy-utility, fp16 efficiency
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#2f6fde", "#1f9d63", "#c7801a", "#8b5cd6", "#d0453b", "#0f9bb0"]


def _integer_axes(fig):
    """Epochs, parties and batch indices are counts: never show 0.5-style ticks on those axes."""
    from matplotlib.ticker import MaxNLocator
    for ax in fig.axes:
        if ax.get_xlabel().split(" ")[0] in ("epoch", "parties", "batch", "seed"):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def save_fig(fig, figdir, name):
    """Write each figure as PNG and SVG so the dashboard and a paper can both use it (item 30)."""
    _integer_axes(fig)
    fig.savefig(os.path.join(figdir, name + ".png"))
    try:
        fig.savefig(os.path.join(figdir, name + ".svg"))
    except Exception:
        pass
    plt.close(fig)


def write_csv(figdir, name, header, rows):
    with open(os.path.join(figdir, name + ".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "legend.frameon": False})


# ------------------------------------------------------------------ loading
class Bench:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.m = json.load(open(os.path.join(self.path, "manifest.json")))
        self.s = {}
        for e in self.m["sessions"]:
            f = os.path.join(self.path, "sessions", f"{e['session_id']}.json")
            if os.path.exists(f):
                self.s[e["session_id"]] = json.load(open(f))
        self.baselines = {b["seed"]: json.load(open(os.path.join(self.path, b["file"])))
                          for b in self.m.get("baselines", [])}
        self.data_report = None
        for cand in (os.path.join(self.path, "data", "data_report.json"),
                     os.path.join(self.m["args"].get("data_dir") or "", "data_report.json")):
            if cand and os.path.exists(cand):
                self.data_report = json.load(open(cand))
                break

    def suite(self, name):
        return [(e, self.s[e["session_id"]]) for e in self.m["sessions"]
                if e["suite"] == name and e["session_id"] in self.s]

    def batches(self, sid, role=None):
        d = self.s[sid]
        rdir = os.path.join(self.path, "runs", sid)
        recs = []
        if os.path.isdir(rdir):
            for f in os.listdir(rdir):
                if f.startswith("batches_") and f.endswith(".jsonl"):
                    with open(os.path.join(rdir, f)) as fh:
                        recs += [json.loads(line) for line in fh if line.strip()]
        else:
            recs = d.get("recent_batches", [])
        return [r for r in recs if role is None or r.get("role") == role]


def parts(d):
    p = d["participants"]
    if p and isinstance(p[0], str):
        return {pid: ("active" if pid == "active" else "passive") for pid in p}
    return {x["party_id"]: x["role"] for x in p}


def fin(d, role):
    out = [d["finals"][pid] for pid, r in parts(d).items() if r == role and pid in d.get("finals", {})]
    return out[0] if role == "active" and out else out


def last_test(d):
    em = d.get("epoch_metrics") or []
    return (em[-1].get("test") or {}) if em else {}


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else None


def f(v, d=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return f"{v:.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ analyses
def analyse_e1(B, R, figdir):
    runs = [(e, d) for e, d in B.suite("e1") if not e["params"].get("repeat")]
    rep = [(e, d) for e, d in B.suite("e1") if e["params"].get("repeat")]
    rows, per_seed = [], []
    for e, d in runs:
        seed = e["params"]["seed"]
        b = B.baselines.get(seed, {})
        c1 = last_test(d)
        b0 = (b.get("b0_centralized") or {}).get("final_test") or {}
        b1 = (b.get("b1_active_only") or {}).get("final_test") or {}
        lv = b.get("local_vfl") or {}
        act = fin(d, "active") or {}
        diff_local = None
        if lv and act.get("final_train_loss") is not None:
            diff_local = abs(lv["final_train_loss"] - act["final_train_loss"])
        per_seed.append({"seed": seed, "c1_auroc": c1.get("auroc"), "c1_acc": c1.get("accuracy"), "c1_f1": c1.get("f1"),
                         "b0_auroc": b0.get("auroc"), "b0_acc": b0.get("accuracy"),
                         "b1_auroc": b1.get("auroc"), "b1_acc": b1.get("accuracy"),
                         "local_vfl_auroc": (lv.get("final_test") or {}).get("auroc"),
                         "networked_vs_local_loss_absdiff": diff_local,
                         "gap_pp": (b0["auroc"] - c1["auroc"]) * 100 if b0.get("auroc") and c1.get("auroc") else None})
        rows.append([seed, f(b0.get("auroc"), 4), f(b1.get("auroc"), 4), f((lv.get("final_test") or {}).get("auroc"), 4),
                     f(c1.get("auroc"), 4), f(c1.get("accuracy"), 4), f(c1.get("f1"), 4),
                     f(per_seed[-1]["gap_pp"], 2), f(diff_local, 6)])
    R["e1"] = {"per_seed": per_seed}
    gaps = [p["gap_pp"] for p in per_seed if p["gap_pp"] is not None]
    nfr1 = {"value": f"mean B0−VFL AUROC gap {f(mean(gaps), 2)} pp (max {f(max(gaps) if gaps else None, 2)} pp) over {len(gaps)} seeds",
            "status": ("pass" if gaps and mean(gaps) <= 5.0 else "fail") if gaps else "not attempted"}
    # reproducibility (NFR-2)
    nfr2 = {"status": "not attempted", "value": "no repeat run"}
    if rep and runs:
        seed = rep[0][0]["params"]["seed"]
        base = [d for e, d in runs if e["params"]["seed"] == seed]
        if base:
            a, b = base[0], rep[0][1]
            dig = lambda d: sorted({(v or {}).get("order_digest") for v in d.get("alignment", {}).values()})  # noqa
            fa, fb = fin(a, "active") or {}, fin(b, "active") or {}
            la, lb = fa.get("final_train_loss"), fb.get("final_train_loss")
            rel = abs(la - lb) / abs(la) * 100 if la and lb is not None else None
            seeds_loss = [(fin(d, "active") or {}).get("final_train_loss") for _, d in runs]
            cv = float(np.std(seeds_loss) / np.mean(seeds_loss) * 100) if len(seeds_loss) > 1 else None
            same_order = dig(a) == dig(b) and len(dig(a)) == 1 and None not in dig(a)
            same_steps = fa.get("optimizer_steps") == fb.get("optimizer_steps")
            R["e1"]["reproducibility"] = {"same_aligned_order": same_order, "order_digest": dig(a),
                                          "same_committed_steps": same_steps, "steps": fa.get("optimizer_steps"),
                                          "repeat_final_loss_rel_diff_pct": rel, "across_seed_final_loss_cv_pct": cv}
            ok = same_order and same_steps and rel is not None and rel <= 5.0
            nfr2 = {"status": "pass" if ok else ("investigate" if same_order and same_steps else "fail"),
                    "value": f"aligned order identical={same_order}, committed steps identical={same_steps} "
                             f"({fa.get('optimizer_steps')}), same-seed final-loss diff {f(rel, 4)}%, "
                             f"3-seed final-loss CV {f(cv, 2)}%"}
    R["nfr"]["NFR-1"] = nfr1
    R["nfr"]["NFR-2"] = nfr2
    R["tables"]["E1 utility (final epoch, test split)"] = md_table(
        ["seed", "B0 AUROC", "B1 AUROC", "local VFL AUROC", "C1 networked AUROC", "C1 acc", "C1 F1",
         "B0−C1 (pp)", "|loss networked − local|"], rows)
    # figure: utility vs epoch
    if runs:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        for i, (e, d) in enumerate(runs):
            em = d.get("epoch_metrics", [])
            ax[0].plot([m["epoch"] for m in em], [m["test"]["auroc"] for m in em], color=COLORS[0], alpha=0.5 + 0.5 * (i == 0),
                       label="C1 VFL (networked)" if i == 0 else None)
            ax[1].plot([m["epoch"] for m in em], [m["train_loss"] for m in em], color=COLORS[0], alpha=0.5 + 0.5 * (i == 0),
                       label="C1 VFL" if i == 0 else None)
        seed0 = runs[0][0]["params"]["seed"]
        b = B.baselines.get(seed0, {})
        for key, name, c in (("b0_centralized", "B0 centralized", COLORS[1]), ("b1_active_only", "B1 active-only", COLORS[2])):
            if key in b:
                ep = b[key]["epochs"]
                ax[0].plot([x["epoch"] for x in ep], [x["test"]["auroc"] for x in ep], "--", color=c, label=name)
                ax[1].plot([x["epoch"] for x in ep], [x["train_loss"] for x in ep], "--", color=c, label=name)
        ax[0].set(title="Test AUROC vs epoch", xlabel="epoch", ylabel="AUROC")
        ax[1].set(title="Training loss vs epoch", xlabel="epoch", ylabel="BCE loss")
        ax[0].legend()
        fig.tight_layout()
        save_fig(fig, figdir, "fig_utility_vs_epoch")


def passive_breakdown(B, sid):
    recs = [r for r in B.batches(sid, "passive") if r.get("phase") == "train"]
    keys = ["fwd_ms", "encode_ms", "injected_rtt_ms", "network_ms", "server_wait_ms", "server_compute_ms",
            "server_delay_ms", "decode_ms", "bwd_ms", "step_ms"]
    return {k: mean([r.get(k, 0.0) for r in recs]) for k in keys}, recs


def analyse_e2(B, R, figdir):
    runs = B.suite("e2")
    if not runs:
        R["nfr"]["NFR-3"] = {"status": "not attempted", "value": ""}
        R["nfr"]["NFR-5"] = {"status": "not attempted", "value": ""}
        return
    grid = defaultdict(list)
    rows, mism, nrec = [], 0, 0
    for e, d in runs:
        bs, rtt = e["params"]["batch_size"], e["params"]["rtt_ms"]
        ps = fin(d, "passive")
        if not ps:
            continue
        p = ps[0]
        bd, recs = passive_breakdown(B, e["session_id"])
        for r in recs:
            nrec += 1
            mism += int(r.get("tensor_bytes_up") != r.get("expected_tensor_bytes")
                        or r.get("tensor_bytes_down") != r.get("expected_tensor_bytes"))
        n_ep = max(1, len(p.get("epochs") or []))
        net = (d.get("netem") or {}).get("bytes_total")
        n_train = p.get("n_train_entities") or 1
        rec = {"batch_size": bs, "rtt_ms": rtt, "epoch_s_median": p.get("epoch_s_median"),
               "batch_ms_p50": p.get("batch_ms_p50"), "batch_ms_p95": p.get("batch_ms_p95"),
               "throughput": p.get("throughput_batches_per_s"), "throughput_train_only": p.get("throughput_train_only"),
               "mib_epoch_tensor": p.get("mib_per_epoch_tensor"), "mib_epoch_app": p.get("mib_per_epoch_app"),
               "mib_epoch_network": net / 2 ** 20 / n_ep if net else None,
               "kib_entity_tensor": p.get("mib_per_epoch_tensor") * 1024 / n_train,
               "kib_entity_app": p.get("mib_per_epoch_app") * 1024 / n_train,
               "kib_entity_network": net / 1024 / n_ep / n_train if net else None,
               "breakdown_ms": bd, "cpu": (p.get("resources") or {}), "n_train": n_train}
        grid[(bs, rtt)].append(rec)
    agg = {}
    for k, recs in sorted(grid.items()):
        agg[k] = {key: mean([r[key] for r in recs]) if not isinstance(recs[0][key], dict) else recs[0][key]
                  for key in recs[0]}
        a = agg[k]
        rows.append([k[0], f(k[1], 0), f(a["epoch_s_median"]), f(a["batch_ms_p50"], 1), f(a["batch_ms_p95"], 1),
                     f(a["throughput"], 1), f(a["mib_epoch_tensor"], 3), f(a["mib_epoch_app"], 3),
                     f(a["mib_epoch_network"], 3), f(a["kib_entity_app"], 3)])
    R["e2"] = {f"bs{k[0]}_rtt{k[1]:g}": v for k, v in agg.items()}
    R["tables"]["E2 RTT × batch size (passive view)"] = md_table(
        ["batch", "RTT ms", "epoch s (median)", "batch p50 ms", "batch p95 ms", "batches/s", "tensor MiB/epoch",
         "app MiB/epoch", "network MiB/epoch", "app KiB/entity/epoch"], rows)
    bss = sorted({k[0] for k in agg})
    rtts = sorted({k[1] for k in agg})
    # NFR-3
    p95_lan = {bs: agg[(bs, 0.0)]["batch_ms_p95"] for bs in bss if (bs, 0.0) in agg}
    R["nfr"]["NFR-3"] = {"status": ("pass" if p95_lan and max(p95_lan.values()) <= 2000 else "fail") if p95_lan else "not attempted",
                         "value": "; ".join(f"bs{bs}: p95 {f(v, 1)} ms @0ms RTT" for bs, v in p95_lan.items())
                                  + f" (target ≤2000 ms); RTTs measured: {', '.join(f(r, 0) for r in rtts)} ms"}
    # NFR-5
    thr = {bs: agg[(bs, 0.0)]["throughput"] for bs in bss if (bs, 0.0) in agg}
    bottleneck = {}
    for bs in bss:
        if (bs, 0.0) in agg:
            bd = agg[(bs, 0.0)]["breakdown_ms"]
            bottleneck[bs] = max(bd, key=lambda k: bd[k] or 0)
    R["nfr"]["NFR-5"] = {"status": ("pass" if thr and min(thr.values()) >= 5 else "fail") if thr else "not attempted",
                         "value": "; ".join(f"bs{bs}: {f(v, 1)} batches/s (largest component: {bottleneck.get(bs)})"
                                            for bs, v in thr.items()) + " (target ≥5)"}
    # NFR-4
    ex = agg[(bss[0], rtts[0])]
    over_app = ex["mib_epoch_app"] / ex["mib_epoch_tensor"] - 1 if ex["mib_epoch_tensor"] else None
    over_net = ex["mib_epoch_network"] / ex["mib_epoch_app"] - 1 if ex["mib_epoch_network"] else None
    R["nfr"]["NFR-4"] = {"status": "pass" if mism == 0 and nrec > 0 else "fail",
                         "value": f"{nrec} batch records: FP32 tensor payload == rows×embed×4 in {nrec - mism}/{nrec}; "
                                  f"protobuf overhead +{f(over_app * 100 if over_app is not None else None, 1)}% over tensors, "
                                  f"wire (gRPC/HTTP2/TLS, incl. eval+control) +{f(over_net * 100 if over_net is not None else None, 1)}% over app bytes "
                                  f"(bs{bss[0]}); see E2 table for MiB/epoch and KiB/entity"}
    # figures
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for i, bs in enumerate(bss):
        xs = [r for r in rtts if (bs, r) in agg]
        ax[0].plot(xs, [agg[(bs, r)]["epoch_s_median"] for r in xs], "o-", color=COLORS[i], label=f"batch {bs}")
        ax[1].plot(xs, [agg[(bs, r)]["batch_ms_p95"] for r in xs], "o-", color=COLORS[i], label=f"batch {bs}")
        ax[1].plot(xs, [agg[(bs, r)]["batch_ms_p50"] for r in xs], "o--", color=COLORS[i], alpha=0.5)
        ax[2].plot(xs, [agg[(bs, r)]["throughput"] for r in xs], "o-", color=COLORS[i], label=f"batch {bs}")
    ax[0].set(title="Median epoch time vs RTT", xlabel="RTT (ms)", ylabel="seconds")
    ax[1].set(title="Batch time p95 (solid) / p50 (dashed)", xlabel="RTT (ms)", ylabel="ms")
    ax[2].set(title="Throughput vs RTT", xlabel="RTT (ms)", ylabel="committed batches / s")
    ax[2].axhline(5, color="#999", lw=0.8, ls=":")
    ax[0].legend()
    fig.tight_layout()
    save_fig(fig, figdir, "fig_latency_throughput_vs_rtt")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    w = 0.25
    x = np.arange(len(bss))
    for j, (key, name) in enumerate((("kib_entity_tensor", "tensor payload"), ("kib_entity_app", "serialized app"),
                                     ("kib_entity_network", "on the wire"))):
        vals = [agg[(bs, rtts[0])][key] or 0 for bs in bss]
        ax.bar(x + (j - 1) * w, vals, w, color=COLORS[j], label=name)
    ax.set_xticks(x, [f"batch {bs}" for bs in bss])
    ax.set(title="Traffic per aligned training entity per epoch", ylabel="KiB")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, figdir, "fig_bytes_per_entity")
    bs_ref = 64 if 64 in bss else bss[0]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    comps = list(agg[(bs_ref, rtts[0])]["breakdown_ms"].keys())
    bottom = np.zeros(len(rtts))
    for j, c in enumerate(comps):
        vals = np.array([agg[(bs_ref, r)]["breakdown_ms"].get(c) or 0 for r in rtts if (bs_ref, r) in agg])
        if vals.sum() <= 0:
            continue
        ax.bar([f"{r:g} ms" for r in rtts if (bs_ref, r) in agg], vals, bottom=bottom[:len(vals)],
               color=COLORS[j % len(COLORS)] if j < 6 else plt.cm.Greys(0.3 + 0.1 * (j - 6)), label=c.replace("_ms", ""))
        bottom[:len(vals)] += vals
    ax.set(title=f"Where batch time goes (batch {bs_ref}, passive view)", xlabel="RTT", ylabel="mean ms per batch")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    save_fig(fig, figdir, "fig_time_breakdown")


def analyse_eff(B, R, figdir):
    runs = B.suite("eff")
    by = defaultdict(dict)
    for e, d in runs:
        p = fin(d, "passive")
        if not p:
            continue
        net = (d.get("netem") or {}).get("bytes_total")
        n_ep = max(1, len(p[0].get("epochs") or []))
        by[e["params"]["seed"]][e["params"]["codec"]] = {
            "auroc": last_test(d).get("auroc"), "app": p[0]["mib_per_epoch_app"], "tensor": p[0]["mib_per_epoch_tensor"],
            "net": net / 2 ** 20 / n_ep if net else None, "epoch_s": p[0].get("epoch_s_median")}
    pairs = [(s, v["fp32"], v["fp16"]) for s, v in by.items() if "fp32" in v and "fp16" in v]
    res_rows = []
    for s, a, b in pairs:
        res_rows.append([s, f(a["auroc"], 4), f(b["auroc"], 4), f((b["auroc"] - a["auroc"]) * 100, 2),
                         f(a["app"], 4), f(b["app"], 4), f((1 - b["app"] / a["app"]) * 100, 1),
                         f(a["net"], 4), f(b["net"], 4)])
    # resources from the utility runs
    res = []
    for e, d in B.suite("e1")[:1]:
        for pid, fr in d.get("finals", {}).items():
            r = fr.get("resources") or {}
            res.append(f"{pid}: CPU avg {f(r.get('cpu_avg_percent'), 1)}% / max {f(r.get('cpu_max_percent'), 1)}%, "
                       f"RSS max {f(r.get('rss_max_mib'), 0)} MiB")
    if pairs:
        red = mean([(1 - b["app"] / a["app"]) * 100 for _, a, b in pairs])
        dau = mean([(b["auroc"] - a["auroc"]) * 100 for _, a, b in pairs])
        ok = red >= 10 and abs(dau) <= 5
        R["nfr"]["NFR-6"] = {"status": "pass" if ok else "fail",
                             "value": f"fp16 variant: −{f(red, 1)}% app bytes/epoch, ΔAUROC {f(dau, 2)} pp over "
                                      f"{len(pairs)} paired seeds. " + " | ".join(res)}
        R["tables"]["NFR-6 efficiency variant (fp16 vs fp32, paired seeds)"] = md_table(
            ["seed", "AUROC fp32", "AUROC fp16", "Δ pp", "app MiB/ep fp32", "app MiB/ep fp16", "reduction %",
             "wire MiB/ep fp32", "wire MiB/ep fp16"], res_rows)
        R["eff"] = {str(s): {"fp32": a, "fp16": b} for s, a, b in pairs}
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        x = np.arange(len(pairs))
        ax[0].bar(x - 0.2, [a["app"] for _, a, _ in pairs], 0.4, color=COLORS[0], label="fp32")
        ax[0].bar(x + 0.2, [b["app"] for _, _, b in pairs], 0.4, color=COLORS[2], label="fp16")
        ax[0].set_xticks(x, [f"seed {s}" for s, _, _ in pairs])
        ax[0].set(title="App bytes per epoch", ylabel="MiB")
        ax[0].legend()
        ax[1].bar(x - 0.2, [a["auroc"] for _, a, _ in pairs], 0.4, color=COLORS[0])
        ax[1].bar(x + 0.2, [b["auroc"] for _, _, b in pairs], 0.4, color=COLORS[2])
        ax[1].set_xticks(x, [f"seed {s}" for s, _, _ in pairs])
        lo = min(min(a["auroc"], b["auroc"]) for _, a, b in pairs)
        ax[1].set(title="Test AUROC", ylim=(max(0, lo - 0.05), 1.0))
        fig.tight_layout()
        save_fig(fig, figdir, "fig_efficiency_fp16")
    else:
        R["nfr"]["NFR-6"] = {"status": "baseline only", "value": " | ".join(res) or "no efficiency runs"}


def analyse_def(B, R, figdir):
    runs = B.suite("def")
    rows, pts = [], []
    off_time = None
    for e, d in runs:
        em = d.get("epoch_metrics") or []
        if not em:
            continue
        nm = e["params"].get("noise_multiplier", 0.0)
        lk = [m.get("leakage") or {} for m in em]
        a = fin(d, "active") or {}
        p = (fin(d, "passive") or [{}])[0]
        t = p.get("epoch_s_median")
        if nm == 0:
            off_time = t
        rows.append([e["label"], f(nm, 2), f(e["params"].get("clip"), 3), f(last_test(d).get("auroc"), 4),
                     f(lk[-1].get("norm_leak_auc"), 3), f(lk[-1].get("cosine_leak_auc"), 3),
                     f(mean([x.get("cosine_leak_auc") for x in lk]), 3), f(t, 3),
                     f(a.get("server_compute_ms_p50"), 3)])
        pts.append((nm, last_test(d).get("auroc"), lk[-1].get("norm_leak_auc"), lk[-1].get("cosine_leak_auc"), t))
    if not rows:
        R["nfr"]["NFR-8"] = {"status": "not attempted", "value": ""}
        return
    R["tables"]["FR-13 defence: label-leakage attack vs utility"] = md_table(
        ["run", "σ", "clip C", "test AUROC", "norm-attack AUC", "cosine-attack AUC", "cosine AUC (mean over epochs)",
         "epoch s", "active compute p50 ms"], rows)
    R["def"] = [{"noise_multiplier": p[0], "auroc": p[1], "norm_leak_auc": p[2], "cosine_leak_auc": p[3],
                 "epoch_s": p[4]} for p in pts]
    on = [p for p in pts if p[0] > 0]
    txt = ("protected: active-party labels; adversary: passive party observing dL/dH; noise inserted per example "
           "before serialisation; reported as gradient perturbation (no formal (ε,δ) claimed). ")
    if on and pts[0][0] == 0:
        txt += (f"Undefended cosine-attack AUC {f(pts[0][3], 3)} → {', '.join(f'{f(p[3], 3)} @σ={p[0]:g}' for p in on)}; "
                f"test AUROC {f(pts[0][1], 4)} → {', '.join(f(p[1], 4) for p in on)}; epoch-time overhead "
                f"{', '.join(f((p[4] / off_time - 1) * 100 if off_time and p[4] else None, 1) + '%' for p in on)}")
    R["nfr"]["NFR-8"] = {"status": "reported", "value": txt}
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for p in pts:
        ax.scatter(p[3], p[1], color=COLORS[0] if p[0] == 0 else COLORS[4], s=40)
        ax.annotate("off" if p[0] == 0 else f"σ={p[0]:g}", (p[3], p[1]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set(title="Privacy–utility (cosine label-leakage attack)", xlabel="attack AUC (0.5 = no leakage)", ylabel="test AUROC")
    fig.tight_layout()
    save_fig(fig, figdir, "fig_privacy_utility")


def analyse_psi(B, R):
    runs = B.suite("psi")
    rows = []
    for e, d in runs:
        for pid, a in (d.get("alignment") or {}).items():
            rows.append([e["params"].get("alignment"), pid, a.get("private"), f(a.get("n_local"), 0), f(a.get("n_aligned"), 0),
                         f(a.get("align_ms"), 0), f(a.get("app_bytes_up"), 0) if a.get("app_bytes_up") else "—",
                         f(a.get("app_bytes_down"), 0) if a.get("app_bytes_down") else "—"])
    if not rows:
        R["nfr"].setdefault("NFR-8", {"status": "not attempted", "value": ""})
        return
    R["tables"]["FR-12 alignment: DH-PSI vs non-private experimental"] = md_table(
        ["mode", "party", "private", "local rows", "aligned", "time ms", "bytes up", "bytes down"], rows)
    psi = [d for e, d in runs if e["params"].get("alignment") == "dh_psi"]
    exp = [d for e, d in runs if e["params"].get("alignment") == "experimental"]
    if psi:
        ns = {a.get("n_aligned") for a in psi[0]["alignment"].values()}
        ne = {a.get("n_aligned") for a in exp[0]["alignment"].values()} if exp else set()
        expect = (B.data_report or {}).get("n_aligned_expected")
        ok = len(ns) == 1 and (not ne or ns == ne) and (expect is None or ns == {expect})
        disc = next(iter(psi[0]["alignment"].values())).get("disclosure")
        R["psi"] = {"correct_intersection": ok, "n_aligned": list(ns), "expected": expect, "disclosure": disc}
        R["nfr"]["NFR-8"]["value"] = (R["nfr"]["NFR-8"].get("value", "") + f" | PSI: intersection correct={ok} "
                                      f"({list(ns)} vs expected {expect}); disclosure: {disc}")


def analyse_e5(B, R):
    runs = B.suite("e5")
    if not runs:
        R["nfr"]["NFR-10"] = {"status": "not attempted", "value": ""}
        return
    rows, checks = [], []
    for e, d in runs:
        a = fin(d, "active") or {}
        ps = fin(d, "passive") or []
        p = ps[0] if ps else {}
        recs = [r for r in B.batches(e["session_id"], "passive") if r.get("phase") == "train"]
        max_b = max([r["batch_ms"] for r in recs], default=None)
        pc = p.get("counters") or {}
        # exactly one optimizer step per scheduled batch at every party (no extra step from duplicates/retries)
        steps_ok = bool(a) and all(
            x.get("optimizer_steps_this_session") == x.get("scheduled_batches") == x.get("committed_batches")
            for x in [a] + list(ps))
        row = {"label": e["label"], "state": d["state"], "deadline_s": d["config"].get("rpc_deadline_s"),
               "duplicates_suppressed": a.get("duplicates_suppressed"), "duplicate_acks": pc.get("duplicate_acks"),
               "timeouts": pc.get("rpc_timeouts", 0), "retries": pc.get("retries", 0), "max_batch_ms": max_b,
               "active_steps": a.get("optimizer_steps"), "committed": a.get("committed_batches"),
               "no_extra_step": steps_ok, "checkpoint_restore_s": a.get("checkpoint_restore_s"),
               "recovery_first_commit_s": p.get("recovery_first_commit_s"),
               "start_epoch": d["config"].get("start_epoch")}
        checks.append(row)
        rows.append([row["label"], row["state"], f(row["deadline_s"], 1), f(row["duplicates_suppressed"], 0),
                     f(row["duplicate_acks"], 0), f(row["timeouts"], 0), f(row["retries"], 0), f(max_b, 0),
                     f(row["active_steps"], 0), f(row["committed"], 0), f(steps_ok),
                     f(row["recovery_first_commit_s"], 3) if row["label"] == "resumed" else "—"])
    R["tables"]["E5 fault handling"] = md_table(
        ["run", "state", "deadline s", "dups suppressed", "dup acks", "timeouts", "retries", "max batch ms",
         "active steps", "committed", "no extra step", "recovery s"], rows)
    R["e5"] = checks
    by = {c["label"]: c for c in checks}
    parts_ = []
    ok = True
    if "dup+delay-deadline5s" in by:
        c = by["dup+delay-deadline5s"]
        good = c["state"] == "COMPLETED" and c["no_extra_step"] and (c["max_batch_ms"] or 0) >= 1900
        ok &= good
        parts_.append(f"2 s delay with 5 s deadline completed (max batch {f(c['max_batch_ms'], 0)} ms), "
                      f"duplicate suppressed={c['duplicates_suppressed']}, no extra step={c['no_extra_step']}")
    if "dup+delay-deadline1s" in by:
        c = by["dup+delay-deadline1s"]
        good = c["state"] == "COMPLETED" and c["timeouts"] >= 1 and c["no_extra_step"]
        ok &= good
        parts_.append(f"1 s deadline: {c['timeouts']} diagnosable timeout(s), {c['retries']} retry, run "
                      f"{c['state'].lower()}, no extra step={c['no_extra_step']}")
    if "resumed" in by:
        c = by["resumed"]
        good = c["state"] == "COMPLETED"
        ok &= good
        parts_.append(f"epoch-checkpoint restart from epoch {c['start_epoch']}: first committed batch "
                      f"{f(c['recovery_first_commit_s'], 3)} s after START (checkpoint load {f(c['checkpoint_restore_s'], 4)} s)")
    R["nfr"]["NFR-10"] = {"status": "pass" if ok else "fail", "value": "; ".join(parts_)}


def analyse_security(R, path):
    if not path or not os.path.exists(path):
        R["nfr"]["NFR-7"] = {"status": "not attempted", "value": "run bench/security_test.py and pass --security"}
        return
    s = json.load(open(path))
    sm = s["summary"]
    R["security"] = sm
    R["nfr"]["NFR-7"] = {"status": "pass" if sm["PASS"] else "fail",
                         "value": f"invalid certificates rejected {sm['FR-11/NFR-7 invalid certificates rejected']}; "
                                  f"identity binding {sm['identity binding']}; plaintext markers in TLS capture "
                                  f"{sm['plaintext markers in captured TLS traffic']}; sentinel/raw-id log hits "
                                  f"{sm['sentinel/raw-id hits in logs']}"
                                  + (f"; plaintext control found {sm.get('plaintext control: markers found without TLS (scanner works)')} markers"
                                     if "plaintext control: markers found without TLS (scanner works)" in sm else "")}


def analyse_cost(B, R, path):
    """NFR-9: C_r = Σ p_i t_i/3600 + p_storage,h G_r h_r + p_egress E_r ; C_r,1000 = 1000 C_r / N_r."""
    if not path:
        R["nfr"]["NFR-9"] = {"status": "not attempted", "value": "no cloud run / price snapshot supplied (--cost)"}
        return
    import yaml
    c = yaml.safe_load(open(path))
    rows, out = [], []
    for e, d in [(e, B.s[e["session_id"]]) for e in B.m["sessions"] if e["session_id"] in B.s]:
        if d["state"] != "COMPLETED" or not d.get("finished") or not d.get("started"):
            continue
        t = d["finished"] - d["started"]
        vm = sum(v["price_per_hour"] * t / 3600 for v in c["vms"])
        storage = c.get("storage_price_per_gb_hour", 0) * c.get("storage_gb", 0) * t / 3600
        net = (d.get("netem") or {}).get("bytes_total")
        egress_gb = c.get("egress_gb_override") if c.get("egress_gb_override") is not None else ((net or 0) / 1e9 if c.get("egress_paid") else 0)
        egress = c.get("egress_price_per_gb", 0) * egress_gb
        total = vm + storage + egress
        a = d["alignment"].get("active") or next(iter(d["alignment"].values()), {})
        n = (a.get("splits") or {}).get("train") or a.get("n_aligned") or 1
        n_ep = max(1, d.get("completed_epochs") or 1)
        rows.append([e["label"], f(t, 1), f(vm, 5), f(storage, 5), f(egress, 5), f(total, 5),
                     f(1000 * total / n, 5), f(total / n_ep, 5)])
        out.append({"label": e["label"], "seconds": t, "total": total, "per_1000_entities": 1000 * total / n,
                    "per_epoch": total / n_ep})
    R["cost"] = {"snapshot": {k: c.get(k) for k in ("region", "currency", "price_date", "taxes_included", "free_tier_excluded")},
                 "runs": out}
    R["tables"][f"NFR-9 cloud cost ({c.get('currency')}, {c.get('region')}, prices {c.get('price_date')})"] = md_table(
        ["run", "active s", "VM", "storage", "egress", "total", "per 1000 entities", "per epoch"], rows)
    R["nfr"]["NFR-9"] = {"status": "reported" if c.get("measured_cloud_run") else "estimate (prices applied to local timings)",
                         "value": f"{len(out)} runs costed; median per epoch {f(mean([o['per_epoch'] for o in out]), 5)} {c.get('currency')}"}


def requirements_matrix(B, R):
    e1, e2, _e5 = B.suite("e1"), B.suite("e2"), B.suite("e5")
    ok_runs = [d for _, d in B.s.items() if d["state"] == "COMPLETED"]
    bss = sorted({e["params"]["batch_size"] for e, _ in e2})
    any_d = ok_runs[0] if ok_runs else None
    exp_n = (B.data_report or {}).get("n_aligned_expected")
    al_n = {a.get("n_aligned") for a in (any_d or {}).get("alignment", {}).values()}
    psi_ok = (R.get("psi") or {}).get("correct_intersection")
    p_recs = B.batches(e1[0][0]["session_id"], "passive") if e1 else []
    shape_ok = all(r.get("tensor_bytes_up") == r.get("expected_tensor_bytes") for r in p_recs if r.get("phase") == "train")
    rows = [
        ["FR-1", "roles, registration, validation", "all suites; tests/test_coordinator.py",
         f"{len(ok_runs)} sessions completed with 1 active + passive(s); invalid roles rejected (pytest)"],
        ["FR-2", "replaceable alignment", "E1, psi; tests/test_alignment.py",
         f"aligned {sorted(al_n)} vs expected {exp_n}; unmatched ids absent from logs: see NFR-7"],
        ["FR-3", "deterministic batches", "E1 repeat; tests/test_data.py",
         f"same aligned order/steps on repeat: {(R.get('e1') or {}).get('reproducibility', {}).get('same_aligned_order')}; batch digests checked on every message"],
        ["FR-4", "local representations only", "E1 telemetry; tests/test_core.py",
         f"every passive message is a [batch × embed_dim] tensor (payload size check {'ok' if shape_ok else 'FAILED'}); no feature/label fields exist in the protocol"],
        ["FR-5", "top model, loss, gradients", "E1", "finite loss each batch; returned gradients shaped like representations"],
        ["FR-6", "synchronous per batch", "E2, E5", "injected RTT and the 2 s delay appear in batch time (E2/E5 tables)"],
        ["FR-7", "local parameter updates", "tests/test_core.py; E1", "optimizer steps == committed batches at every party"],
        ["FR-8", "message identity + duplicate suppression", "E5", f"duplicates suppressed without extra steps: "
         f"{all(c.get('no_extra_step') for c in R.get('e5', []))}"],
        ["FR-9", "configuration without code edits", "E2", f"batch sizes run: {bss}"],
        ["FR-10", "machine-readable telemetry", "all", "per-batch / per-epoch / per-run JSONL+JSON at each node and coordinator"],
        ["FR-11", "authenticated + encrypted transport", "security_test", (R.get("nfr", {}).get("NFR-7") or {}).get("value", "")],
        ["FR-12", "reviewed PSI (could)", "psi", f"DH-PSI implemented (unreviewed textbook protocol); intersection correct: {psi_ok}"],
        ["FR-13", "inference defence (could)", "def", "gradient perturbation vs norm/cosine label-leakage attacks"],
        ["FR-14", "Flotilla adapter (could)", "—", "not attempted (stretch goal outside the 6-week critical path)"],
    ]
    return md_table(["ID", "capability", "experiment", "evidence"], rows)


def analyse_sweep(B, R, figdir):
    """Scaling (item 36): utility vs number of parties, and the active party's slot wait vs party count."""
    runs = B.suite("sweep")
    if not runs:
        return
    rows, pts = [], {}
    idx = {r["run_id"]: r for r in _load_index(B.path)}
    for e, d in runs:
        p = e["params"]
        meta = idx.get(e["session_id"], {})
        test = last_test(d).get("auroc")
        wait = (fin(d, "active") or {}).get("wait_parties_ms_mean")
        rows.append([p.get("n_parties"), p.get("partition"), p.get("overlap"), f(test, 4),
                     f(meta.get("b0_auroc"), 4), f(meta.get("b1_auroc"), 4), f(wait, 1)])
        key = (p.get("partition"), p.get("overlap"))
        pts.setdefault(key, []).append((p.get("n_parties"), test, wait, meta.get("b1_auroc")))
    R["tables"]["Sweep: scaling with parties x partition x overlap"] = md_table(
        ["parties", "partition", "overlap", "VFL AUROC", "B0 AUROC", "B1 AUROC", "slot wait ms"], rows)
    write_csv(figdir, "sweep", ["parties", "partition", "overlap", "vfl_auroc", "b0_auroc", "b1_auroc", "slot_wait_ms"],
              [[c for c in r] for r in rows])
    R["sweep"] = {f"{k[0]}|ov{k[1]}": sorted(v) for k, v in pts.items()}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, (key, vs) in enumerate(sorted(pts.items())):
        vs = sorted(v for v in vs if v[0] is not None)
        xs = [v[0] for v in vs]
        ax[0].plot(xs, [v[1] for v in vs], "o-", color=COLORS[i % len(COLORS)], label=f"{key[0]} ov{key[1]}")
        ax[1].plot(xs, [v[2] for v in vs if v[2] is not None] or [None] * len(xs), "o-", color=COLORS[i % len(COLORS)])
    ax[0].set(title="Utility vs number of parties", xlabel="parties", ylabel="test AUROC")
    ax[1].set(title="Active-party slot wait vs parties", xlabel="parties", ylabel="mean wait (ms)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, figdir, "fig_scaling_parties")


def analyse_conformance(B, R):
    """Security conformance suite results (item 37), from bench/security_test.run_conformance."""
    path = os.path.join(B.path, "security", "conformance.json")
    if not os.path.exists(path):
        return
    rep = json.load(open(path))
    rows = [[r["case"], r["expected"], r["outcome"], "yes" if r["as_expected"] else "NO",
             "yes" if r.get("server_logged") else "no"] for r in rep["table"]]
    R["tables"]["Security conformance suite (attack -> gate outcome)"] = md_table(
        ["case", "expected", "outcome", "as expected", "logged by node"], rows)
    sm = rep["summary"]
    R["conformance"] = sm
    R["nfr"].setdefault("NFR-7", {})
    prev = R["nfr"]["NFR-7"].get("value", "")
    R["nfr"]["NFR-7"] = {"status": "pass" if sm["all_as_expected"] and R["nfr"]["NFR-7"].get("status", "pass") != "fail" else R["nfr"]["NFR-7"].get("status", "pass"),
                         "value": (prev + "; " if prev else "") + f"conformance suite {sm['as_expected']}/{sm['cases']} "
                         f"cases as designed ({sm['blocked']} blocked, {sm['inference_succeeded']} inference succeeded "
                         "as expected)"}


# ================================================================== figure registry (v2.1 item 5)
# Every analytics figure is a function here. `main()` renders all of them into <bench>/report/figures/ and
# lists them in report.md; the unified UI renders the SAME functions (unfiltered = the report's files, or
# re-rendered with a filter / grouping key), so the UI and report.md always agree. Each figure is written as
# PNG + SVG + CSV (the plotted numbers).
FIGURES: dict = {}


def figure(name, title, group, ctx_keys=()):
    def deco(fn):
        FIGURES[name] = {"name": name, "title": title, "group": group, "fn": fn, "ctx_keys": list(ctx_keys)}
        return fn
    return deco


def _cfg_match(cfg, filt):
    for k, v in (filt or {}).items():
        node = cfg or {}
        for part in k.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if str(node) != str(v):
            return False
    return True


def sessions(B, suite=None, ctx=None):
    filt = (ctx or {}).get("filter") or {}
    out = []
    for e in B.m["sessions"]:
        d = B.s.get(e["session_id"])
        if d is None or (suite and e["suite"] != suite):
            continue
        if filt and not _cfg_match({**(d.get("config") or {}), **(e.get("params") or {})}, filt):
            continue
        out.append((e, d))
    return out


def _c1(B, ctx):
    runs = [(e, d) for e, d in sessions(B, "e1", ctx) if not e["params"].get("repeat")]
    return runs or [(e, d) for e, d in sessions(B, None, ctx) if d.get("epoch_metrics")]


def _band(ax, lists, color, label):
    """mean +/- std over runs, aligned by epoch. Returns csv rows [epoch, mean, std, n]."""
    by = defaultdict(list)
    for xs in lists:
        for ep, v in xs:
            if v is not None:
                by[ep].append(v)
    if not by:
        return []
    eps = sorted(by)
    m = np.array([np.mean(by[e]) for e in eps])
    sd = np.array([np.std(by[e]) for e in eps])
    ax.plot(eps, m, color=color, label=label)
    if any(len(by[e]) > 1 for e in eps):
        ax.fill_between(eps, m - sd, m + sd, color=color, alpha=0.18, lw=0)
    return [[e, float(a), float(b), len(by[e])] for e, a, b in zip(eps, m, sd)]


def _models_epochs(B, ctx):
    """{model: [[(epoch, metrics_dict), ...] per seed]} for C1 (networked), B0 and B1."""
    out = {"C1 VFL (networked)": [[(m["epoch"], m) for m in d.get("epoch_metrics", [])] for _, d in _c1(B, ctx)]}
    for key, name in (("b0_centralized", "B0 centralized"), ("b1_active_only", "B1 active-only")):
        lists = [[(x["epoch"], x) for x in b[key]["epochs"]] for b in B.baselines.values() if key in b]
        if lists:
            out[name] = lists
    return {k: v for k, v in out.items() if any(v)}


@figure("perf_vs_epoch", "AUROC, accuracy, F1 and loss vs epoch (mean ± std over seeds)", "model")
def fig_perf_vs_epoch(B, ctx):
    models = _models_epochs(B, ctx)
    if not models:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    rows = []
    for ax, (key, lab) in zip(axes.ravel(), (("auroc", "test AUROC"), ("accuracy", "test accuracy"),
                                             ("f1", "test F1"), ("loss", "test loss"))):
        for i, (model, lists) in enumerate(models.items()):
            pts = [[(ep, (m.get("test") or {}).get(key)) for ep, m in xs] for xs in lists]
            for r in _band(ax, pts, COLORS[i], model):
                rows.append([model, key] + r)
        ax.set(title=lab, xlabel="epoch")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return fig, (["model", "metric", "epoch", "mean", "std", "n_seeds"], rows)


@figure("roc_confusion", "ROC curves and confusion matrix (macro one-vs-rest for multi-class)", "model")
def fig_roc_confusion(B, ctx):
    models = _models_epochs(B, ctx)
    curves = {}
    for model, lists in models.items():
        for xs in lists:
            if xs and ((xs[-1][1].get("test") or {}).get("curves") or {}).get("roc"):
                curves[model] = xs[-1][1]["test"]["curves"]
                break
    if not curves:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    rows = []
    for i, (model, c) in enumerate(curves.items()):
        roc = c["roc"]
        ax[0].plot(roc["fpr"], roc["tpr"], color=COLORS[i], label=f"{model} ({roc['kind']})")
        rows += [[model, "roc", x, y] for x, y in zip(roc["fpr"], roc["tpr"])]
    ax[0].plot([0, 1], [0, 1], ls=":", color="#999", lw=0.8)
    ax[0].set(title="ROC (test split, final epoch)", xlabel="false positive rate", ylabel="true positive rate")
    ax[0].legend(fontsize=7)
    main_model = next(iter(curves))
    cm = (curves[main_model].get("confusion") or {}).get("matrix")
    if cm:
        cm = np.array(cm)
        ax[1].imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax[1].text(c, r, str(v), ha="center", va="center", fontsize=7 if cm.shape[0] > 6 else 9,
                       color="white" if v > cm.max() / 2 else "black")
        ax[1].set(title=f"Confusion matrix: {main_model}", xlabel="predicted class", ylabel="true class")
        ax[1].grid(False)
        rows += [[main_model, "confusion", r, c, int(v)] for (r, c), v in np.ndenumerate(cm)]
    else:
        ax[1].axis("off")
    fig.tight_layout()
    return fig, (["model", "kind", "x_or_true", "y_or_pred", "count"], rows)


@figure("time_to_target", "Time to target validation AUROC", "model")
def fig_time_to_target(B, ctx):
    rows = []
    for e, d in sessions(B, None, ctx):
        tgt = (d.get("config") or {}).get("target_val_auroc")
        if not tgt:
            continue
        t, cum = None, 0.0
        for m in d.get("epoch_metrics", []):
            cum += m.get("epoch_s") or 0.0
            if m.get("reached_target_s") is not None:
                t = m["reached_target_s"]
                break
            if (m.get("val") or {}).get("auroc") is not None and m["val"]["auroc"] >= tgt:
                t = cum
                break
        rows.append([f"{e['suite']}:{e['label']}", e["session_id"], tgt, t])
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(rows) + 2), 3.4))
    ys = [r[3] if r[3] is not None else 0 for r in rows]
    ax.bar(range(len(rows)), ys, color=[COLORS[0] if r[3] is not None else "#ccc" for r in rows])
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=60, ha="right", fontsize=7)
    for i, r in enumerate(rows):
        if r[3] is None:
            ax.text(i, 0, "not reached", rotation=90, ha="center", va="bottom", fontsize=7, color="#777")
    ax.set(title="Seconds of training until val AUROC ≥ target", ylabel="seconds")
    fig.tight_layout()
    return fig, (["run", "session", "target", "seconds"], rows)


@figure("lopo_contribution", "Leave-one-party-out contribution (AUROC drop without each party)", "model")
def fig_lopo(B, ctx):
    runs = sessions(B, "lopo", ctx)
    full, without = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for e, d in runs:
        auc = last_test(d).get("auroc")
        if auc is None:
            continue
        w = e["params"].get("without")
        (full[e["params"].get("seed")].append(auc) if not w else without[w][e["params"].get("seed")].append(auc))
    if not full or not without:
        return None
    rows = []
    for pid, by_seed in sorted(without.items()):
        drops = [mean(full[s]) - mean(v) for s, v in by_seed.items() if s in full]
        rows.append([pid, float(np.mean(drops)), float(np.std(drops)), len(drops)])
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.bar([r[0] for r in rows], [r[1] for r in rows], yerr=[r[2] for r in rows], color=COLORS[1], capsize=3)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set(title="Contribution = AUROC(all) − AUROC(without party)", ylabel="AUROC drop")
    fig.tight_layout()
    return fig, (["party_left_out", "mean_auroc_drop", "std", "n_seeds"], rows)


@figure("privacy_utility", "Privacy vs utility (label-leakage attack AUC vs test AUROC)", "model")
def fig_privacy_utility(B, ctx):
    rows = []
    for e, d in sessions(B, None, ctx):
        em = d.get("epoch_metrics") or []
        if not em or not em[-1].get("leakage"):
            continue
        dc = (d.get("config") or {}).get("defense") or {}
        lab = ("off" if dc.get("type", "none") == "none" else f"σ={dc.get('noise_multiplier')}")
        if dc.get("label_rr_epsilon") is not None:
            lab += f" ε={dc['label_rr_epsilon']}"
        rows.append([e["session_id"], lab, em[-1]["leakage"].get("cosine_leak_auc"),
                     em[-1]["leakage"].get("norm_leak_auc"), last_test(d).get("auroc")])
    rows = [r for r in rows if r[2] is not None and r[4] is not None]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for r in rows:
        c = COLORS[0] if r[1] == "off" else (COLORS[3] if "ε" in r[1] else COLORS[4])
        ax.scatter(r[2], r[4], color=c, s=36)
        ax.annotate(r[1], (r[2], r[4]), textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.axvline(LEAK_SUCCESS, color="#999", ls=":", lw=0.8)
    ax.set(title="Privacy–utility (cosine attack; right of the line = attack succeeded)",
           xlabel="attack AUC (0.5 = no leakage)", ylabel="test AUROC")
    fig.tight_layout()
    return fig, (["session", "defence", "cosine_leak_auc", "norm_leak_auc", "test_auroc"], rows)


LEAK_SUCCESS = 0.6


def _group_rows(B, ctx, key):
    rows = [r for r in _load_index(B.path) if r.get("test_auroc") is not None
            and _cfg_match({**(r.get("config") or {}), "suite": r.get("suite")}, (ctx or {}).get("filter"))]
    groups = defaultdict(list)
    for r in rows:
        v = (r.get("config") or {}).get(key, r.get(key))
        groups[json.dumps(v, sort_keys=True, default=str) if isinstance(v, (dict, list)) else str(v)].append(r["test_auroc"])
    return groups


def _bar_ci(groups, key, title):
    if not groups:
        return None
    rows = []
    for g, vals in sorted(groups.items()):
        m, sd = float(np.mean(vals)), float(np.std(vals))
        ci = 1.96 * sd / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        rows.append([g, m, sd, ci, len(vals)])
    fig, ax = plt.subplots(figsize=(max(4.5, 0.8 * len(rows) + 2), 3.4))
    ax.bar(range(len(rows)), [r[1] for r in rows], yerr=[r[3] for r in rows], color=COLORS[0], capsize=3)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=30, ha="right", fontsize=7)
    lo = min(r[1] - r[3] for r in rows)
    ax.set_ylim(max(0, lo - 0.05), min(1.0, max(r[1] + r[3] for r in rows) + 0.03))
    for i, r in enumerate(rows):
        ax.text(i, r[1], f"n={r[4]}", ha="center", va="bottom", fontsize=7)
    ax.set(title=title, xlabel=key, ylabel="test AUROC (mean ± 95% CI)")
    fig.tight_layout()
    return fig, ([key, "mean_auroc", "std", "ci95", "n_runs"], rows)


@figure("compare_by", "Test AUROC grouped by any config key (mean ± 95% CI)", "model", ctx_keys=["group"])
def fig_compare_by(B, ctx):
    key = (ctx or {}).get("group") or "bottom_model"
    return _bar_ci(_group_rows(B, ctx, key), key, f"Test AUROC by {key}")


@figure("by_bottom_model", "Per-bottom-model comparison", "model")
def fig_by_bottom_model(B, ctx):
    return _bar_ci(_group_rows(B, ctx, "bottom_model"), "bottom_model", "Test AUROC by bottom model")


@figure("by_partition", "Per-partition comparison", "model")
def fig_by_partition(B, ctx):
    return _bar_ci(_group_rows(B, ctx, "partition"), "partition", "Test AUROC by partition strategy")


def _e2_agg(B, ctx):
    grid = defaultdict(list)
    for e, d in sessions(B, "e2", ctx):
        ps = fin(d, "passive")
        if not ps:
            continue
        p = ps[0]
        bd, _ = passive_breakdown(B, e["session_id"])
        n_ep = max(1, len(p.get("epochs") or []))
        net = (d.get("netem") or {}).get("bytes_total")
        n_train = p.get("n_train_entities") or 1
        grid[(e["params"]["batch_size"], e["params"]["rtt_ms"])].append({
            "epoch_s": p.get("epoch_s_median"), "p50": p.get("batch_ms_p50"), "p95": p.get("batch_ms_p95"),
            "thr": p.get("throughput_batches_per_s"), "tensor": p.get("mib_per_epoch_tensor"),
            "app": p.get("mib_per_epoch_app"), "net": net / 2 ** 20 / n_ep if net else None,
            "n_train": n_train, "bd": bd})
    agg = {}
    for k, recs in grid.items():
        agg[k] = {key: mean([r[key] for r in recs]) for key in recs[0] if key != "bd"}
        agg[k]["bd"] = {c: mean([r["bd"].get(c) for r in recs]) for c in recs[0]["bd"]}
    return agg


@figure("latency_vs_rtt", "Epoch time and p50/p95 batch time vs RTT and batch size", "systems")
def fig_latency(B, ctx):
    agg = _e2_agg(B, ctx)
    if not agg:
        return None
    bss, rtts = sorted({k[0] for k in agg}), sorted({k[1] for k in agg})
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    rows = []
    for i, bs in enumerate(bss):
        xs = [r for r in rtts if (bs, r) in agg]
        ax[0].plot(xs, [agg[(bs, r)]["epoch_s"] for r in xs], "o-", color=COLORS[i % 6], label=f"batch {bs}")
        ax[1].plot(xs, [agg[(bs, r)]["p95"] for r in xs], "o-", color=COLORS[i % 6], label=f"p95 bs{bs}")
        ax[1].plot(xs, [agg[(bs, r)]["p50"] for r in xs], "o--", color=COLORS[i % 6], alpha=0.55)
        rows += [[bs, r, agg[(bs, r)]["epoch_s"], agg[(bs, r)]["p50"], agg[(bs, r)]["p95"]] for r in xs]
    for j, r in enumerate(rtts):
        xs = [bs for bs in bss if (bs, r) in agg]
        ax[2].plot(xs, [agg[(bs, r)]["p95"] for bs in xs], "o-", color=COLORS[j % 6], label=f"RTT {r:g} ms")
    ax[0].set(title="Median epoch time vs RTT", xlabel="RTT (ms)", ylabel="s")
    ax[1].set(title="Batch time p95 (solid) / p50 (dashed)", xlabel="RTT (ms)", ylabel="ms")
    ax[2].set(title="p95 batch time vs batch size", xlabel="batch size", ylabel="ms")
    ax[0].legend(fontsize=7)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    return fig, (["batch_size", "rtt_ms", "epoch_s_median", "batch_ms_p50", "batch_ms_p95"], rows)


@figure("throughput", "Throughput (committed batches/s) vs RTT", "systems")
def fig_throughput(B, ctx):
    agg = _e2_agg(B, ctx)
    if not agg:
        return None
    bss, rtts = sorted({k[0] for k in agg}), sorted({k[1] for k in agg})
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    rows = []
    for i, bs in enumerate(bss):
        xs = [r for r in rtts if (bs, r) in agg]
        ax.plot(xs, [agg[(bs, r)]["thr"] for r in xs], "o-", color=COLORS[i % 6], label=f"batch {bs}")
        rows += [[bs, r, agg[(bs, r)]["thr"]] for r in xs]
    ax.axhline(5, color="#999", ls=":", lw=0.8)
    ax.set(title="Throughput vs RTT (NFR-5 target 5/s dotted)", xlabel="RTT (ms)", ylabel="batches / s")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, (["batch_size", "rtt_ms", "batches_per_s"], rows)


@figure("time_breakdown", "Stacked batch-time breakdown (passive view)", "systems")
def fig_breakdown(B, ctx):
    agg = _e2_agg(B, ctx)
    if not agg:
        return None
    bss, rtts = sorted({k[0] for k in agg}), sorted({k[1] for k in agg})
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    rows = []
    for panel, (keys, labels, title) in enumerate((
            ([(bss[len(bss) // 2], r) for r in rtts], [f"{r:g} ms" for r in rtts], f"by RTT (batch {bss[len(bss) // 2]})"),
            ([(bs, rtts[0]) for bs in bss], [f"bs {bs}" for bs in bss], f"by batch size (RTT {rtts[0]:g} ms)"))):
        keys = [k for k in keys if k in agg]
        comps = list(agg[keys[0]]["bd"]) if keys else []
        bottom = np.zeros(len(keys))
        for j, c in enumerate(comps):
            vals = np.array([agg[k]["bd"].get(c) or 0 for k in keys])
            if vals.sum() <= 0:
                continue
            ax[panel].bar(labels[:len(keys)], vals, bottom=bottom, color=(COLORS + ["#9aa", "#bbb", "#ccd", "#dde"])[j % 10],
                          label=c.replace("_ms", ""))
            bottom += vals
            rows += [[title, lab, c, float(v)] for lab, v in zip(labels, vals)]
        ax[panel].set(title=f"Where batch time goes, {title}", ylabel="mean ms / batch")
    ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return fig, (["panel", "x", "component", "ms"], rows)


@figure("bytes", "Tensor, application and wire bytes per epoch and per entity", "systems")
def fig_bytes(B, ctx):
    agg = _e2_agg(B, ctx)
    if not agg:
        return None
    bss, r0 = sorted({k[0] for k in agg}), sorted({k[1] for k in agg})[0]
    bss = [bs for bs in bss if (bs, r0) in agg]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    x, w, rows = np.arange(len(bss)), 0.26, []
    for j, (key, name) in enumerate((("tensor", "tensor payload"), ("app", "serialized app"), ("net", "on the wire"))):
        per_ep = [agg[(bs, r0)][key] or 0 for bs in bss]
        per_ent = [(agg[(bs, r0)][key] or 0) * 1024 / (agg[(bs, r0)]["n_train"] or 1) for bs in bss]
        ax[0].bar(x + (j - 1) * w, per_ep, w, color=COLORS[j], label=name)
        ax[1].bar(x + (j - 1) * w, per_ent, w, color=COLORS[j], label=name)
        rows += [[bs, name, a, b] for bs, a, b in zip(bss, per_ep, per_ent)]
    for a_, t, u in ((ax[0], "Traffic per epoch", "MiB"), (ax[1], "Traffic per training entity per epoch", "KiB")):
        a_.set_xticks(x, [f"batch {bs}" for bs in bss])
        a_.set(title=t, ylabel=u)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    return fig, (["batch_size", "kind", "mib_per_epoch", "kib_per_entity_epoch"], rows)


@figure("resources", "CPU and RSS per node over time", "systems", ctx_keys=["session"])
def fig_resources(B, ctx):
    cands = [(e, d) for e, d in sessions(B, None, ctx)
             if any((x.get("resources") or {}).get("t") for rows_ in (d.get("party_epochs") or {}).values() for x in rows_)]
    want = (ctx or {}).get("session")
    if want:
        cands = [c for c in cands if c[0]["session_id"] == want]
    if not cands:
        fin_rows = []
        for e, d in sessions(B, None, ctx):
            for pid, f_ in (d.get("finals") or {}).items():
                r = f_.get("resources") or {}
                if r.get("cpu_avg_percent") is not None:
                    fin_rows.append([e["session_id"], pid, r["cpu_avg_percent"], r.get("cpu_max_percent"),
                                     r.get("rss_avg_mib"), r.get("rss_max_mib")])
        if not fin_rows:
            return None
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        last = fin_rows[-len({r[1] for r in fin_rows}):]
        ax[0].bar([r[1] for r in last], [r[2] for r in last], color=COLORS[0])
        ax[1].bar([r[1] for r in last], [r[4] for r in last], color=COLORS[1])
        ax[0].set(title="Mean CPU per node (no per-epoch samples in these runs)", ylabel="%")
        ax[1].set(title="Mean RSS per node", ylabel="MiB")
        fig.tight_layout()
        return fig, (["session", "node", "cpu_avg", "cpu_max", "rss_avg_mib", "rss_max_mib"], fin_rows)
    e, d = cands[-1]
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True)
    rows, t0 = [], d.get("started") or min(x["resources"]["t"] for v in d["party_epochs"].values() for x in v
                                             if (x.get("resources") or {}).get("t"))
    for i, (pid, eps) in enumerate(sorted(d["party_epochs"].items())):
        pts = [(x["resources"]["t"] - t0, x["resources"].get("cpu_percent"), x["resources"].get("rss_mib"))
               for x in eps if (x.get("resources") or {}).get("t")]
        if not pts:
            continue
        ax[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=COLORS[i % 6], label=pid, ms=3)
        ax[1].plot([p[0] for p in pts], [p[2] for p in pts], "o-", color=COLORS[i % 6], ms=3)
        rows += [[pid, *p] for p in pts]
    ax[0].set(title=f"CPU per node ({e['session_id']}, sampled at each epoch end)", ylabel="CPU %")
    ax[1].set(title="Resident memory per node", xlabel="seconds since session start", ylabel="RSS MiB")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    return fig, (["node", "t_s", "cpu_percent", "rss_mib"], rows)


def _device(d, pid):
    env = ((d.get("finals") or {}).get(pid) or {}).get("env") or {}
    return f"{env.get('cpu_count', '?')} CPU {str(env.get('platform', '')).split('-')[0]}".strip()


@figure("slot_wait_stragglers", "Slot wait and dropped batches vs number of parties and device type", "systems")
def fig_slot_wait(B, ctx):
    rows = []
    for e, d in sessions(B, None, ctx):
        a = fin(d, "active") or {}
        if a.get("wait_parties_ms_mean") is None:
            continue
        pol = ((d.get("config") or {}).get("straggler") or {}).get("policy", "wait")
        dropped = sum((p.get("counters") or {}).get("dropped_batches", 0) for p in (fin(d, "passive") or []))
        devices = sorted({_device(d, pid) for pid in parts(d)})
        rows.append([e["session_id"], len(parts(d)), pol, " / ".join(devices), a["wait_parties_ms_mean"], dropped])
    if not rows:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    keys = sorted({(r[2], r[3]) for r in rows})
    for i, k in enumerate(keys):
        pts = sorted((r[1], r[4], r[5]) for r in rows if (r[2], r[3]) == k)
        lab = f"{k[0]} · {k[1]}"
        ax[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=COLORS[i % 6], label=lab)
        ax[1].plot([p[0] for p in pts], [p[2] for p in pts], "o-", color=COLORS[i % 6], label=lab)
    ax[0].set(title="Active-party slot wait", xlabel="parties", ylabel="mean wait (ms)")
    ax[1].set(title="Straggler batches dropped (drop policy)", xlabel="parties", ylabel="batches")
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    return fig, (["session", "n_parties", "straggler_policy", "devices", "slot_wait_ms", "dropped_batches"], rows)


@figure("scaling", "Scaling with parties, overlap and partition", "systems")
def fig_scaling(B, ctx):
    pts = defaultdict(list)
    for e, d in sessions(B, "sweep", ctx):
        p = e["params"]
        pts[(p.get("partition"), p.get("overlap"))].append(
            (p.get("n_parties"), last_test(d).get("auroc"), (fin(d, "active") or {}).get("wait_parties_ms_mean"),
             (d.get("finished") or 0) - (d.get("started") or 0)))
    if not pts:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    rows = []
    for i, (key, vs) in enumerate(sorted(pts.items(), key=lambda kv: str(kv[0]))):
        vs = sorted(v for v in vs if v[0] is not None)
        lab = f"{key[0]} ov{key[1]}"
        for j in range(3):
            ax[j].plot([v[0] for v in vs], [v[j + 1] for v in vs], "o-", color=COLORS[i % 6], label=lab)
        rows += [[key[0], key[1], *v] for v in vs]
    ax[0].set(title="Utility vs parties", xlabel="parties", ylabel="test AUROC")
    ax[1].set(title="Slot wait vs parties", xlabel="parties", ylabel="ms")
    ax[2].set(title="Session wall time vs parties", xlabel="parties", ylabel="s")
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    return fig, (["partition", "overlap", "n_parties", "test_auroc", "slot_wait_ms", "wall_s"], rows)


@figure("codec_comparison", "Codec comparison (fp32 / fp16 / int8 / top-k)", "systems")
def fig_codec(B, ctx):
    by = defaultdict(list)
    for e, d in sessions(B, None, ctx):
        c = (d.get("config") or {}).get("codec")
        p = (fin(d, "passive") or [None])[0]
        if not c or not p or e["suite"] not in ("eff", "e1", "codec"):
            continue
        by[c].append((p.get("mib_per_epoch_tensor"), p.get("mib_per_epoch_app"), last_test(d).get("auroc"),
                      p.get("epoch_s_median")))
    if len(by) < 2:
        return None
    order = [c for c in ("fp32", "fp16", "int8", "topk") if c in by]
    rows = [[c] + [mean([v[i] for v in by[c]]) for i in range(4)] + [len(by[c])] for c in order]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].bar(order, [r[1] or 0 for r in rows], color=COLORS[0], label="tensor")
    ax[0].bar(order, [(r[2] or 0) - (r[1] or 0) for r in rows], bottom=[r[1] or 0 for r in rows], color=COLORS[2],
              label="protobuf overhead")
    ax[1].bar(order, [r[3] or 0 for r in rows], color=COLORS[1])
    ax[1].set_ylim(max(0, min(r[3] or 1 for r in rows) - 0.05), 1.0)
    ax[2].bar(order, [r[4] or 0 for r in rows], color=COLORS[3])
    ax[0].set(title="MiB per epoch", ylabel="MiB")
    ax[1].set(title="Test AUROC", ylabel="AUROC")
    ax[2].set(title="Median epoch time", ylabel="s")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    return fig, (["codec", "mib_tensor", "mib_app", "test_auroc", "epoch_s", "n_runs"], rows)


@figure("psi_cost", "Alignment cost: DH-PSI vs non-private experimental", "systems")
def fig_psi(B, ctx):
    rows = []
    for e, d in sessions(B, None, ctx):
        for pid, a in (d.get("alignment") or {}).items():
            if a.get("align_ms") is None:
                continue
            rows.append([a.get("mode") or (d.get("config") or {}).get("alignment"), pid, a.get("align_ms"),
                         (a.get("app_bytes_up") or 0) + (a.get("app_bytes_down") or 0), a.get("n_aligned")])
    modes = sorted({r[0] for r in rows})
    if not rows or ("dh_psi" not in modes and len(modes) < 2):
        return None
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(modes, [mean([r[2] for r in rows if r[0] == m]) for m in modes], color=COLORS[0])
    ax[1].bar(modes, [mean([r[3] for r in rows if r[0] == m]) / 1024 for m in modes], color=COLORS[2])
    ax[0].set(title="Alignment time per party", ylabel="ms")
    ax[1].set(title="Alignment bytes per party", ylabel="KiB")
    fig.tight_layout()
    return fig, (["mode", "party", "align_ms", "bytes", "n_aligned"], rows)


@figure("e5_timeline", "E5 fault timeline (batch time with injected faults)", "systems")
def fig_e5(B, ctx):
    runs = sessions(B, "e5", ctx)
    if not runs:
        return None
    fig, axes = plt.subplots(len(runs), 1, figsize=(9, 2.1 * len(runs) + 0.4), squeeze=False)
    rows = []
    for ax, (e, d) in zip(axes[:, 0], runs):
        recs = sorted([r for r in B.batches(e["session_id"], "passive") if r.get("phase") == "train"],
                      key=lambda r: (r.get("epoch", 0), r.get("batch", 0), r.get("t", 0)))
        if not recs:
            ax.set(title=f"{e['label']}: no batch records")
            continue
        nb = max(r.get("batch", 0) for r in recs) + 1
        xs = [r.get("epoch", 0) * nb + r.get("batch", 0) for r in recs]
        ax.plot(xs, [r.get("batch_ms") for r in recs], lw=0.9, color=COLORS[0])
        fault = (d.get("config") or {}).get("fault") or {}
        for kind, col in (("delay", COLORS[4]), ("duplicate", COLORS[2])):
            for ep, b in fault.get(kind) or []:
                ax.axvline(ep * nb + b, color=col, ls="--", lw=0.9, label=f"{kind} injected")
        se = (d.get("config") or {}).get("start_epoch") or 0
        if se:
            ax.axvline(se * nb, color=COLORS[3], ls=":", lw=1, label="resumed here")
        ax.set(title=f"{e['label']} ({d['state']})", ylabel="batch ms")
        h, lab = ax.get_legend_handles_labels()
        if h:
            ax.legend(dict(zip(lab, h)).values(), dict(zip(lab, h)).keys(), fontsize=6, loc="upper right")
        rows += [[e["label"], x, r.get("epoch"), r.get("batch"), r.get("batch_ms")] for x, r in zip(xs, recs)]
    axes[-1, 0].set(xlabel="global batch index")
    fig.tight_layout()
    return fig, (["run", "index", "epoch", "batch", "batch_ms"], rows)


@figure("security_passfail", "Security conformance: pass/fail per attack", "systems")
def fig_security(B, ctx):
    rows = []
    path = os.path.join(B.path, "security", "conformance.json")
    if os.path.exists(path):
        rows += [[r["case"], r.get("expected"), r.get("outcome"), bool(r.get("as_expected"))]
                 for r in json.load(open(path)).get("table", [])]
    for rp in sorted(glob.glob(os.path.join(B.path, "attacks", "*", "report.json"))):
        rows += [[r["attack"], r["expected"], r["observed"], bool(r["as_expected"])] for r in json.load(open(rp))["rows"]]
    if not rows:
        return None
    latest = {}
    for r in rows:
        latest[r[0]] = r
    rows = list(latest.values())
    fig, ax = plt.subplots(figsize=(max(5, 0.42 * len(rows) + 2), 3.3))
    ax.bar(range(len(rows)), [1] * len(rows), color=[COLORS[1] if r[3] else COLORS[4] for r in rows])
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=60, ha="right", fontsize=7)
    for i, r in enumerate(rows):
        ax.text(i, 0.5, r[2], rotation=90, ha="center", va="center", fontsize=7, color="white")
    ax.set_yticks([])
    ok = sum(r[3] for r in rows)
    ax.set(title=f"{ok}/{len(rows)} cases as designed (green = as expected, red = not)")
    fig.tight_layout()
    return fig, (["attack", "expected", "observed", "as_expected"], rows)


def render_figure(B, name, outdir, ctx=None, formats=("png", "svg")):
    """Render one registered figure into outdir as <name>.png/.svg/.csv. Returns paths, or None (no data)."""
    spec = FIGURES[name]
    try:
        res = spec["fn"](B, ctx or {})
    except Exception as ex:                                   # one broken figure must not stop the report
        plt.close("all")
        return {"error": f"{type(ex).__name__}: {ex}"}
    if res is None:
        return None
    fig, (header, rows) = res
    _integer_axes(fig)
    os.makedirs(outdir, exist_ok=True)
    out = {}
    for fmt in formats:
        pth = os.path.join(outdir, f"{name}.{fmt}")
        fig.savefig(pth)
        out[fmt] = pth
    plt.close(fig)
    write_csv(outdir, name, header, rows)
    out["csv"] = os.path.join(outdir, f"{name}.csv")
    return out


def render_all(B, outdir, ctx=None) -> dict:
    return {name: render_figure(B, name, outdir, ctx) for name in FIGURES}


def compare_sessions(da: dict, db: dict, la: str, lb: str, outdir: str | None = None):
    """Side-by-side of two runs: config differences, final metrics, and AUROC/loss curves overlaid."""
    def flat(d, p=""):
        o = {}
        for k, v in (d or {}).items():
            if isinstance(v, dict) and k in ("defense", "straggler", "security", "fault", "split"):
                o.update(flat(v, p + k + "."))
            else:
                o[p + k] = v
        return o
    ca, cb = flat(da.get("config")), flat(db.get("config"))
    diff = [[k, ca.get(k), cb.get(k)] for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k)]
    ta, tb = last_test(da), last_test(db)
    metrics = [[k, ta.get(k), tb.get(k)] for k in ("auroc", "accuracy", "f1", "loss")]
    for role in ("passive",):
        pa, pb = (fin(da, role) or [{}])[0], (fin(db, role) or [{}])[0]
        for k in ("epoch_s_median", "batch_ms_p50", "batch_ms_p95", "mib_per_epoch_app"):
            metrics.append([k, pa.get(k), pb.get(k)])
    paths = None
    if outdir:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        for d, lab, c in ((da, la, COLORS[0]), (db, lb, COLORS[2])):
            em = d.get("epoch_metrics") or []
            ax[0].plot([m["epoch"] for m in em], [(m.get("test") or {}).get("auroc") for m in em], "o-", color=c, label=lab, ms=3)
            ax[1].plot([m["epoch"] for m in em], [m.get("train_loss") for m in em], "o-", color=c, ms=3)
        ax[0].set(title="Test AUROC", xlabel="epoch")
        ax[1].set(title="Training loss", xlabel="epoch")
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        os.makedirs(outdir, exist_ok=True)
        paths = {}
        _integer_axes(fig)
        for fmt in ("png", "svg"):
            paths[fmt] = os.path.join(outdir, f"compare.{fmt}")
            fig.savefig(paths[fmt])
        plt.close(fig)
        write_csv(outdir, "compare", ["field", la, lb], diff + metrics)
        paths["csv"] = os.path.join(outdir, "compare.csv")
    return {"config_diff": diff, "metrics": metrics, "figure": paths}


NFR_EVIDENCE = {"NFR-1": ["e1"], "NFR-2": ["e1"], "NFR-3": ["e2"], "NFR-4": ["e2"], "NFR-5": ["e2"], "NFR-6": ["eff"],
                "NFR-7": ["security"], "NFR-8": ["def", "psi"], "NFR-9": [], "NFR-10": ["e5"]}


def nfr_evidence(B) -> dict:
    """NFR -> the run ids that are its evidence (for the scorecard links)."""
    return {k: [e["session_id"] for e in B.m["sessions"] if e["suite"] in suites] for k, suites in NFR_EVIDENCE.items()}


def _load_index(bench_path):
    try:
        return json.load(open(os.path.join(bench_path, "index.json")))
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--security", help="security_report.json from bench/security_test.py")
    ap.add_argument("--cost", help="cloud price snapshot YAML (see configs/cloud_cost.example.yaml)")
    a = ap.parse_args()
    B = Bench(a.bench)
    out = os.path.join(B.path, "report")
    os.makedirs(out, exist_ok=True)
    R = {"nfr": {}, "tables": {}}
    analyse_e1(B, R, out)
    analyse_e2(B, R, out)
    analyse_eff(B, R, out)
    analyse_def(B, R, out)
    analyse_psi(B, R)
    analyse_e5(B, R)
    analyse_sweep(B, R, out)
    analyse_security(R, a.security)
    analyse_conformance(B, R)
    analyse_cost(B, R, a.cost)
    names = {"NFR-1": "Utility", "NFR-2": "Reproducibility", "NFR-3": "Latency", "NFR-4": "Communication",
             "NFR-5": "Throughput", "NFR-6": "Efficiency", "NFR-7": "Transport & logging",
             "NFR-8": "Conditional privacy evaluation", "NFR-9": "Conditional cloud cost", "NFR-10": "Fault handling"}
    nfr_rows = [[k, names[k], R["nfr"].get(k, {}).get("status", "not attempted"), R["nfr"].get(k, {}).get("value", "")]
                for k in names]
    with open(os.path.join(out, "nfr_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "status", "measured"])
        w.writerows(nfr_rows)
    nfr_md = md_table(["NFR", "name", "status", "measured value"], nfr_rows)
    with open(os.path.join(out, "nfr_table.md"), "w") as fh:
        fh.write(nfr_md + "\n")
    req = requirements_matrix(B, R)
    with open(os.path.join(out, "requirements_matrix.md"), "w") as fh:
        fh.write(req + "\n")
    env = next(iter(next(iter(B.s.values()))["finals"].values()), {}).get("env", {}) if B.s else {}
    figs = sorted(x for x in os.listdir(out) if x.endswith(".png"))
    catalogue = render_all(B, os.path.join(out, "figures"))
    R["figures"] = {k: (v or {}).get("png") and os.path.relpath(v["png"], out) for k, v in catalogue.items()}
    R["nfr_evidence"] = nfr_evidence(B)
    lines = ["# VFL benchmark report", "",
             f"Benchmark: `{B.path}`  ·  created {B.m['created']}  ·  {len(B.s)} sessions  ·  wall {f(B.m.get('wall_s'), 0)} s",
             f"Environment: python {env.get('python')}, torch {env.get('torch')}, grpc {env.get('grpc')}, "
             f"host {env.get('hostname')} ({env.get('cpu_count')} CPUs), torch threads {env.get('threads')}, "
             f"deterministic={env.get('deterministic')}", "",
             f"Data: {json.dumps({k: (B.data_report or {}).get(k) for k in ('dataset', 'n_total', 'n_aligned_expected', 'overlap_ratio_actual', 'partition')})}",
             "", "## NFR measured-value / pass / fail / not-attempted", "", nfr_md, "",
             "## Requirement → experiment matrix", "", req, ""]
    for title, tbl in R["tables"].items():
        lines += [f"## {title}", "", tbl, ""]
    lines += ["## Figures", ""] + [f"![{x}]({x})" for x in figs]
    lines += ["", "## Figure catalogue (same functions as the UI's Analytics page)", ""]
    for name, spec in FIGURES.items():
        res = catalogue.get(name)
        if res and res.get("png"):
            lines += [f"### {spec['title']}", "", f"![{name}](figures/{name}.png)  ",
                      f"[SVG](figures/{name}.svg) · [CSV](figures/{name}.csv)", ""]
        else:
            lines += [f"### {spec['title']}", "", "_no data in this benchmark_" if not res else f"_error: {res.get('error')}_", ""]
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    R.pop("tables")
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(R, fh, indent=2, default=str)
    print(nfr_md)
    print(f"\nreport: {os.path.join(out, 'report.md')}  ({len(figs)} figures)")


if __name__ == "__main__":
    main()
