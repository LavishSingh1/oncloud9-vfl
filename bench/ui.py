"""Benchmark, test and analytics backend (PRD v2 items 25-31; extended in v2.1 item 5).

Served by `python -m vfl.cli ui` (the unified UI, port 8090) and, for compatibility, `vfl.cli bench-ui`.
Reads results/ (run index + analyze.py reports) and coordinator runs/ folders, so it works offline.

  Tests        run pytest (all / not-slow / one file) as a job: live output + a pass/fail table (JUnit XML)
  Launcher     every suite of bench/run_benchmarks.py (e1, e2, eff, def, psi, e5, sweep, lopo, security) with
               grids chosen from dropdowns / multi-selects; jobs run one at a time from a queue, with progress,
               cancel and rerun; each finished job is analysed with bench/analyze.py
  Runs         filter / group by any config key, mean +/- std and 95% CI over seeds, compare two runs
  Analytics    every figure from bench/analyze.py's registry (same functions as report.md), filterable,
               as PNG / SVG / CSV
  Scorecard    the NFR table, each NFR linked to its evidence runs
  Export       report bundle (report.md + figures + CSV + index)
"""
from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import re
import sys
import threading
import time
import zipfile

from flask import Flask, jsonify, request, send_file, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
STATIC = os.path.join(ROOT, "vfl", "webui", "static")
SUITES = ["e1", "e2", "eff", "def", "psi", "e5", "sweep", "lopo", "security"]
SUITE_HELP = {"e1": "correctness & utility: B0/B1/local-VFL baselines + networked C1 over seeds, plus a repeat",
              "e2": "network & efficiency: RTT x batch-size sweep", "eff": "codec comparison (fp32/fp16/int8/top-k)",
              "def": "inference defence on/off + noise sweep", "psi": "DH-PSI vs experimental alignment (any number of parties)",
              "e5": "faults: duplicate, delay, stop + resume", "sweep": "scaling: parties x partition x overlap",
              "lopo": "leave-one-party-out contribution (needs >= 2 passives)",
              "security": "security conformance suite on a dedicated deployment"}
_MPL_LOCK = threading.Lock()


def _multi(values):
    return {"widget": "multi", "options": list(values)}


# The launcher form: one entry per run_benchmarks.py flag (tests/test_bench_ui.py checks the two agree).
BENCH_OPTIONS = {
    "suites": {"flag": "--suites", "default": ["e1"], **_multi(SUITES), "help": "suites to run, in order"},
    "dataset": {"flag": "--dataset", "default": "breast_cancer", "widget": "select",
                "options": ["breast_cancer", "synthetic", "digits", "adult", "bank", "credit_default", "mnist",
                            "fashion_mnist"]},
    "data_dir": {"flag": "--data-dir", "default": None, "widget": "dataset",
                 "help": "use a saved dataset instead of generating one"},
    "partitioner": {"flag": "--partitioner", "default": None, "widget": "select", "options": [None, "v1", "v2"]},
    "partition": {"flag": "--partition", "default": None, "widget": "select",
                  "options": [None, "random", "sequential", "dirichlet", "importance", "correlated", "image_strips",
                              "image_grid", "importance_skewed"]},
    "overlap": {"flag": "--overlap", "default": None, "widget": "select", "options": [None, 0.5, 0.8, 0.9, 1.0]},
    "binarize": {"flag": "--binarize", "default": None, "widget": "select", "options": [None, "odd_even", "pair:3,8"]},
    "n": {"flag": "--n", "default": 20000, "widget": "select", "options": [2000, 5000, 20000, 50000]},
    "features": {"flag": "--features", "default": 40, "widget": "select", "options": [10, 20, 40, 80]},
    "config": {"flag": "--config", "default": None, "widget": "config", "help": "session YAML (configs/*.yaml)"},
    "passives": {"flag": "--passives", "default": 1, "widget": "select", "options": list(range(1, 9))},
    "insecure": {"flag": "--insecure", "default": False, "widget": "toggle"},
    "rtt_mode": {"flag": "--rtt-mode", "default": None, "widget": "select", "options": [None, "netem", "app"]},
    "url": {"flag": "--url", "default": None, "widget": "coordinator",
            "help": "remote mode: drive a running deployment (from the Deployment panel)"},
    "epochs": {"flag": "--epochs", "default": 20, "widget": "select", "options": [3, 5, 10, 20, 30, 50]},
    "sweep_epochs": {"flag": "--sweep-epochs", "default": 3, "widget": "select", "options": [1, 2, 3, 5, 10]},
    "fault_epochs": {"flag": "--fault-epochs", "default": 3, "widget": "select", "options": [3, 4, 5]},
    "seeds": {"flag": "--seeds", "default": [1, 2, 3], **_multi(range(1, 11))},
    "sweep_seeds": {"flag": "--sweep-seeds", "default": [1], **_multi(range(1, 6))},
    "rtts": {"flag": "--rtts", "default": [0, 20, 50, 100], **_multi([0, 5, 10, 20, 50, 100, 200, 500])},
    "batch_sizes": {"flag": "--batch-sizes", "default": [32, 64, 128], **_multi([16, 32, 64, 128, 256, 512, 1024])},
    "noise": {"flag": "--noise", "default": [0.5, 1, 2], **_multi([0.25, 0.5, 1, 2, 4, 8])},
    "sweep_parties": {"flag": "--sweep-parties", "default": [2, 3, 4], **_multi(range(2, 10))},
    "sweep_partitions": {"flag": "--sweep-partitions", "default": ["random", "dirichlet"],
                         **_multi(["random", "sequential", "dirichlet", "importance", "correlated"])},
    "sweep_overlaps": {"flag": "--sweep-overlaps", "default": [1.0, 0.8], **_multi([1.0, 0.9, 0.8, 0.6, 0.5])},
    "sweep_rtt": {"flag": "--sweep-rtt", "default": 0.0, "widget": "select", "options": [0.0, 20.0, 50.0, 100.0]},
    "clip": {"flag": "--clip", "default": 0.05, "widget": "select", "options": [0.01, 0.02, 0.05, 0.1, 0.5]},
    "codecs": {"flag": "--codecs", "default": ["fp32", "fp16"], **_multi(["fp32", "fp16", "int8", "topk"])},
}
BENCH_NOT_IN_FORM = {"out": "the run name decides the output folder"}


def run_benchmarks_flags() -> set:
    """Every --flag bench/run_benchmarks.py defines (read from its source: the UI must not import torch)."""
    tree = ast.parse(open(os.path.join(ROOT, "bench", "run_benchmarks.py")).read())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument" and n.args:
            a0 = n.args[0]
            if isinstance(a0, ast.Constant) and str(a0.value).startswith("--"):
                out.add(a0.value)
    return out


def bench_argv(spec: dict, out: str) -> list[str]:
    argv = [os.path.join("bench", "run_benchmarks.py"), "--out", out]
    suites = spec.get("suites") or ([spec["suite"]] if spec.get("suite") else ["e1"])
    if isinstance(suites, str):
        suites = suites.split(",")
    bad = [x for x in suites if x not in SUITES]
    if bad:
        raise ValueError(f"unknown suite(s) {bad}; choose from {SUITES}")
    argv += ["--suites", ",".join(suites)]
    for key, o in BENCH_OPTIONS.items():
        if key == "suites" or key not in spec:
            continue
        v = spec[key]
        if v in (None, "", []):
            continue
        if o["widget"] == "toggle":
            if v is True:
                argv.append(o["flag"])
            continue
        if o["widget"] == "multi":
            vals = v if isinstance(v, list) else str(v).split(",")
            bad = [x for x in vals if x not in o["options"] and _num(x) not in o["options"]]
            if bad:
                raise ValueError(f"{key}: {bad} not offered")
            v = ",".join(str(x) for x in vals)
        elif o["widget"] == "select" and v not in o["options"] and _num(v) not in o["options"]:
            raise ValueError(f"{key}={v!r} not offered")
        argv += [o["flag"], str(v)]
    return argv


def _num(x):
    try:
        f = float(x)
        return int(f) if f.is_integer() and "." not in str(x) else f
    except (TypeError, ValueError):
        return x


class BenchUI:
    def __init__(self, results_dir: str, run_dirs: list[str], workspace: str | None = None):
        from vfl.webui.jobs import JobManager
        self.results = os.path.abspath(results_dir)
        self.run_dirs = [os.path.abspath(r) for r in run_dirs]
        self.workspace = os.path.abspath(workspace) if workspace else None
        os.makedirs(self.results, exist_ok=True)
        self.queue = JobManager(serial=True)            # benchmarks: one at a time, in order
        self.tests = JobManager()
        self._benches: dict = {}
        self.lock = threading.Lock()

    # ------------------------------------------------------------ indexing
    def bench_dirs(self) -> list[str]:
        out = []
        for name in sorted(os.listdir(self.results)):
            d = os.path.join(self.results, name)
            if os.path.isdir(d) and (os.path.exists(os.path.join(d, "index.json")) or
                                     os.path.exists(os.path.join(d, "manifest.json"))):
                out.append(d)
        return out

    def bench_path(self, bench: str) -> str:
        d = os.path.abspath(os.path.join(self.results, bench))
        if not d.startswith(self.results + os.sep) or not os.path.isdir(d):
            raise KeyError(f"unknown benchmark {bench!r}")
        return d

    def runs(self) -> list[dict]:
        rows = []
        for d in self.bench_dirs():
            for r in _load(os.path.join(d, "index.json"), []):
                r = dict(r)
                r["bench"] = os.path.basename(d)
                r["bench_dir"] = d
                rows.append(r)
        return rows

    def config_keys(self) -> dict:
        keys: dict = {}
        for r in self.runs():
            for k, v in {**(r.get("config") or {}), "suite": r.get("suite")}.items():
                keys.setdefault(k, set()).add(json.dumps(v, sort_keys=True, default=str) if isinstance(v, (dict, list)) else str(v))
        return {k: sorted(v) for k, v in sorted(keys.items())}

    def grouped(self, group_keys: list[str], filters: dict) -> list[dict]:
        import statistics
        rows = [r for r in self.runs() if _match({**r.get("config", {}), "suite": r.get("suite")}, filters)]
        groups: dict = {}
        for r in rows:
            key = tuple((k, _hashable(r.get("config", {}).get(k, r.get(k)))) for k in group_keys)
            groups.setdefault(key, []).append(r)
        out = []
        for key, members in groups.items():
            aucs = [m.get("test_auroc") for m in members if m.get("test_auroc") is not None]
            g = {"group": dict(key), "n_runs": len(members),
                 "seeds": sorted((m.get("seed") for m in members), key=lambda x: (x is None, x)),
                 "auroc_mean": _mean(aucs), "auroc_std": (statistics.pstdev(aucs) if len(aucs) > 1 else 0.0),
                 "run_ids": [m["run_id"] for m in members]}
            if len(aucs) > 1:
                g["auroc_ci95"] = 1.96 * g["auroc_std"] / (len(aucs) ** 0.5)
                g["high_variance"] = g["auroc_mean"] and g["auroc_std"] / g["auroc_mean"] > 0.05   # NFR-2 flag
            out.append(g)
        return sorted(out, key=lambda g: json.dumps(g["group"], sort_keys=True, default=str))

    def report(self, bench: str) -> dict:
        d = os.path.join(self.results, bench, "report")
        summary = _load(os.path.join(d, "summary.json"), {})
        figs = sorted(f for f in os.listdir(d)) if os.path.isdir(d) else []
        return {"bench": bench, "has_report": os.path.isdir(d),
                "nfr": summary.get("nfr", {}), "figures": [f for f in figs if f.endswith(".png")],
                "catalogue": summary.get("figures", {}), "nfr_evidence": summary.get("nfr_evidence", {}),
                "security": summary.get("security"), "cost": summary.get("cost")}

    # ------------------------------------------------------------ analytics (bench/analyze.py)
    def _bench(self, bench: str):
        from bench import analyze as A
        path = self.bench_path(bench)
        if not os.path.exists(os.path.join(path, "manifest.json")):
            raise KeyError(f"{bench} has no manifest.json (not a run_benchmarks output)")
        stamp = max(os.path.getmtime(p) for p in [path] + glob.glob(os.path.join(path, "sessions", "*.json")))
        with self.lock:
            hit = self._benches.get(path)
            if hit and hit[0] == stamp:
                return hit[1]
        B = A.Bench(path)
        with self.lock:
            self._benches[path] = (stamp, B)
        return B

    @staticmethod
    def catalogue() -> list[dict]:
        from bench import analyze as A
        return [{k: v for k, v in f.items() if k != "fn"} for f in A.FIGURES.values()]

    def figure(self, bench: str, name: str, fmt: str, ctx: dict) -> str | None:
        from bench import analyze as A
        if name not in A.FIGURES:
            raise KeyError(f"unknown figure {name!r}")
        if fmt not in ("png", "svg", "csv"):
            raise KeyError("format: png, svg or csv")
        path = self.bench_path(bench)
        ctx = {k: v for k, v in (ctx or {}).items() if v}
        if not ctx:                                           # unfiltered = exactly the file report.md shows
            p = os.path.join(path, "report", "figures", f"{name}.{fmt}")
            if os.path.exists(p):
                return p
        key = hashlib.sha256(json.dumps(ctx, sort_keys=True).encode()).hexdigest()[:12]
        outdir = os.path.join(path, "report", "figures", "_ui", key)
        p = os.path.join(outdir, f"{name}.{fmt}")
        if os.path.exists(p) and os.path.getmtime(p) >= os.path.getmtime(os.path.join(path, "manifest.json")):
            return p
        B = self._bench(bench)
        with _MPL_LOCK:
            res = A.render_figure(B, name, outdir, ctx)
        if not res:
            return None
        if res.get("error"):
            raise RuntimeError(res["error"])
        return res.get(fmt)

    def compare(self, a: str, b: str) -> dict:
        from bench import analyze as A
        (ba, sa), (bb, sb) = (x.split(":", 1) for x in (a, b))
        da, db = self._session(ba, sa), self._session(bb, sb)
        out = os.path.join(self.bench_path(ba), "report", "figures", "_compare", f"{sa}__{sb}"[:120])
        with _MPL_LOCK:
            res = A.compare_sessions(da, db, sa, sb, out)
        res["figure"] = {k: os.path.relpath(v, self.results) for k, v in (res.get("figure") or {}).items()}
        return res

    def _session(self, bench, sid):
        p = os.path.join(self.bench_path(bench), "sessions", f"{sid}.json")
        if not re.fullmatch(r"[\w.-]+", sid) or not os.path.exists(p):
            raise KeyError(f"unknown run {bench}:{sid}")
        return json.load(open(p))

    def scorecard(self, bench: str) -> dict:
        rep = self.report(bench)
        names = {"NFR-1": "Utility", "NFR-2": "Reproducibility", "NFR-3": "Latency", "NFR-4": "Communication",
                 "NFR-5": "Throughput", "NFR-6": "Efficiency", "NFR-7": "Transport & logging",
                 "NFR-8": "Conditional privacy evaluation", "NFR-9": "Conditional cloud cost", "NFR-10": "Fault handling"}
        return {"bench": bench, "rows": [{"id": k, "name": n, **(rep["nfr"].get(k) or {"status": "not attempted", "value": ""}),
                                          "evidence": rep["nfr_evidence"].get(k, [])} for k, n in names.items()]}

    # ------------------------------------------------------------ benchmark queue
    def launch(self, spec: dict) -> dict:
        from vfl.webui.jobs import env_with_root
        name = spec.get("name") or f"bench-{time.strftime('%Y%m%d-%H%M%S')}"
        if not re.fullmatch(r"[A-Za-z0-9][\w.-]{0,63}", name):
            raise ValueError("run name: letters, digits, '.', '_' or '-'")
        out = os.path.join(self.results, name)
        spec = dict(spec)
        if spec.get("data_dir") and self.workspace and not os.path.isabs(spec["data_dir"]):
            spec["data_dir"] = os.path.join(self.workspace, "datasets", spec["data_dir"])
        argv = bench_argv(spec, out)
        suites = argv[argv.index("--suites") + 1]

        def analyse(job):
            import subprocess
            r = subprocess.run([sys.executable, os.path.join("bench", "analyze.py"), out], cwd=ROOT,
                               env=env_with_root(), capture_output=True, text=True, timeout=900)
            job.lines.append("\n[analyze.py]\n" + (r.stdout or "")[-3000:] + (r.stderr or "")[-1500:])
            return {"bench": name, "analysed": r.returncode == 0}

        counter = {"n": 0}

        def progress(line):
            if re.search(r"\s(COMPLETED|FAILED|STOPPED) in ", line):
                counter["n"] += 1
                return {"sessions_done": counter["n"]}
            return None

        j = self.queue.submit("bench", [sys.executable] + argv, cwd=ROOT, label=f"{name} [{suites}]",
                              meta={"suite": suites, "out": name, "spec": spec}, on_done=analyse, progress=progress)
        return {"job": j.id, "out": name, "queued": True}

    def rerun(self, jid: str) -> dict:
        j = self.queue.rerun(jid)
        return {"job": j.id}

    def cancel(self, jid):
        return self.queue.cancel(jid)

    def jobs_view(self):
        out = []
        for j in self.queue.list("bench", tail=60):
            out.append({**j, "suite": j["meta"].get("suite"), "out": j["meta"].get("out"), "t": j["queued"]})
        return out

    # ------------------------------------------------------------ pytest
    @staticmethod
    def test_targets() -> list[str]:
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
        return ["all", "not-slow"] + files

    def run_tests(self, target: str) -> dict:
        if target not in self.test_targets():
            raise ValueError(f"unknown test target {target!r}")
        jdir = os.path.join(self.results, "_tests")
        os.makedirs(jdir, exist_ok=True)
        xml = os.path.join(jdir, f"junit-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}.xml")
        # --continue-on-collection-errors: one file that cannot import (e.g. no grpc on this machine) must not
        # hide every other result; the import error still shows as an `error` row in the table
        cmd = [sys.executable, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--continue-on-collection-errors",
               f"--junitxml={xml}"]
        if target == "not-slow":
            cmd += ["-m", "not slow"]
        elif target != "all":
            cmd.append(os.path.join("tests", target))
        j = self.tests.submit("pytest", cmd, cwd=ROOT, label=f"pytest {target}", meta={"target": target, "xml": xml},
                              on_done=lambda job: parse_junit(xml), ok_codes=(0,))
        return {"job": j.id}

    def export_zip(self, bench: str) -> str:
        d = self.bench_path(bench)
        zpath = os.path.join(d, f"{bench}-report.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for root, dirs, files in os.walk(os.path.join(d, "report")):
                dirs[:] = [x for x in dirs if not x.startswith("_")]          # skip UI render caches
                for f in files:
                    fp = os.path.join(root, f)
                    z.write(fp, os.path.relpath(fp, d))
            for name in ("index.json", "manifest.json"):
                if os.path.exists(os.path.join(d, name)):
                    z.write(os.path.join(d, name), name)
        return zpath


def parse_junit(path: str) -> dict:
    import xml.etree.ElementTree as ET
    if not os.path.exists(path):
        return {"tests": [], "counts": {}}
    root = ET.parse(path).getroot()
    rows = []
    for tc in root.iter("testcase"):
        outcome, msg = "passed", ""
        for tag in ("failure", "error", "skipped"):
            el = tc.find(tag)
            if el is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                msg = (el.get("message") or el.text or "")[:400]
                break
        rows.append({"test": tc.get("name"), "file": tc.get("classname"), "time_s": float(tc.get("time") or 0),
                     "outcome": outcome, "message": msg})
    counts = {k: sum(r["outcome"] == k for r in rows) for k in ("passed", "failed", "error", "skipped")}
    return {"tests": rows, "counts": counts}


def _load(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _hashable(v):
    return json.dumps(v, sort_keys=True, default=str) if isinstance(v, (dict, list)) else v


def _match(config, filters):
    for k, v in (filters or {}).items():
        x = config.get(k)
        if str(_hashable(x)) != str(v):
            return False
    return True


# ------------------------------------------------------------------ routes
def register(app: Flask, ui: BenchUI, op):
    """Benchmark / test / analytics routes. `op()` is True for an operator (token or login)."""

    def need_op():
        return None if op() else (jsonify({"error": "operator token required"}), 401)

    @app.get("/report-asset/<bench>/<path:name>")
    def asset(bench, name):
        return send_from_directory(os.path.join(ui.bench_path(bench), "report"), name)

    @app.get("/api/runs")
    def runs():
        return jsonify(ui.runs())

    @app.get("/api/configkeys")
    def configkeys():
        return jsonify(ui.config_keys())

    @app.get("/api/grouped")
    def grouped():
        gk = [k for k in request.args.get("group", "dataset_tag,bottom_model").split(",") if k]
        filt = {k[2:]: v for k, v in request.args.items() if k.startswith("f.")}
        return jsonify(ui.grouped(gk, filt))

    @app.get("/api/report/<bench>")
    def report(bench):
        return jsonify(ui.report(bench))

    @app.get("/api/benches")
    def benches():
        return jsonify([os.path.basename(d) for d in ui.bench_dirs()])

    @app.get("/api/figures")
    def figures():
        return jsonify(ui.catalogue())

    @app.get("/api/figure/<bench>/<name>.<fmt>")
    def figure(bench, name, fmt):
        ctx = {"filter": {k[2:]: v for k, v in request.args.items() if k.startswith("f.")},
               "group": request.args.get("group"), "session": request.args.get("session")}
        ctx = {k: v for k, v in ctx.items() if v}
        try:
            p = ui.figure(bench, name, fmt, ctx)
        except KeyError as e:
            return jsonify({"error": str(e)}), 404
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500
        if not p:
            return jsonify({"error": "no data for this figure in this benchmark"}), 404
        return send_file(p, as_attachment=request.args.get("download") == "1")

    @app.get("/api/compare")
    def compare():
        try:
            return jsonify(ui.compare(request.args["a"], request.args["b"]))
        except (KeyError, ValueError) as e:
            return jsonify({"error": f"compare needs a=BENCH:RUN&b=BENCH:RUN ({e})"}), 400

    @app.get("/api/results-file/<path:rel>")
    def results_file(rel):
        p = os.path.abspath(os.path.join(ui.results, rel))
        if not p.startswith(ui.results + os.sep) or not os.path.exists(p):
            return jsonify({"error": "not found"}), 404
        return send_file(p)

    @app.get("/api/scorecard/<bench>")
    def scorecard(bench):
        return jsonify(ui.scorecard(bench))

    @app.get("/api/bench/options")
    def bench_options():
        return jsonify({"options": BENCH_OPTIONS, "suites": SUITES, "suite_help": SUITE_HELP,
                        "configs": sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "configs", "*.yaml")))})

    @app.get("/api/jobs")
    def jobs():
        return jsonify(ui.jobs_view())

    @app.post("/api/launch")
    def launch():
        if (r := need_op()):
            return r
        try:
            return jsonify(ui.launch(request.get_json(force=True) or {}))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.post("/api/jobs/<jid>/cancel")
    def cancel(jid):
        if (r := need_op()):
            return r
        try:
            return jsonify(ui.cancel(jid))
        except KeyError as e:
            return jsonify({"error": str(e)}), 404

    @app.post("/api/jobs/<jid>/rerun")
    def rerun(jid):
        if (r := need_op()):
            return r
        try:
            return jsonify(ui.rerun(jid))
        except KeyError as e:
            return jsonify({"error": str(e)}), 404

    @app.get("/api/export/<bench>")
    def export(bench):
        try:
            return send_file(ui.export_zip(bench), as_attachment=True, download_name=f"{bench}-report.zip")
        except Exception as e:
            return jsonify({"error": str(e)}), 404

    @app.get("/api/tests/targets")
    def test_targets():
        return jsonify(ui.test_targets())

    @app.post("/api/tests/run")
    def run_tests():
        if (r := need_op()):
            return r
        try:
            return jsonify(ui.run_tests((request.get_json(force=True) or {}).get("target", "not-slow")))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.get("/api/tests/jobs")
    def test_jobs():
        return jsonify(ui.tests.list("pytest", tail=0))

    @app.get("/api/tests/jobs/<jid>")
    def test_job(jid):
        try:
            return jsonify(ui.tests.get(jid).view(tail=1500))
        except KeyError as e:
            return jsonify({"error": str(e)}), 404

    @app.post("/api/tests/jobs/<jid>/cancel")
    def test_cancel(jid):
        if (r := need_op()):
            return r
        return jsonify(ui.tests.cancel(jid))


def create_app(ui: BenchUI, token: str | None) -> Flask:
    """bench-ui on its own: the unified UI shell with the benchmark/test/analytics routes."""
    from vfl import websec
    app = Flask(__name__, static_folder=None)
    websec.install(app)

    def op():
        # constant-time (hmac.compare_digest); ?token= only for page loads, never for actions
        return not token or websec.bearer_ok(token) or websec.query_token_ok(token)

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:n>")
    def static_files(n):
        return send_from_directory(STATIC, n)

    register(app, ui, op)
    return app


def serve(results_dir: str, run_dirs: list[str], host: str, port: int, token: str | None):
    """`vfl.cli bench-ui` = the unified UI (everything reachable from one place)."""
    from vfl.webui import serve as ui_serve
    ui_serve(workspace=".", host=host, port=port, token=token, results=results_dir, runs=run_dirs)
