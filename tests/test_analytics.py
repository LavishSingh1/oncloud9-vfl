"""v2.1 item 5: Tests & Benchmarks and Analytics in the UI.

Uses tests/bench_fixture.py (a benchmark folder with the real layout and made-up numbers), so bench/analyze.py
and every UI route can be checked without running VFL. Needs matplotlib and flask, not torch or grpc.
"""
import hashlib
import io
import os
import subprocess
import sys
import time
import zipfile

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("flask")
from bench import analyze as A  # noqa: E402
from bench import ui as U  # noqa: E402
from bench_fixture import make_bench  # noqa: E402
from vfl.webui import create_app  # noqa: E402
from vfl.webui.jobs import JobManager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = {"Authorization": "Bearer tok", "X-VFL-Request": "1"}       # what the UI sends (CSRF header, v2.3)
MODEL = {"perf_vs_epoch", "roc_confusion", "time_to_target", "lopo_contribution", "privacy_utility", "by_bottom_model",
         "by_partition"}
SYSTEMS = {"latency_vs_rtt", "throughput", "time_breakdown", "bytes", "resources", "slot_wait_stragglers", "scaling",
           "codec_comparison", "psi_cost", "e5_timeline", "security_passfail"}


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    bench = make_bench(str(root / "results" / "bench1"))
    r = subprocess.run([sys.executable, os.path.join("bench", "analyze.py"), bench], cwd=ROOT, capture_output=True,
                       text=True, timeout=600, env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return root


@pytest.fixture
def client(ws):
    app = create_app(str(ws), token="tok")
    return app.test_client(), app.extensions["vfl"]


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def test_registry_covers_every_requested_graph():
    assert MODEL | SYSTEMS <= set(A.FIGURES)
    assert {A.FIGURES[n]["group"] for n in MODEL} == {"model"}
    assert {A.FIGURES[n]["group"] for n in SYSTEMS} == {"systems"}


def test_report_md_and_ui_show_the_same_figures(ws, client):
    c, _ = client
    rep = ws / "results" / "bench1" / "report"
    md = open(rep / "report.md").read()
    for name in MODEL | SYSTEMS:
        for ext in ("png", "svg", "csv"):
            assert (rep / "figures" / f"{name}.{ext}").exists(), f"{name}.{ext}"
        assert f"figures/{name}.png" in md, name
        r = c.get(f"/api/figure/bench1/{name}.png")
        assert r.status_code == 200
        assert _sha(r.data) == _sha(open(rep / "figures" / f"{name}.png", "rb").read()), name   # same file
    assert c.get("/api/figure/bench1/bytes.svg").data.lstrip().startswith(b"<?xml")
    assert b"," in c.get("/api/figure/bench1/bytes.csv").data
    assert c.get("/api/figure/bench1/bytes.csv?download=1").headers["Content-Disposition"].startswith("attachment")


def test_filtered_and_grouped_figures_rerun_the_same_function(client):
    c, _ = client
    a = c.get("/api/figure/bench1/compare_by.png?group=codec")
    b = c.get("/api/figure/bench1/compare_by.png?group=batch_size")
    assert a.status_code == b.status_code == 200 and _sha(a.data) != _sha(b.data)
    f = c.get("/api/figure/bench1/perf_vs_epoch.csv?f.suite=e1")
    assert f.status_code == 200
    assert c.get("/api/figure/bench1/nope.png").status_code == 404


def test_runs_grouping_ci_and_compare(client):
    c, _ = client
    runs = c.get("/api/runs").get_json()
    assert len(runs) == 29 and {"run_id", "suite", "config", "test_auroc"} <= set(runs[0])
    keys = c.get("/api/configkeys").get_json()
    assert {"codec", "batch_size", "suite"} <= set(keys)
    g = c.get("/api/grouped?group=codec&f.suite=eff").get_json()
    assert {x["group"]["codec"] for x in g} == {"fp32", "fp16", "int8"}
    e1 = c.get("/api/grouped?group=suite&f.suite=e1").get_json()[0]
    assert e1["n_runs"] == 4 and "auroc_ci95" in e1 and e1["auroc_std"] >= 0
    a, b = [r["run_id"] for r in runs if r["suite"] == "e1"][:2]
    r = c.get(f"/api/compare?a=bench1:{a}&b=bench1:{b}").get_json()
    assert r["figure"]["png"].endswith(".png")
    assert c.get(f"/api/results-file/{r['figure']['png']}").status_code == 200
    assert c.get("/api/compare?a=x").status_code == 400


def test_nfr_scorecard_links_evidence(client):
    c, _ = client
    rows = c.get("/api/scorecard/bench1").get_json()["rows"]
    assert [r["id"] for r in rows] == [f"NFR-{i}" for i in range(1, 11)]
    assert {r["status"] for r in rows} <= {"pass", "fail", "reported", "not attempted", "partial"}
    assert any(r["evidence"] for r in rows), "no NFR links to an evidence run"


def test_report_bundle_export(client):
    c, _ = client
    r = c.get("/api/export/bench1")
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.data)).namelist()
    assert "report/report.md" in names and "index.json" in names
    assert "report/figures/perf_vs_epoch.png" in names and "report/figures/perf_vs_epoch.csv" in names
    assert not any("/_ui/" in n or "/_compare/" in n for n in names)


# ------------------------------------------------------------------ tests + benchmark launcher
def test_pytest_from_the_ui_gives_a_pass_fail_table(client):
    c, ext = client
    assert c.post("/api/tests/run", json={"target": "test_alignment.py"}).status_code == 401
    assert c.post("/api/tests/run", json={"target": "../etc"}, headers=H).status_code == 400
    jid = c.post("/api/tests/run", json={"target": "test_alignment.py"}, headers=H).get_json()["job"]
    j = ext["bench"].tests.wait(jid, 300)
    assert j["status"] == "done", j["log"][-2000:]
    res = j["result"]
    assert res["counts"]["passed"] >= 1 and res["counts"]["failed"] == 0
    assert all({"test", "file", "outcome", "time_s"} <= set(t) for t in res["tests"])
    live = c.get(f"/api/tests/jobs/{jid}").get_json()
    assert "passed" in live["log"]


def test_launcher_covers_every_suite_and_flag():
    assert set(U.SUITES) == {"e1", "e2", "eff", "def", "psi", "e5", "sweep", "lopo", "security"}
    flags = U.run_benchmarks_flags()
    offered = {o["flag"] for o in U.BENCH_OPTIONS.values()} | {"--out"}
    assert flags <= offered, f"run_benchmarks flags with no launcher field: {flags - offered}"
    argv = U.bench_argv({"suites": ["e2", "eff"], "rtts": [0, 50], "batch_sizes": [32, 128], "codecs": ["fp32", "int8"],
                         "seeds": [1, 2]}, "results/x")
    s = " ".join(argv)
    assert "--suites e2,eff" in s and "--rtts 0,50" in s and "--codecs fp32,int8" in s
    with pytest.raises(ValueError):
        U.bench_argv({"suites": ["e9"]}, "x")
    with pytest.raises(ValueError):
        U.bench_argv({"rtts": [7]}, "x")                       # not one of the offered values


def test_benchmark_queue_runs_in_order_with_cancel_and_rerun():
    q = JobManager(serial=True)
    sleep = [sys.executable, "-c", "import time, sys; print('start', flush=True); time.sleep(float(sys.argv[1]))"]
    a = q.submit("bench", sleep + ["0.5"], label="a")
    b = q.submit("bench", sleep + ["30"], label="b")
    c = q.submit("bench", sleep + ["0.1"], label="c")
    time.sleep(0.2)
    assert q.get(a.id).status == "running" and q.get(c.id).status == "queued"       # one at a time
    q.wait(a.id, 30)
    t0 = time.time()
    while q.get(b.id).status != "running" and time.time() - t0 < 10:
        time.sleep(0.05)
    q.cancel(b.id)
    assert q.wait(b.id, 30)["status"] == "cancelled"
    assert q.wait(c.id, 30)["status"] == "done"
    r = q.rerun(c.id)
    assert q.wait(r.id, 30)["status"] == "done" and "(rerun)" in r.label


def test_launch_route_queues_a_benchmark(client, monkeypatch):
    c, ext = client
    seen = {}

    def fake_submit(kind, cmd, **kw):
        seen.update(kind=kind, cmd=cmd, **kw)
        return type("J", (), {"id": "j1"})()
    monkeypatch.setattr(ext["bench"].queue, "submit", fake_submit)
    r = c.post("/api/launch", json={"name": "grid1", "suites": ["e2"], "rtts": [0, 20], "batch_sizes": [64]}, headers=H)
    assert r.status_code == 200, r.get_json()
    assert seen["kind"] == "bench" and "bench/run_benchmarks.py" in " ".join(seen["cmd"])
    assert "--rtts" in seen["cmd"] and seen["meta"]["out"] == "grid1"
    assert c.post("/api/launch", json={"name": "../x"}, headers=H).status_code == 400
