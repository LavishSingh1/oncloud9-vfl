"""Issue #20 (Order of Phoenix, rt-baseline 07ed305): per-session resources.jsonl files were always empty, epoch rows
had no session key, the coordinator's CPU/RSS was never saved, and the run summary's resource aggregate covered the
whole process lifetime (n_samples 513 for a 0.3 s session) instead of the run (FR-10, NFR-6)."""
import json
import os
import time

import pytest

from vfl.telemetry import JsonlWriter, ResourceMonitor


def _rows(path):
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


# ------------------------------------------------------------------ the monitor's session window
def test_session_window_writes_tagged_samples_and_aggregates_only_that_session(tmp_path):
    m = ResourceMonitor(interval=1.0)
    for _ in range(5):                               # process uptime before the session (the reporter's sleep 30)
        m.sample()
    w = JsonlWriter(str(tmp_path / "resources.jsonl"))
    m.begin(w, {"session": "s1", "party": "active", "role": "active"})
    m.sample()
    m.sample()
    live = m.session_summary()
    assert live["scope"] == "session" and live["n_samples"] == 3                 # begin + 2, not the 5 before
    out = m.end("s1")
    assert out["scope"] == "session" and out["n_samples"] == 4                  # + the closing sample
    assert out["window_s"] >= 0 and out["rss_max_mib"] > 0 and out["cpu_avg_percent"] is not None
    m.sample()                                       # after the session: not written to its file
    w.close()
    rows = _rows(tmp_path / "resources.jsonl")
    assert len(rows) == 4
    assert all(r["session"] == "s1" and r["party"] == "active" and "cpu_percent" in r and "rss_mib" in r
               for r in rows)
    assert m.summary()["scope"] == "process" and m.summary()["n_samples"] == 10    # 5 + begin + 2 + end + 1


def test_a_sub_second_session_still_has_samples(tmp_path):
    m = ResourceMonitor(interval=1.0)                # the monitor thread would not tick in time
    w = JsonlWriter(str(tmp_path / "r.jsonl"))
    m.begin(w, {"session": "short"})
    out = m.end("short")
    w.close()
    assert out["n_samples"] == 2 and len(_rows(tmp_path / "r.jsonl")) == 2


def test_end_only_closes_its_own_session_and_a_new_begin_replaces_the_window(tmp_path):
    m = ResourceMonitor()
    m.begin(None, {"session": "a"})
    assert m.end("b") is None and m.session_summary()["n_samples"] == 1          # still a's window
    m.begin(None, {"session": "b"})
    assert m.end("a") is None and m.end("b")["n_samples"] == 2
    assert m.end("b") is None and m.session_summary()["n_samples"] == 0          # no window open


def test_a_closed_session_file_does_not_break_sampling(tmp_path):
    m = ResourceMonitor()
    w = JsonlWriter(str(tmp_path / "r.jsonl"))
    m.begin(w, {"session": "s"})
    w.close()                                        # e.g. the session log was closed first
    m.sample()                                       # must not raise
    assert m.end("s")["n_samples"] == 3


def test_process_lifetime_buffer_is_bounded():
    m = ResourceMonitor(keep=5)
    for _ in range(12):
        m.sample()
    assert len(m.samples) == 5


def test_monitor_thread_feeds_the_window(tmp_path):
    m = ResourceMonitor(interval=0.05).start()
    w = JsonlWriter(str(tmp_path / "r.jsonl"))
    m.begin(w, {"session": "t"})
    time.sleep(0.4)
    out = m.end("t")
    m.stop()
    w.close()
    assert out["n_samples"] >= 4 and len(_rows(tmp_path / "r.jsonl")) == out["n_samples"]


# ------------------------------------------------------------------ benchmark NFR-6 line
def test_nfr6_line_includes_the_coordinator_and_marks_lifetime_aggregates(tmp_path):
    pytest.importorskip("matplotlib")
    from bench import analyze as A
    from bench_fixture import make_bench
    path = make_bench(str(tmp_path / "b"), seeds=(1,))
    m = json.load(open(os.path.join(path, "manifest.json")))
    sid = [e for e in m["sessions"] if e["suite"] == "e1"][0]["session_id"]
    p = os.path.join(path, "sessions", f"{sid}.json")
    d = json.load(open(p))
    d["coordinator_resources"] = {"scope": "session", "cpu_avg_percent": 3.0, "cpu_max_percent": 7.0,
                                  "rss_avg_mib": 90.0, "rss_max_mib": 95.0, "n_samples": 3}
    d["finals"]["active"]["resources"]["scope"] = "session"
    json.dump(d, open(p, "w"))
    R = {"nfr": {}, "tables": {}}
    fig = str(tmp_path / "fig")
    os.makedirs(fig)
    A.analyse_eff(A.Bench(path), R, fig)
    v = R["nfr"]["NFR-6"]["value"]
    assert "coordinator: CPU avg 3.0% / max 7.0%, RSS max 95 MiB" in v
    assert "active: CPU avg 40.0% / max 60.0%, RSS max 320 MiB |" in v                  # session scope: no tag
    assert "passive-1: CPU avg 20.0% / max 30.0%, RSS max 260 MiB (process lifetime)" in v   # old-format record


# ------------------------------------------------------------------ end to end (needs grpc + torch)
@pytest.mark.slow
def test_per_session_resource_records_end_to_end(tmp_path, data_dir):
    pytest.importorskip("grpc")
    pytest.importorskip("torch")
    from bench.harness import LocalDeployment
    work = tmp_path / "dep"
    with LocalDeployment(str(work), data_dir, 1, tls=True, base_port=58050, ui_base=19480) as dep:
        coord = dep.start()
        time.sleep(6)                                # idle uptime that must NOT count towards the session
        d = coord.run({"epochs": 3, "batch_size": 16}, quiet=True)
        assert d["state"] == "COMPLETED"
        sid = d["session_id"]
        for party in ("active", "passive-1"):
            sdir = work / "logs" / party / sid
            res = _rows(sdir / "resources.jsonl")
            assert res and all(r["session"] == sid and r["party"] == party for r in res)       # was 0 bytes
            ep = _rows(sdir / "epochs.jsonl")
            assert len(ep) == 3 and all(r["session"] == sid for r in ep)                       # was no key
            agg = d["finals"][party]["resources"]
            assert agg["scope"] == "session" and agg["n_samples"] == len(res)
            assert agg["n_samples"] <= d["finals"][party]["session_wall_s"] / 1.0 + 3          # not process uptime
            run_summary = json.load(open(sdir / "run_summary.json"))
            assert run_summary["resources"]["scope"] == "session"
        cres = _rows(work / "runs" / sid / "resources_coordinator.jsonl")                      # was never saved
        assert cres and all(r["session"] == sid and r["node"] == "coordinator" for r in cres)
        assert d["coordinator_resources"]["scope"] == "session" and d["coordinator_resources"]["n_samples"] >= 2
        assert all(r["session"] == sid for r in _rows(work / "runs" / sid / "epochs.jsonl"))
