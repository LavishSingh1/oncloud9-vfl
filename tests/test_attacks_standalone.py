"""v2.1 item 1: standalone attack simulator.

* the CLI runs with no dashboard (Flask is made unimportable) and writes report.json/.csv, summary.txt and
  security.jsonl with the required columns;
* the safeguards (--lab-mode, target allowlist, UI token + acknowledgement) refuse what they should;
* a stopped deployment is reported as an ERROR, never as "blocked" (= a false pass);
* CLI, the coordinator's Security tab and the unified UI's attack page give identical verdicts.
The targets are closed loopback ports, so nothing here needs grpc, torch or a running deployment.
"""
import csv
import json
import os
import socket
import subprocess
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = {"Authorization": "Bearer tok", "X-VFL-Request": "1"}       # what the UI sends (CSRF header, v2.3)
REQUIRED = ["attack", "layer", "expected", "observed", "evidence"]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def dead():
    return f"127.0.0.1:{_free_port()}", f"127.0.0.1:{_free_port()}"


def _cli(args, block_flask=True):
    code = ("import sys\n"
            + ("sys.modules['flask'] = None; sys.modules['werkzeug'] = None\n" if block_flask else "")
            + "from vfl.cli import main\nmain(sys.argv[1:])\n")
    return subprocess.run([sys.executable, "-c", code] + args, cwd=ROOT, capture_output=True, text=True, timeout=300,
                          env={**os.environ, "PYTHONPATH": ROOT})


def test_cli_runs_without_any_dashboard_and_writes_every_output(tmp_path, dead):
    out = tmp_path / "attacks"
    r = _cli(["attack", "inference", "--lab-mode", "--coordinator", dead[0], "--active", dead[1], "--out", str(out)])
    assert r.returncode in (0, 1), r.stderr
    rep = json.load(open(out / "report.json"))
    assert rep["lab_only"] is True and rep["summary"]["cases"] == 2
    for row in rep["rows"]:
        assert all(k in row for k in REQUIRED)
        assert row["evidence"] in ("attacker-reported", "server-observed + attacker-reported")
    rows = list(csv.DictReader(open(out / "report.csv")))
    assert [x["attack"] for x in rows] == ["label_leakage", "membership"] and set(REQUIRED) <= set(rows[0])
    assert next(x for x in rows if x["attack"] == "membership")["observed"] == "succeeded"   # experimental is non-private
    first = open(out / "summary.txt").readline()
    assert first.startswith(("PASS", "FAIL"))
    events = [json.loads(line) for line in open(out / "security.jsonl")]
    assert {e["attack_type"] for e in events} == {"label_leakage", "membership"}
    assert all(e["evidence"] == "attacker-reported" and e["node"] == "lab-attacker" for e in events)


def test_safeguards_refuse(tmp_path, dead):
    r = _cli(["attack", "all", "--coordinator", dead[0]])
    assert r.returncode == 2 and "--lab-mode" in r.stdout
    r = _cli(["attack", "auth", "--lab-mode", "--coordinator", "10.20.30.40:50050"])
    assert r.returncode == 2 and "not in the session's target config" in r.stdout
    cfg = tmp_path / "targets.json"
    cfg.write_text(json.dumps({"coordinator": dead[0], "active": dead[1], "hosts": ["127.0.0.1"]}))
    r = _cli(["attack", "auth", "--lab-mode", "--config", str(cfg), "--coordinator", "10.20.30.40:50050"])
    assert r.returncode == 2                                     # explicit target outside the config: still refused
    r = _cli(["attack", "nonsense", "--lab-mode"])
    assert r.returncode == 2 and "unknown case" in r.stdout


def test_stopped_deployment_is_an_error_not_a_pass(tmp_path, dead):
    out = tmp_path / "a"
    r = _cli(["attack", "auth", "--lab-mode", "--coordinator", dead[0], "--active", dead[1], "--out", str(out)])
    assert r.returncode == 1
    rep = json.load(open(out / "report.json"))
    for row in rep["rows"]:
        assert row["observed"] in ("error", "skipped"), row
        assert row["observed"] != "blocked"
    assert not rep["summary"]["all_as_expected"]
    assert "target unreachable" in next(x for x in rep["rows"] if x["attack"] == "no_cert")["detail"]


# ------------------------------------------------------------------ same verdicts from all three entry points
def _cli_verdicts(tmp_path, dead, case):
    cfg = tmp_path / "t.json"
    cfg.write_text(json.dumps({"coordinator": dead[0], "active": dead[1], "hosts": ["127.0.0.1"]}))
    out = tmp_path / "cli"
    _cli(["attack", case, "--lab-mode", "--config", str(cfg), "--coordinator", dead[0], "--active", dead[1],
          "--certs", str(tmp_path / "certs"), "--out", str(out)])
    return json.load(open(out / "report.json"))["verdicts"]


def _security_tab_verdicts(tmp_path, dead, case):
    pytest.importorskip("flask")
    from vfl.attacks.lab import LabRunner
    from vfl.dashboard import create_app
    state = types.SimpleNamespace(run_dir=str(tmp_path / "runs"), parties={
        "active": {"party_id": "active", "role": "active", "address": dead[1], "hostname": "127.0.0.1"}})
    lab = LabRunner(state, str(tmp_path / "certs"), dead[0], 0, "tok", False)
    lab.url = None                                        # nothing to post observations to in this test
    backend = types.SimpleNamespace(party_id="coord", lab=lab, lab_mode=True, registry=None, default_config={},
                                    ui_state=lambda: {})
    c = create_app(backend, "coordinator", token="tok").test_client()
    assert c.post("/api/lab/attacks", json={"name": case}).status_code == 401           # operator only
    r = c.post("/api/lab/attacks", json={"name": case, "args": {}}, headers=H)
    assert r.status_code == 200, r.get_json()
    j = lab.runner.jobs.wait(r.get_json()["job"], 300)
    assert j["result"], j["log"][-1500:]
    return j["result"]["verdicts"]


def _ui_page_verdicts(tmp_path, dead, case):
    pytest.importorskip("flask")
    from vfl.webui import create_app
    ws = tmp_path / "ws"
    app = create_app(str(ws), token="tok", lab_mode=True)
    d = ws / "deployments" / "lab"
    d.mkdir(parents=True)
    (d / "deployment.json").write_text(json.dumps({"name": "lab", "coordinator": dead[0], "active": dead[1],
                                                   "hosts": ["127.0.0.1"], "certs": str(tmp_path / "certs")}))
    c = app.test_client()
    body = {"target": "lab", "case": case}
    assert c.post("/api/attacks/run", json={**body, "lab_mode_ack": True}).status_code == 401
    assert c.post("/api/attacks/run", json=body, headers=H).status_code == 400          # no acknowledgement
    assert c.post("/api/attacks/run", json={**body, "target": "other", "lab_mode_ack": True},
                  headers=H).status_code == 404                                          # not a known deployment
    r = c.post("/api/attacks/run", json={**body, "lab_mode_ack": True}, headers=H)
    assert r.status_code == 200, r.get_json()
    j = app.extensions["vfl"]["attacks"].jobs.wait(r.get_json()["job"], 300)
    assert j["result"], j["log"][-1500:]
    return j["result"]["verdicts"]


@pytest.mark.parametrize("case", ["all"])
def test_cli_security_tab_and_ui_page_give_identical_verdicts(tmp_path, dead, case):
    a = _cli_verdicts(tmp_path, dead, case)
    b = _security_tab_verdicts(tmp_path, dead, case)
    c = _ui_page_verdicts(tmp_path, dead, case)
    assert len(a) == 16
    assert a == b == c


def test_ui_attack_launcher_needs_lab_mode_and_token(tmp_path):
    pytest.importorskip("flask")
    from vfl.webui import create_app
    c = create_app(str(tmp_path / "a"), token="tok", lab_mode=False).test_client()
    r = c.post("/api/attacks/run", json={"target": "x", "lab_mode_ack": True}, headers=H)
    assert r.status_code == 400 and "--lab-mode" in r.get_json()["error"]
    c = create_app(str(tmp_path / "b"), token=None, lab_mode=True).test_client()
    r = c.post("/api/attacks/run", json={"target": "x", "lab_mode_ack": True})
    assert r.status_code == 400 and "token" in r.get_json()["error"]
