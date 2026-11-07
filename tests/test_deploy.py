"""v2.1 item 4: Deployment panel.

The launch/stop test drives the real API route and bench/harness.LocalDeployment; only the node program is
swapped for tests/fake_node.py, so it runs without grpc/torch. `test_real_nodes_go_live` runs the real
coordinator/active/passives and is skipped when grpc or torch is missing (marked slow).
"""
import json
import os
import subprocess
import sys
import time
import zipfile

import pytest

pytest.importorskip("flask")
psutil = pytest.importorskip("psutil")
from vfl.webui import create_app  # noqa: E402
from vfl.webui.deploy import ACTIVE_NOTE, slot_ports  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE = [sys.executable, os.path.join(ROOT, "tests", "fake_node.py")]
H = {"Authorization": "Bearer tok", "X-VFL-Request": "1"}       # what the UI sends (CSRF header, v2.3)


@pytest.fixture
def ui(tmp_path):
    app = create_app(str(tmp_path / "ws"), token="tok")
    ext = app.extensions["vfl"]
    yield app.test_client(), ext, tmp_path / "ws"
    ext["deploys"].stop_all()


def _dataset(c, ext, name, parties):
    r = c.post("/api/prep/generate", json={"values": {"partitioner": "v2", "parties": parties}, "name": name},
               headers=H).get_json()
    assert ext["jobs"].wait(r["job"], 300)["status"] == "done"


def _certs(c, ext, name, passives):
    r = c.post("/api/prep/certs", json={"name": name, "passives": passives}, headers=H).get_json()
    assert ext["jobs"].wait(r["job"], 120)["status"] == "done"


def _wait_live(c, name, n_nodes, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = c.get(f"/api/deploy/{name}", headers=H).get_json()
        if len(v["nodes"]) == n_nodes and all(n["state"] == "live" for n in v["nodes"]):
            return v
        time.sleep(0.3)
    raise AssertionError(f"not all live: {[(n['name'], n['state']) for n in v['nodes']]}")


def _alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_launch_one_active_two_passives_then_stop_leaves_no_orphans(ui):
    c, ext, ws = ui
    _dataset(c, ext, "ds3", 3)
    ext["deploys"].cmd_prefix = FAKE
    r = c.post("/api/deploy/local", json={"name": "lab", "dataset": "ds3", "n_passive": 2, "tls": "insecure",
                                          "slot": 3}, headers=H)
    assert r.status_code == 200, r.get_json()
    v = _wait_live(c, "lab", 4)
    assert [n["role"] for n in v["nodes"]] == ["coordinator", "active", "passive", "passive"]
    ports = slot_ports(3)
    assert v["nodes"][0]["ui_port"] == ports["ui"] and v["coordinator"].endswith(str(ports["grpc"]))
    assert v["nodes"][0]["dashboard"].startswith("http://127.0.0.1:")
    assert "fake passive passive-2" in c.get("/api/deploy/lab/logs/passive-2", headers=H).get_json()["log"]
    plan = json.load(open(ws / "deployments" / "lab" / "deployment.json"))
    assert plan["coordinator"] == v["coordinator"] and "127.0.0.1" in plan["hosts"]

    workdir = ws / "deployments" / "lab"
    pids = {n: json.load(open(workdir / f"{n}.pids")) for n in ("coordinator", "active", "passive-1", "passive-2")}
    every = [p for d in pids.values() for p in d.values()]
    assert all(_alive(p) for p in every)

    # restart one node: new process, old one (and its child) gone
    old = pids["passive-1"]
    c.post("/api/deploy/lab/nodes/passive-1/restart", headers=H)
    _wait_live(c, "lab", 4)
    assert not _alive(old["node"]) and not _alive(old["child"])

    # a second deployment cannot take the same slot
    r2 = c.post("/api/deploy/local", json={"name": "lab2", "dataset": "ds3", "n_passive": 1, "tls": "insecure",
                                           "slot": 3}, headers=H)
    assert r2.status_code == 400 and "slot 3" in r2.get_json()["error"]

    pids["passive-1"] = json.load(open(workdir / "passive-1.pids"))
    every = [p for d in pids.values() for p in d.values()]
    v = c.post("/api/deploy/lab/stop", headers=H).get_json()
    assert v["running"] is False and all(n["state"] == "stopped" for n in v["nodes"])
    time.sleep(0.5)
    assert [p for p in every if _alive(p)] == [], "orphan processes after stop"


def test_launch_validates_inputs(ui):
    c, ext, ws = ui
    _dataset(c, ext, "ds2", 2)
    ext["deploys"].cmd_prefix = FAKE
    bad = [({"dataset": "nope", "n_passive": 1}, "no saved dataset"),
           ({"dataset": "ds2", "n_passive": 2}, "1 passive part"),
           ({"dataset": "ds2", "n_passive": 9}, "1-8"),
           ({"dataset": "ds2", "n_passive": 1, "tls": "static", "certs": "missing"}, "no ca.crt")]
    for body, msg in bad:
        r = c.post("/api/deploy/local", json={"name": "x", "tls": "insecure", "slot": 4, **body}, headers=H)
        assert r.status_code == 400 and msg in r.get_json()["error"], (body, r.get_json())
    assert c.post("/api/deploy/local", json={"name": "x", "dataset": "ds2"}).status_code == 401


def test_one_active_per_session_is_explained():
    assert "FR-1" in ACTIVE_NOTE and "several deployments" in ACTIVE_NOTE


def test_multi_machine_bundles_hold_only_each_machines_files(ui):
    c, ext, ws = ui
    _dataset(c, ext, "ds3", 3)
    _certs(c, ext, "lan", 2)
    r = c.post("/api/deploy/bundles", json={"name": "lan", "dataset": "ds3", "certs": "lan", "n_passive": 2,
                                            "coordinator_host": "10.0.0.1", "active_host": "10.0.0.2",
                                            "passive_hosts": ["10.0.0.3", "10.0.0.4"]}, headers=H)
    assert r.status_code == 200, r.get_json()
    v = r.get_json()
    assert {m["name"] for m in v["machines"]} == {"coordinator", "active", "passive-1", "passive-2"}
    for m in v["machines"]:
        names = zipfile.ZipFile(ws / m["bundle"]).namelist()
        assert not any(os.path.basename(n) == "ca.key" or n.endswith("intermediate.key") for n in names)
        keys = [n for n in names if n.endswith(".key")]
        assert keys == [f"certs/{m['name']}.key"]                              # its own key, nobody else's
        data = [n for n in names if n.startswith("data/")]
        if m["name"] == "coordinator":
            assert data == []
        else:
            assert sorted(data) == sorted([f"data/{m['name']}.csv", f"data/{m['name']}.manifest.json"])
        assert f"run-{m['name']}.sh" in names and m["cmd"].startswith("python -m vfl.cli ")
    coord = next(m for m in v["machines"] if m["name"] == "coordinator")
    assert "--ui-token" not in coord["cmd"] and len(v["ui_token"]) >= 20             # token via env, not argv (v2.3)
    assert all(v["ui_token"] not in m["cmd"] for m in v["machines"])                  # never in a displayed command
    act = next(m for m in v["machines"] if m["name"] == "active")
    assert "--coordinator 10.0.0.1:50050" in act["cmd"] and "--advertise 10.0.0.2:50051" in act["cmd"]
    z = c.get("/api/deploy/lan/bundles/passive-2.zip", headers=H)
    assert z.status_code == 200 and z.data[:2] == b"PK"
    assert c.get("/api/deploy/lan/bundles/passive-2.zip").status_code == 401        # holds a private key
    assert c.get("/api/deploy/lan/bundles/passive-2.zip?token=tok").status_code == 200
    import io
    run = zipfile.ZipFile(io.BytesIO(c.get("/api/deploy/lan/bundles/coordinator.zip", headers=H).data))
    script = run.read("run-coordinator.sh").decode()
    assert f"export VFL_UI_TOKEN='{v['ui_token']}'" in script and "--ui-token" not in script
    for url in ("/api/deploy", "/api/deploy/lan", "/api/overview"):                 # open views never show the token
        assert v["ui_token"] not in c.get(url).get_data(as_text=True)
    bad = c.post("/api/deploy/bundles", json={"name": "lan2", "dataset": "ds3", "certs": "lan", "n_passive": 2,
                                              "coordinator_host": "x;rm -rf /", "active_host": "a",
                                              "passive_hosts": ["b", "c"]}, headers=H)
    assert bad.status_code == 400


def _real_nodes_available():
    try:
        import grpc  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _real_nodes_available(), reason="needs grpc and torch to run real nodes")
def test_real_nodes_go_live_and_register(ui):
    c, ext, ws = ui
    _dataset(c, ext, "real", 3)
    r = c.post("/api/deploy/local", json={"name": "real", "dataset": "real", "n_passive": 2, "tls": "static",
                                          "slot": 5}, headers=H)
    assert r.status_code == 200, r.get_json()
    _wait_live(c, "real", 4, timeout=120)
    t0 = time.time()
    while time.time() - t0 < 60:
        reg = c.get("/api/deploy/real", headers=H).get_json().get("registered") or []
        if sum(p["alive"] for p in reg) == 3:
            break
        time.sleep(1)
    assert sorted(p["party_id"] for p in reg) == ["active", "passive-1", "passive-2"]
    pids = [n["pid"] for n in c.get("/api/deploy/real", headers=H).get_json()["nodes"]]
    c.post("/api/deploy/real/stop", headers=H)
    time.sleep(1)
    assert not [p for p in pids if _alive(p)]
    leftovers = subprocess.run(["pgrep", "-f", str(ws / "deployments" / "real")], capture_output=True, text=True)
    assert leftovers.stdout.strip() == ""
