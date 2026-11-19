"""Issue #10 (Order of Phoenix, rt-baseline 07ed305): epochs=10^9 was accepted, the active party pre-created one
threading.Event per configured epoch before training (~1.2 KB each) until the OOM killer ended it, and STOP could
not interrupt that loop.

Fix: `epochs` has a documented maximum (config_schema.MAX_EPOCHS, refused with HTTP 400 at the coordinator), and the
active party creates each epoch's event just before that epoch's barrier, so its memory does not grow with `epochs`."""
import ast
import os
import time

import pytest

from vfl import config as cfgmod
from vfl import config_schema as cs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_epochs_has_a_documented_maximum():
    assert cs.MAX_EPOCHS == 10_000
    p = next(p for p in cs.PARAMS if p.key == "epochs")
    assert p.min == 1 and p.max == cs.MAX_EPOCHS
    doc = open(os.path.join(ROOT, "docs", "CONFIG.md"), encoding="utf-8").read()
    row = next(line for line in doc.splitlines() if line.startswith("| `epochs` |"))
    assert "10000" in row.replace(",", "")


@pytest.mark.parametrize("value", [10_001, 1_000_000, 1_000_000_000])
def test_too_many_epochs_are_refused_on_every_path(value):
    with pytest.raises(ValueError, match=r"epochs must be <= 10000"):           # --set / CLI overrides
        cfgmod.load_session_config(None, [f"epochs={value}"])
    base = cfgmod.load_session_config(None, [])
    with pytest.raises(ValueError, match=r"epochs must be <= 10000"):           # coordinator: merged config
        cfgmod.validate_session_config(cfgmod.deep_merge(base, {"epochs": value}))


def test_the_maximum_itself_is_accepted():
    cfg = cfgmod.load_session_config(None, [f"epochs={cs.MAX_EPOCHS}"])
    cfgmod.validate_session_config(cfg)
    assert cfg["epochs"] == cs.MAX_EPOCHS


def test_active_party_does_not_preallocate_per_epoch_state():
    """No loop over range(cfg["epochs"]) may store into epoch_events (the reported cause, active.py:531 on
    rt-baseline). Checked on the source so it runs without grpc/torch."""
    tree = ast.parse(open(os.path.join(ROOT, "vfl", "active.py"), encoding="utf-8").read())
    for loop in (n for n in ast.walk(tree) if isinstance(n, ast.For)):
        it = loop.iter
        if not (isinstance(it, ast.Call) and getattr(it.func, "id", "") == "range"):
            continue
        if not any(isinstance(a, ast.Subscript) and getattr(a.slice, "value", None) == "epochs" for a in it.args):
            continue                                       # only loops whose bound is cfg["epochs"] (last arg)
        if len(it.args) == 1:                              # range(cfg["epochs"]): a pure pre-allocation loop
            stores = [t for n in ast.walk(loop) if isinstance(n, ast.Assign) for t in n.targets
                      if isinstance(t, ast.Subscript) and getattr(t.value, "attr", "") == "epoch_events"]
            assert not stores, f"line {loop.lineno}: pre-allocates epoch_events for every configured epoch"


# ------------------------------------------------------------------ end to end (needs grpc + torch)
@pytest.mark.slow
def test_coordinator_refuses_huge_epochs_and_stop_ends_a_long_session(tmp_path, data_dir):
    pytest.importorskip("grpc")
    pytest.importorskip("torch")
    from bench.harness import LocalDeployment, http
    with LocalDeployment(str(tmp_path / "dep"), data_dir, 1, tls=True, base_port=57050, ui_base=19380) as dep:
        coord = dep.start()
        # the reporter's request: refused up front with a message naming the parameter and the bound
        with pytest.raises(RuntimeError) as e:
            coord.start({"epochs": 1_000_000_000, "dataset_tag": "synthetic"})
        assert str(e.value).startswith("400") and "epochs must be <= 10000" in str(e.value)
        assert all(p["alive"] for p in coord.state()["parties"])                # nothing reached the parties

        # the largest allowed run trains, and STOP ends it within seconds
        sid = coord.start({"epochs": cs.MAX_EPOCHS})["session_id"]
        t0 = time.time()
        while time.time() - t0 < 60:
            d = http("GET", f"{coord.url}/api/sessions/{sid}")
            if (d.get("completed_epochs") or 0) >= 1:
                break
            time.sleep(0.3)
        assert d["state"] == "RUNNING" and d["completed_epochs"] >= 1
        coord.stop(sid)
        t_stop = time.time()
        while time.time() - t_stop < 15:
            d = http("GET", f"{coord.url}/api/sessions/{sid}")
            if d["state"] in ("STOPPED", "COMPLETED", "FAILED"):
                break
            time.sleep(0.2)
        assert d["state"] == "STOPPED", d["state"]
        assert time.time() - t_stop < 15
        coord.wait_idle()
        assert all(p["alive"] for p in coord.state()["parties"])
