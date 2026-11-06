"""v2.3: passive-side defences against a malicious active party, and network-facing dashboards that need the token
for every view (the two residual risks left open by the red-team report)."""
import os
import stat

import numpy as np
import pytest

from vfl import passive_guard as G


# ---------------------------------------------------------------------------------------------- crafted gradients
def test_clipping_defeats_targeted_probing_and_scaling():
    rng = np.random.default_rng(0)
    honest = rng.normal(0, 0.01, (64, 16)).astype(np.float32)
    probe = honest.copy()
    probe[0] *= 1e4                                     # the lab 'probe' attack: one row singled out
    out, n = G.clip_rows(probe, 0.1)
    assert n == 1 and np.linalg.norm(out, axis=1).max() <= 0.1 + 1e-6
    assert np.allclose(out[1:], honest[1:])             # honest rows untouched
    same, n0 = G.clip_rows(honest, None)
    assert n0 == 0 and same is honest


def test_gradient_monitor_flags_a_scaling_attack_but_not_honest_decay():
    rng = np.random.default_rng(1)
    mon = G.GradientMonitor()
    for t in range(200):                                # honest: gradients shrink as training converges
        g = rng.normal(0, 0.02 / (1 + t / 20), (64, 16))
        assert mon.check(g) is None, t
    assert mon.flags == 0
    for _ in range(3):                                  # the lab 'scale:50' attack
        assert mon.check(rng.normal(0, 1.0, (64, 16))) is not None
    assert mon.should_stop()
    assert mon.check(np.full((4, 16), np.nan)) is not None


def test_gradient_monitor_baseline_cannot_be_drifted_by_flagged_batches():
    mon = G.GradientMonitor(warmup=5)
    for _ in range(20):
        mon.check(np.full((8, 4), 0.01))
    m = mon.mean
    for _ in range(50):
        mon.check(np.full((8, 4), 10.0))
    assert mon.mean == m


def test_ridge_audit_detects_invertible_representations():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2000, 10))
    A = rng.normal(size=(10, 16))
    hijacked = X @ A                                    # what feature-space hijacking aims for: H invertible to X
    tr, ev = np.arange(1500), np.arange(1500, 2000)
    assert G.ridge_reconstruction_r2(hijacked[tr], X[tr], hijacked[ev], X[ev]) > 0.99
    unrelated = rng.normal(size=(2000, 16))
    assert G.ridge_reconstruction_r2(unrelated[tr], X[tr], unrelated[ev], X[ev]) < 0.05
    noisy = hijacked + rng.normal(0, 8.0, hijacked.shape)   # representation noise lowers what can be recovered
    assert G.ridge_reconstruction_r2(noisy[tr], X[tr], noisy[ev], X[ev]) < 0.9
    res = G.leakage_audit(lambda idx: hijacked[idx], X, tr, ev)
    assert res["ridge_r2"] > 0.99 and res["rows_eval"] == 500


def test_guard_is_off_by_default_and_configurable():
    from vfl.config import load_session_config
    g = G.guard_config(load_session_config(None))
    assert g == {"grad_clip": None, "max_drift": None, "grad_monitor": "off", "leakage_audit": "off", "r2_max": 0.9}
    cfg = load_session_config(None, ["security.passive_grad_clip=0.1", "security.grad_monitor=stop",
                                     "security.leakage_audit=warn", "security.leakage_r2_max=0.5",
                                     "security.max_param_drift=2.0"])
    g = G.guard_config(cfg)
    assert g["grad_clip"] == 0.1 and g["grad_monitor"] == "stop" and g["r2_max"] == 0.5 and g["max_drift"] == 2.0
    with pytest.raises(ValueError):
        load_session_config(None, ["security.grad_monitor=maybe"])
    sec = load_session_config(os.path.join(os.path.dirname(__file__), "..", "configs", "session.secure.yaml"))
    assert G.guard_config(sec)["grad_monitor"] != "off" and G.guard_config(sec)["leakage_audit"] != "off"


def test_parameter_drift_is_bounded():
    pytest.importorskip("torch")
    from vfl.config import load_session_config
    from vfl.core import PassiveCore
    cfg = load_session_config(None, ["security.max_param_drift=0.5", "lr=0.1"])
    X = np.random.default_rng(0).normal(size=(64, 6)).astype(np.float32)
    core = PassiveCore(cfg, X, "passive-1")
    idx = np.arange(32)
    for _ in range(50):                                 # a malicious active pushing hard in one direction
        h = core.forward(idx)
        core.backward_step(np.full(h.shape, 5.0, np.float32))
    assert core.drift_projections > 0 and core.last_drift <= 0.5 + 1e-6


def test_malicious_active_modes_are_lab_only():
    pytest.importorskip("torch")
    pytest.importorskip("grpc")
    from vfl.active import ActiveNode
    node = ActiveNode.__new__(ActiveNode)
    node.lab_mode, node.malicious = True, "probe"
    import collections
    node.counters = collections.Counter()
    g = {"passive-1": np.ones((4, 3), np.float32)}
    out = ActiveNode.tamper_gradients(node, g, 0, 0)
    assert out["passive-1"][0, 0] == 1e4 and g["passive-1"][0, 0] == 1.0


# ---------------------------------------------------------------------------------------------- dashboards
class _Party:
    party_id = "passive-1"

    def ui_state(self):
        return {"role": "passive", "node": "passive-1", "data": {"feature_names": ["income", "age"]}}


def test_network_facing_dashboard_needs_the_token_for_every_view(tmp_path):
    """v2.2: a party dashboard on 0.0.0.0 showed feature names, row counts and the session config to anyone on
    the LAN, token or not (the token only guarded actions)."""
    from vfl.dashboard import create_app, resolve_ui_auth
    token, private = resolve_ui_auth("0.0.0.0", None, None, "passive-1", str(tmp_path))
    assert private and token
    path = tmp_path / "passive-1" / "ui_token"
    assert path.read_text().strip() == token and stat.S_IMODE(os.stat(path).st_mode) == 0o600
    c = create_app(_Party(), "passive", token=token, private_views=private).test_client()
    assert c.get("/api/state").status_code == 401
    assert c.get("/metrics").status_code == 401
    assert c.get("/").status_code == 200                 # static shell only, no data
    r = c.get("/api/state", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200 and "income" in r.get_data(as_text=True)
    assert c.get(f"/api/state?token={token}").status_code == 200
    # loopback and the explicit opt-out keep the v2.2 behaviour
    assert resolve_ui_auth("127.0.0.1", None, None, "x", str(tmp_path)) == (None, False)
    assert resolve_ui_auth("0.0.0.0", "t", None, "x", str(tmp_path), public_views=True) == ("t", False)
    c2 = create_app(_Party(), "passive").test_client()
    assert c2.get("/api/state").status_code == 200


def test_bundles_give_every_dashboard_its_own_token(tmp_path):
    pytest.importorskip("flask")
    import zipfile
    from vfl import security as sec
    from vfl.data import prepare_vertical_dataset
    from vfl.webui.dataprep import Workspace
    from vfl.webui.deploy import DeploymentManager
    ws = Workspace(str(tmp_path / "ws"))
    prepare_vertical_dataset(ws.path("datasets", "d"), "breast_cancer", n_passive=1, overlap=0.9, seed=0)
    sec.generate_all(ws.path("certs", "c"), ["coordinator", "active", "passive-1"])
    out = DeploymentManager(ws).make_bundles("lan", "d", "c", 1, "10.0.0.1", "10.0.0.2", ["10.0.0.3"])
    toks = out["ui_tokens"]
    assert set(toks) == {"coordinator", "active", "passive-1"} and len(set(toks.values())) == 3
    for m in out["machines"]:
        assert toks[m["name"]] not in m["cmd"]
        with zipfile.ZipFile(os.path.join(ws.root, m["bundle"])) as z:
            assert f"export VFL_UI_TOKEN='{toks[m['name']]}'" in z.read(f"run-{m['name']}.sh").decode()
