"""Regression tests for the NFR-8 / NFR-2 reporting gaps (Order of Phoenix, rt-baseline 07ed305).

Part A: the grad-noise privacy block must carry its adjacency, L2 sensitivity and composition, so the epsilon can be
checked and re-derived from the report alone; the benchmark's NFR-8 line and the docs must state them too.
Part B: NFR-2 must report final-loss variation across three same-seed repetitions, and label the across-seed CV with
the number of seeds actually run (not a hard-coded "3-seed")."""
import ast
import json
import math
import os
import types

import pytest

from vfl import defense

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the reporter's session: defense.type=grad_noise, clip=0.05, noise_multiplier=1.0, epochs=5, 1 passive
REPORTED = {"defense": {"type": "grad_noise", "clip": 0.05, "noise_multiplier": 1.0}}
REPORTED_EPS = 30.12663110385034


# ------------------------------------------------------------------ Part A: the privacy block
def test_grad_channel_reports_adjacency_sensitivity_and_composition():
    gc = defense.privacy_report(REPORTED, 5, n_passive=1)["grad_channel"]
    assert gc["epsilon"] == pytest.approx(REPORTED_EPS, abs=1e-12)          # the number itself is unchanged
    assert gc["delta"] == 1e-5 and gc["rdp_order"] == 2
    assert "one aligned entity" in gc["adjacency"] and "label" in gc["adjacency"]
    assert gc["protected_information"].startswith("each aligned entity's training label")
    assert gc["sensitivity"]["norm"] == "L2" and gc["sensitivity"]["formula"] == "2C"
    assert gc["sensitivity"]["value"] == pytest.approx(0.1)
    assert gc["noise_std"] == pytest.approx(0.05) and gc["effective_noise_multiplier"] == pytest.approx(0.5)
    comp = gc["composition"]
    assert comp["releases_per_label"] == 5 and "once per label per epoch" in comp["rule"]
    assert "no amplification by subsampling" in comp["rule"] and "RDP" in comp["accountant"]


def test_epsilon_can_be_rederived_from_the_reported_fields_alone():
    for sigma, clip, epochs, n_passive in ((1.0, 0.05, 5, 1), (2.0, 0.1, 10, 3), (0.5, 1.0, 1, 2)):
        cfg = {"defense": {"type": "grad_noise", "clip": clip, "noise_multiplier": sigma, "delta": 1e-6}}
        gc = defense.privacy_report(cfg, epochs, n_passive)["grad_channel"]
        z = gc["noise_std"] / gc["sensitivity"]["value"]
        a, n = gc["rdp_order"], gc["composition"]["releases_per_label"]
        by_hand = n * a / (2 * z * z) + math.log((a - 1) / a) - (math.log(gc["delta"]) + math.log(a)) / (a - 1)
        assert gc["epsilon"] == pytest.approx(max(0.0, by_hand), rel=1e-12)
        n_all = gc["composition"]["releases_per_label_if_passives_collude"]
        assert n_all == epochs * n_passive
        eps_all, _ = defense.gaussian_rdp_to_dp(z, n_all, gc["delta"])
        assert gc["epsilon_if_passives_collude"] == pytest.approx(eps_all if n_passive > 1 else gc["epsilon"])


def test_label_dp_block_states_adjacency_and_single_release():
    ld = defense.privacy_report({"defense": {"label_rr_epsilon": 2.0}}, 10, n_passive=3)["label_dp"]
    assert ld["adjacency"] == defense.LABEL_ADJACENCY and ld["sensitivity"] is None
    assert ld["composition"]["releases_per_label"] == 1 and "post-processing" in ld["composition"]["rule"]


def test_privacy_block_is_json_serialisable():
    pr = defense.privacy_report({"defense": {**REPORTED["defense"], "label_rr_epsilon": 1.0}}, 5, 2)
    assert json.loads(json.dumps(pr))["grad_channel"]["adjacency"]


def test_docs_define_adjacency_for_grad_noise_not_only_randomized_response():
    doc = open(os.path.join(ROOT, "docs", "PROJECT_DOCUMENTATION.md"), encoding="utf-8").read()
    gn = doc[doc.index("* **Gradient noise** (`grad_noise`)"):doc.index("* Both are evaluated")]
    for needle in ("adjacency", "2C", "one batch per epoch", "E × n_passive", "30.13"):
        assert needle in gn, needle
    row = next(line for line in doc.splitlines() if line.startswith("| NFR-8 Privacy evaluation"))
    assert "sensitivity and composition" in row


# ------------------------------------------------------------------ benchmark analysis (NFR-8 and NFR-2)
def _analyse(tmp_path, seeds=(1, 2), repeats=2, edit=None):
    pytest.importorskip("matplotlib")
    from bench import analyze as A
    from bench_fixture import make_bench
    path = make_bench(str(tmp_path / "b"), seeds=seeds, repeats=repeats)
    if edit:
        edit(path)
    B = A.Bench(path)
    R = {"nfr": {}, "tables": {}}
    fig = str(tmp_path / "fig")
    os.makedirs(fig, exist_ok=True)
    A.analyse_e1(B, R, fig)
    A.analyse_def(B, R, fig)
    return R


def test_nfr8_line_and_table_carry_adjacency_sensitivity_and_composition(tmp_path):
    R = _analyse(tmp_path)
    v = R["nfr"]["NFR-8"]["value"]
    assert "one-label-changed adjacency" in v and "L2 sensitivity 2C=0.1000" in v
    assert "once per label per epoch" in v and "RDP accountant" in v and "label ε=2.000 (δ=0)" in v
    table = R["tables"]["NFR-8 privacy accounting: adjacency, sensitivity, composition"]
    assert "| sigma1 | Gaussian on dL/dH" in table and "2C = 0.1000 (C = 0.0500)" in table


def test_nfr8_rederives_assumptions_for_results_recorded_before_the_fix(tmp_path):
    def old_block(path):       # an rt-baseline privacy block: epsilon present, no adjacency/sensitivity
        for fn in os.listdir(os.path.join(path, "sessions")):
            p = os.path.join(path, "sessions", fn)
            d = json.load(open(p))
            if (d["config"].get("defense") or {}).get("type") == "grad_noise":
                d["finals"]["active"]["privacy"] = {"grad_channel": {"epsilon": 1.0, "delta": 1e-5, "rdp_order": 2}}
                json.dump(d, open(p, "w"))
    v = _analyse(tmp_path, edit=old_block)["nfr"]["NFR-8"]["value"]
    assert "sigma1: dL/dH-channel ε=26.127" in v and "2C=0.1000" in v      # re-derived from the run's config


def test_nfr2_label_uses_the_number_of_seeds_actually_run(tmp_path):
    v = _analyse(tmp_path, seeds=(1, 2), repeats=3)["nfr"]["NFR-2"]["value"]
    assert "2-seed final-loss CV" in v and "3-seed" not in v
    v1 = _analyse(tmp_path / "one", seeds=(1,), repeats=3)["nfr"]["NFR-2"]["value"]
    assert "across-seed final-loss CV n/a (1 seed)" in v1 and "1-seed final-loss CV" not in v1


def test_nfr2_single_same_seed_repeat_is_not_reported_as_pass(tmp_path):
    R = _analyse(tmp_path, repeats=2)           # the rt-baseline layout: original + one repeat
    assert R["nfr"]["NFR-2"]["status"] == "partial"
    assert "NFR-2 asks for 3 repetitions, only 2 ran" in R["nfr"]["NFR-2"]["value"]


def test_nfr2_three_repetitions_pass_and_every_repetition_counts(tmp_path):
    R = _analyse(tmp_path, repeats=3)
    rp = R["e1"]["reproducibility"]
    assert R["nfr"]["NFR-2"]["status"] == "pass" and rp["repetitions"] == 3
    assert len(rp["repetition_final_losses"]) == 3
    assert R["nfr"]["NFR-2"]["value"].startswith("3 same-seed repetitions (seed 1)")

    def drift_third(path):     # only the LAST repetition drifts: the old code compared run 1 vs run 2 and passed
        m = json.load(open(os.path.join(path, "manifest.json")))
        sid = [e for e in m["sessions"] if e["params"].get("repeat_index") == 2][0]["session_id"]
        p = os.path.join(path, "sessions", f"{sid}.json")
        d = json.load(open(p))
        d["finals"]["active"]["final_train_loss"] *= 1.10
        json.dump(d, open(p, "w"))
    R2 = _analyse(tmp_path / "drift", repeats=3, edit=drift_third)
    assert R2["nfr"]["NFR-2"]["status"] == "investigate"
    assert R2["e1"]["reproducibility"]["repeat_final_loss_rel_diff_pct"] > 5.0


# ------------------------------------------------------------------ the runner schedules three repetitions
def test_run_benchmarks_defaults_to_three_repetitions():
    tree = ast.parse(open(os.path.join(ROOT, "bench", "run_benchmarks.py")).read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
             and n.args and getattr(n.args[0], "value", None) == "--repeats"]
    assert calls and {k.arg: k.value.value for k in calls[0].keywords if k.arg == "default"} == {"default": 3}


def test_e1_runs_the_first_seed_three_times(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("grpc")
    from bench import run_benchmarks as RB
    seen = []
    stub = types.SimpleNamespace(a=types.SimpleNamespace(seeds=[1, 2], repeats=3), data=None,
                                 session=lambda suite, label, cfg, params=None: seen.append((label, params)))
    RB.Runner.e1(stub)
    assert [p["seed"] for _, p in seen] == [1, 2, 1, 1]
    assert [p.get("repeat_index") for _, p in seen if p.get("repeat")] == [1, 2]
