"""v2 items 32, 33: label randomized response (formal eps) and the RDP accountant."""
import numpy as np

from vfl import defense


def test_randomized_response_flip_rate_matches_epsilon():
    y = np.random.default_rng(0).integers(0, 2, 20000)
    out = defense.randomized_response(y, 2, {"label_rr_epsilon": 1.0}, seed=1)
    import math
    keep = math.exp(1.0) / (math.exp(1.0) + 1)
    assert abs((out == y).mean() - keep) < 0.02           # ~73% kept for eps=1, k=2


def test_privacy_report_formal_vs_not():
    pr = defense.privacy_report({"defense": {"label_rr_epsilon": 2.0}}, 10)
    assert pr["formal_label_dp"] and pr["label_dp"]["epsilon"] == 2.0 and pr["label_dp"]["delta"] == 0.0
    pr2 = defense.privacy_report({"defense": {"type": "grad_noise", "noise_multiplier": 1.0}}, 5)
    assert not pr2["formal_label_dp"]
    gc = pr2["grad_channel"]
    assert gc["epsilon"] > 0 and "NOT end-to-end" in gc["scope"]


def test_rdp_epsilon_decreases_with_more_noise():
    e_lo, _ = defense.gaussian_rdp_to_dp(0.5, 10, 1e-5)
    e_hi, _ = defense.gaussian_rdp_to_dp(4.0, 10, 1e-5)
    assert e_hi < e_lo


def test_rep_noise_changes_output_only_when_on():
    h = np.ones((4, 8), np.float32)
    assert np.array_equal(defense.apply_rep_noise(h, {}, 1, 1, "p"), h)
    noisy = defense.apply_rep_noise(h, {"rep_noise_sigma": 1.0, "rep_clip": 1.0}, 1, 1, "p")
    assert not np.array_equal(noisy, h)
