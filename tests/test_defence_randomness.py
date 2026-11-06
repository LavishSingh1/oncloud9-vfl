"""v2.1 privacy fix: defence randomness must be secret to the party that adds it.

In v2 the gradient noise, the randomized-response flips and the representation noise were all derived from the
session seed, which the coordinator sends to every party - so the party each defence protects against could
recompute the noise and remove it exactly. These tests reproduce that attack and check it now fails.
"""
import hashlib
import math

import numpy as np
import pytest

from vfl import defense

D = {"type": "grad_noise", "clip": 0.05, "noise_multiplier": 4.0, "label_rr_epsilon": 0.5, "rep_noise_sigma": 0.5,
     "rep_clip": 1.0}
SEED = 42


def _clip(g, C):
    return g * np.minimum(1.0, C / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12))


def _passive_strips_noise(sent):
    """What a passive party can compute from the session config alone (seed, clip, sigma, epoch, batch, its id)."""
    noise = defense._rng(SEED, 3, 7, "passive-1").normal(0.0, D["noise_multiplier"] * D["clip"], size=sent.shape)
    return sent - noise


def test_v2_seed_noise_is_removable_which_is_why_it_is_not_the_default():
    g = np.random.default_rng(0).normal(0, 0.01, (64, 16)).astype(np.float32)
    sent = defense.apply_defense(g, D, SEED, 3, 7, "passive-1")                    # secret=None: v2 behaviour
    assert np.abs(_passive_strips_noise(sent) - _clip(g, D["clip"])).max() < 1e-6


def test_secret_noise_cannot_be_removed():
    g = np.random.default_rng(0).normal(0, 0.01, (64, 16)).astype(np.float32)
    secret = defense.new_secret()
    sent = defense.apply_defense(g, D, SEED, 3, 7, "passive-1", secret)
    err = np.abs(_passive_strips_noise(sent) - _clip(g, D["clip"]))
    assert err.mean() > 0.5 * D["noise_multiplier"] * D["clip"] * math.sqrt(2 / math.pi)   # ~ the noise scale
    # deterministic given the secret (a retried batch gets the same reply) and different per secret
    assert np.array_equal(sent, defense.apply_defense(g, D, SEED, 3, 7, "passive-1", secret))
    assert not np.array_equal(sent, defense.apply_defense(g, D, SEED, 3, 7, "passive-1", defense.new_secret()))


def _invert_rr_with_public_mask(y_rr, eps):
    keep = math.exp(eps) / (math.exp(eps) + 1)
    r = np.random.default_rng(int(hashlib.sha256(f"rr|{SEED}".encode()).hexdigest()[:12], 16))
    flip = r.random(len(y_rr)) >= keep
    return np.where(flip, 1 - y_rr, y_rr)


def test_randomized_response_flips_are_secret():
    y = np.random.default_rng(1).integers(0, 2, 20000)
    keep = math.exp(0.5) / (math.exp(0.5) + 1)
    v2 = defense.randomized_response(y, 2, D, SEED)
    assert np.mean(_invert_rr_with_public_mask(v2, 0.5) == y) == 1.0               # v2: fully invertible
    fixed = defense.randomized_response(y, 2, D, SEED, defense.new_secret())
    assert abs(np.mean(fixed == y) - keep) < 0.02                                  # still eps-RR
    # with an independent secret mask, "inverting" agrees with the truth w.p. keep^2 + (1-keep)^2 (~0.53 here):
    # worse than simply trusting the noisy labels, so the public mask no longer helps the attacker
    inv = np.mean(_invert_rr_with_public_mask(fixed, 0.5) == y)
    assert abs(inv - (keep ** 2 + (1 - keep) ** 2)) < 0.02 and inv < keep


def test_representation_noise_is_secret():
    h = np.random.default_rng(2).normal(0, 1, (32, 8)).astype(np.float32)
    s = defense.new_secret()
    sent = defense.apply_rep_noise(h, D, SEED, 5, "passive-1", s)
    guess = defense._rng(SEED, -1, 5, "rep|passive-1").normal(0.0, 0.5, size=h.shape)  # the active's attempt
    assert np.abs((sent - guess) - _clip(h, 1.0)).mean() > 0.2


def test_privacy_report_is_honest_about_scope_source_and_collusion():
    pr = defense.privacy_report({"defense": {"label_rr_epsilon": 2.0}}, 10)
    assert pr["formal_label_dp"] and "NOT covered: evaluation metrics" in pr["label_dp"]["scope"]
    seed = defense.privacy_report({"defense": {"label_rr_epsilon": 2.0, "noise_source": "seed"}}, 10)
    assert not seed["formal_label_dp"] and "NOT a privacy defence" in seed["warning"]
    gn = {"defense": {"type": "grad_noise", "noise_multiplier": 2.0}}
    one = defense.privacy_report(gn, 10, n_passive=1)["grad_channel"]
    three = defense.privacy_report(gn, 10, n_passive=3)["grad_channel"]
    assert one["epsilon"] == three["epsilon"]                                       # one curious passive
    assert three["epsilon_if_passives_collude"] > three["epsilon"]                  # all passives pooling views


# ------------------------------------------------------------------ the cores (need torch)
def test_cores_use_secret_noise_and_resume_keeps_the_same_labels():
    pytest.importorskip("torch")
    from vfl.config import load_session_config
    from vfl.core import ActiveCore, PassiveCore
    cfg = load_session_config(None, ["embed_dim=4", "defense.label_rr_epsilon=1.0"])
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(300, 5)).astype(np.float32), rng.integers(0, 2, 300)
    a1 = ActiveCore(cfg, X, y, "active", ["passive-1"])
    a2 = ActiveCore(cfg, X, y, "active", ["passive-1"])
    assert a1.noise_secret and a1.noise_secret != a2.noise_secret
    st = a1.state(0)
    a2.load(st)
    assert np.array_equal(a1.y_train, a2.y_train)                                   # resume: same noisy labels
    seedcfg = load_session_config(None, ["embed_dim=4", "defense.label_rr_epsilon=1.0", "defense.noise_source=seed"])
    b1, b2 = ActiveCore(seedcfg, X, y, "active", ["passive-1"]), ActiveCore(seedcfg, X, y, "active", ["passive-1"])
    assert b1.noise_secret is None and np.array_equal(b1.y_train, b2.y_train)       # v2: reproducible
    priv = load_session_config(None, ["embed_dim=4", "security.private_init=true"])
    w = [PassiveCore(c, X, "passive-1").param_snapshot() for c in (cfg, cfg, priv)]
    assert np.array_equal(w[0], w[1]) and not np.array_equal(w[0], w[2])
