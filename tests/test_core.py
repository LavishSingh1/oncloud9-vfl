"""FR-4 / FR-5 / FR-7: representations only, correctly shaped gradients, parameters change;
local VFL equals the transport-free composition used by the networked services."""
import numpy as np

from vfl import codec
from vfl.core import ActiveCore, PassiveCore
from vfl.defense import LeakageMeter, apply_defense


def _setup(cfg, n=200, da=5, dp=7):
    rng = np.random.default_rng(0)
    Xa, Xp = rng.normal(size=(n, da)).astype(np.float32), rng.normal(size=(n, dp)).astype(np.float32)
    y = (Xa[:, 0] + Xp[:, 0] > 0).astype(np.float32)
    return ActiveCore(cfg, Xa, y, "active", ["passive-1"]), PassiveCore(cfg, Xp, "passive-1"), y


def test_forward_backward_shapes_and_updates(cfg):
    act, pas, _ = _setup(cfg)
    idx = np.arange(32)
    before = pas.param_snapshot().copy()
    h = pas.forward(idx)
    assert h.shape == (32, cfg["embed_dim"])                       # a representation, not 7 raw columns
    loss, grads, _ = act.train_step(idx, {"passive-1": h}, 0, 0)
    assert np.isfinite(loss)
    assert grads["passive-1"].shape == h.shape
    pas.backward_step(grads["passive-1"])
    assert not np.allclose(before, pas.param_snapshot())           # FR-7
    assert act.optimizer_steps == pas.optimizer_steps == 1


def test_codec_roundtrip_and_byte_count():
    a = np.random.default_rng(1).normal(size=(64, 16)).astype(np.float32)
    t = codec.encode(a, "fp32")
    assert len(t.data) == codec.expected_tensor_bytes(64, 16, "fp32") == 64 * 16 * 4
    assert np.array_equal(codec.decode(t), a)
    t16 = codec.encode(a, "fp16")
    assert len(t16.data) == 64 * 16 * 2
    assert np.allclose(codec.decode(t16), a, atol=1e-2)


def test_shape_mismatch_rejected(cfg):
    import pytest
    act, pas, _ = _setup(cfg)
    with pytest.raises(ValueError):
        act.train_step(np.arange(32), {"passive-1": np.zeros((31, cfg["embed_dim"]), np.float32)}, 0, 0)


def test_defense_reduces_leakage(cfg):
    act, pas, y = _setup(cfg)
    m_off, m_on = LeakageMeter(), LeakageMeter()
    for b in range(5):
        idx = np.arange(b * 32, (b + 1) * 32)
        loss, g, _ = act.train_step(idx, {"passive-1": pas.forward(idx)}, 0, b)
        pas.backward_step(g["passive-1"])
        m_off.update(g["passive-1"], y[idx])
        noisy = apply_defense(g["passive-1"], {"type": "grad_noise", "clip": 0.01, "noise_multiplier": 4.0}, 1, 0, b, "p")
        m_on.update(noisy, y[idx])
    assert m_off.result()["cosine_leak_auc"] > 0.9
    assert m_on.result()["cosine_leak_auc"] < m_off.result()["cosine_leak_auc"]
