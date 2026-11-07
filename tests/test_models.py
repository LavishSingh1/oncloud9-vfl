"""v2 items 3,4,6: bottom-model registry, CNN/GRU shape + gradient flow, multi-class, determinism."""
import numpy as np
import pytest
import torch

from vfl.config import load_session_config
from vfl.core import ActiveCore, PassiveCore, infer_num_classes, utility_metrics
from vfl.models import BOTTOM_MODELS, configure_torch, make_bottom, n_params, resolve_bottom_kind


def cfg(**kw):
    return load_session_config(None, [f"{k}={v}" for k, v in kw.items()])


def test_registry_and_shape_requirements():
    c = cfg(embed_dim=8, hidden_dim=16)
    for kind in ("mlp", "linear"):
        m = make_bottom(kind, 20, None, c)
        out = m(torch.from_numpy(np.random.rand(4, 20).astype(np.float32)))
        assert out.shape == (4, 8)
    with pytest.raises(ValueError, match="shape"):
        make_bottom("cnn", 24, None, c)              # cnn needs a shape
    with pytest.raises(ValueError, match="shape"):
        make_bottom("cnn", 24, [1, 8, 4], c)         # 32 != 24


def test_cnn_and_gru_and_cnn1d_forward_backward():
    c = cfg(embed_dim=8)
    for kind, shape, dim in (("cnn", [1, 8, 4], 32), ("cnn1d", [3, 16], 48), ("gru", [3, 16], 48)):
        torch.manual_seed(0)
        m = make_bottom(kind, dim, shape, c)
        x = torch.from_numpy(np.random.RandomState(1).rand(5, dim).astype(np.float32)).requires_grad_(True)
        out = m(x)
        assert out.shape == (5, 8), kind
        out.sum().backward()
        grads = [p.grad for p in m.parameters()]
        assert all(g is not None for g in grads) and n_params(m) > 0        # gradient flow reaches every param
        assert x.grad is not None and np.isfinite(x.grad.numpy()).all()


def test_per_party_model_mapping():
    c = cfg(bottom_model='{"active": "cnn", "default": "mlp"}')
    assert resolve_bottom_kind(c, "active") == "cnn"
    assert resolve_bottom_kind(c, "passive-1") == "mlp"
    assert set(BOTTOM_MODELS) >= {"mlp", "cnn", "cnn1d", "gru", "text", "linear"}


def test_multiclass_loss_and_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 200)
    assert infer_num_classes(y, {"num_classes": None}) == 4
    Xa = rng.normal(size=(200, 6)).astype(np.float32)
    core = ActiveCore(cfg(embed_dim=6, num_classes=4), Xa, y, "active", ["passive-1"])
    assert not core.binary and core.num_classes == 4
    h = {"passive-1": rng.normal(size=(16, 6)).astype(np.float32)}
    loss, grads, _ = core.train_step(np.arange(16), h, 0, 0)
    assert np.isfinite(loss) and grads["passive-1"].shape == (16, 6)
    prob = core.predict(np.arange(16), h)
    assert prob.shape == (16, 4)
    m = utility_metrics(y[:16], prob)
    assert m["num_classes"] == 4 and m["auroc_kind"] == "macro one-vs-rest"


def test_determinism_flag_set():
    env = configure_torch(1, True)
    assert env["threads"] == 1 and env["deterministic"]
