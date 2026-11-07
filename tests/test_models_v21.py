"""v2.1 item 2: new bottom-model options (kernel, pooling, layers, text projection). Defaults must rebuild the
v2 networks exactly (same layers, same init order -> same weights and outputs for the same seed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from vfl.config import load_session_config  # noqa: E402
from vfl.models import CNN1dBottom, CNNBottom, GRUBottom, make_bottom, n_params  # noqa: E402


# ---- the v2 classes, verbatim, as the reference
class V2CNN(nn.Module):
    def __init__(self, shape, embed, channels=(16, 32)):
        super().__init__()
        c, h, w = (int(v) for v in shape)
        self.shape = (c, h, w)
        c1, c2 = (int(v) for v in channels)
        pool = min(h, w) >= 4
        layers = [nn.Conv2d(c, c1, 3, padding=1), nn.ReLU(), nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU()]
        if pool:
            layers.append(nn.MaxPool2d(2))
            h, w = h // 2, w // 2
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c2 * h * w, embed)

    def forward(self, x):
        c, h, w = self.shape
        z = self.features(x.reshape(x.shape[0], c, h, w))
        return self.head(z.reshape(z.shape[0], -1))


class V2CNN1d(nn.Module):
    def __init__(self, shape, embed, channels=(16, 32)):
        super().__init__()
        c, t = (int(v) for v in shape)
        self.shape = (c, t)
        c1, c2 = (int(v) for v in channels)
        self.features = nn.Sequential(nn.Conv1d(c, c1, 5, padding=2), nn.ReLU(),
                                      nn.Conv1d(c1, c2, 5, padding=2), nn.ReLU())
        self.head = nn.Linear(c2, embed)

    def forward(self, x):
        c, t = self.shape
        return self.head(self.features(x.reshape(x.shape[0], c, t)).mean(dim=2))


class V2GRU(nn.Module):
    def __init__(self, shape, embed, hidden=32):
        super().__init__()
        c, t = (int(v) for v in shape)
        self.shape = (c, t)
        self.gru = nn.GRU(c, int(hidden), batch_first=True)
        self.head = nn.Linear(int(hidden), embed)

    def forward(self, x):
        c, t = self.shape
        _, h = self.gru(x.reshape(x.shape[0], c, t).permute(0, 2, 1))
        return self.head(h.reshape(x.shape[0], -1))


def _same(a, b, dim):
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    x = torch.from_numpy(np.random.RandomState(3).rand(4, dim).astype(np.float32))
    assert torch.equal(a(x), b(x))


@pytest.mark.parametrize("kind,shape,ref", [("cnn", [1, 8, 8], V2CNN), ("cnn", [3, 2, 3], V2CNN),
                                            ("cnn1d", [3, 16], V2CNN1d), ("gru", [3, 16], V2GRU)])
def test_defaults_rebuild_the_v2_networks_exactly(kind, shape, ref):
    cfg = load_session_config(None, ["embed_dim=8"])
    dim = int(np.prod(shape))
    torch.manual_seed(11)
    new = make_bottom(kind, dim, shape, cfg)
    torch.manual_seed(11)
    old = ref(shape, 8)
    _same(new, old, dim)


def test_text_hidden_zero_is_the_v2_linear_projection():
    cfg = load_session_config(None, ["embed_dim=8"])
    m = make_bottom("text", 20, None, cfg)
    assert len(list(m.net)) == 1
    m2 = make_bottom("text", 20, None, load_session_config(None, ["embed_dim=8", "text_hidden=32"]))
    assert len(list(m2.net)) == 3 and n_params(m2) > n_params(m)


@pytest.mark.parametrize("sets,shape", [
    (["bottom_model=cnn", "cnn_kernel=5", "cnn_pool=avg"], [1, 8, 8]),
    (["bottom_model=cnn", "cnn_kernel=7", "cnn_pool=none", "cnn_channels=[8, 16, 32]"], [3, 6, 6]),
    (["bottom_model=cnn", "cnn_pool=max"], [1, 2, 3]),
    (["bottom_model=cnn1d", "cnn1d_kernel=3", "cnn1d_pool=max", "cnn_channels=[16]"], [4, 20]),
    (["bottom_model=gru", "gru_layers=3", "gru_hidden=16"], [3, 12]),
])
def test_new_options_forward_backward(sets, shape):
    cfg = load_session_config(None, ["embed_dim=6"] + sets)
    dim = int(np.prod(shape))
    m = make_bottom(cfg["bottom_model"], dim, shape, cfg)
    x = torch.from_numpy(np.random.RandomState(1).rand(5, dim).astype(np.float32)).requires_grad_(True)
    out = m(x)
    assert out.shape == (5, 6)
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters()) and x.grad is not None


def test_option_effects_are_real():
    assert len([m for m in CNNBottom([1, 8, 8], 4, (8, 16, 32)).features if isinstance(m, nn.Conv2d)]) == 3
    assert isinstance(CNNBottom([1, 8, 8], 4, pool="avg").features[-1], nn.AvgPool2d)
    assert not any(isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)) for m in CNNBottom([1, 8, 8], 4, pool="none").features)
    assert CNNBottom([1, 8, 8], 4, kernel=5).features[0].kernel_size == (5, 5)
    assert CNN1dBottom([2, 10], 4, kernel=7).features[0].kernel_size == (7,)
    assert GRUBottom([2, 10], 4, layers=2).gru.num_layers == 2
    with pytest.raises(ValueError, match="2x2"):
        CNNBottom([1, 1, 4], 4, pool="max")
