"""Split PyTorch models: a registry of bottom models (one per party), the top model at the
active party, and centralized reference models for the baselines.

Bottom-model registry (session key `bottom_model`, a name or a per-party mapping):

  mlp     Linear-ReLU-Linear (v1 default; `hidden_dim: 0` makes it linear)
  linear  single Linear layer
  cnn     2-D CNN over image tiles; flattened CSV pixels are reshaped with the manifest `shape` [C,H,W]
          (`cnn_channels`, `cnn_kernel`, `cnn_pool`)
  cnn1d   1-D CNN over sensor streams; `shape` [C,T] (`cnn_channels`, `cnn1d_kernel`, `cnn1d_pool`)
  gru     GRU over sensor streams; `shape` [C,T] (`gru_hidden`, `gru_layers`)
  text    trainable projection of FROZEN text-encoder embeddings (the encoder ran offline at the party, see
          `vfl.cli embed-text`; it is a feature extractor, not part of split learning). `text_hidden: 0` is
          the v2 linear projection; > 0 adds one hidden layer.

The v2.1 options (kernel, pooling, layers, text_hidden) default to the v2 networks exactly: same layers in
the same order, so the same seed gives the same weights. Keys and defaults live in vfl/config_schema.py.

Every bottom model maps [batch, input_dim] -> [batch, embed_dim], so the wire contract and the top
model are the same whatever each party uses.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn

from .config_schema import BOTTOM_MODELS  # single source of truth (torch-free)
SHAPED = {"cnn": 3, "cnn1d": 2, "gru": 2}            # model -> required len(shape)


def party_seed(seed: int, party_id: str) -> int:
    """Stable per-party init seed so local simulation and networked runs match exactly."""
    h = int(hashlib.sha256(party_id.encode()).hexdigest()[:8], 16)
    return (seed * 1_000_003 + h) % (2 ** 31)


def configure_torch(threads: int = 1, deterministic: bool = True) -> dict:
    """Fixed thread count + deterministic kernels so CPU runs are repeatable (v2 item 6)."""
    torch.set_num_threads(max(1, int(threads)))
    try:                                   # may only be set once per process
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    if deterministic:
        torch.use_deterministic_algorithms(True)
    return {"torch": torch.__version__, "threads": torch.get_num_threads(),
            "deterministic": bool(deterministic)}


# ------------------------------------------------------------------ bottom models
class BottomModel(nn.Module):
    """v1 MLP bottom (kept byte-for-byte so v1 runs reproduce)."""

    def __init__(self, in_dim: int, hidden: int, embed: int):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed))
        else:
            self.net = nn.Sequential(nn.Linear(in_dim, embed))

    def forward(self, x):
        return self.net(x)


class CNNBottom(nn.Module):
    """[Conv(k)-ReLU] x len(channels) - [Pool2] - Linear over a [C,H,W] tile (v2: channels (16, 32), k=3, auto pool)."""

    def __init__(self, shape, embed: int, channels=(16, 32), kernel: int = 3, pool: str = "auto"):
        super().__init__()
        c, h, w = (int(v) for v in shape)
        self.shape = (c, h, w)
        chs = [int(v) for v in channels]
        k = int(kernel)
        layers, cin = [], c
        for co in chs:
            layers += [nn.Conv2d(cin, co, k, padding=k // 2), nn.ReLU()]
            cin = co
        if pool == "auto":
            pool = "max" if min(h, w) >= 4 else "none"
        if pool in ("max", "avg"):
            if min(h, w) < 2:
                raise ValueError(f"cnn_pool={pool!r} needs a tile of at least 2x2, got {h}x{w}")
            layers.append(nn.MaxPool2d(2) if pool == "max" else nn.AvgPool2d(2))
            h, w = h // 2, w // 2
        elif pool != "none":
            raise ValueError(f"unknown cnn_pool {pool!r}")
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(chs[-1] * h * w, embed)

    def forward(self, x):
        c, h, w = self.shape
        z = self.features(x.reshape(x.shape[0], c, h, w))
        return self.head(z.reshape(z.shape[0], -1))


class CNN1dBottom(nn.Module):
    """[Conv1d(k)-ReLU] x len(channels) - pool over time - Linear over a [C,T] window (v2: k=5, mean pool)."""

    def __init__(self, shape, embed: int, channels=(16, 32), kernel: int = 5, pool: str = "mean"):
        super().__init__()
        c, t = (int(v) for v in shape)
        self.shape = (c, t)
        chs = [int(v) for v in channels]
        k = int(kernel)
        if pool not in ("mean", "max"):
            raise ValueError(f"unknown cnn1d_pool {pool!r}")
        self.pool = pool
        layers, cin = [], c
        for co in chs:
            layers += [nn.Conv1d(cin, co, k, padding=k // 2), nn.ReLU()]
            cin = co
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(chs[-1], embed)

    def forward(self, x):
        c, t = self.shape
        z = self.features(x.reshape(x.shape[0], c, t))
        return self.head(z.mean(dim=2) if self.pool == "mean" else z.amax(dim=2))


class GRUBottom(nn.Module):
    """GRU over a [C,T] window (time-major after transpose); last layer's final hidden -> Linear."""

    def __init__(self, shape, embed: int, hidden: int = 32, layers: int = 1):
        super().__init__()
        c, t = (int(v) for v in shape)
        self.shape = (c, t)
        self.layers = int(layers)
        self.gru = nn.GRU(c, int(hidden), num_layers=self.layers, batch_first=True)
        self.head = nn.Linear(int(hidden), embed)

    def forward(self, x):
        c, t = self.shape
        seq = x.reshape(x.shape[0], c, t).permute(0, 2, 1)      # [B, T, C]
        _, h = self.gru(seq)                                     # [layers, B, hidden]
        last = h.reshape(x.shape[0], -1) if self.layers == 1 else h[-1]
        return self.head(last)


def conv_options(cfg: dict, kind: str) -> dict:
    """Constructor options for a shaped bottom model from the session config (defaults = v2)."""
    if kind == "cnn":
        return {"channels": cfg.get("cnn_channels") or (16, 32), "kernel": int(cfg.get("cnn_kernel") or 3),
                "pool": cfg.get("cnn_pool") or "auto"}
    if kind == "cnn1d":
        return {"channels": cfg.get("cnn_channels") or (16, 32), "kernel": int(cfg.get("cnn1d_kernel") or 5),
                "pool": cfg.get("cnn1d_pool") or "mean"}
    if kind == "gru":
        return {"hidden": int(cfg.get("gru_hidden") or 32), "layers": int(cfg.get("gru_layers") or 1)}
    raise ValueError(kind)


def resolve_bottom_kind(cfg: dict, party_id: str) -> str:
    bm = cfg.get("bottom_model") or "mlp"
    if isinstance(bm, dict):
        bm = bm.get(party_id, bm.get("default", "mlp"))
    if bm not in BOTTOM_MODELS:
        raise ValueError(f"unknown bottom_model {bm!r} for {party_id}; choose from {BOTTOM_MODELS}")
    return bm


def make_bottom(kind: str, in_dim: int, shape, cfg: dict) -> nn.Module:
    E = int(cfg["embed_dim"])
    if kind in SHAPED:
        if not shape or len(shape) != SHAPED[kind]:
            raise ValueError(f"bottom_model {kind!r} needs a {SHAPED[kind]}-D input shape from the party "
                             f"manifest (got {shape}); prepare data with an image/sensor partition")
        if int(np.prod(shape)) != int(in_dim):
            raise ValueError(f"shape {shape} does not match {in_dim} inputs")
    if kind == "mlp":
        return BottomModel(in_dim, int(cfg["hidden_dim"]), E)
    if kind == "linear":
        return BottomModel(in_dim, 0, E)
    if kind == "text":
        return BottomModel(in_dim, int(cfg.get("text_hidden") or 0), E)
    if kind == "cnn":
        return CNNBottom(shape, E, **conv_options(cfg, "cnn"))
    if kind == "cnn1d":
        return CNN1dBottom(shape, E, **conv_options(cfg, "cnn1d"))
    if kind == "gru":
        return GRUBottom(shape, E, **conv_options(cfg, "gru"))
    raise ValueError(kind)


# ------------------------------------------------------------------ top / centralized
def output_dim(num_classes: int) -> int:
    """Binary tasks keep one logit + BCE (v1); k > 2 classes use k logits + cross-entropy."""
    return 1 if int(num_classes) <= 2 else int(num_classes)


class TopModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int = 1):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        else:
            self.net = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, z):
        return self.net(z)


class CentralizedMLP(nn.Module):
    """B0 reference for MLP parties: dense layers over the joined features with capacity >= the split
    model (the split model's first two layers are block-diagonal; here they are dense)."""

    def __init__(self, in_dim: int, hidden_total: int, embed_total: int, top_hidden: int, out_dim: int = 1):
        super().__init__()
        layers = []
        if hidden_total > 0:
            layers += [nn.Linear(in_dim, hidden_total), nn.ReLU(), nn.Linear(hidden_total, embed_total)]
        else:
            layers += [nn.Linear(in_dim, embed_total)]
        layers += TopModel(embed_total, top_hidden, out_dim).net
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Stacked(nn.Module):
    """bottom -> top as one module (B0 for CNN/sequence families, B1 active-only)."""

    def __init__(self, bottom: nn.Module, top: nn.Module):
        super().__init__()
        self.bottom = bottom
        self.top = top

    def forward(self, x):
        return self.top(self.bottom(x))


def make_optimizer(params, cfg: dict):
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=float(cfg["lr"]), momentum=float(cfg.get("momentum", 0.0)))
    return torch.optim.Adam(params, lr=float(cfg["lr"]))


def n_params(*modules) -> int:
    return int(sum(p.numel() for m in modules for p in m.parameters()))
