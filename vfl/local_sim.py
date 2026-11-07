"""Single-process references on the SAME aligned entities, split, preprocessing and seeds.

* local VFL simulation - identical split model/optimisers/batch order/codec as the networked
  system, without any network (correctness oracle: a networked fp32 run must match it).
* B0 centralized      - one model over the joined features, from the SAME model family as the
                        VFL run and with at least its capacity (v2 item 5):
                          all mlp/linear/text parties -> dense MLP (v1)
                          all cnn image tiles          -> CNN over the re-assembled full image
                          all cnn1d / gru streams      -> same family over all channels
                          mixed families               -> the split architecture trained jointly
                                                          in one process ("B0-joint")
* B1 active-only      - the active party's own bottom-model family + top model, its features only.

Reading every party's data in one process is for EXPERIMENTS ONLY; it is exactly what a real
deployment avoids.
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import codec
from .alignment import align_locally
from .config import alignment_salt
from .core import ActiveCore, PassiveCore, aligned_features, infer_num_classes, utility_metrics
from .data import PartyData, epoch_batches
from .models import (CNN1dBottom, CNNBottom, CentralizedMLP, GRUBottom, Stacked, TopModel, configure_torch,
                     conv_options, make_bottom, make_optimizer, n_params, output_dim, party_seed,
                     resolve_bottom_kind)


def _party_file(data_dir: str, name: str):
    for ext in (".csv", ".npz"):
        p = os.path.join(data_dir, name + ext)
        if os.path.exists(p):
            return p
    return None


def load_parties(data_dir: str):
    a = _party_file(data_dir, "active")
    if not a:
        raise FileNotFoundError(f"no active.csv / active.npz in {data_dir}")
    active = PartyData.load(a, require_label=True)
    passives = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "passive-*.csv")) + glob.glob(os.path.join(data_dir, "passive-*.npz"))):
        pid = os.path.splitext(os.path.basename(path))[0]
        passives[pid] = PartyData.load(path, require_label=False)
    if not passives:
        raise FileNotFoundError(f"no passive-* files in {data_dir}")
    return active, passives


def prepare(data_dir: str, cfg: dict):
    active, passives = load_parties(data_dir)
    pids = sorted(passives)
    res = align_locally([active.ids] + [passives[p].ids for p in pids], alignment_salt(cfg))
    Xa, splits = aligned_features(active, res[0].rows, cfg)
    y = active.y[np.asarray(res[0].rows)]
    Xp = {p: aligned_features(passives[p], r.rows, cfg)[0] for p, r in zip(pids, res[1:])}
    shapes = {"active": active.shape, **{p: passives[p].shape for p in pids}}
    return Xa, y, Xp, splits, res[0].n_aligned, shapes


def _eval_split(predict, idx, y, bs):
    if not len(idx):
        return None
    probs = np.concatenate([predict(idx[i:i + bs]) for i in range(0, len(idx), bs)])
    return utility_metrics(y[idx], probs)


def run_local_vfl(data_dir: str, cfg: dict, active_id: str = "active", log=print, return_cores: bool = False):
    configure_torch(cfg.get("torch_threads", 1), cfg.get("deterministic", True))
    Xa, y, Xp, splits, n_al, shapes = prepare(data_dir, cfg)
    pids = sorted(Xp)
    act = ActiveCore(cfg, Xa, y, active_id, pids, shape=shapes["active"])
    pas = {p: PassiveCore(cfg, Xp[p], p, shape=shapes[p]) for p in pids}
    cast = lambda a: codec.roundtrip(a, cfg)  # noqa: E731  same lossy codec as the wire
    rows, t0 = [], time.time()
    for epoch in range(cfg["epochs"]):
        te, losses = time.time(), []
        for b, idx in enumerate(epoch_batches(splits["train"], cfg["batch_size"], cfg["seed"], epoch)):
            hs = {p: cast(pas[p].forward(idx)) for p in pids}
            loss, grads, _ = act.train_step(idx, hs, epoch, b)
            for p in pids:
                pas[p].backward_step(cast(grads[p]))
            losses.append(loss)
        pred = lambda idx: act.predict(idx, {p: cast(pas[p].embed(idx)) for p in pids})  # noqa: E731
        row = {"epoch": epoch, "epoch_s": time.time() - te, "train_loss": float(np.mean(losses)),
               "val": _eval_split(pred, splits["val"], y, cfg["eval_batch_size"]),
               "test": _eval_split(pred, splits["test"], y, cfg["eval_batch_size"]),
               "leakage": act.epoch_leakage(), "optimizer_steps": act.optimizer_steps}
        rows.append(row)
        log(f"[local-vfl] epoch {epoch}: loss={row['train_loss']:.5f} test_auroc={_f(row['test']['auroc'])}")
    out = {"kind": "local_vfl", "n_aligned": n_al, "epochs": rows, "wall_s": time.time() - t0,
           "params": act.n_params + sum(p.n_params for p in pas.values()), "final_test": rows[-1]["test"],
           "final_train_loss": rows[-1]["train_loss"], "num_classes": act.num_classes,
           "bottom_models": {active_id: act.kind, **{p: pas[p].kind for p in pids}}}
    if return_cores:
        return out, act, pas, splits
    return out


def _f(v):
    return "n/a" if v is None else f"{v:.4f}"


def _train_single(model, X, y, splits, cfg, kind, num_classes, log=print) -> dict:
    binary = num_classes <= 2
    opt = make_optimizer(model.parameters(), cfg)
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    yt = torch.from_numpy(y.astype(np.float32) if binary else y.astype(np.int64))
    rows, t0 = [], time.time()

    def predict(idx):
        model.eval()
        with torch.no_grad():
            out = model(Xt[idx])
            if binary:
                return torch.sigmoid(out.reshape(-1)).numpy().astype(np.float64)
            return torch.softmax(out, dim=1).numpy().astype(np.float64)

    for epoch in range(cfg["epochs"]):
        te, losses = time.time(), []
        model.train()
        for idx in epoch_batches(splits["train"], cfg["batch_size"], cfg["seed"], epoch):
            out = model(Xt[idx])
            loss = (F.binary_cross_entropy_with_logits(out.reshape(-1), yt[idx]) if binary
                    else F.cross_entropy(out, yt[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().numpy()))
        row = {"epoch": epoch, "epoch_s": time.time() - te, "train_loss": float(np.mean(losses)),
               "val": _eval_split(predict, splits["val"], y, cfg["eval_batch_size"]),
               "test": _eval_split(predict, splits["test"], y, cfg["eval_batch_size"])}
        rows.append(row)
        log(f"[{kind}] epoch {epoch}: loss={row['train_loss']:.5f} test_auroc={_f(row['test']['auroc'])}")
    return {"kind": kind, "epochs": rows, "wall_s": time.time() - t0, "params": n_params(model),
            "final_test": rows[-1]["test"], "final_train_loss": rows[-1]["train_loss"]}


class _Joint(nn.Module):
    """The split architecture trained centrally: every party's bottom on its own column block."""

    def __init__(self, bottoms: list, widths: list[int], top: nn.Module):
        super().__init__()
        self.widths = widths
        for i, b in enumerate(bottoms):
            setattr(self, f"b{i}", b)
        self.n = len(bottoms)
        self.top = top

    def forward(self, x):
        hs, start = [], 0
        for i, w in enumerate(self.widths):
            hs.append(getattr(self, f"b{i}")(x[:, start:start + w]))
            start += w
        return self.top(torch.cat(hs, dim=1))


def _full_image_order(data_dir: str, names: list[str]):
    """Permutation that puts the parties' pixel columns back into original image order (needs the
    experimenter's full manifest)."""
    path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(path):
        return None, None
    m = json.load(open(path))
    if not m.get("image_shape"):
        return None, None
    cols = {p["name"]: p["columns"] for p in m["parties"]}
    joined = [c for n in names for c in cols[n]]
    try:
        idx = [int(c.split("_")[1]) for c in joined]
    except (IndexError, ValueError):
        return None, None
    return np.argsort(idx), m["image_shape"]


def build_b0(data_dir: str, cfg: dict, names: list[str], shapes: dict, widths: list[int], k_cls: int):
    """Returns (model, kind_label, column_permutation_or_None)."""
    kinds = [resolve_bottom_kind(cfg, n) for n in names]
    k = len(names)
    E, out = int(cfg["embed_dim"]), output_dim(k_cls)
    torch.manual_seed(party_seed(cfg["seed"], "b0-centralized"))
    if all(kd in ("mlp", "linear", "text") for kd in kinds):
        hidden = (int(cfg["hidden_dim"]) * k if all(kd == "mlp" for kd in kinds) else
                  int(cfg.get("text_hidden") or 0) * k if all(kd == "text" for kd in kinds) else 0)
        return CentralizedMLP(sum(widths), hidden, E * k, cfg["top_hidden"], out), "B0-centralized (dense MLP)", None
    ch = [int(c) * k for c in (cfg.get("cnn_channels") or (16, 32))]
    if all(kd == "cnn" for kd in kinds):
        perm, img = _full_image_order(data_dir, names)
        if perm is not None:
            return (Stacked(CNNBottom(img, E * k, **{**conv_options(cfg, "cnn"), "channels": ch}),
                            TopModel(E * k, cfg["top_hidden"], out)),
                    "B0-centralized (CNN on full image)", perm)
    if all(kd in ("cnn1d", "gru") for kd in kinds) and len({kd for kd in kinds}) == 1:
        ts = {tuple(shapes[n])[1] for n in names}
        if len(ts) == 1:
            full = [sum(int(shapes[n][0]) for n in names), ts.pop()]
            gru = conv_options(cfg, "gru")
            bottom = (CNN1dBottom(full, E * k, **{**conv_options(cfg, "cnn1d"), "channels": ch})
                      if kinds[0] == "cnn1d" else GRUBottom(full, E * k, gru["hidden"] * k, gru["layers"]))
            return Stacked(bottom, TopModel(E * k, cfg["top_hidden"], out)), f"B0-centralized ({kinds[0]} on all channels)", None
    bottoms = [make_bottom(kd, w, shapes[n], cfg) for kd, w, n in zip(kinds, widths, names)]
    return _Joint(bottoms, widths, TopModel(E * k, cfg["top_hidden"], out)), "B0-joint (split architecture, centralized)", None


def run_baselines(data_dir: str, cfg: dict, which=("b0", "b1", "local_vfl"), log=print) -> dict:
    configure_torch(cfg.get("torch_threads", 1), cfg.get("deterministic", True))
    Xa, y, Xp, splits, n_al, shapes = prepare(data_dir, cfg)
    pids = sorted(Xp)
    k_cls = infer_num_classes(y, cfg)
    names = ["active"] + pids
    out = {"n_aligned": n_al, "splits": {k: int(len(v)) for k, v in splits.items()}, "num_classes": k_cls,
           "positive_rate": float(np.mean(y)) if k_cls <= 2 else None, "config": cfg,
           "bottom_models": {n: resolve_bottom_kind(cfg, n) for n in names}}
    if "b0" in which:
        Xj = np.concatenate([Xa] + [Xp[p] for p in pids], axis=1)
        widths = [Xa.shape[1]] + [Xp[p].shape[1] for p in pids]
        m, label, perm = build_b0(data_dir, cfg, names, shapes, widths, k_cls)
        if perm is not None:
            Xj = Xj[:, perm]
        r = _train_single(m, Xj, y, splits, cfg, label, k_cls, log)
        r["family"] = label
        out["b0_centralized"] = r
    if "b1" in which:
        torch.manual_seed(party_seed(cfg["seed"], "active"))
        kind = resolve_bottom_kind(cfg, "active")
        model = Stacked(make_bottom(kind, Xa.shape[1], shapes["active"], cfg),
                        TopModel(int(cfg["embed_dim"]), cfg["top_hidden"], output_dim(k_cls)))
        r = _train_single(model, Xa, y, splits, cfg, "B1-active-only", k_cls, log)
        r["family"] = kind
        out["b1_active_only"] = r
    if "local_vfl" in which:
        out["local_vfl"] = run_local_vfl(data_dir, cfg, log=log)
    if "b0_centralized" in out and "local_vfl" in out:
        out["capacity_check"] = {"b0_params": out["b0_centralized"]["params"],
                                 "vfl_params": out["local_vfl"]["params"],
                                 "b0_at_least_vfl": out["b0_centralized"]["params"] >= out["local_vfl"]["params"]}
    return out


def save(obj: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
