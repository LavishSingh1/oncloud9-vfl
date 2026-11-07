"""Transport-independent VFL training core.

The same ActiveCore/PassiveCore objects are driven by the networked services
(active.py / passive.py) and by the single-process simulator (local_sim.py), so a
networked fp32 run can be checked against an in-process run tensor-for-tensor.

Binary tasks (the v1 default) use one logit + BCE; tasks with k > 2 classes use k logits +
cross-entropy and report macro one-vs-rest AUROC (v2 item 4).
"""
from __future__ import annotations

import os
import secrets
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import defense
from .data import PartyData, Standardizer, make_splits
from .metrics_curves import curves
from .models import TopModel, make_bottom, make_optimizer, n_params, output_dim, party_seed, resolve_bottom_kind


# ------------------------------------------------------------------ metrics
def _macro_ovr_auroc(y: np.ndarray, prob: np.ndarray):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(prob.shape[1]):
        pos = y == c
        if pos.any() and (~pos).any():
            aucs.append(roc_auc_score(pos.astype(int), prob[:, c]))
    return float(np.mean(aucs)) if aucs else None


def utility_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    """Binary: prob is P(y=1) of shape [n]. Multi-class: prob is [n, k] softmax output."""
    from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim == 1:
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        pred = (prob >= 0.5).astype(int)
        out = {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, pred)),
               "f1": float(f1_score(y_true, pred, zero_division=0)),
               "loss": float(log_loss(y_true, prob, labels=[0, 1]))}
        out["auroc"] = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None
        out["curves"] = curves(y_true, prob)                  # v2.1: ROC points + confusion counts (aggregates)
        return out
    k = prob.shape[1]
    prob = np.clip(prob, 1e-7, 1.0)
    prob = prob / prob.sum(axis=1, keepdims=True)
    pred = prob.argmax(axis=1)
    return {"n": int(len(y_true)), "num_classes": k, "accuracy": float(accuracy_score(y_true, pred)),
            "f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "loss": float(log_loss(y_true, prob, labels=list(range(k)))),
            "auroc": _macro_ovr_auroc(y_true, prob), "auroc_kind": "macro one-vs-rest",
            "curves": curves(y_true, prob)}


def infer_num_classes(y: np.ndarray, cfg: dict) -> int:
    """`num_classes: null` -> 2 if labels are {0,1}, else max+1 (labels must be 0..k-1 integers)."""
    y = np.asarray(y)
    if not np.all(np.isfinite(y)) or not np.all(np.equal(np.mod(y, 1), 0)) or (len(y) and y.min() < 0):
        raise ValueError("labels must be non-negative integers 0..k-1")
    k_data = int(y.max()) + 1 if len(y) else 2
    k = cfg.get("num_classes")
    if k is None:
        return max(2, k_data)
    if int(k) < k_data:
        raise ValueError(f"num_classes={k} but labels go up to {k_data - 1}")
    return int(k)


def noise_secret(cfg: dict) -> str | None:
    """This party's secret for its defence randomness (v2.1). `defense.noise_source: seed` restores the v2
    seed-derived noise, which every party can recompute (reproducible, NOT private)."""
    return None if (cfg.get("defense") or {}).get("noise_source", "secret") == "seed" else defense.new_secret()


def init_seed(cfg: dict, party_id: str, passive: bool) -> int:
    """Model-init seed. `security.private_init` (opt-in) makes a passive party's bottom-model initialisation secret:
    with the shared seed, the active party can compute the passive's initial weights, which helps it invert the
    representations it receives (white-box model inversion)."""
    if passive and (cfg.get("security") or {}).get("private_init"):
        return secrets.randbits(31)
    return party_seed(cfg["seed"], party_id)


def aligned_features(data: PartyData, rows: list[int], cfg: dict, return_scaler: bool = False):
    """Reorder local rows into the shared aligned order, split, and standardise using
    this party's train rows only. Returns (X_aligned, splits[, scaler])."""
    X = data.X[np.asarray(rows, dtype=np.int64)] if len(rows) else data.X[:0]
    splits = make_splits(len(rows), cfg["seed"], cfg["split"]["val"], cfg["split"]["test"])
    if len(splits["train"]) == 0:
        raise ValueError("aligned training set is empty")
    scaler = Standardizer(X[splits["train"]])
    X = scaler(X)
    return (X, splits, scaler) if return_scaler else (X, splits)


class _Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        ms = (now - self.t) * 1000.0
        self.t = now
        return ms


class ActiveCore:
    """Label owner: local bottom model + top model + loss."""

    def __init__(self, cfg: dict, X: np.ndarray, y: np.ndarray, party_id: str, passive_ids: list[str],
                 shape=None, scaler: Standardizer | None = None):
        self.cfg = cfg
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.num_classes = infer_num_classes(y, cfg)
        self.binary = self.num_classes <= 2
        self.y = np.asarray(y, dtype=np.float32 if self.binary else np.int64)
        # Training labels: optionally randomised response (formal label-DP, v2 item 32). The flips and the gradient
        # noise come from a secret only this party holds (v2.1); it is checkpointed so a resumed run keeps the SAME
        # noisy labels (a fresh draw would be a second release of every label).
        self.noise_secret = noise_secret(cfg)
        self._randomize_labels()
        self.party_id = party_id
        self.passive_ids = sorted(passive_ids)
        self.kind = resolve_bottom_kind(cfg, party_id)
        self.scaler = scaler
        torch.manual_seed(init_seed(cfg, party_id, passive=False))
        E = cfg["embed_dim"]
        self.bottom = make_bottom(self.kind, self.X.shape[1], shape, cfg)
        self.top = TopModel(E * (1 + len(self.passive_ids)), cfg["top_hidden"], output_dim(self.num_classes))
        self.opt = make_optimizer(list(self.bottom.parameters()) + list(self.top.parameters()), cfg)
        self.optimizer_steps = 0
        self.leak = defense.LeakageMeter()

    def _randomize_labels(self):
        self.y_train = defense.randomized_response(self.y, self.num_classes, self.cfg.get("defense"), self.cfg["seed"],
                                                   self.noise_secret)
        self.label_flip_rate = float(np.mean(self.y_train != self.y)) if len(self.y) else 0.0

    @property
    def n_params(self) -> int:
        return n_params(self.bottom, self.top)

    def _z(self, x, passive_h: dict[str, np.ndarray], requires_grad: bool, dropped=()):
        h_a = self.bottom(x)
        hs = []
        for p in self.passive_ids:
            if p in dropped:            # straggler policy: this party's input is zero for the batch
                hs.append(torch.from_numpy(np.zeros((x.shape[0], self.cfg["embed_dim"]), np.float32)))
                continue
            arr = passive_h[p]
            if arr.shape != (x.shape[0], self.cfg["embed_dim"]):
                raise ValueError(f"representation from {p} has shape {arr.shape}, expected "
                                 f"({x.shape[0]}, {self.cfg['embed_dim']})")
            t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
            if requires_grad:
                t.requires_grad_(True)
            hs.append(t)
        return torch.cat([h_a] + hs, dim=1), hs

    def loss_fn(self, out, y):
        if self.binary:
            return F.binary_cross_entropy_with_logits(out.reshape(-1), y)
        return F.cross_entropy(out, y)

    def train_step(self, idx: np.ndarray, passive_h: dict[str, np.ndarray], epoch: int, batch: int,
                   dropped=()):
        """One committed batch. Returns (loss, {party: gradient sent or None if dropped}, timings)."""
        tm = _Timer()
        self.bottom.train()
        self.top.train()
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.y_train[idx])
        z, hs = self._z(x, passive_h, requires_grad=True, dropped=dropped)
        loss = self.loss_fn(self.top(z), y)
        fwd_ms = tm.lap()
        self.opt.zero_grad()
        loss.backward()
        bwd_ms = tm.lap()
        self.opt.step()
        self.optimizer_steps += 1
        step_ms = tm.lap()
        grads = {}
        loss_v = float(loss.detach().numpy())
        if not np.isfinite(loss_v):
            raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {batch}")
        for p, h in zip(self.passive_ids, hs):
            if p in dropped:
                grads[p] = None
                continue
            g = h.grad.detach().numpy().astype(np.float32)
            g = defense.apply_defense(g, self.cfg.get("defense"), self.cfg["seed"], epoch, batch, p,
                                      self.noise_secret)
            if self.cfg.get("attack_eval", True):
                self.leak.update(g, self.y[idx])            # attacker's target: the TRUE labels
            grads[p] = g
        return loss_v, grads, {"fwd_ms": fwd_ms, "bwd_ms": bwd_ms, "step_ms": step_ms,
                               "defense_ms": tm.lap()}

    def predict(self, idx: np.ndarray, passive_h: dict[str, np.ndarray]) -> np.ndarray:
        """Binary: P(y=1) [n]. Multi-class: softmax probabilities [n, k]."""
        self.bottom.eval()
        self.top.eval()
        with torch.no_grad():
            z, _ = self._z(torch.from_numpy(self.X[idx]), passive_h, requires_grad=False)
            out = self.top(z)
            if self.binary:
                return torch.sigmoid(out.reshape(-1)).numpy().astype(np.float64)
            return torch.softmax(out, dim=1).numpy().astype(np.float64)

    def epoch_leakage(self) -> dict:
        r = self.leak.result()
        self.leak.reset()
        return r

    # checkpoints ------------------------------------------------------
    def state(self, epoch: int) -> dict:
        st = {"bottom": self.bottom.state_dict(), "top": self.top.state_dict(),
              "opt": self.opt.state_dict(), "epoch": epoch, "optimizer_steps": self.optimizer_steps,
              "bottom_model": self.kind, "num_classes": self.num_classes}
        if self.scaler is not None:          # needed to apply the model to new data
            st["scaler"] = {"mean": torch.as_tensor(self.scaler.mean), "std": torch.as_tensor(self.scaler.std)}
        if self.noise_secret:                # stays on this party's disk with its labels
            st["noise_secret"] = self.noise_secret
        return st

    def load(self, st: dict):
        self.bottom.load_state_dict(st["bottom"])
        self.top.load_state_dict(st["top"])
        self.opt.load_state_dict(st["opt"])
        self.optimizer_steps = int(st["optimizer_steps"])
        if st.get("noise_secret") and st["noise_secret"] != self.noise_secret:
            self.noise_secret = st["noise_secret"]
            self._randomize_labels()         # resume with the same randomized-response labels


class PassiveCore:
    """Feature owner: local bottom model only. Never sees labels or loss."""

    def __init__(self, cfg: dict, X: np.ndarray, party_id: str, shape=None, scaler: Standardizer | None = None):
        self.cfg = cfg
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.party_id = party_id
        self.kind = resolve_bottom_kind(cfg, party_id)
        self.scaler = scaler
        self.noise_secret = noise_secret(cfg)                 # for representation noise (never sent)
        torch.manual_seed(init_seed(cfg, party_id, passive=True))
        self.bottom = make_bottom(self.kind, self.X.shape[1], shape, cfg)
        self.opt = make_optimizer(self.bottom.parameters(), cfg)
        self.optimizer_steps = 0
        self._h = None
        self._rep_batch = 0
        # v2.3 trust region against a malicious active party (security.max_param_drift, vfl/passive_guard.py)
        self.max_drift = (cfg.get("security") or {}).get("max_param_drift")
        self._theta0 = [p.detach().clone() for p in self.bottom.parameters()] if self.max_drift else None
        self.drift_projections = 0
        self.last_drift = 0.0

    @property
    def n_params(self) -> int:
        return n_params(self.bottom)

    def _release(self, h: np.ndarray) -> np.ndarray:
        """Optional representation noise (v2 item 33) applied to what leaves the node."""
        self._rep_batch += 1
        return defense.apply_rep_noise(h, self.cfg.get("defense"), self.cfg["seed"], self._rep_batch, self.party_id,
                                       self.noise_secret)

    def forward(self, idx: np.ndarray) -> np.ndarray:
        self.bottom.train()
        self._h = self.bottom(torch.from_numpy(self.X[idx]))
        return self._release(self._h.detach().numpy().astype(np.float32))

    def backward_step(self, grad: np.ndarray) -> dict:
        if self._h is None:
            raise RuntimeError("backward_step called without a forward pass")
        if grad.shape != tuple(self._h.shape):
            raise ValueError(f"gradient shape {grad.shape} != representation shape {tuple(self._h.shape)}")
        tm = _Timer()
        self.opt.zero_grad()
        self._h.backward(torch.from_numpy(np.ascontiguousarray(grad, dtype=np.float32)))
        bwd = tm.lap()
        self.opt.step()
        if self._theta0 is not None:
            self._project_drift()
        self.optimizer_steps += 1
        self._h = None
        return {"bwd_ms": bwd, "step_ms": tm.lap()}

    def _project_drift(self):
        """Pull the parameters back onto the ball ||theta - theta0|| <= r * ||theta0||."""
        with torch.no_grad():
            params = list(self.bottom.parameters())
            num = torch.sqrt(sum(((p - p0) ** 2).sum() for p, p0 in zip(params, self._theta0)))
            den = torch.sqrt(sum((p0 ** 2).sum() for p0 in self._theta0)) + 1e-12
            rel = float(num / den)
            if rel > float(self.max_drift):
                f = float(self.max_drift) / rel
                for p, p0 in zip(params, self._theta0):
                    p.copy_(p0 + (p - p0) * f)
                self.drift_projections += 1
                rel = float(self.max_drift)
            self.last_drift = rel

    def skip_step(self):
        """Straggler policy dropped this party for the batch: discard the forward pass, no update."""
        self._h = None

    def embed(self, idx: np.ndarray, release: bool = True) -> np.ndarray:
        self.bottom.eval()
        with torch.no_grad():
            h = self.bottom(torch.from_numpy(self.X[idx])).numpy().astype(np.float32)
        return self._release(h) if release else h

    def param_snapshot(self) -> np.ndarray:
        return np.concatenate([p.detach().numpy().ravel() for p in self.bottom.parameters()])

    def state(self, epoch: int) -> dict:
        st = {"bottom": self.bottom.state_dict(), "opt": self.opt.state_dict(), "epoch": epoch,
              "optimizer_steps": self.optimizer_steps, "bottom_model": self.kind}
        if self.scaler is not None:
            st["scaler"] = {"mean": torch.as_tensor(self.scaler.mean), "std": torch.as_tensor(self.scaler.std)}
        if self._theta0 is not None:                  # the trust-region centre survives a resume
            st["theta0"] = [t.clone() for t in self._theta0]
        return st

    def load(self, st: dict):
        self.bottom.load_state_dict(st["bottom"])
        self.opt.load_state_dict(st["opt"])
        self.optimizer_steps = int(st["optimizer_steps"])
        if self._theta0 is not None and st.get("theta0"):
            self._theta0 = [t.clone() for t in st["theta0"]]


def save_checkpoint(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _np_safe_globals():
    """numpy types that v2.2 checkpoints contain (the standardiser's mean/std arrays). Allow-listed for the
    restricted unpickler; v2.3 checkpoints store tensors and need none of these."""
    import numpy as _np
    out = [_np.ndarray, _np.dtype]
    for mod in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            out.append(__import__(mod, fromlist=["_reconstruct"])._reconstruct)
            break
        except (ImportError, AttributeError):
            continue
    out += [type(_np.dtype(t)) for t in ("float32", "float64", "int64", "int32", "bool")]
    return out


def load_checkpoint(path: str) -> dict:
    """weights_only=True: tensors, containers and primitives only. v2.2 turned weights_only off, i.e. did a full
    pickle load, and the path came from the session config (resume_from), so whoever controlled the session
    config could run code on every party (RT-02)."""
    try:
        with torch.serialization.safe_globals(_np_safe_globals()):
            return torch.load(path, map_location="cpu", weights_only=True)
    except AttributeError:                            # torch < 2.4 has no safe_globals: tensors-only checkpoints
        return torch.load(path, map_location="cpu", weights_only=True)
