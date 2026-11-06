"""Passive-side defences against a MALICIOUS active party (v2.3, closes most of red-team residual risk 2).

The active party is the only node that sends a passive party anything that changes its model: dL/dH for every
batch. A deviating active party can craft those gradients instead of back-propagating an honest loss, for example:

  * feature-space hijacking / model steering (Pasquini et al., CCS 2021): gradients from an attacker
    discriminator push the passive's bottom model into a representation space the attacker can invert, so H
    reveals the passive's raw features;
  * targeted probing: huge gradients on chosen rows to make individual entities stand out in later H;
  * sabotage: scaled gradients that blow up the passive's model.

A passive party cannot verify gradients (it does not hold the labels or the top model), but it holds its own raw
features. That makes the strongest check possible: measure how much of X can be recovered from the H it sends.
The guard has four independent layers. All are opt-in, so the defaults and the attack demos are unchanged.

  security.passive_grad_clip   clip every received gradient row to an L2 bound. This bounds each entity's
                               influence per step (cf. DP-SGD clipping) and defeats scaling and targeted probing.
  security.max_param_drift     keep the bottom model within a relative L2 ball around its initialisation. This
                               bounds how far ANY gradient sequence can steer the model (a trust region).
  security.grad_monitor        off | warn | stop. A running z-score on the gradient norm flags gradients that
                               jump away from the honest trend. Detection only, like the active's anomaly monitor.
  security.leakage_audit       off | warn | stop. After every epoch the passive fits a ridge decoder H -> X on
                               its own train rows and scores R^2 on held-out rows, using exactly the H it sent.
                               Above security.leakage_r2_max it warns or stops. This catches hijacking (H becomes
                               invertible) and also honest-but-curious leakage, which is equally real.

Honest limits: the audit measures LINEAR recoverability (ridge). A non-linear attacker decoder can do better, so
treat the R^2 as a lower bound on leakage. Clipping and the drift bound limit steering but do not prove the
absence of leakage; only representation noise (defense.rep_noise_sigma) changes what leaves the node.
"""
from __future__ import annotations

import numpy as np

ACTIONS = ("off", "warn", "stop")


class GuardStop(RuntimeError):
    """Raised when a guard configured with action=stop decides the session must end at this party."""


def clip_rows(grad: np.ndarray, max_norm: float | None) -> tuple[np.ndarray, int]:
    """Scale every row with L2 norm > max_norm down to max_norm. Returns (clipped gradient, rows clipped)."""
    if not max_norm:
        return grad, 0
    g = np.asarray(grad, dtype=np.float32)
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    over = norms[:, 0] > float(max_norm)
    if not over.any():
        return g, 0
    scale = np.where(norms > float(max_norm), float(max_norm) / np.maximum(norms, 1e-12), 1.0)
    return (g * scale).astype(np.float32), int(over.sum())


class GradientMonitor:
    """Running mean/variance (Welford) of the per-batch mean gradient-row norm. A batch is flagged when it is more
    than `z` standard deviations ABOVE the running mean after `warmup` batches. Flagged batches are not added
    to the statistics, so an attacker cannot drift the baseline upwards with them. The standard deviation has a
    floor of 5 % of the mean, so near-constant honest gradients do not flag on noise."""

    def __init__(self, z: float = 6.0, warmup: int = 10, stop_after: int = 3):
        self.z, self.warmup, self.stop_after = float(z), int(warmup), int(stop_after)
        self.n, self.mean, self.m2 = 0, 0.0, 0.0
        self.flags = 0
        self.consecutive = 0

    def check(self, grad: np.ndarray) -> dict | None:
        x = float(np.mean(np.linalg.norm(grad, axis=1))) if grad.size else 0.0
        if not np.isfinite(x):
            self.flags += 1
            self.consecutive += 1
            return {"norm": x, "z": float("inf"), "consecutive": self.consecutive}
        if self.n >= self.warmup:
            sd = max((self.m2 / max(self.n - 1, 1)) ** 0.5, 0.05 * abs(self.mean), 1e-12)
            zz = (x - self.mean) / sd
            if zz > self.z:
                self.flags += 1
                self.consecutive += 1
                return {"norm": x, "z": zz, "mean": self.mean, "consecutive": self.consecutive}
        self.consecutive = 0
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        return None

    def should_stop(self) -> bool:
        return self.consecutive >= self.stop_after


def ridge_reconstruction_r2(H_fit: np.ndarray, X_fit: np.ndarray, H_eval: np.ndarray, X_eval: np.ndarray,
                            lam: float = 1e-2) -> float:
    """R^2 of the best linear (ridge) decoder H -> X, fit on one set of rows and scored on another. The score is
    averaged over the feature columns that vary, and is 0 when H explains nothing and 1 when X is fully
    recoverable."""
    H_fit, X_fit = np.asarray(H_fit, np.float64), np.asarray(X_fit, np.float64)
    H_eval, X_eval = np.asarray(H_eval, np.float64), np.asarray(X_eval, np.float64)
    if len(H_fit) < 2 or len(H_eval) < 2:
        return float("nan")
    hm, xm = H_fit.mean(0), X_fit.mean(0)
    A, B = H_fit - hm, X_fit - xm
    W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ B)
    pred = (H_eval - hm) @ W + xm
    var = X_eval.var(0)
    keep = var > 1e-12
    if not keep.any():
        return float("nan")
    r2 = 1.0 - ((X_eval - pred) ** 2).mean(0)[keep] / var[keep]
    return float(np.clip(r2, -1.0, 1.0).mean())


def leakage_audit(embed, X: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray, seed: int = 0,
                  max_rows: int = 4000) -> dict:
    """`embed(idx)` must return exactly what the party SENDS for those rows (after representation noise). The
    decoder is fit on train rows and scored on validation/test rows, all held by this party alone."""
    rng = np.random.default_rng(seed)
    tr = np.asarray(train_idx)
    ev = np.asarray(eval_idx)
    if len(tr) > max_rows:
        tr = rng.choice(tr, max_rows, replace=False)
    if len(ev) > max_rows:
        ev = rng.choice(ev, max_rows, replace=False)
    Xf = X.reshape(len(X), -1)
    r2 = ridge_reconstruction_r2(embed(tr), Xf[tr], embed(ev), Xf[ev])
    return {"ridge_r2": r2, "rows_fit": int(len(tr)), "rows_eval": int(len(ev))}


def guard_config(cfg: dict) -> dict:
    sec = cfg.get("security") or {}
    return {"grad_clip": sec.get("passive_grad_clip"), "max_drift": sec.get("max_param_drift"),
            "grad_monitor": sec.get("grad_monitor") or "off", "leakage_audit": sec.get("leakage_audit") or "off",
            "r2_max": float(sec.get("leakage_r2_max") if sec.get("leakage_r2_max") is not None else 0.9)}
