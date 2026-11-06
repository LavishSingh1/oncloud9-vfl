"""Defences against training-time inference, their privacy accounting, and the attack metrics
used to evaluate them (FR-13, NFR-8; v2 items 32, 33).

Threat model: honest-but-curious parties. A passive party may try to infer the active party's
labels from the gradients dL/dH it receives; the active party may try to infer a passive party's
features from the representations H it receives. Malicious (deviating) parties are outside the
core threat model (see vfl/attacks/malicious.py for lab-only experiments).

Mechanisms (all opt-in through `defense:` in the session config)

  type: grad_noise          per-example clip of dL/dH rows to L2 <= C, then N(0, (sigma*C)^2).
                            Reported as GRADIENT PERTURBATION. An RDP accountant gives (eps, delta)
                            for the dL/dH channel only; it does NOT bound label leakage through
                            the top model's parameters, so it is not an end-to-end guarantee.
  label_rr_epsilon: e       randomized response on the training labels before training
                            (Ghazi et al., NeurIPS 2021). The trained model and every gradient sent to a
                            passive party are post-processing of the noisy labels, so they are formally
                            e-label-DP (delta = 0) w.r.t. one entity's label - provided the flips are secret
                            (v2.1: a party-local secret, see new_secret). Validation/test metrics use the TRUE
                            labels and are outside this scope. This is the mechanism that satisfies NFR-8 strictly.
  rep_noise_sigma: s        passive parties clip each representation row to rep_clip and add
                            N(0, (s*rep_clip)^2) before sending (defence against feature
                            reconstruction; no formal guarantee claimed).

Attack metrics (evaluator view, computed at the active party on the EXACT tensors sent):
  norm_leak_auc   - AUROC of ||g_i|| as a score for y_i (binary tasks only)
  cosine_leak_auc - AUROC of cos(g_i, g_ref) for predicting [y_i == y_ref] (any number of classes)
Both are reported as max(auc, 1 - auc): 0.5 = no leakage, 1.0 = labels fully exposed.
"""
from __future__ import annotations

import hashlib
import math
import secrets

import numpy as np
from sklearn.metrics import roc_auc_score

GUARANTEE_TEXT = "gradient perturbation (clip + Gaussian noise); NOT a formal end-to-end DP guarantee"
LEAK_SUCCESS_AUC = 0.6        # attack AUC above this is reported as "succeeded"


def new_secret() -> str:
    """A party-local secret for its defence randomness (v2.1). Never sent to anyone: every party knows the session
    seed, so noise derived from the seed alone can be recomputed and removed by the very party it protects against."""
    return secrets.token_hex(32)


def _rng(seed: int, epoch: int, batch: int, party: str, secret: str | None = None) -> np.random.Generator:
    """secret=None reproduces the v2 seed-derived noise (`defense.noise_source: seed`, lab use only)."""
    key = f"{seed}|{epoch}|{batch}|{party}" + (f"|{secret}" if secret else "")
    h = int(hashlib.sha256(key.encode()).hexdigest()[:12 if not secret else 32], 16)
    return np.random.default_rng(h)


def apply_defense(grad: np.ndarray, cfg: dict, seed: int, epoch: int, batch: int, party: str,
                  secret: str | None = None) -> np.ndarray:
    d = cfg or {}
    if d.get("type", "none") == "none":
        return grad
    if d["type"] == "grad_noise":
        C = float(d.get("clip", 1.0))
        sigma = float(d.get("noise_multiplier", 1.0))
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        clipped = grad * np.minimum(1.0, C / np.maximum(norms, 1e-12))
        noise = _rng(seed, epoch, batch, party, secret).normal(0.0, sigma * C, size=grad.shape)
        return (clipped + noise).astype(np.float32)
    raise ValueError(f"unknown defense {d['type']!r}")


def randomized_response(y: np.ndarray, num_classes: int, cfg: dict, seed: int, secret: str | None = None) -> np.ndarray:
    """k-ary randomized response: keep the label w.p. e^eps/(e^eps + k - 1), else a uniformly
    random OTHER class. eps-label-DP for each entity's label - provided the flip randomness is secret
    (`secret`); with secret=None the flips follow from the shared seed and can be inverted by any party."""
    eps = (cfg or {}).get("label_rr_epsilon")
    if eps is None:
        return y
    k = max(2, int(num_classes))
    eps = float(eps)
    keep = math.exp(eps) / (math.exp(eps) + k - 1)
    key = f"rr|{seed}" + (f"|{secret}" if secret else "")
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:12 if not secret else 32], 16))
    yi = np.asarray(y).astype(np.int64)
    flip = rng.random(len(yi)) >= keep
    other = (yi + rng.integers(1, k, size=len(yi))) % k
    out = np.where(flip, other, yi)
    return out.astype(np.asarray(y).dtype)


def apply_rep_noise(h: np.ndarray, cfg: dict, seed: int, counter: int, party: str,
                    secret: str | None = None) -> np.ndarray:
    d = cfg or {}
    s = float(d.get("rep_noise_sigma") or 0.0)
    if s <= 0:
        return h
    C = float(d.get("rep_clip") or 1.0)
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    hc = h * np.minimum(1.0, C / np.maximum(norms, 1e-12))
    return (hc + _rng(seed, -1, counter, "rep|" + party, secret).normal(0.0, s * C, size=h.shape)).astype(np.float32)


# ------------------------------------------------------------------ accounting
RDP_ORDERS = [1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 48, 64, 128, 256]


def gaussian_rdp_to_dp(z: float, compositions: int, delta: float) -> tuple[float, float]:
    """(eps, best alpha) for `compositions` Gaussian mechanisms with noise multiplier z
    (std / L2-sensitivity), using RDP(alpha) = alpha / (2 z^2) and the conversion of
    Balle et al. 2020: eps = rdp + log((alpha-1)/alpha) - (log delta + log alpha) / (alpha - 1)."""
    if z <= 0:
        return math.inf, None
    best = (math.inf, None)
    for a in RDP_ORDERS:
        rdp = compositions * a / (2 * z * z)
        eps = rdp + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        if eps < best[0]:
            best = (eps, a)
    return max(0.0, best[0]), best[1]


LABEL_DP_SCOPE = ("formal for everything derived from the TRAINING labels: the trained model and every gradient sent "
                  "to passive parties. NOT covered: evaluation metrics (val/test AUROC, accuracy, loss, ROC/confusion, "
                  "attack AUCs) are computed on the TRUE labels at the active party and reported to the coordinator "
                  "and dashboards; val/test labels never enter training")
SEED_NOISE_WARNING = ("defense.noise_source is 'seed': the noise follows from the session seed every party holds, so the "
                      "party it protects against can recompute and remove it - NOT a privacy defence (lab comparisons only)")


def privacy_report(cfg: dict, epochs_completed: int, n_passive: int = 1) -> dict:
    """What can honestly be claimed about label privacy for this run."""
    d = cfg.get("defense") or {}
    seed_noise = d.get("noise_source", "secret") == "seed"
    out = {"label_dp": None, "grad_channel": None, "rep_noise": None, "noise_source": d.get("noise_source", "secret")}
    if seed_noise:
        out["warning"] = SEED_NOISE_WARNING
    if d.get("label_rr_epsilon") is not None:
        out["label_dp"] = {"mechanism": "randomized response on training labels",
                           "epsilon": float(d["label_rr_epsilon"]), "delta": 0.0,
                           "scope": SEED_NOISE_WARNING if seed_noise else LABEL_DP_SCOPE,
                           "unit": "one entity's label", "formal": not seed_noise}
    if d.get("type") == "grad_noise":
        sigma = float(d.get("noise_multiplier", 1.0))
        delta = float(d.get("delta", 1e-5))
        # Changing one label can move that entity's clipped row from g to g' (both ||.|| <= C):
        # L2 sensitivity 2C, noise std sigma*C  ->  z = sigma / 2. Each label is used once per epoch.
        eps, alpha = gaussian_rdp_to_dp(sigma / 2.0, max(1, epochs_completed), delta)
        # each passive party receives its own noisy dL/dH rows for every label: colluding passives see
        # n_passive releases per epoch, so their joint budget composes over epochs x passives
        eps_all, _ = gaussian_rdp_to_dp(sigma / 2.0, max(1, epochs_completed) * max(1, int(n_passive)), delta)
        out["grad_channel"] = {"mechanism": "per-example clip + Gaussian noise on dL/dH",
                               "clip": float(d.get("clip", 1.0)), "noise_multiplier": sigma,
                               "epochs": epochs_completed, "epsilon": eps, "delta": delta, "rdp_order": alpha,
                               "epsilon_if_passives_collude": eps_all if int(n_passive) > 1 else eps,
                               "n_passive": int(n_passive),
                               "scope": "dL/dH channel only - NOT end-to-end (labels also influence the top "
                                        "model's parameters, which shape later gradients)",
                               "guarantee": GUARANTEE_TEXT}
    if float(d.get("rep_noise_sigma") or 0) > 0:
        out["rep_noise"] = {"sigma": float(d["rep_noise_sigma"]), "clip": float(d.get("rep_clip") or 1.0),
                            "scope": "perturbation of representations; no formal guarantee claimed"}
    out["formal_label_dp"] = out["label_dp"] is not None and not seed_noise
    return out


def describe(cfg: dict) -> dict:
    d = cfg or {}
    out = {"type": d.get("type", "none")}
    if d.get("type", "none") != "none":
        out.update({"clip_norm": float(d.get("clip", 1.0)),
                    "noise_multiplier": float(d.get("noise_multiplier", 1.0)),
                    "protected_information": "active-party labels",
                    "adversary": "honest-but-curious passive party observing returned gradients",
                    "noise_insertion_point": "per-example rows of dL/dH before serialisation at the active party",
                    "guarantee": GUARANTEE_TEXT})
    if d.get("label_rr_epsilon") is not None:
        out["label_rr_epsilon"] = float(d["label_rr_epsilon"])
    if float(d.get("rep_noise_sigma") or 0) > 0:
        out["rep_noise_sigma"] = float(d["rep_noise_sigma"])
        out["rep_clip"] = float(d.get("rep_clip") or 1.0)
    return out


# ------------------------------------------------------------------ attack metrics
class LeakageMeter:
    """Accumulates attack scores over one epoch of sent gradients."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.norm_scores, self.labels = [], []
        self.cos_scores, self.same = [], []

    def update(self, grad: np.ndarray, y: np.ndarray):
        if len(y) < 2:
            return
        g = grad.astype(np.float64)
        n = np.linalg.norm(g, axis=1)
        self.norm_scores.append(n)
        self.labels.append(np.asarray(y))
        ref = g[0]
        cos = g[1:] @ ref / (np.maximum(n[1:], 1e-12) * max(np.linalg.norm(ref), 1e-12))
        self.cos_scores.append(cos)
        self.same.append((np.asarray(y)[1:] == np.asarray(y)[0]).astype(int))

    @staticmethod
    def _leak(labels, scores):
        labels = np.concatenate(labels)
        scores = np.concatenate(scores)
        if len(np.unique(labels)) != 2:
            return None
        auc = roc_auc_score(labels, scores)
        return float(max(auc, 1.0 - auc))

    def result(self) -> dict:
        if not self.labels:
            return {}
        binary = set(np.unique(np.concatenate(self.labels)).tolist()) <= {0, 1, 0.0, 1.0}
        return {"norm_leak_auc": self._leak(self.labels, self.norm_scores) if binary else None,
                "cosine_leak_auc": self._leak(self.same, self.cos_scores)}


def leak_outcome(auc) -> str:
    if auc is None:
        return "not measured"
    return "succeeded" if auc > LEAK_SUCCESS_AUC else "blocked"
