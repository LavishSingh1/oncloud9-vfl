"""ROC curve points and confusion matrix for the analytics page (v2.1 item 5). Torch-free.

Only aggregate statistics leave the active party: a down-sampled ROC curve (<= 61 points) and confusion-matrix
counts, the same kind of information as the AUROC / accuracy already reported. No per-entity scores or labels.
Multi-class tasks get the macro one-vs-rest ROC (per-class TPR interpolated on a common FPR grid, averaged).
"""
from __future__ import annotations

import numpy as np

GRID = np.linspace(0.0, 1.0, 61)
MAX_CONFUSION_CLASSES = 20


def roc_points(y: np.ndarray, prob: np.ndarray) -> dict | None:
    from sklearn.metrics import roc_curve
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim == 1:
        if len(np.unique(y)) != 2:
            return None
        fpr, tpr, _ = roc_curve(y, prob)
        return {"kind": "binary", "fpr": GRID.round(4).tolist(),
                "tpr": np.interp(GRID, fpr, tpr).round(4).tolist()}
    tprs = []
    for c in range(prob.shape[1]):
        pos = y == c
        if pos.any() and (~pos).any():
            fpr, tpr, _ = roc_curve(pos.astype(int), prob[:, c])
            tprs.append(np.interp(GRID, fpr, tpr))
    if not tprs:
        return None
    return {"kind": "macro one-vs-rest", "fpr": GRID.round(4).tolist(),
            "tpr": np.mean(tprs, axis=0).round(4).tolist(), "n_classes_used": len(tprs)}


def confusion(y: np.ndarray, prob: np.ndarray) -> dict | None:
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= 0.5).astype(int) if prob.ndim == 1 else prob.argmax(axis=1)
    k = 2 if prob.ndim == 1 else prob.shape[1]
    if k > MAX_CONFUSION_CLASSES:
        return None
    m = np.zeros((k, k), dtype=int)
    np.add.at(m, (np.clip(y, 0, k - 1), np.clip(pred, 0, k - 1)), 1)
    return {"labels": list(range(k)), "matrix": m.tolist(), "rows": "true class", "cols": "predicted class"}


def curves(y, prob) -> dict:
    try:
        return {"roc": roc_points(y, prob), "confusion": confusion(y, prob)}
    except Exception:                                   # metrics must never break training
        return {}
