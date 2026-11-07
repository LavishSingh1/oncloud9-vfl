"""Extra datasets and modalities for the v2 partitioner (v2 items 2, 7, 8).

  npz images     large image sets (e.g. CIFAR-10 saved as .npz) split into strips/tiles and written
                 as per-party .npz files (no giant CSVs)            --dataset npz --npz FILE
  sensor streams multichannel time series, one sensor stream (group of channels) per party/device
                 synthetic_sensors (offline) | har (UCI HAR, --har-dir "UCI HAR Dataset")
  multimodal     multimodal_digits: active = left image half (cnn), passive-1 = tabular ink statistics
                 of the right half (mlp), passive-2 = generated text describing the right half, turned
                 into features by a FROZEN text encoder at that party (text)
  embed-text     run a frozen sentence encoder over a party's own text column, offline, at the party.
                 The encoder is a feature extractor; only the projection layer after it is trained by
                 VFL (this is NOT split learning through the language model).

All writers produce the same per-party manifest format as vfl/partition.py.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import partition as P
from .data import (ID_COL, LABEL_COL, SENTINEL_ID_PREFIX, SENTINEL_VALUE, write_party_manifests)

IMAGE_STRATEGIES = ("image_strips", "image_grid", "sequential", "random")


# ------------------------------------------------------------------ helpers
def _binarize_arrays(X, y, rule):
    if rule is None:
        return X, y
    if rule == "odd_even":
        return X, (y % 2).astype(int)
    if rule.startswith("one_vs_rest:"):
        return X, (y == int(rule.split(":")[1])).astype(int)
    if rule.startswith("pair:"):
        a, b = (int(v) for v in rule.split(":")[1].split(","))
        keep = np.isin(y, [a, b])
        return X[keep], (y[keep] == b).astype(int)
    raise ValueError(f"unknown binarize rule {rule!r}")


def _write_party(out_dir, name, ids, X, cols, y, fmt, sentinel):
    if sentinel:
        k = 5
        ids = np.concatenate([ids.astype(object), np.array([f"{SENTINEL_ID_PREFIX}{name}-{i:04d}" for i in range(k)],
                                                            dtype=object)])
        X = np.concatenate([X, np.full((k, X.shape[1]), SENTINEL_VALUE, X.dtype)])
        if y is not None:
            y = np.concatenate([y, np.ones(k, y.dtype)])
    if fmt == "npz":
        path = os.path.join(out_dir, f"{name}.npz")
        arrays = {"ids": np.asarray(ids).astype(str), "X": X.astype(np.float32), "columns": np.asarray(cols)}
        if y is not None:
            arrays["y"] = y.astype(np.int64)
        np.savez_compressed(path, **arrays)
    else:
        path = os.path.join(out_dir, f"{name}.csv")
        df = pd.DataFrame(X, columns=cols)
        df.insert(0, ID_COL, ids)
        if y is not None:
            df[LABEL_COL] = y
        df.to_csv(path, index=False)
    return path, len(ids)


def _finish(out_dir, manifest, party_infos, common, extra_report=None):
    full = os.path.join(out_dir, "manifest.json")
    manifest["parties"] = party_infos
    with open(full, "w") as fh:
        json.dump(manifest, fh, indent=2)
    write_party_manifests(out_dir, party_infos, common, full)
    report = {"dataset": manifest["dataset"], "seed": manifest["seed"], "n_total": manifest["entities_total"],
              "overlap_target": manifest["overlap_requested"], "partition": manifest["strategy"], "partitioner": "v2",
              "num_classes": manifest["num_classes"], "n_aligned_expected": manifest["intersection_size"],
              "overlap_ratio_actual": manifest["intersection_ratio"],
              "parties": {p["name"]: {"rows": p["rows"], "n_features": p["model_input_dim"],
                                      "has_labels": p["role"] == "active",
                                      "file": os.path.join(out_dir, p["file"])} for p in party_infos}}
    report.update(extra_report or {})
    with open(os.path.join(out_dir, "data_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return manifest


def _split_rows(n_rows, n_parties, overlap, seed, id_style):
    return P.make_ids(n_rows, id_style, seed), P.overlap_masks(n_rows, n_parties, overlap, seed)


# ------------------------------------------------------------------ images (npz) - item 2
def load_npz_images(path: str):
    """Accepts keys x/X/images/data and y/labels/targets; layouts (N,H,W), (N,H,W,C) or (N,C,H,W)."""
    with np.load(path, allow_pickle=False) as z:
        keys = {k.lower(): k for k in z.files}
        xk = next((keys[k] for k in ("x", "images", "data", "x_train") if k in keys), None)
        yk = next((keys[k] for k in ("y", "labels", "targets", "y_train") if k in keys), None)
        if xk is None or yk is None:
            raise ValueError(f"{path}: need image and label arrays (keys like x/y), found {z.files}")
        X, y = z[xk], np.asarray(z[yk]).reshape(-1)
    if X.ndim == 3:
        X = X[:, None]
    elif X.ndim == 4 and X.shape[-1] in (1, 3, 4) and X.shape[1] not in (1, 3, 4):
        X = X.transpose(0, 3, 1, 2)
    elif X.ndim != 4:
        raise ValueError(f"{path}: images must be (N,H,W), (N,H,W,C) or (N,C,H,W), got {X.shape}")
    X = X.astype(np.float32)
    if X.max() > 1.5:
        X /= 255.0
    return X, pd.factorize(y, sort=True)[0]


def prepare_images(out_dir, X, y, name, n_parties, strategy="image_strips", seed=0, overlap=1.0, binarize=None,
                   id_style="hash", grid=None, fmt="npz", sentinel=False):
    """X: (N, C, H, W) float32. Writes per-party image tiles/strips."""
    if strategy not in IMAGE_STRATEGIES:
        raise ValueError(f"image arrays support {IMAGE_STRATEGIES}")
    X, y = _binarize_arrays(X, y, binarize)
    N, C, H, W = X.shape
    flat = X.reshape(N, -1)
    if strategy in ("image_strips", "image_grid"):
        groups = P._image_tiles((C, H, W), n_parties, strategy, grid)
    else:
        cols = [f"px_{i}" for i in range(C * H * W)]
        if strategy == "random":
            cols = list(np.random.default_rng(seed).permutation(cols))
        groups = [list(b) for b in np.array_split(np.array(cols, dtype=object), n_parties)]
    ids, held = _split_rows(N, n_parties, overlap, seed, id_style)
    os.makedirs(out_dir, exist_ok=True)
    names = ["active"] + [f"passive-{i}" for i in range(1, n_parties)]
    infos = []
    for p, (pname, cols) in enumerate(zip(names, groups)):
        rows = np.flatnonzero(held[p])
        rows = rows[np.random.default_rng(seed + 100 + p).permutation(len(rows))]
        idx = [int(c.split("_")[1]) for c in cols]
        path, n = _write_party(out_dir, pname, ids[rows], flat[rows][:, idx], cols,
                               y[rows] if p == 0 else None, fmt, sentinel)
        info = {"name": pname, "role": "active" if p == 0 else "passive", "file": os.path.basename(path),
                "rows": n, "source_columns": len(cols), "model_input_dim": len(cols),
                "categorical_columns": [], "columns": cols}
        if strategy in ("image_strips", "image_grid"):
            info["shape"] = P._tile_shape((C, H, W), cols)
        infos.append(info)
    inter = int(held.all(axis=0).sum())
    k = int(len(np.unique(y)))
    name_full = name + (f"[{binarize}]" if binarize else "")
    m = {"dataset": name_full, "n_parties": n_parties, "strategy": strategy, "seed": seed,
         "params": {"grid": grid, "id_style": id_style, "format": fmt}, "entities_total": int(N),
         "overlap_requested": overlap, "intersection_size": inter, "intersection_ratio": round(inter / N, 4),
         "label_balance": round(float(np.mean(y)), 4) if k == 2 else None, "num_classes": k,
         "image_shape": [C, H, W], "experiment_only": False}
    return _finish(out_dir, m, infos, {"partitioner": "v2", "dataset": name_full, "strategy": strategy,
                                       "overlap": overlap, "seed": seed, "binarize": binarize, "num_classes": k})


# ------------------------------------------------------------------ sensor streams - item 7
HAR_SIGNALS = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z",
               "total_acc_x", "total_acc_y", "total_acc_z"]
HAR_STREAMS = {"body_acc": [0, 1, 2], "body_gyro": [3, 4, 5], "total_acc": [6, 7, 8]}


def synthetic_sensors(n=3000, channels=6, T=64, classes=4, seed=0):
    """Offline stand-in for wearable data: each activity class has its own per-channel frequency,
    amplitude and phase pattern, plus noise and random time shifts."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, n)
    base_f = rng.uniform(0.5, 4.0, size=(classes, channels))
    amp = rng.uniform(0.3, 1.5, size=(classes, channels))
    t = np.arange(T) / T
    X = np.empty((n, channels, T), np.float32)
    for i in range(n):
        c = y[i]
        shift = rng.uniform(0, 2 * np.pi)
        f = base_f[c] * rng.uniform(0.9, 1.1, size=channels)
        X[i] = amp[c][:, None] * np.sin(2 * np.pi * f[:, None] * t[None, :] * 4 + shift) + rng.normal(0, 0.6, (channels, T))
    names = [f"{s}_{a}" for s in ("acc", "gyro", "mag", "baro") for a in "xyz"][:channels]
    streams = {}
    for ci, nm in enumerate(names):
        streams.setdefault(nm.split("_")[0], []).append(ci)
    return X, y, names, streams


def load_har(har_dir: str):
    """UCI HAR (Anguita et al. 2013): 9 inertial channels x 128 steps, 6 activities (train+test)."""
    Xs, ys = [], []
    for part in ("train", "test"):
        sig = [np.loadtxt(os.path.join(har_dir, part, "Inertial Signals", f"{s}_{part}.txt")) for s in HAR_SIGNALS]
        Xs.append(np.stack(sig, axis=1).astype(np.float32))
        ys.append(np.loadtxt(os.path.join(har_dir, part, f"y_{part}.txt")).astype(int) - 1)
    return np.concatenate(Xs), np.concatenate(ys), list(HAR_SIGNALS), dict(HAR_STREAMS)


def prepare_sensors(out_dir, X, y, channel_names, streams, name, n_parties, seed=0, overlap=1.0, id_style="hash",
                    fmt="csv", sentinel=False):
    """One sensor stream per party when n_parties == number of streams, otherwise channels are dealt
    out in contiguous blocks. Party columns are channel-major: <channel>_t000 ... (shape [C_p, T])."""
    N, C, T = X.shape
    stream_list = list(streams.values())
    if n_parties == len(stream_list):
        groups = stream_list
    else:
        groups = [list(map(int, g)) for g in np.array_split(np.arange(C), n_parties)]
        if any(len(g) == 0 for g in groups):
            raise ValueError(f"{n_parties} parties but only {C} channels")
    ids, held = _split_rows(N, n_parties, overlap, seed, id_style)
    os.makedirs(out_dir, exist_ok=True)
    names = ["active"] + [f"passive-{i}" for i in range(1, n_parties)]
    infos = []
    for p, (pname, chans) in enumerate(zip(names, groups)):
        rows = np.flatnonzero(held[p])
        rows = rows[np.random.default_rng(seed + 100 + p).permutation(len(rows))]
        cols = [f"{channel_names[c]}_t{t:03d}" for c in chans for t in range(T)]
        Xp = X[rows][:, chans, :].reshape(len(rows), -1)
        path, n = _write_party(out_dir, pname, ids[rows], Xp, cols, y[rows] if p == 0 else None, fmt, sentinel)
        infos.append({"name": pname, "role": "active" if p == 0 else "passive", "file": os.path.basename(path),
                      "rows": n, "source_columns": len(cols), "model_input_dim": len(cols),
                      "categorical_columns": [], "columns": cols, "shape": [len(chans), T],
                      "streams": [channel_names[c] for c in chans]})
    inter = int(held.all(axis=0).sum())
    k = int(len(np.unique(y)))
    m = {"dataset": name, "n_parties": n_parties, "strategy": "sensor_streams", "seed": seed,
         "params": {"id_style": id_style, "format": fmt}, "entities_total": int(N), "overlap_requested": overlap,
         "intersection_size": inter, "intersection_ratio": round(inter / N, 4),
         "label_balance": round(float(np.mean(y)), 4) if k == 2 else None, "num_classes": k,
         "image_shape": None, "series_shape": [C, T], "experiment_only": False}
    return _finish(out_dir, m, infos, {"partitioner": "v2", "dataset": name, "strategy": "sensor_streams",
                                       "overlap": overlap, "seed": seed, "binarize": None, "num_classes": k})


# ------------------------------------------------------------------ text encoders - item 8
TEXT_ENCODERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "distilbert": "distilbert-base-uncased",
    "hashing": "hashing-vectorizer + truncated SVD (offline fallback)",
}


def encode_texts(texts: list[str], encoder: str = "minilm", dim: int = 64, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Frozen encoder -> [n, d] float32. Nothing here is trained by VFL."""
    texts = [str(t) for t in texts]
    if encoder == "minilm":
        from sentence_transformers import SentenceTransformer       # optional dependency
        model = SentenceTransformer(TEXT_ENCODERS["minilm"], device="cpu")
        E = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
        return E.astype(np.float32), {"encoder": TEXT_ENCODERS["minilm"], "dim": int(E.shape[1]), "frozen": True}
    if encoder == "distilbert":
        import torch
        from transformers import AutoModel, AutoTokenizer                # optional dependency
        tok = AutoTokenizer.from_pretrained(TEXT_ENCODERS["distilbert"])
        mdl = AutoModel.from_pretrained(TEXT_ENCODERS["distilbert"]).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                b = tok(texts[i:i + 32], padding=True, truncation=True, max_length=128, return_tensors="pt")
                h = mdl(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                out.append(((h * m).sum(1) / m.sum(1)).numpy())
        E = np.concatenate(out).astype(np.float32)
        return E, {"encoder": TEXT_ENCODERS["distilbert"] + " (mean pooled)", "dim": int(E.shape[1]), "frozen": True}
    if encoder == "hashing":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
        H = HashingVectorizer(n_features=2 ** 12, alternate_sign=False, ngram_range=(1, 2)).transform(texts)
        H = TfidfTransformer().fit_transform(H)
        d = max(2, min(dim, H.shape[1] - 1, H.shape[0] - 1))
        E = TruncatedSVD(d, random_state=seed).fit_transform(H).astype(np.float32)
        return E, {"encoder": TEXT_ENCODERS["hashing"], "dim": d, "frozen": True,
                   "note": "unsupervised, fitted on this party's own text only"}
    raise ValueError(f"unknown text encoder {encoder!r}; choose from {list(TEXT_ENCODERS)}")


def embed_text_file(csv_in: str, text_col: str, out_csv: str, encoder: str = "minilm", dim: int = 64,
                    manifest_in: str | None = None) -> dict:
    """Party-side: replace a text column by frozen-encoder embeddings (+ update its party manifest)."""
    df = pd.read_csv(csv_in, dtype={ID_COL: str})
    if text_col not in df.columns:
        raise ValueError(f"{csv_in}: no column {text_col!r}")
    E, info = encode_texts(df[text_col].fillna("").tolist(), encoder, dim)
    keep = [c for c in df.columns if c not in (text_col,)]
    out = df[keep].copy()
    emb_cols = [f"emb_{i:03d}" for i in range(E.shape[1])]
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(E, columns=emb_cols)], axis=1)
    if LABEL_COL in out.columns:                      # keep label last for the active party
        out = out[[c for c in out.columns if c != LABEL_COL] + [LABEL_COL]]
    out.to_csv(out_csv, index=False)
    feats = [c for c in out.columns if c not in (ID_COL, LABEL_COL)]
    mpath = manifest_in or os.path.splitext(csv_in)[0] + ".manifest.json"
    if os.path.exists(mpath):
        m = json.load(open(mpath))
        m.update({"file": os.path.basename(out_csv), "rows": int(len(out)), "model_input_dim": len(feats),
                  "columns": feats, "text_encoder": info})
        with open(os.path.splitext(out_csv)[0] + ".manifest.json", "w") as fh:
            json.dump(m, fh, indent=2)
    return info


# ------------------------------------------------------------------ multimodal demo - item 8
_BANDS = ["top", "upper middle", "lower middle", "bottom"]


def _describe(half: np.ndarray, rng) -> str:
    """Noisy natural-language description of an 8x4 image half (ink per horizontal band)."""
    words = []
    for b in range(4):
        ink = half[2 * b:2 * b + 2].mean()
        level = "heavy" if ink > 0.45 else "some" if ink > 0.2 else "faint" if ink > 0.05 else "no"
        if rng.random() < 0.15:          # descriptions are imperfect
            level = rng.choice(["heavy", "some", "faint", "no"])
        words.append(f"{level} ink at the {_BANDS[b]}")
    lean = "leans right" if half[:, 2:].mean() > half[:, :2].mean() else "stays near the centre"
    fill = rng.choice(["the stroke", "the right side", "this half", "the scan"])
    return f"{fill}: " + ", ".join(words) + f"; it {lean}."


def prepare_multimodal_digits(out_dir, seed=0, overlap=1.0, binarize=None, id_style="hash",
                              text_encoder="hashing", sentinel=False):
    from sklearn.datasets import load_digits
    d = load_digits()
    imgs = (d.images / 16.0).astype(np.float32)                      # (N, 8, 8)
    y = d.target.astype(int)
    imgs, y = _binarize_arrays(imgs, y, binarize)
    N = len(y)
    rng = np.random.default_rng(seed)
    left, right = imgs[:, :, :4], imgs[:, :, 4:]
    tab = np.concatenate([right.sum(axis=2), right.sum(axis=1), right.reshape(N, -1).mean(1, keepdims=True),
                          np.abs(right - right[:, ::-1]).reshape(N, -1).mean(1, keepdims=True)], axis=1)
    tab_cols = [f"row_ink_{i}" for i in range(8)] + [f"col_ink_{i}" for i in range(4)] + ["mean_ink", "asymmetry"]
    texts = [_describe(right[i], rng) for i in range(N)]
    ids, held = _split_rows(N, 3, overlap, seed, id_style)
    os.makedirs(out_dir, exist_ok=True)
    infos = []
    # active: left half pixels (image, cnn)
    rows = np.flatnonzero(held[0])
    cols = [f"px_{r * 8 + c}" for r in range(8) for c in range(4)]
    path, n = _write_party(out_dir, "active", ids[rows], left[rows].reshape(len(rows), -1), cols, y[rows], "csv", sentinel)
    infos.append({"name": "active", "role": "active", "file": os.path.basename(path), "rows": n,
                  "source_columns": 32, "model_input_dim": 32, "categorical_columns": [], "columns": cols,
                  "shape": [1, 8, 4], "modality": "image"})
    # passive-1: tabular statistics of the right half (mlp)
    rows = np.flatnonzero(held[1])
    path, n = _write_party(out_dir, "passive-1", ids[rows], tab[rows], tab_cols, None, "csv", sentinel)
    infos.append({"name": "passive-1", "role": "passive", "file": os.path.basename(path), "rows": n,
                  "source_columns": len(tab_cols), "model_input_dim": len(tab_cols), "categorical_columns": [],
                  "columns": tab_cols, "modality": "tabular"})
    # passive-2: raw text (what the party holds) -> frozen encoder at the party -> embeddings (text)
    rows = np.flatnonzero(held[2])
    raw = pd.DataFrame({ID_COL: ids[rows], "text": [texts[i] for i in rows]})
    raw.to_csv(os.path.join(out_dir, "passive-2.text.csv"), index=False)
    E, enc = encode_texts(raw["text"].tolist(), text_encoder, 32, seed)
    emb_cols = [f"emb_{i:03d}" for i in range(E.shape[1])]
    path, n = _write_party(out_dir, "passive-2", ids[rows], E, emb_cols, None, "csv", sentinel)
    infos.append({"name": "passive-2", "role": "passive", "file": os.path.basename(path), "rows": n,
                  "source_columns": 1, "model_input_dim": E.shape[1], "categorical_columns": [], "columns": emb_cols,
                  "modality": "text", "text_encoder": enc})
    inter = int(held.all(axis=0).sum())
    k = int(len(np.unique(y)))
    name = "multimodal_digits" + (f"[{binarize}]" if binarize else "")
    m = {"dataset": name, "n_parties": 3, "strategy": "multimodal", "seed": seed,
         "params": {"id_style": id_style, "text_encoder": enc}, "entities_total": int(N),
         "overlap_requested": overlap, "intersection_size": inter, "intersection_ratio": round(inter / N, 4),
         "label_balance": round(float(np.mean(y)), 4) if k == 2 else None, "num_classes": k,
         "image_shape": None, "experiment_only": False,
         "recommended_bottom_model": {"default": "mlp", "active": "cnn", "passive-2": "text"}}
    out = _finish(out_dir, m, infos, {"partitioner": "v2", "dataset": name, "strategy": "multimodal",
                                      "overlap": overlap, "seed": seed, "binarize": binarize, "num_classes": k})
    pm = os.path.join(out_dir, "passive-2.manifest.json")
    j = json.load(open(pm))
    j["text_encoder"] = enc
    json.dump(j, open(pm, "w"), indent=2)
    return out
