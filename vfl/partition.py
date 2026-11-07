"""
vfl_partition.py - split a dataset vertically into n parties for VFL experiments.

Output matches the layout the VFL system already uses:
    <out>/active.csv          id, <active features...>, label
    <out>/passive-1.csv       id, <passive-1 features...>
    ...
    <out>/passive-{n-1}.csv
    <out>/manifest.json       how the split was made + per-party statistics

Party 0 is the active (label-owning) party.

Feature-split strategies
    sequential    contiguous column blocks (simple, deterministic)
    random        features shuffled, then near-equal blocks
    dirichlet     VertiBench-style imbalance: party shares ~ Dirichlet(alpha);
                  small alpha -> very uneven feature counts and importance
    importance    rank by mutual information; put the top `top_frac` of
                  importance at `informative_party` (e.g. 0 = active, 1 = passive-1)
    correlated    cluster correlated features; each party gets whole clusters
                  (realistic: one institution holds related measurements)
    manual        explicit column lists per party (--columns-json)
    image_strips  split H x W images into vertical strips (n strips)
    image_grid    split images into a rows x cols grid of tiles (n = rows*cols)

Entity overlap
    overlap=1.0   every party holds every entity (default)
    overlap<1.0   only that fraction is held by all parties; each remaining entity
                  is dropped from a random non-empty subset of parties, so the
                  intersection really has to be computed by alignment.

Rows are shuffled independently per party, and ids can be opaque strings, so
nothing about the row order gives away the alignment.

Experiment-only note: strategies `importance` and `correlated` look at all
features (and labels, for `importance`) in one place to *design* the split.
That is benchmark set-up, not something a real deployment could do; the
manifest records it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STRATEGIES = ("sequential", "random", "dirichlet", "importance", "correlated",
              "manual", "image_strips", "image_grid")


# --------------------------------------------------------------------------
# Dataset container and loaders
# --------------------------------------------------------------------------
@dataclass
class Dataset:
    X: pd.DataFrame                      # features (tabular) or flattened pixels
    y: np.ndarray                        # labels (int)
    name: str
    image_shape: tuple | None = None     # (C, H, W) if X holds flattened images
    categorical: list = field(default_factory=list)


def load_dataset(name: str, csv: str | None = None, label: str | None = None,
                 n_samples: int = 20000, n_features: int = 40, seed: int = 0,
                 binarize: str | None = None) -> Dataset:
    """Built-in loaders. Offline: breast_cancer, digits, synthetic, csv.
    Online (needs openml.org access): adult, bank, credit_default, mnist, fashion_mnist."""
    from sklearn import datasets as skd

    if name == "breast_cancer":
        d = skd.load_breast_cancer(as_frame=True)
        return Dataset(d.data, d.target.to_numpy(), name)

    if name == "digits":                                   # 8x8 grayscale, offline
        d = skd.load_digits()
        X = pd.DataFrame(d.data, columns=[f"px_{i}" for i in range(64)])
        return _maybe_binarize(Dataset(X, d.target, name, image_shape=(1, 8, 8)), binarize)

    if name == "synthetic":
        Xa, y = skd.make_classification(
            n_samples=n_samples, n_features=n_features,
            n_informative=max(2, n_features // 4), n_redundant=max(1, n_features // 8),
            n_clusters_per_class=2, class_sep=1.0, flip_y=0.02, random_state=seed)
        X = pd.DataFrame(Xa, columns=[f"f{i:03d}" for i in range(n_features)])
        return Dataset(X, y, name)

    if name == "csv":
        if not csv or not label:
            raise ValueError("--dataset csv needs --csv PATH and --label COLUMN")
        df = pd.read_csv(csv)
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        y_raw = df.pop(label)
        y = pd.factorize(y_raw, sort=True)[0]
        cats = [c for c in df.columns if _is_cat(df[c])]
        return Dataset(df, y, os.path.basename(csv), categorical=cats)

    openml = {"adult": ("adult", 2), "bank": ("bank-marketing", 1),
              "credit_default": ("default-of-credit-card-clients", 1),
              "mnist": ("mnist_784", 1), "fashion_mnist": ("Fashion-MNIST", 1)}
    if name in openml:
        oname, ver = openml[name]
        d = skd.fetch_openml(oname, version=ver, as_frame=True, parser="auto")
        X, y_raw = d.data.copy(), d.target
        y = pd.factorize(y_raw, sort=True)[0]
        if name in ("mnist", "fashion_mnist"):
            X = X.astype(np.float32) / 255.0
            X.columns = [f"px_{i}" for i in range(X.shape[1])]
            return _maybe_binarize(Dataset(X, y, name, image_shape=(1, 28, 28)), binarize)
        cats = [c for c in X.columns if _is_cat(X[c])]
        return Dataset(X, y, name, categorical=cats)

    raise ValueError(f"unknown dataset {name!r}")


def _maybe_binarize(ds: Dataset, rule: str | None) -> Dataset:
    """Turn a multi-class image task into binary so binary AUROC/leakage attacks apply."""
    if rule is None:
        return ds
    if rule == "odd_even":
        ds.y = (ds.y % 2).astype(int)
    elif rule.startswith("one_vs_rest:"):
        k = int(rule.split(":")[1])
        ds.y = (ds.y == k).astype(int)
    elif rule.startswith("pair:"):                       # pair:3,8 keeps only classes 3 and 8
        a, b = (int(v) for v in rule.split(":")[1].split(","))
        keep = np.isin(ds.y, [a, b])
        ds.X, ds.y = ds.X.loc[keep].reset_index(drop=True), (ds.y[keep] == b).astype(int)
    else:
        raise ValueError(f"unknown binarize rule {rule!r}")
    ds.name += f"[{rule}]"
    return ds


# --------------------------------------------------------------------------
# Feature assignment
# --------------------------------------------------------------------------
def _is_cat(s: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categoricals so statistics can be computed (design-time only)."""
    return X.apply(lambda s: pd.Series(pd.factorize(s)[0], index=s.index) if _is_cat(s) else s)


def _blocks(cols: list, n: int) -> list[list]:
    return [list(b) for b in np.array_split(np.array(cols, dtype=object), n)]


def _fill_empty(groups: list[list], rng: np.random.Generator) -> list[list]:
    """Every party must own at least one feature: steal from the largest party."""
    for g in groups:
        while not g:
            donor = max(groups, key=len)
            if len(donor) < 2:
                raise ValueError("more parties than features")
            g.append(donor.pop(rng.integers(len(donor))))
    return groups


def _importance(ds: Dataset, cols: list, seed: int, max_rows: int = 5000) -> pd.Series:
    """Mutual information with the label (experiment-only: needs all features + labels)."""
    from sklearn.feature_selection import mutual_info_classif
    X = ds.X[cols]
    if len(X) > max_rows:
        idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], ds.y[idx]
    else:
        y = ds.y
    Xn = _numeric(X)
    Xn = Xn.fillna(Xn.median(numeric_only=True))
    discrete = [c in ds.categorical for c in cols]
    mi = mutual_info_classif(Xn.to_numpy(float), y, discrete_features=discrete, random_state=seed)
    return pd.Series(mi, index=cols)


def assign_features(ds: Dataset, n: int, strategy: str, seed: int = 0, alpha: float = 1.0,
                    top_frac: float = 0.5, informative_party: int = 0,
                    columns: list[list] | None = None, grid: tuple | None = None) -> list[list]:
    rng = np.random.default_rng(seed)
    cols = list(ds.X.columns)

    if strategy in ("image_strips", "image_grid"):
        if ds.image_shape is None:
            raise ValueError(f"{strategy} needs an image dataset")
        return _image_tiles(ds.image_shape, n, strategy, grid)

    if strategy == "sequential":
        return _fill_empty(_blocks(cols, n), rng)

    if strategy == "random":
        return _fill_empty(_blocks(list(rng.permutation(cols)), n), rng)

    if strategy == "dirichlet":
        share = rng.dirichlet([alpha] * n)
        owner = rng.choice(n, size=len(cols), p=share)
        return _fill_empty([[c for c, o in zip(cols, owner) if o == p] for p in range(n)], rng)

    if strategy == "importance":
        imp = _importance(ds, cols, seed).sort_values(ascending=False)
        ranked = list(imp.index)
        k = max(1, int(round(top_frac * len(ranked))))
        top, rest = ranked[:k], list(rng.permutation(ranked[k:]))
        groups = [[] for _ in range(n)]
        groups[informative_party] = list(top)
        others = [p for p in range(n) if p != informative_party]
        for p, block in zip(others, _blocks(rest, len(others))):
            groups[p] = block
        return _fill_empty(groups, rng)

    if strategy == "correlated":
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        num = _numeric(ds.X[cols])
        corr = num.corr().abs().fillna(0).to_numpy(copy=True)
        np.fill_diagonal(corr, 1.0)
        dist = squareform(1 - corr, checks=False).clip(min=0)
        labels = fcluster(linkage(dist, "average"), t=n, criterion="maxclust")
        clusters = [[c for c, l in zip(cols, labels) if l == k] for k in sorted(set(labels))]
        clusters.sort(key=len, reverse=True)
        groups = [[] for _ in range(n)]
        for cl in clusters:                      # greedy: biggest cluster to smallest party
            min(groups, key=len).extend(cl)
        return _fill_empty(groups, rng)

    if strategy == "manual":
        if not columns or len(columns) != n:
            raise ValueError(f"manual strategy needs {n} column lists")
        flat = [c for g in columns for c in g]
        if len(flat) != len(set(flat)):
            raise ValueError("a column is assigned to more than one party")
        missing = set(flat) - set(cols)
        if missing:
            raise ValueError(f"unknown columns: {sorted(missing)}")
        return [list(g) for g in columns]

    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")


def _image_tiles(shape: tuple, n: int, strategy: str, grid: tuple | None) -> list[list]:
    c, h, w = shape
    if strategy == "image_strips":
        rows, cols = 1, n
    else:
        rows, cols = grid or _near_square(n)
        if rows * cols != n:
            raise ValueError(f"grid {rows}x{cols} does not give {n} parties")
    hb = np.array_split(np.arange(h), rows)
    wb = np.array_split(np.arange(w), cols)
    groups = []
    for r in range(rows):
        for q in range(cols):
            idx = [ch * h * w + i * w + j for ch in range(c) for i in hb[r] for j in wb[q]]
            groups.append([f"px_{k}" for k in idx])
    return groups


def _near_square(n: int) -> tuple:
    r = int(math.sqrt(n))
    while n % r:
        r -= 1
    return r, n // r


# --------------------------------------------------------------------------
# Entity overlap and ids
# --------------------------------------------------------------------------
def make_ids(n_rows: int, style: str, seed: int) -> np.ndarray:
    if style == "int":
        return np.arange(n_rows)
    salt = f"vfl-{seed}"                                   # opaque, stable per seed
    return np.array([hashlib.sha1(f"{salt}:{i}".encode()).hexdigest()[:12] for i in range(n_rows)])


def overlap_masks(n_rows: int, n: int, overlap: float, seed: int) -> np.ndarray:
    """Boolean [n_parties, n_rows]: which party holds which entity."""
    if not 0 < overlap <= 1:
        raise ValueError("overlap must be in (0, 1]")
    rng = np.random.default_rng(seed + 1)
    held = np.ones((n, n_rows), dtype=bool)
    if overlap >= 1:
        return held
    shared = rng.random(n_rows) < overlap
    for r in np.flatnonzero(~shared):
        drop = rng.random(n) < 0.5                         # random non-empty subset dropped
        if not drop.any():
            drop[rng.integers(n)] = True
        held[drop, r] = False
    return held


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def partition(ds: Dataset, n_parties: int, out_dir: str, strategy: str = "random",
              seed: int = 0, overlap: float = 1.0, id_style: str = "hash",
              alpha: float = 1.0, top_frac: float = 0.5, informative_party: int = 0,
              columns: list[list] | None = None, grid: tuple | None = None,
              one_hot: bool = True, label_col: str = "label",
              evaluate: bool = False) -> dict:
    """Split `ds` into `n_parties` CSVs in `out_dir`. Returns the manifest dict."""
    if n_parties < 2:
        raise ValueError("need at least 2 parties (1 active + >=1 passive)")
    groups = assign_features(ds, n_parties, strategy, seed, alpha, top_frac,
                             informative_party, columns, grid)

    ids = make_ids(len(ds.X), id_style, seed)
    held = overlap_masks(len(ds.X), n_parties, overlap, seed)
    names = ["active"] + [f"passive-{i}" for i in range(1, n_parties)]
    os.makedirs(out_dir, exist_ok=True)

    imp = None
    if evaluate or strategy == "importance":
        tab_cols = [c for g in groups for c in g]
        imp = _importance(ds, tab_cols, seed) if ds.image_shape is None or len(tab_cols) <= 4096 else None

    parties = []
    for p, (name, cols) in enumerate(zip(names, groups)):
        rows = np.flatnonzero(held[p])
        df = ds.X.iloc[rows][cols].copy()
        cats = [c for c in cols if c in ds.categorical]
        if one_hot and cats:                               # encode AFTER the split: a categorical
            df = pd.get_dummies(df, columns=cats, dtype=np.float32)   # stays with one party
        df.insert(0, "id", ids[rows])
        if p == 0:
            df[label_col] = ds.y[rows]
        df = df.sample(frac=1.0, random_state=seed + 100 + p).reset_index(drop=True)
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)

        info = {"name": name, "role": "active" if p == 0 else "passive", "file": os.path.basename(path),
                "rows": int(len(df)), "source_columns": len(cols),
                "model_input_dim": int(df.shape[1] - 1 - (p == 0)),
                "categorical_columns": cats, "columns": cols}
        if ds.image_shape is not None and strategy in ("image_strips", "image_grid"):
            info["shape"] = _tile_shape(ds.image_shape, cols)
        if imp is not None:
            info["importance_share"] = round(float(imp[cols].sum() / max(imp.sum(), 1e-12)), 4)
        parties.append(info)

    inter = int(held.all(axis=0).sum())
    manifest = {
        "dataset": ds.name, "n_parties": n_parties, "strategy": strategy, "seed": seed,
        "params": {"alpha": alpha, "top_frac": top_frac, "informative_party": informative_party,
                   "grid": grid, "one_hot": one_hot, "id_style": id_style},
        "entities_total": int(len(ds.X)), "overlap_requested": overlap,
        "intersection_size": inter, "intersection_ratio": round(inter / len(ds.X), 4),
        "label_balance": round(float(np.mean(ds.y)), 4) if len(np.unique(ds.y)) == 2 else None,
        "num_classes": int(len(np.unique(ds.y))),
        "image_shape": list(ds.image_shape) if ds.image_shape else None,
        "experiment_only": strategy in ("importance", "correlated") or evaluate,
        "parties": parties,
    }
    if evaluate:
        manifest["reference_auroc"] = evaluate_split(ds, groups, seed)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _tile_shape(image_shape: tuple, cols: list) -> list:
    c, h, w = image_shape
    idx = np.array([int(x.split("_")[1]) for x in cols])
    rows = np.unique((idx % (h * w)) // w)
    colsx = np.unique(idx % w)
    return [c, len(rows), len(colsx)]


def evaluate_split(ds: Dataset, groups: list[list], seed: int) -> dict:
    """Quick logistic-regression AUROC per party alone vs all features.
    Tells you how much each party *could* contribute before running real VFL.
    Experiment-only: reads all parties' features and labels in one place."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(ds.y)) != 2:
        return {"note": "binary labels only"}
    X = _numeric(ds.X)
    X = X.fillna(X.median(numeric_only=True))
    tr, te = train_test_split(np.arange(len(X)), test_size=0.3, random_state=seed, stratify=ds.y)

    def auc(cols):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(X.iloc[tr][cols], ds.y[tr])
        return round(float(roc_auc_score(ds.y[te], m.predict_proba(X.iloc[te][cols])[:, 1])), 4)

    names = ["active"] + [f"passive-{i}" for i in range(1, len(groups))]
    res = {n: auc(g) for n, g in zip(names, groups)}
    res["all_features"] = auc([c for g in groups for c in g])
    return res


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="breast_cancer | digits | synthetic | csv | adult | bank | credit_default | mnist | fashion_mnist")
    ap.add_argument("--csv"); ap.add_argument("--label")
    ap.add_argument("--parties", type=int, required=True, help="total parties incl. the active one")
    ap.add_argument("--strategy", default="random", choices=STRATEGIES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlap", type=float, default=1.0)
    ap.add_argument("--id-style", default="hash", choices=["hash", "int"])
    ap.add_argument("--alpha", type=float, default=1.0, help="dirichlet concentration (small = uneven)")
    ap.add_argument("--top-frac", type=float, default=0.5, help="importance: fraction of top features")
    ap.add_argument("--informative-party", type=int, default=0, help="importance: 0=active, 1=passive-1, ...")
    ap.add_argument("--columns-json", help="manual: JSON list of column lists, one per party")
    ap.add_argument("--grid", help="image_grid: ROWSxCOLS, e.g. 2x2")
    ap.add_argument("--binarize", help="odd_even | one_vs_rest:K | pair:A,B (image datasets)")
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--n-features", type=int, default=40)
    ap.add_argument("--no-one-hot", action="store_true")
    ap.add_argument("--evaluate", action="store_true", help="add per-party reference AUROC to the manifest")
    a = ap.parse_args(argv)

    ds = load_dataset(a.dataset, a.csv, a.label, a.n_samples, a.n_features, a.seed, a.binarize)
    columns = json.load(open(a.columns_json)) if a.columns_json else None
    grid = tuple(int(v) for v in a.grid.lower().split("x")) if a.grid else None
    m = partition(ds, a.parties, a.out, a.strategy, a.seed, a.overlap, a.id_style, a.alpha,
                  a.top_frac, a.informative_party, columns, grid, not a.no_one_hot, evaluate=a.evaluate)

    print(f"{m['dataset']}: {m['entities_total']} entities -> {m['n_parties']} parties "
          f"({m['strategy']}), intersection {m['intersection_size']} ({m['intersection_ratio']:.0%})")
    for p in m["parties"]:
        extra = f"  importance {p['importance_share']:.0%}" if "importance_share" in p else ""
        shape = f"  shape {p['shape']}" if "shape" in p else ""
        print(f"  {p['name']:<11} rows {p['rows']:>6}  inputs {p['model_input_dim']:>4}{shape}{extra}")
    if "reference_auroc" in m:
        print("  reference AUROC (logistic regression):", m["reference_auroc"])
    print(f"wrote {a.out}/manifest.json")


if __name__ == "__main__":
    main()
