"""Datasets, vertical partitioning, per-party loading, splits and batch schedules.

`prepare_vertical_dataset` writes one CSV per party (as each institution would
hold it), with shuffled rows and partial entity overlap so alignment is real.
The active party's CSV carries the label column; passive CSVs never do.

Splits and batch schedules are derived only from (seed, aligned order), so every
party computes identical batch membership without exchanging indices (FR-3).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ID_COL = "id"
LABEL_COL = "label"
SENTINEL_VALUE = 987654.321
SENTINEL_ID_PREFIX = "SENTINEL-UNMATCHED-"


# ---------------------------------------------------------------- sources
def load_source(name: str, n: int = 20000, n_features: int = 40, seed: int = 0,
                csv_path: str | None = None, label: str | None = None):
    """Return (X: DataFrame, y: ndarray[int]) for a public/synthetic tabular task."""
    if name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(as_frame=True)
        X = d.data.copy()
        X.columns = [c.replace(" ", "_") for c in X.columns]
        return X, d.target.to_numpy().astype(int)
    if name == "synthetic":
        from sklearn.datasets import make_classification
        Xa, y = make_classification(
            n_samples=n, n_features=n_features, n_informative=max(2, n_features // 3),
            n_redundant=max(0, n_features // 6), n_clusters_per_class=2,
            weights=[0.7, 0.3], flip_y=0.02, class_sep=1.0, random_state=seed)
        X = pd.DataFrame(Xa, columns=[f"f{i:03d}" for i in range(n_features)])
        return X, y.astype(int)
    if name == "csv":
        if not csv_path or not label:
            raise ValueError("--csv and --label are required for dataset=csv")
        df = pd.read_csv(csv_path)
        y_raw = df.pop(label)
        classes = sorted(y_raw.unique())
        if len(classes) != 2:
            raise ValueError(f"label column must be binary, found {classes[:5]}")
        y = (y_raw == classes[1]).astype(int).to_numpy()
        X = pd.get_dummies(df, dummy_na=False).astype(float).fillna(0.0)
        return X, y
    raise ValueError(f"unknown dataset {name!r}")


def partition_features(X: pd.DataFrame, y: np.ndarray, n_parties: int, method: str,
                       seed: int, ratios: list[float] | None = None) -> list[list[str]]:
    """Split columns among parties. Index 0 is the active party.

    random | sequential : plain splits by `ratios`
    importance_balanced : columns sorted by mutual information, dealt round-robin
    importance_skewed   : most informative columns go to the passive parties
    """
    cols = list(X.columns)
    ratios = ratios or [1.0 / n_parties] * n_parties
    if len(ratios) != n_parties:
        raise ValueError("ratios must have one entry per party")
    rng = np.random.default_rng(seed)
    if method in ("importance_balanced", "importance_skewed"):
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(X.to_numpy(), y, random_state=seed)
        order = [cols[i] for i in np.argsort(-mi)]
        if method == "importance_balanced":
            parts = [[] for _ in range(n_parties)]
            for i, c in enumerate(order):
                parts[i % n_parties].append(c)
            return parts
        cols = order[::-1]  # least informative first => active gets the weakest
    elif method == "random":
        cols = list(rng.permutation(cols))
    elif method != "sequential":
        raise ValueError(f"unknown partition method {method!r}")
    counts = np.floor(np.array(ratios) / sum(ratios) * len(cols)).astype(int)
    counts[-1] = len(cols) - counts[:-1].sum()
    if (counts <= 0).any():
        raise ValueError("every party needs at least one feature")
    out, start = [], 0
    for c in counts:
        out.append(list(cols[start:start + c]))
        start += c
    return out


def prepare_vertical_dataset(out_dir: str, dataset: str = "breast_cancer", n_passive: int = 1,
                             overlap: float = 0.9, partition: str = "random", seed: int = 0,
                             ratios: list[float] | None = None, n: int = 20000,
                             n_features: int = 40, csv_path: str | None = None,
                             label: str | None = None, sentinel: bool = False) -> dict:
    X, y = load_source(dataset, n=n, n_features=n_features, seed=seed, csv_path=csv_path, label=label)
    n_parties = n_passive + 1
    parts = partition_features(X, y, n_parties, partition, seed, ratios)
    N = len(X)
    ids = np.array([f"E{i:07d}" for i in range(N)])
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(N)
    n_common = int(round(overlap * N))
    common = set(perm[:n_common].tolist())
    rest = perm[n_common:]
    names = ["active"] + [f"passive-{i + 1}" for i in range(n_passive)]
    members = {nm: set(common) for nm in names}
    # Non-overlapping rows: each is held by a random strict subset of parties.
    for r in rest:
        k = rng.integers(1, n_parties) if n_parties > 1 else 1
        for nm in rng.choice(names, size=k, replace=False):
            members[nm].add(int(r))
    os.makedirs(out_dir, exist_ok=True)
    report = {"dataset": dataset, "seed": seed, "n_total": N, "overlap_target": overlap,
              "partition": partition, "parties": {}}
    try:
        from sklearn.feature_selection import mutual_info_classif
        mi = dict(zip(X.columns, mutual_info_classif(X.to_numpy(), y, random_state=seed)))
    except Exception:  # pragma: no cover
        mi = {}
    for p_idx, nm in enumerate(names):
        rows = np.array(sorted(members[nm]))
        df = X.iloc[rows][parts[p_idx]].copy()
        df.insert(0, ID_COL, ids[rows])
        if p_idx == 0:
            df[LABEL_COL] = y[rows]
        if sentinel:
            # Rows nobody else holds get sentinel ids/values: they must never appear in logs.
            k = 5
            sent = pd.DataFrame({c: [SENTINEL_VALUE] * k for c in parts[p_idx]})
            sent.insert(0, ID_COL, [f"{SENTINEL_ID_PREFIX}{nm}-{i:04d}" for i in range(k)])
            if p_idx == 0:
                sent[LABEL_COL] = 1
            df = pd.concat([df, sent], ignore_index=True)
        df = df.sample(frac=1.0, random_state=seed + 7 + p_idx).reset_index(drop=True)
        path = os.path.join(out_dir, f"{nm}.csv")
        df.to_csv(path, index=False)
        report["parties"][nm] = {
            "rows": int(len(df)), "n_features": len(parts[p_idx]), "features": parts[p_idx],
            "mutual_info_sum": float(sum(mi.get(c, 0.0) for c in parts[p_idx])),
            "has_labels": p_idx == 0, "file": path,
        }
    all_sets = [members[nm] for nm in names]
    inter = set.intersection(*all_sets)
    report["n_aligned_expected"] = len(inter)
    report["overlap_ratio_actual"] = len(inter) / N
    if len(inter):
        report["positive_rate_aligned"] = float(y[sorted(inter)].mean())
    with open(os.path.join(out_dir, "data_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    write_party_manifests(out_dir, [
        {"name": nm, "role": "active" if p["has_labels"] else "passive", "file": os.path.basename(p["file"]),
         "rows": p["rows"], "model_input_dim": p["n_features"], "columns": p["features"],
         "categorical_columns": []}
        for nm, p in report["parties"].items()],
        {"partitioner": "v1", "dataset": dataset, "strategy": partition, "overlap": overlap, "seed": seed,
         "binarize": None, "num_classes": int(len(np.unique(y)))},
        os.path.join(out_dir, "data_report.json"))
    return report


# ---------------------------------------------------------------- v2 partitioner + manifests
MANIFEST_SUFFIX = ".manifest.json"
PARTY_MANIFEST_FORMAT = "vfl-party-manifest/1"
V1_DATASETS = ("breast_cancer", "synthetic", "csv")
V1_STRATEGIES = ("random", "sequential", "importance_balanced", "importance_skewed")
V2_DATASETS = ("breast_cancer", "digits", "synthetic", "csv", "adult", "bank", "credit_default",
               "mnist", "fashion_mnist", "npz", "synthetic_sensors", "har", "multimodal_digits")


def _sha256_file(path: str) -> str:
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def write_party_manifests(out_dir: str, parties: list[dict], common: dict, full_manifest_path: str) -> list[str]:
    """Write one `<party>.manifest.json` per party holding ONLY what that party needs.

    The full manifest (manifest.json / data_report.json) lists every party's columns, label balance,
    importance shares and reference AUROC; it stays with the experimenter. Party files contain the
    party's own schema plus split metadata used for consistency checks. Label-derived information
    (number of classes, binarize rule) is written to the active party's file only.
    """
    full_hash = _sha256_file(full_manifest_path)
    paths = []
    for p in parties:
        active = p["role"] == "active"
        part = {k: common[k] for k in ("partitioner", "strategy", "overlap", "seed")}
        part["dataset"] = str(common["dataset"]).split("[")[0]       # drop "[binarize-rule]" suffix
        part["manifest_sha256"] = full_hash
        if active:
            part["binarize"] = common.get("binarize")
            part["num_classes"] = common.get("num_classes")
        m = {"format": PARTY_MANIFEST_FORMAT, "name": p["name"], "role": p["role"], "file": p["file"],
             "rows": int(p["rows"]), "model_input_dim": int(p["model_input_dim"]),
             "columns": list(p["columns"]), "categorical_columns": list(p.get("categorical_columns") or []),
             "partition": part}
        if p.get("shape"):
            m["shape"] = [int(v) for v in p["shape"]]
        path = os.path.join(out_dir, os.path.splitext(p["file"])[0] + MANIFEST_SUFFIX)
        with open(path, "w") as fh:
            json.dump(m, fh, indent=2)
        paths.append(path)
    return paths


def _append_sentinels(path: str, name: str, active: bool, k: int = 5) -> int:
    """Rows nobody else holds, with sentinel ids/values: they must never appear in logs."""
    df = pd.read_csv(path, dtype={ID_COL: str})
    feats = [c for c in df.columns if c not in (ID_COL, LABEL_COL)]
    sent = pd.DataFrame({c: [SENTINEL_VALUE] * k for c in feats})
    sent.insert(0, ID_COL, [f"{SENTINEL_ID_PREFIX}{name}-{i:04d}" for i in range(k)])
    if active:
        sent[LABEL_COL] = 1
    df = pd.concat([df, sent], ignore_index=True)
    df.to_csv(path, index=False)
    return len(df)


def prepare_v2(out_dir: str, dataset: str = "breast_cancer", n_parties: int = 2, strategy: str = "random",
               seed: int = 0, overlap: float = 1.0, binarize: str | None = None, id_style: str = "hash",
               alpha: float = 1.0, top_frac: float = 0.5, informative_party: int | None = None,
               columns: list[list] | None = None, grid: tuple | None = None, one_hot: bool = True,
               n_samples: int = 20000, n_features: int = 40, csv_path: str | None = None,
               label: str | None = None, sentinel: bool = False, evaluate: bool = False) -> dict:
    """Run the vendored v2 partitioner (vfl/partition.py), then add per-party manifests, optional
    sentinel rows and a v1-compatible data_report.json for the benchmark analysis."""
    from . import partition as P
    if dataset in ("npz", "synthetic_sensors", "har", "multimodal_digits"):
        raise ValueError(f"dataset {dataset!r} is prepared by vfl.datasets (use `prepare-data`)")
    if strategy == "importance_skewed":          # v1 name: most informative columns at a passive party
        strategy = "importance"
        informative_party = 1 if informative_party is None else informative_party
    elif strategy == "importance_balanced":
        raise ValueError("importance_balanced exists only in the v1 partitioner; with v2 use "
                         "--partition importance --informative-party K, or dirichlet/random")
    if strategy not in P.STRATEGIES:
        raise ValueError(f"unknown v2 strategy {strategy!r}; choose from {P.STRATEGIES}")
    ds = P.load_dataset(dataset, csv_path, label, n_samples, n_features, seed, binarize)
    if binarize and ds.image_shape is None:
        raise ValueError("--binarize applies to image datasets (digits, mnist, fashion_mnist)")
    m = P.partition(ds, n_parties, out_dir, strategy, seed, overlap, id_style, alpha, top_frac,
                    0 if informative_party is None else informative_party, columns, grid, one_hot,
                    LABEL_COL, evaluate)
    full = os.path.join(out_dir, "manifest.json")
    if sentinel:
        for p in m["parties"]:
            p["rows"] = _append_sentinels(os.path.join(out_dir, p["file"]), p["name"], p["role"] == "active")
        m["sentinel_rows_per_party"] = 5
        with open(full, "w") as fh:
            json.dump(m, fh, indent=2)
    write_party_manifests(out_dir, m["parties"],
                          {"partitioner": "v2", "dataset": m["dataset"], "strategy": strategy,
                           "overlap": overlap, "seed": seed, "binarize": binarize,
                           "num_classes": m["num_classes"]}, full)
    report = {"dataset": m["dataset"], "seed": seed, "n_total": m["entities_total"], "overlap_target": overlap,
              "partition": strategy, "partitioner": "v2", "num_classes": m["num_classes"],
              "n_aligned_expected": m["intersection_size"], "overlap_ratio_actual": m["intersection_ratio"],
              "parties": {p["name"]: {"rows": p["rows"], "n_features": p["model_input_dim"],
                                      "has_labels": p["role"] == "active",
                                      "file": os.path.join(out_dir, p["file"])} for p in m["parties"]}}
    with open(os.path.join(out_dir, "data_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return m


def load_party_manifest(path: str) -> dict:
    with open(path) as fh:
        m = json.load(fh)
    if m.get("format") != PARTY_MANIFEST_FORMAT:
        raise ValueError(f"{path}: not a party manifest (format {m.get('format')!r})")
    return m


# ---------------------------------------------------------------- party data
@dataclass
class PartyData:
    ids: list[str]
    X: np.ndarray                 # float32 [n_rows, n_features]
    feature_names: list[str]
    y: np.ndarray | None = None   # only at the active party
    manifest: dict | None = None  # this party's own manifest, if one was found

    @property
    def role(self) -> str:
        return "active" if self.y is not None else "passive"

    @property
    def input_dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def shape(self) -> list[int] | None:
        """[C, H, W] for image tiles (used by CNN bottom models), else None."""
        return (self.manifest or {}).get("shape")

    @classmethod
    def load(cls, path: str, require_label: bool = False, manifest_path: str | None = None) -> "PartyData":
        """CSV (default) or .npz party file."""
        if path.endswith(".npz"):
            return cls.from_npz(path, require_label, manifest_path)
        return cls.from_csv(path, require_label, manifest_path)

    @classmethod
    def from_npz(cls, path: str, require_label: bool = False, manifest_path: str | None = None) -> "PartyData":
        """Large image/sensor sets: arrays `ids` [n], `X` [n, d] float32, `columns` [d], `y` [n] (active only)."""
        with np.load(path, allow_pickle=False) as z:
            if "ids" not in z.files or "X" not in z.files:
                raise ValueError(f"{path}: npz party file needs arrays 'ids' and 'X'")
            ids = [str(i) for i in z["ids"]]
            X = np.ascontiguousarray(z["X"], dtype=np.float32)
            cols = [str(c) for c in z["columns"]] if "columns" in z.files else [f"x{i}" for i in range(X.shape[1])]
            y = z["y"].astype(np.float32) if "y" in z.files else None
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate entity ids")
        if y is not None and not require_label:
            raise ValueError(f"{path}: passive party data must not contain labels")
        if y is None and require_label:
            raise ValueError(f"{path}: active party data must contain labels ('y')")
        out = cls(ids=ids, X=X, feature_names=cols, y=y)
        if manifest_path is None:
            auto = os.path.splitext(path)[0] + MANIFEST_SUFFIX
            manifest_path = auto if os.path.exists(auto) else None
        if manifest_path:
            out.manifest = load_party_manifest(manifest_path)
            out._check_manifest(path)
        return out

    @classmethod
    def from_csv(cls, path: str, require_label: bool = False, manifest_path: str | None = None) -> "PartyData":
        """Load a party CSV. If `<stem>.manifest.json` sits next to it (or `manifest_path` is given),
        the manifest's role, row count, model_input_dim, column order and shape are verified."""
        df = pd.read_csv(path, dtype={ID_COL: str})
        if ID_COL not in df.columns:
            raise ValueError(f"{path}: missing '{ID_COL}' column")
        if df[ID_COL].duplicated().any():
            raise ValueError(f"{path}: duplicate entity ids")
        y = None
        if LABEL_COL in df.columns:
            y = df.pop(LABEL_COL).to_numpy().astype(np.float32)
            if require_label is False:
                # A passive party must not hold labels in this design.
                raise ValueError(f"{path}: passive party data must not contain '{LABEL_COL}'")
        elif require_label:
            raise ValueError(f"{path}: active party data must contain '{LABEL_COL}'")
        ids = df.pop(ID_COL).astype(str).tolist()
        X = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        out = cls(ids=ids, X=X, feature_names=list(df.columns), y=y)
        if manifest_path is None:
            auto = os.path.splitext(path)[0] + MANIFEST_SUFFIX
            manifest_path = auto if os.path.exists(auto) else None
        if manifest_path:
            out.manifest = load_party_manifest(manifest_path)
            out._check_manifest(path)
        return out

    def _check_manifest(self, path: str):
        m = self.manifest
        want = "active" if self.y is not None else "passive"
        problems = []
        if m["role"] != want:
            problems.append(f"manifest role is {m['role']!r} but this file is loaded as {want!r}")
        if m["rows"] != len(self.ids):
            problems.append(f"manifest says {m['rows']} rows, file has {len(self.ids)}")
        if m["model_input_dim"] != self.input_dim:
            problems.append(f"manifest model_input_dim {m['model_input_dim']} != {self.input_dim} feature columns")
        if not m.get("categorical_columns") and m.get("columns") and m["columns"] != self.feature_names:
            problems.append("feature column order differs from the manifest")
        if m.get("shape") and int(np.prod(m["shape"])) != self.input_dim:
            problems.append(f"shape {m['shape']} does not match {self.input_dim} inputs")
        if problems:
            raise ValueError(f"{path}: data does not match its manifest: " + "; ".join(problems))

    def summary(self) -> dict:
        """Counts and split metadata only: safe to send to the coordinator."""
        s = {"rows": len(self.ids), "n_features": int(self.X.shape[1]), "has_labels": self.y is not None}
        if self.manifest:
            s["partition"] = dict(self.manifest["partition"])
            if self.shape:
                s["shape"] = list(self.shape)
            if self.manifest.get("text_encoder"):
                s["text_encoder"] = self.manifest["text_encoder"].get("encoder")
        return s


# ---------------------------------------------------------------- splits & schedule
def make_splits(n_aligned: int, seed: int, val: float, test: float) -> dict[str, np.ndarray]:
    """Entity-level split over the ALIGNED order; identical at every party."""
    perm = np.random.default_rng(seed + 1001).permutation(n_aligned)
    n_test = int(round(test * n_aligned))
    n_val = int(round(val * n_aligned))
    return {"test": np.sort(perm[:n_test]), "val": np.sort(perm[n_test:n_test + n_val]),
            "train": np.sort(perm[n_test + n_val:])}


def epoch_batches(train_idx: np.ndarray, batch_size: int, seed: int, epoch: int) -> list[np.ndarray]:
    """Deterministic shuffled mini-batches for one epoch (drop nothing)."""
    order = np.random.default_rng((seed * 100003 + epoch) % (2 ** 32)).permutation(len(train_idx))
    shuffled = train_idx[order]
    return [shuffled[i:i + batch_size] for i in range(0, len(shuffled), batch_size)]


def eval_batches(splits: dict, eval_batch_size: int) -> list[tuple[str, np.ndarray]]:
    out = []
    for name in ("val", "test"):
        idx = splits[name]
        for i in range(0, len(idx), eval_batch_size):
            out.append((name, idx[i:i + eval_batch_size]))
    return out


def batch_digest(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:16]


class Standardizer:
    """Per-party feature standardisation fitted on that party's TRAIN rows only."""

    def __init__(self, X_train: np.ndarray):
        self.mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        self.std = np.where(std < 1e-8, 1.0, std)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)
