"""Data preparation from the UI (v2.1 item 3).

Every `prepare-data` option is described once in PREP_OPTIONS (widget, presets, when it applies). The UI
never re-implements data preparation: `build_argv()` turns the form into a `vfl.cli prepare-data` argv,
validated by the real CLI parser, and the job runs `python -m vfl.cli prepare-data ...` with the datasets
folder as its working directory. Same code, same relative paths, so a UI-prepared dataset is
byte-identical to `cd datasets && python -m vfl.cli prepare-data ... --out NAME` (tests/test_dataprep.py).

Workspace layout (all under the UI's --workspace):
  datasets/<name>/        what prepare-data wrote (party CSV/NPZ + manifests + manifest.json/data_report.json)
  datasets/.index/        the options each saved dataset was made with (kept OUTSIDE the dataset folder)
  datasets/.inputs/       files the form generated for the CLI (manual column assignment JSON)
  uploads/                CSV / NPZ uploads, extracted HAR zips, embed-text outputs
  certs/<name>/           certificates made from the UI (`vfl.cli certs`)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import zipfile

V1_DATASETS = ("breast_cancer", "synthetic", "csv")
V2_DATASETS = ("breast_cancer", "digits", "synthetic", "csv", "adult", "bank", "credit_default", "mnist",
               "fashion_mnist", "npz", "synthetic_sensors", "har", "multimodal_digits")
ONLINE = ("adult", "bank", "credit_default", "mnist", "fashion_mnist")
V1_STRATEGIES = ("random", "sequential", "importance_balanced", "importance_skewed")
TAB_STRATEGIES = ("random", "sequential", "dirichlet", "importance", "correlated", "manual", "importance_skewed")
IMAGE_DATASETS = ("digits", "mnist", "fashion_mnist", "npz")
NPZ_STRATEGIES = ("image_strips", "image_grid", "sequential", "random")
BINARIZE = ("odd_even",) + tuple(f"one_vs_rest:{k}" for k in range(10)) + ("pair:3,8", "pair:1,7", "pair:4,9")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def kind_of(v: dict) -> str:
    ds = v.get("dataset")
    if ds in ("synthetic_sensors", "har"):
        return "sensor"
    if ds == "multimodal_digits":
        return "multimodal"
    if ds in IMAGE_DATASETS:
        return "image"
    return "tabular"


def strategies_for(v: dict) -> tuple:
    if v.get("partitioner") != "v2":
        return V1_STRATEGIES
    k = kind_of(v)
    if v.get("dataset") == "npz":
        return NPZ_STRATEGIES
    if k == "image":
        return ("image_strips", "image_grid") + TAB_STRATEGIES
    if k == "tabular":
        return TAB_STRATEGIES
    return ()


def grids_for(n: int) -> list[str]:
    return [f"{r}x{n // r}" for r in range(1, n + 1) if n % r == 0]


def ratio_presets(n: int) -> list:
    eq = ",".join(f"{1 / n:.4g}" for _ in range(n))
    act_big = ",".join(["0.5"] + [f"{0.5 / (n - 1):.4g}"] * (n - 1))
    act_small = ",".join(["0.2"] + [f"{0.8 / (n - 1):.4g}"] * (n - 1))
    return [None, eq, act_big, act_small]


# Each option: dest (argparse), flag, widget, default, options(values)->list or tuple, applies(values)->bool, help.
# widget: select | toggle | upload | column | dir | columns_table
def _opts(**kw):
    kw.setdefault("applies", lambda v: True)
    kw.setdefault("widget", "select")
    return kw


PREP_OPTIONS = [
    _opts(dest="partitioner", flag="--partitioner", default="v2", options=lambda v: ["v1", "v2"],
          help="v1 = original generator (reproduces v1 results); v2 = vfl/partition.py (all strategies, manifests)."),
    _opts(dest="dataset", flag="--dataset", default="breast_cancer",
          options=lambda v: list(V2_DATASETS if v.get("partitioner") == "v2" else V1_DATASETS),
          help="Built-in dataset, an uploaded CSV, an .npz image file, HAR, synthetic sensors or the multimodal set. "
               "adult/bank/credit_default/mnist/fashion_mnist download from openml.org on first use."),
    _opts(dest="csv", flag="--csv", widget="upload", default=None, accept=".csv",
          options=lambda v: [], applies=lambda v: v.get("dataset") == "csv",
          help="Uploaded CSV with one row per entity."),
    _opts(dest="label", flag="--label", widget="column", default=None, options=lambda v: [],
          applies=lambda v: v.get("dataset") == "csv",
          help="Label column of the CSV (v1 needs it binary; v2 accepts k classes)."),
    _opts(dest="npz", flag="--npz", widget="upload", default=None, accept=".npz", options=lambda v: [],
          applies=lambda v: v.get("dataset") == "npz", help="Image file with arrays x and y (e.g. CIFAR-10 via np.savez)."),
    _opts(dest="har_dir", flag="--har-dir", widget="dir", default=None, options=lambda v: [],
          applies=lambda v: v.get("dataset") == "har",
          help="The extracted 'UCI HAR Dataset' folder (upload the zip; it is extracted under uploads/)."),
    _opts(dest="parties", flag="--parties", default=2,
          options=lambda v: [3] if v.get("dataset") == "multimodal_digits" else list(range(2, 10)),
          help="Total parties including the one active (label-owning) party; passives = parties - 1."),
    _opts(dest="partition", flag="--partition", default="random", options=lambda v: list(strategies_for(v)),
          applies=lambda v: bool(strategies_for(v)),
          help="How columns are split between parties. Sensor and multimodal datasets use a fixed split."),
    _opts(dest="ratios", flag="--ratios", default=None, options=lambda v: ratio_presets(int(v.get("parties") or 2)),
          applies=lambda v: v.get("partitioner") != "v2", labels={None: "even (default)"},
          help="v1: feature share per party, active first."),
    _opts(dest="alpha", flag="--alpha", default=1.0, options=lambda v: [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0],
          applies=lambda v: v.get("partitioner") == "v2" and v.get("partition") == "dirichlet",
          help="Dirichlet concentration: small = very uneven column shares."),
    _opts(dest="top_frac", flag="--top-frac", default=0.5, options=lambda v: [0.1, 0.2, 0.3, 0.5, 0.7],
          applies=lambda v: v.get("partitioner") == "v2" and v.get("partition") == "importance",
          help="importance: fraction of the most informative columns given to one party."),
    _opts(dest="informative_party", flag="--informative-party", default=None,
          options=lambda v: [None] + list(range(int(v.get("parties") or 2))),
          applies=lambda v: v.get("partitioner") == "v2" and v.get("partition") in ("importance", "importance_skewed"),
          labels={None: "default (active; passive-1 for importance_skewed)"},
          help="importance: which party gets the top columns (0 = active, 1 = passive-1, ...)."),
    _opts(dest="columns_json", flag="--columns-json", widget="columns_table", default=None, options=lambda v: [],
          applies=lambda v: v.get("partitioner") == "v2" and v.get("partition") == "manual",
          help="manual: assign every column to one party (written to a JSON file for the CLI)."),
    _opts(dest="grid", flag="--grid", default=None, options=lambda v: [None] + grids_for(int(v.get("parties") or 2)),
          applies=lambda v: v.get("partitioner") == "v2" and v.get("partition") == "image_grid",
          labels={None: "near-square (default)"}, help="image_grid: ROWSxCOLS tiles; rows x cols must equal parties."),
    _opts(dest="overlap", flag="--overlap", default=None, options=lambda v: [None, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
          labels={None: "default (v1 0.9, v2 1.0)"}, help="Fraction of entities every party holds (alignment is real)."),
    _opts(dest="id_style", flag="--id-style", default="hash", options=lambda v: ["hash", "int"],
          applies=lambda v: v.get("partitioner") == "v2", help="Opaque hashed ids or plain integers."),
    _opts(dest="binarize", flag="--binarize", default=None, options=lambda v: [None] + list(BINARIZE),
          applies=lambda v: v.get("partitioner") == "v2" and kind_of(v) in ("image", "sensor", "multimodal"),
          labels={None: "keep all classes"}, help="Turn a multi-class task binary (so binary AUROC / the norm attack apply)."),
    _opts(dest="n", flag="--n", default=20000, options=lambda v: [1000, 3000, 5000, 10000, 20000, 50000],
          applies=lambda v: v.get("dataset") in ("synthetic", "synthetic_sensors"),
          help="Rows (synthetic). synthetic_sensors uses 3000 when left at 20000."),
    _opts(dest="features", flag="--features", default=40, options=lambda v: [10, 20, 40, 80, 160],
          applies=lambda v: v.get("dataset") == "synthetic", help="Features (synthetic)."),
    _opts(dest="channels", flag="--channels", default=6, options=lambda v: [3, 6, 9, 12],
          applies=lambda v: v.get("dataset") == "synthetic_sensors", help="Sensor channels (3 per stream)."),
    _opts(dest="timesteps", flag="--timesteps", default=64, options=lambda v: [32, 64, 128, 256],
          applies=lambda v: v.get("dataset") == "synthetic_sensors", help="Window length."),
    _opts(dest="classes", flag="--classes", default=4, options=lambda v: [2, 3, 4, 6, 8],
          applies=lambda v: v.get("dataset") == "synthetic_sensors", help="Activity classes."),
    _opts(dest="text_encoder", flag="--text-encoder", default="hashing", options=lambda v: ["hashing", "minilm", "distilbert"],
          applies=lambda v: v.get("dataset") == "multimodal_digits",
          help="Frozen text encoder at the text party (hashing works offline; the others need requirements-optional)."),
    _opts(dest="format", flag="--format", default=None, options=lambda v: [None, "csv", "npz"],
          applies=lambda v: v.get("partitioner") == "v2" and v.get("dataset") in ("npz", "synthetic_sensors", "har"),
          labels={None: "auto"}, help="Party file format."),
    _opts(dest="seed", flag="--seed", default=0, options=lambda v: [0, 1, 2, 3, 4, 5, 7, 42], help="Seed of the split."),
    _opts(dest="no_one_hot", flag="--no-one-hot", widget="toggle", default=False, options=lambda v: [False, True],
          applies=lambda v: v.get("partitioner") == "v2" and kind_of(v) == "tabular",
          help="Keep categorical columns un-encoded (default: one-hot AFTER the split, so a categorical stays with one party)."),
    _opts(dest="evaluate", flag="--evaluate", widget="toggle", default=False, options=lambda v: [False, True],
          applies=lambda v: v.get("partitioner") == "v2" and kind_of(v) in ("tabular", "image"),
          help="Add per-party reference AUROC to manifest.json (experiment-only: uses every party's data in one place)."),
    _opts(dest="sentinel", flag="--sentinel", widget="toggle", default=False, options=lambda v: [False, True],
          help="Add sentinel rows for the log-audit test (they must never appear in any log)."),
]
BY_DEST = {o["dest"]: o for o in PREP_OPTIONS}
# prepare-data arguments the form deliberately does not expose, and why
NOT_IN_FORM = {"out": "the saved dataset name decides the output folder",
               "passives": "superseded by the parties dropdown (--parties)"}


def defaults() -> dict:
    return {o["dest"]: o["default"] for o in PREP_OPTIONS}


def normalise(values: dict) -> dict:
    v = {**defaults(), **{k: x for k, x in (values or {}).items() if k in BY_DEST}}
    if v["dataset"] not in BY_DEST["dataset"]["options"](v):
        v["dataset"] = "breast_cancer"
    if v["dataset"] == "multimodal_digits":
        v["parties"] = 3
    st = strategies_for(v)
    if st and v.get("partition") not in st:
        v["partition"] = st[0]
    if v.get("grid") and v["grid"] not in grids_for(int(v["parties"])):
        v["grid"] = None
    return v


def form(values: dict, ctx: dict | None = None) -> dict:
    """The fields to draw for the current values: visibility and options are decided here, in one place."""
    v = normalise(values)
    ctx = ctx or {}
    fields = []
    for o in PREP_OPTIONS:
        if not o["applies"](v):
            continue
        f = {"dest": o["dest"], "flag": o["flag"], "widget": o["widget"], "value": v.get(o["dest"]),
             "default": o["default"], "help": o["help"], "accept": o.get("accept")}
        if o["widget"] in ("select", "toggle"):
            opts = o["options"](v)
            labels = o.get("labels") or {}
            f["options"] = [{"value": x, "label": labels.get(x, "on" if x is True else "off" if x is False else str(x))}
                            for x in opts]
        elif o["widget"] == "upload":
            f["options"] = [x for x in ctx.get("uploads", []) if x.endswith(o.get("accept") or "")]
        elif o["widget"] == "dir":
            f["options"] = ctx.get("upload_dirs", [])
        elif o["widget"] == "column":
            f["options"] = ctx.get("csv_columns", [])
        fields.append(f)
    return {"values": v, "fields": fields, "kind": kind_of(v), "passives": int(v["parties"]) - 1,
            "notes": _notes(v)}


def _notes(v: dict) -> list[str]:
    out = []
    if v["dataset"] in ONLINE:
        out.append(f"{v['dataset']} is downloaded from openml.org on first use (needs network on this machine).")
    if v.get("partitioner") == "v2" and v.get("partition") in ("importance", "correlated"):
        out.append("importance / correlated splits use every column and the labels in one place: experiment-only.")
    if v.get("partitioner") != "v2" and v["dataset"] == "csv":
        out.append("the v1 partitioner needs a binary label column.")
    return out


def build_argv(values: dict, out: str, inputs_dir: str | None = None) -> list[str]:
    """`prepare-data` argv for these form values, checked by the real CLI parser."""
    v = normalise(values)
    argv = ["prepare-data"]
    for o in PREP_OPTIONS:
        if not o["applies"](v):
            continue
        x = v.get(o["dest"])
        if o["widget"] == "toggle":
            if x:
                argv.append(o["flag"])
            continue
        if x is None or x == "":
            if o["dest"] in ("csv", "label", "npz", "har_dir"):
                raise ValueError(f"{o['dest']} is required for dataset {v['dataset']}")
            if o["dest"] == "columns_json":
                raise ValueError("manual strategy: assign every column to a party")
            continue
        if o["dest"] == "columns_json":
            n = int(v["parties"])
            if not isinstance(x, list) or len(x) != n or any(not g for g in x):
                raise ValueError(f"manual strategy: give each of the {n} parties at least one column")
            if not inputs_dir:
                raise ValueError("manual strategy needs an inputs folder")
            os.makedirs(inputs_dir, exist_ok=True)
            digest = hashlib.sha256(json.dumps(x).encode()).hexdigest()[:12]
            path = os.path.join(os.path.abspath(inputs_dir), f"columns-{digest}.json")
            with open(path, "w") as fh:
                json.dump(x, fh)
            x = path
        argv += [o["flag"], str(x)]
    argv += ["--out", out]
    from ..cli import build_parser
    try:
        build_parser().parse_args(argv)
    except SystemExit as e:
        raise ValueError(f"prepare-data rejected these options ({' '.join(argv)})") from e
    return argv


# ---------------------------------------------------------------- summaries
def summarize(d: str) -> dict:
    """What the preview / saved-dataset list shows. Reads only what prepare-data wrote."""
    mpath, rpath = os.path.join(d, "manifest.json"), os.path.join(d, "data_report.json")
    out = {"dir": d, "name": os.path.basename(d)}
    if os.path.exists(mpath):
        m = json.load(open(mpath))
        out.update({"partitioner": "v2", "dataset": m.get("dataset"), "strategy": m.get("strategy"),
                    "n_parties": m.get("n_parties"), "entities_total": m.get("entities_total"),
                    "intersection_size": m.get("intersection_size"), "intersection_ratio": m.get("intersection_ratio"),
                    "num_classes": m.get("num_classes"), "label_balance": m.get("label_balance"),
                    "image_shape": m.get("image_shape"), "experiment_only": m.get("experiment_only"),
                    "reference_auroc": m.get("reference_auroc"), "overlap": m.get("overlap_requested"),
                    "parties": [{"name": p["name"], "role": p["role"], "file": p["file"], "rows": p["rows"],
                                 "inputs": p.get("model_input_dim"), "n_columns": len(p.get("columns") or []),
                                 "shape": p.get("shape"), "importance_share": p.get("importance_share")}
                                for p in m.get("parties", [])]})
    elif os.path.exists(rpath):
        r = json.load(open(rpath))
        out.update({"partitioner": r.get("partitioner", "v1"), "dataset": r.get("dataset"), "strategy": r.get("partition"),
                    "n_parties": len(r.get("parties", {})), "entities_total": r.get("n_total"),
                    "intersection_size": r.get("n_aligned_expected"),
                    "intersection_ratio": r.get("overlap_ratio_actual"), "overlap": r.get("overlap_target"),
                    "label_balance": r.get("positive_rate_aligned"), "num_classes": r.get("num_classes", 2),
                    "parties": [{"name": n, "role": "active" if p.get("has_labels") else "passive",
                                 "file": os.path.basename(p.get("file", "")), "rows": p.get("rows"),
                                 "inputs": p.get("n_features"), "n_columns": p.get("n_features"),
                                 "mutual_info_sum": p.get("mutual_info_sum")}
                                for n, p in r.get("parties", {}).items()]})
    else:
        raise FileNotFoundError(f"{d} holds no prepare-data output")
    out["label_distribution"] = _label_distribution(d, out["parties"])
    if out.get("reference_auroc") is not None:
        out["reference_auroc_note"] = ("experiment-only: logistic regression on each party's columns and on all "
                                       "columns together, computed with every party's data and the labels in one place")
    return out


def _label_distribution(d, parties):
    act = next((p for p in parties if p["role"] == "active"), None)
    if not act:
        return None
    path = os.path.join(d, act["file"])
    try:
        if path.endswith(".csv"):
            import pandas as pd
            y = pd.read_csv(path, usecols=["label"])["label"]
        else:
            import numpy as np
            with np.load(path, allow_pickle=False) as z:
                key = "label" if "label" in z.files else "y" if "y" in z.files else None
                if key is None:
                    return None
                import pandas as pd
                y = pd.Series(z[key])
        vc = y.value_counts(normalize=True).sort_index()
        return {str(k): round(float(x), 4) for k, x in vc.items()}
    except Exception:
        return None


# ---------------------------------------------------------------- workspace
class Workspace:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        for sub in ("datasets", "uploads", "certs", "deployments", "results", "runs"):
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)

    def path(self, *p):
        return os.path.join(self.root, *p)

    @property
    def datasets(self):
        return self.path("datasets")

    # datasets ------------------------------------------------------
    def check_name(self, name: str) -> str:
        if not isinstance(name, str) or not NAME_RE.fullmatch(name) or name.startswith(".") or ".." in name:
            raise ValueError("dataset/cert names: letters, digits, '.', '_' or '-' (max 64), not starting with '.'")
        return name

    def list_datasets(self) -> list[dict]:
        out = []
        for n in sorted(os.listdir(self.datasets)):
            d = os.path.join(self.datasets, n)
            if n.startswith(".") or not os.path.isdir(d):
                continue
            try:
                s = summarize(d)
            except Exception:
                continue
            meta = self.dataset_meta(n)
            s["created"] = meta.get("created") or os.path.getmtime(d)
            s["options"] = meta.get("options")
            s["argv"] = meta.get("argv")
            out.append(s)
        return out

    def dataset_meta(self, name: str) -> dict:
        try:
            return json.load(open(self.path("datasets", ".index", f"{name}.json")))
        except Exception:
            return {}

    def save_meta(self, name: str, meta: dict):
        os.makedirs(self.path("datasets", ".index"), exist_ok=True)
        with open(self.path("datasets", ".index", f"{name}.json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=str)

    def delete_dataset(self, name: str):
        self.check_name(name)
        shutil.rmtree(self.path("datasets", name), ignore_errors=True)
        try:
            os.remove(self.path("datasets", ".index", f"{name}.json"))
        except FileNotFoundError:
            pass

    # uploads -------------------------------------------------------
    def save_upload(self, filename: str, stream) -> dict:
        base = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(filename or "upload"))[:96] or "upload"
        dest = self.path("uploads", base)
        with open(dest, "wb") as fh:
            shutil.copyfileobj(stream, fh)
        out = {"file": base}
        if base.lower().endswith(".zip"):
            target = self.path("uploads", os.path.splitext(base)[0])
            try:
                safe_extract(dest, target)
            except (ValueError, zipfile.BadZipFile) as e:
                shutil.rmtree(target, ignore_errors=True)
                os.remove(dest)
                raise ValueError(f"zip refused: {e}")
            out["dir"] = os.path.basename(target)
        return out

    def uploads(self) -> dict:
        files, dirs = [], []
        up = self.path("uploads")
        for n in sorted(os.listdir(up)):
            p = os.path.join(up, n)
            (dirs if os.path.isdir(p) else files).append(n)
        # an extracted HAR zip usually nests 'UCI HAR Dataset'; offer the inner folders too
        nested = []
        for d in dirs:
            for n in sorted(os.listdir(os.path.join(up, d))):
                if os.path.isdir(os.path.join(up, d, n)) and not n.startswith("__"):
                    nested.append(f"{d}/{n}")
        return {"files": files, "dirs": dirs + nested}

    def upload_path(self, rel: str) -> str:
        p = os.path.abspath(self.path("uploads", rel))
        if not p.startswith(self.path("uploads") + os.sep):
            raise ValueError("path outside uploads/")
        return p

    def csv_columns(self, rel: str) -> list[str]:
        import pandas as pd
        return [c for c in pd.read_csv(self.upload_path(rel), nrows=0).columns if c != "id"]

    # certificates --------------------------------------------------
    def list_certs(self) -> list[dict]:
        out = []
        for n in sorted(os.listdir(self.path("certs"))):
            d = self.path("certs", n)
            if not os.path.isdir(d):
                continue
            files = sorted(os.listdir(d))
            out.append({"name": n, "mode": "pki" if "pki.json" in files else "static",
                        "nodes": sorted(f[:-4] for f in files if f.endswith(".crt") and f not in ("ca.crt", "intermediate.crt")),
                        "created": os.path.getmtime(d)})
        return out


MAX_EXTRACT_BYTES = 8 * 2 ** 30
MAX_EXTRACT_MEMBERS = 100_000


def safe_extract(zpath: str, target: str, max_bytes: int = MAX_EXTRACT_BYTES, max_members: int = MAX_EXTRACT_MEMBERS):
    """Zip-slip safe, and bounded: v2.2 had no size limit, so a 10 MB zip bomb could fill the disk (RT-16)."""
    root = os.path.abspath(target)
    with zipfile.ZipFile(zpath) as z:
        infos = z.infolist()
        if len(infos) > max_members:
            raise ValueError(f"zip has {len(infos)} members (limit {max_members})")
        if sum(m.file_size for m in infos) > max_bytes:
            raise ValueError(f"zip expands beyond {max_bytes // 2 ** 20} MiB")
        for m in infos:
            dest = os.path.abspath(os.path.join(root, m.filename))
            if not (dest == root or dest.startswith(root + os.sep)):
                raise ValueError(f"unsafe path in zip: {m.filename}")
            if (m.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"symbolic link in zip: {m.filename}")
        os.makedirs(target, exist_ok=True)
        written = 0
        for m in infos:                                  # declared sizes can lie: count what is really written
            dest = os.path.abspath(os.path.join(root, m.filename))
            if m.is_dir():
                os.makedirs(dest, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(m) as src, open(dest, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(f"zip expands beyond {max_bytes // 2 ** 20} MiB")
                    out.write(chunk)


def columns_for(values: dict, ws: Workspace) -> list[str]:
    """Columns available for the manual assignment table / label dropdown."""
    v = normalise(values)
    if v["dataset"] == "csv":
        if not v.get("csv"):
            return []
        return [c for c in ws.csv_columns(v["csv"]) if c != v.get("label")]
    if kind_of(v) in ("sensor", "multimodal") or v["dataset"] == "npz":
        return []
    from .. import partition as P
    ds = P.load_dataset(v["dataset"], None, None, int(v.get("n") or 20000), int(v.get("features") or 40),
                        int(v.get("seed") or 0), None)
    return [str(c) for c in ds.X.columns]


def certs_argv(name: str, n_passive: int, mode: str = "static", hosts: list[str] | None = None,
               active_id: str = "active", hours: float | None = None, domain: str | None = None,
               extra: list[str] | None = None) -> list[str]:
    if mode not in ("static", "pki"):
        raise ValueError("mode must be static or pki")
    if not 1 <= int(n_passive) <= 8:
        raise ValueError("passives must be 1-8")
    argv = ["certs", "--out", os.path.join("certs", name), "--passives", str(int(n_passive)), "--mode", mode,
            "--active-id", active_id]
    for h in hosts or []:
        if not re.match(r"^[A-Za-z0-9.:-]{1,253}$", h):
            raise ValueError(f"bad host {h!r}")
    if hosts:
        argv += ["--hosts", *hosts]
    if extra:
        argv += ["--extra", *extra]
    if mode == "pki":
        if hours:
            argv += ["--hours", str(float(hours))]
        if domain:
            argv += ["--domain", domain]
    from ..cli import build_parser
    build_parser().parse_args(argv)
    return argv


def local_addresses() -> list[str]:
    out = ["127.0.0.1"]
    try:
        import psutil
        import socket
        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if a.family == socket.AF_INET and a.address not in out:
                    out.append(a.address)
    except Exception:
        pass
    return out


def now() -> float:
    return time.time()
