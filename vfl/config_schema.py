"""Session-config schema: the single source of truth for every training hyperparameter (v2.1 item 2).

Each entry says what one config key is: type, default, the preset values the dashboard offers in its
dropdown, the form group it sits in, help text, and which bottom models it applies to. Generated from it:

  * `vfl.config.DEFAULT_SESSION`             (defaults)
  * `vfl.config.validate_session_config`     (per-key type / range / choice checks)
  * `vfl.cli start|baseline --set k=v`       (unknown keys and bad values are refused before anything runs)
  * the coordinator's "Start a session" form (GET /api/config/schema; dropdowns and toggles only)
  * docs/CONFIG.md                           (`python -m vfl.cli config docs`)

`tests/test_config_schema.py` scans the training code for every key it reads and fails if one has no
entry here, so a new key cannot ship without appearing in the form, the validator and the docs.

This module must not import torch or grpc: the CLI client and the form use it on machines without them.

Presets are what the UI offers, not hard limits: `--set lr=0.002` is valid (it is a positive float).
Hard limits are `choices`, `min` / `max` and the type.
"""
from __future__ import annotations

import copy
import difflib
import json
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = 1

BOTTOM_MODELS = ("mlp", "linear", "cnn", "cnn1d", "gru", "text")
CODECS = ("fp32", "fp16", "int8", "topk")
# every split strategy any partitioner can write into a manifest (v1, v2, image, sensor, multimodal)
PARTITION_STRATEGIES = ("random", "sequential", "importance_balanced", "importance_skewed", "dirichlet",
                        "importance", "correlated", "manual", "image_strips", "image_grid", "sensor_streams",
                        "multimodal")
BINARIZE_RULES = ("odd_even",) + tuple(f"one_vs_rest:{k}" for k in range(10)) + ("pair:3,8", "pair:1,7", "pair:4,9")

GROUPS = (
    ("training", "Training"),
    ("model", "Model sizes"),
    ("bottom", "Bottom models"),
    ("align", "Alignment & codec"),
    ("network", "Network & timing"),
    ("defense", "Defences"),
    ("straggler", "Straggler"),
    ("fault", "Fault injection"),
    ("data", "Resume & data checks"),
    ("security", "Tensor validation"),
    ("runtime", "Runtime & telemetry"),
    ("labels", "Run labels"),
)
GROUP_IDS = tuple(g for g, _ in GROUPS)

# widget kinds the form knows how to draw (no free-text inputs anywhere)
WIDGETS = ("select", "toggle", "party_models", "pairs", "session", "parties", "salt", "name", "dataset")
TYPES = ("int", "float", "bool", "str", "enum", "int_list", "pairs", "bottom_map", "str_list")


@dataclass(frozen=True)
class Param:
    key: str                         # dotted path, e.g. "defense.clip"
    type: str                        # one of TYPES
    default: Any
    group: str
    help: str
    presets: tuple = ()              # dropdown values (the default is always offered too)
    choices: tuple = ()              # hard constraint (enums)
    nullable: bool = False           # None allowed
    null_label: str = "off"          # how None is shown in the dropdown
    min: float | None = None
    max: float | None = None
    min_exclusive: bool = False
    odd: bool = False                # int must be odd (conv kernels with "same" padding)
    applies_to: tuple = ()           # bottom-model kinds; () = applies whatever the models are
    show_if: tuple = ()              # ((key, (values...)), ...) all must hold; ("!", key, (values...)) negates
    widget: str = "select"
    labels: tuple = ()               # ((value, label), ...) for nicer option text
    unit: str = ""
    advanced: bool = False           # drawn inside the group's "advanced" fold
    internal: bool = False           # set by the system: not in the form, refused by --set
    read: bool = True                # False = declared for the record but not read by the code
    in_defaults: bool = True         # appears in DEFAULT_SESSION
    since: str = "v1"

    def to_json(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["labels"] = [[json.dumps(v), lab] for v, lab in self.labels]
        d["show_if"] = [list(c) for c in self.show_if]
        return json.loads(json.dumps(d))           # tuples -> lists


def _p(*a, **kw) -> Param:
    return Param(*a, **kw)


LR = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)

PARAMS: tuple[Param, ...] = (
    # ------------------------------------------------------------------ run labels
    _p("name", "str", "vfl-run", "labels", "Session name recorded in logs and the run index.", widget="name"),
    _p("dataset_tag", "str", "unspecified", "labels",
       "Informational dataset label recorded in logs. The form offers the datasets named in the parties' manifests.",
       widget="dataset"),

    # ------------------------------------------------------------------ training
    _p("seed", "int", 42, "training", "Seed for model init, batch order, splits and defence noise.",
       presets=(0, 1, 2, 3, 4, 5, 7, 42, 123, 2026), min=0),
    _p("epochs", "int", 10, "training", "Training epochs.", presets=(1, 2, 3, 5, 10, 15, 20, 30, 50, 100), min=1),
    _p("batch_size", "int", 64, "training", "Training mini-batch size (entities per exchanged batch).",
       presets=(16, 32, 64, 128, 256, 512, 1024), min=1),
    _p("lr", "float", 0.001, "training", "Learning rate (every party's optimizer uses the same value).",
       presets=LR, min=0, min_exclusive=True),
    _p("optimizer", "enum", "adam", "training", "Optimizer at every party.", choices=("adam", "sgd")),
    _p("momentum", "float", 0.0, "training", "SGD momentum (ignored by adam).", presets=(0.0, 0.5, 0.9, 0.95, 0.99),
       min=0, max=1, show_if=(("optimizer", ("sgd",)),)),
    _p("target_val_auroc", "float", None, "training",
       "Validation AUROC that counts as 'converged' for the time-to-target metric (training does not stop).",
       presets=(0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99), nullable=True, null_label="none", min=0, max=1),
    _p("split.val", "float", 0.15, "training", "Validation fraction of the aligned entities.",
       presets=(0.0, 0.1, 0.15, 0.2, 0.25), min=0, max=1, advanced=True),
    _p("split.test", "float", 0.15, "training", "Test fraction of the aligned entities.",
       presets=(0.1, 0.15, 0.2, 0.25, 0.3), min=0, max=1, min_exclusive=True, advanced=True),
    _p("eval_batch_size", "int", 512, "training", "Batch size for validation/test passes.",
       presets=(128, 256, 512, 1024, 2048), min=1, advanced=True),

    # ------------------------------------------------------------------ model sizes
    _p("hidden_dim", "int", 64, "model", "MLP bottom hidden width (0 = a single linear layer).",
       presets=(0, 16, 32, 64, 128, 256), min=0, applies_to=("mlp",)),
    _p("embed_dim", "int", 16, "model",
       "Representation width each party sends: the wire contract is [batch x embed_dim].",
       presets=(2, 4, 8, 16, 32, 64, 128), min=1),
    _p("top_hidden", "int", 32, "model", "Top-model hidden width at the active party (0 = linear top).",
       presets=(0, 16, 32, 64, 128, 256), min=0),
    _p("num_classes", "int", None, "model",
       "Number of classes. Empty = inferred from the labels (2 -> one logit + BCE, k > 2 -> cross-entropy).",
       presets=(2, 3, 4, 5, 6, 8, 10, 20, 100), nullable=True, null_label="auto (from labels)", min=2),

    # ------------------------------------------------------------------ bottom models
    _p("bottom_model", "bottom_map", "mlp", "bottom",
       "Bottom model per party. One kind for everyone, or a mapping {party_id: kind, default: kind}.",
       choices=BOTTOM_MODELS, widget="party_models"),
    _p("cnn_channels", "int_list", [16, 32], "bottom",
       "Conv widths, one conv layer per entry (cnn and cnn1d). The default [16, 32] is the v2 network.",
       presets=([8, 16], [16, 32], [32, 64], [16, 32, 64], [32, 64, 128], [16]), applies_to=("cnn", "cnn1d")),
    _p("cnn_kernel", "int", 3, "bottom", "2-D conv kernel size (odd; 'same' padding keeps the tile size).",
       presets=(3, 5, 7), min=1, odd=True, applies_to=("cnn",), since="v2.1"),
    _p("cnn_pool", "enum", "auto", "bottom",
       "Pooling after the 2-D convs. auto = 2x2 max-pool when the tile is at least 4x4 (v2 behaviour).",
       choices=("auto", "max", "avg", "none"), applies_to=("cnn",), since="v2.1"),
    _p("cnn1d_kernel", "int", 5, "bottom", "1-D conv kernel size over time (odd).",
       presets=(3, 5, 7, 9), min=1, odd=True, applies_to=("cnn1d",), since="v2.1"),
    _p("cnn1d_pool", "enum", "mean", "bottom", "How the 1-D conv features are pooled over time (mean is v2).",
       choices=("mean", "max"), applies_to=("cnn1d",), since="v2.1"),
    _p("gru_hidden", "int", 32, "bottom", "GRU hidden size.", presets=(16, 32, 64, 128), min=1, applies_to=("gru",)),
    _p("gru_layers", "int", 1, "bottom", "Stacked GRU layers (the last layer's final state feeds the head).",
       presets=(1, 2, 3), min=1, max=4, applies_to=("gru",), since="v2.1"),
    _p("text_hidden", "int", 0, "bottom",
       "Text bottom: hidden width of the trainable projection over the frozen encoder embeddings "
       "(0 = linear projection to embed_dim, v2). The encoder itself is chosen at data preparation.",
       presets=(0, 32, 64, 128), min=0, applies_to=("text",), since="v2.1"),

    # ------------------------------------------------------------------ alignment & codec
    _p("alignment", "enum", "experimental", "align",
       "Entity alignment. experimental is NON-PRIVATE (salted hashes reveal the intersection); "
       "dh_psi is Diffie-Hellman PSI for any number of parties (two-party protocol with one passive, "
       "multi-party through the active party with more).",
       choices=("experimental", "dh_psi"),
       labels=(("experimental", "experimental (NON-private)"), ("dh_psi", "DH-PSI (private)"))),
    _p("alignment_salt", "str", None, "align",
       "Salt for experimental alignment hashes. Empty = derived from the seed (reproducible order).",
       nullable=True, null_label="derived from seed", widget="salt", advanced=True),
    _p("psi_group_bits", "int", 2048, "align",
       "Recorded for the run log only: DH-PSI (two- and multi-party) always uses the RFC 3526 2048-bit group "
       "(not read by the code).",
       choices=(2048,), read=False, advanced=True, show_if=(("alignment", ("dh_psi",)),)),
    _p("codec", "enum", "fp32", "align", "Wire encoding of representations and gradients.", choices=CODECS,
       labels=(("fp32", "fp32 (exact)"), ("fp16", "fp16"), ("int8", "int8 (per-row scale)"), ("topk", "top-k sparse"))),
    _p("topk_fraction", "float", 0.25, "align", "top-k codec: fraction of entries kept per row.",
       presets=(0.05, 0.1, 0.25, 0.5, 0.75), min=0, max=1, min_exclusive=True, show_if=(("codec", ("topk",)),)),

    # ------------------------------------------------------------------ network & timing
    _p("inject_rtt_ms", "float", 0.0, "network", "Application-level RTT emulated at passive parties.",
       presets=(0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0), min=0, unit="ms"),
    _p("rpc_deadline_s", "float", 30.0, "network", "Per-RPC deadline before a retry.",
       presets=(5.0, 10.0, 30.0, 60.0, 120.0, 300.0), min=0, min_exclusive=True, unit="s"),
    _p("max_retries", "int", 3, "network", "Retries per RPC on UNAVAILABLE / DEADLINE_EXCEEDED.",
       presets=(0, 1, 2, 3, 5, 10), min=0),
    _p("retry_backoff_s", "float", 0.2, "network", "Linear back-off step between retries.",
       presets=(0.05, 0.1, 0.2, 0.5, 1.0), min=0, unit="s", advanced=True),
    _p("slot_timeout_s", "float", 120.0, "network", "How long the active party waits for every passive activation.",
       presets=(30.0, 60.0, 120.0, 300.0, 600.0), min=0, min_exclusive=True, unit="s"),
    _p("barrier_timeout_s", "float", 300.0, "network", "Epoch-barrier timeout at the coordinator.",
       presets=(60.0, 120.0, 300.0, 600.0, 1800.0), min=0, min_exclusive=True, unit="s"),

    # ------------------------------------------------------------------ defences
    _p("defense.type", "enum", "none", "defense",
       "Gradient perturbation on dL/dH (clip + Gaussian noise). Not a formal end-to-end DP guarantee.",
       choices=("none", "grad_noise")),
    _p("defense.clip", "float", 0.05, "defense", "Per-row L2 clip C on dL/dH (rows scale like 1/batch_size).",
       presets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0), min=0, min_exclusive=True,
       show_if=(("defense.type", ("grad_noise",)),)),
    _p("defense.noise_multiplier", "float", 1.0, "defense", "Gaussian noise std as a multiple of C.",
       presets=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0), min=0, show_if=(("defense.type", ("grad_noise",)),)),
    _p("defense.delta", "float", 1e-5, "defense", "delta for the (eps, delta) report of the gradient channel.",
       presets=(1e-6, 1e-5, 1e-4), min=0, max=1, min_exclusive=True, advanced=True,
       show_if=(("defense.type", ("grad_noise",)),)),
    _p("defense.label_rr_epsilon", "float", None, "defense",
       "Randomized response on training labels: the ONLY formal guarantee here (the trained model and the "
       "gradients sent to passives are eps-label-DP; validation/test metrics use true labels).",
       presets=(0.5, 1.0, 2.0, 3.0, 4.0, 8.0), nullable=True, null_label="off", min=0, min_exclusive=True),
    _p("defense.rep_noise_sigma", "float", 0.0, "defense",
       "Passive-side representation noise (clip to rep_clip, add N(0, (sigma*rep_clip)^2)); no formal guarantee.",
       presets=(0.0, 0.01, 0.05, 0.1, 0.5, 1.0), min=0),
    _p("defense.rep_clip", "float", 1.0, "defense", "Row clip applied before representation noise.",
       presets=(0.5, 1.0, 2.0, 5.0), min=0, min_exclusive=True,
       show_if=(("!", "defense.rep_noise_sigma", (0.0,)),)),
    _p("defense.noise_source", "enum", "secret", "defense",
       "Where defence randomness (gradient noise, randomized-response flips, representation noise) comes from. "
       "secret: a secret each party draws locally and never sends. seed: v2 behaviour, derived from the shared "
       "session seed - reproducible, but every party can recompute and REMOVE the noise, so it is not a defence.",
       choices=("secret", "seed"), labels=(("secret", "secret per party (private)"),
                                           ("seed", "session seed (v2; NOT private)")),
       advanced=True, since="v2.1"),
    _p("attack_eval", "bool", True, "defense",
       "Evaluator-side label-leakage metrics (norm / cosine attack AUC) on the gradients actually sent.",
       widget="toggle"),

    # ------------------------------------------------------------------ straggler
    _p("straggler.policy", "enum", "wait", "straggler",
       "wait = every batch waits for all parties (v1); drop = proceed without parties that miss the deadline.",
       choices=("wait", "drop")),
    _p("straggler.slot_deadline_s", "float", 5.0, "straggler", "drop policy: per-batch wait at the active party.",
       presets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0), min=0, min_exclusive=True, unit="s",
       show_if=(("straggler.policy", ("drop",)),)),

    # ------------------------------------------------------------------ fault injection
    _p("fault.duplicate", "pairs", [], "fault",
       "Passive parties re-send the message for these (epoch, batch) slots; the active party must not step twice.",
       widget="pairs"),
    _p("fault.delay", "pairs", [], "fault", "The active party delays its reply for these (epoch, batch) slots.",
       widget="pairs"),
    _p("fault.delay_s", "float", 2.0, "fault", "Length of each injected reply delay.",
       presets=(0.5, 1.0, 2.0, 5.0, 10.0), min=0, unit="s"),

    # ------------------------------------------------------------------ resume & data checks
    _p("resume_from", "str", None, "data", "Resume a past session from its last complete epoch.",
       nullable=True, null_label="new run", widget="session"),
    _p("partition", "str", None, "data",
       "Refuse to start unless every party's manifest reports this split strategy (empty = don't check).",
       presets=PARTITION_STRATEGIES, nullable=True, null_label="don't check"),
    _p("overlap", "float", None, "data", "Refuse to start unless the manifests report this entity overlap.",
       presets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0), nullable=True, null_label="don't check", min=0, max=1,
       min_exclusive=True),
    _p("binarize", "str", None, "data", "Refuse to start unless the active party's labels used this binarize rule.",
       presets=BINARIZE_RULES, nullable=True, null_label="don't check"),
    _p("participants", "str_list", None, "data", "Parties to include. Empty = every live party.",
       nullable=True, null_label="all live parties", widget="parties", advanced=True),
    _p("min_passive", "int", 1, "data", "Minimum live passive parties needed to start.",
       presets=(1, 2, 3, 4, 5, 6, 7, 8), min=1, advanced=True),
    _p("checkpoint", "bool", True, "data", "Save a checkpoint at every party after each epoch (needed to resume).",
       widget="toggle"),
    _p("start_epoch", "int", 0, "data", "First epoch of this run (set by the coordinator when resuming).",
       min=0, internal=True, in_defaults=False),

    # ------------------------------------------------------------------ tensor validation
    _p("security.validate_tensors", "bool", True, "security", "Shape + finiteness checks on every H and dL/dH.",
       widget="toggle"),
    _p("security.max_row_norm", "float", None, "security",
       "Reject representation rows with a larger L2 norm (active party).",
       presets=(1.0, 5.0, 10.0, 50.0, 100.0), nullable=True, null_label="no bound", min=0, min_exclusive=True),
    _p("security.max_grad_row_norm", "float", None, "security",
       "Reject gradient rows with a larger L2 norm (passive parties).",
       presets=(0.01, 0.1, 1.0, 10.0), nullable=True, null_label="no bound", min=0, min_exclusive=True),
    _p("security.private_init", "bool", False, "security",
       "Passive parties initialise their bottom models from a local secret instead of the shared seed, so the active "
       "party cannot compute their starting weights (which helps it invert representations). Off keeps runs "
       "reproducible from the seed.", widget="toggle", since="v2.1"),
    _p("security.anomaly_detection", "bool", False, "security",
       "Flag near-constant / outlier representations (free-riders, poisoning).", widget="toggle"),
    _p("security.max_align_items", "int", None, "security",
       "Most ids a party may submit for alignment. Empty = 4 x the active party's rows + 10000 (PSI makes every "
       "party work per submitted id, and each one is a membership query).", presets=(10000, 100000, 1000000, 5000000),
       nullable=True, null_label="automatic", min=1, advanced=True, since="v2.3"),
    # ------------------------------------------------------------------ passive guard (v2.3): malicious active
    _p("security.passive_grad_clip", "float", None, "security",
       "Passive parties clip every received gradient row to this L2 norm (bounds how much a malicious active party "
       "can move them per step; defeats scaled / targeted gradients).", presets=(0.01, 0.1, 1.0, 10.0),
       nullable=True, null_label="off", min=0, min_exclusive=True, since="v2.3"),
    _p("security.max_param_drift", "float", None, "security",
       "Passive bottom models stay within this relative L2 distance of their initialisation (a trust region: no "
       "gradient sequence can steer them further).", presets=(0.5, 1.0, 2.0, 5.0, 10.0), nullable=True,
       null_label="off", min=0, min_exclusive=True, advanced=True, since="v2.3"),
    _p("security.grad_monitor", "enum", "off", "security",
       "Passive parties flag gradients whose norm jumps far above the running trend; stop = leave the session "
       "after 3 consecutive flags.", choices=("off", "warn", "stop"), since="v2.3"),
    _p("security.leakage_audit", "enum", "off", "security",
       "After each epoch every passive measures how well its own raw features can be reconstructed (ridge "
       "decoder) from the representations it sent; warn or stop above leakage_r2_max.",
       choices=("off", "warn", "stop"), since="v2.3"),
    _p("security.leakage_r2_max", "float", 0.9, "security",
       "Largest acceptable feature-reconstruction R^2 for the leakage audit (1 = features fully recoverable).",
       presets=(0.3, 0.5, 0.7, 0.8, 0.9, 0.95), min=0, max=1,
       show_if=(("security.leakage_audit", ("warn", "stop")),), since="v2.3"),

    # ------------------------------------------------------------------ runtime
    _p("torch_threads", "int", 1, "runtime", "PyTorch CPU threads per party process.",
       presets=(1, 2, 4, 8), min=1),
    _p("deterministic", "bool", True, "runtime", "Deterministic kernels so CPU runs repeat exactly.", widget="toggle"),
    _p("telemetry_flush_s", "float", 2.0, "runtime", "How often parties flush batch telemetry to the coordinator.",
       presets=(0.5, 1.0, 2.0, 5.0), min=0, min_exclusive=True, unit="s", advanced=True),
)

BY_KEY: dict[str, Param] = {p.key: p for p in PARAMS}
SECTIONS = sorted({p.key.split(".")[0] for p in PARAMS if "." in p.key})


# ---------------------------------------------------------------- nested <-> dotted
def get_path(cfg: dict, key: str, missing=KeyError):
    node = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            if missing is KeyError:
                raise KeyError(key)
            return missing
        node = node[part]
    return node


def set_path(cfg: dict, key: str, value):
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def default_session() -> dict:
    out: dict = {}
    for p in PARAMS:
        if p.in_defaults:
            set_path(out, p.key, copy.deepcopy(p.default))
    return out


def leaf_keys(d: dict, prefix: str = "") -> list[str]:
    """Dotted keys of a nested override dict. Section dicts (defense, fault, ...) are walked; values that are
    themselves dicts (a bottom_model mapping) stay one leaf."""
    out = []
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and key in SECTIONS:
            out += leaf_keys(v, key + ".")
        else:
            out.append(key)
    return out


# ---------------------------------------------------------------- validation
def _is_num(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return v == v                                      # not NaN
    if isinstance(v, str):                                 # YAML reads `1e-3` as a string; v2 accepted it
        try:
            float(v)
            return True
        except ValueError:
            return False
    return False


def check_value(p: Param, v) -> str | None:
    """None if `v` is valid for `p`, else a short reason."""
    if v is None:
        return None if p.nullable else "must be set"
    t = p.type
    if t == "bool":
        return None if isinstance(v, bool) else "must be true or false"
    if t == "int":
        if isinstance(v, bool) or not isinstance(v, int):
            return "must be an integer"
    elif t == "float":
        if not _is_num(v):
            return "must be a number"
    elif t == "str":
        if not isinstance(v, str) or not v.strip():
            return "must be a non-empty string"
    elif t == "enum":
        return None if v in p.choices else f"must be one of {', '.join(map(str, p.choices))}"
    elif t == "int_list":
        if not isinstance(v, (list, tuple)) or not 1 <= len(v) <= 6 or \
                any(isinstance(c, bool) or not isinstance(c, int) or c <= 0 for c in v):
            return "must be a list of 1-6 positive integers"
        return None
    elif t == "pairs":
        if not isinstance(v, (list, tuple)) or any(
                not isinstance(x, (list, tuple)) or len(x) != 2 or
                any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in x) for x in v):
            return "must be a list of [epoch, batch] pairs of non-negative integers"
        return None
    elif t == "bottom_map":
        kinds = list(v.values()) if isinstance(v, dict) else [v]
        if not kinds or any(k not in BOTTOM_MODELS for k in kinds) or \
                (isinstance(v, dict) and any(not isinstance(k, str) or not k for k in v)):
            return f"must be one of {BOTTOM_MODELS} or a mapping party -> kind"
        return None
    elif t == "str_list":
        if not isinstance(v, (list, tuple)) or not v or any(not isinstance(s, str) or not s for s in v):
            return "must be a non-empty list of party ids"
        return None
    if p.choices and v not in p.choices:
        return f"must be one of {', '.join(map(str, p.choices))}"
    if t in ("int", "float"):
        x = float(v)
        if p.min is not None and (x < p.min or (p.min_exclusive and x == p.min)):
            return f"must be {'>' if p.min_exclusive else '>='} {p.min:g}"
        if p.max is not None and x > p.max:
            return f"must be <= {p.max:g}"
        if p.odd and int(v) % 2 == 0:
            return "must be odd"
    return None


def validate(cfg: dict) -> list[str]:
    """Per-key schema errors for a full session config (missing keys use their defaults)."""
    errs = []
    for p in PARAMS:
        v = get_path(cfg, p.key, missing=_MISSING)
        if v is _MISSING:
            continue
        r = check_value(p, v)
        if r:
            errs.append(f"{p.key} {r}")
    return errs


_MISSING = object()


def suggest(key: str) -> str:
    m = difflib.get_close_matches(key, list(BY_KEY), n=1)
    return f" (did you mean {m[0]!r}?)" if m else ""


def validate_overrides(overrides: dict) -> list[str]:
    """Check user overrides (--set / form / YAML): unknown keys, system-only keys and bad values."""
    errs = []
    for key in leaf_keys(overrides):
        p = BY_KEY.get(key)
        if p is None:
            errs.append(f"unknown config key {key!r}{suggest(key)}")
            continue
        if p.internal:
            errs.append(f"{key} is set by the system and cannot be overridden"
                        + (" (use --resume-from)" if key == "start_epoch" else ""))
            continue
        r = check_value(p, get_path(overrides, key))
        if r:
            errs.append(f"{key} {r}")
    return errs


# ---------------------------------------------------------------- form <-> config
def form_params() -> list[Param]:
    return [p for p in PARAMS if not p.internal]


def coerce(p: Param, v):
    """Normalise a value coming from JSON (JS sends 2 for 2.0)."""
    if v is None:
        return None
    if p.type == "float" and _is_num(v) and not isinstance(v, str):
        return float(v)
    if p.type == "int_list":
        return [int(c) for c in v]
    if p.type == "pairs":
        return [[int(a), int(b)] for a, b in v]
    if p.type == "str_list":
        return list(v)
    return v


def form_values(cfg: dict) -> dict:
    """Flat {dotted_key: value} for every form field, taken from `cfg` (defaults where missing). This is what
    the dashboard pre-fills (config_form.js `initialValues` computes the same map; a test checks they agree)."""
    out = {}
    for p in form_params():
        v = get_path(cfg, p.key, missing=_MISSING)
        out[p.key] = copy.deepcopy(p.default if v is _MISSING else v)
    return out


def config_from_form(form: dict) -> dict:
    """Nested session config from a submitted form. Every field must be a schema key; values are validated."""
    if not isinstance(form, dict):
        raise ValueError("form must be an object of {key: value}")
    errs, out = [], {}
    for key, v in form.items():
        p = BY_KEY.get(key)
        if p is None:
            errs.append(f"unknown config key {key!r}{suggest(key)}")
            continue
        if p.internal:
            errs.append(f"{key} is set by the system")
            continue
        try:
            v = coerce(p, v)
        except (TypeError, ValueError):
            errs.append(f"{key} has the wrong shape")
            continue
        r = check_value(p, v)
        if r:
            errs.append(f"{key} {r}")
            continue
        set_path(out, key, v)
    if errs:
        raise ValueError("invalid form: " + "; ".join(errs))
    return out


def option_values(p: Param) -> list:
    """Every value the form offers for a select-type field (presets + choices + default + None if nullable)."""
    vals = []
    if p.nullable:
        vals.append(None)
    for v in tuple(p.choices) + tuple(p.presets) + (p.default,):
        if v is None:
            continue
        if not any(json.dumps(v) == json.dumps(w) for w in vals):
            vals.append(v)
    return vals


# ---------------------------------------------------------------- export
def schema_json() -> dict:
    return {"version": SCHEMA_VERSION, "groups": [{"id": g, "label": lab} for g, lab in GROUPS],
            "bottom_models": list(BOTTOM_MODELS), "params": [p.to_json() for p in PARAMS if not p.internal],
            "defaults": form_values(default_session())}


def _fmt(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v
    return json.dumps(v)


def render_markdown() -> str:
    """docs/CONFIG.md: generated, do not edit by hand (`python -m vfl.cli config docs > docs/CONFIG.md`)."""
    lines = ["# Session configuration reference", "",
             "<!-- generated from vfl/config_schema.py by `python -m vfl.cli config docs`; do not edit by hand -->", "",
             "Every key the training code reads. Set them in `configs/session.yaml`, with "
             "`python -m vfl.cli start --set key=value`, or in the coordinator dashboard's session form "
             "(dropdowns of the presets below). Presets are the dropdown values, not limits: "
             "`--set` accepts any value that passes the type and range check.", "",
             ] + render_table().splitlines() + ["", HONEST_NOTE, ""]
    return "\n".join(lines)


HONEST_NOTE = ("Honest limits: only `defense.label_rr_epsilon` gives a formal (end-to-end) label-DP guarantee. "
               "`defense.type: grad_noise` reports an (eps, delta) for the dL/dH channel only. "
               "`alignment: experimental` is non-private. The text encoder is chosen when the data is prepared "
               "(`prepare-data --text-encoder`, `embed-text --encoder`), not here.")
README_START = "<!-- config-table:start (generated by `python -m vfl.cli config docs --readme README.md`) -->"
README_END = "<!-- config-table:end -->"


def render_table() -> str:
    """The hyperparameter table alone (README.md embeds it between README_START / README_END)."""
    lines = ["| Key | Type | Default | Dropdown presets | Applies to | Description |",
             "|---|---|---|---|---|---|"]
    for gid, label in GROUPS:
        rows = [p for p in PARAMS if p.group == gid]
        if not rows:
            continue
        lines.append(f"| **{label}** | | | | | |")
        for p in rows:
            typ = {"bottom_map": "kind or {party: kind}", "int_list": "list of int", "pairs": "list of [epoch, batch]",
                   "str_list": "list of party ids"}.get(p.type, p.type) + (" or null" if p.nullable else "")
            rng = []
            if p.min is not None:
                rng.append(f"{'>' if p.min_exclusive else '>='} {p.min:g}")
            if p.max is not None:
                rng.append(f"<= {p.max:g}")
            if p.odd:
                rng.append("odd")
            if rng:
                typ += " (" + ", ".join(rng) + ")"
            opts = p.choices or p.presets
            pre = ", ".join(f"`{_fmt(v)}`" for v in opts) if opts else {
                "toggle": "toggle", "pairs": "epoch/batch rows", "party_models": "one dropdown per party",
                "session": "past sessions", "parties": "detected parties", "salt": "derived / random",
                "name": "suggested names", "dataset": "datasets in manifests"}.get(p.widget, "")
            notes = p.help.replace("|", "\\|")
            if p.internal:
                notes += " *(system-set; not in the form)*"
            if not p.read:
                notes += " *(not read by the code)*"
            if p.since != "v1":
                notes += f" *(new in {p.since}; default = previous behaviour)*"
            applies = ", ".join(p.applies_to) or "all"
            for c in p.show_if:
                neg, k, vals = (True, c[1], c[2]) if c[0] == "!" else (False, c[0], c[1])
                applies += f"; when {k} {'≠' if neg else '='} {'/'.join(_fmt(v) for v in vals)}"
            lines.append(f"| `{p.key}` | {typ} | `{_fmt(p.default)}` | {pre} | {applies} | {notes} |")
    return "\n".join(lines)


def update_readme(path: str) -> bool:
    """Replace the README block between README_START and README_END with the current table."""
    text = open(path).read()
    if README_START not in text or README_END not in text:
        raise ValueError(f"{path} has no config-table markers")
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    new = head + README_START + "\n" + render_table() + "\n" + README_END + tail
    if new != text:
        with open(path, "w") as fh:
            fh.write(new)
    return new != text
