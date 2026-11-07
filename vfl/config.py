"""Session configuration: defaults, YAML loading, overrides and validation (FR-9).

Everything that changes an experiment (dataset tag, seed, batch size, model
dimensions, epochs, RTT injection, codec, defence, faults) lives here so that no
source edit is needed between runs.
"""
from __future__ import annotations

import copy
import json
from typing import Any

import yaml

from . import config_schema as schema

# Generated from vfl/config_schema.py (the single source of truth for keys, defaults, presets and help).
# tests/test_config_schema.py pins the v2 defaults so a schema edit cannot silently change a v1/v2 run.
DEFAULT_SESSION: dict[str, Any] = schema.default_session()


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_value(text: str) -> Any:
    """Parse a CLI override value using YAML rules (numbers, bools, lists)."""
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, sets: list[str] | None) -> dict:
    """Apply `a.b=c` style overrides."""
    cfg = copy.deepcopy(cfg)
    for item in sets or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parse_value(val)
    return cfg


def parse_sets(sets: list[str] | None) -> dict:
    """`--set` strings -> nested override dict (no defaults merged)."""
    return apply_overrides({}, sets)


def check_overrides(overrides: dict) -> None:
    """Refuse unknown keys, system-only keys and bad values in user overrides (schema-driven)."""
    errs = schema.validate_overrides(overrides)
    if errs:
        raise ValueError("invalid override: " + "; ".join(errs))


def load_session_config(path: str | None = None, sets: list[str] | None = None,
                        extra: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_SESSION)
    if path:
        with open(path) as fh:
            loaded = yaml.safe_load(fh) or {}
        cfg = deep_merge(cfg, loaded.get("session", loaded))
    if extra:
        cfg = deep_merge(cfg, extra)
    if sets:
        check_overrides(parse_sets(sets))
    cfg = apply_overrides(cfg, sets)
    validate_session_config(cfg)
    return cfg


# keys the code indexes with cfg[...] (a missing one is an error, not a silent default)
REQUIRED = ("epochs", "batch_size", "embed_dim", "eval_batch_size", "hidden_dim", "lr", "seed", "top_hidden",
            "alignment", "codec", "optimizer")


def validate_session_config(cfg: dict) -> None:
    """Per-key checks come from the schema; cross-field checks live here. Keys a v2 config lacks (e.g. the
    v2.1 conv options) are fine: the code reads them with defaults equal to the previous behaviour."""
    errs = [f"{k} is missing" for k in REQUIRED if k not in cfg]
    errs += schema.validate(cfg)
    sp = cfg.get("split", {})
    try:
        if not (0 < sp.get("test", 0) < 1 and 0 <= sp.get("val", 0) < 1 and sp["val"] + sp["test"] < 1):
            errs.append("split.val/test must be fractions with val+test < 1")
    except (TypeError, KeyError):
        errs.append("split.val/test must be fractions with val+test < 1")
    # ids that become file paths at every party (checkpoints/<party>/<resume_from>/...): v2.2 accepted any string,
    # so an absolute path made each party torch.load() a file of the operator's choosing (RT-02)
    from .ids import check_party_id, check_session_id
    if cfg.get("resume_from") is not None:
        try:
            check_session_id(cfg["resume_from"])
        except ValueError as e:
            errs.append(f"resume_from: {e}")
    for pid in cfg.get("participants") or []:
        try:
            check_party_id(pid)
        except ValueError as e:
            errs.append(f"participants: {e}")
    if errs:
        raise ValueError("invalid session config: " + "; ".join(errs))


def alignment_salt(cfg: dict) -> str:
    return cfg.get("alignment_salt") or f"vfl-align-v1:{cfg['seed']}"


def dumps(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)
