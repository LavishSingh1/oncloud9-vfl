"""Standalone attack simulator (v2.1 item 1): the one library behind every entry point.

    python -m vfl.cli attack all --lab-mode --coordinator HOST:50050 --active HOST:50051 \\
        --certs certs/ --config deployment.json --out results/attacks/

Needs only gRPC endpoints, certificates and a target config; no dashboard or Flask process. The Security tab
(coordinator --lab-mode) and the unified UI's Security & Attacks page both run this same command through
vfl/attacks/jobs.py, so the three entry points produce the same verdicts for the same attack.

Safeguards (all three must hold before anything is sent):
  * --lab-mode         explicit acknowledgement that the target is your own lab deployment
  * target allowlist   coordinator / active / enroll hosts must appear in the target config (`hosts`, or the
                       addresses it lists). Without --config only loopback targets are allowed.
  * UI token           the two UI entry points only launch for an operator (dashboard token / login)

Outputs (in --out, a folder; a path ending in .json writes report.json there and the rest beside it):
  report.json     every case: attack, layer, expected outcome, observed outcome, as-expected, evidence source
  report.csv      the same rows as a table
  summary.txt     PASS / FAIL line + counts
  security.jsonl  one attacker-side record per case (same schema as the nodes' security.jsonl)
Evidence: every row is "attacker-reported" (what the probe saw). If the target config lists the nodes'
log folders (`log_dirs`, as the Deployment panel's deployment.json does) the suite also reads their
security.jsonl and marks rows "server-observed" when a node logged a matching event during that case.
"""
from __future__ import annotations

import csv
import glob
import ipaddress
import json
import os
import time

from . import CASES, FAMILIES

LOOPBACK = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
FIELDS = ["attack", "family", "layer", "expected", "observed", "as_expected", "evidence", "server_observed",
          "method", "detail"]


class Refused(Exception):
    """A safeguard refused the run (exit code 2)."""


def resolve(case: str) -> list[str]:
    names = list(CASES) if case == "all" else list(FAMILIES.get(case) or [case])
    bad = [n for n in names if n not in CASES]
    if bad:
        raise Refused(f"unknown case(s) {bad}; choose from {list(CASES)}, a family {list(FAMILIES)} or 'all'")
    return names


def load_targets(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(text) or {}
    return json.loads(text)


def host_of(addr: str | None) -> str | None:
    if not addr:
        return None
    a = addr.strip()
    if a.startswith("["):                                        # [::1]:50050
        return a[1:a.index("]")]
    if a.count(":") == 1:
        return a.split(":")[0]
    try:
        ipaddress.ip_address(a)
        return a
    except ValueError:
        return a.rsplit(":", 1)[0]


def allowed_hosts(targets: dict) -> set:
    if not targets:
        return set(LOOPBACK)
    hs = {str(h) for h in targets.get("hosts") or []}
    for k in ("coordinator", "active", "enroll"):
        h = host_of(targets.get(k))
        if h:
            hs.add(h)
    return hs


def check_targets(a, targets: dict):
    allowed = allowed_hosts(targets)
    for label, addr in (("coordinator", a.coordinator), ("active", a.active), ("enroll", getattr(a, "enroll", None))):
        h = host_of(addr)
        if h and h not in allowed:
            raise Refused(f"refusing to target {label} {addr}: host {h!r} is not in the session's target config "
                          f"({sorted(allowed)}). Pass --config with your lab deployment's hosts.")


def apply_targets(a, targets: dict):
    """Fill options the user left at their defaults from the target config (e.g. the Deployment panel's
    deployment.json). Explicit CLI values always win."""
    for key in ("coordinator", "active"):
        if targets.get(key) and getattr(a, key, None) in (None, "", _DEFAULTS[key]):
            setattr(a, key, targets[key])
    for key in ("enroll", "session", "data_dir"):
        if targets.get(key) and not getattr(a, key, None):
            setattr(a, key, targets[key])
    if targets.get("certs") and getattr(a, "certs", None) in (None, "", "certs"):
        a.certs = targets["certs"]


_DEFAULTS = {"coordinator": "localhost:50050", "active": "localhost:50051"}


def _family(case):
    return next((f for f, cs in FAMILIES.items() if case in cs), "")


def _server_events(log_dirs, t0, t1) -> list[dict]:
    out = []
    for d in log_dirs or []:
        for path in glob.glob(os.path.join(d, "**", "security.jsonl"), recursive=True):
            try:
                with open(path) as fh:
                    for line in fh:
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        if r.get("evidence") != "attacker-reported" and t0 - 0.5 <= float(r.get("t", 0)) <= t1 + 2.0:
                            out.append(r)
            except OSError:
                continue
    return out


def enrich(results: list[dict], log_dirs=None) -> list[dict]:
    from .runner import _layer
    rows = []
    for r in results:
        srv = _server_events(log_dirs, r.get("t_start", 0), r.get("t_end", 0)) if log_dirs else []
        types = sorted({e.get("attack_type") for e in srv if e.get("attack_type")})
        rows.append({"attack": r["case"], "family": _family(r["case"]), "layer": _layer(r["case"]),
                     "expected": r["expected"], "observed": r["outcome"], "as_expected": bool(r["as_expected"]),
                     "evidence": "server-observed + attacker-reported" if types else "attacker-reported",
                     "server_observed": types, "method": r.get("method"), "detail": r.get("detail", ""),
                     "t_start": r.get("t_start"), "t_end": r.get("t_end"),
                     **({"metric": r["metric"]} if "metric" in r else {})})
    return rows


def out_paths(out: str | None) -> dict:
    if not out:
        return {}
    if out.endswith(".json"):
        d = os.path.dirname(os.path.abspath(out)) or "."
        stem = os.path.splitext(os.path.basename(out))[0]
        return {"dir": d, "json": os.path.abspath(out), "csv": os.path.join(d, stem + ".csv"),
                "summary": os.path.join(d, stem + ".summary.txt"), "jsonl": os.path.join(d, "security.jsonl")}
    d = os.path.abspath(out)
    return {"dir": d, "json": os.path.join(d, "report.json"), "csv": os.path.join(d, "report.csv"),
            "summary": os.path.join(d, "summary.txt"), "jsonl": os.path.join(d, "security.jsonl")}


def write_outputs(report: dict, out: str | None) -> dict:
    p = out_paths(out)
    if not p:
        return {}
    os.makedirs(p["dir"], exist_ok=True)
    with open(p["json"], "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(p["csv"], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in report["rows"]:
            w.writerow({**r, "server_observed": ";".join(r["server_observed"]), "as_expected": "yes" if r["as_expected"] else "NO"})
    sm = report["summary"]
    with open(p["summary"], "w") as fh:
        fh.write(f"{'PASS' if sm['all_as_expected'] else 'FAIL'}: {sm['as_expected']}/{sm['cases']} cases as designed "
                 f"(blocked {sm['blocked']}, detected {sm['detected']}, succeeded {sm['succeeded']}, "
                 f"skipped {sm['skipped']}, error {sm['error']})\n")
        for r in report["rows"]:
            fh.write(f"  {'ok ' if r['as_expected'] else 'BAD'} {r['attack']:<24} expected {r['expected']:<9} "
                     f"observed {r['observed']:<9} [{r['evidence']}]\n")
    from ..telemetry import SecurityLog, now_iso
    log = SecurityLog("lab-attacker", p["jsonl"])
    for r in report["rows"]:
        log.ingest({"t": r.get("t_end") or time.time(), "time": now_iso(), "node": "lab-attacker", "source_ip": None,
                    "claimed_identity": None, "cert_identity": None, "attack_type": r["attack"], "layer": r["layer"],
                    "outcome": r["observed"] if r["observed"] in ("blocked", "detected", "succeeded") else "detected",
                    "defence": "", "evidence": "attacker-reported", "method": r.get("method"),
                    "session": report.get("session"),
                    "detail": f"expected {r['expected']}, observed {r['observed']}: {r['detail']}"[:500]})
    if log._fh:
        log._fh.close()
    return p


def run(a) -> dict:
    """Run the suite for parsed CLI args `a`. Raises Refused when a safeguard says no."""
    if not getattr(a, "lab_mode", False):
        raise Refused("refusing to run: pass --lab-mode to confirm this is your own test deployment")
    targets = load_targets(getattr(a, "config", None))
    apply_targets(a, targets)
    names = resolve(a.case)
    check_targets(a, targets)
    from .runner import Harness
    t0 = time.time()
    raw = Harness(a).run(names)
    rows = enrich(raw["results"], targets.get("log_dirs") or getattr(a, "log_dirs", None))
    counts = {k: sum(r["observed"] == k for r in rows) for k in ("blocked", "detected", "succeeded", "reported",
                                                                  "skipped", "error")}
    ok = sum(r["as_expected"] for r in rows)
    report = {"tool": "vfl.attacks", "lab_only": True, "case": a.case, "started": t0, "wall_s": time.time() - t0,
              "targets": {"coordinator": a.coordinator, "active": a.active, "enroll": getattr(a, "enroll", None),
                          "allowed_hosts": sorted(allowed_hosts(targets))},
              "session": getattr(a, "session", None), "rows": rows, "results": raw["results"],
              "verdicts": {r["attack"]: {"observed": r["observed"], "as_expected": r["as_expected"]} for r in rows},
              "summary": {"cases": len(rows), "as_expected": ok, "all_as_expected": ok == len(rows),
                          "by_case": {r["attack"]: r["observed"] for r in rows}, **counts,
                          "inference_succeeded": sum(r["observed"] == "succeeded" and r["family"] == "inference"
                                                     for r in rows)},
              "honest_note": "Inference cases measure what an AUTHORISED party can learn from tensors it legitimately "
                             "receives; mTLS does not prevent it. Only label_rr_epsilon gives a formal guarantee."}
    report["files"] = write_outputs(report, getattr(a, "out", None))
    return report
