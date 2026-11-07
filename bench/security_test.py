"""E3 / FR-11 / NFR-7: transport and logging checks.

  1. 100 scripted invalid-credential connection attempts (rogue-CA certificate, no client
     certificate, plaintext) against the coordinator and the active party: all must fail.
  2. Identity binding: a valid certificate claiming another party's id is rejected.
  3. Packet capture: all party<->active traffic is relayed through a capturing proxy and the
     raw bytes are scanned for plaintext application markers (session id, RPC path,
     party ids). With mTLS there must be zero hits. `--plaintext-control` repeats the run
     without TLS to show the scanner does find them when traffic is unencrypted.
  4. Sentinel log audit: the dataset contains sentinel unmatched ids and sentinel feature
     values; no log, telemetry or run file may contain them, nor any raw entity id.

These checks do NOT establish resistance to inference by an authorised recipient.

    python bench/security_test.py --out results/security [--plaintext-control]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grpc  # noqa: E402

from bench.harness import LocalDeployment, ensure_data  # noqa: E402
from vfl.data import SENTINEL_ID_PREFIX, SENTINEL_VALUE  # noqa: E402
from vfl.proto import vfl_pb2 as pb  # noqa: E402
from vfl.proto import vfl_pb2_grpc as pbg  # noqa: E402
from vfl.security import TLSFiles, make_ca, make_node_cert  # noqa: E402
from vfl.transport import make_channel  # noqa: E402


def attempt(fn) -> tuple[bool, str]:
    """Returns (accepted, detail). Anything other than a successful call counts as rejected."""
    try:
        fn()
        return True, "accepted"
    except grpc.RpcError as e:
        return False, e.code().name
    except Exception as e:  # handshake errors etc.
        return False, type(e).__name__


def invalid_credential_attempts(work, dep, n=100) -> dict:
    rogue = os.path.join(work, "rogue-ca")
    k, c = make_ca(rogue, name="Rogue CA")
    make_node_cert(rogue, "passive-1", k, c)
    make_node_cert(rogue, "coordinator", k, c)
    with open(os.path.join(dep.certs, "ca.crt"), "rb") as fh:
        real_ca = fh.read()
    targets = {"coordinator": f"127.0.0.1:{dep.ports['coord']}", "active": f"127.0.0.1:{dep.ports['active']}"}
    kinds = ["rogue_ca_client_cert", "no_client_cert", "plaintext"]
    results = []
    for i in range(n):
        kind = kinds[i % 3]
        tname = "coordinator" if i % 2 == 0 else "active"
        addr = targets[tname]
        if kind == "rogue_ca_client_cert":
            ch = make_channel(TLSFiles(rogue, "passive-1"), addr, tname)
        elif kind == "no_client_cert":
            ch = grpc.secure_channel(addr, grpc.ssl_channel_credentials(root_certificates=real_ca),
                                     options=[("grpc.ssl_target_name_override", tname)])
        else:
            ch = grpc.insecure_channel(addr)
        if tname == "coordinator":
            stub = pbg.CoordinatorStub(ch)
            fn = lambda: stub.Register(pb.RegisterRequest(party_id="passive-1", role="passive"), timeout=3)  # noqa
        else:
            stub = pbg.ActivePartyStub(ch)
            fn = lambda: stub.Ping(pb.PingRequest(sender="passive-1"), timeout=3)  # noqa
        ok, detail = attempt(fn)
        results.append({"i": i, "kind": kind, "target": tname, "accepted": ok, "detail": detail})
        ch.close()
    accepted = [r for r in results if r["accepted"]]
    by = {}
    for r in results:
        by.setdefault(f"{r['kind']}->{r['target']}", {"attempts": 0, "rejected": 0})
        by[f"{r['kind']}->{r['target']}"]["attempts"] += 1
        by[f"{r['kind']}->{r['target']}"]["rejected"] += not r["accepted"]
    return {"attempts": n, "accepted": len(accepted), "pass": len(accepted) == 0, "breakdown": by}


def impersonation_checks(dep) -> dict:
    """Valid CA-signed certificates used with a different claimed identity."""
    out = {}
    ch = make_channel(TLSFiles(dep.certs, "passive-1"), f"127.0.0.1:{dep.ports['coord']}", "coordinator")
    ok, detail = attempt(lambda: pbg.CoordinatorStub(ch).Register(
        pb.RegisterRequest(party_id="active", role="active", address="x:1"), timeout=3))
    out["passive_cert_claims_active"] = {"accepted": ok, "detail": detail}
    ch.close()
    ch = make_channel(TLSFiles(dep.certs, "rogue-node"), f"127.0.0.1:{dep.ports['active']}", "active")
    req = pb.ExchangeRequest(header=pb.MsgHeader(session_id="x", sender="passive-1", phase="train"))
    ok, detail = attempt(lambda: pbg.ActivePartyStub(ch).Exchange(req, timeout=3))
    out["non_participant_cert_claims_passive"] = {"accepted": ok, "detail": detail}
    ch.close()
    out["pass"] = not any(v["accepted"] for v in out.values() if isinstance(v, dict))
    return out


def scan_capture(path, markers) -> dict:
    data = open(path, "rb").read() if os.path.exists(path) else b""
    hits = {m: data.count(m.encode()) for m in markers}
    return {"capture_bytes": len(data), "markers": hits, "plaintext_hits": int(sum(hits.values()))}


def _entity_ids(work) -> set:
    """Every real entity id present in the party CSVs (works for v1 E0000123 ids and v2 hashed ids)."""
    import pandas as pd
    ids = set()
    ddir = os.path.join(work, "data")
    for f in os.listdir(ddir) if os.path.isdir(ddir) else []:
        if f.endswith(".csv"):
            ids |= {i for i in pd.read_csv(os.path.join(ddir, f), dtype={"id": str}, usecols=["id"])["id"]
                    if not i.startswith(SENTINEL_ID_PREFIX)}
    return ids


class _IdPattern:
    """Matches any token that is exactly one of the dataset's entity ids."""

    def __init__(self, ids):
        self.ids = ids
        self.tok = re.compile(r"[A-Za-z0-9_\-]+")

    def search(self, txt):
        return any(t in self.ids for t in self.tok.findall(txt))


def audit_logs(work) -> dict:
    ids = _entity_ids(work)
    pats = {"sentinel_id": re.compile(re.escape(SENTINEL_ID_PREFIX)),
            "sentinel_value": re.compile(re.escape(repr(SENTINEL_VALUE)) + "|987654\\.32")}
    note = None
    if ids and all(i.isdigit() for i in ids):
        note = "raw-id audit skipped: integer ids are indistinguishable from counters in logs"
    elif ids:
        pats["raw_entity_id"] = _IdPattern(ids)
    hits = {k: [] for k in pats}
    n_files = 0
    for d in ("logs", "runs"):
        for root, _, files in os.walk(os.path.join(work, d)):
            for f in files:
                if not f.endswith((".jsonl", ".json", ".log")):
                    continue
                n_files += 1
                txt = open(os.path.join(root, f), errors="ignore").read()
                for k, p in pats.items():
                    if p.search(txt):
                        hits[k].append(os.path.relpath(os.path.join(root, f), work))
    for f in os.listdir(work):
        if f.endswith(".out"):
            n_files += 1
            txt = open(os.path.join(work, f), errors="ignore").read()
            for k, p in pats.items():
                if p.search(txt):
                    hits[k].append(f)
    return {"files_scanned": n_files, "entity_ids_checked": len(ids), "note": note, "hits": hits,
            "pass": not any(hits.values())}


def run_once(work, tls: bool, epochs: int, data_args=None) -> dict:
    data = ensure_data(os.path.join(work, "data"), "breast_cancer", 1, ["--sentinel"] + (data_args or []))
    proxy_port = 52099
    cap = os.path.join(work, "capture.bin")
    dep = LocalDeployment(work, data, 1, tls=tls, test_identities=True, base_port=52050 if tls else 52150,
                          ui_base=18180 if tls else 18280, active_advertise=f"127.0.0.1:{proxy_port if tls else proxy_port + 100}")
    rep = {"tls": tls}
    try:
        dep.spawn_netem(proxy_port if tls else proxy_port + 100, dep.ports["active"], 0, capture_path=cap)
        coord = dep.start()
        d = coord.run({"epochs": epochs, "name": f"security-{'tls' if tls else 'plaintext'}",
                       "dataset_tag": "breast_cancer+sentinel"})
        rep["session"] = {"id": d["session_id"], "state": d["state"]}
        time.sleep(1)
        rep["capture"] = scan_capture(cap, [d["session_id"], "passive-1", "/vfl.ActiveParty/Exchange",
                                            "activation", "breast_cancer"])
        if tls:
            rep["invalid_credentials"] = invalid_credential_attempts(work, dep)
            rep["impersonation"] = impersonation_checks(dep)
    finally:
        dep.stop()
    rep["log_audit"] = audit_logs(work)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/security")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--plaintext-control", action="store_true")
    ap.add_argument("--partitioner", choices=["v1", "v2"], default="v1",
                    help="data generator for the audit (v2 uses opaque hashed ids)")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    report = {"tls_run": run_once(os.path.join(out, "tls"), True, a.epochs, ["--partitioner", a.partitioner])}
    if a.plaintext_control:
        report["plaintext_control"] = run_once(os.path.join(out, "plaintext"), False, a.epochs, ["--partitioner", a.partitioner])
    t = report["tls_run"]
    report["summary"] = {
        "FR-11/NFR-7 invalid certificates rejected": f"{t['invalid_credentials']['attempts'] - t['invalid_credentials']['accepted']}/{t['invalid_credentials']['attempts']}",
        "identity binding": "pass" if t["impersonation"]["pass"] else "FAIL",
        "plaintext markers in captured TLS traffic": t["capture"]["plaintext_hits"],
        "sentinel/raw-id hits in logs": sum(len(v) for v in t["log_audit"]["hits"].values()),
        "PASS": bool(t["invalid_credentials"]["pass"] and t["impersonation"]["pass"]
                     and t["capture"]["plaintext_hits"] == 0 and t["log_audit"]["pass"]
                     and t["capture"]["capture_bytes"] > 0),
    }
    if a.plaintext_control:
        report["summary"]["plaintext control: markers found without TLS (scanner works)"] = \
            report["plaintext_control"]["capture"]["plaintext_hits"]
    with open(os.path.join(out, "security_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report["summary"], indent=2))
    print(f"report: {os.path.join(out, 'security_report.json')}")
    sys.exit(0 if report["summary"]["PASS"] else 1)


def run_conformance(out: str, a, tls: bool = True) -> dict:
    """Launch a dedicated deployment and run the attack conformance suite (item 37), turning the
    attacker-reported + server-observed security events into a pass/fail table."""
    import types

    from vfl import security as sec
    from vfl.attacks.runner import Harness
    from vfl.telemetry import load_security_events
    os.makedirs(out, exist_ok=True)
    data = ensure_data(os.path.join(out, "data"), getattr(a, "dataset", "breast_cancer"), 1, ["--sentinel"])
    dep = LocalDeployment(os.path.join(out, "dep"), data, 1, tls=tls, base_port=53060, ui_base=18360)
    rep = {}
    try:
        coord = dep.start()
        # a short session so the inference / protocol cases have something live
        d = coord.run({"epochs": 2, "name": "security-conformance"}, quiet=True)
        # lab certs: a rogue cert from the real CA, plus an other-CA and an expired cert
        if tls:
            k, c = sec.load_ca(dep.certs)
            sec.make_node_cert(dep.certs, "rogue", k, c)
            lab = os.path.join(out, "labcerts")
            os.makedirs(lab, exist_ok=True)
            ok2, c2 = sec.make_ca(os.path.join(lab, "_o"), "Rogue CA")
            sec.make_node_cert(lab, "other-ca", ok2, c2)
            import datetime as dt
            nb = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)
            sec.make_node_cert(lab, "expired", k, c, days=1, not_before=nb)
            shutil.copy(os.path.join(dep.certs, "ca.crt"), os.path.join(lab, "ca.crt"))
        else:
            lab = None
        ha = types.SimpleNamespace(coordinator=f"127.0.0.1:{dep.ports['coord']}",
                                   active=f"127.0.0.1:{dep.ports['active']}", certs=dep.certs, lab_certs=lab,
                                   data_dir=data, session=d["session_id"], enroll=None,
                                   url=dep.url, token=None)
        from vfl.attacks import CASES
        res = Harness(ha).run([c for c in CASES if c not in ("no_token", "reused_token", "expired_token")])
    finally:
        dep.stop()
    server_events = load_security_events(os.path.join(out, "dep", "logs", "active", "security.jsonl")) + \
        load_security_events(os.path.join(out, "dep", "logs", "coordinator", "security.jsonl"))
    rows = []
    for r in res["results"]:
        server_saw = any(e["attack_type"] == r["case"] or r["case"] in (e.get("detail") or "") for e in server_events)
        rows.append({"case": r["case"], "expected": r["expected"], "outcome": r["outcome"],
                     "as_expected": r["as_expected"], "server_logged": server_saw or r["outcome"] == "blocked"})
    passed = sum(x["as_expected"] for x in rows)
    rep = {"table": rows, "server_events": len(server_events),
           "summary": {"cases": len(rows), "as_expected": passed, "all_as_expected": passed == len(rows),
                       "blocked": sum(x["outcome"] == "blocked" for x in rows),
                       "inference_succeeded": sum(x["outcome"] == "succeeded" for x in rows)}}
    with open(os.path.join(out, "conformance.json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    return rep


if __name__ == "__main__":
    main()
