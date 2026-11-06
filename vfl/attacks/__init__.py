"""Security conformance suite (PRD v2 item 14; standalone since v2.1 item 1). LAB USE ONLY - it runs
against the team's own test deployment to confirm the live security gate (vfl/transport.py) and the
honest-but-curious privacy accounting behave as designed. It mints no new capability: every case is a client
that sends a request the gate should refuse, then records whether the gate refused it.

    python -m vfl.cli attack all --lab-mode --coordinator HOST:50050 --active HOST:50051 \
        --certs certs/ --config deployment.json --out results/attacks/

No dashboard is needed (vfl/attacks/suite.py has the outputs and safeguards). With --url/--token it also posts
each observation to the coordinator's security feed. The coordinator's Security tab and the unified UI's
Security & Attacks page run this same command (vfl/attacks/jobs.py).

Each case returns {outcome: blocked|detected|succeeded, ...}. "blocked" is the expected, healthy result for the
network/auth/protocol/enrollment cases: the gate rejected the request. The inference cases measure what an
authorised party could already infer from tensors it legitimately receives - mTLS does not prevent that.

Case families: auth (missing/other-CA/expired/mismatched certificate), protocol (non-participant, wrong
session, replay, wrong digest, non-finite tensor), enrollment (absent/reused/expired token), inference
(label-leakage, membership) and deviating-party (free-rider, scaled representation) detection.
"""
from __future__ import annotations

import os

CASES = ("no_cert", "other_ca_cert", "expired_cert", "mismatched_cert", "non_participant",
         "unknown_session", "replay", "wrong_digest", "non_finite_tensor",
         "no_token", "reused_token", "expired_token",
         "label_leakage", "membership", "free_rider", "scaled_representation")
FAMILIES = {"auth": CASES[0:4], "protocol": CASES[4:9], "enrollment": CASES[9:12],
            "inference": CASES[12:14], "deviating_party": CASES[14:16]}


def add_attack_parser(sub, handler):
    p = sub.add_parser("attack", help="LAB-ONLY security conformance suite (see vfl/attacks)")
    p.add_argument("case", help="a case name, a family (auth|protocol|enrollment|inference|deviating_party), or 'all'")
    p.add_argument("--lab-mode", action="store_true", required=False,
                   help="required acknowledgement that this runs against your own test deployment")
    p.add_argument("--config", help="target config (JSON/YAML): coordinator, active, enroll, hosts, log_dirs, certs - "
                   "e.g. the Deployment panel's deployment.json. Targets outside it are refused; without it only "
                   "loopback targets are allowed")
    p.add_argument("--log-dirs", nargs="*", help="node log folders to read security.jsonl from (server-observed evidence)")
    p.add_argument("--url", default=None, help="optional coordinator dashboard URL (posts attacker observations "
                   "to its security feed; not needed for the suite)")
    p.add_argument("--token", default=None, help="dashboard operator token (with --url)")
    p.add_argument("--coordinator", default="localhost:50050")
    p.add_argument("--active", default="localhost:50051")
    p.add_argument("--certs", default="certs", help="the deployment's trust dir (for a valid-but-wrong certificate)")
    p.add_argument("--lab-certs", help="dir with test-only certificates (other CA, expired, extra identities)")
    p.add_argument("--data-dir", help="per-party data, for the inference cases (experimenter view)")
    p.add_argument("--session", help="a running session id, for protocol/inference cases")
    p.add_argument("--enroll", help="enrollment address, for the enrollment cases")
    p.add_argument("--out", help="output folder (report.json, report.csv, summary.txt, security.jsonl); "
                   "a path ending in .json writes the report there and the rest beside it")
    p.set_defaults(fn=handler)


def run_attack_cli(a) -> int:
    import json
    from .suite import Refused, run
    try:
        report = run(a)
    except Refused as e:
        print(str(e), flush=True)
        return 2
    print(json.dumps(report["summary"], indent=2))
    for r in report["rows"]:
        print(f"  {'ok ' if r['as_expected'] else 'BAD'} {r['attack']:<24} {r['layer']:<11} expected {r['expected']:<9} "
              f"observed {r['observed']:<9} [{r['evidence']}]")
    if report.get("files"):
        print(f"report: {report['files']['json']} (+ {os.path.basename(report['files']['csv'])}, "
              f"{os.path.basename(report['files']['summary'])}, security.jsonl)")
    return 0 if report["summary"]["all_as_expected"] else 1
