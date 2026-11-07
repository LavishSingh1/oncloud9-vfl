"""Command-line entry point:  python -m vfl.cli <command> [options]

  certs         generate the private CA and per-node mTLS certificates
  prepare-data  build per-party CSVs from a public/synthetic dataset (vertical split)
  coordinator   run the session-management node (gRPC + operator dashboard)
  active        run the active (label-owning) party
  passive       run a passive (feature-owning) party
  start         start a session through the coordinator's HTTP API (--set keys are checked against the schema)
  config        session-config schema: docs table, JSON schema, defaults, validate a config + --set
  ui            unified UI (port 8090): every command above from one page
  attack-ui     the unified UI on its Security & Attacks page (lab mode)
  attack        LAB-ONLY security conformance suite (no dashboard needed)
  stop / status / wait   control and observe sessions
  baseline      run B0 centralized / B1 active-only / local VFL references
  netem         TCP relay that adds RTT and counts wire bytes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def _http(method: str, url: str, body: dict | None = None, token: str | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode().replace("NaN", "null"))
    except urllib.error.HTTPError as e:
        msg = e.read().decode()
        try:
            msg = json.loads(msg).get("error", msg)
        except Exception:
            pass
        raise SystemExit(f"error: {e.code} {msg}")
    except urllib.error.URLError as e:
        raise SystemExit(f"error: cannot reach {url}: {e.reason}")


def _addr_default(v, port):
    return v if ":" in v else f"{v}:{port}"


def cmd_certs(a):
    from . import security as sec
    nodes = ["coordinator", a.active_id] + [f"passive-{i + 1}" for i in range(a.passives)] + (a.extra or [])
    if a.mode == "static":
        files = sec.generate_all(a.out, nodes, a.hosts, roles={a.active_id: "active", "coordinator": "coordinator"})
        print(f"wrote {len(files)} files to {a.out}: {', '.join(files)}")
        print("Copy ca.crt plus <node>.crt/<node>.key to each machine (never copy ca.key).")
        return
    if os.path.exists(os.path.join(a.out, "pki.json")):
        print(f"{a.out} already holds a PKI; issuing node certificates only")
    else:
        meta = sec.make_pki(a.out, a.domain)
        print(f"root CA fingerprint (SHA-256): {meta['root_sha256']}")
    issued = []
    for n in ([] if a.enroll_only else nodes[1:]):
        if n == "coordinator":
            continue
        role = "active" if n == a.active_id else "passive"
        issued.append(sec.issue_static_pki(a.out, n, role, a.hours))
    with open(os.path.join(a.out, "issued_static.json"), "w") as fh:
        json.dump(issued, fh, indent=2)
    print(f"pki in {a.out}: ca.crt (trust anchor), intermediate.crt/.key (coordinator only), "
          f"{len(issued)} node certificate(s) valid {a.hours:g} h")
    print(f"IMPORTANT: move {os.path.join(a.out, 'root', 'ca.key')} OFFLINE now; it is only needed to make a new "
          "intermediate.")
    print("Parties can instead enroll with a token: python -m vfl.cli token create --role passive")


def cmd_prepare(a):
    from .data import V1_DATASETS, V1_STRATEGIES, V2_DATASETS, prepare_v2, prepare_vertical_dataset
    n_passive = (a.parties - 1) if a.parties else a.passives
    if n_passive < 1:
        raise SystemExit("error: need at least one passive party (--parties >= 2)")
    if a.partitioner == "v1":
        bad = []
        if a.dataset not in V1_DATASETS:
            bad.append(f"dataset {a.dataset!r}")
        if a.partition not in V1_STRATEGIES:
            bad.append(f"strategy {a.partition!r}")
        for flag, val in (("--binarize", a.binarize), ("--grid", a.grid), ("--columns-json", a.columns_json),
                          ("--evaluate", a.evaluate), ("--informative-party", a.informative_party)):
            if val:
                bad.append(flag)
        if bad:
            raise SystemExit(f"error: {', '.join(bad)} need(s) the v2 partitioner: add --partitioner v2")
        ratios = [float(x) for x in a.ratios.split(",")] if a.ratios else None
        overlap = 0.9 if a.overlap is None else a.overlap
        rep = prepare_vertical_dataset(a.out, a.dataset, n_passive, overlap, a.partition, a.seed, ratios,
                                       a.n, a.features, a.csv, a.label, a.sentinel)
        for nm, p in rep["parties"].items():
            print(f"{nm:>10}: {p['rows']:>6} rows x {p['n_features']:>3} features "
                  f"(MI sum {p['mutual_info_sum']:.3f}){' + labels' if p['has_labels'] else ''} -> {p['file']}")
        print(f"expected aligned entities: {rep['n_aligned_expected']} ({rep['overlap_ratio_actual']:.1%} of {rep['n_total']})")
        print(f"report: {os.path.join(a.out, 'data_report.json')}")
    else:
        if a.dataset not in V2_DATASETS:
            raise SystemExit(f"error: unknown dataset {a.dataset!r}")
        if a.ratios:
            raise SystemExit("error: --ratios is v1-only; with v2 use --partition dirichlet --alpha A or manual")
        columns = None
        if a.columns_json:
            with open(a.columns_json) as fh:
                columns = json.load(fh)
        grid = tuple(int(v) for v in a.grid.lower().split("x")) if a.grid else None
        overlap = 1.0 if a.overlap is None else a.overlap
        try:
            if a.dataset in ("npz", "synthetic_sensors", "har", "multimodal_digits"):
                m = _prepare_special(a, n_passive + 1, overlap, grid)
            else:
                m = None
        except (ValueError, FileNotFoundError, ImportError) as e:
            raise SystemExit(f"error: {e}")
        try:
            m = m or prepare_v2(a.out, a.dataset, n_passive + 1, a.partition, a.seed,
                           1.0 if a.overlap is None else a.overlap, a.binarize, a.id_style, a.alpha, a.top_frac,
                           a.informative_party, columns, grid, not a.no_one_hot, a.n, a.features, a.csv, a.label,
                           a.sentinel, a.evaluate)
        except ValueError as e:
            raise SystemExit(f"error: {e}")
        print(f"{m['dataset']}: {m['entities_total']} entities -> {m['n_parties']} parties ({m['strategy']}), "
              f"intersection {m['intersection_size']} ({m['intersection_ratio']:.0%}), {m['num_classes']} classes")
        for p in m["parties"]:
            extra = f"  importance {p['importance_share']:.0%}" if "importance_share" in p else ""
            shape = f"  shape {p['shape']}" if "shape" in p else ""
            print(f"  {p['name']:<11} rows {p['rows']:>6}  inputs {p['model_input_dim']:>4}{shape}{extra}")
        if "reference_auroc" in m:
            print("  reference AUROC (logistic regression):", m["reference_auroc"])
        if m["num_classes"] > 2:
            print(f"note: {m['num_classes']} classes -> cross-entropy + macro one-vs-rest AUROC "
                  "(norm label-leakage attack is binary-only; the cosine attack still applies)")
        if m.get("recommended_bottom_model"):
            print(f"recommended: --set 'bottom_model={json.dumps(m['recommended_bottom_model'])}'")
        print(f"full manifest (keep with the experimenter): {os.path.join(a.out, 'manifest.json')}")
    print("give each party ONLY its <name>.csv|.npz and <name>.manifest.json")


def _prepare_special(a, n_parties, overlap, grid):
    from . import datasets as D
    fmt = a.format
    if a.dataset == "npz":
        if not a.npz:
            raise ValueError("--dataset npz needs --npz FILE (arrays x/y, e.g. CIFAR-10 saved with np.savez)")
        X, y = D.load_npz_images(a.npz)
        strategy = a.partition if a.partition in D.IMAGE_STRATEGIES else "image_strips"
        return D.prepare_images(a.out, X, y, os.path.splitext(os.path.basename(a.npz))[0], n_parties, strategy,
                                a.seed, overlap, a.binarize, a.id_style, grid, fmt or "npz", a.sentinel)
    if a.dataset in ("synthetic_sensors", "har"):
        if a.dataset == "har":
            if not a.har_dir:
                raise ValueError("--dataset har needs --har-dir 'path/to/UCI HAR Dataset'")
            X, y, names, streams = D.load_har(a.har_dir)
        else:
            X, y, names, streams = D.synthetic_sensors(a.n if a.n != 20000 else 3000, a.channels, a.timesteps,
                                                       a.classes, a.seed)
        if a.binarize:
            X, y = D._binarize_arrays(X, y, a.binarize)
        return D.prepare_sensors(a.out, X, y, names, streams, a.dataset + (f"[{a.binarize}]" if a.binarize else ""),
                                 n_parties, a.seed, overlap, a.id_style, fmt or "csv", a.sentinel)
    if n_parties != 3:
        raise ValueError("multimodal_digits always has 3 parties (image, tabular, text)")
    return D.prepare_multimodal_digits(a.out, a.seed, overlap, a.binarize, a.id_style, a.text_encoder, a.sentinel)


def cmd_embed_text(a):
    from .datasets import embed_text_file
    try:
        info = embed_text_file(a.csv, a.text_col, a.out, a.encoder, a.dim)
    except ImportError as e:
        raise SystemExit(f"error: {e}. Install the optional encoder packages (requirements-optional.txt) "
                         "or use --encoder hashing (offline)")
    print(f"wrote {a.out} with {info['dim']}-d embeddings from {info['encoder']} (frozen; only the projection "
          "layer after it is trained)")


def _logging(node, a):
    from .telemetry import setup_logging
    setup_logging(node, os.path.join(a.log_dir, node), a.log_level)


LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")


def _tls(a):
    if a.insecure:
        host = (getattr(a, "listen", None) or "127.0.0.1").rsplit(":", 1)[0]
        if host not in LOOPBACK and not getattr(a, "allow_insecure_remote", False):
            # without TLS there is no identity at all: anyone who can reach the port can register as any party
            raise SystemExit(f"refusing --insecure on {a.listen}: without mTLS anyone on the network can join as "
                             "any party. Listen on 127.0.0.1, or add --allow-insecure-remote for a throw-away lab")
        print("WARNING: running WITHOUT TLS (local development only)", file=sys.stderr)
        return None
    if not os.path.exists(os.path.join(a.tls_dir, "ca.crt")):
        raise SystemExit(f"no certificates in {a.tls_dir}; run `python -m vfl.cli certs` or pass --insecure")
    return a.tls_dir


def _node_opts(a) -> dict:
    return {"rate_limit_rps": a.rate_limit_rps, "max_peer_concurrency": a.max_peer_concurrency,
            "max_concurrent_rpcs": a.max_concurrent_rpcs, "lab_mode": a.lab_mode, "ui_users": a.ui_users,
            "ui_tls": a.ui_tls, "log_dir": a.log_dir, "ui_public_views": a.ui_public_views}


def cmd_coordinator(a):
    _logging("coordinator", a)
    from .coordinator import serve
    o = _node_opts(a)
    o.update({"join_mode": a.join_mode, "enroll_listen": a.enroll_listen, "cert_hours": a.cert_hours,
              "lab_certs": a.lab_certs})
    serve(a.listen, a.ui_host, a.ui_port, _tls(a), a.run_dir, a.config, a.ui_token, o)


def _maybe_enroll(a, role: str):
    """pki mode: obtain a certificate before starting (key generated locally)."""
    if not (a.enroll_token or a.join_approval):
        return
    from .enrollment import enroll_party, fetch_root
    crt = os.path.join(a.tls_dir, f"{a.id}.crt")
    if os.path.exists(crt) and not a.re_enroll:
        print(f"{crt} exists; skipping enrollment (use --re-enroll to replace it)")
        return
    if not a.enroll:
        raise SystemExit("--enroll COORD_HOST:50049 is required for enrollment")
    try:
        root = fetch_root(a.enroll_url, a.ca_file, a.enroll_token, a.ca_sha256)
        info = enroll_party(a.tls_dir, a.id, role, a.enroll, root, a.enroll_token, log=print)
    except Exception as e:
        raise SystemExit(f"enrollment failed: {e}")
    print(f"enrolled {a.id} as {info['role']}: serial {info['serial']}, valid until "
          f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(info['not_after']))}")


def cmd_active(a):
    _logging(a.id, a)
    _maybe_enroll(a, "active")
    from .active import serve
    advertise = a.advertise or a.listen.replace("0.0.0.0", "localhost")
    o = _node_opts(a)
    o["enroll_addr"] = a.enroll
    o["malicious"] = a.malicious
    serve(a.id, a.data, _addr_default(a.coordinator, 50050), a.listen, advertise, _tls(a), a.ui_host, a.ui_port,
          a.log_dir, a.ckpt_dir, a.ui_token, a.public_ui_url, a.manifest, o)


def cmd_passive(a):
    _logging(a.id, a)
    _maybe_enroll(a, "passive")
    from .passive import serve
    o = _node_opts(a)
    o.update({"enroll_addr": a.enroll, "malicious": a.malicious})
    serve(a.id, a.data, _addr_default(a.coordinator, 50050), _tls(a), a.ui_host, a.ui_port, a.log_dir,
          a.ckpt_dir, a.ui_token, a.public_ui_url, a.manifest, o)


def cmd_token(a):
    body = {"role": a.role, "name": a.name, "ttl_s": a.ttl}
    r = _http("POST", f"{a.url.rstrip('/')}/api/pki/tokens", body, a.token)
    print(r["token"])
    print(f"# single-use, role={r['role']}{', name=' + r['name'] if r.get('name') else ''}, expires "
          f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['expires']))}", file=sys.stderr)


def cmd_ui_user(a):
    import getpass
    from .dashboard import add_user
    pw = a.password or getpass.getpass(f"password for {a.name}: ")
    if len(pw) < 8:
        raise SystemExit("password must be at least 8 characters")
    add_user(a.file, a.name, pw, a.role)
    print(f"saved {a.name} ({a.role}) to {a.file}; start nodes with --ui-users {a.file}")


def cmd_attack(a):
    from .attacks import run_attack_cli
    sys.exit(run_attack_cli(a))


def cmd_bench_ui(a):
    from bench.ui import serve as bench_serve
    bench_serve(a.results, a.runs, a.host, a.port, a.token)


def cmd_ui(a):
    from .webui import serve
    serve(a.workspace, a.host, a.port, a.token, a.results, a.runs, a.coordinator_url, a.coordinator_token,
          a.lab_mode, getattr(a, "page", "dataprep"), a.session_config)


def _session_overrides(a) -> dict:
    """YAML overrides + --set, checked against the config schema before anything is sent (item 2).
    --set: unknown keys / bad values are errors. YAML: bad values are errors, unknown keys only warn
    (a v2 YAML with an extra key keeps working)."""
    from . import config_schema as schema
    from .config import apply_overrides, deep_merge, parse_sets
    import yaml
    cfg = {}
    if a.config:
        with open(a.config) as fh:
            loaded = yaml.safe_load(fh) or {}
        cfg = loaded.get("session", loaded)
        errs = schema.validate_overrides(cfg)
        unknown = [e for e in errs if e.startswith("unknown config key")]
        for e in unknown:
            print(f"warning: {a.config}: {e} (ignored by the training code)", file=sys.stderr)
        bad = [e for e in errs if e not in unknown]
        if bad:
            raise SystemExit(f"error: {a.config}: " + "; ".join(bad))
    errs = schema.validate_overrides(parse_sets(a.set))
    if errs:
        raise SystemExit("error: " + "; ".join(errs) + "\n(see `python -m vfl.cli config docs` for every key)")
    return apply_overrides(deep_merge({}, cfg), a.set)


def cmd_config(a):
    """Schema tools: docs table, JSON schema, defaults, and local validation of a config + --set."""
    from . import config_schema as schema
    from .config import DEFAULT_SESSION, load_session_config
    import yaml
    if a.action == "docs":
        text = schema.render_markdown()
        if getattr(a, "readme", None):
            changed = schema.update_readme(a.readme)
            print(f"{'updated' if changed else 'already current'}: hyperparameter table in {a.readme}")
            if not a.out:
                return
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(text)
            print(f"wrote {a.out}")
        else:
            print(text, end="")
    elif a.action == "schema":
        print(json.dumps(schema.schema_json(), indent=2))
    elif a.action == "defaults":
        print(yaml.safe_dump({"session": DEFAULT_SESSION}, sort_keys=False), end="")
    else:
        _session_overrides(a)                               # unknown keys / bad values in --set
        try:
            cfg = load_session_config(a.config, a.set)
        except ValueError as e:
            raise SystemExit(f"error: {e}")
        print(json.dumps(cfg, indent=2, default=str))
        print("valid", file=sys.stderr)


def cmd_start(a):
    overrides = _session_overrides(a)
    resume = a.resume_from or overrides.pop("resume_from", None)   # --set resume_from=X works too
    overrides.pop("resume_from", None)
    body = {"config": overrides, "resume_from": resume}
    r = _http("POST", f"{a.url.rstrip('/')}/api/sessions", body, a.token)
    print(json.dumps(r, indent=2))
    if a.wait:
        _wait(a.url, r["session_id"], a.timeout)


def cmd_stop(a):
    print(json.dumps(_http("POST", f"{a.url.rstrip('/')}/api/sessions/{a.session}/stop", {}, a.token), indent=2))


def cmd_status(a):
    st = _http("GET", f"{a.url.rstrip('/')}/api/state")
    print("parties:")
    for p in st["parties"]:
        print(f"  {p['party_id']:<14} {p['role']:<8} {'alive' if p['alive'] else 'DEAD ':<6} {p['status']:<8} "
              f"{p.get('hostname', '')}")
    print("sessions:")
    for s in st["sessions"][:15]:
        print(f"  {s['session_id']:<28} {s['state']:<10} epochs {s['completed_epochs']}/{s['epochs']} "
              f"bs={s['batch_size']} rtt={s.get('inject_rtt_ms')} test_auroc={s.get('last_test_auroc')}")


def _wait(url, sid, timeout):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        d = _http("GET", f"{url.rstrip('/')}/api/sessions/{sid}")
        msg = f"{d['state']} epochs {d.get('completed_epochs')}/{d['config']['epochs']}"
        if msg != last:
            print(f"[{time.strftime('%H:%M:%S')}] {sid}: {msg}", flush=True)
            last = msg
        if d["state"] in ("COMPLETED", "FAILED", "STOPPED"):
            return d
        time.sleep(1.0)
    raise SystemExit(f"timeout waiting for {sid}")


def cmd_wait(a):
    d = _wait(a.url, a.session, a.timeout)
    print(json.dumps({k: d.get(k) for k in ("session_id", "state", "completed_epochs", "errors")}, indent=2))


def cmd_baseline(a):
    from .config import load_session_config
    from .local_sim import run_baselines, save
    cfg = load_session_config(a.config, a.set)
    which = tuple(a.which.split(","))
    res = run_baselines(a.data_dir, cfg, which)
    save(res, a.out)
    for k in ("b0_centralized", "b1_active_only", "local_vfl"):
        if k in res:
            t = res[k]["final_test"]
            print(f"{k:>16}: test AUROC {t['auroc']:.4f}  acc {t['accuracy']:.4f}  F1 {t['f1']:.4f}  "
                  f"params {res[k]['params']}")
    print(f"saved {a.out}")


def cmd_netem(a):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [netem] %(message)s")
    from .netem_proxy import run
    run(a.listen, a.target, a.rtt_ms, a.stats, a.capture)


def build_parser() -> argparse.ArgumentParser:
    """The full CLI parser. The unified UI builds prepare-data/certs/attack argv against it, and
    tests/test_webui.py walks its subcommands to check every one has a UI route."""
    ap = argparse.ArgumentParser(prog="python -m vfl.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("certs", help="generate CA + node certificates")
    p.add_argument("--mode", default="static", choices=["static", "pki"],
                   help="static: one CA, long-lived certs (v1); pki: offline root + name-constrained intermediate, "
                        "short-lived certs with role in the SAN")
    p.add_argument("--domain", default="oncloud9", help="pki trust domain")
    p.add_argument("--hours", type=float, default=24 * 30, help="pki: validity of hand-distributed node certs")
    p.add_argument("--enroll-only", action="store_true", help="pki: create the CA only; parties enroll with tokens")
    p.add_argument("--out", default="certs")
    p.add_argument("--passives", type=int, default=1)
    p.add_argument("--active-id", default="active")
    p.add_argument("--extra", nargs="*", help="additional node ids (e.g. rogue-test)")
    p.add_argument("--hosts", nargs="*", default=[], help="extra IPs/hostnames for SANs (optional)")
    p.set_defaults(fn=cmd_certs)

    p = sub.add_parser("prepare-data", help="create vertically partitioned per-party CSVs + manifests")
    p.add_argument("--partitioner", default="v1", choices=["v1", "v2"],
                   help="v1 = original generator (default, reproduces earlier results); v2 = vfl/partition.py")
    p.add_argument("--dataset", default="breast_cancer",
                   help="v1: breast_cancer | synthetic | csv;  v2 adds digits | adult | bank | credit_default | "
                        "mnist | fashion_mnist (the last five download from openml.org)")
    p.add_argument("--csv"); p.add_argument("--label")
    p.add_argument("--out", default="data")
    p.add_argument("--passives", type=int, default=1, help="number of passive parties")
    p.add_argument("--parties", type=int, help="total parties incl. the active one (overrides --passives)")
    p.add_argument("--overlap", type=float, default=None, help="entity overlap (default: v1 0.9, v2 1.0)")
    p.add_argument("--partition", default="random",
                   help="v1: random | sequential | importance_balanced | importance_skewed;  "
                        "v2: sequential | random | dirichlet | importance | correlated | manual | image_strips | "
                        "image_grid (importance_skewed is accepted as importance at passive-1)")
    p.add_argument("--ratios", help="v1: feature share per party, active first, e.g. 0.4,0.6")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=20000, help="rows for synthetic")
    p.add_argument("--features", type=int, default=40, help="features for synthetic")
    p.add_argument("--sentinel", action="store_true", help="add sentinel rows for the log-audit test")
    g = p.add_argument_group("v2 partitioner options")
    g.add_argument("--binarize", help="image datasets: odd_even | one_vs_rest:K | pair:A,B")
    g.add_argument("--id-style", default="hash", choices=["hash", "int"])
    g.add_argument("--alpha", type=float, default=1.0, help="dirichlet concentration (small = uneven)")
    g.add_argument("--top-frac", type=float, default=0.5, help="importance: fraction of top features")
    g.add_argument("--informative-party", type=int, help="importance: 0=active, 1=passive-1, ...")
    g.add_argument("--columns-json", help="manual: JSON list of column lists, one per party")
    g.add_argument("--grid", help="image_grid: ROWSxCOLS, e.g. 2x2")
    g.add_argument("--no-one-hot", action="store_true")
    g.add_argument("--evaluate", action="store_true", help="add per-party reference AUROC to manifest.json")
    g.add_argument("--format", choices=["csv", "npz"], help="party file format (npz default for --dataset npz)")
    g.add_argument("--npz", help="--dataset npz: image file with arrays x and y (e.g. CIFAR-10)")
    g.add_argument("--har-dir", help="--dataset har: path to the extracted 'UCI HAR Dataset' folder")
    g.add_argument("--channels", type=int, default=6, help="synthetic_sensors: channels (3 per sensor stream)")
    g.add_argument("--timesteps", type=int, default=64, help="synthetic_sensors: window length")
    g.add_argument("--classes", type=int, default=4, help="synthetic_sensors: activity classes")
    g.add_argument("--text-encoder", default="hashing", choices=["minilm", "distilbert", "hashing"],
                   help="multimodal_digits: frozen text encoder at the text party")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("embed-text", help="party-side: frozen text encoder -> embedding columns")
    p.add_argument("--csv", required=True, help="party CSV with an id column and a text column")
    p.add_argument("--text-col", default="text")
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", default="minilm", choices=["minilm", "distilbert", "hashing"])
    p.add_argument("--dim", type=int, default=64, help="hashing encoder: SVD dimensions")
    p.set_defaults(fn=cmd_embed_text)

    def node_common(p, ui_port):
        p.add_argument("--tls-dir", default="certs")
        p.add_argument("--insecure", action="store_true", help="disable mTLS (local dev only, loopback only)")
        p.add_argument("--allow-insecure-remote", action="store_true",
                       help="LAB ONLY: allow --insecure on a non-loopback address (anyone can join)")
        p.add_argument("--ui-host", default="127.0.0.1", help="0.0.0.0 to expose the dashboard on the LAN")
        p.add_argument("--ui-port", type=int, default=ui_port, help="0 disables the dashboard")
        p.add_argument("--ui-token", default=os.environ.get("VFL_UI_TOKEN"),
                       help="bearer token required for dashboard POST actions")
        p.add_argument("--log-dir", default="logs")
        p.add_argument("--log-level", default="INFO")
        p.add_argument("--public-ui-url", help="dashboard URL other machines should use, e.g. "
                       "http://192.168.1.30:8082 (shown as the 'open' link on the coordinator)")
        g = p.add_argument_group("hardening (v2)")
        g.add_argument("--rate-limit-rps", type=float, help="per-peer request rate limit on the gRPC port")
        g.add_argument("--max-peer-concurrency", type=int, default=16,
                       help="concurrent calls per peer identity (certificate name; IP without TLS); 0 = no cap")
        g.add_argument("--max-concurrent-rpcs", type=int, default=256, help="server-wide concurrent call cap")
        g.add_argument("--ui-users", help="users file for dashboard logins (vfl.cli ui-user add)")
        g.add_argument("--ui-tls", action="store_true", help="serve the dashboard over HTTPS with the node cert")
        g.add_argument("--ui-public-views", action="store_true",
                       help="show dashboard views without the token even when --ui-host is not loopback (v2.2 "
                            "behaviour, trusted demo networks only)")
        g.add_argument("--lab-mode", action="store_true",
                       help="LAB ONLY: enable the attack simulator / malicious-party switches")


    def enroll_args(p):
        g = p.add_argument_group("enrollment (pki mode)")
        g.add_argument("--enroll", help="coordinator enrollment address, e.g. 192.168.1.10:50049")
        g.add_argument("--enroll-token", help="single-use join token from the coordinator")
        g.add_argument("--join-approval", action="store_true", help="ask the operator to approve this party")
        g.add_argument("--enroll-url", help="coordinator dashboard URL to fetch the root CA (checked against the token)")
        g.add_argument("--ca-file", help="root CA certificate obtained out of band")
        g.add_argument("--ca-sha256", help="root CA SHA-256 fingerprint (pins a root fetched with --enroll-url)")
        g.add_argument("--re-enroll", action="store_true", help="replace an existing certificate")

    p = sub.add_parser("coordinator", help="run the coordinator / session manager")
    p.add_argument("--listen", default="0.0.0.0:50050")
    p.add_argument("--config", default="configs/session.yaml", help="default session config")
    p.add_argument("--run-dir", default="runs")
    p.add_argument("--join-mode", default="static", choices=["static", "token", "approval"],
                   help="pki: how new parties obtain certificates")
    p.add_argument("--enroll-listen", default="0.0.0.0:50049", help="pki: enrollment service address ('' = off)")
    p.add_argument("--cert-hours", type=float, default=24.0, help="pki: lifetime of enrolled certificates")
    p.add_argument("--lab-certs", help="lab mode: folder with test certificates for the attack simulator")
    node_common(p, 8080)
    p.set_defaults(fn=cmd_coordinator)

    p = sub.add_parser("active", help="run the active (label-owning) party")
    p.add_argument("--id", default="active")
    p.add_argument("--data", required=True)
    p.add_argument("--coordinator", default="localhost:50050")
    p.add_argument("--listen", default="0.0.0.0:50051")
    p.add_argument("--advertise", help="address passives use to reach this node, e.g. 192.168.1.20:50051")
    p.add_argument("--manifest", help="this party's manifest (default: <data stem>.manifest.json if present)")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--malicious", help="LAB ONLY (needs --lab-mode): scale:K | drift:K | probe (crafted gradients, "
                   "to demonstrate the passive guard)")
    node_common(p, 8081)
    enroll_args(p)
    p.set_defaults(fn=cmd_active)

    p = sub.add_parser("passive", help="run a passive (feature-owning) party")
    p.add_argument("--id", default="passive-1")
    p.add_argument("--data", required=True)
    p.add_argument("--manifest", help="this party's manifest (default: <data stem>.manifest.json if present)")
    p.add_argument("--coordinator", default="localhost:50050")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--malicious", help="LAB ONLY (needs --lab-mode): free_rider | scale:K | noise:S | nan")
    node_common(p, 8082)
    enroll_args(p)
    p.set_defaults(fn=cmd_passive)

    p = sub.add_parser("token", help="pki: create a single-use join token")
    p.add_argument("action", choices=["create"])
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--token", default=os.environ.get("VFL_UI_TOKEN"), help="dashboard operator token")
    p.add_argument("--role", default="passive", choices=["active", "passive"])
    p.add_argument("--name", help="bind the token to one party id")
    p.add_argument("--ttl", type=float, default=3600, help="seconds until the token expires")
    p.set_defaults(fn=cmd_token)

    p = sub.add_parser("ui-user", help="add a dashboard user (scrypt-hashed password)")
    p.add_argument("action", choices=["add"])
    p.add_argument("name")
    p.add_argument("--role", default="viewer", choices=["viewer", "operator"])
    p.add_argument("--file", default="ui_users.json")
    p.add_argument("--password", help="(prompted if omitted)")
    p.set_defaults(fn=cmd_ui_user)

    from .attacks import add_attack_parser
    add_attack_parser(sub, cmd_attack)

    p = sub.add_parser("bench-ui", help="benchmark & analytics dashboard (offline; port 8090)")
    p.add_argument("--results", default="results")
    p.add_argument("--runs", nargs="*", default=["runs"], help="coordinator run folders to index")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--token", default=os.environ.get("VFL_UI_TOKEN"), help="required for launching experiments")
    p.set_defaults(fn=cmd_bench_ui)

    def ui_args(p, page):
        p.add_argument("--workspace", default=".", help="folder for datasets/, uploads/, certs/, deployments/, results/")
        p.add_argument("--host", default="127.0.0.1", help="bind address (keep 127.0.0.1 unless you add --token)")
        p.add_argument("--port", type=int, default=8090)
        p.add_argument("--token", default=os.environ.get("VFL_UI_TOKEN"), help="operator token for every action")
        p.add_argument("--results", help="benchmark results folder (default WORKSPACE/results)")
        p.add_argument("--runs", nargs="*", help="coordinator run folders to index")
        p.add_argument("--coordinator-url", help="an existing coordinator dashboard to drive (besides UI deployments)")
        p.add_argument("--coordinator-token", help="that coordinator's UI token (default: --token)")
        p.add_argument("--session-config", help="default session YAML for UI deployments (configs/session.yaml)")
        p.add_argument("--lab-mode", action="store_true", help="LAB ONLY: enable the attack page's launcher")
        p.set_defaults(fn=cmd_ui, page=page)

    p = sub.add_parser("ui", help="unified UI: data prep, deployment, training, attacks, tests, analytics (port 8090)")
    ui_args(p, "dataprep")
    p = sub.add_parser("attack-ui", help="the unified UI opened on its Security & Attacks page (implies --lab-mode)")
    ui_args(p, "attacks")
    p.set_defaults(lab_mode=True)

    def ctl(p):
        p.add_argument("--url", default="http://localhost:8080", help="coordinator dashboard URL")
        p.add_argument("--token", default=os.environ.get("VFL_UI_TOKEN"))

    p = sub.add_parser("start", help="start a session")
    ctl(p)
    p.add_argument("--config", help="YAML with session overrides")
    p.add_argument("--set", action="append", default=[], help="override, e.g. --set batch_size=128")
    p.add_argument("--resume-from")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--timeout", type=float, default=7200)
    p.set_defaults(fn=cmd_start)

    p = sub.add_parser("stop"); ctl(p); p.add_argument("session"); p.set_defaults(fn=cmd_stop)
    p = sub.add_parser("status"); ctl(p); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("wait"); ctl(p); p.add_argument("session")
    p.add_argument("--timeout", type=float, default=7200); p.set_defaults(fn=cmd_wait)

    p = sub.add_parser("config", help="session-config schema: docs table, JSON schema, defaults, validate")
    p.add_argument("action", choices=["docs", "schema", "defaults", "validate"])
    p.add_argument("--config", help="validate: YAML with session overrides")
    p.add_argument("--set", action="append", default=[], help="validate: override, e.g. --set lr=0.003")
    p.add_argument("--out", help="docs: write to this file (e.g. docs/CONFIG.md) instead of stdout")
    p.add_argument("--readme", help="docs: also refresh the generated table inside this README")
    p.set_defaults(fn=cmd_config)

    p = sub.add_parser("baseline", help="B0 centralized, B1 active-only and local VFL references")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--config", default="configs/session.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--which", default="b0,b1,local_vfl")
    p.add_argument("--out", default="results/baselines.json")
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("netem", help="delay/measure relay")
    p.add_argument("--listen", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--rtt-ms", type=float, default=0.0)
    p.add_argument("--stats")
    p.add_argument("--capture")
    p.set_defaults(fn=cmd_netem)

    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
