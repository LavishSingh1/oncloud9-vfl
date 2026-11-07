"""Unified UI (v2.1 item 6): `python -m vfl.cli ui` (port 8090, 127.0.0.1 by default).

One page with Data preparation · Deployment · Training/Sessions · Security & Attacks · Tests & Benchmarks ·
Analytics · Reports. It links to the node dashboards (coordinator :8080, parties :8081+) rather than replacing
them. Every CLI subcommand has a UI route (CLI_ROUTES; tests/test_webui.py walks the CLI parser to check).

The UI does not re-implement anything: data preparation, certificates, baselines, attacks and benchmarks run
the same `python -m vfl.cli ...` / bench scripts as jobs; deployments reuse bench/harness.LocalDeployment;
training actions are forwarded to the chosen coordinator's own HTTP API.

Access: GET routes are read-only views. Every action (POST) needs the operator token when one is configured
(--token / VFL_UI_TOKEN), as before. Attack runs additionally need a configured token, the UI started with
--lab-mode, and an explicit per-run acknowledgement.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from werkzeug.serving import make_server

from .. import websec
from .dataprep import Workspace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC = os.path.join(os.path.dirname(__file__), "static")
DASH_STATIC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "static")
PAGES = ("dataprep", "deployment", "training", "attacks", "tests", "analytics", "reports")

# CLI subcommand -> (UI page, method, route). Every subcommand must appear here (tests/test_webui.py).
CLI_ROUTES = {
    "certs": ("dataprep", "POST", "/api/prep/certs"),
    "prepare-data": ("dataprep", "POST", "/api/prep/generate"),
    "embed-text": ("dataprep", "POST", "/api/prep/embed-text"),
    "coordinator": ("deployment", "POST", "/api/deploy/local"),
    "active": ("deployment", "POST", "/api/deploy/local"),
    "passive": ("deployment", "POST", "/api/deploy/local"),
    "netem": ("deployment", "POST", "/api/deploy/local"),            # rtt_relay_ms starts the netem relay
    "start": ("training", "POST", "/api/train/start"),
    "stop": ("training", "POST", "/api/train/stop"),
    "status": ("training", "GET", "/api/train/state"),
    "wait": ("training", "GET", "/api/train/sessions/<sid>"),
    "config": ("training", "GET", "/api/train/schema"),
    "baseline": ("training", "POST", "/api/train/baseline"),
    "token": ("training", "POST", "/api/train/token"),
    "ui-user": ("reports", "POST", "/api/admin/ui-user"),
    "attack": ("attacks", "POST", "/api/attacks/run"),
    "attack-ui": ("attacks", "GET", "/"),
    "bench-ui": ("tests", "GET", "/"),
    "ui": ("dataprep", "GET", "/"),
}


def _http(method, url, body=None, token=None, timeout=10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode().replace("NaN", "null") or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except ValueError:
            return e.code, {"error": str(e)}
    except (urllib.error.URLError, OSError) as e:
        return 502, {"error": f"cannot reach {url}: {getattr(e, 'reason', e)}"}


def create_app(workspace: str = ".", token: str | None = None, results: str | None = None, runs=None,
               coordinator_url: str | None = None, coordinator_token: str | None = None, lab_mode: bool = False,
               start_page: str = "dataprep", session_config: str | None = None) -> Flask:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from bench import ui as bench_ui
    from . import dataprep as D
    from .deploy import ACTIVE_NOTE, DeploymentManager, dataset_parties
    from .jobs import JobManager, cli_cmd

    ws = Workspace(workspace)
    jobs = JobManager()
    deploys = DeploymentManager(ws, session_config)
    from ..attacks import CASES, FAMILIES
    from ..attacks.jobs import AttackJobs
    attacks = AttackJobs(ws.path("results", "attacks"))
    bench = bench_ui.BenchUI(results or ws.path("results"), runs or [ws.path("runs")], workspace=ws.root)
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3
    app.extensions["vfl"] = {"ws": ws, "jobs": jobs, "deploys": deploys, "attacks": attacks, "bench": bench}
    websec.install(app)                                  # Host / Origin / CSRF checks + security headers (RT-03)

    def op():
        return not token or websec.bearer_ok(token) or websec.query_token_ok(token)

    def body():
        return request.get_json(force=True, silent=True) or {}

    def guard(fn):
        def inner(*a, **kw):
            if request.method != "GET" and not op():
                return jsonify({"error": "operator token required (start the UI with --token and open ?token=...)"}), 401
            try:
                out = fn(*a, **kw)
            except (ValueError, FileNotFoundError) as e:
                return jsonify({"error": str(e)}), 400
            except KeyError as e:
                return jsonify({"error": str(e).strip("'\"")}), 404
            return out if isinstance(out, (Response, tuple)) else jsonify(out)
        inner.__name__ = fn.__name__
        return inner

    # ------------------------------------------------------------ shell
    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:n>")
    def static_files(n):
        return send_from_directory(STATIC, n)

    @app.get("/dashboard-static/<path:n>")
    def dash_static(n):
        return send_from_directory(DASH_STATIC, n)

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "role": "ui"})

    @app.get("/api/whoami")
    def whoami():
        return jsonify({"operator": op(), "token_configured": bool(token), "lab_mode": lab_mode,
                        "start_page": start_page, "workspace": ws.root, "pages": PAGES})

    @app.get("/api/cli-map")
    def cli_map():
        return jsonify({k: {"page": p, "method": m, "route": r} for k, (p, m, r) in CLI_ROUTES.items()})

    @app.get("/api/overview")
    def overview():
        return jsonify({"datasets": len(ws.list_datasets()), "certs": len(ws.list_certs()),
                        "deployments": deploys.list(), "benches": [os.path.basename(d) for d in bench.bench_dirs()],
                        "jobs": jobs.list(tail=0)[:20]})

    # ------------------------------------------------------------ workspace jobs
    @app.get("/api/ws-jobs")
    def ws_jobs():
        return jsonify(jobs.list(request.args.get("kind"), tail=0))

    @app.get("/api/ws-jobs/<jid>")
    @guard
    def ws_job(jid):
        return jobs.get(jid).view(tail=int(request.args.get("tail", 400)))

    @app.post("/api/ws-jobs/<jid>/cancel")
    @guard
    def ws_job_cancel(jid):
        return jobs.cancel(jid)

    # ------------------------------------------------------------ data preparation (item 3)
    def prep_ctx(values):
        up = ws.uploads()
        ctx = {"uploads": up["files"], "upload_dirs": up["dirs"]}
        if (values or {}).get("dataset") == "csv" and values.get("csv"):
            try:
                ctx["csv_columns"] = ws.csv_columns(values["csv"])
            except Exception:
                ctx["csv_columns"] = []
        return ctx

    def resolve_paths(values: dict) -> dict:
        v = dict(values or {})
        for k in ("csv", "npz"):
            if v.get(k):
                v[k] = ws.upload_path(v[k])
        if v.get("har_dir"):
            v["har_dir"] = ws.upload_path(v["har_dir"])
        return v

    @app.post("/api/prep/form")
    @guard
    def prep_form():
        v = body().get("values") or {}
        return D.form(v, prep_ctx(v))

    @app.post("/api/prep/columns")
    @guard
    def prep_columns():
        return {"columns": D.columns_for(body().get("values") or {}, ws)}

    def summarize_job(out_dir, keep_meta=None):
        def done(job):
            if job.returncode != 0:
                return None
            s = D.summarize(out_dir)
            if keep_meta:
                ws.save_meta(os.path.basename(out_dir), {**keep_meta, "created": time.time(), "summary": s})
            return s
        return done

    @app.post("/api/prep/preview")
    @guard
    def prep_preview():
        v = D.normalise(body().get("values") or {})
        pv = dict(v)
        if pv.get("partitioner") == "v2" and D.kind_of(pv) in ("tabular", "image"):
            pv["evaluate"] = True                     # preview always shows the experiment-only reference AUROC
        pid = f"preview-{os.urandom(4).hex()}"
        argv = D.build_argv(resolve_paths(pv), pid, ws.path("datasets", ".inputs"))
        d = ws.path("datasets", ".preview")
        os.makedirs(d, exist_ok=True)
        out_dir = os.path.join(d, pid)

        def preview_done(job):
            import shutil
            try:
                return summarize_job(out_dir)(job)
            finally:                                   # a preview is never saved: drop the scratch folder
                shutil.rmtree(out_dir, ignore_errors=True)

        j = jobs.submit("prep-preview", cli_cmd(argv), cwd=d, label=f"preview {v['dataset']}",
                        meta={"values": v}, on_done=preview_done, timeout=1800)
        return {"job": j.id}

    @app.post("/api/prep/generate")
    @guard
    def prep_generate():
        b = body()
        v = D.normalise(b.get("values") or {})
        name = ws.check_name(b.get("name") or "")
        if os.path.exists(ws.path("datasets", name)):
            if not b.get("overwrite"):
                raise ValueError(f"dataset {name!r} exists; choose overwrite to replace it")
            ws.delete_dataset(name)
        argv = D.build_argv(resolve_paths(v), name, ws.path("datasets", ".inputs"))
        j = jobs.submit("prepare-data", cli_cmd(argv), cwd=ws.datasets, label=f"prepare {name}",
                        meta={"name": name, "values": v},
                        on_done=summarize_job(ws.path("datasets", name), {"options": v, "argv": argv}), timeout=3600)
        return {"job": j.id, "argv": argv}

    @app.get("/api/prep/datasets")
    def prep_datasets():
        return jsonify(ws.list_datasets())

    @app.get("/api/prep/datasets/<name>")
    @guard
    def prep_dataset(name):
        d = ws.path("datasets", ws.check_name(name))
        if not os.path.isdir(d):
            raise KeyError(f"unknown dataset {name}")
        return {**D.summarize(d), **ws.dataset_meta(name), "parties_by_role": dataset_parties(d)}

    @app.delete("/api/prep/datasets/<name>")
    @guard
    def prep_delete(name):
        ws.delete_dataset(name)
        return {"ok": True}

    @app.post("/api/uploads")
    @guard
    def upload():
        f = request.files.get("file")
        if not f:
            raise ValueError("multipart field 'file' required")
        return ws.save_upload(f.filename, f.stream)

    @app.get("/api/uploads")
    def uploads():
        return jsonify(ws.uploads())

    @app.get("/api/uploads/columns")
    @guard
    def upload_columns():
        return {"columns": ws.csv_columns(request.args["file"])}

    @app.post("/api/prep/embed-text")
    @guard
    def embed_text():
        b = body()
        src = b.get("csv") or ""
        enc = b.get("encoder", "hashing")
        dim = int(b.get("dim", 64))
        if enc not in ("hashing", "minilm", "distilbert") or dim not in (16, 32, 64, 128, 256):
            raise ValueError("encoder: hashing|minilm|distilbert; dim: 16-256")
        out = f"{os.path.splitext(os.path.basename(src))[0]}-{enc}{dim}.csv"
        cols = ws.csv_columns(src)
        if b.get("text_col") not in cols:
            raise ValueError(f"text column must be one of {cols}")
        argv = ["embed-text", "--csv", ws.upload_path(src), "--text-col", b["text_col"], "--out",
                ws.path("uploads", out), "--encoder", enc, "--dim", str(dim)]
        from ..cli import build_parser
        build_parser().parse_args(argv)
        j = jobs.submit("embed-text", cli_cmd(argv), cwd=ws.root, label=f"embed-text {src}", meta={"out": out})
        return {"job": j.id, "out": out}

    @app.get("/api/prep/certs")
    def certs_list():
        return jsonify({"sets": ws.list_certs(), "addresses": D.local_addresses()})

    @app.post("/api/prep/certs")
    @guard
    def certs_make():
        b = body()
        name = ws.check_name(b.get("name") or "")
        if os.path.exists(ws.path("certs", name)) and not b.get("overwrite"):
            raise ValueError(f"certificate set {name!r} exists")
        # v2.2 added a CA-signed 'rogue-node' identity to EVERY certificate set (RT-09). Test identities for the
        # attack simulator are now opt-in (lab sets only) and the coordinator refuses them outside --lab-mode.
        extra = ["rogue-node"] if b.get("lab_identities") is True else None
        argv = D.certs_argv(name, int(b.get("passives", 1)), b.get("mode", "static"), b.get("hosts") or [],
                            hours=b.get("hours"), domain=b.get("domain"), extra=extra)
        j = jobs.submit("certs", cli_cmd(argv), cwd=ws.root, label=f"certs {name}", meta={"name": name})
        return {"job": j.id, "argv": argv}

    # ------------------------------------------------------------ deployment (item 4)
    @app.get("/api/deploy")
    def deploy_list():
        return jsonify({"deployments": deploys.list(), "active_note": ACTIVE_NOTE,
                        "datasets": [{"name": d["name"], "parties": dataset_parties(d["dir"])} for d in ws.list_datasets()],
                        "certs": ws.list_certs(), "addresses": D.local_addresses(), "slots": list(range(5))})

    @app.post("/api/deploy/local")
    @guard
    def deploy_local():
        b = body()
        return deploys.launch_local(b.get("name") or f"local-{ws.check_name(b.get('dataset') or 'x')}",
                                    b.get("dataset") or "",
                                    int(b.get("n_passive", 1)), b.get("tls", "static"), b.get("certs") or None,
                                    int(b.get("slot", 0)), bool(b.get("lab_mode")), b.get("ui_token") or token,
                                    rtt_relay_ms=b.get("rtt_relay_ms"))

    @app.get("/api/deploy/<name>")
    @guard
    def deploy_view(name):
        return deploys.view(name)

    @app.post("/api/deploy/<name>/stop")
    @guard
    def deploy_stop(name):
        return deploys.stop(name)

    @app.post("/api/deploy/<name>/nodes/<node>/<action>")
    @guard
    def deploy_node(name, node, action):
        if action == "restart":
            return deploys.restart_node(name, node)
        if action == "stop":
            return deploys.stop_node(name, node)
        raise ValueError("action: restart or stop")

    @app.get("/api/deploy/<name>/logs/<node>")
    @guard
    def deploy_log(name, node):
        return {"node": node, "log": deploys.log(name, node, int(request.args.get("n", 200)))}

    @app.post("/api/deploy/bundles")
    @guard
    def deploy_bundles():
        b = body()
        return deploys.make_bundles(b.get("name") or "lan", b.get("dataset") or "", b.get("certs") or "",
                                    int(b.get("n_passive", 1)), b.get("coordinator_host") or "",
                                    b.get("active_host") or "", b.get("passive_hosts") or [], b.get("tls", "static"))

    @app.get("/api/deploy/<name>/bundles/<node>.zip")
    @guard
    def deploy_bundle(name, node):
        if not op():                         # a bundle holds the node's private key: operators only, even for GET
            return jsonify({"error": "operator token required to download a bundle (it contains a private key)"}), 401
        p = ws.path("deployments", ws.check_name(name), "bundles", f"{node}.zip")
        if not os.path.exists(p) or os.path.basename(p) != f"{node}.zip":
            raise KeyError("no such bundle")
        return send_file(p, as_attachment=True)

    # ------------------------------------------------------------ training / sessions (forwarded to a coordinator)
    def coordinators() -> dict:
        out = {}
        if coordinator_url:
            out["(configured)"] = {"url": coordinator_url, "token": coordinator_token or token}
        for d in deploys.list():
            if d.get("coordinator_url"):
                out[d["name"]] = {"url": d["coordinator_url"], "token": token}
        return out

    def coord(name):
        cs = coordinators()
        if not cs:
            raise ValueError("no coordinator: launch a deployment first, or start the UI with --coordinator-url")
        if not name:
            name = next(iter(cs))
        if name not in cs:
            raise KeyError(f"unknown coordinator {name!r}; choose from {list(cs)}")
        return cs[name]

    def fwd(method, name, path, payload=None):
        c = coord(name)
        status, js = _http(method, c["url"].rstrip("/") + path, payload, c["token"])
        return (jsonify(js), status) if status >= 400 else js

    @app.get("/api/train/coordinators")
    def train_coords():
        return jsonify([{"name": k, "url": v["url"]} for k, v in coordinators().items()])

    @app.get("/api/train/schema")
    @guard
    def train_schema():
        from .. import config_schema as cs
        try:
            c = coord(request.args.get("coord"))
            status, js = _http("GET", c["url"].rstrip("/") + "/api/config/schema", None, c["token"], timeout=4)
            if status == 200:
                js["datasets"] = [d["name"] for d in ws.list_datasets()]
                return js
        except (ValueError, KeyError):
            pass
        from ..config import load_session_config
        cfg = load_session_config(session_config or os.path.join(ROOT, "configs", "session.yaml"))
        return {**cs.schema_json(), "current": cs.form_values(cfg), "datasets": [d["name"] for d in ws.list_datasets()],
                "offline": True}

    @app.post("/api/train/validate")
    @guard
    def train_validate():
        return fwd("POST", body().get("coord"), "/api/config/validate", {k: v for k, v in body().items() if k != "coord"})

    @app.get("/api/train/state")
    @guard
    def train_state():
        return fwd("GET", request.args.get("coord"), "/api/state")

    @app.get("/api/train/sessions/<sid>")
    @guard
    def train_session(sid):
        return fwd("GET", request.args.get("coord"), f"/api/sessions/{sid}")

    @app.post("/api/train/start")
    @guard
    def train_start():
        b = body()
        return fwd("POST", b.get("coord"), "/api/sessions", {k: v for k, v in b.items() if k != "coord"})

    @app.post("/api/train/stop")
    @guard
    def train_stop():
        b = body()
        return fwd("POST", b.get("coord"), f"/api/sessions/{b.get('session')}/stop", {})

    @app.post("/api/train/token")
    @guard
    def train_token():
        b = body()
        return fwd("POST", b.get("coord"), "/api/pki/tokens",
                   {"role": b.get("role", "passive"), "name": b.get("name"), "ttl_s": float(b.get("ttl_s", 3600))})

    @app.post("/api/train/baseline")
    @guard
    def train_baseline():
        from .. import config_schema as cs
        b = body()
        ds = ws.check_name(b.get("dataset") or "")
        if not os.path.isdir(ws.path("datasets", ds)):
            raise KeyError(f"unknown dataset {ds}")
        which = b.get("which") or "b0,b1,local_vfl"
        if not set(which.split(",")) <= {"b0", "b1", "local_vfl"}:
            raise ValueError("which: b0, b1, local_vfl")
        cfg = cs.config_from_form(b["form"]) if b.get("form") else {}
        cfg.pop("resume_from", None)
        sets = [f"{k}={json.dumps(cs.get_path(cfg, k))}" for k in cs.leaf_keys(cfg)]
        out = ws.path("results", "baselines", f"{ds}-{time.strftime('%Y%m%d-%H%M%S')}.json")
        argv = ["baseline", "--data-dir", ws.path("datasets", ds), "--which", which, "--out", out]
        for s_ in sets:
            argv += ["--set", s_]
        j = jobs.submit("baseline", cli_cmd(argv), cwd=ROOT, label=f"baseline {ds}", meta={"out": out},
                        on_done=lambda job: json.load(open(out)) if os.path.exists(out) else None, timeout=7200)
        return {"job": j.id, "out": out}

    # ------------------------------------------------------------ security & attacks (item 1)
    @app.get("/api/attacks")
    def attacks_view():
        targets = []
        for d in deploys.list():
            p = ws.path("deployments", d["name"], "deployment.json")
            if os.path.exists(p):
                targets.append({"name": d["name"], "coordinator": d.get("coordinator"), "active": d.get("active"),
                                "running": d.get("running"), "lab_mode": d.get("lab_mode")})
        return jsonify({"cases": list(CASES), "families": {k: list(v) for k, v in FAMILIES.items()},
                        "targets": targets, "jobs": attacks.view(), "lab_mode": lab_mode,
                        "token_configured": bool(token), "datasets": [d["name"] for d in ws.list_datasets()]})

    @app.post("/api/attacks/run")
    @guard
    def attacks_run():
        b = body()
        if not lab_mode:
            raise ValueError("start the UI with --lab-mode (or use `vfl.cli attack-ui`) to run the attack simulator")
        if not token:
            raise ValueError("attack runs need an operator token: start the UI with --token")
        if b.get("lab_mode_ack") is not True:
            raise ValueError("confirm that the target is your own lab deployment")
        name = ws.check_name(b.get("target") or "")
        plan = ws.path("deployments", name, "deployment.json")
        if not os.path.exists(plan):
            raise KeyError(f"no deployment {name!r} (its deployment.json is the target allowlist)")
        p = json.load(open(plan))
        data_dir = ws.path("datasets", ws.check_name(b["dataset"])) if b.get("dataset") else None
        return attacks.launch(b.get("case") or "all", coordinator=p["coordinator"], active=p.get("active"),
                              certs=p.get("certs") or ws.path("deployments", name, "certs"), config=plan,
                              session=b.get("session") or None, data_dir=data_dir, url=p.get("coordinator_url"),
                              token=token, log_dirs=p.get("log_dirs"))

    @app.get("/api/attacks/jobs/<jid>")
    @guard
    def attack_job(jid):
        return next((j for j in attacks.view() if j["id"] == jid), None) or (jsonify({"error": "unknown job"}), 404)

    @app.post("/api/attacks/jobs/<jid>/cancel")
    @guard
    def attack_cancel(jid):
        return attacks.cancel(jid)

    @app.get("/api/attacks/jobs/<jid>/report.<fmt>")
    @guard
    def attack_report(jid, fmt):
        j = next((x for x in attacks.view() if x["id"] == jid), None)
        if not j or fmt not in ("json", "csv"):
            raise KeyError("no report")
        p = os.path.join(j["out"], f"report.{fmt}")
        if not os.path.exists(p):
            raise KeyError("report not written yet")
        return send_file(p, as_attachment=True, download_name=f"attack-{j['name']}-{jid}.{fmt}")

    # ------------------------------------------------------------ admin
    @app.post("/api/admin/ui-user")
    @guard
    def ui_user():
        from ..dashboard import add_user
        b = body()
        if len(b.get("password") or "") < 8:
            raise ValueError("password must be at least 8 characters")
        if not D.NAME_RE.fullmatch(b.get("name") or ""):
            raise ValueError("user name: letters, digits, '.', '_' or '-'")
        path = ws.path("ui_users.json")
        add_user(path, b["name"], b["password"], b.get("role", "viewer"))
        return {"ok": True, "file": path, "hint": f"start nodes with --ui-users {path}"}

    # ------------------------------------------------------------ tests, benchmarks, analytics, reports (item 5)
    bench_ui.register(app, bench, op)
    return app


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def serve(workspace=".", host="127.0.0.1", port=8090, token=None, results=None, runs=None, coordinator_url=None,
          coordinator_token=None, lab_mode=False, start_page="dataprep", session_config=None):
    if host not in LOOPBACK_HOSTS and not token:
        # this UI starts processes (deployments, benchmarks, attacks): never expose it without a token
        raise SystemExit(f"refusing to bind {host} without --token: the UI can launch processes. "
                         "Add --token (or VFL_UI_TOKEN), or keep the default 127.0.0.1")
    if lab_mode and not token:
        print("note: --lab-mode without --token: the attack launcher stays disabled until a token is set", flush=True)
    app = create_app(workspace, token, results, runs, coordinator_url, coordinator_token, lab_mode, start_page,
                     session_config)
    srv = make_server(host, port, app, threaded=True)
    shown = "localhost" if host in ("0.0.0.0", "") else host
    q = f"?token={token}" if token else ""
    print(f"VFL UI: http://{shown}:{port}/{q}#{start_page}   (workspace: {app.extensions['vfl']['ws'].root})", flush=True)
    if not token:
        print("note: no --token set; anyone with access to this machine's loopback can run actions")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.extensions["vfl"]["deploys"].stop_all()
