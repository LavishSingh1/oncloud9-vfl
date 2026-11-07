"""Web dashboards: one per node (coordinator / active / passive), same frontend.

The coordinator dashboard is the operator console (parties, sessions, start/stop, cross-party
telemetry, Security tab, inference-attack panel, identities/join requests, lab attack launcher).
Party dashboards show that party's local view, including an explicit "what leaves this node"
privacy panel. Dashboards never expose raw rows.

Access control (v2 item 24), from weakest to strongest:
  none            v1 behaviour (bind to 127.0.0.1)
  --ui-token T    POST actions need `Authorization: Bearer T` (or ?token=T)
  --ui-users F    every page needs a login; users have role viewer (read) or operator (read + actions);
                  passwords stored as scrypt hashes (`vfl.cli ui-user add`); the token still works for scripts
  --ui-tls        serve HTTPS with the node's certificate (browsers must trust certs/ca.crt)
Every node also serves Prometheus metrics at /metrics (v2 item 34).
"""
from __future__ import annotations

import functools
import hashlib
import hmac
import json
import logging
import os
import secrets
import ssl
import tempfile
import threading

from flask import Flask, Response, jsonify, redirect, request, send_from_directory, session
from werkzeug.serving import make_server

from .. import websec

log = logging.getLogger("vfl.dashboard")
STATIC = os.path.join(os.path.dirname(__file__), "static")


# ------------------------------------------------------------------ users
def hash_password(password: str, salt: bytes | None = None) -> dict:
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return {"salt": salt.hex(), "hash": h.hex()}


def check_password(rec: dict, password: str) -> bool:
    h = hashlib.scrypt(password.encode(), salt=bytes.fromhex(rec["salt"]), n=2 ** 14, r=8, p=1, dklen=32)
    return hmac.compare_digest(h.hex(), rec["hash"])


def load_users(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as fh:
        return json.load(fh).get("users", {})


def add_user(path: str, name: str, password: str, role: str = "viewer"):
    if role not in ("viewer", "operator"):
        raise ValueError("role must be viewer or operator")
    data = {"users": {}}
    if os.path.exists(path):
        data = json.load(open(path))
    data["users"][name] = {**hash_password(password), "role": role}
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VFL sign in</title><link rel="stylesheet" href="/static/style.css"></head><body>
<main style="max-width:360px;margin:12vh auto;padding:0 16px"><section class="card"><div class="card-head"><h2>Sign in</h2>
<span class="muted">{node}</span></div><form method="post" class="form" action="/login">
<label>User<input name="user" autocomplete="username" autofocus></label>
<label style="margin-top:10px">Password<input name="password" type="password" autocomplete="current-password"></label>
<div class="form-actions"><button class="btn primary" type="submit">Sign in</button><span class="form-msg err">{err}</span></div>
</form></section></main></body></html>"""


# ------------------------------------------------------------------ prometheus
def _esc(v) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def prom_metrics(st: dict) -> str:
    """Text exposition format; values come from the same state the dashboard shows."""
    lines = []

    seen = set()

    def m(name, value, help_=None, mtype="gauge", **labels):
        if value is None:
            return
        if help_ and name not in seen:
            seen.add(name)
            lines.append(f"# HELP {name} {help_}")
            lines.append(f"# TYPE {name} {mtype}")
        lab = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items())
        lines.append(f"{name}{{{lab}}} {float(value)}" if lab else f"{name} {float(value)}")

    node, role = st.get("node"), st.get("role")
    m("vfl_info", 1, "node identity", node=node, role=role, tls=str(bool(st.get("tls"))).lower())
    m("vfl_uptime_seconds", st.get("uptime_s"), "process uptime")
    res = st.get("resources") or {}
    m("vfl_cpu_percent", res.get("cpu_percent"), "process CPU percent")
    m("vfl_rss_mib", res.get("rss_mib"), "process resident memory (MiB)")
    sec = st.get("security") or {}
    for atype, outs in (sec.get("by_type") or {}).items():
        for outcome, c in outs.items():
            m("vfl_security_events_total", c, "security events by type and outcome", "counter",
              attack_type=atype, outcome=outcome)
    for k, v in (sec.get("gate") or st.get("gate") or {}).items():
        m(f"vfl_gate_{k}_total", v, f"security gate {k}", "counter")
    if role == "coordinator":
        parties = st.get("parties") or []
        m("vfl_parties", sum(1 for p in parties if p.get("alive")), "live parties")
        for s in st.get("sessions") or []:
            pass
        cur = st.get("current") or {}
        m("vfl_session_running", 1 if cur.get("state") in ("STARTING", "RUNNING") else 0, "a session is running")
        m("vfl_session_completed_epochs", cur.get("completed_epochs"), "completed epochs in the current session")
    else:
        c = st.get("counters") or {}
        for key, metric in (("committed_batches", "vfl_committed_batches_total"), ("retries", "vfl_retries_total"),
                            ("rpc_timeouts", "vfl_rpc_timeouts_total"),
                            ("duplicates_suppressed", "vfl_duplicates_suppressed_total"),
                            ("dropped_batches", "vfl_dropped_batches_total")):
            m(metric, c.get(key, 0), key.replace("_", " "), "counter")
        for key in ("tensor_bytes_up", "tensor_bytes_down", "tensor_bytes_in", "tensor_bytes_out",
                    "app_bytes_in", "app_bytes_out"):
            if key in c:
                m("vfl_bytes_total", c[key], "bytes by kind", "counter", kind=key)
        pr = st.get("progress") or {}
        m("vfl_epoch", pr.get("epoch"), "current epoch")
        m("vfl_last_batch_ms", pr.get("last_batch_ms"), "last batch latency (passive)")
        m("vfl_last_loss", pr.get("last_loss"), "last training loss (active, aggregate)")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ who may look
LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")


def resolve_ui_auth(host: str, token: str | None, users: str | None, node: str, log_dir: str | None,
                    public_views: bool = False) -> tuple[str | None, bool]:
    """(token, private_views). A dashboard reachable from the network (host not loopback) shows its views only to
    someone holding the token or a login: v2.2 party dashboards on 0.0.0.0 showed feature names, row counts and
    the session config (seed, alignment salt) to the whole LAN. With no token and no users configured, a random
    token is generated and written to <log_dir>/<node>/ui_token (owner-only). `public_views` (--ui-public-views)
    restores the v2.2 behaviour for a trusted demo network."""
    if host in LOOPBACK or public_views or not host:
        return token, False
    if not token and not users:
        token = secrets.token_urlsafe(18)
        path = os.path.join(log_dir or "logs", node, "ui_token")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(token + "\n")
        log.warning("dashboard listens on %s without --ui-token: generated one; it is in %s (open the dashboard "
                    "with ?token=<it>)", host, path)
    return token, True


# ------------------------------------------------------------------ app
def create_app(backend, role: str, token: str | None = None, users: str | None = None,
               private_views: bool = False) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.secret_key = secrets.token_bytes(32)
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict",
                      MAX_CONTENT_LENGTH=16 * 2 ** 20)
    websec.install(app)                                  # Host / Origin / CSRF checks + security headers (RT-03)
    throttle = websec.LoginThrottle()
    user_db = load_users(users)
    node_name = getattr(backend, "party_id", None) or getattr(backend, "id", role)

    def current_role() -> str | None:
        """'operator' | 'viewer' | None (not signed in)."""
        if token and (websec.bearer_ok(token) or websec.query_token_ok(token)):
            return "operator"
        if user_db:
            u = session.get("user")
            return user_db[u]["role"] if u in user_db else None
        if private_views:
            return None                                    # network-facing: every view needs the token (v2.3)
        return "viewer" if token else "operator"          # loopback: open, or token-only for actions

    def need(level):
        def deco(fn):
            @functools.wraps(fn)
            def inner(*a, **kw):
                r = current_role()
                if r is None:
                    if request.path.startswith("/api/") or request.path == "/metrics":
                        return jsonify({"error": "sign in required"}), 401
                    return redirect("/login")
                if level == "operator" and r != "operator":
                    return jsonify({"error": "operator role or UI token required"}), 401 if not user_db else 403
                return fn(*a, **kw)
            return inner
        return deco

    @app.get("/login")
    def login_page():
        if not user_db:
            return redirect("/")
        return LOGIN_PAGE.format(node=node_name, err="")

    @app.post("/login")
    def login():
        u, pw = request.form.get("user", "")[:64], request.form.get("password", "")[:256]
        keys = (f"ip:{request.remote_addr}", f"user:{u}")
        if throttle.blocked(*keys):
            return LOGIN_PAGE.format(node=node_name, err="too many failed sign-ins; try again in a few minutes"), 429
        rec = user_db.get(u)
        if rec and check_password(rec, pw):
            throttle.ok(*keys)
            session.clear()
            session["user"] = u
            return redirect("/")
        throttle.fail(*keys)
        from ..telemetry import SECURITY
        SECURITY.event("dashboard_login_failed", "auth", "blocked", "password login", "server-observed",
                       request.remote_addr, u, None, "/login", detail="bad username or password")
        if throttle.blocked(*keys):
            return LOGIN_PAGE.format(node=node_name, err="too many failed sign-ins; try again in a few minutes"), 429
        return LOGIN_PAGE.format(node=node_name, err="wrong user or password"), 401

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect("/login" if user_db else "/")

    @app.get("/")
    def index():
        # the page itself is a static shell (no data); with logins it still requires one
        if user_db and current_role() is None:
            return redirect("/login")
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_files(name):
        return send_from_directory(STATIC, name)

    @app.get("/healthz")
    def health():
        return jsonify({"ok": True, "role": role})

    @app.get("/api/whoami")
    def whoami():
        return jsonify({"role": current_role(), "user": session.get("user"), "logins": bool(user_db),
                        "token_configured": bool(token)})

    @app.get("/api/state")
    @need("viewer")
    def state():
        return Response(json.dumps(backend.ui_state(), default=_jd), mimetype="application/json")

    @app.get("/metrics")
    @need("viewer")
    def metrics():
        return Response(prom_metrics(backend.ui_state()), mimetype="text/plain; version=0.0.4")

    if role == "coordinator":
        def guarded(fn, *a, **kw):
            try:
                return jsonify(fn(*a, **kw))
            except (ValueError, KeyError) as e:
                return jsonify({"error": str(e)}), 400

        def _from_form(body: dict) -> dict:
            """{form: {dotted_key: value}, resume_from?} -> {config, resume_from}; schema-validated (item 2)."""
            from .. import config_schema as cs
            cfg = cs.config_from_form(body.get("form") or {})
            resume = cfg.pop("resume_from", None) or body.get("resume_from") or None
            return {"config": cfg, "resume_from": resume}

        @app.post("/api/sessions")
        @need("operator")
        def start():
            body = request.get_json(force=True) or {}
            if "form" in body:                       # schema form; {config: ...} bodies (v2 / CLI) still work
                try:
                    body = _from_form(body)
                except ValueError as e:
                    return jsonify({"error": str(e)}), 400
            return guarded(backend.start_session, body)

        @app.get("/api/config/schema")
        @need("viewer")
        def config_schema():
            from .. import config_schema as cs
            return jsonify({**cs.schema_json(), "current": cs.form_values(backend.default_config)})

        @app.post("/api/config/validate")
        @need("viewer")
        def config_validate():
            from .. import config as cfgmod
            body = request.get_json(force=True) or {}
            try:
                out = _from_form(body) if "form" in body else {"config": body.get("config") or {},
                                                               "resume_from": body.get("resume_from")}
                if "form" not in body:
                    cfgmod.check_overrides(out["config"])
                cfgmod.validate_session_config(cfgmod.deep_merge(backend.default_config, out["config"]))
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            return jsonify({"ok": True, **out})

        @app.post("/api/sessions/<sid>/stop")
        @need("operator")
        def stop(sid):
            return guarded(backend.stop_session, sid)

        @app.get("/api/sessions/<sid>")
        @need("viewer")
        def detail(sid):
            d = backend.session_detail(sid)
            if d is None:
                return jsonify({"error": "unknown session"}), 404
            return Response(json.dumps(d, default=_jd), mimetype="application/json")

        @app.get("/api/sessions")
        @need("viewer")
        def sessions():
            return Response(json.dumps(backend.ui_state()["sessions"], default=_jd), mimetype="application/json")

        @app.get("/api/config/default")
        @need("viewer")
        def default_cfg():
            return jsonify(backend.default_config)

        # ---- security feed (items 12, 20, 21)
        @app.get("/api/security")
        @need("viewer")
        def security():
            from ..telemetry import SECURITY
            try:
                n = max(1, min(int(request.args.get("n", 500)), 2000))
            except ValueError:
                n = 500
            return Response(json.dumps({**SECURITY.summary(), "recent": SECURITY.recent(n)}, default=_jd),
                            mimetype="application/json")

        @app.post("/api/security/report")
        @need("operator")
        def security_report():
            """Attacker-reported outcomes from the lab attack simulator."""
            from ..telemetry import SECURITY
            if not backend.lab_mode:
                return jsonify({"error": "attacker reports are accepted only in --lab-mode"}), 403
            body = request.get_json(force=True) or {}
            for r in (body.get("events") or [])[:200]:
                if isinstance(r, dict) and r.get("attack_type"):
                    SECURITY.event(r["attack_type"], r.get("layer", "protocol"), r.get("outcome", "detected"),
                                   r.get("defence", ""), "attacker-reported", r.get("source_ip"),
                                   r.get("claimed_identity"), r.get("cert_identity"), r.get("method"),
                                   r.get("session"), r.get("detail", ""), node=r.get("node") or "attacker",
                                   forward=False)
            return jsonify({"ok": True})

        @app.get("/api/inference")
        @need("viewer")
        def inference():
            return Response(json.dumps(backend.inference_points(), default=_jd), mimetype="application/json")

        # ---- identities / enrollment (items 17-19, 22)
        @app.get("/api/pki/root.pem")
        def root_pem():
            path = os.path.join(backend.tls.dir or "", "ca.crt")
            if not backend.tls.enabled or not os.path.exists(path):
                return jsonify({"error": "no CA"}), 404
            return Response(open(path, "rb").read(), mimetype="application/x-pem-file")

        def _reg():
            if not backend.registry:
                raise ValueError("identity management needs pki mode (vfl.cli certs --mode pki)")
            return backend.registry

        @app.get("/api/pki")
        @need("viewer")
        def pki():
            return guarded(lambda: _reg().view())

        @app.post("/api/pki/tokens")
        @need("operator")
        def make_tok():
            b = request.get_json(force=True) or {}
            return guarded(lambda: _reg().create_token(b.get("role", "passive"), b.get("name") or None,
                                                       float(b.get("ttl_s", 3600))))

        @app.errorhandler(TypeError)
        def _bad_body(e):
            return jsonify({"error": f"bad request: {e}"}), 400

        @app.post("/api/pki/mode")
        @need("operator")
        def set_mode():
            b = request.get_json(force=True) or {}
            return guarded(lambda: (_reg().set_mode(b.get("mode")), _reg().view())[1])

        @app.post("/api/pki/requests/<rid>/<action>")
        @need("operator")
        def decide(rid, action):
            if action not in ("approve", "deny"):
                return jsonify({"error": "action must be approve or deny"}), 400
            return guarded(lambda: _reg().decide(rid, action == "approve"))

        @app.post("/api/pki/revoke/<serial>")
        @need("operator")
        def revoke(serial):
            b = request.get_json(silent=True) or {}
            return guarded(backend.revoke_identity, serial, b.get("reason") or "revoked by operator")

        # ---- lab attack launcher (items 14, 20)
        @app.get("/api/lab")
        @need("viewer")
        def lab():
            return jsonify(backend.lab.view() if backend.lab else {"enabled": False})

        @app.post("/api/lab/attacks")
        @need("operator")
        def launch():
            if not backend.lab:
                return jsonify({"error": "start the coordinator with --lab-mode to use the attack simulator"}), 403
            b = request.get_json(force=True) or {}
            return guarded(backend.lab.launch, b.get("name"), b.get("args") or {})

        @app.post("/api/lab/jobs/<jid>/cancel")
        @need("operator")
        def cancel(jid):
            if not backend.lab:
                return jsonify({"error": "lab mode off"}), 403
            return guarded(backend.lab.cancel, jid)

    return app


def _ssl_context(tls) -> ssl.SSLContext:
    ca, key, crt = tls.read()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    d = tempfile.mkdtemp(prefix="vfl-ui-")
    kp, cp = os.path.join(d, "k.pem"), os.path.join(d, "c.pem")
    with open(kp, "wb") as fh:
        fh.write(key)
    os.chmod(kp, 0o600)
    with open(cp, "wb") as fh:
        fh.write(crt)
    ctx.load_cert_chain(cp, kp)
    os.remove(kp)
    return ctx


def run_app(app: Flask, host: str, port: int, tls=None):
    """`tls`: a TLSFiles to serve HTTPS with the node certificate (None/False = HTTP)."""
    if not port:
        return None
    ctx = _ssl_context(tls) if tls else None
    srv = make_server(host, port, app, threaded=True, ssl_context=ctx)
    t = threading.Thread(target=srv.serve_forever, daemon=True, name="dashboard")
    t.start()
    shown = "localhost" if host in ("0.0.0.0", "") else host
    log.info("dashboard: %s://%s:%d", "https" if ctx else "http", shown, port)
    return srv


def _jd(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # pragma: no cover
        pass
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)
