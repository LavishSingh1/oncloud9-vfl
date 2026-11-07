"""HTTP hardening shared by the node dashboards, the unified UI and bench-ui (v2.3 red-team fixes RT-03, RT-18).

v2.2 dashboards started without a token treated every request as the operator and parsed any body as JSON
(`get_json(force=True)`), and nothing checked the Host or Origin headers. So any web page the operator opened could
POST to http://127.0.0.1:8080 / :8090 (CSRF: start sessions, mint join tokens, create UI users, upload files,
launch deployments), and a DNS-rebinding page could also READ the replies (join tokens, bundles).

`install(app)` adds, for every request:
  * Host allow-list: IP literals, localhost, this machine's names, and VFL_UI_ALLOWED_HOSTS (comma separated).
    A rebinding attack needs the attacker's own domain in Host, so it is refused.
  * for state-changing methods: a cross-site Origin is refused, and the request must be JSON
    (application/json forces a CORS preflight the server never grants) or carry the X-VFL-Request header
    (custom headers also force a preflight) - the UIs send it on every call, including uploads.
  * security headers: a strict CSP (script-src 'self', so injected markup and javascript: links cannot run),
    no framing, no MIME sniffing, no Referer (tokens never leak through links).
"""
from __future__ import annotations

import hmac
import ipaddress
import os
import socket
import threading
import time

from flask import jsonify, request

CSRF_HEADER = "X-VFL-Request"
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'")


def _machine_names() -> set:
    names = {"localhost", "localhost.localdomain"}
    try:
        h = socket.gethostname()
        names |= {h.lower(), socket.getfqdn().lower(), h.split(".")[0].lower()}
    except OSError:
        pass
    extra = os.environ.get("VFL_UI_ALLOWED_HOSTS", "")
    names |= {x.strip().lower() for x in extra.split(",") if x.strip()}
    return names


def _host_only(hostport: str) -> str:
    h = (hostport or "").strip().lower()
    if h.startswith("["):                               # [::1]:8080
        return h[1:h.find("]")] if "]" in h else h
    if h.count(":") == 1:
        h = h.split(":", 1)[0]
    return h


def host_allowed(hostport: str, allowed: set) -> bool:
    h = _host_only(hostport)
    if not h:
        return False
    try:
        ipaddress.ip_address(h)
        return True                                     # an IP literal cannot be a rebinding attack
    except ValueError:
        pass
    return h in allowed or h.endswith(".localhost")


def tokens_equal(given: str | None, token: str | None) -> bool:
    return bool(token) and given is not None and hmac.compare_digest(str(given).encode(), str(token).encode())


def bearer_ok(token: str | None) -> bool:
    """Operator token from the Authorization header (constant-time)."""
    h = request.headers.get("Authorization", "")
    return h.startswith("Bearer ") and tokens_equal(h[7:], token)


def query_token_ok(token: str | None) -> bool:
    """?token= is accepted for page loads (GET) only; actions need the header, so a token never has to sit in
    a URL that browsers, proxies and server logs keep."""
    return request.method in SAFE_METHODS and tokens_equal(request.args.get("token"), token)


def install(app, extra_hosts: set | None = None):
    allowed = _machine_names() | {x.lower() for x in (extra_hosts or set())}

    @app.before_request
    def _guard():
        if not host_allowed(request.host, allowed):
            return jsonify({"error": f"host {request.host!r} not allowed (DNS-rebinding protection); "
                                     "add it to VFL_UI_ALLOWED_HOSTS"}), 403
        if request.method in SAFE_METHODS:
            return None
        origin = request.headers.get("Origin")
        same_origin = False
        if origin and origin != "null":
            from urllib.parse import urlsplit
            o = urlsplit(origin)
            if (o.netloc or "").lower() != (request.host or "").lower():
                return jsonify({"error": "cross-site request refused"}), 403
            same_origin = True                          # browsers always send Origin on cross-site POSTs
        elif origin == "null":
            return jsonify({"error": "cross-site request refused"}), 403
        is_json = (request.mimetype or "") == "application/json"
        if not (is_json or same_origin) and request.headers.get(CSRF_HEADER) != "1":
            return jsonify({"error": f"state-changing requests must be JSON or carry {CSRF_HEADER}: 1"}), 403
        return None

    @app.after_request
    def _headers(resp):
        resp.headers.setdefault("Content-Security-Policy", CSP)
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        resp.headers.setdefault("Cache-Control", "no-store")
        return resp
    return app


class LoginThrottle:
    """Per-client and per-user failure counter: after `limit` failures in `window` s, refuse logins (429) for
    `lockout` s. Scrypt alone still allowed ~10 guesses/s per core."""

    def __init__(self, limit: int = 5, window: float = 300.0, lockout: float = 300.0):
        self.limit, self.window, self.lockout = limit, window, lockout
        self.fails: dict = {}
        self.locked: dict = {}
        self.lock = threading.Lock()

    def blocked(self, *keys) -> bool:
        now = time.time()
        with self.lock:
            return any(self.locked.get(k, 0) > now for k in keys)

    def fail(self, *keys):
        now = time.time()
        with self.lock:
            if len(self.fails) > 10000:
                self.fails.clear()
            for k in keys:
                lst = [t for t in self.fails.get(k, []) if now - t < self.window] + [now]
                self.fails[k] = lst
                if len(lst) >= self.limit:
                    self.locked[k] = now + self.lockout

    def ok(self, *keys):
        with self.lock:
            for k in keys:
                self.fails.pop(k, None)
