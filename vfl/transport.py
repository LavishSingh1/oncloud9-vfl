"""gRPC servers and channels with mutual TLS, plus the live security gate (v2 items 9, 11).

SecurityGate is a grpc.ServerInterceptor installed on every server. For each call it checks, in order:
  rate     per-peer token bucket (--rate-limit-rps) and per-peer concurrent calls (--max-peer-concurrency)
  auth     certificate revoked (serial on the coordinator's revocation list) or expired
  auth     identity: the name in the CERTIFICATE (SAN URI in pki mode, CN in static mode) must equal the
           identity claimed in the message (sender / party_id) - a stolen or borrowed certificate cannot
           speak for another party
  auth     role: in pki mode the certificate's role must be allowed for the method
           (e.g. only passive parties may call Exchange/Align)
Application handlers add protocol/data checks (wrong session, digest mismatch, malformed or
non-finite tensors, norm bounds, duplicate replays) through `reject(...)`. Every rejection is written to
security.jsonl with the cause. Connection caps: grpc `maximum_concurrent_rpcs` per server.

Handshake failures (no certificate, rogue CA, expired certificate) are refused by TLS before any
application code runs; they are visible to the TLS-aware gate in netem_proxy.py and to the attacker.
"""
from __future__ import annotations

import collections
import threading
import time
from concurrent import futures

import grpc

import os

from .security import TLSFiles, describe_cert, effective_role, server_role
from .telemetry import SECURITY

# Largest gRPC message. Big enough for PSI over ~1M ids; tune with VFL_GRPC_MAX_MSG_MB. Together with the
# per-identity concurrency cap this bounds the memory one authorised peer can make a node allocate.
MAX_MSG = int(float(os.environ.get("VFL_GRPC_MAX_MSG_MB", "256")) * 1024 * 1024)
DEFAULT_WORKERS = 64
_OPTS = [("grpc.max_send_message_length", MAX_MSG), ("grpc.max_receive_message_length", MAX_MSG),
         ("grpc.keepalive_time_ms", 30000), ("grpc.keepalive_timeout_ms", 10000)]
# Servers must accept the clients' 30 s keepalive pings even while a call is idle (e.g. a party
# waiting at an epoch barrier for a slow device). gRPC's default server policy rejects pings more
# often than every 5 min and then closes the connection (GOAWAY "too_many_pings"). The pings also
# keep NAT / firewall mappings open on internet paths.
_SERVER_OPTS = _OPTS + [("grpc.http2.min_ping_interval_without_data_ms", 10000),
                        ("grpc.http2.max_ping_strikes", 0),
                        ("grpc.keepalive_permit_without_calls", 1)]

DEFAULT_MAX_CONCURRENT_RPCS = 256      # far above normal load (one Exchange per passive + barriers)


class Policy:
    """Per-method rule: where the claimed identity lives, and which certificate roles may call."""

    def __init__(self, claim, roles=None):
        self.claim = claim          # callable(request) -> claimed identity or None
        self.roles = set(roles) if roles else None


COORDINATOR_POLICY = {
    "Register": Policy(lambda r: r.party_id, ("active", "passive")),
    "Heartbeat": Policy(lambda r: r.party_id, ("active", "passive")),
    "Barrier": Policy(lambda r: r.party_id, ("active", "passive")),
    "Report": Policy(lambda r: r.party_id, ("active", "passive")),
}
ACTIVE_POLICY = {
    "Align": Policy(lambda r: r.header.sender, ("passive",)),
    "Exchange": Policy(lambda r: r.header.sender, ("passive",)),
    "Ping": Policy(lambda r: r.sender or None, None),
}


class _Bucket:
    def __init__(self, rate: float, burst: float):
        self.rate, self.burst = rate, burst
        self.tokens, self.t = burst, time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


def peer_ip(context) -> str | None:
    try:
        p = context.peer() or ""
    except Exception:
        return None
    if p.startswith("ipv4:"):
        return p[5:].rsplit(":", 1)[0]
    if p.startswith("ipv6:"):
        return p[5:].rsplit(":", 1)[0].strip("[]")
    return p or None


def peer_cert(context) -> dict | None:
    """{name, role, serial, not_after, ...} of the calling peer's certificate, or None (insecure)."""
    try:
        ac = context.auth_context() or {}
    except Exception:
        return None
    pem = ac.get("x509_pem_cert")
    if pem:
        v = pem[0]
        try:
            return describe_cert(v if isinstance(v, bytes) else v.encode())
        except Exception:
            pass
    cn = ac.get("x509_common_name")
    if cn:
        v = cn[0]
        return {"name": v.decode() if isinstance(v, bytes) else str(v), "role": None, "serial": None,
                "not_after": None}
    return None


def peer_identity(context) -> str | None:
    c = peer_cert(context)
    return c["name"] if c else None


def reject(context, code, message: str, attack_type: str, layer: str, defence: str, claimed=None,
           session=None, method=None, outcome="blocked"):
    """Log a security event (server-observed) and abort the RPC."""
    c = peer_cert(context)
    SECURITY.event(attack_type, layer, outcome, defence, "server-observed", peer_ip(context), claimed,
                   c["name"] if c else None, method, session, message)
    context.abort(code, message)


class SecurityGate(grpc.ServerInterceptor):
    def __init__(self, tls: TLSFiles, policy: dict, revoked=lambda: set(), rate_limit_rps: float | None = None,
                 max_peer_concurrency: int | None = None):
        self.tls = tls
        self.pki = bool(tls.pki) if tls.enabled else False      # read pki.json once, not per call
        self.policy = policy
        self.revoked = revoked
        self.rate = rate_limit_rps
        self.max_conc = max_peer_concurrency
        self.buckets: dict = {}
        self.active = collections.Counter()
        self.lock = threading.Lock()
        self.stats = collections.Counter()

    def _admit(self, key: str) -> str | None:
        with self.lock:
            if self.rate:
                b = self.buckets.get(key)
                if b is None:
                    if len(self.buckets) > 10000:            # bounded state
                        self.buckets.clear()
                    b = self.buckets[key] = _Bucket(float(self.rate), max(1.0, 2.0 * float(self.rate)))
                if not b.take():
                    return "rate"
            if self.max_conc and self.active[key] >= int(self.max_conc):
                return "concurrency"
            self.active[key] += 1
        return None

    def _release(self, ip):
        with self.lock:
            self.active[ip] -= 1

    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        if handler is None or handler.unary_unary is None:
            return handler
        method = handler_call_details.method.rsplit("/", 1)[-1]
        pol = self.policy.get(method)
        inner = handler.unary_unary

        def wrapped(request, context):
            ip = peer_ip(context) or "?"
            # limits are per IDENTITY when there is one (parties behind one NAT, or a whole local deployment on
            # 127.0.0.1, do not share a budget), per IP otherwise (insecure mode, the enrollment port)
            c = peer_cert(context) if self.tls.enabled else None
            key = f"id:{c['name']}" if c and c.get("name") else f"ip:{ip}"
            self.stats["calls"] += 1
            why = self._admit(key)
            if why:
                self.stats["rate_limited"] += 1
                claimed = _safe_claim(pol, request)
                reject(context, grpc.StatusCode.RESOURCE_EXHAUSTED,
                       f"{'rate limit' if why == 'rate' else 'per-peer concurrency cap'} exceeded for {ip}",
                       "flood", "rate", "token bucket" if why == "rate" else "per-peer concurrency cap",
                       claimed, method=method)
            try:
                if self.tls.enabled:
                    self._check_identity(request, context, pol, method)
                return inner(request, context)
            finally:
                if not why:
                    self._release(key)

        return grpc.unary_unary_rpc_method_handler(wrapped, request_deserializer=handler.request_deserializer,
                                                   response_serializer=handler.response_serializer)

    def _check_identity(self, request, context, pol, method):
        c = peer_cert(context)
        claimed = _safe_claim(pol, request)
        if c is None:
            reject(context, grpc.StatusCode.UNAUTHENTICATED, "no client certificate", "no_cert", "auth", "mTLS",
                   claimed, method=method)
        if c.get("serial") and c["serial"] in self.revoked():
            reject(context, grpc.StatusCode.PERMISSION_DENIED, f"certificate {c['serial']} is revoked",
                   "revoked_cert", "auth", "revocation list", claimed, method=method)
        if c.get("not_after") and c["not_after"] < time.time():
            reject(context, grpc.StatusCode.UNAUTHENTICATED, "certificate expired", "expired_cert", "auth",
                   "certificate validity", claimed, method=method)
        if pol is not None and claimed is None and pol.roles:
            # v2.2 skipped the identity binding when the message named nobody (claim ""), RT-17
            reject(context, grpc.StatusCode.PERMISSION_DENIED, "message does not name its sender",
                   "identity_mismatch", "auth", "certificate-bound identity", claimed, method=method)
        if claimed is not None and claimed != c["name"]:
            reject(context, grpc.StatusCode.PERMISSION_DENIED,
                   f"certificate identity {c['name']!r} does not match claimed sender {claimed!r}",
                   "identity_mismatch", "auth", "certificate-bound identity", claimed, method=method)
        role = effective_role(c, self.pki)
        if pol and pol.roles and role not in pol.roles:
            # v2.2 only checked roles for certificates that carried one, i.e. never in static mode (RT-01)
            reject(context, grpc.StatusCode.PERMISSION_DENIED,
                   f"role {role!r} may not call {method}", "bad_role", "auth", "certificate role", claimed,
                   method=method)


def _safe_claim(pol, request):
    if not pol:
        return None
    try:
        v = pol.claim(request)
        return v or None
    except Exception:
        return None


def _key_matches(key_pem: bytes, cert_pem: bytes) -> bool:
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    try:
        k = serialization.load_pem_private_key(key_pem, None).public_key()
        c = x509.load_pem_x509_certificate(cert_pem).public_key()
        f = serialization.PublicFormat.SubjectPublicKeyInfo
        return k.public_bytes(serialization.Encoding.DER, f) == c.public_bytes(serialization.Encoding.DER, f)
    except Exception:
        return False


def _server_credentials(tls: TLSFiles, require_client_auth: bool = True):
    """Credentials that re-read the node's key/certificate when the files change (auto-renewal)."""
    ca, key, crt = tls.read()
    state = {"mtime": tls.mtime()}

    def fetch():
        m = tls.mtime()
        if m != state["mtime"]:
            try:
                ca2, key2, crt2 = tls.read()
                if not _key_matches(key2, crt2):        # files mid-replacement: try on the next handshake
                    return None
            except OSError:
                return None
            state["mtime"] = m
            return grpc.ssl_server_certificate_configuration([(key2, crt2)], root_certificates=ca2)
        return None

    initial = grpc.ssl_server_certificate_configuration([(key, crt)], root_certificates=ca)
    return grpc.dynamic_ssl_server_credentials(initial, fetch, require_client_authentication=require_client_auth)


def make_server(tls: TLSFiles, listen: str, workers: int = DEFAULT_WORKERS, gate: SecurityGate | None = None,
                max_concurrent_rpcs: int | None = DEFAULT_MAX_CONCURRENT_RPCS, require_client_auth: bool = True):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers), options=_SERVER_OPTS,
                         interceptors=[gate] if gate else None, maximum_concurrent_rpcs=max_concurrent_rpcs)
    if tls.enabled:
        port = server.add_secure_port(listen, _server_credentials(tls, require_client_auth))
    else:
        port = server.add_insecure_port(listen)
    if not port:
        raise RuntimeError(f"could not bind gRPC server on {listen}")
    return server, port


def make_channel(tls: TLSFiles, target: str, server_name: str, role: str | None = None):
    """`server_name` is the peer's node id. In pki mode the client pins <id>.<role>.vfl.<domain>, where role is
    the role the server must hold (coordinator, or active: parties only ever dial those two)."""
    if tls.enabled:
        ca, key, crt = tls.read()
        creds = grpc.ssl_channel_credentials(root_certificates=ca, private_key=key, certificate_chain=crt)
        pinned = tls.server_name(server_name, role or server_role(server_name))
        return grpc.secure_channel(target, creds, options=_OPTS + [("grpc.ssl_target_name_override", pinned)])
    return grpc.insecure_channel(target, options=_OPTS)


def check_identity(context, tls: TLSFiles, claimed: str):
    """Kept for compatibility: the SecurityGate performs this check for every method."""
    if not tls.enabled:
        return
    if peer_identity(context) != claimed:
        reject(context, grpc.StatusCode.PERMISSION_DENIED,
               f"certificate identity {peer_identity(context)!r} does not match claimed sender {claimed!r}",
               "identity_mismatch", "auth", "certificate-bound identity", claimed)
