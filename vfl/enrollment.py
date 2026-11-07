"""Identity registry and enrollment service (v2 items 17-19), run by the coordinator in pki mode.

Join modes (switchable at runtime from the Identities panel or `--join-mode`):
  static    enrollment disabled; certificates are distributed by hand (v1 behaviour)
  token     a party presents a single-use, expiring token; the token also carries the root CA
            fingerprint so the party can verify it is talking to the right coordinator
  approval  the request appears under "Join requests"; an operator approves or denies it

State lives in <run_dir>/pki/: tokens.json (only SHA-256 of token secrets), issued.json, revoked.json,
requests.json. Issued certificates are short-lived (default 24 h) and renewed by the parties
(RenewRequest = new CSR signed with the current key). Revoked serials are rejected by every node's
security gate (parties receive the list in heartbeat replies).
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

import grpc

from . import security as sec
from .proto import vfl_pb2 as pb
from .proto import vfl_pb2_grpc as pbg
from .telemetry import SECURITY
from .transport import peer_ip

JOIN_MODES = ("static", "token", "approval")
RESERVED_NAMES = ("coordinator",)
MAX_CSR_BYTES = 8192
MAX_PENDING = 50
MAX_PENDING_PER_IP = 5
MAX_TOKEN_TTL_S = 7 * 24 * 3600


class IdentityRegistry:
    def __init__(self, pki_dir: str, state_dir: str, join_mode: str = "static", cert_hours: float = 24.0):
        self.pki_dir = pki_dir
        self.dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.lock = threading.RLock()
        self.join_mode = join_mode if join_mode in JOIN_MODES else "static"
        self.cert_hours = float(cert_hours)
        self.meta = json.load(open(os.path.join(pki_dir, "pki.json")))
        self.tokens = self._load("tokens.json", {})
        self.issued = self._load("issued.json", {})
        self.revoked = set(self._load("revoked.json", []))
        self.requests = self._load("requests.json", {})

    # ------------------------------------------------------------ persistence
    def _load(self, name, default):
        try:
            return json.load(open(os.path.join(self.dir, name)))
        except Exception:
            return default

    def _save(self):
        for name, obj in (("tokens.json", self.tokens), ("issued.json", self.issued),
                          ("revoked.json", sorted(self.revoked)), ("requests.json", self.requests)):
            tmp = os.path.join(self.dir, name + ".tmp")
            with open(tmp, "w") as fh:
                json.dump(obj, fh, indent=2)
            os.replace(tmp, os.path.join(self.dir, name))

    # ------------------------------------------------------------ operator actions
    def create_token(self, role: str = "passive", name: str | None = None, ttl_s: float = 3600) -> dict:
        if role not in ("active", "passive"):
            raise ValueError("token role must be active or passive")
        if not float(ttl_s) <= MAX_TOKEN_TTL_S:          # (a non-positive ttl makes an already-expired token)
            raise ValueError(f"token lifetime must be at most {MAX_TOKEN_TTL_S // 3600} h")
        if name:
            self._check_name(name)
        token, tid, h = sec.make_token(self.meta["root_sha256"])
        with self.lock:
            self.tokens[tid] = {"id": tid, "secret_sha256": h, "role": role, "name": name,
                                "created": time.time(), "expires": time.time() + float(ttl_s), "used": False}
            self._save()
        return {"token": token, "id": tid, "role": role, "name": name, "expires": self.tokens[tid]["expires"]}

    def set_mode(self, mode: str):
        if mode not in JOIN_MODES:
            raise ValueError(f"join mode must be one of {JOIN_MODES}")
        self.join_mode = mode

    def decide(self, request_id: str, approve: bool):
        with self.lock:
            r = self.requests.get(request_id)
            if not r or r["status"] != "pending":
                raise ValueError("no such pending request")
            r["status"] = "approved" if approve else "denied"
            r["decided"] = time.time()
            self._save()
            return r

    def revoke(self, serial: str, reason: str = "revoked by operator"):
        """Revoke an IDENTITY: the chosen certificate and every other unexpired certificate issued to the same
        name before now (renewals). v2.2 revoked one serial, so a party that had already renewed kept a valid
        certificate and could renew it forever (RT-06)."""
        with self.lock:
            if serial not in self.issued:
                raise ValueError(f"unknown serial {serial}")
            name, now = self.issued[serial]["name"], time.time()
            for ser, rec in self.issued.items():
                if ser == serial or (rec.get("name") == name and not rec.get("revoked")
                                     and (rec.get("not_after") or now + 1) > now):
                    self.revoked.add(ser)
                    rec["revoked"] = now
                    rec["revoke_reason"] = reason if ser == serial else f"{reason} (same identity as {serial})"
            self._save()
        SECURITY.event("revocation", "enrollment", "detected", "revocation list", "server-observed",
                       claimed=self.issued[serial]["name"], cert_identity=self.issued[serial]["name"],
                       detail=f"serial {serial} revoked: {reason}")
        return self.issued[serial]

    def record_static(self, info: dict, via: str = "static"):
        with self.lock:
            self.issued.setdefault(info["serial"], {**info, "via": via, "issued": time.time(), "revoked": None})
            self._save()

    def view(self) -> dict:
        now = time.time()
        with self.lock:
            return {"mode": self.join_mode, "trust_domain": self.meta["trust_domain"],
                    "root_sha256": self.meta["root_sha256"], "cert_hours": self.cert_hours,
                    "identities": sorted(self.issued.values(), key=lambda x: -x.get("issued", 0))[:200],
                    "revoked": sorted(self.revoked),
                    "requests": sorted([r for r in self.requests.values()], key=lambda x: -x["t"])[:100],
                    "tokens": [{k: v for k, v in t.items() if k != "secret_sha256"} | {"expired": t["expires"] < now}
                               for t in sorted(self.tokens.values(), key=lambda x: -x["created"])[:100]]}

    # ------------------------------------------------------------ checks
    def _check_name(self, name: str):
        """Names a new enrollment may take. 'coordinator' is the coordinator's own identity, lab test identities
        are for the attack simulator, and a name that already holds a valid certificate belongs to someone: v2.2
        let an unnamed token mint any of these (RT-05)."""
        from .ids import check_party_id, is_lab_identity
        check_party_id(name)
        if name in RESERVED_NAMES or is_lab_identity(name):
            raise ValueError(f"{name!r} is a reserved identity")

    def _in_use(self, name: str) -> str | None:
        now = time.time()
        for ser, rec in self.issued.items():
            if rec.get("name") == name and not rec.get("revoked") and ser not in self.revoked \
                    and (rec.get("not_after") or 0) > now:
                return ser
        return None

    @staticmethod
    def _check_csr(csr_pem: bytes, name: str):
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        if not csr_pem or len(csr_pem) > MAX_CSR_BYTES:
            raise ValueError("missing or oversized CSR")
        try:
            csr = x509.load_pem_x509_csr(csr_pem)
        except Exception:
            raise ValueError("CSR is not a PEM certificate request")
        if not csr.is_signature_valid:
            raise ValueError("CSR signature invalid")
        cn = csr.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        if not cn or cn[0].value != name:
            raise ValueError("CSR common name does not match the requested party id")

    # ------------------------------------------------------------ issuing
    def _issue(self, csr_pem: bytes, name: str, role: str, via: str):
        cert, chain, root = sec.issue_from_csr(self.pki_dir, csr_pem, name, role, self.cert_hours)
        pem = sec._pem(cert)
        info = sec.describe_cert(pem)
        with self.lock:
            self.issued[info["serial"]] = {**info, "via": via, "issued": time.time(), "revoked": None}
            self._save()
        return pb.EnrollReply(status="issued", cert_pem=pem, chain_pem=chain, root_pem=root,
                              not_after_unix=int(info["not_after"]))

    def enroll(self, req, context) -> pb.EnrollReply:
        ip = peer_ip(context)
        name, role = (req.party_id or "").strip(), req.role
        if self.join_mode == "static":
            SECURITY.event("enroll_disabled", "enrollment", "blocked", "join mode static", source_ip=ip, claimed=name,
                           method="Enroll", detail="enrollment attempted while join mode is static")
            return pb.EnrollReply(status="error", error="enrollment is disabled (join mode static)")
        if not name or len(name) > 64 or any(c in name for c in "/\\ \t\n.") or role not in ("active", "passive"):
            return pb.EnrollReply(status="error", error="invalid party id or role")
        try:
            self._check_name(name)
            if not req.request_id:
                self._check_csr(req.csr_pem, name)         # before a token is spent or a request is stored
        except ValueError as e:
            SECURITY.event("bad_enrollment", "enrollment", "blocked", "enrollment request validation", source_ip=ip,
                           claimed=name, method="Enroll", detail=str(e))
            return pb.EnrollReply(status="error", error=str(e))
        if self.join_mode == "token":
            return self._enroll_token(req, ip, name, role)
        return self._enroll_approval(req, ip, name, role)

    def _squatting(self, ip, name):
        held = self._in_use(name)
        if not held:
            return None
        SECURITY.event("identity_squatting", "enrollment", "blocked", "one valid certificate per identity",
                       source_ip=ip, claimed=name, method="Enroll", detail=f"{name} already holds serial {held}")
        return pb.EnrollReply(status="denied", error=f"{name!r} already has a valid certificate "
                                                      "(revoke it first to re-enroll)")

    def _enroll_token(self, req, ip, name, role):
        def deny(kind, msg):
            SECURITY.event(kind, "enrollment", "blocked", "single-use expiring join token", source_ip=ip,
                           claimed=name, method="Enroll", detail=msg)
            return pb.EnrollReply(status="denied", error=msg)
        if not req.token:
            return deny("no_token", "a join token is required")
        try:
            t = sec.parse_token(req.token)
        except ValueError as e:
            return deny("bad_token", str(e))
        with self.lock:
            rec = self.tokens.get(t["id"])
            if not rec or rec["secret_sha256"] != t["secret_hash"]:
                return deny("bad_token", "unknown token")
            if rec["used"]:
                return deny("reused_token", f"token {t['id']} was already used by {rec.get('used_by')}")
            if rec["expires"] < time.time():
                return deny("expired_token", f"token {t['id']} expired")
            if t["ca_prefix"] != self.meta["root_sha256"][:16]:
                return deny("bad_token", "token was issued for a different CA")
            if rec["role"] != role or (rec.get("name") and rec["name"] != name):
                return deny("token_scope", f"token is for role={rec['role']} name={rec.get('name')}")
            squat = self._squatting(ip, name)
            if squat:
                return squat
            rec["used"], rec["used_by"], rec["used_at"] = True, name, time.time()
            self._save()
        try:
            return self._issue(req.csr_pem, name, role, f"token {t['id']}")
        except ValueError as e:
            return pb.EnrollReply(status="error", error=str(e))

    def _enroll_approval(self, req, ip, name, role):
        squat = self._squatting(ip, name)
        if squat:
            return squat
        with self.lock:
            rid = req.request_id
            if rid:
                r = self.requests.get(rid)
                if not r or r["name"] != name:
                    return pb.EnrollReply(status="error", error="unknown request id")
                if r["status"] == "pending":
                    return pb.EnrollReply(status="pending", request_id=rid)
                if r["status"] == "denied":
                    return pb.EnrollReply(status="denied", error="request denied by operator", request_id=rid)
                if r["status"] == "issued":
                    return pb.EnrollReply(status="error", error="request already fulfilled", request_id=rid)
                r["status"] = "issued"
                self._save()
                csr = r["csr_pem"].encode()
            else:
                pending = [x for x in self.requests.values() if x["status"] == "pending"]
                if len(pending) >= MAX_PENDING or sum(x.get("source_ip") == ip for x in pending) >= MAX_PENDING_PER_IP:
                    SECURITY.event("join_flood", "enrollment", "blocked", "pending-request cap", source_ip=ip,
                                   claimed=name, method="Enroll", detail=f"{len(pending)} requests already pending")
                    return pb.EnrollReply(status="denied", error="too many pending join requests; try later")
                rid = uuid.uuid4().hex
                from .ids import one_line
                self.requests[rid] = {"id": rid, "name": name, "role": role, "hostname": one_line(req.hostname, 128),
                                      "source_ip": ip, "t": time.time(), "status": "pending",
                                      "csr_pem": req.csr_pem.decode()}
                self._save()
                SECURITY.event("join_request", "enrollment", "detected", "operator approval", source_ip=ip,
                               claimed=name, method="Enroll", detail=f"{name} ({role}) asks to join from {req.hostname}")
                return pb.EnrollReply(status="pending", request_id=rid)
        try:
            rep = self._issue(csr, name, role, f"approval {rid}")
            rep.request_id = rid
            return rep
        except ValueError as e:
            return pb.EnrollReply(status="error", error=str(e))

    def renew(self, req, context) -> pb.EnrollReply:
        ip = peer_ip(context)
        try:
            name, role, serial = sec.verify_renewal(self.pki_dir, req.cert_pem, req.csr_pem, req.signature)
        except ValueError as e:
            SECURITY.event("bad_renewal", "enrollment", "blocked", "proof of possession + issuer check",
                           source_ip=ip, method="Renew", detail=str(e))
            return pb.EnrollReply(status="denied", error=str(e))
        if serial in self.revoked or (serial in self.issued and self.issued[serial].get("revoked")):
            SECURITY.event("revoked_cert", "enrollment", "blocked", "revocation list", source_ip=ip, claimed=name,
                           cert_identity=name, method="Renew", detail=f"renewal with revoked serial {serial}")
            return pb.EnrollReply(status="denied", error="certificate revoked")
        try:
            return self._issue(req.csr_pem, name, role, f"renewal of {serial}")
        except ValueError as e:
            return pb.EnrollReply(status="error", error=str(e))


class EnrollmentServicer(pbg.EnrollmentServicer):
    def __init__(self, reg: IdentityRegistry):
        self.reg = reg

    def Enroll(self, request, context):
        return self.reg.enroll(request, context)

    def Renew(self, request, context):
        return self.reg.renew(request, context)


# ------------------------------------------------------------------ party side
def fetch_root(enroll_url: str | None, ca_file: str | None, token: str | None,
               ca_sha256: str | None = None) -> bytes:
    """Root CA for the enrollment channel. A root fetched from the dashboard must be pinned, by the join token's
    fingerprint prefix or by --ca-sha256: v2.2 trusted whatever a plain-HTTP fetch returned in approval mode, so
    anyone on the path could hand the party their own CA (RT-13)."""
    if ca_file:
        root = open(ca_file, "rb").read()
    elif enroll_url:
        if not token and not ca_sha256:
            raise ValueError("refusing to trust a root CA fetched over the network without a pin: pass the join "
                             "token, --ca-sha256 (the fingerprint the coordinator printed) or --ca-file")
        import urllib.request
        with urllib.request.urlopen(enroll_url.rstrip("/") + "/api/pki/root.pem", timeout=15) as r:
            root = r.read()
    else:
        raise ValueError("need --ca-file or --enroll-url to obtain the root CA certificate")
    if token:
        want = sec.parse_token(token)["ca_prefix"]
        got = sec.fingerprint(root)[:16]
        if want != got:
            raise ValueError(f"root CA fingerprint {got} does not match the token ({want}); refusing to enroll")
    if ca_sha256:
        import hmac
        got = sec.fingerprint(root)
        if not hmac.compare_digest(got, ca_sha256.lower().replace(":", "")):
            raise ValueError(f"root CA fingerprint {got} does not match --ca-sha256; refusing to enroll")
    return root


def enroll_party(tls_dir: str, party_id: str, role: str, enroll_addr: str, root_pem: bytes,
                 token: str | None = None, poll_s: float = 3.0, timeout_s: float = 3600, log=print) -> dict:
    """Generate a key locally, send a CSR, wait for approval if needed, and install the certificate."""
    key_pem, csr_pem = sec.make_csr(party_id)
    creds = grpc.ssl_channel_credentials(root_certificates=root_pem)
    domain = None
    ch = grpc.secure_channel(enroll_addr, creds, options=[("grpc.ssl_target_name_override",
                                                           sec.dns_name("coordinator", _domain_hint(root_pem),
                                                                        "coordinator"))])
    stub = pbg.EnrollmentStub(ch)
    req = pb.EnrollRequest(party_id=party_id, role=role, csr_pem=csr_pem, token=token or "",
                           hostname=os.uname().nodename)
    t0 = time.time()
    while True:
        rep = stub.Enroll(req, timeout=30)
        if rep.status == "issued":
            break
        if rep.status == "pending":
            if not req.request_id:
                log(f"join request {rep.request_id} is waiting for operator approval on the coordinator dashboard")
            req.request_id = rep.request_id
            if time.time() - t0 > timeout_s:
                raise TimeoutError("join request was not approved in time")
            time.sleep(poll_s)
            continue
        raise PermissionError(f"enrollment {rep.status}: {rep.error}")
    ch.close()
    install(tls_dir, party_id, key_pem, rep.cert_pem + rep.chain_pem, rep.root_pem, domain or _domain_hint(rep.root_pem))
    return sec.describe_cert(rep.cert_pem)


def _domain_hint(root_pem: bytes) -> str:
    from cryptography import x509
    cn = x509.load_pem_x509_certificate(root_pem).subject.rfc4514_string()
    if "(" in cn and ")" in cn:
        return cn.split("(", 1)[1].split(")", 1)[0]
    return sec.TRUST_DOMAIN


def install(tls_dir: str, node: str, key_pem: bytes, chain_pem: bytes, root_pem: bytes, domain: str):
    os.makedirs(tls_dir, exist_ok=True)
    sec._write(os.path.join(tls_dir, "ca.crt"), root_pem)
    sec._write(os.path.join(tls_dir, f"{node}.key"), key_pem, private=True)
    sec._write(os.path.join(tls_dir, f"{node}.crt"), chain_pem)
    if not os.path.exists(os.path.join(tls_dir, "pki.json")):
        sec._write(os.path.join(tls_dir, "pki.json"), json.dumps(
            {"mode": "pki", "trust_domain": domain, "root_sha256": sec.fingerprint(root_pem)}).encode())


def renew_party(tls: sec.TLSFiles, enroll_addr: str) -> dict:
    ca, key_pem, chain = tls.read()
    new_key_pem, csr_pem = sec.make_csr(tls.node)
    sig = sec.sign_bytes(key_pem, csr_pem)
    ch = grpc.secure_channel(enroll_addr, grpc.ssl_channel_credentials(root_certificates=ca),
                             options=[("grpc.ssl_target_name_override", tls.server_name("coordinator"))])
    rep = pbg.EnrollmentStub(ch).Renew(pb.RenewRequest(cert_pem=chain, csr_pem=csr_pem, signature=sig), timeout=30)
    ch.close()
    if rep.status != "issued":
        raise PermissionError(f"renewal {rep.status}: {rep.error}")
    install(tls.dir, tls.node, new_key_pem, rep.cert_pem + rep.chain_pem, rep.root_pem, (tls.pki or {}).get(
        "trust_domain", sec.TRUST_DOMAIN))
    return sec.describe_cert(rep.cert_pem)
