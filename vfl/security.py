"""Certificates and identities (FR-11; v2 items 17-19).

Two modes, chosen by `vfl.cli certs --mode`:

static (v1 default)
    One private CA signs one long-lived certificate per node; the Common Name is the node id.
    Certificates are copied to each machine by hand.

pki (v2)
    root/ca.key       OFFLINE root (move it off the coordinator after setup)
    intermediate.*    online issuing CA held by the coordinator, NAME-CONSTRAINED to
                      DNS *.vfl.<domain> and URIs under spiffe://<domain>/, path length 0
    leaf certificates short-lived (24 h by default) and renewed automatically; identity and role
                      are in the SAN URI  spiffe://<domain>/vfl/<role>/<name>  and DNS <name>.vfl.<domain>.
                      Parties generate their own keys and get certificates through enrollment
                      (static | token | approval); the security gate trusts the CERTIFICATE's
                      name and role, never the claim inside a message. Revoked serials are rejected.

mTLS protects against network observers and unauthenticated peers. It does NOT stop an authorised
recipient from inferring information from tensors it legitimately receives.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import ipaddress
import json
import os
import secrets
import time

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

TRUST_DOMAIN = "oncloud9"
ROLES = ("coordinator", "active", "passive")


def _now():
    return dt.datetime.now(dt.timezone.utc)


def _key():
    return ec.generate_private_key(ec.SECP256R1())


def _pem_key(k) -> bytes:
    return k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                           serialization.NoEncryption())


def _pem(cert) -> bytes:
    return cert.public_bytes(serialization.Encoding.PEM)


def _write(path: str, data: bytes, private: bool = False):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    if private:
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def fingerprint(cert_pem: bytes) -> str:
    cert = x509.load_pem_x509_certificate(cert_pem)
    return hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()


def dns_name(node: str, domain: str = TRUST_DOMAIN, role: str | None = None) -> str:
    """pki DNS SAN. v2.3 puts the ROLE in the name (<node>.<role>.vfl.<domain>) and clients pin it, so a passive
    certificate can never serve as the coordinator or the active party, whatever name it was enrolled under
    (v2.2 pinned <node>.vfl.<domain>, which a passive enrolled as 'coordinator' satisfied)."""
    return f"{node}.{role}.vfl.{domain}" if role else f"{node}.vfl.{domain}"


def spiffe_uri(node: str, role: str, domain: str = TRUST_DOMAIN) -> str:
    return f"spiffe://{domain}/vfl/{role}/{node}"


def role_of_node(node: str) -> str:
    return "coordinator" if node == "coordinator" else "active" if node == "active" else "passive"


def server_role(peer: str) -> str:
    """Clients only ever dial the coordinator or the active party."""
    return "coordinator" if peer == "coordinator" else "active"


def effective_role(info: dict | None, pki: bool) -> str | None:
    """The role a certificate speaks for. pki and v2.3 static certificates carry it in the SAN URI. A legacy
    (v2.2) static certificate has none: its role follows from its name (coordinator / active / anything else =
    passive), which is how the static certificate set was issued. v2.2 skipped every role check for such
    certificates, so a passive certificate could register as the active party (RT-01)."""
    if not info:
        return None
    if info.get("role"):
        return info["role"]
    return None if pki else role_of_node(info.get("name") or "")


# ------------------------------------------------------------------ static mode (v1)
def make_ca(out_dir: str, name: str = "VFL Private CA", days: int = 365):
    os.makedirs(out_dir, exist_ok=True)
    key = _key()
    subj = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OnCloud9 VFL")])
    now = _now()
    cert = (x509.CertificateBuilder().subject_name(subj).issuer_name(subj).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=days))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                         content_commitment=False, key_encipherment=False,
                                         data_encipherment=False, key_agreement=False,
                                         encipher_only=False, decipher_only=False), critical=True)
            .sign(key, hashes.SHA256()))
    _write(os.path.join(out_dir, "ca.crt"), _pem(cert))
    _write(os.path.join(out_dir, "ca.key"), _pem_key(key), private=True)
    return key, cert


def load_ca(out_dir: str):
    with open(os.path.join(out_dir, "ca.key"), "rb") as fh:
        key = serialization.load_pem_private_key(fh.read(), None)
    with open(os.path.join(out_dir, "ca.crt"), "rb") as fh:
        cert = x509.load_pem_x509_certificate(fh.read())
    return key, cert


def make_node_cert(out_dir: str, node: str, ca_key, ca_cert, hosts: list[str] | None = None, days: float = 365,
                   not_before=None, role: str | None = None):
    key = _key()
    role = role or role_of_node(node)
    if role not in ROLES:
        raise ValueError(f"bad role {role!r}")
    # v2.3: the role is in the certificate (SAN URI), so the gate checks it in static mode too (RT-01)
    sans = [x509.DNSName(node), x509.DNSName("localhost"), x509.UniformResourceIdentifier(spiffe_uri(node, role))]
    for h in ["127.0.0.1"] + list(hosts or []):
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            sans.append(x509.DNSName(h))
    nb = not_before or (_now() - dt.timedelta(minutes=5))
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node)]))
            .issuer_name(ca_cert.subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(nb)
            .not_valid_after(nb + dt.timedelta(days=days))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH,
                                                  ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256()))
    _write(os.path.join(out_dir, f"{node}.crt"), _pem(cert))
    _write(os.path.join(out_dir, f"{node}.key"), _pem_key(key), private=True)
    return cert


def generate_all(out_dir: str, nodes: list[str], hosts: list[str] | None = None,
                 roles: dict | None = None) -> list[str]:
    if os.path.exists(os.path.join(out_dir, "ca.key")):
        ca_key, ca_cert = load_ca(out_dir)
    else:
        ca_key, ca_cert = make_ca(out_dir)
    for n in nodes:
        make_node_cert(out_dir, n, ca_key, ca_cert, hosts, role=(roles or {}).get(n))
    return sorted(os.listdir(out_dir))


# ------------------------------------------------------------------ pki mode (v2)
def make_pki(out_dir: str, domain: str = TRUST_DOMAIN, root_days: int = 3650, int_days: int = 365) -> dict:
    """Offline root + name-constrained online intermediate. Root key goes to <out>/root/."""
    os.makedirs(os.path.join(out_dir, "root"), exist_ok=True)
    rkey = _key()
    rname = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"VFL Root CA ({domain})"),
                       x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OnCloud9 VFL")])
    now = _now()
    ku_ca = x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                          key_encipherment=False, data_encipherment=False, key_agreement=False,
                          encipher_only=False, decipher_only=False)
    root = (x509.CertificateBuilder().subject_name(rname).issuer_name(rname).public_key(rkey.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=root_days))
            .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
            .add_extension(ku_ca, critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(rkey.public_key()), critical=False)
            .sign(rkey, hashes.SHA256()))
    ikey = _key()
    iname = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"VFL Issuing CA ({domain})"),
                       x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OnCloud9 VFL")])
    nc = x509.NameConstraints(permitted_subtrees=[x509.DNSName(f"vfl.{domain}"),
                                                  x509.UniformResourceIdentifier(domain)],
                              excluded_subtrees=None)
    inter = (x509.CertificateBuilder().subject_name(iname).issuer_name(rname).public_key(ikey.public_key())
             .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
             .not_valid_after(now + dt.timedelta(days=int_days))
             .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
             .add_extension(ku_ca, critical=True)
             .add_extension(nc, critical=True)
             .add_extension(x509.SubjectKeyIdentifier.from_public_key(ikey.public_key()), critical=False)
             .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(rkey.public_key()), critical=False)
             .sign(rkey, hashes.SHA256()))
    _write(os.path.join(out_dir, "root", "ca.key"), _pem_key(rkey), private=True)
    _write(os.path.join(out_dir, "ca.crt"), _pem(root))                      # trust anchor for everyone
    _write(os.path.join(out_dir, "intermediate.crt"), _pem(inter))
    _write(os.path.join(out_dir, "intermediate.key"), _pem_key(ikey), private=True)
    meta = {"mode": "pki", "trust_domain": domain, "root_sha256": fingerprint(_pem(root)),
            "created": int(time.time())}
    _write(os.path.join(out_dir, "pki.json"), json.dumps(meta, indent=2).encode())
    return meta


def load_issuer(pki_dir: str):
    with open(os.path.join(pki_dir, "intermediate.key"), "rb") as fh:
        ikey = serialization.load_pem_private_key(fh.read(), None)
    with open(os.path.join(pki_dir, "intermediate.crt"), "rb") as fh:
        icert_pem = fh.read()
    with open(os.path.join(pki_dir, "ca.crt"), "rb") as fh:
        root_pem = fh.read()
    meta = json.load(open(os.path.join(pki_dir, "pki.json")))
    return ikey, x509.load_pem_x509_certificate(icert_pem), icert_pem, root_pem, meta


def make_csr(node: str, key=None) -> tuple[bytes, bytes]:
    """(key_pem, csr_pem) - the key is generated locally and never leaves the party."""
    key = key or _key()
    csr = (x509.CertificateSigningRequestBuilder()
           .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node)]))
           .sign(key, hashes.SHA256()))
    return _pem_key(key), csr.public_bytes(serialization.Encoding.PEM)


def issue_leaf(ikey, icert, public_key, node: str, role: str, hours: float = 24.0, domain: str = TRUST_DOMAIN,
               not_before=None):
    if role not in ROLES:
        raise ValueError(f"bad role {role!r}")
    nb = not_before or (_now() - dt.timedelta(minutes=5))
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node)]))
            .issuer_name(icert.subject).public_key(public_key)
            .serial_number(x509.random_serial_number()).not_valid_before(nb)
            .not_valid_after(nb + dt.timedelta(hours=hours))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(dns_name(node, domain, role)),
                                                        x509.UniformResourceIdentifier(spiffe_uri(node, role, domain))]),
                           critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH,
                                                  ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ikey.public_key()), critical=False)
            .sign(ikey, hashes.SHA256()))
    return cert


def issue_from_csr(pki_dir: str, csr_pem: bytes, node: str, role: str, hours: float = 24.0):
    csr = x509.load_pem_x509_csr(csr_pem)
    if not csr.is_signature_valid:
        raise ValueError("CSR signature invalid")
    cn = csr.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    if not cn or cn[0].value != node:
        raise ValueError(f"CSR common name {cn[0].value if cn else None!r} != requested id {node!r}")
    ikey, icert, icert_pem, root_pem, meta = load_issuer(pki_dir)
    cert = issue_leaf(ikey, icert, csr.public_key(), node, role, hours, meta["trust_domain"])
    return cert, icert_pem, root_pem


def issue_static_pki(pki_dir: str, node: str, role: str | None = None, hours: float = 24.0 * 30,
                     not_before=None) -> dict:
    """Issue a leaf (key generated here) for hand distribution in pki mode."""
    ikey, icert, icert_pem, root_pem, meta = load_issuer(pki_dir)
    key = _key()
    cert = issue_leaf(ikey, icert, key.public_key(), node, role or role_of_node(node), hours, meta["trust_domain"],
                      not_before)
    _write(os.path.join(pki_dir, f"{node}.key"), _pem_key(key), private=True)
    _write(os.path.join(pki_dir, f"{node}.crt"), _pem(cert) + icert_pem)
    return describe_cert(_pem(cert))


def sign_bytes(key_pem: bytes, data: bytes) -> bytes:
    key = serialization.load_pem_private_key(key_pem, None)
    return key.sign(data, ec.ECDSA(hashes.SHA256()))


def verify_renewal(pki_dir: str, cert_pem: bytes, csr_pem: bytes, signature: bytes) -> tuple[str, str, str]:
    """Check the presented leaf was issued by our intermediate, is unexpired, and that the caller holds
    its key (signature over the new CSR). Returns (name, role, serial)."""
    leaf = x509.load_pem_x509_certificate(cert_pem)
    ikey, icert, _, _, _ = load_issuer(pki_dir)
    try:
        icert.public_key().verify(leaf.signature, leaf.tbs_certificate_bytes, ec.ECDSA(leaf.signature_hash_algorithm))
    except InvalidSignature:
        raise ValueError("certificate was not issued by this coordinator's CA")
    if leaf.not_valid_after_utc < _now():
        raise ValueError("certificate expired; re-enroll with a token")
    try:
        leaf.public_key().verify(signature, csr_pem, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        raise ValueError("proof-of-possession signature invalid")
    info = describe_cert(cert_pem)
    if not info["role"]:
        raise ValueError("certificate has no VFL role")
    return info["name"], info["role"], info["serial"]


# ------------------------------------------------------------------ identity parsing
_CACHE: dict = {}
_CACHE_MAX = 4096


def describe_cert(pem: bytes) -> dict:
    """{name, role, serial, not_after, issuer, uri} from a PEM leaf (first cert in a chain)."""
    key = hashlib.sha1(pem).digest()
    if key in _CACHE:
        return _CACHE[key]
    cert = x509.load_pem_x509_certificate(pem)
    cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    name, role, uri = (cn[0].value if cn else None), None, None
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        for u in san.get_values_for_type(x509.UniformResourceIdentifier):
            if u.startswith("spiffe://") and "/vfl/" in u:
                parts = u.split("/vfl/", 1)[1].split("/")
                if len(parts) == 2:
                    role, name, uri = parts[0], parts[1], u
    except x509.ExtensionNotFound:
        pass
    info = {"name": name, "role": role, "uri": uri, "serial": format(cert.serial_number, "x"),
            "not_before": cert.not_valid_before_utc.timestamp(), "not_after": cert.not_valid_after_utc.timestamp(),
            "issuer": cert.issuer.rfc4514_string()}
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = info
    return info


def leaf_dns_names(pem: bytes) -> list[str]:
    try:
        cert = x509.load_pem_x509_certificate(pem)
        return cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(
            x509.DNSName)
    except Exception:
        return []


# ------------------------------------------------------------------ tokens
def make_token(ca_sha256: str) -> tuple[str, str, str]:
    """Returns (token, token_id, secret_hash). token = vfl1.<id>.<secret>.<ca-fingerprint-prefix>."""
    tid = secrets.token_hex(4)
    secret = base64.urlsafe_b64encode(secrets.token_bytes(18)).decode().rstrip("=")
    return f"vfl1.{tid}.{secret}.{ca_sha256[:16]}", tid, hashlib.sha256(secret.encode()).hexdigest()


def parse_token(token: str) -> dict:
    parts = (token or "").split(".")
    if len(parts) != 4 or parts[0] != "vfl1":
        raise ValueError("malformed join token")
    return {"id": parts[1], "secret_hash": hashlib.sha256(parts[2].encode()).hexdigest(), "ca_prefix": parts[3]}


# ------------------------------------------------------------------ files for one node
class TLSFiles:
    """Paths for one node; `None` dir => insecure (local development only)."""

    def __init__(self, tls_dir: str | None, node: str):
        self.dir = tls_dir
        self.node = node

    @property
    def enabled(self) -> bool:
        return bool(self.dir)

    @property
    def pki(self) -> dict | None:
        if not self.dir:
            return None
        p = os.path.join(self.dir, "pki.json")
        return json.load(open(p)) if os.path.exists(p) else None

    def server_name(self, peer: str, role: str | None = None) -> str:
        """Name a client pins when talking to `peer` (certificate DNS SAN). In pki mode it includes the role the
        peer must hold (coordinator, or active for every party server)."""
        meta = self.pki
        return dns_name(peer, meta["trust_domain"], role or server_role(peer)) if meta else peer

    def paths(self):
        return (os.path.join(self.dir, "ca.crt"), os.path.join(self.dir, f"{self.node}.key"),
                os.path.join(self.dir, f"{self.node}.crt"))

    def read(self):
        out = []
        for p in self.paths():
            with open(p, "rb") as fh:
                out.append(fh.read())
        return tuple(out)

    def mtime(self) -> float:
        try:
            return max(os.path.getmtime(p) for p in self.paths()[1:])
        except OSError:
            return 0.0

    def leaf_info(self) -> dict | None:
        try:
            return describe_cert(self.read()[2])
        except Exception:
            return None
