"""v2 items 15, 17: token enrollment, single-use and expiry."""
import pytest

from vfl import security as sec
from vfl.enrollment import IdentityRegistry
from vfl.proto import vfl_pb2 as pb


class _Ctx:
    def peer(self): return "ipv4:10.0.0.5:4000"
    def auth_context(self): return {}


def _reg(tmp_path, mode="token"):
    d = str(tmp_path / "pki")
    sec.make_pki(d)
    return IdentityRegistry(d, str(tmp_path / "state"), mode)


def _enroll(reg, name, token):
    _, csr = sec.make_csr(name)
    return reg.enroll(pb.EnrollRequest(party_id=name, role="passive", csr_pem=csr, token=token), _Ctx())


def test_token_issues_once(tmp_path):
    reg = _reg(tmp_path)
    tok = reg.create_token("passive")["token"]
    rep = _enroll(reg, "passive-9", tok)
    assert rep.status == "issued" and rep.cert_pem
    assert sec.describe_cert(rep.cert_pem)["role"] == "passive"


def test_token_single_use(tmp_path):
    reg = _reg(tmp_path)
    tok = reg.create_token("passive")["token"]
    assert _enroll(reg, "passive-9", tok).status == "issued"
    again = _enroll(reg, "passive-9", tok)
    assert again.status == "denied" and "already used" in again.error


def test_token_expiry(tmp_path):
    reg = _reg(tmp_path)
    tok = reg.create_token("passive", ttl_s=-1)["token"]       # already expired
    assert _enroll(reg, "passive-9", tok).status == "denied"


def test_static_mode_refuses_enrollment(tmp_path):
    reg = _reg(tmp_path, mode="static")
    tok = sec.make_token("x" * 64)[0]
    assert _enroll(reg, "p", tok).status == "error"


def test_revocation_list(tmp_path):
    reg = _reg(tmp_path)
    tok = reg.create_token("passive")["token"]
    rep = _enroll(reg, "passive-9", tok)
    serial = sec.describe_cert(rep.cert_pem)["serial"]
    reg.revoke(serial, "test")
    assert serial in reg.revoked
