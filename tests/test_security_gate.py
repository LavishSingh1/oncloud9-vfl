"""v2 items 9-12, 17-19: security gate, revocation, PKI identity, enrollment tokens, security log."""
import time

import grpc
import numpy as np
import pytest

from vfl import security as sec
from vfl.proto import vfl_pb2 as pb
from vfl.telemetry import SecurityLog
from vfl.transport import ACTIVE_POLICY, COORDINATOR_POLICY, SecurityGate, _Bucket


def test_token_bucket_rate_limit():
    b = _Bucket(rate=5, burst=5)
    assert sum(b.take() for _ in range(5)) == 5
    assert not b.take()


def test_pki_roles_and_name_in_san(tmp_path):
    d = str(tmp_path)
    sec.make_pki(d)
    info = sec.issue_static_pki(d, "passive-1", "passive")
    assert info["role"] == "passive" and info["name"] == "passive-1"
    pem = open(f"{d}/passive-1.crt", "rb").read()
    desc = sec.describe_cert(pem)
    assert desc["uri"] == "spiffe://oncloud9/vfl/passive/passive-1"


def test_renewal_requires_proof_of_possession(tmp_path):
    d = str(tmp_path)
    sec.make_pki(d)
    sec.issue_static_pki(d, "passive-1", "passive")
    cert = open(f"{d}/passive-1.crt", "rb").read()
    key = open(f"{d}/passive-1.key", "rb").read()
    new_key, csr = sec.make_csr("passive-1")
    name, role, serial = sec.verify_renewal(d, cert, csr, sec.sign_bytes(key, csr))
    assert name == "passive-1" and role == "passive"
    with pytest.raises(ValueError, match="possession"):
        sec.verify_renewal(d, cert, csr, sec.sign_bytes(new_key, csr))   # signed with the wrong key


def test_security_log_structured_event(tmp_path):
    log = SecurityLog("active", str(tmp_path / "security.jsonl"))
    log.event("identity_mismatch", "auth", "blocked", "certificate-bound identity",
              source_ip="10.0.0.9", claimed="active", cert_identity="passive-1", method="Register")
    assert log.summary()["totals"]["blocked"] == 1
    rec = log.recent()[-1]
    assert rec["attack_type"] == "identity_mismatch" and rec["evidence"] == "server-observed"
    import json
    line = json.loads(open(str(tmp_path / "security.jsonl")).read().strip())
    assert line["cert_identity"] == "passive-1" and "label" not in line


def test_gate_policies_cover_methods():
    assert "Exchange" in ACTIVE_POLICY and ACTIVE_POLICY["Exchange"].roles == {"passive"}
    assert "Register" in COORDINATOR_POLICY


@pytest.mark.slow
def test_gate_rejects_mismatched_identity_and_revoked(tmp_path, data_dir):
    from bench.harness import LocalDeployment
    with LocalDeployment(str(tmp_path / "dep"), data_dir, 1, tls=True, base_port=59050, ui_base=19950) as dep:
        dep.start()
        from vfl.transport import make_channel
        from vfl.proto import vfl_pb2_grpc as pbg
        from vfl.security import TLSFiles, load_ca, make_node_cert
        k, c = load_ca(dep.certs)
        make_node_cert(dep.certs, "rogue", k, c)
        ch = make_channel(TLSFiles(dep.certs, "rogue"), f"127.0.0.1:{dep.ports['coord']}", "coordinator")
        with pytest.raises(grpc.RpcError) as e:      # rogue cert claiming to be the active party
            pbg.CoordinatorStub(ch).Register(pb.RegisterRequest(party_id="active", role="active", address="x:1"), timeout=5)
        assert e.value.code() == grpc.StatusCode.PERMISSION_DENIED
        ch.close()
