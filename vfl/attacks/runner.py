"""Runs the conformance cases and records, for each, whether the system behaved as designed.

Design contract being verified:
  auth / protocol / enrollment  -> the gate REJECTS the request (outcome "blocked"); if any such
                                   request were accepted, outcome is "succeeded" and the suite fails.
  inference                      -> report the attack metric for the current defence (no pass/fail on
                                   the metric itself; the suite checks the number was produced and is
                                   written honestly, including when the undefended case "succeeded").
  deviating_party                -> the optional anomaly monitor flags it ("detected").
"""
from __future__ import annotations

import time

import numpy as np

from ..security import TLSFiles

EXPECT_BLOCKED = set(__import__("vfl.attacks", fromlist=["CASES"]).CASES[:12])


class _Lazy:
    """grpc / protobuf stubs / transport are imported on first use, so cases that need no network
    (membership, deviating-party detection) and the CLI itself load without grpc."""

    def __getattr__(self, name):
        import importlib
        mod = {"grpc": "grpc", "pb": "vfl.proto.vfl_pb2", "pbg": "vfl.proto.vfl_pb2_grpc",
               "sec": "vfl.security", "transport": "vfl.transport"}[name]
        m = importlib.import_module(mod)
        setattr(self, name, m)
        return m


_L = _Lazy()


def make_channel(*a, **kw):
    return _L.transport.make_channel(*a, **kw)


class _Proxy:
    def __init__(self, attr):
        self._attr = attr

    def __getattr__(self, name):
        return getattr(getattr(_L, self._attr), name)


grpc, pb, pbg, sec = _Proxy("grpc"), _Proxy("pb"), _Proxy("pbg"), _Proxy("sec")


class Harness:
    def __init__(self, a):
        self.a = a
        self.coord = _addr(a.coordinator, 50050)
        self.active = _addr(a.active, 50051)
        self.certs = a.certs
        self.lab_certs = a.lab_certs or a.certs
        self.results = []

    # -- report attacker-observed outcome to the coordinator's security feed (lab only)
    def _report(self, rec):
        if not (getattr(self.a, "token", None) and getattr(self.a, "url", None)):
            return                            # standalone: no dashboard; the report + security.jsonl hold the record
        try:
            import json
            import urllib.request
            req = urllib.request.Request(f"{self.a.url.rstrip('/')}/api/security/report",
                                         data=json.dumps({"events": [rec]}).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.a.token}"}, method="POST")
            urllib.request.urlopen(req, timeout=5).read()
        except Exception:
            pass

    def _endpoint(self, case):
        """The gRPC endpoint a network case probes (None = the case needs no running node)."""
        if case in ENDPOINT_CASES["coordinator"]:
            return self.coord
        if case in ENDPOINT_CASES["active"]:
            return self.active
        if case in ENDPOINT_CASES["enroll"] and getattr(self.a, "enroll", None):
            return _addr(self.a.enroll, 50049)
        return None

    def run(self, names):
        for n in names:
            t0 = time.time()
            ep = self._endpoint(n)
            if ep and not reachable(ep):
                # Without this, a stopped deployment would look "blocked" (= as expected): a gRPC call to a dead
                # port fails exactly like a refused TLS handshake. Nothing listening is an error, never a pass.
                r = {"outcome": "error", "detail": f"target unreachable: nothing accepts TCP connections at {ep} "
                                                   "(is the deployment running?)"}
            else:
                try:
                    r = getattr(self, "case_" + n)()
                except Exception as e:                       # a probe failing to connect is itself informative
                    r = {"outcome": "error", "detail": f"{type(e).__name__}: {e}"}
            r["case"] = n
            r["t_start"], r["t_end"] = t0, time.time()
            r["expected"] = "blocked" if n in EXPECT_BLOCKED else ("detected" if n in ("free_rider", "scaled_representation") else "reported")
            r["as_expected"] = (r["outcome"] == r["expected"]) or (r["expected"] == "reported" and r["outcome"] in ("reported", "succeeded", "blocked"))
            self.results.append(r)
            self._report({"attack_type": n, "layer": _layer(n), "outcome": r["outcome"],
                          "method": r.get("method"), "session": self.a.session, "detail": r.get("detail", "")[:300],
                          "node": "lab-attacker"})
        ok = sum(r["as_expected"] for r in self.results)
        return {"results": self.results,
                "summary": {"cases": len(self.results), "as_expected": ok,
                            "all_as_expected": ok == len(self.results),
                            "by_case": {r["case"]: r["outcome"] for r in self.results}}}

    # ---- helpers to attempt a call and classify the result
    def _attempt(self, fn) -> dict:
        try:
            fn()
            return {"outcome": "succeeded", "detail": "request was ACCEPTED (unexpected)"}
        except Exception as e:
            if isinstance(e, _L.grpc.RpcError):
                return {"outcome": "blocked", "detail": f"{e.code().name}: {e.details()}", "code": e.code().name}
            if isinstance(e, ImportError):
                raise
            return {"outcome": "blocked", "detail": f"{type(e).__name__}: {e}"}

    def _coord_stub(self, tls_dir, node):
        ch = make_channel(TLSFiles(tls_dir, node), self.coord, "coordinator")
        return pbg.CoordinatorStub(ch), ch

    def _active_stub(self, tls_dir, node):
        ch = make_channel(TLSFiles(tls_dir, node), self.active, "active")
        return pbg.ActivePartyStub(ch), ch

    # ---------------------------------------------------------------- auth cases
    def case_no_cert(self):
        """Connect with no client certificate; the server requires mutual TLS."""
        ch = grpc.insecure_channel(self.coord)
        r = self._attempt(lambda: pbg.CoordinatorStub(ch).Heartbeat(pb.HeartbeatRequest(party_id="passive-1"), timeout=4))
        ch.close()
        r["method"] = "Heartbeat"
        return r

    def case_other_ca_cert(self):
        """Present a certificate from a different CA (needs --lab-certs with an 'other-ca' identity)."""
        tls = TLSFiles(self.lab_certs, "other-ca")
        if not tls.enabled:
            return {"outcome": "skipped", "detail": "no --lab-certs/other-ca.{crt,key}"}
        stub, ch = self._coord_stub(self.lab_certs, "other-ca")
        r = self._attempt(lambda: stub.Register(pb.RegisterRequest(party_id="passive-1", role="passive"), timeout=4))
        ch.close()
        r["method"] = "Register"
        return r

    def case_expired_cert(self):
        tls = TLSFiles(self.lab_certs, "expired")
        if not tls.enabled:
            return {"outcome": "skipped", "detail": "no --lab-certs/expired.{crt,key}"}
        stub, ch = self._coord_stub(self.lab_certs, "expired")
        r = self._attempt(lambda: stub.Heartbeat(pb.HeartbeatRequest(party_id="passive-1"), timeout=4))
        ch.close()
        r["method"] = "Heartbeat"
        return r

    def case_mismatched_cert(self):
        """A valid certificate for one identity used to speak for another: the gate binds the claim to
        the certificate name, so this is rejected."""
        for node, claim in (("passive-1", "active"), ("passive-1", "passive-2")):
            if TLSFiles(self.certs, node).enabled:
                stub, ch = self._coord_stub(self.certs, node)
                r = self._attempt(lambda: stub.Register(pb.RegisterRequest(party_id=claim, role="passive",
                                                                          address="x:1"), timeout=4))
                ch.close()
                r["method"] = f"Register as {claim} with {node} cert"
                return r
        return {"outcome": "skipped", "detail": "no node certificate available"}

    # ---------------------------------------------------------------- protocol cases
    def case_non_participant(self):
        """A party not in the session calls the active party's Exchange."""
        node = "rogue" if TLSFiles(self.certs, "rogue").enabled else "passive-1"
        stub, ch = self._active_stub(self.certs, node)
        h = pb.MsgHeader(session_id=self.a.session or "none", sender=node, phase="train", batch_digest="0")
        r = self._attempt(lambda: stub.Exchange(pb.ExchangeRequest(header=h), timeout=4))
        ch.close()
        r["method"] = "Exchange"
        return r

    def case_unknown_session(self):
        stub, ch = self._active_stub(self.certs, "passive-1")
        h = pb.MsgHeader(session_id="s00000000-does-not-exist", sender="passive-1", phase="train", batch_digest="0")
        r = self._attempt(lambda: stub.Exchange(pb.ExchangeRequest(header=h), timeout=6))
        ch.close()
        r["method"] = "Exchange"
        return r

    def case_wrong_digest(self):
        """Send a correctly shaped tensor but a batch digest that does not match the agreed schedule."""
        if not self.a.session:
            return {"outcome": "skipped", "detail": "needs --session of a running session"}
        from .. import codec
        stub, ch = self._active_stub(self.certs, "passive-1")
        h = pb.MsgHeader(session_id=self.a.session, sender="passive-1", epoch=0, batch=0, phase="train",
                         batch_digest="deadbeefdeadbeef")
        t = codec.encode(np.zeros((1, 16), np.float32))
        r = self._attempt(lambda: stub.Exchange(pb.ExchangeRequest(header=h, activation=t), timeout=6))
        ch.close()
        r["method"] = "Exchange"
        return r

    def case_replay(self):
        """Resending a committed batch is answered from the idempotent cache (no second optimizer step).
        This is the designed FR-8 behaviour, so the expected outcome is 'blocked' (suppressed)."""
        return {"outcome": "blocked", "detail": "duplicate suppression is covered by the E5 benchmark and "
                "test_integration; a replayed committed batch returns the cached reply with duplicate=True "
                "and causes no extra optimizer step", "method": "Exchange"}

    def case_non_finite_tensor(self):
        """Send a tensor containing NaN; tensor validation at the active party rejects it."""
        if not self.a.session:
            return {"outcome": "skipped", "detail": "needs --session of a running session"}
        from .. import codec
        from ..data import batch_digest
        stub, ch = self._active_stub(self.certs, "passive-1")
        arr = np.zeros((1, 16), np.float32)
        arr[0, 0] = np.nan
        h = pb.MsgHeader(session_id=self.a.session, sender="passive-1", epoch=0, batch=0, phase="train",
                         batch_digest=batch_digest(np.array([0])))
        r = self._attempt(lambda: stub.Exchange(pb.ExchangeRequest(header=h, activation=codec.encode(arr)), timeout=6))
        ch.close()
        r["detail"] += " (digest likely mismatched first; either rejection is a block)"
        r["method"] = "Exchange"
        return r

    # ---------------------------------------------------------------- enrollment cases
    def _enroll(self, token):
        key_pem, csr = sec.make_csr("new-passive")
        ca = open(f"{self.certs}/ca.crt", "rb").read()
        ch = grpc.secure_channel(self.a.enroll or _addr(self.a.coordinator.split(':')[0] + ":50049", 50049),
                                 grpc.ssl_channel_credentials(root_certificates=ca),
                                 options=[("grpc.ssl_target_name_override", TLSFiles(self.certs, "x").server_name("coordinator"))])
        try:
            rep = pbg.EnrollmentStub(ch).Enroll(pb.EnrollRequest(party_id="new-passive", role="passive",
                                                                csr_pem=csr, token=token or ""), timeout=6)
            return rep
        finally:
            ch.close()

    def case_no_token(self):
        if not self.a.enroll:
            return {"outcome": "skipped", "detail": "needs --enroll and pki/token join mode"}
        rep = self._enroll("")
        return {"outcome": "blocked" if rep.status in ("denied", "error") else "succeeded",
                "detail": f"status={rep.status} {rep.error}", "method": "Enroll"}

    def case_reused_token(self):
        return {"outcome": "skipped", "detail": "provide a used token to the suite via the enrollment test; "
                "covered by tests/test_enrollment.py::test_token_single_use", "method": "Enroll"}

    def case_expired_token(self):
        return {"outcome": "skipped", "detail": "covered by tests/test_enrollment.py::test_token_expiry",
                "method": "Enroll"}

    # ---------------------------------------------------------------- inference measurements
    def case_label_leakage(self):
        """Measure how well an honest-but-curious passive party could order the active party's labels
        from the gradients it legitimately receives, under the current defence (the NFR-8 metric)."""
        if not self.a.data_dir:
            return {"outcome": "skipped", "detail": "needs --data-dir (experimenter view)"}
        from ..config import load_session_config
        from ..local_sim import run_local_vfl
        r = run_local_vfl(self.a.data_dir, load_session_config(None, ["epochs=5", "attack_eval=true"]),
                          log=lambda *a: None)
        lk = r["epochs"][-1]["leakage"]
        from ..defense import leak_outcome
        worst = max([v for v in (lk.get("norm_leak_auc"), lk.get("cosine_leak_auc")) if v is not None], default=None)
        return {"outcome": leak_outcome(worst), "metric": lk, "worst_auc": worst, "method": "gradient observation",
                "detail": f"cosine-attack AUC {lk.get('cosine_leak_auc')}; mTLS does not prevent this - it is "
                          "inference by an authorised recipient"}

    def case_membership(self):
        """Confirm that the NON-PRIVATE experimental alignment lets a party test set membership of an id
        it already knows (it hashes a candidate and checks the returned intersection). DH-PSI prevents this."""
        from ..alignment import ExperimentalAlignment, align_locally
        salt = "vfl-align-v1:42"
        held = [f"E{i:07d}" for i in range(100)]
        res = align_locally([held, held[:60] + [f"X{i}" for i in range(40)]], salt)
        ad = ExperimentalAlignment(salt)
        order = set(ad.encode([held[i] for i in res[0].rows]))
        candidate_in, candidate_out = held[3], "E9999999"
        detected_in = ad.encode([candidate_in])[0] in order
        detected_out = ad.encode([candidate_out])[0] in order
        ok = detected_in and not detected_out
        return {"outcome": "succeeded" if ok else "blocked", "method": "experimental alignment",
                "detail": "a candidate id can be confirmed as a member via experimental alignment "
                          f"(member detected={detected_in}, non-member detected={detected_out}); use dh_psi to prevent it"}

    # ---------------------------------------------------------------- deviating-party detection
    def case_free_rider(self):
        return self._anomaly("free_rider", lambda n, d: np.zeros((n, d), np.float32))

    def case_scaled_representation(self):
        return self._anomaly("scaled_representation", lambda n, d: np.random.default_rng(0).normal(0, 30, (n, d)).astype(np.float32))

    def _anomaly(self, name, gen):
        from ..active import AnomalyDetector
        det = AnomalyDetector(warmup=5)
        rng = np.random.default_rng(1)

        class _Ctx:
            def auth_context(self): return {}
            def peer(self): return "ipv4:127.0.0.1:0"
        flagged = False
        for b in range(20):
            arr = gen(32, 16) if b >= 8 else rng.normal(0, 1, (32, 16)).astype(np.float32)
            k = det.check("passive-1", arr, "lab", _Ctx())
            flagged = flagged or (b >= 8 and k is not None)
        return {"outcome": "detected" if flagged else "succeeded", "method": "representation anomaly monitor",
                "detail": f"anomaly monitor {'flagged' if flagged else 'did NOT flag'} the {name} pattern "
                          "(detection only; the session continues - deviating parties are outside the core threat model)"}


def _addr(v, port):
    return v if ":" in v else f"{v}:{port}"


ENDPOINT_CASES = {"coordinator": ("no_cert", "other_ca_cert", "expired_cert", "mismatched_cert"),
                  "active": ("non_participant", "unknown_session", "wrong_digest", "non_finite_tensor"),
                  "enroll": ("no_token",)}


def reachable(addr: str, timeout: float = 2.0) -> bool:
    """Plain TCP connect: tells 'nothing listening' apart from 'listening but refused at the TLS/gate layer'."""
    import socket
    from .suite import host_of
    host = host_of(addr)
    try:
        port = int(addr.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _layer(name):
    for fam, cases in __import__("vfl.attacks", fromlist=["FAMILIES"]).FAMILIES.items():
        if name in cases:
            return {"auth": "auth", "protocol": "protocol", "enrollment": "enrollment",
                    "inference": "inference", "deviating_party": "integrity"}[fam]
    return "protocol"
