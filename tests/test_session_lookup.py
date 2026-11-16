"""Regression tests for stale / unknown session ids at the active party (red-team report: "10-second wait loop in
session_for"). On rt-baseline every unknown id was parked for 10 s, so a caller with a 3 s deadline saw
DEADLINE_EXCEEDED and 16 stale calls filled the per-identity concurrency cap. Driven in-process: a fake coordinator
answers heartbeats, a fake gRPC context carries the caller's deadline and identity.
"""
import collections
import json
import threading
import time
import types

import pytest

pytest.importorskip("grpc")
pytest.importorskip("torch")
import grpc  # noqa: E402

from vfl import active as A  # noqa: E402


class Aborted(Exception):
    def __init__(self, code, details):
        super().__init__(f"{code}: {details}")
        self.code, self.details = code, details


class Ctx:
    def __init__(self, timeout=None, ident="passive-1"):
        self.end = None if timeout is None else time.time() + timeout
        self.ident = ident

    def time_remaining(self):
        return None if self.end is None else max(0.0, self.end - time.time())

    def peer(self):
        return "ipv4:127.0.0.1:5555"

    def auth_context(self):
        return {"x509_common_name": [self.ident.encode()]}

    def abort(self, code, details):
        raise Aborted(code, details)


class FakeCoord:
    """Holds at most one START for the active node and hands it out on the next heartbeat, like the coordinator."""

    def __init__(self):
        self.pending, self.down = None, False

    def Heartbeat(self, req, timeout=None):
        time.sleep(0.005)
        if self.down:
            err = grpc.RpcError()
            err.code, err.details = (lambda: grpc.StatusCode.UNAVAILABLE), (lambda: "coordinator down")
            raise err
        cmd, js = ("START", json.dumps(self.pending)) if self.pending else ("NONE", "")
        self.pending = None
        return types.SimpleNamespace(known=True, command=cmd, session_json=js, revoked_serials=[])


@pytest.fixture
def node(monkeypatch):
    monkeypatch.setattr(A.SECURITY, "event", lambda *a, **k: None)
    n = object.__new__(A.ActiveNode)              # no data, TLS or dashboard: only the session-lookup state
    n.party_id, n.status, n.progress, n.revoked = "active", "idle", {}, set()
    n.lock, n.stop_event, n.history = threading.RLock(), threading.Event(), []
    n.session_id, n._session_thread, n.coord_ok, n.coord_error = None, None, True, ""
    n.sess, n.sess_cv, n._sess_waiters = None, threading.Condition(), collections.Counter()
    n._hb_now, n._hb_cv, n._hb_started, n._hb_ok, n._hb_last_start = threading.Event(), threading.Condition(), 0, 0, 0.0
    n.coord, done = FakeCoord(), threading.Event()

    def on_start(payload):                         # stands in for PartyNode._on_start + run_session publishing sess
        n.session_id = payload["session_id"]

        def run():
            time.sleep(0.02)
            s = A.ActiveSession(n, n.session_id, {}, [{"party_id": "passive-1", "role": "passive"}])
            with n.sess_cv:
                n.sess = s
                n.sess_cv.notify_all()
            done.wait()
        n._session_thread = threading.Thread(target=run, daemon=True)
        n._session_thread.start()

    n._on_start = on_start
    threading.Thread(target=n.heartbeat_loop, daemon=True).start()
    yield n
    done.set()


def call(n, sid, timeout=None, ident="passive-1"):
    t0 = time.time()
    try:
        return "OK", n.session_for(sid, Ctx(timeout, ident), ident, "Exchange").id, time.time() - t0
    except Aborted as e:
        return e.code, e.details, time.time() - t0


def test_unknown_session_fails_fast_with_a_diagnostic(node):
    code, details, dt = call(node, "stale-1", timeout=3.0)
    assert code == grpc.StatusCode.FAILED_PRECONDITION and "stale-1" in details
    assert dt < 0.5                                 # rt-baseline: 3.0 s and DEADLINE_EXCEEDED


def test_stale_calls_do_not_hold_concurrency_slots(node):
    out = []
    threads = [threading.Thread(target=lambda i=i: out.append(call(node, f"stale-{i}", 12.0))) for i in range(16)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert time.time() - t0 < 1.5                   # rt-baseline: 10 s, with all 16 slots held
    assert {c for c, _, _ in out} == {grpc.StatusCode.FAILED_PRECONDITION}
    assert sum(node._sess_waiters.values()) == 0


def test_start_race_is_still_tolerated(node):
    """A passive party can hear START up to one heartbeat before the active party: that call must succeed."""
    node.coord.pending = {"session_id": "real-session"}
    assert call(node, "real-session", 30.0)[:2] == ("OK", "real-session")
    assert call(node, "real-session", 2.0)[0] == "OK"


def test_other_or_ended_sessions_are_refused_immediately(node):
    node.coord.pending = {"session_id": "real-session"}
    assert call(node, "real-session", 30.0)[0] == "OK"
    code, details, dt = call(node, "other-session", 30.0)
    assert code == grpc.StatusCode.FAILED_PRECONDITION and "different session" in details and dt < 0.05
    node.sess.closed = True
    node.history.insert(0, {"session_id": "real-session"})
    code, details, dt = call(node, "real-session", 30.0)
    assert code == grpc.StatusCode.FAILED_PRECONDITION and "ended" in details and dt < 0.05


def test_malformed_session_id_is_invalid_argument(node):
    assert call(node, "../../etc", 5.0)[0] == grpc.StatusCode.INVALID_ARGUMENT


def test_waits_are_capped_by_the_callers_deadline_and_per_peer(node):
    node.coord.down = True                          # the only case where the grace period is actually spent
    code, _, dt = call(node, "stale-x", timeout=1.0)
    assert code == grpc.StatusCode.FAILED_PRECONDITION and dt < 1.0
    out = []
    threads = [threading.Thread(target=lambda i=i: out.append(call(node, f"s-{i}", 12.0))) for i in range(10)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    refused = [d for _, d, dt in out if dt < 0.3]
    assert len(refused) == 10 - A.ActiveNode.MAX_SESSION_WAITERS_PER_PEER
    assert all("too many" in d for d in refused)
    # another identity still gets its own waiting budget
    assert call(node, "s-other", 0.5, ident="passive-2")[0] == grpc.StatusCode.FAILED_PRECONDITION
    for t in threads:
        t.join()
    assert max(dt for _, _, dt in out) < A.ActiveNode.START_GRACE_S + 0.3
