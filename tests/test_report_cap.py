"""Issue #16: one session participant could grow the coordinator's memory and disk without limit.

On rt-baseline every Report(kind="epoch") was appended to Session.party_epochs, added to epoch_done and written to
epochs.jsonl, whatever the epoch number and however often it was repeated. A party reports each epoch of the
session exactly once, so the coordinator now refuses epochs outside 0..epochs-1 and acknowledges a repeat without
storing it: per-session state is bounded by epochs x parties.
"""
import json
import os
import tempfile

import pytest

pytest.importorskip("grpc")
import vfl.coordinator as C  # noqa: E402
from vfl.config import load_session_config  # noqa: E402
from vfl.coordinator import CoordinatorState  # noqa: E402
from vfl.proto import vfl_pb2 as pb  # noqa: E402
from vfl.security import TLSFiles  # noqa: E402


@pytest.fixture
def session(monkeypatch):
    """The reporter's setup: a coordinator with one active and one passive party and a 2-epoch session."""
    st = CoordinatorState(tempfile.mkdtemp(), load_session_config(None), TLSFiles(None, "coordinator"))
    monkeypatch.setattr(C, "check_data_consistency", lambda cfg, parties: None)
    st.register(pb.RegisterRequest(party_id="active", role="active", address="h:1"))
    st.register(pb.RegisterRequest(party_id="passive-1", role="passive"))
    st.parties["active"]["status"] = st.parties["passive-1"]["status"] = "idle"
    sid = st.start_session({"config": {"epochs": 2}})["session_id"]
    return st, st.sessions[sid]


def _report(st, s, party, payload):
    return st.report(pb.ReportRequest(session_id=s.id, party_id=party, kind="epoch", payload_json=json.dumps(payload)))


def _jsonl_rows(s):
    path = os.path.join(s.dir, "epochs.jsonl")
    if not os.path.exists(path):
        return 0
    with open(path) as fh:
        return sum(1 for _ in fh)


def test_reported_flood_no_longer_grows_state(session):
    """The reproduction from the issue: 20 000 epoch reports from passive-1 into a 2-epoch session."""
    st, s = session
    for i in range(1, 20001):
        _report(st, s, "passive-1", {"epoch": i})
    for i in range(20000):
        _report(st, s, "passive-1", {"epoch": i % 2})
    assert len(s.party_epochs["passive-1"]) == 2                  # one row per real epoch, not 20 001
    assert set(s.epoch_done) <= {0, 1}
    assert _jsonl_rows(s) == 2


@pytest.mark.parametrize("epoch", [-1, 2, 10 ** 9, "x", None])
def test_out_of_range_epochs_are_refused(session, epoch):
    st, s = session
    rep = _report(st, s, "passive-1", {"epoch": epoch})
    assert not rep.ok
    assert not s.party_epochs and _jsonl_rows(s) == 0


def test_a_repeat_is_acknowledged_but_not_stored(session):
    """An RPC retry of the same epoch report must not fail the party, and must not be recorded twice."""
    st, s = session
    first = _report(st, s, "passive-1", {"epoch": 0, "train_loss": 0.5})
    again = _report(st, s, "passive-1", {"epoch": 0, "train_loss": 99.0})
    assert first.ok and again.ok
    assert s.party_epochs["passive-1"] == [{"epoch": 0, "train_loss": 0.5}]


def test_normal_session_reports_are_all_kept(session):
    """Every party reporting every epoch once (what the real parties do) is recorded exactly as before."""
    st, s = session
    for e in range(2):
        for party in ("active", "passive-1"):
            assert _report(st, s, party, {"epoch": e, "train_loss": 0.1 * (e + 1)}).ok
    assert {p: len(v) for p, v in s.party_epochs.items()} == {"active": 2, "passive-1": 2}
    assert s.epoch_done[0] == s.epoch_done[1] == {"active", "passive-1"}
    assert _jsonl_rows(s) == 4
