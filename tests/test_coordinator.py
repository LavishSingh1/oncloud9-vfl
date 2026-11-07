"""FR-1: role validation, membership and session preconditions."""
import json

import pytest

from vfl.config import load_session_config
from vfl.coordinator import CoordinatorState
from vfl.proto import vfl_pb2 as pb
from vfl.security import TLSFiles


@pytest.fixture
def state(tmp_path):
    return CoordinatorState(str(tmp_path), load_session_config(None), TLSFiles(None, "coordinator"))


def reg(state, pid, role, address="", summary=None):
    return state.register(pb.RegisterRequest(party_id=pid, role=role, address=address,
                                             summary_json=json.dumps(summary or {})))


def test_rejects_invalid_roles_and_ids(state):
    assert reg(state, "x", "observer") == (False, "invalid role 'observer'; expected one of ('active', 'passive')")
    assert not reg(state, "", "passive")[0]
    assert not reg(state, "bad id", "passive")[0]
    assert not reg(state, "a", "active")[0]                      # active must advertise an address
    assert not reg(state, "p", "passive", summary={"has_labels": True})[0]


def test_single_active_and_session_preconditions(state):
    assert reg(state, "active", "active", "h:1")[0]
    assert not reg(state, "active-2", "active", "h:2")[0]
    with pytest.raises(ValueError, match="passive"):
        state.start_session({})
    assert reg(state, "passive-1", "passive")[0]
    brief = state.start_session({"config": {"epochs": 1}})
    assert brief["participants"] == ["active", "passive-1"]
    with pytest.raises(ValueError, match="still running"):
        state.start_session({})


def test_start_delivered_via_heartbeat(state):
    reg(state, "active", "active", "h:1")
    reg(state, "passive-1", "passive")
    sid = state.start_session({})["session_id"]
    r = state.heartbeat(pb.HeartbeatRequest(party_id="passive-1", status="idle"))
    assert r.command == "START" and json.loads(r.session_json)["session_id"] == sid
    assert state.heartbeat(pb.HeartbeatRequest(party_id="passive-1", status="running")).command == "NONE"
    assert not state.heartbeat(pb.HeartbeatRequest(party_id="ghost")).known


def test_dh_psi_with_three_or_more_parties(state):
    """v2.1: DH-PSI is no longer limited to two parties (multi-party protocol, vfl/alignment.MultiPSIHub)."""
    reg(state, "active", "active", "h:1")
    reg(state, "passive-1", "passive")
    reg(state, "passive-2", "passive")
    brief = state.start_session({"config": {"alignment": "dh_psi"}})
    assert brief["alignment"] == "dh_psi" and len(brief["participants"]) == 3


def test_config_validation():
    with pytest.raises(ValueError):
        load_session_config(None, ["batch_size=0"])
    with pytest.raises(ValueError):
        load_session_config(None, ["codec=int4"])
