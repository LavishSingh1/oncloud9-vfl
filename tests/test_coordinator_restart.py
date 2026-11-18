"""Issue #13: a session that was live when the coordinator died must not become a zombie after a restart.

On rt-baseline the restarted coordinator listed such a session as STARTING 0/N forever, `stop` answered
"400 unknown session", and runs/<sid>/summary.json kept saying RUNNING. A restart is simulated here by building a
second CoordinatorState on the same run directory without finishing the first one, which is what a kill -9 leaves.
"""
import json

import pytest

from vfl.config import load_session_config
from vfl.coordinator import INTERRUPTED_REASON, CoordinatorState
from vfl.proto import vfl_pb2 as pb
from vfl.security import TLSFiles


def coordinator(run_dir):
    return CoordinatorState(str(run_dir), load_session_config(None), TLSFiles(None, "coordinator"))


def register_parties(state):
    for pid, role, addr in (("active", "active", "h:1"), ("passive-1", "passive", "")):
        assert state.register(pb.RegisterRequest(party_id=pid, role=role, address=addr, summary_json="{}"))[0]


def crash_mid_run(run_dir, epochs_done=1):
    """Start a 40-epoch session, let `epochs_done` epochs finish (as Report does), then 'kill' the process."""
    state = coordinator(run_dir)
    register_parties(state)
    sid = state.start_session({"config": {"epochs": 40}})["session_id"]
    s = state.sessions[sid]
    s.state = "RUNNING"
    for e in range(epochs_done):
        s.epoch_done[e] = set(s.party_ids)
    s.epoch_metrics.append({"epoch": epochs_done - 1, "test": {"auroc": 0.91}})
    s.persist()                                         # summary.json now says RUNNING, index still says STARTING
    return sid


def listed(state, sid):
    return next(h for h in state.history if h["session_id"] == sid)


def test_interrupted_session_is_closed_on_restart(tmp_path):
    sid = crash_mid_run(tmp_path, epochs_done=1)
    state = coordinator(tmp_path)                       # the restarted coordinator
    assert state.recovered == [sid]
    h = listed(state, sid)
    assert h["state"] == "FAILED" and h["completed_epochs"] == 1 and h["epochs"] == 40
    assert h["stop_reason"] == INTERRUPTED_REASON and h["errors"][-1]["reason"] == INTERRUPTED_REASON
    assert h["last_test_auroc"] == 0.91 and h["finished"]
    disk = json.loads((tmp_path / sid / "summary.json").read_text())
    assert disk["state"] == "FAILED" and disk["completed_epochs"] == 1
    assert disk["errors"][-1]["reason"] == INTERRUPTED_REASON
    index = json.loads((tmp_path / "sessions_index.json").read_text())
    assert next(x for x in index if x["session_id"] == sid)["state"] == "FAILED"
    shown = next(x for x in state.ui_state()["sessions"] if x["session_id"] == sid)   # what `status` prints
    assert shown["state"] == "FAILED" and shown["completed_epochs"] == 1


def test_stop_on_a_session_from_before_the_restart_is_a_clear_no_op(tmp_path):
    sid = crash_mid_run(tmp_path)
    state = coordinator(tmp_path)
    out = state.stop_session(sid)
    assert out["state"] == "FAILED" and "already ended (FAILED)" in out["note"]
    assert listed(state, sid)["state"] == "FAILED"     # unchanged
    with pytest.raises(ValueError, match="unknown session"):
        state.stop_session("s20990101-000000-dead")


def test_resume_still_works_after_the_restart(tmp_path):
    sid = crash_mid_run(tmp_path, epochs_done=2)
    state = coordinator(tmp_path)
    register_parties(state)
    new = state.start_session({"resume_from": sid})["session_id"]
    cfg = state.sessions[new].config
    assert cfg["resume_from"] == sid and cfg["start_epoch"] == 2 and cfg["epochs"] == 40


def test_recovery_is_idempotent_and_leaves_finished_sessions_alone(tmp_path):
    sid = crash_mid_run(tmp_path)
    first = coordinator(tmp_path)
    assert first.recovered == [sid]
    before = (tmp_path / sid / "summary.json").read_text()
    second = coordinator(tmp_path)
    assert second.recovered == []
    assert (tmp_path / sid / "summary.json").read_text() == before


def write_index(run_dir, entries):
    (run_dir / "sessions_index.json").write_text(json.dumps(entries))


def test_other_live_states_and_edge_cases(tmp_path):
    def entry(sid, state, done=0):
        return {"session_id": sid, "state": state, "created": 1.0, "epochs": 5, "completed_epochs": done}

    ok = "s20261117-000000-0001"      # index says STOPPING, no summary.json (died before the first report)
    synced = "s20261117-000000-0002"  # summary.json already COMPLETED, index write lost
    done = "s20261117-000000-0003"    # finished normally: must not be touched
    (tmp_path / synced).mkdir()
    (tmp_path / synced / "summary.json").write_text(json.dumps(
        {"session_id": synced, "state": "COMPLETED", "completed_epochs": 5, "finished": 9.0,
         "epoch_metrics": [{"test": {"auroc": 0.97}}], "errors": []}))
    write_index(tmp_path, [entry(ok, "STOPPING", 3), entry(synced, "RUNNING", 4), entry(done, "COMPLETED", 5),
                           {"session_id": "../../etc", "state": "RUNNING"}, "garbage"])
    state = coordinator(tmp_path)
    assert sorted(state.recovered) == [ok, synced]
    assert listed(state, ok)["state"] == "STOPPED" and listed(state, ok)["completed_epochs"] == 3
    s = listed(state, synced)
    assert s["state"] == "COMPLETED" and s["completed_epochs"] == 5 and s["last_test_auroc"] == 0.97
    assert "errors" not in s or not any(e.get("reason") == INTERRUPTED_REASON for e in s["errors"])
    assert json.loads((tmp_path / synced / "summary.json").read_text())["state"] == "COMPLETED"
    assert listed(state, done) == entry(done, "COMPLETED", 5)
    assert not (tmp_path.parent / "etc").exists()
