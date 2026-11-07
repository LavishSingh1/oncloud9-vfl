"""End-to-end over real processes and gRPC with mTLS (slower: ~30-60 s).

Checks: networked fp32 run == single-process VFL (tensor-level correctness), FR-8 duplicate
suppression and retry after a deadline, exactly one optimizer step per committed batch."""

import pytest

from bench.harness import LocalDeployment
from vfl.config import load_session_config
from vfl.local_sim import run_local_vfl


@pytest.mark.slow
def test_networked_matches_local_and_faults(tmp_path, data_dir):
    with LocalDeployment(str(tmp_path / "dep"), data_dir, 1, tls=True, base_port=54050, ui_base=19080) as dep:
        coord = dep.start()
        d = coord.run({"epochs": 3, "batch_size": 32, "seed": 5}, quiet=True)
        assert d["state"] == "COMPLETED"
        act = d["finals"]["active"]
        local = run_local_vfl(data_dir, load_session_config(None, ["epochs=3", "batch_size=32", "seed=5"]),
                              log=lambda *a: None)
        assert abs(local["final_train_loss"] - act["final_train_loss"]) < 1e-6
        assert abs(local["final_test"]["auroc"] - act["final_test"]["auroc"]) < 1e-9

        fault = {"duplicate": [[0, 1]], "delay": [[1, 2]], "delay_s": 1.5}
        d = coord.run({"epochs": 2, "batch_size": 32, "fault": fault, "rpc_deadline_s": 0.5}, quiet=True)
        assert d["state"] == "COMPLETED"
        a, p = d["finals"]["active"], d["finals"]["passive-1"]
        assert a["duplicates_suppressed"] >= 2                   # injected replay + retried request
        assert p["counters"]["rpc_timeouts"] >= 1
        for x in (a, p):
            assert x["optimizer_steps_this_session"] == x["scheduled_batches"] == x["committed_batches"]
