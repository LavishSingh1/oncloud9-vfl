"""v2 item 1: partitioner integration, per-party manifests, split checks, v1 default unchanged."""
import hashlib
import json
import os

import numpy as np
import pytest

from vfl import partition as P
from vfl.alignment import align_locally
from vfl.coordinator import check_data_consistency
from vfl.data import PartyData, prepare_v2, prepare_vertical_dataset

LABEL_DERIVED = {"label_balance", "num_classes", "binarize", "importance_share", "reference_auroc"}


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _load_all(d, n_passive):
    act = PartyData.from_csv(os.path.join(d, "active.csv"), require_label=True)
    pas = [PartyData.from_csv(os.path.join(d, f"passive-{i}.csv"), require_label=False)
           for i in range(1, n_passive + 1)]
    return act, pas


def test_default_prepare_data_is_still_v1(tmp_path):
    from vfl.cli import main
    main(["prepare-data", "--dataset", "breast_cancer", "--out", str(tmp_path / "cli")])
    prepare_vertical_dataset(str(tmp_path / "direct"), "breast_cancer", 1, 0.9, "random", 0)
    for f in ("active.csv", "passive-1.csv"):
        assert _sha(tmp_path / "cli" / f) == _sha(tmp_path / "direct" / f)
    m = json.load(open(tmp_path / "cli" / "passive-1.manifest.json"))
    assert m["partition"]["partitioner"] == "v1" and m["role"] == "passive"


def test_v2_tabular_round_trip(tmp_path):
    d = str(tmp_path)
    m = prepare_v2(d, "breast_cancer", 3, "random", seed=3, overlap=0.8)
    act, pas = _load_all(d, 2)
    res = align_locally([act.ids] + [p.ids for p in pas], "salt")
    assert res[0].n_aligned == m["intersection_size"] < m["entities_total"]
    ds = P.load_dataset("breast_cancer")
    src_row = {i: r for r, i in enumerate(P.make_ids(len(ds.X), "hash", 3))}
    for party, r, info in zip([act] + pas, res, m["parties"]):
        assert party.input_dim == info["model_input_dim"]
        for k in (0, 17, res[0].n_aligned - 1):
            row = r.rows[k]
            assert party.ids[row] == act.ids[res[0].rows[k]]          # same entity at every party
            want = ds.X.iloc[src_row[party.ids[row]]][info["columns"]].to_numpy(np.float32)
            assert np.allclose(party.X[row], want, rtol=1e-6)        # values survive the CSV round trip
    assert np.array_equal(act.y[res[0].rows], ds.y[[src_row[act.ids[i]] for i in res[0].rows]])


def test_v2_image_tiles_reshape_to_original(tmp_path):
    d = str(tmp_path)
    m = prepare_v2(d, "digits", 3, "image_strips", seed=1)
    act, pas = _load_all(d, 2)
    ds = P.load_dataset("digits")
    imgs = ds.X.to_numpy(np.float32).reshape(-1, 1, 8, 8)
    src_row = {i: r for r, i in enumerate(P.make_ids(len(ds.X), "hash", 1))}
    start = 0
    for party, info in zip([act] + pas, m["parties"]):
        c, h, w = party.shape
        assert party.shape == info["shape"] and c * h * w == party.input_dim
        tile = party.X[5].reshape(c, h, w)                           # what a CNN bottom model will see
        assert np.array_equal(tile, imgs[src_row[party.ids[5]], :, :, start:start + w])
        start += w
    assert m["num_classes"] == 10


def test_party_manifests_leak_nothing_label_derived_or_foreign(tmp_path):
    d = str(tmp_path)
    prepare_v2(d, "breast_cancer", 3, "importance", seed=0, overlap=0.9, evaluate=True)
    full = json.load(open(os.path.join(d, "manifest.json")))
    assert "reference_auroc" in full and full["label_balance"] is not None      # experimenter-only file
    for i in (1, 2):
        pm = json.load(open(os.path.join(d, f"passive-{i}.manifest.json")))
        flat = json.dumps(pm)
        assert not any(k in flat for k in LABEL_DERIVED)
        own = PartyData.from_csv(os.path.join(d, f"passive-{i}.csv")).feature_names
        assert pm["columns"] == own
        others = {c for p in full["parties"] if p["name"] != f"passive-{i}" for c in p["columns"]}
        assert not others & set(flat.split('"'))
    am = json.load(open(os.path.join(d, "active.manifest.json")))
    assert am["partition"]["num_classes"] == 2 and "importance_share" not in am


def test_manifest_mismatch_is_rejected(tmp_path):
    d = str(tmp_path)
    prepare_v2(d, "breast_cancer", 2, "sequential")
    with pytest.raises(ValueError, match="role"):
        PartyData.from_csv(os.path.join(d, "passive-1.csv"), manifest_path=os.path.join(d, "active.manifest.json"))
    mp = os.path.join(d, "passive-1.manifest.json")
    m = json.load(open(mp))
    m["model_input_dim"] += 1
    json.dump(m, open(mp, "w"))
    with pytest.raises(ValueError, match="model_input_dim"):
        PartyData.from_csv(os.path.join(d, "passive-1.csv"))


def test_v2_sentinels_and_v1_aliases(tmp_path):
    m = prepare_v2(str(tmp_path / "s"), "breast_cancer", 2, "random", sentinel=True)
    act, pas = _load_all(str(tmp_path / "s"), 1)
    assert sum(i.startswith("SENTINEL-UNMATCHED-") for i in act.ids) == 5
    assert m["parties"][1]["rows"] == len(pas[0].ids)
    m2 = prepare_v2(str(tmp_path / "k"), "breast_cancer", 2, "importance_skewed")
    assert m2["strategy"] == "importance" and m2["params"]["informative_party"] == 1
    with pytest.raises(ValueError, match="v1 partitioner"):
        prepare_v2(str(tmp_path / "b"), "breast_cancer", 2, "importance_balanced")
    with pytest.raises(ValueError, match="image datasets"):
        prepare_v2(str(tmp_path / "c"), "breast_cancer", 2, "random", binarize="odd_even")


def test_multiclass_trains_or_binarizes(tmp_path):
    # item 1 generates multi-class data; item 4 trains it (cross-entropy + macro one-vs-rest AUROC).
    from vfl.config import load_session_config
    from vfl.local_sim import run_local_vfl
    d = str(tmp_path)
    prepare_v2(d, "digits", 2, "image_strips")
    assert json.load(open(os.path.join(d, "active.manifest.json")))["partition"]["num_classes"] == 10
    r = run_local_vfl(d, load_session_config(None, ["epochs=1", "bottom_model=cnn"]), log=lambda *a: None)
    assert r["num_classes"] == 10 and r["final_test"]["auroc"] > 0.5
    # binarize still works for the binary attack metrics
    prepare_v2(str(tmp_path / "b"), "digits", 2, "image_strips", binarize="odd_even")
    r2 = run_local_vfl(str(tmp_path / "b"), load_session_config(None, ["epochs=1"]), log=lambda *a: None)
    assert r2["num_classes"] == 2 and r2["final_test"]["auroc"] > 0.5


def _party(pid, role, **part):
    base = {"partitioner": "v2", "strategy": "random", "overlap": 0.9, "seed": 0, "manifest_sha256": "aaa"}
    if role == "active":
        base.update(binarize=None, num_classes=2)
    base.update(part)
    return {"party_id": pid, "role": role, "summary": {"partition": base}}


def test_coordinator_split_checks():
    ok = [_party("active", "active"), _party("passive-1", "passive")]
    check_data_consistency({}, ok)
    check_data_consistency({"partition": "random", "overlap": 0.9}, ok)
    with pytest.raises(ValueError, match="different prepare-data runs"):
        check_data_consistency({}, [_party("active", "active"), _party("passive-1", "passive", manifest_sha256="bbb")])
    with pytest.raises(ValueError, match="session expects 'dirichlet'"):
        check_data_consistency({"partition": "dirichlet"}, ok)
    with pytest.raises(ValueError, match="overlap"):
        check_data_consistency({"overlap": 1.0}, ok)
    with pytest.raises(ValueError, match="binarize"):
        check_data_consistency({"binarize": "odd_even"}, ok)
    with pytest.raises(ValueError, match="no data manifest"):
        check_data_consistency({"partition": "random"}, [_party("active", "active"),
                                                         {"party_id": "passive-1", "role": "passive", "summary": {}}])
    # multi-class is allowed (item 4); only a num_classes mismatch is rejected
    check_data_consistency({}, [_party("active", "active", num_classes=10), _party("passive-1", "passive")])
    with pytest.raises(ValueError, match="num_classes"):
        check_data_consistency({"num_classes": 2}, [_party("active", "active", num_classes=10),
                                                    _party("passive-1", "passive")])


def test_new_config_keys_validated():
    from vfl.config import load_session_config
    assert load_session_config(None)["partition"] is None                  # v1 default: no check
    with pytest.raises(ValueError, match="overlap"):
        load_session_config(None, ["overlap=1.5"])


def test_log_audit_detects_hashed_ids(tmp_path):
    from bench.security_test import audit_logs
    w = tmp_path
    prepare_v2(str(w / "data"), "breast_cancer", 2, "random")
    os.makedirs(w / "logs")
    (w / "logs" / "a.log").write_text("batch 3 took 4.2 ms; session s20261023-1200-ab12\n")
    assert audit_logs(str(w))["pass"]
    leaked = PartyData.from_csv(str(w / "data" / "passive-1.csv")).ids[7]
    (w / "logs" / "b.jsonl").write_text(json.dumps({"oops": leaked}) + "\n")
    rep = audit_logs(str(w))
    assert not rep["pass"] and rep["hits"]["raw_entity_id"]


@pytest.mark.slow
def test_networked_three_party_v2_run(tmp_path):
    from bench.harness import LocalDeployment
    d = str(tmp_path / "data")
    prepare_v2(d, "digits", 3, "image_strips", overlap=0.9, binarize="odd_even")
    with LocalDeployment(str(tmp_path / "dep"), d, 2, tls=True, base_port=56050, ui_base=19680) as dep:
        coord = dep.start()
        with pytest.raises(RuntimeError, match="session expects 'random'"):
            coord.start({"epochs": 1, "partition": "random"})
        r = coord.run({"epochs": 2, "partition": "image_strips", "overlap": 0.9, "binarize": "odd_even"},
                      quiet=True)
        assert r["state"] == "COMPLETED"
        assert {a["n_aligned"] for a in r["alignment"].values()} == {json.load(open(os.path.join(d, "manifest.json")))["intersection_size"]}
