"""FR-3 / FR-9: deterministic splits and mini-batches."""
import numpy as np

from vfl.data import PartyData, batch_digest, epoch_batches, make_splits


def test_splits_disjoint_and_deterministic():
    s1, s2 = make_splits(500, 7, 0.15, 0.15), make_splits(500, 7, 0.15, 0.15)
    for k in s1:
        assert np.array_equal(s1[k], s2[k])
    allidx = np.concatenate(list(s1.values()))
    assert len(allidx) == 500 and len(set(allidx.tolist())) == 500


def test_batches_identical_for_same_seed_and_cover_train():
    tr = make_splits(500, 7, 0.15, 0.15)["train"]
    b1, b2 = epoch_batches(tr, 32, 3, 0), epoch_batches(tr, 32, 3, 0)
    assert [x.tolist() for x in b1] == [x.tolist() for x in b2]
    assert sorted(np.concatenate(b1).tolist()) == sorted(tr.tolist())
    assert [batch_digest(x) for x in b1] == [batch_digest(x) for x in b2]
    b_next = epoch_batches(tr, 32, 3, 1)
    assert [x.tolist() for x in b_next] != [x.tolist() for x in b1]


def test_three_batch_sizes(cfg):
    tr = make_splits(400, 1, 0.1, 0.1)["train"]
    for bs in (32, 64, 128):
        b = epoch_batches(tr, bs, 1, 0)
        assert len(b) == int(np.ceil(len(tr) / bs))


def test_party_files_respect_label_ownership(data_dir):
    import os
    import pytest
    a = PartyData.from_csv(os.path.join(data_dir, "active.csv"), require_label=True)
    assert a.y is not None
    p = PartyData.from_csv(os.path.join(data_dir, "passive-1.csv"), require_label=False)
    assert p.y is None
    with pytest.raises(ValueError):
        PartyData.from_csv(os.path.join(data_dir, "active.csv"), require_label=False)   # passive holding labels
