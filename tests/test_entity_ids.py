"""Issue #2: entity ids must be wide enough to never collide in practice, and duplicates must be refused.

v2.3.1 generated ids as SHA-1(salt:row) truncated to 12 hex chars (48 bits). With seed 0, rows 18,125,834 and
20,457,688 received the same id (a877eff6ba91). When two colliding rows ended up at different parties (overlap < 1)
alignment joined two different entities without any error.
"""
import hashlib
import re

import numpy as np
import pytest

from vfl import partition as P
from vfl.alignment import align_locally
from vfl.data import PartyData

OLD_COLLIDING_ROWS = (18_125_834, 20_457_688)        # seed 0, found by exhaustive search of the old scheme


def _old_id(seed, row):                               # the v2.3.1 scheme, kept here only as a regression oracle
    return hashlib.new("sha1", f"vfl-{seed}:{row}".encode(), usedforsecurity=False).hexdigest()[:12]


def test_old_scheme_really_collided():
    a, b = OLD_COLLIDING_ROWS
    assert _old_id(0, a) == _old_id(0, b) == "a877eff6ba91"


def test_new_ids_do_not_collide_on_the_known_pair():
    a, b = OLD_COLLIDING_ROWS
    assert P.entity_id("vfl-0", a) != P.entity_id("vfl-0", b)


def test_ids_are_128_bit_sha256_hex():
    ids = P.make_ids(1000, "hash", 0)
    assert all(re.fullmatch(r"[0-9a-f]{32}", i) for i in ids)
    assert ids[7] == hashlib.sha256(b"vfl-0:7").hexdigest()[:32]


def test_ids_are_unique_and_stable_per_seed():
    a = P.make_ids(200_000, "hash", 5)
    assert len(set(a.tolist())) == len(a)
    assert np.array_equal(a, P.make_ids(200_000, "hash", 5))                 # reproducible (NFR-2)
    assert not np.array_equal(a[:100], P.make_ids(100, "hash", 6))           # seed changes the ids


def test_duplicate_ids_are_refused():
    with pytest.raises(ValueError, match="duplicate id"):
        P.check_unique_ids(np.array(["a", "b", "a"]))
    P.check_unique_ids(np.arange(10))                                         # int style passes


def test_unknown_id_style_is_rejected():
    with pytest.raises(ValueError, match="unknown id style"):
        P.make_ids(10, "sha1", 0)


def test_collision_can_no_longer_reach_the_party_files(tmp_path, monkeypatch):
    """Force a collision between an active-only row and a passive-only row (the silent case).
    partition() must stop before writing any file instead of producing a wrong join."""
    seed, overlap = 3, 0.8
    held = P.overlap_masks(569, 2, overlap, seed)
    i = int(np.flatnonzero(held[0] & ~held[1])[0])
    j = int(np.flatnonzero(held[1] & ~held[0])[0])
    real = P.entity_id

    def colliding(salt, row):
        return real(salt, i if row == j else row)

    monkeypatch.setattr(P, "entity_id", colliding)
    with pytest.raises(ValueError, match="duplicate id"):
        P.partition(P.load_dataset("breast_cancer"), 2, str(tmp_path), "random", seed, overlap)
    assert not (tmp_path / "active.csv").exists()


def test_manifest_records_id_scheme_and_alignment_still_matches(tmp_path):
    m = P.partition(P.load_dataset("breast_cancer"), 2, str(tmp_path), "random", 3, 0.8)
    assert m["params"]["id_scheme"] == "sha256-128"
    act = PartyData.from_csv(str(tmp_path / "active.csv"), require_label=True)
    pas = PartyData.from_csv(str(tmp_path / "passive-1.csv"))
    res = align_locally([act.ids, pas.ids], "salt")
    assert res[0].n_aligned == m["intersection_size"]
    assert [act.ids[r] for r in res[0].rows] == [pas.ids[r] for r in res[1].rows]   # same entity, every slot
