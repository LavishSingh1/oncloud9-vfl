"""Issue #19: dh_psi alignment was not reproducible.

DH-PSI orders the intersection by values blinded with per-session secret exponents, so two sessions with the same
seed got a different aligned order (and so a different train/test split and different batches), and order_digest
was never set. canonical_order re-sorts each party's matched rows locally by the salted id hash that experimental
alignment uses: no extra message, no extra disclosure, the same order at every party and in every session.
"""
import random

import pytest

from vfl.alignment import align_locally, canonical_order, psi_locally
from vfl.config import alignment_salt

SALT = alignment_salt({"seed": 42})


def _id_lists(n_parties, seed=0):
    rng = random.Random(seed)
    common = [f"e{i}" for i in range(60)]
    lists = []
    for p in range(n_parties):
        own = common + [f"p{p}-x{i}" for i in range(15)]     # non-matching ids at every party
        rng.shuffle(own)
        lists.append(own)
    return lists


def _ids_in_order(res, ids):
    return [ids[r] for r in res.rows]


@pytest.mark.parametrize("n_parties", [2, 3])
def test_raw_psi_order_changes_between_sessions(n_parties):
    """The reported weakness: the protocol's own output order depends on the per-session secrets."""
    lists = _id_lists(n_parties)
    a = _ids_in_order(psi_locally(lists)[0], lists[0])
    b = _ids_in_order(psi_locally(lists)[0], lists[0])
    assert sorted(a) == sorted(b)          # same intersection...
    assert a != b                          # ...in a different order (60! orders; equal by chance ~never)


@pytest.mark.parametrize("n_parties", [2, 3])
def test_canonical_order_is_identical_across_sessions_and_parties(n_parties):
    lists = _id_lists(n_parties)
    runs = []
    for _ in range(3):
        res = [canonical_order(r, ids, SALT) for r, ids in zip(psi_locally(lists), lists)]
        orders = [_ids_in_order(r, ids) for r, ids in zip(res, lists)]
        assert all(o == orders[0] for o in orders)                  # every party: same entity per position
        assert len({r.order_digest for r in res}) == 1 and res[0].order_digest
        runs.append((orders[0], res[0].order_digest))
    assert runs[0] == runs[1] == runs[2]                            # every session: same order and digest


@pytest.mark.parametrize("n_parties", [2, 3])
def test_canonical_order_matches_experimental(n_parties):
    """Same intersection, same salt -> the same aligned order and order_digest as experimental mode."""
    lists = _id_lists(n_parties)
    exp = align_locally(lists, SALT)
    psi = [canonical_order(r, ids, SALT) for r, ids in zip(psi_locally(lists), lists)]
    for e, p in zip(exp, psi):
        assert p.rows == e.rows and p.order_digest == e.order_digest
        assert p.mode == "dh_psi" and p.private and not e.private   # still reported as the private adapter


def test_canonical_order_keeps_the_result_fields():
    lists = _id_lists(2)
    raw = psi_locally(lists)[1]
    res = canonical_order(raw, lists[1], SALT)
    assert sorted(res.rows) == sorted(raw.rows)
    assert (res.mode, res.private, res.n_local, res.disclosure) == (raw.mode, raw.private, raw.n_local,
                                                                    raw.disclosure)
