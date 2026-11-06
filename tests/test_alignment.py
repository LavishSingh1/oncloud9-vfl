"""FR-2 / FR-12: alignment returns the reference intersection in one shared order."""
import random

from vfl.alignment import DHPSIParty, ExperimentalAlignment, align_locally


def _ids():
    rng = random.Random(0)
    common = [f"E{i:07d}" for i in range(200)]
    a = common + [f"A{i}" for i in range(30)]
    b = common[:150] + [f"B{i}" for i in range(40)]
    rng.shuffle(a)
    rng.shuffle(b)
    return a, b, set(common[:150])


def test_experimental_alignment_matches_reference_and_order():
    a, b, ref = _ids()
    ra, rb = align_locally([a, b], "salt")
    assert ra.n_aligned == rb.n_aligned == len(ref)
    assert [a[i] for i in ra.rows] == [b[i] for i in rb.rows]   # same entity at every position
    assert {a[i] for i in ra.rows} == ref
    assert not ra.private and "NON-PRIVATE" in ra.disclosure
    assert ra.order_digest == rb.order_digest


def test_experimental_alignment_deterministic():
    a, b, _ = _ids()
    r1 = align_locally([a, b], "salt")
    r2 = align_locally([a, b], "salt")
    assert r1[0].rows == r2[0].rows and r1[0].order_digest == r2[0].order_digest


def test_salt_changes_hashes():
    ad1, ad2 = ExperimentalAlignment("s1"), ExperimentalAlignment("s2")
    assert ad1.encode(["x"]) != ad2.encode(["x"])


def test_dh_psi_two_party():
    a, b, ref = _ids()
    passive, active = DHPSIParty(b), DHPSIParty(a)
    r1 = passive.blind_own()
    own_double = active.reblind(r1)
    peer_blind = active.blind_own()
    res_p, positions = passive.finish_initiator(own_double, passive.reblind(peer_blind))
    res_a = active.finish_responder(positions)
    assert res_p.n_aligned == res_a.n_aligned == len(ref)
    assert [b[i] for i in res_p.rows] == [a[i] for i in res_a.rows]
    assert {a[i] for i in res_a.rows} == ref
    assert res_p.private
    # blinded values reveal nothing obvious: no raw id bytes inside
    assert all(b"E00" not in x for x in r1)
