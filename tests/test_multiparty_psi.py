"""v2.1: DH-PSI for three or more parties (vfl/alignment.MultiPSIHub + run_psi_member, vfl/psi_wire.py).

Needs no grpc or torch: the protocol is driven in-process, and the wire test serialises every request and
reply with the real protobuf messages, exactly as the gRPC transport would carry them.
"""
import random
import threading

import pytest

from vfl import psi_wire
from vfl.alignment import (MULTI_PSI_DISCLOSURE, PSI_DISCLOSURE, ROUND_RESULT, ROUND_SUBMIT, ROUND_WORK, DHPSIParty,
                           MultiPSIHub, PSIProtocolError, psi_locally, run_psi_member, to_bytes)
from vfl.proto import vfl_pb2 as pb


def _lists(n_parties, n_ids=120, keep=0.8, seed=0):
    rng = random.Random(seed)
    base = [f"E{i:06d}" for i in range(n_ids)]
    out = []
    for _ in range(n_parties):
        ids = rng.sample(base, int(keep * n_ids)) + [f"X{rng.random()}" for _ in range(5)]   # some private-only ids
        rng.shuffle(ids)
        out.append(ids)
    return out


def _check(lists, results):
    want = set(lists[0]).intersection(*map(set, lists[1:]))
    seqs = [[ids[i] for i in r.rows] for ids, r in zip(lists, results)]
    assert set(seqs[0]) == want and len(seqs[0]) == len(want)
    assert all(s == seqs[0] for s in seqs), "parties disagree on the aligned order"
    return want


@pytest.mark.parametrize("n", [3, 4, 6])
def test_every_party_gets_the_global_intersection_in_one_order(n):
    lists = _lists(n, seed=n)
    res = psi_locally(lists)
    want = _check(lists, res)
    assert 0 < len(want) < len(lists[0])
    assert all(r.private and r.mode == "dh_psi" and r.disclosure == MULTI_PSI_DISCLOSURE for r in res)


def test_two_parties_keep_the_v2_protocol():
    lists = _lists(2, seed=9)
    res = psi_locally(lists)
    _check(lists, res)
    assert all(r.disclosure == PSI_DISCLOSURE for r in res)
    with pytest.raises(ValueError, match=">= 2 passive"):
        MultiPSIHub(lists[0], "active", ["passive-1"])


# ------------------------------------------------------------------ over the wire (serialised protobuf)
def _wire_run(lists, names, retry=False):
    hub = MultiPSIHub(lists[0], names[0], names[1:])
    seen = {n: [] for n in names[1:]}               # every set a passive was handed, as it received it
    finals = {}

    def member(name, ids):
        def call(rnd, items=None, task="", wait_s=0.0):
            hdr = pb.MsgHeader(session_id="s", sender=name, msg_type=f"psi_r{rnd}")
            raw = psi_wire.request(hdr, rnd, items, task).SerializeToString()
            req = pb.AlignRequest.FromString(raw)                       # what the active party receives
            rep = pb.AlignReply.FromString(psi_wire.hub_reply(hub, req.header.sender, req, wait_s=0.3)
                                           .SerializeToString())
            if retry and rnd in (ROUND_SUBMIT, ROUND_RESULT):           # the RPC helper retried: must be harmless
                rep2 = psi_wire.hub_reply(hub, req.header.sender, req, wait_s=0.3)
                assert rep2.ok, rep2.error
            d = psi_wire.reply_dict(rep)
            if d.get("status") == "task":
                seen[name].append((d["task"], set(d["items"])))
            if d.get("status") == "done":
                finals[name] = d
            return d
        finals[name + ":result"] = run_psi_member(ids, call, deadline_s=120, wait_s=0.3)[0]

    ts = [threading.Thread(target=member, args=(n, ids)) for n, ids in zip(names[1:], lists[1:])]
    for t in ts:
        t.start()
    for t in ts:
        t.join(120)
    return hub, seen, finals


@pytest.mark.parametrize("retry", [False, True])
def test_full_message_flow_over_protobuf(retry):
    names = ["active", "passive-1", "passive-2", "passive-3"]
    lists = _lists(4, seed=11)
    hub, seen, finals = _wire_run(lists, names, retry=retry)
    _check(lists, [hub.result] + [finals[n + ":result"] for n in names[1:]])
    for n in names[1:]:
        assert finals[n]["n_parties"] == 4 and finals[n]["n_aligned"] == len(hub.result.rows)


# ------------------------------------------------------------------ privacy properties
def test_intermediate_holders_can_match_nothing():
    """Every set a passive re-blinds carries a different key product, so even though the underlying ids
    overlap heavily, no two sets a passive sees share a single value; and it never sees its own set again."""
    names = ["active", "passive-1", "passive-2", "passive-3"]
    lists = _lists(4, keep=0.95, seed=3)
    hub, seen, finals = _wire_run(lists, names)
    for n in names[1:]:
        owners = [o for o, _ in seen[n]]
        assert n not in owners and len(owners) == len(set(owners)) == 3           # each other set exactly once
        sets = [v for _, v in seen[n]]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                assert not (sets[i] & sets[j]), f"{n} could match two sets it re-blinded"


def test_passives_learn_only_their_positions_and_the_hub_keeps_nothing():
    names = ["active", "passive-1", "passive-2"]
    lists = _lists(3, seed=5)
    hub, seen, finals = _wire_run(lists, names)
    for n in names[1:]:
        assert set(finals[n]) == {"status", "task", "items", "indices", "n_aligned", "n_parties"}
        assert finals[n]["items"] == [] and finals[n]["task"] == ""
        assert len(finals[n]["indices"]) == len(hub.result.rows)
    assert all(not s["items"] for s in hub.sets.values())              # fully blinded sets dropped after use
    assert hub.positions.keys() == {"passive-1", "passive-2"}           # nothing about the active's own list


def test_blinded_values_carry_no_id_bytes():
    p = DHPSIParty(["E000123", "E000124"])
    assert all(b"E000" not in x and len(x) == 256 for x in p.blind_own())


# ------------------------------------------------------------------ protocol abuse is refused
def test_protocol_abuse_is_refused():
    lists = _lists(3, seed=7)
    hub = MultiPSIHub(lists[0], "active", ["passive-1", "passive-2"])
    p1, p2 = DHPSIParty(lists[1]), DHPSIParty(lists[2])
    with pytest.raises(PSIProtocolError, match="not a passive"):
        hub.submit("rogue", p1.blind_own())
    with pytest.raises(ValueError, match="out of range"):
        hub.submit("passive-1", [to_bytes(1)])
    hub.submit("passive-1", p1.blind_own())
    with pytest.raises(PSIProtocolError, match="already submitted"):
        hub.submit("passive-1", DHPSIParty(lists[1]).blind_own())      # a different set: not a retry
    w = hub.work("passive-1", 0.2)                                      # active's set waits for passive-1
    assert w["status"] == "task" and w["task"] == "active"
    with pytest.raises(PSIProtocolError, match="not asked"):
        hub.result_of("passive-2", "active", p2.reblind(w["items"]))   # only the assigned holder may answer
    with pytest.raises(PSIProtocolError, match="expected"):
        hub.result_of("passive-1", "active", p1.reblind(w["items"])[:-1])
    with pytest.raises(PSIProtocolError):                               # the run is aborted after a bad result
        hub.work("passive-2", 0.1)


def test_v2_two_party_client_gets_a_clear_error():
    lists = _lists(3, seed=8)
    hub = MultiPSIHub(lists[0], "active", ["passive-1", "passive-2"])
    req = pb.AlignRequest(header=pb.MsgHeader(sender="passive-1"), mode="dh_psi", round=1,
                          items=DHPSIParty(lists[1]).blind_own())
    rep = psi_wire.hub_reply(hub, "passive-1", req)
    assert not rep.ok and "multi-party" in rep.error


def test_member_stops_on_error_reply():
    def call(rnd, items=None, task="", wait_s=0.0):
        return {} if rnd == ROUND_SUBMIT else {"error": "session stopped"}
    with pytest.raises(RuntimeError, match="session stopped"):
        run_psi_member(["a", "b"], call, deadline_s=5)
    assert ROUND_WORK == 12
