"""Protobuf mapping for multi-party DH-PSI (v2.1). Imports protobuf only (no grpc), so the full message flow
can be tested without a network: tests serialise every request and reply exactly as gRPC would.

Rounds (AlignRequest.round, mode "dh_psi", sessions with >= 2 passive parties):
  11 submit   passive -> active: H(x)^k_i for its own ids (shuffled)
  12 work     passive -> active, long-poll: reply status "task" (a set to re-blind, `task` = its owner),
              "wait" (nothing yet) or "done" (`indices` = this party's positions of the global intersection)
  13 result   passive -> active: the set `task`, re-blinded with the passive's key
Two-party sessions keep the v2 rounds 1 and 2 unchanged.
"""
from __future__ import annotations

from .alignment import ROUND_RESULT, ROUND_SUBMIT, ROUND_WORK, MultiPSIHub, PSIProtocolError
from .proto import vfl_pb2 as pb

WORK_WAIT_S = 5.0                     # server-side long-poll; the client's RPC deadline must exceed it


def request(header, rnd: int, items=None, task: str = "") -> pb.AlignRequest:
    return pb.AlignRequest(header=header, mode="dh_psi", round=int(rnd), items=list(items or []), task=task or "")


def hub_reply(hub: MultiPSIHub, sender: str, req: pb.AlignRequest, wait_s: float = WORK_WAIT_S) -> pb.AlignReply:
    try:
        if req.round == ROUND_SUBMIT:
            hub.submit(sender, list(req.items))
            return pb.AlignReply(ok=True, n_parties=hub.n)
        if req.round == ROUND_WORK:
            w = hub.work(sender, wait_s)
            return pb.AlignReply(ok=True, status=w["status"], task=w.get("task", ""), items=w.get("items", []),
                                 indices=w.get("indices", []), n_aligned=w.get("n_aligned", 0), n_parties=hub.n)
        if req.round == ROUND_RESULT:
            hub.result_of(sender, req.task, list(req.items))
            return pb.AlignReply(ok=True, n_parties=hub.n)
        if req.round in (1, 2):
            return pb.AlignReply(ok=False, error=f"this session has {hub.n} parties: use multi-party PSI "
                                                 f"rounds {ROUND_SUBMIT}-{ROUND_RESULT} (upgrade this party)")
        return pb.AlignReply(ok=False, error=f"bad PSI round {req.round}")
    except (PSIProtocolError, ValueError) as e:
        return pb.AlignReply(ok=False, error=str(e))


def reply_dict(rep: pb.AlignReply) -> dict:
    if not rep.ok:
        return {"error": rep.error or "rejected"}
    return {"status": rep.status, "task": rep.task, "items": list(rep.items), "indices": list(rep.indices),
            "n_aligned": rep.n_aligned, "n_parties": rep.n_parties}
