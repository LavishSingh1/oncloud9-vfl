"""Replaceable entity-alignment interface (FR-2) and the conditional DH-PSI adapter (FR-12).

ExperimentalAlignment
    Salted SHA-256 of the entity ids is sent to the active party, which returns the
    ordered intersection. Deterministic and reproducible, but NOT privacy-preserving:
    a party holding a candidate id list can confirm membership by hashing it.
    It is always reported as "non-private experimental alignment".

DHPSIParty (two parties, semi-honest)
    Diffie-Hellman / commutative-encryption PSI (Meadows 1986; Huberman et al. 1999)
    over the RFC 3526 2048-bit MODP group, hashing ids into the quadratic-residue
    subgroup. Output policy: BOTH parties learn the intersection (required to train)
    and both learn the other's set SIZE. Assumes honest-but-curious parties. It is a
    textbook protocol implemented here for study; it has not had an external review.

MultiPSIHub + run_psi_member (three or more parties, semi-honest; v2.1)
    The same commutative blinding generalised to N parties over the system's star topology
    (passives can only reach the active party). Every party's set is blinded by EVERY party's
    key, hop by hop in a fixed ring order (active, passive-1, ..., passive-m), with the active
    party relaying sets between passives. Only fully blinded values H(x)^(k_A k_1 ... k_m) are
    comparable, so an intermediate holder can match nothing (a partially blinded set carries a
    different key product from every other set it sees). The active party intersects the fully
    blinded sets and tells each party the positions of the intersection in its own (secretly
    shuffled) list. Leakage, stated precisely in MULTI_PSI_DISCLOSURE: everyone learns the
    global intersection (its own rows) and every set size; the active party, which holds all
    fully blinded sets, additionally learns which passive parties hold each of ITS OWN ids and
    the size of every sub-intersection (e.g. |passive-1 ∩ passive-2|). Passives learn nothing
    about other parties' ids beyond the global intersection. Removing the active party's extra
    view needs an MPSI built on oblivious PRFs or homomorphic encryption (future work).

Both adapters produce the same object: an ordered list of local row indices, one
per aligned entity, in an order shared by every party.
"""
from __future__ import annotations

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass

# RFC 3526, group 14 (2048-bit MODP). p is a safe prime: p = 2q + 1.
MODP_2048_P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A0879"
    "8E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B"
    "0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA4836"
    "1C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804"
    "F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6"
    "955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
MODP_2048_Q = (MODP_2048_P - 1) // 2
ELEM_BYTES = 256


@dataclass
class AlignmentResult:
    rows: list[int]          # local row index for each aligned position
    mode: str
    private: bool
    n_local: int
    disclosure: str
    order_digest: str | None = None   # hash of the shared aligned order (reproducibility check)

    @property
    def n_aligned(self) -> int:
        return len(self.rows)

    def report(self) -> dict:
        """Counts and policy only - never ids."""
        return {"mode": self.mode, "private": self.private, "n_local": self.n_local,
                "n_aligned": self.n_aligned, "disclosure": self.disclosure, "order_digest": self.order_digest}


# ---------------------------------------------------------------- experimental
EXPERIMENTAL_DISCLOSURE = (
    "NON-PRIVATE experimental alignment: the active party receives salted SHA-256 hashes of "
    "every passive-party id and can test candidate ids; all parties learn the intersection.")


def salted_hash(salt: str, entity_id: str) -> bytes:
    return hashlib.sha256(f"{salt}\x1f{entity_id}".encode()).digest()


class ExperimentalAlignment:
    name = "experimental"
    private = False

    def __init__(self, salt: str):
        self.salt = salt

    def encode(self, ids: list[str]) -> list[bytes]:
        return [salted_hash(self.salt, i) for i in ids]

    @staticmethod
    def intersect(sets: list[list[bytes]]) -> list[bytes]:
        """Ordered intersection (sorted by hash => same order for everyone)."""
        common = set(sets[0])
        for s in sets[1:]:
            common &= set(s)
        return sorted(common)

    def rows_for(self, ids: list[str], ordered_hashes: list[bytes]) -> AlignmentResult:
        pos = {h: i for i, h in enumerate(self.encode(ids))}
        rows = [pos[h] for h in ordered_hashes]
        digest = hashlib.sha256(b"".join(ordered_hashes)).hexdigest()[:16]
        return AlignmentResult(rows, self.name, False, len(ids), EXPERIMENTAL_DISCLOSURE, digest)


def align_locally(id_lists: list[list[str]], salt: str) -> list[AlignmentResult]:
    """Single-process alignment used by baselines/local simulation; same order as networked."""
    ad = ExperimentalAlignment(salt)
    encoded = [ad.encode(ids) for ids in id_lists]
    order = ad.intersect(encoded)
    return [ad.rows_for(ids, order) for ids in id_lists]


# ---------------------------------------------------------------- DH-PSI
PSI_DISCLOSURE = (
    "DH-PSI (semi-honest, unreviewed implementation): both parties learn the intersection and "
    "each other's set size; non-matching ids are not revealed under the DDH assumption.")


def hash_to_group(entity_id: str, salt: str = "vfl-psi-v1") -> int:
    """Hash to the order-q subgroup of quadratic residues: (H(x) mod p)^2."""
    h = b""
    ctr = 0
    while len(h) < ELEM_BYTES + 16:
        h += hashlib.sha512(f"{salt}|{ctr}|{entity_id}".encode()).digest()
        ctr += 1
    v = int.from_bytes(h, "big") % MODP_2048_P
    return pow(v, 2, MODP_2048_P)


def to_bytes(v: int) -> bytes:
    return v.to_bytes(ELEM_BYTES, "big")


def from_bytes(b: bytes) -> int:
    v = int.from_bytes(b, "big")
    if not (1 < v < MODP_2048_P - 1):
        raise ValueError("PSI element out of range")
    return v


class DHPSIParty:
    """One side of two-party DH-PSI. The secret exponent never leaves the object."""

    name = "dh_psi"
    private = True

    def __init__(self, ids: list[str]):
        self.ids = ids
        # 256-bit short exponent (~128-bit security, standard practice for DH over
        # safe-prime MODP groups); keeps pow() cost low enough for ~10^4 ids.
        self._k = secrets.randbits(256) | (1 << 255)
        self._order = list(range(len(ids)))
        secrets.SystemRandom().shuffle(self._order)  # hide local row order

    def blind_own(self) -> list[bytes]:
        """H(x)^k for own ids, in a shuffled order."""
        return [to_bytes(pow(hash_to_group(self.ids[i]), self._k, MODP_2048_P)) for i in self._order]

    def reblind(self, items: list[bytes]) -> list[bytes]:
        """(peer's H(y)^j)^k, preserving order."""
        return [to_bytes(pow(from_bytes(b), self._k, MODP_2048_P)) for b in items]

    def finish_initiator(self, own_double: list[bytes], peer_double: list[bytes]
                         ) -> tuple[AlignmentResult, list[int]]:
        """Initiator (passive) holds H(x)^{ab} for its ids (in blind_own order) and
        H(y)^{ab} for the responder's ids (in the responder's shuffled order).
        Returns its alignment and, for the responder, the positions of matches in
        the responder's list ordered by the shared double-blinded value."""
        mine = {d: self._order[pos] for pos, d in enumerate(own_double)}
        peer = {d: pos for pos, d in enumerate(peer_double)}
        common = sorted(set(mine) & set(peer))
        rows = [mine[d] for d in common]
        peer_positions = [peer[d] for d in common]
        return AlignmentResult(rows, self.name, True, len(self.ids), PSI_DISCLOSURE), peer_positions

    def finish_responder(self, positions: list[int], disclosure: str = PSI_DISCLOSURE) -> AlignmentResult:
        """Responder (active; every party in the multi-party protocol) maps positions in its shuffled
        blinded list back to rows."""
        if len(set(positions)) != len(positions) or any(p < 0 or p >= len(self._order) for p in positions):
            raise ValueError("invalid PSI positions")
        rows = [self._order[p] for p in positions]
        return AlignmentResult(rows, self.name, True, len(self.ids), disclosure)


# ---------------------------------------------------------------- multi-party DH-PSI (v2.1)
MULTI_PSI_DISCLOSURE = (
    "multi-party DH-PSI (semi-honest, star topology through the active party, unreviewed implementation): every "
    "party learns the global intersection and every party's set size; the active party, which compares the fully "
    "blinded sets, also learns which passive parties hold each of its OWN ids and the size of every "
    "sub-intersection; passive parties learn nothing about other parties' ids beyond the global intersection. "
    "Non-matching ids are not revealed under the DDH assumption.")
ROUND_SUBMIT, ROUND_WORK, ROUND_RESULT = 11, 12, 13


class PSIProtocolError(ValueError):
    """A party sent something the multi-party protocol does not allow (reported, never trusted)."""


class MultiPSIHub:
    """Active-party side of N-party DH-PSI (N >= 3). Thread-safe: each passive's RPC thread calls submit(),
    work() (long-poll) and result(). Holds only blinded values; the active's secret key stays in self.me."""

    def __init__(self, ids: list[str], active_id: str, passive_ids: list[str], on_finish=None,
                 item_cap: int | None = None):
        if len(passive_ids) < 2:
            raise ValueError("MultiPSIHub is for >= 2 passive parties; use DHPSIParty for two parties")
        self.active_id = active_id
        self.ring = [active_id] + sorted(passive_ids)
        self.n = len(self.ring)
        self.passives = set(passive_ids)
        self.me = DHPSIParty(ids)
        from .limits import psi_item_cap
        self.item_cap = item_cap or psi_item_cap(len(ids))   # every submitted item is re-blinded by EVERY party (RT-08)
        self.on_finish = on_finish
        self.cv = threading.Condition()
        self.sets: dict[str, dict] = {}                 # owner -> {items, keys applied, holder}
        self.queue: dict[str, list[str]] = {p: [] for p in passive_ids}
        self.positions: dict[str, list[int]] = {}
        self.sizes: dict[str, int] = {}
        self.result: AlignmentResult | None = None
        self.error: str | None = None
        self.finished = False
        self._submitted: dict[str, str] = {}              # sender -> digest, so a retried submit is idempotent
        self._applied: set = set()                        # (sender, task) already applied: a retried result is acked
        own = self.me.blind_own()                        # expensive: outside the lock
        with self.cv:
            self.sizes[active_id] = len(own)
            self.sets[active_id] = {"items": own, "keys": 1, "holder": None}
            self._advance(active_id)

    # -- internal (call with self.cv held)
    def _next_hop(self, owner: str) -> str:
        return self.ring[(self.ring.index(owner) + self.sets[owner]["keys"]) % self.n]

    def _advance(self, owner: str):
        s = self.sets[owner]
        while s["keys"] < self.n:
            hop = self._next_hop(owner)
            if hop == self.active_id:                    # the active party's own turn: blind locally
                s["items"] = self.me.reblind(s["items"])
                s["keys"] += 1
                continue
            s["holder"] = hop
            self.queue[hop].append(owner)
            self.cv.notify_all()
            return
        s["holder"] = None
        if len(self.sets) == self.n and all(x["keys"] == self.n for x in self.sets.values()):
            self._finish()

    def _finish(self):
        full = {o: x["items"] for o, x in self.sets.items()}
        common = set(full[self.active_id])
        for o in self.ring[1:]:
            common &= set(full[o])
        order = sorted(common)                           # shared order: by fully blinded value
        for o in self.ring:
            pos = {v: i for i, v in enumerate(full[o])}
            self.positions[o] = [pos[v] for v in order]
        self.result = self.me.finish_responder(self.positions.pop(self.active_id), MULTI_PSI_DISCLOSURE)
        for o in self.sets:                               # keep nothing but positions
            self.sets[o]["items"] = []
        self.finished = True
        self.cv.notify_all()
        if self.on_finish:
            self.on_finish(self.result)

    def _fail(self, msg: str):
        self.error = msg
        self.cv.notify_all()
        raise PSIProtocolError(msg)

    # -- RPC entry points
    def submit(self, sender: str, items: list[bytes]):
        if sender not in self.passives:
            raise PSIProtocolError(f"{sender} is not a passive participant")
        if len(items) > self.item_cap:
            raise PSIProtocolError(f"{sender} submitted {len(items)} items; the limit is {self.item_cap}")
        for b in items:
            from_bytes(b)                                # range check before anything is stored
        digest = hashlib.sha256(b"".join(items)).hexdigest()
        with self.cv:
            if sender in self.sets:
                if self._submitted.get(sender) == digest:
                    return                               # the same message again (an RPC retry)
                raise PSIProtocolError(f"{sender} already submitted its set")
            self._submitted[sender] = digest
            self.sizes[sender] = len(items)
            self.sets[sender] = {"items": list(items), "keys": 1, "holder": None}
            self._advance(sender)

    def work(self, sender: str, wait_s: float = 5.0) -> dict:
        """Long-poll: a set to re-blind, 'done' with this party's positions, or 'wait'."""
        if sender not in self.passives:
            raise PSIProtocolError(f"{sender} is not a passive participant")
        deadline = time.time() + wait_s
        with self.cv:
            while True:
                if self.error:
                    raise PSIProtocolError(self.error)
                if self.queue[sender]:
                    owner = self.queue[sender][0]
                    return {"status": "task", "task": owner, "items": list(self.sets[owner]["items"]),
                            "n_parties": self.n}
                if self.finished:
                    return {"status": "done", "indices": list(self.positions.get(sender, [])),
                            "n_aligned": len(self.result.rows), "n_parties": self.n}
                left = deadline - time.time()
                if left <= 0:
                    return {"status": "wait", "n_parties": self.n}
                self.cv.wait(min(left, 0.5))

    def result_of(self, sender: str, task: str, items: list[bytes]):
        """`sender` returns the set `task` re-blinded with its key."""
        for b in items:
            from_bytes(b)
        with self.cv:
            q = self.queue.get(sender)
            if q is None:
                raise PSIProtocolError(f"{sender} is not a passive participant")
            if task not in q:
                if (sender, task) in self._applied:
                    return                               # already applied (an RPC retry)
                raise PSIProtocolError(f"{sender} was not asked to re-blind {task!r}")
            s = self.sets[task]
            if len(items) != len(s["items"]):
                self._fail(f"{sender} returned {len(items)} items for {task!r}, expected {len(s['items'])}")
            q.remove(task)
            self._applied.add((sender, task))
            s["items"] = list(items)
            s["keys"] += 1
            s["holder"] = None
            self._advance(task)

    def abort(self, msg: str = "alignment aborted"):
        with self.cv:
            if not self.finished:
                self.error = msg
                self.cv.notify_all()


def run_psi_member(ids: list[str], call, deadline_s: float = 3600.0, wait_s: float = 5.0) -> tuple[AlignmentResult, dict]:
    """Passive-party side of multi-party DH-PSI. `call(round, items=None, task="", wait_s=0)` sends one Align
    request to the active party and returns a dict (status, task, items, indices, n_aligned, n_parties, error).
    Returns the alignment and counters (rounds, items re-blinded)."""
    me = DHPSIParty(ids)
    stats = {"tasks": 0, "items_reblinded": 0, "polls": 0}
    r = call(ROUND_SUBMIT, items=me.blind_own())
    if r.get("error"):
        raise RuntimeError(f"PSI submit rejected: {r['error']}")
    t_end = time.time() + deadline_s
    while True:
        if time.time() > t_end:
            raise TimeoutError("multi-party PSI did not finish in time")
        r = call(ROUND_WORK, wait_s=wait_s)
        stats["polls"] += 1
        if r.get("error"):
            raise RuntimeError(f"PSI rejected: {r['error']}")
        st = r.get("status")
        if st == "task":
            out = me.reblind(list(r["items"]))
            stats["tasks"] += 1
            stats["items_reblinded"] += len(out)
            rr = call(ROUND_RESULT, items=out, task=r["task"])
            if rr.get("error"):
                raise RuntimeError(f"PSI result rejected: {rr['error']}")
        elif st == "done":
            stats["n_parties"] = r.get("n_parties")
            return me.finish_responder(list(r["indices"]), MULTI_PSI_DISCLOSURE), stats
        elif st != "wait":
            raise RuntimeError(f"unexpected PSI status {st!r}")


def psi_locally(id_lists: list[list[str]], names: list[str] | None = None) -> list[AlignmentResult]:
    """Run the networked PSI protocols in one process (tests, the attack harness): two parties use the v2
    two-party exchange, three or more use MultiPSIHub + run_psi_member over threads."""
    names = names or ["active"] + [f"passive-{i}" for i in range(1, len(id_lists))]
    if len(id_lists) == 2:
        act, pas = DHPSIParty(id_lists[0]), DHPSIParty(id_lists[1])
        own = pas.blind_own()
        res_p, positions = pas.finish_initiator(act.reblind(own), pas.reblind(act.blind_own()))
        return [act.finish_responder(positions), res_p]
    hub = MultiPSIHub(id_lists[0], names[0], names[1:])
    out: dict[str, AlignmentResult] = {}
    errors: list[BaseException] = []

    def member(name, ids):
        def call(rnd, items=None, task="", wait_s=0.0):
            if rnd == ROUND_SUBMIT:
                hub.submit(name, items)
                return {}
            if rnd == ROUND_WORK:
                return hub.work(name, wait_s)
            hub.result_of(name, task, items)
            return {}
        try:
            out[name] = run_psi_member(ids, call, deadline_s=600, wait_s=0.5)[0]
        except BaseException as e:                       # surfaced by the caller
            errors.append(e)

    ts = [threading.Thread(target=member, args=(n, ids)) for n, ids in zip(names[1:], id_lists[1:])]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errors:
        raise errors[0]
    return [hub.result] + [out[n] for n in names[1:]]
