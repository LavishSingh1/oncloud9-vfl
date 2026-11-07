"""Protocol limits enforced on what an AUTHORISED peer may ask of a node (v2.3 red-team hardening).

mTLS decides who may talk; these limits decide how much one authorised party can make the others do. They are
far above anything the honest protocol needs, so they never change a correct run.
"""
from __future__ import annotations

# PSI: a party may submit at most this many blinded ids, relative to the active party's own set. Each item costs
# every other party one 2048-bit modular exponentiation, so an unbounded set is a CPU denial of service, and in
# two-party PSI it is also an unbounded membership oracle against the active party's ids.
PSI_FACTOR = 4
PSI_SLACK = 10_000
PSI_HARD_MAX = 5_000_000
# experimental alignment: salted hashes, 32 B each
ALIGN_HARD_MAX = 20_000_000

# coordinator: report and registration payloads (JSON text)
MAX_REPORT_JSON = 8 * 2 ** 20
MAX_SUMMARY_JSON = 1 * 2 ** 20
MAX_PROGRESS_JSON = 64 * 2 ** 10
MAX_RECORDS_PER_REPORT = 2000
MAX_SECURITY_EVENTS_PER_REPORT = 500

# training slots: how far ahead of the active party's current epoch a passive may send
EPOCH_LOOKAHEAD = 1
# ...and how many batches beyond the last committed one (each passive sends batch b+1 only after b's reply)
SLOT_AHEAD = 4


def psi_item_cap(n_active: int, cfg: dict | None = None) -> int:
    explicit = ((cfg or {}).get("security") or {}).get("max_align_items")
    if explicit:
        return min(PSI_HARD_MAX, int(explicit))
    return min(PSI_HARD_MAX, PSI_FACTOR * max(int(n_active), 1) + PSI_SLACK)


def check_slot(phase: str, epoch: int, batch: int, n_batches: int, current_epoch: int | None,
               epochs: int | None = None, committed: int | None = None) -> None:
    """Refuse a training/eval slot the honest schedule could not produce. v2.2 accepted negative indices (Python
    wrapped them onto real batches under a different slot key) and any epoch, so one passive could create slots
    faster than they complete, evict the live slot from the 64-slot window and park server threads."""
    if phase not in ("train", "eval"):
        raise ValueError(f"unknown phase {phase!r}")
    if not 0 <= int(batch) < int(n_batches):
        raise ValueError(f"no {phase} batch {batch} (0..{n_batches - 1})")
    if epoch < 0 or (epochs is not None and epoch >= epochs):
        raise ValueError(f"epoch {epoch} outside the session")
    # one epoch back is allowed: with the straggler `drop` policy a slow party may still be finishing the
    # previous epoch when the active party has moved on (its late message is answered as dropped)
    if current_epoch is not None and not current_epoch - 1 <= epoch <= current_epoch + EPOCH_LOOKAHEAD:
        raise ValueError(f"epoch {epoch} is not near the current epoch {current_epoch}")
    if committed is not None and batch > committed + SLOT_AHEAD:
        raise ValueError(f"{phase} batch {batch} is too far ahead of the schedule ({committed} committed)")
