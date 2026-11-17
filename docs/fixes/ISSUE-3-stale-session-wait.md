# Issue — Stale session ids held for 10 s at the active party

| | |
|---|---|
| Issue | "10-second wait loop in `session_for`: stale session ids time out clients and exhaust the per-peer concurrency cap" #3 |
| Reported by | Order of Phoenix (baseline `rt-baseline`, commit `07ed305`) |
| Severity / triage | `minor`, `confirmed` |
| Requirement | FR-8 (message identity and duplicate handling), NFR-10 (fault handling and diagnostic timeout) |
| Component | `vfl/active.py` (`ActiveNode.session_for`, `_finish`, `ActiveSession.handle_exchange`), `vfl/party_base.py` (heartbeat loop) |

## 1. Summary

The report is **valid**. `session_for` waited up to 10 s for *any* session id it did not recognise, whatever the
caller's deadline. A client with a 3 s deadline got `DEADLINE_EXCEEDED` instead of a diagnostic, and every waiting
call kept one of the caller's `--max-peer-concurrency` slots, so 16 stale calls blocked the 17th with
`RESOURCE_EXHAUSTED`.

The wait exists for one legitimate reason: a passive party can receive START from the coordinator up to one
heartbeat (~1 s) before the active party does, and its first `Align` must not fail. **The wait cannot simply be
removed:** the passive party's `_call` retries only `UNAVAILABLE` and `DEADLINE_EXCEEDED`, so an early
`FAILED_PRECONDITION` would kill a real session at startup.

While fixing it we found a second wait of the same kind: `handle_exchange` waited up to `slot_timeout_s` (120 s by
default) for `ready` on a session that had already failed during alignment, so `ready` was never going to be set.

## 2. The fix: work out which case applies, and wait only for the START race

| Case | v2.3.1 | Now |
|---|---|---|
| Id of the current live session | served | served (lock-free fast path) |
| Id of a session that has ended on this node | served the dead session object | `FAILED_PRECONDITION` "it has already ended", immediately |
| Another session is running here | waited 10 s | `FAILED_PRECONDITION` "a different session is running", immediately |
| Malformed id | waited 10 s | `INVALID_ARGUMENT`, immediately |
| START applied, session object being built | waited (ms) | waits (ms) |
| Idle, new id | waited 10 s | **polls the coordinator now** (early heartbeat). If no START with that id is pending, `FAILED_PRECONDITION` within milliseconds; if one is, it is applied and the call proceeds |
| Coordinator unreachable | waited 10 s | waits at most `START_GRACE_S` = 3 s **and** never past the caller's deadline minus 0.2 s, then `FAILED_PRECONDITION` |

Further limits:
- **At most 2 waiting calls per peer identity** (`MAX_SESSION_WAITERS_PER_PEER`). Any more are refused at once, so
  stale ids can never fill the 16-slot budget. A real passive party has one `Align` in flight at startup.
- **Ended sessions are marked closed** in `_finish`. `handle_exchange` waits for `ready` in 0.25 s steps and returns
  `FAILED_PRECONDITION` once the session is closed, instead of holding the call for `slot_timeout_s`.
- **Nudged heartbeats are at least 0.1 s apart** (`HB_MIN_GAP_S`), so a flood of stale calls cannot become a flood
  of heartbeats on the coordinator.

`PartyNode.poll_coordinator(timeout)` returns True only after a heartbeat that *began after the call* has succeeded.
A heartbeat already in flight may predate the passive party's START, so it is not counted.

## 3. Tests

`tests/test_session_lookup.py` (6 tests; 5 fail on `rt-baseline`, the START-race test passes on both on purpose):

| Test | Checks |
|---|---|
| `test_unknown_session_fails_fast_with_a_diagnostic` | the reported case 1: 3 s deadline, `FAILED_PRECONDITION` in < 0.5 s (baseline: 3.0 s, `DEADLINE_EXCEEDED`) |
| `test_stale_calls_do_not_hold_concurrency_slots` | the reported case 2: 16 concurrent stale calls are all answered in < 1.5 s and no waiter slot is leaked (baseline: 10 s) |
| `test_start_race_is_still_tolerated` | a call that arrives before the active party hears START still succeeds |
| `test_other_or_ended_sessions_are_refused_immediately` | a different running session, and an ended session, are refused in < 50 ms |
| `test_malformed_session_id_is_invalid_argument` | `../../etc` is refused with `INVALID_ARGUMENT` |
| `test_waits_are_capped_by_the_callers_deadline_and_per_peer` | with the coordinator down: the answer arrives before the client deadline, 8 of 10 calls from one identity are refused at once, and another identity is unaffected |

## 4. How to verify

```bash
pytest -q tests/test_session_lookup.py tests/test_redteam.py tests/test_security_gate.py tests/test_integration.py
bash scripts/stop_local.sh; bash scripts/run_local.sh     # restart so the patched code is running
python3 scripts/verify_stale_session.py --sessions 3      # exit status 0 = resolved
bash scripts/stop_local.sh
```

`scripts/verify_stale_session.py` runs the reporter's three steps with pass/fail checks:
- step 1 gets `FAILED_PRECONDITION` in under 1 s;
- step 2's 16 calls are all answered in seconds;
- step 3 is not `RESOURCE_EXHAUSTED`.

`--sessions 3` then trains three real sessions back to back, which exercises the START race. The last one runs
while 8 threads flood the active party with stale ids. All three must reach `COMPLETED`.

The reporter's step 3 expects "Call succeeded!" for `real-session`, but the coordinator never started a session with
that id either, so a fast `FAILED_PRECONDITION` is the correct answer. The bug was the `RESOURCE_EXHAUSTED`.

## 5. Files changed

| File | Change |
|---|---|
| `vfl/active.py` | `session_for` rewritten as above; `ActiveNode._finish` marks the session closed; `ActiveSession.closed`; `handle_exchange` stops waiting once the session is closed; the experimental-alignment wait also checks `closed` |
| `vfl/party_base.py` | heartbeat sequence counters, `poll_coordinator()`, minimum 0.1 s between early heartbeats |
| `tests/test_session_lookup.py` | new regression tests |
| `scripts/verify_stale_session.py` | live check against a running deployment (reporter's steps + real sessions) |
