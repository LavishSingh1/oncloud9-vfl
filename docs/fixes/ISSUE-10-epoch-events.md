# Issue — epochs=10⁹ accepted; active party pre-allocates per-epoch state until OOM; STOP cannot interrupt

| | |
|---|---|
| Issue | `epochs` without an upper bound: active-party memory grows ~1.2 KB per configured epoch before training, OOM kill, STOP ignored #10 |
| Reported by | Order of Phoenix (baseline `rt-baseline`, commit `07ed305`) |
| Severity / triage | `major`, `confirmed` |
| Requirement | FR-9 (configure epochs without source edits), red-team rubric: crash/outage from an extreme input; cited NFR-10 only loosely applies (see 1) |
| Component | `vfl/active.py` (`ActiveNode` session loop, `on_stop`), `vfl/config_schema.py` (`epochs`) |

## 1. Validation

We tried to refute each claim before fixing it.

| # | Claim | Evidence we checked | Refutation tried | Verdict |
|---|---|---|---|---|
| 1 | `epochs=1000000` is accepted | `config_schema.py:109` had `min=1` and no maximum; `validate_session_config` accepted `epochs=10⁹` | none: the docs list only `int (>= 1)` | **Valid** |
| 2 | memory grows ~1.2 KB per epoch before training | `active.py:615-616` (`:531-532` on `rt-baseline`) ran `for e in range(cfg["epochs"]): s.epoch_events[e] = threading.Event()` after alignment and before the first epoch. A `threading.Event` measures **1,236 bytes** (tracemalloc, 100k events), so 10⁶ epochs is about 1.2 GB, matching the reporter's 0.39 → 1.83 GB | *"Fixed by the #3/#13 merges."* No: the loop is unchanged on `main` @ `95d8a66` | **Valid** |
| 3 | large values get the active party OOM-killed | follows from 2: 10⁹ × 1.2 KB ≈ 1.2 TB | none | **Valid** |
| 4 | STOP does not interrupt it | the loop has no stop check; `stop_event` is first read after the loop. `on_stop` also walked `epoch_events` without a lock while that loop was growing it | none | **Valid** |

The pre-allocation wasn't needed in the first place: `epoch_events` is a `defaultdict(threading.Event)`. The loop's
only possible purpose was to avoid a race where the waiting thread and the committing thread each create their own
event for the same epoch. We keep that guarantee by creating each epoch's event just before that epoch's barrier
(section 2).

**Refutations considered for validity and severity:**
- *"No PRD clause sets an epochs bound, and NFR-10 is about duplicates, delays and checkpoints."* True: the cited
  NFR-10 does not really apply. But the defect does not depend on a PRD bound. FR-9 (Must) makes `epochs` an
  operator-facing setting, and a value accepted by validation should not crash a party or disable STOP. The course
  rubric lists "a crash or service outage" as Major in its own right, and its required tests include extreme inputs.
- *"Only an authenticated operator can start a session (`@need("operator")`), and the PRD trusts the operator."*
  This makes it a robustness defect rather than a security one, but a legitimate operator can trigger it with a
  typo (extra zeros), and STOP then cannot recover it. It reduces exposure, not validity.
- *"Avoidable by not using huge values."* Nothing documented a limit, so the operator had no way to know.

We agree with **Major**: an extreme but accepted input crashes the active party and the session fails.

## 2. The fix

| Expected outcome (from the issue) | Now |
|---|---|
| a request above a documented maximum is rejected with HTTP 400 naming `epochs` and the range | `config_schema.MAX_EPOCHS = 10_000`; `epochs` has `min=1, max=10000`. Refused by `--set`, the form, `/api/config/validate` and `POST /api/sessions` (HTTP 400 `invalid session config: epochs must be <= 10000`) before anything reaches the parties. `docs/CONFIG.md` and the README table show `int (>= 1, <= 10000)`. Every PRD experiment uses ≤ 100 epochs |
| memory stays flat with respect to `epochs` | the pre-allocation loop is gone. Each epoch's event is created under `s.cv` just before that epoch's barrier, and dropped once the epoch's row is recorded, so at most one is live |
| STOP during setup ends the session within seconds | the unbounded setup loop no longer exists; setup does no work per configured epoch. `on_stop` walks the events under `s.cv` |

**Why creating the event before the barrier is race-free:** the epoch barrier goes through the coordinator
(`PartyNode.barrier`) and releases no party into epoch *e* until every party, the active one included, has reached
it. So no batch of epoch *e*, and therefore no eval commit that sets the event, can happen before the active party
has created the event.

## 3. Tests

`tests/test_epoch_bounds.py`: 7 tests. The 6 unit tests fail on `rt-baseline`. The end-to-end test needs `grpc`
and `torch` (marked `slow`).

| Test | Checks |
|---|---|
| `test_epochs_has_a_documented_maximum` | schema `min=1, max=10000`; `docs/CONFIG.md` shows the bound |
| `test_too_many_epochs_are_refused_on_every_path` (×3: 10 001, 10⁶, 10⁹) | refused by CLI overrides and by the coordinator's merged-config validation, message names `epochs` and `10000` |
| `test_the_maximum_itself_is_accepted` | 10 000 is valid |
| `test_active_party_does_not_preallocate_per_epoch_state` | no `range(cfg["epochs"])` loop stores into `epoch_events` (the reported line) |
| `test_coordinator_refuses_huge_epochs_and_stop_ends_a_long_session` | live mTLS deployment: the reporter's `epochs=10⁹` gets HTTP 400 and every party stays alive; a 10 000-epoch session trains, and STOP brings it to `STOPPED` within 15 s with every party alive |

## 4. How to verify

```bash
pytest -q tests/test_epoch_bounds.py tests/test_config_schema.py tests/test_integration.py
DATA=data bash scripts/run_local.sh
python -m vfl.cli start --set epochs=1000000000 --set dataset_tag=synthetic    # refused: epochs must be <= 10000
python -m vfl.cli start --set epochs=10000 --set dataset_tag=synthetic         # trains; active RSS stays near idle
python -m vfl.cli stop <session_id>                                            # STOPPED within seconds
bash scripts/stop_local.sh
```

## 5. Files changed

| File | Change |
|---|---|
| `vfl/config_schema.py` | `MAX_EPOCHS = 10_000`; `epochs` gets `max=MAX_EPOCHS` |
| `vfl/active.py` | no per-epoch pre-allocation; each epoch's event created before its barrier and dropped after its row; `on_stop` walks the events under `s.cv` |
| `docs/CONFIG.md`, `README.md` | regenerated parameter tables (`int (>= 1, <= 10000)`) |
| `tests/test_epoch_bounds.py` | new regression tests |
