# Issue #13 — Sessions left live after a coordinator restart

| | |
|---|---|
| Issue | [#13](https://github.com/LavishSingh1/oncloud9-vfl/issues/13) "In-flight session becomes a zombie after a coordinator restart" |
| Reported by | Order of Phoenix (baseline `rt-baseline`, commit `07ed305`) |
| Severity / triage | `minor`, `confirmed` |
| Requirement | FR-8 / NFR-10 (state and fault handling) |
| Status | Fix in branch `fix/13-zombie-session` (linked PR); `retest-requested` from the reporting red team |
| Component | `vfl/coordinator.py` (`CoordinatorState._load_index`, `_recover_interrupted_sessions`, `stop_session`) |

## 1. Summary

The report is **valid**. The coordinator keeps live sessions in memory and writes the session index
(`runs/sessions_index.json`) only when a session starts and when it ends. If the process is killed in between, the
index still says `STARTING` and `summary.json` still says `RUNNING`. A restarted coordinator loaded that index as
history, so:

* `status` and the dashboard listed the session as `STARTING 0/40` forever;
* `stop` looked only at in-memory sessions and answered `400 unknown session`;
* `summary.json` was never corrected.

Checkpoints and resume were not affected, as the report says.

## 2. Root cause

Nothing reconciled the on-disk session records with reality when the coordinator started. A session from a previous
process can never finish: its in-memory state is gone, and its parties end it themselves after losing the
coordinator (about 60 s of retries). Its last recorded state was therefore stale by construction.

## 3. The fix

**On start-up** (`CoordinatorState.__init__` → `_recover_interrupted_sessions`), every index entry in a live state
(`STARTING`, `RUNNING`, `STOPPING`, `FAILING`) is closed:

| Recorded state | Becomes | Recorded reason |
|---|---|---|
| `STOPPING` (operator had asked to stop) | `STOPPED` | "interrupted: the coordinator restarted while this session was live" |
| `STARTING`, `RUNNING`, `FAILING` | `FAILED` | same |
| any live state, but `summary.json` already terminal (process died between writing `summary.json` and the index) | the state in `summary.json` | its own reason; no error added |

For each closed session:

* **Index and `status`/dashboard:** state, real `completed_epochs` (from `summary.json`), last test AUROC, finish
  time, reason, and an entry in `errors`.
* **`summary.json`:** state, reason, finish time, plus an entry in `errors` and in `events`. It is written
  atomically (temporary file, then `os.replace`). Everything else is kept, including `completed_epochs`,
  `epoch_metrics` and the config.
* **Coordinator log:** one warning line per closed session.

The existing terminal states are reused rather than adding `INTERRUPTED`. `vfl.cli wait` and `bench/harness.py`
treat exactly `COMPLETED`, `FAILED` and `STOPPED` as final, so a new state would make them wait forever. The reason
string records that the session was interrupted.

**`stop` on a session from before the restart** returns that session's record with
`"note": "session already ended (FAILED); nothing to stop"` instead of `400 unknown session`. An id the coordinator
has never seen is still refused.

**Resume** is unchanged and still works: `start --resume-from <sid>` reads the session's config and
`completed_epochs` from disk and continues from the last completed epoch.

**Robustness:** `_load_index` now keeps only well-formed entries (objects with a string `session_id`). Before, one
malformed entry in `sessions_index.json` made every later index write fail. Entries with an invalid session id are
never used to build a file path.

## 4. Tests

`tests/test_coordinator_restart.py` (5 tests). A restart is simulated by building a second `CoordinatorState` on the
same run directory without finishing the first, which is what `kill -9` leaves on disk.

| Test | Checks |
|---|---|
| `test_interrupted_session_is_closed_on_restart` | the reported case: after the restart the session is `FAILED` with 1 of 40 epochs in the index, in `summary.json` and in what `status` prints, with the reason and last AUROC |
| `test_stop_on_a_session_from_before_the_restart_is_a_clear_no_op` | `stop` returns the record with "already ended (FAILED)" and changes nothing; an unknown id is still refused |
| `test_resume_still_works_after_the_restart` | resume from the interrupted session starts at its last completed epoch with its original config |
| `test_recovery_is_idempotent_and_leaves_finished_sessions_alone` | a second restart changes nothing |
| `test_other_live_states_and_edge_cases` | `STOPPING` → `STOPPED`; a stale index entry whose `summary.json` is `COMPLETED` is synced without adding an error; finished sessions are untouched; malformed entries and path-like ids are ignored |

On `rt-baseline`, 4 of the 5 behaviours fail. The resume test passes on both, as it should: it guards behaviour the
report says already works. Run unchanged on `rt-baseline`, the file does not import at all, because
`INTERRUPTED_REASON` does not exist there.

## 5. How to verify

```bash
pytest -q tests/test_coordinator_restart.py tests/test_coordinator.py
bash scripts/verify_coordinator_restart.sh       # exit status 0 = resolved; about 2.5 minutes
```

`scripts/verify_coordinator_restart.sh` runs the reporter's steps against a real local deployment (mTLS, separate
processes) and checks each expected outcome:

1. Start a 40-epoch session.
2. `kill -9` the coordinator after the first epoch report.
3. Wait for the parties to give up, then restart the coordinator.
4. Check that `status` shows `FAILED` with the completed epochs (not `STARTING`).
5. Check that `stop` returns the "already ended" note.
6. Check that `summary.json` says `FAILED` with at least one epoch, and that the checkpoints are still there.
7. Resume from the interrupted session; it must start at its last completed epoch and reach `COMPLETED`.

## 6. Files changed

| File | Change |
|---|---|
| `vfl/coordinator.py` | `TERMINAL_STATES`, `INTERRUPTED_REASON`; `_recover_interrupted_sessions()` called at start-up; `stop_session` no-op for earlier sessions; `_load_index` skips malformed entries |
| `tests/test_coordinator_restart.py` | new regression tests |
| `scripts/verify_coordinator_restart.sh` | live check with the reporter's steps |
| `CHANGELOG-v2.md`, `CONTRIBUTORS.md` | changelog entry; ownership of the new files |
| `docs/fixes/ISSUE-13-coordinator-restart.md` | this document |
