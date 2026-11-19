# Issue — empty per-session resources.jsonl, unkeyed epoch rows, coordinator CPU/RSS not recorded, lifetime aggregate reported as per-run

| | |
|---|---|
| Issue | per-session resource records missing or mis-scoped (FR-10 / NFR-6) #20 |
| Reported by | Order of Phoenix (baseline `rt-baseline`, commit `07ed305`; also on `main` @ `95d8a66`) |
| Severity / triage | `minor`, `confirmed` (severity as filed) |
| Requirement | FR-10 (Must): "per-batch and per-run records for … resource use, keyed by run/session ID"; NFR-6: "Record coordinator/party CPU and memory use" |
| Component | `vfl/telemetry.py` (`ResourceMonitor`), `vfl/party_base.py`, `vfl/active.py`, `vfl/passive.py`, `vfl/coordinator.py`, `bench/analyze.py` |

## 1. Validation

We tried to refute each claim before fixing it. Every cited line matches `main`.

| # | Claim | Evidence we checked | Refutation tried | Verdict |
|---|---|---|---|---|
| 1 | `resources.jsonl` is 0 bytes for every session | `party_base.py:64` builds `ResourceMonitor(interval=1.0)` with no writer; `SessionLog` opens `resources.jsonl` (`telemetry.py:164`) and nothing ever writes it. README:524 and docs §5.2 (NFR-6 row) cite it as the evidence | *"Per-session CPU/RSS is recorded anyway: every epoch row carries a `resources` snapshot."* True, but only one snapshot per epoch, none during setup or alignment, and not in the file the docs point to | **Valid** |
| 2 | epoch rows have no `session` key; batch rows do | `active.py:674` / `passive.py:234` write the row as is; `party_base.py:306` tags batch rows; the coordinator's `runs/<sid>/epochs.jsonl` rows had only `party` | *"Already keyed: the files live in `logs/<party>/<session>/` and `runs/<session>/`."* Largely true, so this part alone is weak. Records stop being self-describing once files are combined, and batch rows already carry the key, so we made them consistent (no risk) | **Weak, fixed for consistency** |
| 3 | the coordinator's CPU/RSS is saved in no run record | `coordinator.py:134` starts a monitor; it only feeds the live `/api/state` value (`:623`) | none: NFR-6 names the coordinator explicitly | **Valid** |
| 4 | the per-run aggregate covers the whole process, not the session | the monitor starts with the node and never resets; `run_summary.json` and `summary.json` finals took `monitor.summary()` over all samples since process start (reporter: `n_samples` 513 for a 0.3 s session). The benchmark's NFR-6 line printed it as the run's CPU/RSS | none | **Valid** |

Found along the way: the lifetime sample list grew by one tuple per second forever. It is now bounded.

## 2. The fix

`ResourceMonitor` gets a **per-session window**:
- `begin(writer, tags)` opens it, and each sample from then on is also written, tagged, to that session's file.
- `session_summary()` aggregates the window only.
- `end(session)` closes it.
- One sample is taken at begin and one at end, so even a sub-second session has records.
- The lifetime buffer is capped at 3,600 samples and only feeds the live dashboard value.
- Every aggregate now says what it covers: `"scope": "session"` or `"process"`, plus `window_s` and `interval_s`.

| Claim | Now |
|---|---|
| 1 | each party opens the window at START, so `logs/<party>/<session>/resources.jsonl` gets one row per second: `{"session", "party", "role", "t", "cpu_percent", "rss_mib"}` |
| 2 | epoch rows carry `session` at the party (`epochs.jsonl`) and at the coordinator (`runs/<sid>/epochs.jsonl`) |
| 3 | the coordinator opens its own window when it creates a session: `runs/<sid>/resources_coordinator.jsonl` (`{"session", "node": "coordinator", …}`) and `coordinator_resources` in `summary.json` and the session API |
| 4 | `finals.<party>.resources` and `run_summary.json` hold the session window (`_finish` records it for every outcome, including stopped and failed sessions). The NFR-6 benchmark line lists the coordinator too, and marks old lifetime aggregates "(process lifetime)" |

README (file table) and docs §5.2 (NFR-6 row) now describe these files.

## 3. Tests

`tests/test_resource_records.py`: 8 tests. The 7 unit tests fail on `rt-baseline`. The end-to-end test needs `grpc`
and `torch` (marked `slow`).

| Test | Checks |
|---|---|
| `test_session_window_writes_tagged_samples_and_aggregates_only_that_session` | samples before the session are excluded; rows are tagged; nothing written after `end` |
| `test_a_sub_second_session_still_has_samples` | a session shorter than one interval still gets 2 rows |
| `test_end_only_closes_its_own_session_and_a_new_begin_replaces_the_window` | windows cannot be closed by the wrong session |
| `test_a_closed_session_file_does_not_break_sampling` | sampling survives a closed log file |
| `test_process_lifetime_buffer_is_bounded` | lifetime buffer capped |
| `test_monitor_thread_feeds_the_window` | the background thread writes into the window |
| `test_nfr6_line_includes_the_coordinator_and_marks_lifetime_aggregates` | benchmark NFR-6 line |
| `test_per_session_resource_records_end_to_end` | live deployment, 6 s idle then a 3-epoch session: both parties' `resources.jsonl` non-empty and tagged; epoch rows keyed; `n_samples` matches the file and the session length, not uptime; coordinator file and `coordinator_resources` present |

## 4. How to verify (the reporter's steps)

```bash
DATA=data bash scripts/run_local.sh
sleep 30
python -m vfl.cli start --set epochs=6 --set batch_size=16 --wait
S=$(ls -t runs | grep '^s2' | head -1); echo "session $S"
ls -l logs/active/$S/resources.jsonl logs/passive-1/$S/resources.jsonl runs/$S/resources_coordinator.jsonl   # non-empty
grep -c '"session"' logs/active/$S/epochs.jsonl logs/active/$S/batches.jsonl                               # 6 and N
python - <<'P'
import json, os
S = os.popen('ls -t runs | grep ^s2 | head -1').read().strip()
d = json.load(open(f"runs/{S}/summary.json"))
for p, fr in sorted(d['finals'].items()):
    r = fr['resources']; print(p, r['scope'], 'n_samples', r['n_samples'], 'window_s', round(r['window_s'], 1))
r = d['coordinator_resources']; print('coordinator', r['scope'], 'n_samples', r['n_samples'])
P
bash scripts/stop_local.sh
```

Expected: `n_samples` is about the session length in seconds plus 2, not the process uptime.

## 5. Files changed

| File | Change |
|---|---|
| `vfl/telemetry.py` | `ResourceMonitor`: session window (`begin` / `session_summary` / `end`), `scope` on aggregates, bounded lifetime buffer, thread-safe `sample()` |
| `vfl/party_base.py` | window opened at START into `resources.jsonl`, closed in `_finish`; the session aggregate recorded for every outcome |
| `vfl/active.py`, `vfl/passive.py` | `session_summary()` in the run summary; `session` key on epoch rows |
| `vfl/coordinator.py` | coordinator window per session → `resources_coordinator.jsonl`, `coordinator_resources` in summary and session API; `session` key on epoch rows |
| `bench/analyze.py` | NFR-6 line includes the coordinator; lifetime aggregates labelled |
| `README.md`, `docs/PROJECT_DOCUMENTATION.md` | file table and NFR-6 row |
| `tests/test_resource_records.py` | new regression tests |
