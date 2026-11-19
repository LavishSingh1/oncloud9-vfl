# Issue #11: oversized model sizes in the session config exhaust party memory or segfault a party

| | |
|---|---|
| Issue | #11 "A single oversized number in the session config makes a party exhaust memory, segfault, or fail with an unhelpful error" |
| Reported by | Order of Phoenix (baseline `rt-baseline`, commit `07ed305`) |
| Severity / triage | `major`, `confirmed` |
| Requirement | FR-9 (model dimensions configurable without source edits), red-team rubric: crash or outage from an extreme input. The cited NFR-10 only loosely applies (see 1) |
| Component | `vfl/config_schema.py`, new `vfl/model_budget.py`, `vfl/models.py` (`configure_torch`), `vfl/core.py`, `vfl/coordinator.py` (`check_data_consistency`) |

## 1. Validation

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| 1 | `hidden_dim`, `embed_dim`, `num_classes`, `top_hidden`, `cnn_channels`, `gru_hidden`, `torch_threads` have a minimum but no maximum | `config_schema.py` on `rt-baseline`: `min=` only (for example `:306` for `torch_threads`). `docs/CONFIG.md` listed only `int (>= 1)` | **Valid** |
| 2 | `hidden_dim=10^9` OOM-kills both parties | each party's MLP bottom is `Linear(n_features, 10^9)`: 20 × 10^9 floats = 80 GB at a 20-feature passive | **Valid** |
| 3 | `num_classes=10^9` OOM-kills only the active party | only the active party builds the top model, `Linear(top_hidden, 10^9)` = 32 × 10^9 floats. `check_data_consistency` refused only a value *below* the label count | **Valid** |
| 4 | `embed_dim=10^7` fails with `EncodeError` and no mention of `embed_dim` | passive bottom `Linear(64, 10^7)` = 2.6 GB, active bottom + top about 5.2 GB; a `[64 × 10^7]` fp32 tensor is 2.4 GiB, above protobuf's 2 GiB and gRPC's 256 MiB message limits | **Valid** |
| 5 | `torch_threads=10^6` segfaults | `models.py:42` passed `max(1, int(threads))` straight to `torch.set_num_threads`; the reporter's one-line repro segfaults with exit 139 at the first multi-threaded op | **Valid** |
| 6 | STOP does not work during model construction | the coordinator closed the session after its 30 s stop grace while the parties were still inside one PyTorch allocation; Python cannot interrupt that call | **Valid, by consequence of 2** |

**Refutations considered:**
- *"The PRD sets no bound and NFR-10 is about duplicates, delays and checkpoints."* True. The cited NFR-10 does not
  really apply. But FR-9 (Must) makes model dimensions operator settings, and a value that validation accepts should
  not kill a party. This is the same reasoning the team accepted for `epochs` in issue #10 (same reporter, same
  class of defect). The rubric counts a crash or outage as Major in its own right.
- *"Only an authenticated operator can start a session."* That makes it a robustness defect, not a security one.
  A typo (extra zeros) by a legitimate operator triggers it, and the failure reason ("stopped heartbeating") does not
  point to the cause.
- *"Avoidable by not using huge values."* Nothing documented a limit.

We agree with **Major**.

## 2. The fix

| Expected outcome (from the issue) | Now |
|---|---|
| each size key and `torch_threads` has a schema max; out-of-range values get HTTP 400 naming the field and limit | `hidden_dim`, `top_hidden`, `text_hidden` <= 8192; `embed_dim` <= 4096; `gru_hidden` <= 4096; each `cnn_channels` entry <= 1024; `cnn_kernel`, `cnn1d_kernel` <= 31; `num_classes` <= 1000; `torch_threads` <= 256 (constants at the top of `config_schema.py`). Refused by `--set`, the form, `/api/config/validate` and `POST /api/sessions` (HTTP 400, for example `invalid session config: embed_dim must be <= 4096`). `docs/CONFIG.md` and the README table show the bounds. The largest preset is far below every bound (width 256, embed 128, 100 classes, 8 threads) |
| `torch_threads` clamped to the CPU count or capped; no segfault | `models.thread_count` caps the value at the CPUs the process may use (`sched_getaffinity`) and at 256, logs a warning when it caps, and the summary records the count actually used |
| no party can be driven to an unbounded allocation through the API | per-key bounds make every allocation finite. Bounded values can still combine into a model too big for a party's data (an 8192-wide layer is 6.5 MB over 200 features and 6.5 GB over 200,000). `vfl/model_budget.py` estimates each party's parameters (limit 50 M), one batch's activations (limit 2^28 values) and the `[batch × embed_dim]` wire tensor (gRPC message limit), and refuses the config naming the keys to reduce. It runs at the coordinator at session start, from each party's manifest (`rows`, `n_features`, `shape`), and again at each party just before it builds its model, from its real aligned data |
| make STOP work during model construction | construction is now bounded (about a second for the largest permitted model), and each party checks for STOP right after building its model. A single PyTorch allocation still cannot be interrupted mid-call; the size check is what keeps that call short |

`MAX_GRPC_MSG` moved from `transport.py` to `limits.py`, so the torch-free size check and the transport use the same
value (`VFL_GRPC_MAX_MSG_MB`, default 256).

## 3. Tests

`tests/test_config_size_bounds.py`: 39 tests. The 24 schema and size-check tests need neither torch nor grpc. The 15
model, party and coordinator tests skip without them.

| Test | Checks |
|---|---|
| `test_reported_values_are_refused_with_field_and_limit` (×4) | the reporter's four values are refused, naming the field and the limit |
| `test_every_size_key_has_a_maximum_in_every_entry_point` (×9) | each bounded key: the maximum is valid, max + 1 is refused by `--set`, the form and the full-config check |
| `test_cnn_channels_entries_are_bounded`, `test_every_preset_and_default_is_within_the_maxima`, `test_limits_are_in_the_generated_docs` | list bound; no preset broken; docs regenerated |
| `test_combination_of_valid_values_too_big_for_the_data_is_refused`, `test_embed_dim_report_case_is_refused_before_the_parties_run`, `test_activation_bound`, `test_top_model_counts_at_the_active_party_only`, `test_missing_manifest_counts_skip_only_the_model_part`, `test_defaults_and_presets_fit_on_realistic_data` | size check: refuses what cannot fit, accepts the presets on realistic data; `test_partial_config_uses_schema_defaults`: a partial config (as `tests/test_partition.py` passes) gets schema defaults |
| `test_parameter_estimate_matches_pytorch` (×12) | the estimate equals `n_params(make_bottom(...))` and the top model for every bottom kind and option (torch) |
| `test_party_refuses_before_allocating` | `PassiveCore` raises before building a 1.6 G-parameter model (torch) |
| `test_torch_threads_capped_no_segfault` | the reporter's `10^6` runs a matmul in a subprocess and exits 0 (torch) |
| `test_coordinator_refuses_at_start_and_parties_stay_idle` | each reported value and an oversized combination are refused at `start_session`; no session is created; a normal start still works (grpc, torch) |

## 4. How to verify

```bash
pytest -q tests/test_config_size_bounds.py tests/test_config_schema.py tests/test_coordinator.py tests/test_models.py
bash /tmp/reset.sh data-synth
python -m vfl.cli start --set hidden_dim=1000000000 --set dataset_tag=synthetic   # 400: hidden_dim must be <= 8192
python -m vfl.cli start --set num_classes=1000000000 --set dataset_tag=synthetic  # 400: num_classes must be <= 1000
python -m vfl.cli start --set embed_dim=10000000 --set dataset_tag=synthetic      # 400: embed_dim must be <= 4096
python -m vfl.cli start --set torch_threads=1000000 --set dataset_tag=synthetic   # 400: torch_threads must be <= 256
bash /tmp/rss.sh; grep '^oom_kill ' /sys/fs/cgroup/memory.events                 # both parties alive, oom_kill unchanged
python -m vfl.cli start --set torch_threads=256 --set dataset_tag=synthetic --wait   # COMPLETED; threads capped to the CPU count
```

## 5. Files changed

| File | Change |
|---|---|
| `vfl/config_schema.py` | size and thread maxima; `int_list` entries respect `max`; docs table shows `entries <= N` |
| `vfl/model_budget.py` | new: torch-free size check (parameters, activations, wire tensor) |
| `vfl/coordinator.py` | `check_data_consistency` runs the size check for every party at session start |
| `vfl/core.py` | `ActiveCore` / `PassiveCore` run the size check before allocating |
| `vfl/models.py` | `available_cpus`, `thread_count`; `configure_torch` caps `torch_threads` |
| `vfl/active.py`, `vfl/passive.py` | STOP check right after the model is built |
| `vfl/limits.py`, `vfl/transport.py` | `MAX_GRPC_MSG` shared from `limits.py` |
| `docs/CONFIG.md`, `README.md` | regenerated parameter tables |
| `dev.md`, `CHANGELOG-v2.md`, `CONTRIBUTORS.md` | module notes, changelog entry, file ownership |
| `tests/test_config_size_bounds.py` | new regression tests |
