# Issue — NFR-8 grad-noise ε reported without its adjacency; NFR-2 "3-seed" label and single repeat

| | |
|---|---|
| Issue | "Privacy and reproducibility reporting gaps" #21 |
| Reported by | Order of Phoenix (baseline `rt-baseline`, commit `07ed305`; also on `main` @ `95d8a66`) |
| Severity / triage | `minor`, `confirmed` |
| Requirement | NFR-8 (privacy evaluation: adjacency and composition), NFR-2 (reproducibility reporting) |
| Component | `vfl/defense.py` (`privacy_report`), `bench/analyze.py` (`analyse_e1`, `analyse_def`), `bench/run_benchmarks.py` (`e1`, `--repeats`), `bench/ui.py`, `docs/PROJECT_DOCUMENTATION.md` |

## 1. Validation

We tried to refute each claim before fixing it. Every evidence line the reporter cites matches `main` @ `95d8a66`
(`vfl/defense.py:155, 160-163`; `docs/PROJECT_DOCUMENTATION.md:207, 255-256`; `bench/analyze.py:190, 194`;
`bench/run_benchmarks.py:345`). We recomputed the reported ε by hand: `z = σ/2 = 0.5`, RDP at α = 2 is 20,
ε = 20 + ln ½ − (ln 10⁻⁵ + ln 2) = 30.1266, which matches the reported value.

| # | Claim | PRD clause | Refutation tried | Verdict |
|---|---|---|---|---|
| A1 | grad-noise ε is reported without its adjacency definition | NFR-8: "report … (ε, δ) … the protected information, adjacency definition, clipping/noise parameters, composition accountant"; §2 DP: "state … neighboring-dataset definition … account for all relevant releases across batches and epochs; and report … sampling assumptions, (ε, δ)" | *"grad noise is labelled perturbation, so the DP items are not 'applicable'."* Fails: the block reports a numeric (ε, δ), and once a DP parameter is reported its adjacency applies. §2 lets noise be reported as perturbation only when sensitivity bounds or composition accounting are *missing*; here both exist in the code and were not written into the report. *"The docs state it."* Fails: §6.3 defines neighbouring datasets for randomized response only. For grad noise the assumption appears only in a source comment. | **Valid** |
| A2 | sensitivity and composition are not reported | §2 DP: "valid sensitivity bounds and composition accounting", "sampling assumptions"; NFR-8 "composition accountant" | NFR-8 does not list sensitivity by name. But §2 makes it a condition of the ε, and NFR-8 asks for the composition accountant, which was reported only as `rdp_order`. | **Valid** (sensitivity via §2) |
| A3 | the docs say adjacency is reported when it is not | NFR-8 | none: `PROJECT_DOCUMENTATION.md:207` lists adjacency as a reported output and no report contains it | **Valid** |
| — | (not claimed) protected information, adversary and noise-insertion point | NFR-8 | these were already reported, in `finals.active.defense` (`defense.describe`) and in the NFR-8 benchmark line | already met |
| B1 | "3-seed" is hard-coded | NFR-2 (and honest reporting of evidence) | *"Avoidable: with the default `--seeds 1,2,3` the label is right."* True, but `--seeds` is a supported flag and any other value mislabels the evidence | **Valid** |
| B2 | NFR-2 passes on one same-seed repeat instead of three repetitions | NFR-2: "The same seed/configuration yields identical aligned order and committed-step count; report final-loss variation across three repetitions and investigate variation above 5%" | *"'Three repetitions' means three seeds, and the default run already reports a 3-seed CV."* Fails twice. (1) The sentence's subject is the same seed/configuration, and the PRD writes "three seeds" where it means seeds (NFR-1, §7.3), so "repetitions" is deliberate; our own docs already read NFR-2 as a "same-seed repeat". (2) Even on the three-seeds reading the old check is wrong: it never compared the across-seed CV with 5%, so the reporter's 9.02% CV still passed. No flag could run more than one repeat, so the gap could not be avoided by configuration either. | **Valid** |

Neither part changes a number that was already reported. Both are reporting defects, so we agree with `minor`.

**Part A.** The `grad_channel` block had `epsilon`, `delta` and `rdp_order` but not the assumptions behind them, so
a reader could not tell what the ε protects or re-derive it. The docs also said adjacency was reported.

**Part B.** `bench/analyze.py` always printed "3-seed final-loss CV" whatever `--seeds` was, and NFR-2 passed on
one same-seed repeat (two runs).

## 2. The fix

### Part A: every reported ε carries its assumptions

`privacy_report` now writes these fields into `grad_channel` (the existing fields are unchanged):

| Field | Value for the reporter's run | Meaning |
|---|---|---|
| `protected_information` | each aligned entity's training label | what the ε protects |
| `adjacency` | replace-one label adjacency | neighbouring training sets differ in one aligned entity's label |
| `sensitivity` | `{"norm": "L2", "value": 0.1, "formula": "2C", "derivation": …}` | one label can move its clipped row from `g` to `g'` (both ≤ C) |
| `noise_std`, `effective_noise_multiplier` | 0.05, 0.5 | `σC`, and `z = σC / 2C = σ/2` |
| `composition` | `releases_per_label: 5`, `…_if_passives_collude`, `rule`, `accountant` | once per label per epoch; × n_passive if passives collude; no subsampling amplification; RDP + Balle et al. 2020 |

`label_dp` (randomized response) gets `protected_information`, `adjacency`, `sensitivity: null` and a `composition`
saying the labels are randomized once per session and everything after that is post-processing.

We checked the "once per label per epoch" claim against the code before writing it into the report.
`data.epoch_batches` splits a full permutation of the training rows into batches and drops none. `core.py` draws each
passive party's noise separately, so colluding passive parties see `E × n_passive` releases.

`bench/analyze.py` puts the same assumptions into the NFR-8 line and a new table, **"NFR-8 privacy accounting:
adjacency, sensitivity, composition"**: one row per reported ε (grad-noise and randomized response). Results recorded
before this fix lack the new fields, so for those the block is re-derived from the session's own config with the
same accountant, and old benchmark folders still report the assumptions.

`docs/PROJECT_DOCUMENTATION.md` §6.3 now defines adjacency, sensitivity and composition for gradient noise. Before,
it defined them only for randomized response. It also has a worked example that reproduces ε ≈ 30.13. The §5.2
NFR-8 row says where these fields are reported.

### Part B: three repetitions, and the real seed count

- `bench/run_benchmarks.py --repeats N` (default **3**) runs the first seed and configuration N times in total,
  the original included. `--repeats 1` turns NFR-2 off ("not attempted"). The launcher in `bench/ui.py` offers the
  flag.
- NFR-2 now checks **all** repetitions: identical aligned order, identical committed steps, and final-loss spread
  ((max − min) / mean) ≤ 5%. The CV across repetitions is reported as well.
- Status: `pass` needs ≥ 3 repetitions. Fewer gives `partial`, with "NFR-2 asks for 3 repetitions, only 2 ran"
  (this is what an `rt-baseline` results folder now shows). A spread above 5% gives `investigate`, as the PRD asks.
- The across-seed CV is labelled with the number of distinct seeds that ran ("2-seed final-loss CV …"). With a single
  seed it says "across-seed final-loss CV n/a (1 seed)". Repeats are not counted as seeds.

## 3. Tests

`tests/test_privacy_repro_reporting.py`: 12 tests. 11 fail on `rt-baseline`. The runner test needs `torch` and
`grpc` and is skipped without them.

| Test | Checks |
|---|---|
| `test_grad_channel_reports_adjacency_sensitivity_and_composition` | reporter's config: ε unchanged (30.12663…), adjacency, 2C = 0.1, σC = 0.05, 5 releases |
| `test_epsilon_can_be_rederived_from_the_reported_fields_alone` | ε recomputed by hand from `noise_std`, `sensitivity`, `composition`, `rdp_order`, `delta` only, three configs incl. colluding passives |
| `test_label_dp_block_states_adjacency_and_single_release` | randomized response: adjacency, one release, post-processing |
| `test_privacy_block_is_json_serialisable` | the block still goes into `summary.json` |
| `test_docs_define_adjacency_for_grad_noise_not_only_randomized_response` | §6.3 grad-noise bullet states adjacency, 2C, composition, worked ε |
| `test_nfr8_line_and_table_carry_adjacency_sensitivity_and_composition` | benchmark NFR-8 line and accounting table |
| `test_nfr8_rederives_assumptions_for_results_recorded_before_the_fix` | an old-format privacy block is re-derived from the run's config |
| `test_nfr2_label_uses_the_number_of_seeds_actually_run` | 2 seeds → "2-seed", never "3-seed"; 1 seed → "n/a (1 seed)" |
| `test_nfr2_single_same_seed_repeat_is_not_reported_as_pass` | the old layout (original + 1 repeat) → `partial` |
| `test_nfr2_three_repetitions_pass_and_every_repetition_counts` | 3 repetitions → `pass`; only the 3rd drifting by 10% → `investigate` (the old code compared runs 1 and 2 only) |
| `test_run_benchmarks_defaults_to_three_repetitions` | `--repeats` defaults to 3 |
| `test_e1_runs_the_first_seed_three_times` | E1 schedules seeds 1, 2, 1, 1 for `--seeds 1,2` |

`tests/bench_fixture.py` takes a `repeats` argument. The default of 2 keeps the old layout, so the existing
analytics tests are unchanged.

## 4. How to verify

```bash
pytest -q tests/test_privacy_repro_reporting.py tests/test_defense.py tests/test_defence_randomness.py tests/test_analytics.py
```

Then repeat the reporter's steps:

```bash
DATA=data bash scripts/run_local.sh
python -m vfl.cli start --set epochs=5 --set defense.type=grad_noise --set defense.clip=0.05 --set defense.noise_multiplier=1.0 --wait
python - <<'P'
import json, glob
d = sorted(glob.glob('runs/s2*/'))[-1]
gc = json.load(open(d + 'summary.json'))['finals']['active']['privacy']['grad_channel']
print(gc['epsilon'], gc['adjacency'], gc['sensitivity']['value'], gc['composition']['releases_per_label'])
P
# expected: 30.12663110385034, replace-one label adjacency ..., 0.1, 5

python bench/run_benchmarks.py --out results/bench-quick --seeds 1,2 --epochs 5 --rtts 0,50 --batch-sizes 32,128 --noise 1
python bench/analyze.py results/bench-quick
grep -E "NFR-2|NFR-8" results/bench-quick/report/report.md
# NFR-2: "3 same-seed repetitions (seed 1) ... 2-seed final-loss CV ..."; NFR-8: "Accounting: ... adjacency ... 2C=0.1000 ..."
bash scripts/stop_local.sh
```

The quick benchmark now runs one more E1 session (the third same-seed repetition), which adds a few seconds.

## 5. Files changed

| File | Change |
|---|---|
| `vfl/defense.py` | `privacy_report`: adjacency, protected information, sensitivity, noise std, composition and accountant for `grad_channel`; adjacency and composition for `label_dp` |
| `bench/analyze.py` | NFR-8 accounting text and table (re-derived for old results); NFR-2 over all repetitions with `partial`/`investigate`, real seed count in the label |
| `bench/run_benchmarks.py` | `--repeats` (default 3); E1 schedules the extra repetitions |
| `bench/ui.py` | launcher field for `--repeats` |
| `docs/PROJECT_DOCUMENTATION.md` | §5.2 NFR-2 and NFR-8 rows; §6.3 gradient-noise adjacency, sensitivity, composition and a worked example |
| `README.md` | E1 description: three same-seed repetitions |
| `tests/bench_fixture.py` | `repeats` argument |
| `tests/test_privacy_repro_reporting.py` | new regression tests |
