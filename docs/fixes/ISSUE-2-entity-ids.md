# Issue #2 — Entity identifier width and uniqueness

| | |
|---|---|
| Issue | [#2](https://github.com/LavishSingh1/oncloud9-vfl/issues/2) "Insecure and truncated 48-bit SHA-1 entity identifier generation risks cross-entity collision and data contamination" |
| Reported by | Order of Phoenix (baseline `rt-baseline`, commit `07ed305`) |
| Severity / triage | `major`, `confirmed` (silent data corruption under partial overlap) |
| Status | Fix in v2.3.2 (branch `fix/2-entity-ids`, linked PR); `retest-requested` from the reporting red team |
| Component | `vfl/partition.py` → `make_ids` (data preparation, FR-2 / FR-3) |
| Owner | M. Divya (data and partitioning) · reviewer: Lavish Singh |

## 1. Summary

The report is **valid**. In v2.3.1, `make_ids` built every hashed entity id as SHA-1 of `vfl-<seed>:<row>`,
truncated to 12 hex characters (48 bits). Two different rows can therefore receive the same id, and nothing
stopped such a duplicate from reaching the party files. We reproduced a real collision and showed that, under
partial overlap, it makes alignment join two different entities without any error.

The fix derives ids from the first **128 bits of SHA-256** and adds an **explicit uniqueness check** before any
party file is written, so a duplicate id can no longer reach training, whatever its cause.

## 2. The code before the fix

```python
def make_ids(n_rows: int, style: str, seed: int) -> np.ndarray:
    if style == "int":
        return np.arange(n_rows)
    salt = f"vfl-{seed}"
    return np.array([hashlib.sha1(f"{salt}:{i}".encode()).hexdigest()[:12] for i in range(n_rows)])
```

`make_ids` runs once, during `prepare-data` (`--partitioner v2`, and the image/sensor/multimodal generators in
`vfl/datasets.py`). The same id is written to every party file that holds that row; alignment later matches parties
on these ids.

## 3. Verification

### 3.1 A real collision exists

An exhaustive search of the old scheme for `seed = 0` found:

| Row | Old id |
|---|---|
| 18,125,834 | `a877eff6ba91` |
| 20,457,688 | `a877eff6ba91` |

Any dataset with more than 20,457,688 rows prepared with seed 0 would have contained a duplicate id. This pair is
now a regression test (`tests/test_entity_ids.py::test_old_scheme_really_collided`).

### 3.2 What a duplicate id does

We injected one duplicate into a 569-row dataset and ran the real pipeline (`partition` → `PartyData` →
`align_locally`):

| Situation | Result in v2.3.1 |
|---|---|
| Full overlap (`overlap = 1`): both colliding rows are in every party file | **Loud failure** — `PartyData` refuses the file: `duplicate entity ids` |
| Partial overlap: one colliding row only at the active party, the other only at a passive party | **Silent mismatch** — each file loads, alignment pairs the active party's row with a *different* entity's passive features, and training proceeds on the wrong join |

The second case is the data-contamination effect described in the report.

### 3.3 How likely it was

Probability of at least one collision, $p \approx 1 - e^{-n(n-1)/2^{b+1}}$:

| Dataset | Rows | Old (48-bit) | New (128-bit) |
|---|---:|---:|---:|
| breast_cancer | 569 | 5.7 × 10⁻¹⁰ | 4.7 × 10⁻³⁴ |
| digits | 1,797 | 5.7 × 10⁻⁹ | 4.7 × 10⁻³³ |
| synthetic (default) | 20,000 | 7.1 × 10⁻⁷ | 5.9 × 10⁻³¹ |
| adult | 48,842 | 4.2 × 10⁻⁶ | 3.5 × 10⁻³⁰ |
| mnist / fashion_mnist | 70,000 | 8.7 × 10⁻⁶ | 7.2 × 10⁻³⁰ |
| 1 million rows | 1,000,000 | 1.8 × 10⁻³ | 1.5 × 10⁻²⁷ |
| 10 million rows | 10,000,000 | 0.16 | 1.5 × 10⁻²⁵ |

For the datasets used in this project's experiments (≤ 70,000 rows) the old risk was very small, but it grew
quickly for large `csv` or `synthetic` inputs, and the failure mode (a silent wrong join) is serious enough that it
should not depend on luck.

### 3.4 Scope of the impact

* **Affected:** correctness of the alignment for hashed ids (`--id-style hash`, the default). `--id-style int` and
  the v1 generator (`E0000000`-style ids) were never affected.
* **Not affected:** the alignment protocols themselves. Experimental alignment hashes ids with full 256-bit salted
  SHA-256 (`alignment.salted_hash`) and DH-PSI maps them into a 2048-bit group; neither truncates.
* **Security relevance of SHA-1:** these ids are pseudonyms of row numbers created by the experimenter during data
  preparation; the input is a counter, not a secret or a real-world identifier, so no attacker chooses the inputs
  and SHA-1's known collision attacks do not apply. The actual defect was the 48-bit width together with the
  missing uniqueness check. SHA-256 is used now regardless, which also clears the `bandit B324` finding.

## 4. Root cause

1. **Width.** 48 bits gives a birthday bound of about 2²⁴ ≈ 16.7 million rows.
2. **No check.** Uniqueness was assumed rather than verified when ids were generated. The loader's duplicate check
   only sees one party's file, so it cannot catch two colliding rows that end up at different parties.

## 5. The fix

```python
ID_SCHEME = "sha256-128"
ID_HEX_CHARS = 32          # 128 bits

def entity_id(salt: str, row: int) -> str:
    return hashlib.sha256(f"{salt}:{row}".encode()).hexdigest()[:ID_HEX_CHARS]

def check_unique_ids(ids) -> None:
    ...  # raises ValueError("... duplicate id(s) ...") if any id repeats

def make_ids(n_rows: int, style: str, seed: int) -> np.ndarray:
    if style == "int":
        return np.arange(n_rows)
    if style != "hash":
        raise ValueError(f"unknown id style {style!r} (expected 'hash' or 'int')")
    ids = np.array([entity_id(f"vfl-{seed}", i) for i in range(n_rows)])
    check_unique_ids(ids)
    return ids
```

* **SHA-256, 128 bits.** Collisions become negligible at any realistic size (below 10⁻²⁰ even at 10⁹ rows), and
  32 hex characters keep the party CSVs readable.
* **Explicit uniqueness check.** The guarantee is enforced, not only made improbable. It runs before any file is
  written, over *all* rows, so it also covers the cross-party case the loader cannot see.
* **Strict style.** An unknown `id_style` now raises instead of silently falling back to hashing.
* **Traceability.** Every manifest records `params.id_scheme` (`sha256-128`, or `row-index` for `int`).

## 6. Tests

`tests/test_entity_ids.py` (8 tests; 6 of them fail on v2.3.1 and all pass after the fix):

| Test | Checks |
|---|---|
| `test_old_scheme_really_collided` | the reported weakness is real: rows 18,125,834 and 20,457,688 collided under the old scheme |
| `test_new_ids_do_not_collide_on_the_known_pair` | the same pair gets different ids now |
| `test_ids_are_128_bit_sha256_hex` | format is 32 lowercase hex chars and equals SHA-256(`vfl-<seed>:<row>`)[:32] |
| `test_ids_are_unique_and_stable_per_seed` | 200,000 ids are unique, identical across runs with the same seed (NFR-2), different for another seed |
| `test_duplicate_ids_are_refused` | `check_unique_ids` raises on a duplicate |
| `test_unknown_id_style_is_rejected` | unknown styles are refused |
| `test_collision_can_no_longer_reach_the_party_files` | a forced collision between an active-only and a passive-only row stops `partition()` before any file is written |
| `test_manifest_records_id_scheme_and_alignment_still_matches` | manifest records the scheme; aligned slots hold the same entity at every party |

The existing data-preparation, data and configuration-schema suites still pass.

## 7. Compatibility

* **Prepared data on disk** from v2.3.1 still loads: party files accept any unique string ids.
* **New preparations** with `--id-style hash` produce different id strings. Alignment orders entities by their
  salted hash, so the aligned order and the seeded train/validation/test split change. Results remain fully
  reproducible for a given seed (NFR-2), but they are not bit-identical to runs on data prepared by v2.3.1. Re-run
  `prepare-data` before comparing new results with old ones.
* No configuration key, CLI flag or wire-protocol message changed.

## 8. How to verify

```bash
pytest -q tests/test_entity_ids.py
python -m vfl.cli prepare-data --partitioner v2 --dataset breast_cancer --passives 1 --out /tmp/ids-check
python -c "import json; print(json.load(open('/tmp/ids-check/manifest.json'))['params']['id_scheme'])"   # sha256-128
head -3 /tmp/ids-check/active.csv                                                                           # 32-char ids
bandit -r vfl/partition.py -t B324                                                                          # no issues
```

## 9. Files changed

| File | Change |
|---|---|
| `vfl/partition.py` | `entity_id`, `check_unique_ids`, `ID_SCHEME`; `make_ids` uses them; manifest records `id_scheme` |
| `tests/test_entity_ids.py` | new regression tests |
| `CHANGELOG-v2.md` | v2.3.2 entry |
| `dev.md`, `CONTRIBUTORS.md` | module description and test ownership |
| `docs/fixes/ISSUE-2-entity-ids.md` | this document |
