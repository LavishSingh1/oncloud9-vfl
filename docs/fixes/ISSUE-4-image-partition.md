# Issue #4 — Image partitions that leave a party with no features

| | |
|---|---|
| Issue | [#4](https://github.com/LavishSingh1/oncloud9-vfl/issues/4) "Image partitioning allocates zero features to trailing parties when party count exceeds image dimension, crashing PyTorch Conv2d" |
| Reported by | Order of Phoenix (baseline `rt-baseline`, commit `07ed305`) |
| Severity / triage | `minor`, `confirmed` |
| Status | Fix in branch `fix/4-image-partition` (linked PR); `retest-requested` from the reporting red team |
| Component | `vfl/partition.py` (`_image_tiles`, `partition`), `vfl/datasets.py`, `vfl/data.py`, `vfl/models.py` |
| Owner | Amisha Sadhukhan|

## 1. Summary

The report is **valid**. `_image_tiles` split the image with `np.array_split` without checking the number of
strips or grid cells against the image size. With more parties than pixel columns (or grid rows than pixel rows)
the last party received an empty tile: its manifest said `model_input_dim: 0` and `shape: [1, 0, 0]`, and a CNN
bottom model then crashed on its first forward pass with a PyTorch error.

While reproducing it we found the same gap in two related places: an `image_grid` with more rows than the image
height, and a `manual` partition with an empty column list. The fix closes all of them, at four layers.

## 2. Reproduction on `rt-baseline`

| Command | Result on `rt-baseline` |
|---|---|
| `prepare-data --partitioner v2 --dataset digits --parties 9 --partition image_strips` | `passive-8` written with 0 inputs, shape `[1, 0, 0]` |
| `prepare-data --partitioner v2 --dataset digits --parties 9 --partition image_grid --grid 9x1` | `passive-8` written with 0 inputs |
| `prepare-data --partitioner v2 --dataset breast_cancer --parties 2 --partition manual --columns-json <file with an empty list>` | `passive-1` written with 0 inputs |
| `baseline --set bottom_model=cnn` on the first output | `RuntimeError: ... Kernel size can't be greater than actual input size` |

**Root cause.** `np.array_split(np.arange(8), 9)` returns eight one-element arrays and one empty array. The tabular
strategies already guarantee a non-empty party through `_fill_empty`, but the image and `manual` paths had no
equivalent check, and nothing downstream refused a zero-feature party.

## 3. The fix

| Layer | Change | Effect |
|---|---|---|
| Image tiling (`_image_tiles`) | Refuse `rows > h` or `cols > w` | Clear message: "a 8x8 image cannot be split into 1x9 tiles … (at most 8 parties)" |
| Every partition (`check_nonempty_groups`, called by `partition()` and `datasets.prepare_images`) | Refuse any party with no features, naming it | Covers `manual` and any future strategy; runs before any file is written |
| Loading (`PartyData.from_csv`, `from_npz`) | Refuse a party file with no feature columns | Data prepared by `rt-baseline` fails fast with a clear message |
| Models (`make_bottom`) | Refuse `in_dim < 1` | Clear `ValueError` instead of a PyTorch convolution error |

`prepare-data` with the reported command now exits with status 1, prints the error and writes nothing.

**Unchanged:** every valid partition. Thin tiles remain allowed. For example, 8 one-pixel-wide strips of an 8×8
image, or an 8×8 grid of single pixels, still work, and the CNN bottom model already handles tiles smaller than its
pooling window by disabling pooling.

## 4. Tests

`tests/test_partition_boundaries.py` (9 tests; 6 of them fail on `rt-baseline`):

| Test | Checks |
|---|---|
| `test_image_strips_more_parties_than_columns_is_refused` | the reported case (digits, 9 strips) is refused and nothing is written |
| `test_image_grid_more_rows_than_pixels_is_refused` | the grid variant (9×1) is refused |
| `test_manual_partition_with_an_empty_party_is_refused` | `manual` with an empty column list is refused, naming the party |
| `test_largest_valid_image_partitions_still_work` (3 cases) | 8 strips, an 8×8 grid and the default 2×2 grid still work, cover all 64 pixels, and leave no party empty |
| `test_check_nonempty_groups_names_the_party` | the error names every empty party |
| `test_party_file_without_features_is_refused_at_load` | zero-feature CSV and `.npz` party files are refused |
| `test_bottom_model_refuses_zero_inputs` | `make_bottom` refuses 0 inputs for MLP and CNN; a 1-pixel-wide CNN strip is still accepted |

The existing data-preparation, partition and model suites still pass.

## 5. How to verify

```bash
pytest -q tests/test_partition_boundaries.py
python -m vfl.cli prepare-data --partitioner v2 --dataset digits --parties 9 --partition image_strips --out /tmp/vfl_digits_9p
echo $?                      # 1, with the error message above; /tmp/vfl_digits_9p is not created
python -m vfl.cli prepare-data --partitioner v2 --dataset digits --parties 8 --partition image_strips --out /tmp/vfl_digits_8p
python -m vfl.cli baseline --data-dir /tmp/vfl_digits_8p --set bottom_model=cnn --set epochs=1    # trains
```

## 6. Files changed

| File | Change |
|---|---|
| `vfl/partition.py` | bounds check in `_image_tiles`; `check_nonempty_groups`; called in `partition()` |
| `vfl/datasets.py` | `prepare_images` calls `check_nonempty_groups` |
| `vfl/data.py` | CSV and `.npz` loaders refuse zero feature columns |
| `vfl/models.py` | `make_bottom` refuses zero inputs |
| `tests/test_partition_boundaries.py` | new regression tests |
| `CONTRIBUTORS.md` | ownership of the new test file |
| `docs/fixes/ISSUE-4-image-partition.md` | this document |
