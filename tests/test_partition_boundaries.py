"""Issue #4: no partition may give a party zero features.

On rt-baseline, `prepare-data --dataset digits --parties 9 --partition image_strips` split an 8-pixel-wide image
into 9 strips; np.array_split gave passive-8 an empty strip, the manifest said model_input_dim 0, and a CNN bottom
model crashed on its first forward pass. The same happened for an image_grid with more rows than pixels and for a
manual partition with an empty column list.
"""
import json

import numpy as np
import pandas as pd
import pytest

from vfl import partition as P
from vfl.data import PartyData


@pytest.fixture(scope="module")
def digits():
    return P.load_dataset("digits")                # 8x8 images


def test_image_strips_more_parties_than_columns_is_refused(digits, tmp_path):
    with pytest.raises(ValueError, match="cannot be split into 1x9 tiles"):
        P.partition(digits, 9, str(tmp_path), "image_strips")
    assert not any(tmp_path.iterdir())             # nothing written


def test_image_grid_more_rows_than_pixels_is_refused(digits, tmp_path):
    with pytest.raises(ValueError, match="cannot be split into 9x1 tiles"):
        P.partition(digits, 9, str(tmp_path), "image_grid", grid=(9, 1))


def test_manual_partition_with_an_empty_party_is_refused(tmp_path):
    ds = P.load_dataset("breast_cancer")
    with pytest.raises(ValueError, match="passive-1 with no features"):
        P.partition(ds, 2, str(tmp_path), "manual", columns=[["mean radius", "mean texture"], []])


@pytest.mark.parametrize("strategy,n,grid", [("image_strips", 8, None), ("image_grid", 64, (8, 8)),
                                             ("image_grid", 4, None)])
def test_largest_valid_image_partitions_still_work(digits, tmp_path, strategy, n, grid):
    m = P.partition(digits, n, str(tmp_path), strategy, grid=grid)
    dims = [p["model_input_dim"] for p in m["parties"]]
    assert min(dims) >= 1 and sum(dims) == 64          # every pixel assigned, every party non-empty


def test_check_nonempty_groups_names_the_party():
    with pytest.raises(ValueError, match="active, passive-2"):
        P.check_nonempty_groups([[], ["a"], []])
    P.check_nonempty_groups([["a"], ["b"]])


def test_party_file_without_features_is_refused_at_load(tmp_path):
    """Data prepared by rt-baseline may still contain a zero-feature party file: fail fast and clearly."""
    p = tmp_path / "passive-8.csv"
    pd.DataFrame({"id": ["a", "b"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="no feature columns"):
        PartyData.from_csv(str(p))
    q = tmp_path / "passive-8.npz"
    np.savez(q, ids=np.array(["a", "b"]), X=np.zeros((2, 0), np.float32))
    with pytest.raises(ValueError, match="no feature columns"):
        PartyData.from_npz(str(q))


def test_bottom_model_refuses_zero_inputs():
    pytest.importorskip("torch")
    from vfl.models import make_bottom
    cfg = {"embed_dim": 8, "hidden_dim": 16}
    for kind, shape in (("mlp", None), ("cnn", [1, 0, 0])):
        with pytest.raises(ValueError, match="at least one input feature"):
            make_bottom(kind, 0, shape, cfg)
    assert make_bottom("cnn", 8, [1, 8, 1], cfg) is not None   # 1-pixel-wide strips remain valid
