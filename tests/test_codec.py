"""v2 items 10, 34: codec round-trips, byte accounting and tensor validation."""
import numpy as np
import pytest  # noqa: F401

from vfl import codec


def test_roundtrip_and_byte_count():
    a = np.random.default_rng(0).normal(size=(32, 16)).astype(np.float32)
    for c in ("fp32", "fp16", "int8", "topk"):
        t = codec.encode(a, c, topk_fraction=0.5)
        assert codec.tensor_bytes(t) == codec.expected_tensor_bytes(32, 16, c, 0.5), c
        back = codec.decode(t)
        assert back.shape == a.shape
        if c == "fp32":
            assert np.array_equal(back, a)
        elif c == "fp16":
            assert np.allclose(back, a, atol=1e-2)
        elif c == "int8":
            assert np.abs(back - a).max() < np.abs(a).max() / 50
        else:
            assert (back != 0).sum(axis=1).max() <= 8      # kept k=8 of 16 per row


def test_int8_smaller_than_fp32():
    assert codec.expected_tensor_bytes(64, 32, "int8") < codec.expected_tensor_bytes(64, 32, "fp32")


def test_validation_shape_finite_norm():
    good = np.ones((4, 8), np.float32)
    codec.validate(good, (4, 8), max_row_norm=100)
    with pytest.raises(codec.TensorRejected, match="bad_shape"):
        codec.validate(good, (4, 7))
    bad = good.copy(); bad[0, 0] = np.nan
    with pytest.raises(codec.TensorRejected, match="non_finite"):
        codec.validate(bad, (4, 8))
    big = good * 100
    with pytest.raises(codec.TensorRejected, match="norm_bound"):
        codec.validate(big, (4, 8), max_row_norm=1.0)


def test_malformed_payload_rejected():
    t = codec.encode(np.zeros((2, 4), np.float32), "fp32")
    t.data = t.data[:-4]
    with pytest.raises(codec.TensorRejected):
        codec.decode(t)
