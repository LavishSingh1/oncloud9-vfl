"""Issue #7: randomized-response label DP overflowed for large epsilon.

On rt-baseline (07ed305) `defense.label_rr_epsilon` had no upper bound, so `config validate` accepted e.g. 1000.0,
and `randomized_response` computed math.exp(eps), which raises OverflowError above eps ~ 709. The active party then
failed while building its core. The fix computes the keep probability in a form that cannot overflow, bounds the
schema to (0, 100], and refuses non-finite or non-positive values with a clear message.
"""
import math

import numpy as np
import pytest

from vfl import config_schema as schema
from vfl import defense
from vfl.config import load_session_config


def _old_keep(eps: float, k: int) -> float:
    """The v2.3.1 formula, kept here to show the new one gives the same probabilities."""
    return math.exp(eps) / (math.exp(eps) + k - 1)


def _old_randomized_response(y, k, eps, seed):
    """The v2.3.1 randomized_response with secret=None, for bit-for-bit comparison."""
    import hashlib
    rng = np.random.default_rng(int(hashlib.sha256(f"rr|{seed}".encode()).hexdigest()[:12], 16))
    yi = np.asarray(y).astype(np.int64)
    flip = rng.random(len(yi)) >= _old_keep(eps, k)
    other = (yi + rng.integers(1, k, size=len(yi))) % k
    return np.where(flip, other, yi).astype(np.asarray(y).dtype)


def test_old_formula_really_overflowed():
    """The reported weakness is real: the old expression cannot be evaluated at eps = 1000."""
    with pytest.raises(OverflowError):
        _old_keep(1000.0, 2)


@pytest.mark.parametrize("eps", [710.0, 1000.0, 1e6])
def test_large_epsilon_no_longer_overflows(eps):
    """Called directly (bypassing the schema), a huge epsilon now means 'keep every label', not a crash."""
    y = np.random.default_rng(0).integers(0, 2, 10_000)
    out = defense.randomized_response(y, 2, {"label_rr_epsilon": eps}, seed=1)
    assert np.array_equal(out, y)
    assert defense.rr_keep_probability(eps, 10) == 1.0


@pytest.mark.parametrize("k", [2, 3, 10, 100])
@pytest.mark.parametrize("eps", [1e-6, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 40.0, 100.0])
def test_keep_probability_equals_the_closed_form(eps, k):
    assert defense.rr_keep_probability(eps, k) == pytest.approx(_old_keep(eps, k), rel=1e-12, abs=0)


@pytest.mark.parametrize("k", [2, 10])
@pytest.mark.parametrize("eps", [0.5, 1.0, 2.0, 3.0, 4.0, 8.0])
def test_existing_runs_flip_exactly_the_same_labels(eps, k):
    """NFR-2: for every preset epsilon, the new code flips the same labels as v2.3.1 (same seed)."""
    y = np.random.default_rng(3).integers(0, k, 200_000)
    new = defense.randomized_response(y, k, {"label_rr_epsilon": eps}, seed=7)
    assert np.array_equal(new, _old_randomized_response(y, k, eps, seed=7))


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_randomized_response_refuses_non_finite_or_non_positive(bad):
    with pytest.raises(ValueError, match="label_rr_epsilon"):
        defense.randomized_response(np.zeros(4, int), 2, {"label_rr_epsilon": bad}, seed=0)


@pytest.mark.parametrize("bad", ["1000.0", "100.5", "inf", "0", "-1"])
def test_schema_refuses_out_of_range_epsilon(bad):
    """The reported reproduction step 2: `config validate` must now reject the value before any session starts."""
    with pytest.raises(ValueError, match="label_rr_epsilon"):
        load_session_config(None, [f"defense.label_rr_epsilon={bad}"])


@pytest.mark.parametrize("ok", ["0.5", "8.0", "100"])
def test_schema_still_accepts_valid_epsilon(ok):
    cfg = load_session_config(None, [f"defense.label_rr_epsilon={ok}"])
    assert cfg["defense"]["label_rr_epsilon"] == float(ok)


def test_schema_records_the_bound():
    p = next(p for p in schema.PARAMS if p.key == "defense.label_rr_epsilon")
    assert p.min == 0 and p.min_exclusive and p.max == 100.0 and p.nullable


# ------------------------------------------------------------------ same class of defect, every float key
FLOAT_KEYS = [p.key for p in schema.PARAMS if p.type == "float" and not p.internal]


@pytest.mark.parametrize("key", FLOAT_KEYS)
@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan, "inf", "nan"])
def test_every_float_key_refuses_non_finite_values(key, bad):
    """rt-baseline accepted inf for every float key without an upper bound (lr, every timeout, defence clip/noise);
    an infinite timeout then crashed threading waits with OverflowError. Strings cover YAML ('inf', quoted 'nan')."""
    p = next(p for p in schema.PARAMS if p.key == key)
    assert schema.check_value(p, bad) is not None


@pytest.mark.parametrize("override", ["lr=inf", "rpc_deadline_s=inf", "barrier_timeout_s=inf",
                                      "defense.noise_multiplier=inf", "fault.delay_s=nan"])
def test_non_finite_overrides_are_refused_at_load(override):
    with pytest.raises(ValueError, match="finite"):
        load_session_config(None, [override])


def test_non_finite_values_in_yaml_are_refused(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text("session:\n  lr: .inf\n")
    with pytest.raises(ValueError, match="lr"):
        load_session_config(str(f))
    f.write_text("session:\n  slot_timeout_s: 'nan'\n")
    with pytest.raises(ValueError, match="slot_timeout_s"):
        load_session_config(str(f))


def test_finite_float_values_still_pass():
    p = next(p for p in schema.PARAMS if p.key == "lr")
    assert schema.check_value(p, 0.003) is None and schema.check_value(p, "1e-3") is None
