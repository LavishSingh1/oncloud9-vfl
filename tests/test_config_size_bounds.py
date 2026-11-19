"""Red-team issue (Order of Phoenix, NFR-10 / robustness): one oversized number in the session config made the parties
allocate until the kernel OOM-killed them (hidden_dim, num_classes, embed_dim), or segfaulted PyTorch (torch_threads),
and the session failed with only "party stopped heartbeating". v2.3.1's schema set a minimum but no maximum.

Fix: a schema maximum for every model-size key and torch_threads (HTTP 400 naming the field and the limit), a size
check over the parties' real input widths (vfl/model_budget.py) at the coordinator and again at each party before it
allocates, and torch_threads capped at the CPU count.

The schema and size-check tests need neither torch nor grpc; the model and coordinator tests skip without them.
"""
import os
import subprocess
import sys

import pytest

from vfl import config_schema as cs
from vfl import model_budget as mb
from vfl.config import load_session_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOUNDED = {"hidden_dim": cs.MAX_WIDTH, "top_hidden": cs.MAX_WIDTH, "text_hidden": cs.MAX_WIDTH,
           "embed_dim": cs.MAX_EMBED, "gru_hidden": cs.MAX_RNN_HIDDEN, "num_classes": cs.MAX_CLASSES,
           "cnn_kernel": cs.MAX_KERNEL, "cnn1d_kernel": cs.MAX_KERNEL, "torch_threads": cs.MAX_TORCH_THREADS}


# ------------------------------------------------------------------ schema maxima (HTTP 400 at start)
@pytest.mark.parametrize("value", ["hidden_dim=1000000000", "num_classes=1000000000", "embed_dim=10000000",
                                   "torch_threads=1000000"])
def test_reported_values_are_refused_with_field_and_limit(value):
    key = value.split("=")[0]
    with pytest.raises(ValueError, match=rf"{key} must be <= {BOUNDED[key]}"):
        load_session_config(None, [value, "dataset_tag=synthetic"])


@pytest.mark.parametrize("key,limit", sorted(BOUNDED.items()))
def test_every_size_key_has_a_maximum_in_every_entry_point(key, limit):
    over = limit + 1 if key not in ("cnn_kernel", "cnn1d_kernel") else limit + 2        # kernels must be odd
    assert cs.check_value(cs.BY_KEY[key], limit) is None
    assert cs.validate_overrides({key: over}) == [f"{key} must be <= {limit}"]          # --set and the API
    with pytest.raises(ValueError, match=f"{key} must be <= {limit}"):
        cs.config_from_form({key: over})                                                # dashboard form
    with pytest.raises(ValueError, match=f"{key} must be <= {limit}"):                  # full config (coordinator)
        from vfl.config import validate_session_config
        validate_session_config({**load_session_config(None), key: over})


def test_cnn_channels_entries_are_bounded():
    p = cs.BY_KEY["cnn_channels"]
    assert cs.check_value(p, [16, cs.MAX_CHANNELS]) is None
    assert cs.check_value(p, [16, cs.MAX_CHANNELS + 1]) == f"entries must be <= {cs.MAX_CHANNELS}"
    assert cs.check_value(p, [16, 0]) == "must be a list of 1-6 positive integers"


def test_every_preset_and_default_is_within_the_maxima():
    for p in cs.PARAMS:
        for v in cs.option_values(p):
            assert cs.check_value(p, v) is None, (p.key, v)


def test_limits_are_in_the_generated_docs():
    md = cs.render_markdown()
    assert f"(>= 0, <= {cs.MAX_WIDTH})" in md and f"entries <= {cs.MAX_CHANNELS}" in md
    with open(os.path.join(ROOT, "docs", "CONFIG.md")) as fh:
        assert fh.read().strip() == md.strip(), "docs/CONFIG.md is stale: run `python -m vfl.cli config docs`"


# ------------------------------------------------------------------ size check over the parties' real inputs
DEF = load_session_config(None)


def test_defaults_and_presets_fit_on_realistic_data():
    mb.check_party("active", "mlp", DEF, n_features=40, rows=20_000, role="active", n_passive=3, num_classes=10)
    big = {**DEF, "batch_size": 1024, "eval_batch_size": 2048, "cnn_channels": [32, 64, 128], "embed_dim": 128}
    mb.check_party("passive-1", "cnn", big, n_features=784, shape=[1, 28, 28], rows=60_000)
    mb.check_party("passive-1", "gru", {**big, "gru_hidden": 128, "gru_layers": 2}, n_features=9 * 128,
                   shape=[9, 128], rows=10_000)


def test_combination_of_valid_values_too_big_for_the_data_is_refused():
    cfg = {**DEF, "hidden_dim": cs.MAX_WIDTH}
    mb.check_party("passive-1", "mlp", cfg, n_features=200, rows=20_000)                 # 1.8 M params: fine
    with pytest.raises(ValueError, match=r"too large for passive-1: .*parameters over its 200000 input features"
                                         r".*reduce hidden_dim, embed_dim"):
        mb.check_party("passive-1", "mlp", cfg, n_features=200_000, rows=20_000)        # 1.6 G params


def test_embed_dim_report_case_is_refused_before_the_parties_run():
    """The reported embed_dim=10^7 failed after ~8 s with 'EncodeError: Failed to serialize proto'. The schema now
    stops it; the largest legal embed_dim with a batch too big for one gRPC message is refused by name."""
    cfg = {**DEF, "embed_dim": cs.MAX_EMBED, "eval_batch_size": 65_536}
    with pytest.raises(ValueError, match=r"\[65536 x 4096\] tensor .* message limit; reduce embed_dim"):
        mb.check_party("passive-1", "mlp", cfg, n_features=20, rows=100_000)
    mb.check_party("passive-1", "mlp", cfg, n_features=20, rows=5_000)    # a batch never exceeds the party's rows


def test_activation_bound():
    cfg = {**DEF, "batch_size": 4096, "cnn_channels": [cs.MAX_CHANNELS] * 3}
    with pytest.raises(ValueError, match="activation values .* reduce batch_size, eval_batch_size or cnn_channels"):
        mb.check_party("passive-1", "cnn", cfg, n_features=3 * 64 * 64, shape=[3, 64, 64], rows=50_000)


def test_top_model_counts_at_the_active_party_only():
    cfg = {**DEF, "top_hidden": cs.MAX_WIDTH, "embed_dim": cs.MAX_EMBED}
    mb.check_party("passive-1", "linear", cfg, n_features=10, rows=1000)
    with pytest.raises(ValueError, match="and the top model .* reduce embed_dim, top_hidden, num_classes"):
        mb.check_party("active", "linear", cfg, n_features=10, rows=1000, role="active", n_passive=3)


def test_missing_manifest_counts_skip_only_the_model_part():
    mb.check_party("p", "mlp", {**DEF, "hidden_dim": cs.MAX_WIDTH}, n_features=None)
    with pytest.raises(ValueError, match="message limit"):
        mb.check_party("p", "mlp", {**DEF, "embed_dim": cs.MAX_EMBED, "batch_size": 100_000}, n_features=None)


def test_partial_config_uses_schema_defaults():
    """check_data_consistency is also called with partial configs (tests/test_partition.py passes {})."""
    mb.check_party("passive-1", "mlp", {}, n_features=40, rows=100)
    with pytest.raises(ValueError, match="reduce hidden_dim"):
        mb.check_party("passive-1", "mlp", {"hidden_dim": cs.MAX_WIDTH}, n_features=200_000, rows=100)


def test_model_budget_is_torch_free():
    code = ("import sys; import vfl.model_budget, vfl.config_schema; "
            "assert 'torch' not in sys.modules and 'grpc' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


# ------------------------------------------------------------------ against the real models (torch)
SHAPED_CASES = [
    ("mlp", 30, None, {}), ("mlp", 30, None, {"hidden_dim": 0}), ("linear", 30, None, {}),
    ("text", 384, None, {"text_hidden": 64}), ("text", 384, None, {}),
    ("cnn", 3 * 8 * 8, [3, 8, 8], {}), ("cnn", 3 * 6 * 6, [3, 6, 6], {"cnn_kernel": 7, "cnn_pool": "none",
                                                                      "cnn_channels": [8, 16, 32]}),
    ("cnn", 1 * 3 * 3, [1, 3, 3], {"cnn_pool": "auto"}), ("cnn", 2 * 8 * 8, [2, 8, 8], {"cnn_pool": "avg"}),
    ("cnn1d", 4 * 20, [4, 20], {"cnn1d_kernel": 3, "cnn_channels": [16]}),
    ("gru", 4 * 20, [4, 20], {}), ("gru", 4 * 20, [4, 20], {"gru_hidden": 16, "gru_layers": 3}),
]


@pytest.mark.parametrize("kind,in_dim,shape,over", SHAPED_CASES)
def test_parameter_estimate_matches_pytorch(kind, in_dim, shape, over):
    pytest.importorskip("torch")
    from vfl.models import TopModel, make_bottom, n_params, output_dim
    cfg = {**DEF, **over}
    assert mb.bottom_estimate(kind, in_dim, shape, cfg).params == n_params(make_bottom(kind, in_dim, shape, cfg))
    for k, n_passive in ((2, 1), (10, 3)):
        top = TopModel(cfg["embed_dim"] * (1 + n_passive), cfg["top_hidden"], output_dim(k))
        assert mb.top_estimate(cfg, n_passive, k).params == n_params(top)


def test_party_refuses_before_allocating():
    pytest.importorskip("torch")
    import numpy as np
    from vfl.core import PassiveCore
    X = np.zeros((16, 200_000), dtype=np.float32)
    with pytest.raises(ValueError, match="too large for passive-1"):         # 1.6 G params: would need ~26 GB
        PassiveCore({**DEF, "hidden_dim": cs.MAX_WIDTH}, X, "passive-1")


def test_torch_threads_capped_no_segfault():
    """`torch.set_num_threads(10**6)` segfaulted at the first multi-threaded op (exit 139). Run in a subprocess:
    a regression must not take pytest down with it, and the thread count is process-global."""
    pytest.importorskip("torch")
    from vfl.models import available_cpus, thread_count
    assert thread_count(10 ** 6) == min(available_cpus(), cs.MAX_TORCH_THREADS)
    assert thread_count(1) == 1 and thread_count(0) == 1
    code = ("import torch; from vfl.models import configure_torch, available_cpus; env = configure_torch(10**6); "
            "a = torch.randn(512, 512); print(float((a @ a).sum())); "
            "assert env['threads'] == torch.get_num_threads() <= available_cpus()")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (r.returncode, r.stderr[-500:])


# ------------------------------------------------------------------ coordinator: refused at start (HTTP 400)
@pytest.fixture
def state(tmp_path):
    pytest.importorskip("grpc")
    pytest.importorskip("torch")
    from vfl.coordinator import CoordinatorState
    from vfl.security import TLSFiles
    return CoordinatorState(str(tmp_path), load_session_config(None), TLSFiles(None, "coordinator"))


def _reg(state, pid, role, summary, address=""):
    import json
    from vfl.proto import vfl_pb2 as pb
    ok, why = state.register(pb.RegisterRequest(party_id=pid, role=role, address=address,
                                                summary_json=json.dumps(summary)))
    assert ok, why


def test_coordinator_refuses_at_start_and_parties_stay_idle(state):
    _reg(state, "active", "active", {"rows": 20_000, "n_features": 20, "has_labels": True}, address="h:1")
    _reg(state, "passive-1", "passive", {"rows": 20_000, "n_features": 200_000, "has_labels": False})
    for over, msg in (({"hidden_dim": 10 ** 9}, "hidden_dim must be <= 8192"),
                      ({"num_classes": 10 ** 9}, "num_classes must be <= 1000"),
                      ({"embed_dim": 10 ** 7}, "embed_dim must be <= 4096"),
                      ({"torch_threads": 10 ** 6}, "torch_threads must be <= 256"),
                      ({"hidden_dim": 8192}, "too large for passive-1: .*reduce hidden_dim")):
        with pytest.raises(ValueError, match=msg):
            state.start_session({"config": {**over, "dataset_tag": "synthetic"}})
        assert state.current is None                       # nothing was sent to a party
    assert state.start_session({"config": {"hidden_dim": 64, "epochs": 1}})["participants"] == ["active", "passive-1"]
