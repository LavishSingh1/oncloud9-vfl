"""Issue #12: an unbounded inject_rtt_ms left a passive party stuck after STOP.

On rt-baseline (07ed305) `inject_rtt_ms` had no upper bound and the passive party emulated it with
time.sleep(rtt_half), which STOP cannot interrupt. With inject_rtt_ms=1e9 the passive slept ~5.8 days per half-RTT,
the coordinator closed the session as STOPPED after its 30 s grace, and every new session was refused with
"parties still busy". The fix bounds the schema (inject_rtt_ms <= 5000 ms, fault.delay_s <= 300 s) and makes both
emulated delays wait on the stop event, so STOP ends them at once.
"""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vfl import config_schema as schema
from vfl.config import load_session_config
from vfl.party_base import StopRequested
from vfl.passive import PassiveNode


def _param(key):
    return next(p for p in schema.PARAMS if p.key == key)


@pytest.mark.parametrize("key,ok,bad", [("inject_rtt_ms", [0, 100, 500, 5000], [5000.1, 1e6, 1e9]),
                                         ("fault.delay_s", [0, 2.0, 10.0, 300], [300.5, 1e9])])
def test_delay_keys_are_bounded(key, ok, bad):
    p = _param(key)
    for v in ok:
        assert schema.check_value(p, v) is None, (key, v)
    for v in bad:
        assert schema.check_value(p, v) is not None, (key, v)


@pytest.mark.parametrize("override", ["inject_rtt_ms=1000000000", "fault.delay_s=1000000000"])
def test_reported_values_are_refused_at_load(override):
    with pytest.raises(ValueError):
        load_session_config(None, [override])


def test_prd_rtt_sweep_still_accepted():
    """NFR-3 / E2 sweep 0, 20, 50, 100 ms; the documented presets go to 500 ms."""
    for rtt in (0, 20, 50, 100, 200, 500):
        assert load_session_config(None, [f"inject_rtt_ms={rtt}"])["inject_rtt_ms"] == rtt


def _fake_passive():
    """Just enough of a PassiveNode for _train_batch to reach the emulated uplink delay."""
    node = SimpleNamespace(stop_event=threading.Event(), malicious=None, session_id="s", party_id="passive-1",
                           counters={}, core=SimpleNamespace(forward=lambda idx: np.zeros((len(idx), 4), np.float32)))
    node._tamper = lambda h, idx, cfg: h
    node._header = lambda *a, **kw: PassiveNode._header(node, *a, **kw)
    node._call = lambda *a, **kw: pytest.fail("the exchange must not be sent once STOP is requested")
    return node


def test_stop_interrupts_the_emulated_rtt():
    """Bypassing the schema (rtt_half = 500 000 s), STOP must still end the batch within a second."""
    node = _fake_passive()
    threading.Timer(0.2, node.stop_event.set).start()
    t0 = time.monotonic()
    with pytest.raises(StopRequested):
        PassiveNode._train_batch(node, stub=None, cfg={"codec": "fp32"}, epoch=0, b=0, idx=np.arange(8),
                                 rtt_half=1e9 / 2000.0, inject_duplicate=False)
    assert time.monotonic() - t0 < 2.0
