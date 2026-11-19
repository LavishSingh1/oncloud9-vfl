"""Size check for the models a session config would build, before any party allocates them.

The per-key maxima in `vfl/config_schema.py` stop absurd single values (hidden_dim=10^9). They cannot stop a
combination of valid values that is still too big for a party: the size of a bottom model also depends on that
party's input width, which only its data manifest knows (a 4096-wide hidden layer is 3 MB over 200 features and
3 GB over 200,000). This module estimates, from the session config and each party's `rows` / `n_features` /
`shape`:

  * the parameter count of the party's bottom model (plus the top model at the active party);
  * the elements in one batch's layer outputs (training batch and evaluation batch);
  * the bytes of the [batch x embed_dim] tensor sent on the wire (representation and its gradient).

and refuses the config with a message that names the keys to reduce. It runs at the coordinator when a session
starts (HTTP 400, with the parties' manifests) and again at each party just before it builds its model, with its
real aligned data.

The limits are far above every configuration in the PRD experiments, the presets and the tests; they only refuse
configs that cannot run on a lab machine. The estimates use the same layer arithmetic as `vfl/models.py` (a test
checks the parameter counts against PyTorch). They are not a memory model: optimizer state, autograd buffers and
codec copies add a small multiple on top.

This module must not import torch or grpc: the coordinator's start check and the CLI use it without them.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config_schema import default_session
from .limits import MAX_GRPC_MSG

# ~0.8 GB of weights + gradients + Adam state (16 bytes per parameter). The largest preset model is < 1 M.
MAX_PARTY_PARAMS = 50_000_000
# elements in one batch's layer outputs: 1 GiB of float32 (= codec.MAX_ELEMENTS)
MAX_BATCH_ACTIVATIONS = 1 << 28
WIRE_HEADROOM = 0.9          # leave room for the protobuf envelope, ids and digests

SHAPED = {"cnn": 3, "cnn1d": 2, "gru": 2}
SIZE_KEYS = {"mlp": ("hidden_dim", "embed_dim"), "linear": ("embed_dim",), "text": ("text_hidden", "embed_dim"),
             "cnn": ("cnn_channels", "cnn_kernel", "embed_dim"), "cnn1d": ("cnn_channels", "cnn1d_kernel", "embed_dim"),
             "gru": ("gru_hidden", "gru_layers", "embed_dim"), "top": ("top_hidden", "num_classes", "embed_dim")}


@dataclass
class Estimate:
    params: int = 0
    act_per_row: int = 0        # layer-output elements per row (sum over layers): what training keeps
    widest_per_row: int = 0     # widest single layer output per row: the evaluation peak

    def add_layer(self, params: int, out_per_row: int) -> None:
        self.params += int(params)
        self.act_per_row += int(out_per_row)
        self.widest_per_row = max(self.widest_per_row, int(out_per_row))


def _linear(est: Estimate, n_in: int, n_out: int) -> None:
    est.add_layer(n_in * n_out + n_out, n_out)


def _dense(est: Estimate, n_in: int, hidden: int, n_out: int) -> None:
    """BottomModel / TopModel: Linear-ReLU-Linear, or one Linear when hidden == 0."""
    if hidden > 0:
        _linear(est, n_in, hidden)
        est.add_layer(0, hidden)                     # ReLU output
        _linear(est, hidden, n_out)
    else:
        _linear(est, n_in, n_out)


def _channels(cfg: dict) -> list[int]:
    return [int(c) for c in (cfg.get("cnn_channels") or (16, 32))]


def bottom_estimate(kind: str, in_dim: int, shape, cfg: dict) -> Estimate:
    """Parameters and activations of `vfl.models.make_bottom(kind, in_dim, shape, cfg)` (same arithmetic)."""
    E, est = int(cfg["embed_dim"]), Estimate()
    in_dim = int(in_dim)
    if kind == "mlp":
        _dense(est, in_dim, int(cfg["hidden_dim"]), E)
    elif kind == "linear":
        _dense(est, in_dim, 0, E)
    elif kind == "text":
        _dense(est, in_dim, int(cfg.get("text_hidden") or 0), E)
    elif kind == "cnn":
        c, h, w = (int(v) for v in shape)
        k, cin = int(cfg.get("cnn_kernel") or 3), c
        for co in _channels(cfg):                    # 'same' padding: every conv keeps h x w
            est.add_layer(cin * co * k * k + co, co * h * w)
            cin = co
        pool = cfg.get("cnn_pool") or "auto"
        if pool == "auto":
            pool = "max" if min(h, w) >= 4 else "none"
        if pool in ("max", "avg") and min(h, w) >= 2:
            h, w = h // 2, w // 2
            est.add_layer(0, cin * h * w)
        _linear(est, cin * h * w, E)
    elif kind == "cnn1d":
        c, t = (int(v) for v in shape)
        k, cin = int(cfg.get("cnn1d_kernel") or 5), c
        for co in _channels(cfg):
            est.add_layer(cin * co * k + co, co * t)
            cin = co
        _linear(est, cin, E)
    elif kind == "gru":
        c, t = (int(v) for v in shape)
        hid, layers = int(cfg.get("gru_hidden") or 32), int(cfg.get("gru_layers") or 1)
        for i in range(layers):                      # torch GRU: 3 gates, weight_ih + weight_hh + 2 biases
            n_in = c if i == 0 else hid
            est.add_layer(3 * hid * (n_in + hid + 2), 4 * hid * t)    # output sequence + gate activations
        _linear(est, hid, E)
    else:
        raise ValueError(f"unknown bottom_model {kind!r}")
    return est


def top_estimate(cfg: dict, n_passive: int, num_classes: int) -> Estimate:
    """`TopModel(embed_dim * (1 + n_passive), top_hidden, output_dim(num_classes))`."""
    est = Estimate()
    out = 1 if int(num_classes) <= 2 else int(num_classes)
    _dense(est, int(cfg["embed_dim"]) * (1 + int(n_passive)), int(cfg["top_hidden"]), out)
    return est


def wire_bytes_per_row(cfg: dict) -> float:
    """Upper bound on the encoded bytes of one row of a [batch x embed_dim] tensor."""
    per_elem = 4.0                                    # fp32; fp16 and int8 are smaller
    if cfg.get("codec") == "topk":                   # int32 index + fp32 value per kept entry
        per_elem = max(per_elem, 8.0 * float(cfg.get("topk_fraction") or 0.25))
    return int(cfg["embed_dim"]) * per_elem


def _keys(*kinds: str) -> str:
    out: list[str] = []
    for k in kinds:
        out += [x for x in SIZE_KEYS[k] if x not in out]
    return ", ".join(out)


def check_party(party_id: str, kind: str, cfg: dict, *, n_features: int | None, shape=None,
                rows: int | None = None, role: str = "passive", n_passive: int = 1,
                num_classes: int | None = None) -> None:
    """Raise ValueError if the models `party_id` would build for `cfg` are beyond the limits above.

    `n_features` None (no manifest information) skips the parameter/activation part; the wire check needs only
    the config. `rows` caps the batch sizes (a batch never holds more rows than the party has)."""
    cfg = {**default_session(), **cfg}      # a session config is complete; partial ones (tests, tools) get defaults
    errs = []
    cap = (lambda b: min(int(b), int(rows))) if rows else (lambda b: int(b))
    b_train, b_eval = cap(cfg["batch_size"]), cap(cfg["eval_batch_size"])
    b_wire = max(b_train, b_eval)
    wire = b_wire * wire_bytes_per_row(cfg)
    if wire > MAX_GRPC_MSG * WIRE_HEADROOM:
        errs.append(f"one [batch x embed_dim] = [{b_wire} x {cfg['embed_dim']}] tensor is about {wire / 2**20:,.0f} MiB "
                    f"on the wire, above the {MAX_GRPC_MSG / 2**20:,.0f} MiB message limit; "
                    f"reduce embed_dim, batch_size or eval_batch_size")
    if n_features and (kind not in SHAPED or (shape and len(shape) == SHAPED[kind])):
        est = bottom_estimate(kind, n_features, shape, cfg)
        kinds = [kind]
        if role == "active":
            top = top_estimate(cfg, n_passive, num_classes or 2)
            est.params += top.params
            est.act_per_row += top.act_per_row
            est.widest_per_row = max(est.widest_per_row, top.widest_per_row)
            kinds.append("top")
        what = f"its {kind} bottom model" + (" and the top model" if role == "active" else "")
        if est.params > MAX_PARTY_PARAMS:
            errs.append(f"{what} would have about {est.params:,} parameters over its {n_features} input features "
                        f"(limit {MAX_PARTY_PARAMS:,}); reduce {_keys(*kinds)}")
        acts = max(b_train * est.act_per_row, b_eval * est.widest_per_row)
        if acts > MAX_BATCH_ACTIVATIONS:
            errs.append(f"one batch through {what} would produce about {acts:,} activation values "
                        f"(limit {MAX_BATCH_ACTIVATIONS:,}); reduce batch_size, eval_batch_size or {_keys(*kinds)}")
    if errs:
        raise ValueError(f"session config too large for {party_id}: " + "; ".join(errs))
