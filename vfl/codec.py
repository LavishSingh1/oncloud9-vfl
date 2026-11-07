"""Tensor <-> protobuf encoding, byte accounting (NFR-4), lossy efficiency codecs (NFR-6, v2 item 34)
and validation of every received representation / gradient (v2 item 10).

Every codec carries the same logical tensor: [batch x embed_dim] float rows.
  fp32   little-endian float32                     (v1 default, exact)
  fp16   little-endian float16                     (v1 efficiency variant)
  int8   per-row symmetric scale (float32) + int8   (~4x smaller than fp32)
  topk   per row keep the k largest-|x| entries: uint16 index + float16 value (sparse)
"""
from __future__ import annotations

import numpy as np

from .proto import vfl_pb2 as pb

_DTYPES = {"fp32": ("float32", np.float32), "fp16": ("float16", np.float16)}
WIRE_DTYPES = ("float32", "float16", "int8rows", "topk")


class TensorRejected(ValueError):
    """Raised when an incoming tensor fails validation; `reason` is a short machine-readable tag."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def _topk_k(cols: int, cfg_or_frac) -> int:
    frac = cfg_or_frac if isinstance(cfg_or_frac, (int, float)) else float((cfg_or_frac or {}).get("topk_fraction", 0.25))
    return max(1, min(cols, int(round(frac * cols))))


def encode(arr: np.ndarray, codec: str = "fp32", topk_fraction: float = 0.25) -> pb.Tensor:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if codec in _DTYPES:
        name, dt = _DTYPES[codec]
        b = a.astype(np.dtype(dt).newbyteorder("<"))
        return pb.Tensor(shape=list(a.shape), dtype=name, data=b.tobytes())
    if a.ndim != 2:
        raise ValueError(f"{codec} codec needs a 2-D tensor")
    rows, cols = a.shape
    if codec == "int8":
        scale = (np.abs(a).max(axis=1) / 127.0).astype("<f4")
        scale[scale == 0] = 1.0
        q = np.clip(np.round(a / scale[:, None]), -127, 127).astype(np.int8)
        return pb.Tensor(shape=[rows, cols], dtype="int8rows", data=scale.tobytes() + q.tobytes())
    if codec == "topk":
        k = _topk_k(cols, topk_fraction)
        if cols > 65535:
            raise ValueError("topk codec supports up to 65535 columns")
        idx = np.argsort(-np.abs(a), axis=1, kind="stable")[:, :k]
        idx.sort(axis=1)
        val = np.take_along_axis(a, idx, axis=1).astype("<f2")
        header = np.array([k], dtype="<u4").tobytes()
        return pb.Tensor(shape=[rows, cols], dtype="topk", data=header + idx.astype("<u2").tobytes() + val.tobytes())
    raise ValueError(f"unknown codec {codec!r}")


MAX_ELEMENTS = 1 << 28                 # 1 GiB of float32: far above any batch x embed_dim


def decode(t: pb.Tensor, expected_shape: tuple | None = None) -> np.ndarray:
    """Decode a wire tensor. Every malformed input raises TensorRejected (v2.2 let integer overflow in the shape
    product, or a huge declared topk width, reach numpy and surface as unlogged INTERNAL errors / large
    allocations, RT-14). With `expected_shape` the declared shape is checked BEFORE anything is allocated."""
    try:
        return _decode(t, expected_shape)
    except TensorRejected:
        raise
    except Exception as e:                       # anything numpy raises on hostile input
        raise TensorRejected("malformed", f"{type(e).__name__}: {e}"[:200])


def _decode(t: pb.Tensor, expected_shape: tuple | None) -> np.ndarray:
    if t.dtype not in WIRE_DTYPES:
        raise TensorRejected("bad_dtype", f"unsupported tensor dtype {t.dtype!r}")
    shape = tuple(int(s) for s in t.shape)
    if any(s < 0 for s in shape) or len(shape) > 4:
        raise TensorRejected("bad_shape", f"invalid shape {shape}")
    if expected_shape is not None and shape != tuple(expected_shape):
        raise TensorRejected("bad_shape", f"shape {shape} != expected {tuple(expected_shape)}")
    n = 1
    for d in shape:
        n *= d                                   # Python ints: no overflow
    n = n if shape else 0
    if n > MAX_ELEMENTS:
        raise TensorRejected("bad_shape", f"tensor of {n} elements exceeds the {MAX_ELEMENTS} limit")
    if t.dtype in ("float32", "float16"):
        dt = np.dtype(t.dtype).newbyteorder("<")
        if len(t.data) != n * dt.itemsize:
            raise TensorRejected("malformed", f"tensor payload {len(t.data)} bytes != expected {n * dt.itemsize} "
                                              f"for {shape} {t.dtype}")
        return np.frombuffer(t.data, dtype=dt).reshape(shape).astype(np.float32)
    if len(shape) != 2:
        raise TensorRejected("bad_shape", f"{t.dtype} needs a 2-D shape, got {shape}")
    rows, cols = shape
    if t.dtype == "int8rows":
        if len(t.data) != rows * 4 + rows * cols:
            raise TensorRejected("malformed", f"int8rows payload {len(t.data)} bytes != {rows * 4 + rows * cols}")
        scale = np.frombuffer(t.data[:rows * 4], dtype="<f4")
        q = np.frombuffer(t.data[rows * 4:], dtype=np.int8).reshape(rows, cols)
        return (q.astype(np.float32) * scale[:, None]).astype(np.float32)
    if len(t.data) < 4:
        raise TensorRejected("malformed", "topk payload too short")
    k = int(np.frombuffer(t.data[:4], dtype="<u4")[0])
    if not 1 <= k <= cols or len(t.data) != 4 + rows * k * 4:
        raise TensorRejected("malformed", f"topk payload {len(t.data)} bytes inconsistent with k={k}, shape {shape}")
    idx = np.frombuffer(t.data[4:4 + rows * k * 2], dtype="<u2").reshape(rows, k).astype(np.int64)
    if idx.size and idx.max() >= cols:
        raise TensorRejected("malformed", "topk index out of range")
    val = np.frombuffer(t.data[4 + rows * k * 2:], dtype="<f2").reshape(rows, k).astype(np.float32)
    out = np.zeros((rows, cols), np.float32)
    np.put_along_axis(out, idx, val, axis=1)
    return out


def encode_cfg(arr: np.ndarray, cfg: dict) -> pb.Tensor:
    return encode(arr, cfg.get("codec", "fp32"), float(cfg.get("topk_fraction", 0.25)))


def roundtrip(arr: np.ndarray, cfg: dict) -> np.ndarray:
    """Exactly what the receiver decodes (used by the local simulator to mirror the wire)."""
    if cfg.get("codec", "fp32") == "fp32":
        return np.asarray(arr, dtype=np.float32)
    return decode(encode_cfg(arr, cfg))


def validate(arr: np.ndarray, expected_shape: tuple, max_row_norm: float | None = None) -> np.ndarray:
    """Shape, finiteness and optional per-row L2 bound. Raises TensorRejected."""
    if tuple(arr.shape) != tuple(expected_shape):
        raise TensorRejected("bad_shape", f"shape {tuple(arr.shape)} != expected {tuple(expected_shape)}")
    if not np.all(np.isfinite(arr)):
        raise TensorRejected("non_finite", f"{int(np.sum(~np.isfinite(arr)))} NaN/Inf values")
    if max_row_norm is not None and arr.size:
        m = float(np.max(np.linalg.norm(arr, axis=1)))
        if m > float(max_row_norm):
            raise TensorRejected("norm_bound", f"row L2 norm {m:.3g} exceeds bound {float(max_row_norm):.3g}")
    return arr


def tensor_bytes(t: pb.Tensor) -> int:
    return len(t.data)


def expected_tensor_bytes(rows: int, cols: int, codec: str, topk_fraction: float = 0.25) -> int:
    if codec in _DTYPES:
        return rows * cols * np.dtype(_DTYPES[codec][1]).itemsize
    if codec == "int8":
        return rows * 4 + rows * cols
    if codec == "topk":
        return 4 + rows * _topk_k(cols, topk_fraction) * 4
    raise ValueError(codec)
