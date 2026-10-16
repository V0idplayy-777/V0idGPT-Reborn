#!/usr/bin/env python3
"""
NeuronForge standalone inference runtime (Python / NumPy)
=========================================================

Runs a model exported from NeuronForge without NeuronForge.
Requires Python 3.8+ and NumPy. INFERENCE ONLY: no training, no gradients.

The maths mirrors NeuronForge's engine: NHWC layout, the same layer formulas
(GELU tanh approximation, numerically stable sigmoid, max-shifted softmax),
Dropout disabled, BatchNorm using running statistics. Intermediate results are
rounded to float32 at the same layer boundaries as the browser engine.
NumPy's matrix multiply accumulates in a different order than the JavaScript
loops, so results agree to within a small tolerance rather than bit-for-bit.

Usage:
    from neuronforge_model import load_model
    model = load_model()                 # reads bundle.json next to this file
    print(model.predict([0.25, 0.75]))
    print(model.verify())

Command line:
    python neuronforge_model.py --info
    python neuronforge_model.py --verify
    python neuronforge_model.py --predict 0.25,0.75
    python neuronforge_model.py --generate "hello there" --max-tokens 40
"""
from __future__ import annotations

import base64
import json
import re
import math
import os
import random
import re
import sys

import numpy as np

RUNTIME_VERSION = "1.3.0"
BUNDLE_FORMAT = "neuronforge-standalone"
BUNDLE_VERSION = 1
SUPPORTED_LAYERS = (
    "dense", "flatten", "reshape", "dropout", "batchnorm", "layernorm", "conv2d", "maxpool2d", "avgpool2d",
    "gap2d", "embedding", "attention", "transformer_block", "rnn", "lstm", "gru", "residual_dense", "residual_conv",
    "identity", "relu", "leakyrelu", "gelu", "silu", "sigmoid", "tanh", "softmax",
)
ACTIVATION_LAYERS = ("relu", "leakyrelu", "gelu", "silu", "sigmoid", "tanh", "softmax")
F32 = np.float32
F64 = np.float64
_M32 = 0xFFFFFFFF


# --------------------------------------------------------------------------- helpers

def _num(v, d):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        n = float(v)
    except (TypeError, ValueError):
        return d
    return n if math.isfinite(n) else d


def _int(v, d):
    return int(_num(v, d))


def _bool(v, d):
    return v if isinstance(v, bool) else d


def _kernel(v):
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return int(v[0]), int(v[1])
    n = _int(v, 3)
    return n, n


def _padding(c):
    p = c.get("padding")
    return "same" if p is None else str(p)


def _act_name(c):
    a = c.get("activation")
    return "none" if a is None else str(a)


def _numel(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _checksum(key: str, data: bytes) -> str:
    """Checksum over utf8(key) + tensor bytes; identical to the JS runtime."""
    arr = np.frombuffer(key.encode("utf-8") + data, dtype=np.uint8).astype(np.uint64)
    s1 = int(arr.sum()) % 2 ** 32
    w = (np.arange(arr.size, dtype=np.uint64) % np.uint64(65521)) + np.uint64(1)
    s2 = int((arr * w).sum()) % 2 ** 32
    return f"{s1:08x}{s2:08x}"


def _decode_tensor(entry, key, check_sums):
    if not entry or not isinstance(entry.get("data"), str):
        raise ValueError(f"Missing weight tensor '{key}'")
    raw = base64.b64decode(entry["data"])
    shape = [int(s) for s in entry["shape"]]
    n = _numel(shape)
    if len(raw) != n * 4:
        raise ValueError(f"Weight tensor '{key}' has {len(raw)} bytes but shape {shape} needs {n * 4}")
    if check_sums and entry.get("checksum") and _checksum(key, raw) != entry["checksum"]:
        raise ValueError(f"Checksum mismatch for weight tensor '{key}': bundle corrupted, edited, or tensor on the wrong layer")
    return np.frombuffer(raw, dtype="<f4").astype(F32).reshape(shape)


# --------------------------------------------------------------------------- shapes

def _conv_geom(H, W, kH, kW, stride, padding):
    if padding == "same":
        oh = math.ceil(H / stride)
        ow = math.ceil(W / stride)
        return oh, ow, max(0, (oh - 1) * stride + kH - H), max(0, (ow - 1) * stride + kW - W)
    return (H - kH) // stride + 1, (W - kW) // stride + 1, 0, 0


def _residual_width(c, din):
    u = int(round(_num(c.get("units"), 0)))
    return u if u > 0 else din


def _residual_conv_plan(c, s, name):
    """Geometry of a Residual Conv block: two k×k convolutions, stride on the first only."""
    if len(s) != 3:
        raise ValueError(f"Layer '{name}' (residual_conv): expects [H, W, C], got {s}")
    H, W, Cin = s
    kH, kW = _kernel(c.get("kernelSize"))
    stride = max(1, int(round(_num(c.get("stride"), 1))))
    padding = _padding(c)
    raw_f = int(round(_num(c.get("filters"), 0)))
    filters = raw_f if raw_f > 0 else Cin
    g1 = _conv_geom(H, W, kH, kW, stride, padding)
    g2 = _conv_geom(g1[0], g1[1], kH, kW, 1, padding)
    if g2[0] <= 0 or g2[1] <= 0:
        raise ValueError(f"Layer '{name}' (residual_conv): kernels do not fit input {s}")
    identity = stride == 1 and filters == Cin and g2[0] == H and g2[1] == W
    if not identity:
        proj = _conv_geom(H, W, 1, 1, stride, "valid")
        if proj[0] != g2[0] or proj[1] != g2[1]:
            raise ValueError(
                f"Layer '{name}' (residual_conv): shortcut {s} cannot be projected onto block output [{g2[0]}, {g2[1]}, {filters}]"
            )
    return {
        "H": H, "W": W, "Cin": Cin, "kH": kH, "kW": kW, "stride": stride, "padding": padding,
        "filters": filters, "outH": g2[0], "outW": g2[1], "identityShortcut": identity,
        "g1": g1, "g2": g2, "proj": _conv_geom(H, W, 1, 1, stride, "valid"),
    }


def _infer_shape(t, c, s, name):
    def fail(msg):
        raise ValueError(f"Layer '{name}' ({t}): {msg}")

    if t == "dense":
        if len(s) != 1:
            fail(f"expects rank-1 input, got {s}")
        return [_int(c.get("units"), 8)]
    if t == "flatten":
        return [_numel(s)]
    if t == "reshape":
        target = [int(v) for v in c.get("targetShape", [1])]
        if _numel(target) != _numel(s):
            fail(f"cannot reshape {s} to {target}")
        return target
    if t in ("dropout", "batchnorm", "layernorm", "identity") + ACTIVATION_LAYERS:
        return list(s)
    if t == "conv2d":
        if len(s) != 3:
            fail(f"expects [H, W, C], got {s}")
        kH, kW = _kernel(c.get("kernelSize"))
        oh, ow, _, _ = _conv_geom(s[0], s[1], kH, kW, _int(c.get("stride"), 1), _padding(c))
        return [oh, ow, _int(c.get("filters"), 8)]
    if t in ("maxpool2d", "avgpool2d"):
        if len(s) != 3:
            fail(f"expects [H, W, C], got {s}")
        p = _int(c.get("poolSize"), 2)
        st = _int(c.get("stride"), p)
        oh = (s[0] - p) // st + 1
        ow = (s[1] - p) // st + 1
        if oh <= 0 or ow <= 0:
            fail(f"pool {p}x{p} does not fit {s}")
        return [oh, ow, s[2]]
    if t == "gap2d":
        if len(s) != 3:
            fail(f"expects [H, W, C], got {s}")
        return [s[2]]
    if t == "embedding":
        if len(s) != 1:
            fail(f"expects [seqLen], got {s}")
        return [s[0], _int(c.get("embeddingDim"), 8)]
    if t in ("rnn", "lstm", "gru"):
        if len(s) != 2:
            fail(f"expects [seqLen, features], got {s}")
        u = _int(c.get("units"), 32)
        return [s[0], u] if _bool(c.get("returnSequences"), False) else [u]
    if t in ("attention", "transformer_block"):
        if len(s) != 2:
            fail(f"expects a sequence [seqLen, features], got {s}")
        plan = _attention_plan(c, s, name)
        return list(_attention_out(plan, s[0]))
    if t in ("attention", "transformer_block"):
        if len(s) != 2:
            fail(f"expects a sequence [seqLen, features], got {s}")
        plan = _attention_plan(c, s, name)
        return list(_attention_out(plan, s[0]))
    if t == "residual_dense":
        if len(s) != 1:
            fail(f"expects rank-1 input, got {s}")
        return [_residual_width(c, s[0])]
    if t == "residual_conv":
        plan = _residual_conv_plan(c, s, name)
        return [plan["outH"], plan["outW"], plan["filters"]]
    raise ValueError(f"Unsupported layer type '{t}' in layer '{name}'")


def _expected_params(t, c, s, out):
    if t == "dense":
        p = {"W": [s[0], out[0]]}
        if _bool(c.get("useBias"), True):
            p["b"] = [out[0]]
        return p
    if t == "conv2d":
        kH, kW = _kernel(c.get("kernelSize"))
        p = {"K": [kH, kW, s[2], out[2]]}
        if _bool(c.get("useBias"), True):
            p["b"] = [out[2]]
        return p
    if t == "batchnorm":
        f = s[-1]
        return {"gamma": [f], "beta": [f], "runningMean": [f], "runningVar": [f]}
    if t == "layernorm":
        f = s[-1]
        return {"gamma": [f], "beta": [f]}
    if t == "embedding":
        return {"W": [_int(c.get("vocabSize"), 32), out[1]]}
    if t == "rnn":
        h = _int(c.get("units"), 32)
        return {"Wxh": [s[1], h], "Whh": [h, h], "b": [h]}
    if t == "lstm":
        h = _int(c.get("units"), 32)
        return {"Wx": [s[1], 4 * h], "Wh": [h, 4 * h], "b": [4 * h]}
    if t == "gru":
        h = _int(c.get("units"), 32)
        return {
            "Wxz": [s[1], h], "Wxr": [s[1], h], "Wxn": [s[1], h],
            "Whz": [h, h], "Whr": [h, h], "Whn": [h, h],
            "bz": [h], "br": [h], "bn": [h],
        }
    if t == "attention":
        plan = _attention_plan(c, s, "attention")
        m = plan["M"]
        p = {"Wq": [m, m], "Wk": [m, m], "Wv": [m, m], "Wo": [m, m]}
        if plan["useBias"]:
            p["bq"] = [m]
            p["bk"] = [m]
            p["bv"] = [m]
            p["bo"] = [m]
        if s[1] != m:
            p["Win"] = [s[1], m]
            p["bin_b"] = [m]
        return p
    if t == "transformer_block":
        plan = _attention_plan(c, s, "transformer_block")
        m = plan["M"]
        ff = _ff_width(c, m)
        p = {
            "Wq": [m, m], "Wk": [m, m], "Wv": [m, m], "Wo": [m, m],
            "ln1_gamma": [m], "ln1_beta": [m], "ln2_gamma": [m], "ln2_beta": [m],
            "W1": [m, ff], "W2": [ff, m],
        }
        if plan["useBias"]:
            p["bq"] = [m]
            p["bk"] = [m]
            p["bv"] = [m]
            p["bo"] = [m]
            p["b1"] = [ff]
            p["b2"] = [m]
        if s[1] != m:
            p["Win"] = [s[1], m]
            p["bin_b"] = [m]
        return p
    if t == "attention":
        plan = _attention_plan(c, s, "attention")
        m = plan["M"]
        p = {"Wq": [m, m], "Wk": [m, m], "Wv": [m, m], "Wo": [m, m]}
        if plan["useBias"]:
            p["bq"] = [m]
            p["bk"] = [m]
            p["bv"] = [m]
            p["bo"] = [m]
        if s[1] != m:
            p["Win"] = [s[1], m]
            p["bin_b"] = [m]
        return p
    if t == "transformer_block":
        plan = _attention_plan(c, s, "transformer_block")
        m = plan["M"]
        ff = _ff_width(c, m)
        p = {
            "Wq": [m, m], "Wk": [m, m], "Wv": [m, m], "Wo": [m, m],
            "ln1_gamma": [m], "ln1_beta": [m], "ln2_gamma": [m], "ln2_beta": [m],
            "W1": [m, ff], "W2": [ff, m],
        }
        if plan["useBias"]:
            p["bq"] = [m]
            p["bk"] = [m]
            p["bv"] = [m]
            p["bo"] = [m]
            p["b1"] = [ff]
            p["b2"] = [m]
        if s[1] != m:
            p["Win"] = [s[1], m]
            p["bin_b"] = [m]
        return p
    if t == "residual_dense":
        w = _residual_width(c, s[0])
        p = {"W1": [s[0], w], "W2": [w, w]}
        if _bool(c.get("useBias"), True):
            p["b1"] = [w]
            p["b2"] = [w]
        if w != s[0]:
            p["Wp"] = [s[0], w]
        return p
    if t == "residual_conv":
        plan = _residual_conv_plan(c, s, "residual_conv")
        p = {"K1": [plan["kH"], plan["kW"], plan["Cin"], plan["filters"]], "K2": [plan["kH"], plan["kW"], plan["filters"], plan["filters"]]}
        if _bool(c.get("useBias"), True):
            p["b1"] = [plan["filters"]]
            p["b2"] = [plan["filters"]]
        if not plan["identityShortcut"]:
            p["Kp"] = [1, 1, plan["Cin"], plan["filters"]]
        return p
    return {}


# --------------------------------------------------------------------------- ops

def _attention_plan(c, s, name):
    """Normalised attention config for a declared input shape, with validation."""
    if len(s) != 2:
        raise ValueError(f"Layer '{name}': attention expects a sequence [seqLen, features], got {list(s)}")
    raw_m = int(round(_num(c.get("dModel"), 0)))
    m = raw_m if raw_m > 0 else int(s[1])
    heads = max(1, int(round(_num(c.get("heads"), 4))))
    if m <= 0 or m % heads != 0:
        raise ValueError(f"Layer '{name}': dModel {m} is not divisible by {heads} heads")
    pool = c.get("pool")
    if pool not in ("last", "mean"):
        pool = "none"
    return {
        "M": m,
        "heads": heads,
        "dh": m // heads,
        "causal": _bool(c.get("causal"), True),
        "useBias": _bool(c.get("useBias"), True),
        "invScale": _num(c.get("scale"), 1) / math.sqrt(m / heads),
        "pool": pool,
    }


def _ff_width(c, m):
    raw = int(round(_num(c.get("ffDim"), 0)))
    return raw if raw > 0 else 4 * m


def _attention_out(plan, T):
    return (T, plan["M"]) if plan["pool"] == "none" else (plan["M"],)


def _softmax_rows(x):
    """Row-wise softmax; rows that are entirely -inf become uniform."""
    m = np.max(x, axis=-1, keepdims=True)
    finite = np.isfinite(m)
    safe = np.where(finite, m, 0.0)
    e = np.exp(np.where(np.isfinite(x), x - safe, -np.inf))
    total = e.sum(axis=-1, keepdims=True)
    out = e / np.where(total > 0, total, 1.0)
    uniform = np.full_like(out, 1.0 / x.shape[-1])
    return np.where(finite, out, uniform)


def _layer_norm_rows(x, gamma, beta, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _mha_forward(xin, B, T, plan, P):
    """Multi-head self-attention over a (B*T, M) input, matching the engine's maths."""
    M, H, dh = plan["M"], plan["heads"], plan["dh"]
    scale = plan["invScale"]
    xt = np.asarray(xin, dtype=F64).reshape(B * T, M)
    Q = xt @ P["Wq"].astype(F64)
    K = xt @ P["Wk"].astype(F64)
    V = xt @ P["Wv"].astype(F64)
    if P.get("bq") is not None:
        Q = Q + P["bq"].astype(F64)
    if P.get("bk") is not None:
        K = K + P["bk"].astype(F64)
    if P.get("bv") is not None:
        V = V + P["bv"].astype(F64)
    Q = Q.reshape(B, T, M)
    K = K.reshape(B, T, M)
    V = V.reshape(B, T, M)
    O = np.zeros((B, T, M), dtype=F64)
    upper = np.triu(np.ones((T, T), dtype=bool), 1) if plan["causal"] else None
    for b in range(B):
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            qh = Q[b, :, sl]
            kh = K[b, :, sl]
            vh = V[b, :, sl]
            scores = (qh @ kh.T) * scale
            if upper is not None:
                scores = np.where(upper, -np.inf, scores)
            attn = _softmax_rows(scores)
            O[b, :, sl] = attn @ vh
    y = O.reshape(B * T, M) @ P["Wo"].astype(F64)
    if P.get("bo") is not None:
        y = y + P["bo"].astype(F64)
    return y


def _pool_attention(y, B, T, plan):
    """(B, T, M) -> the layer's declared output."""
    if plan["pool"] == "none":
        return y.reshape(B, T, plan["M"])
    if plan["pool"] == "last":
        return y.reshape(B, T, plan["M"])[:, -1, :]
    return y.reshape(B, T, plan["M"]).mean(axis=1)


def _attention_plan(c, s, name):
    """Normalised attention config for a declared input shape, with validation."""
    if len(s) != 2:
        raise ValueError(f"Layer '{name}': attention expects a sequence [seqLen, features], got {list(s)}")
    raw_m = int(round(_num(c.get("dModel"), 0)))
    m = raw_m if raw_m > 0 else int(s[1])
    heads = max(1, int(round(_num(c.get("heads"), 4))))
    if m <= 0 or m % heads != 0:
        raise ValueError(f"Layer '{name}': dModel {m} is not divisible by {heads} heads")
    pool = c.get("pool")
    if pool not in ("last", "mean"):
        pool = "none"
    return {
        "M": m,
        "heads": heads,
        "dh": m // heads,
        "causal": _bool(c.get("causal"), True),
        "useBias": _bool(c.get("useBias"), True),
        "invScale": _num(c.get("scale"), 1) / math.sqrt(m / heads),
        "pool": pool,
    }


def _ff_width(c, m):
    raw = int(round(_num(c.get("ffDim"), 0)))
    return raw if raw > 0 else 4 * m


def _attention_out(plan, T):
    return (T, plan["M"]) if plan["pool"] == "none" else (plan["M"],)


def _softmax_rows(x):
    """Row-wise softmax; rows that are entirely -inf become uniform."""
    m = np.max(x, axis=-1, keepdims=True)
    finite = np.isfinite(m)
    safe = np.where(finite, m, 0.0)
    e = np.exp(np.where(np.isfinite(x), x - safe, -np.inf))
    total = e.sum(axis=-1, keepdims=True)
    out = e / np.where(total > 0, total, 1.0)
    uniform = np.full_like(out, 1.0 / x.shape[-1])
    return np.where(finite, out, uniform)


def _layer_norm_rows(x, gamma, beta, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _mha_forward(xin, B, T, plan, P):
    """Multi-head self-attention over a (B*T, M) input, matching the engine's maths."""
    M, H, dh = plan["M"], plan["heads"], plan["dh"]
    scale = plan["invScale"]
    xt = np.asarray(xin, dtype=F64).reshape(B * T, M)
    Q = xt @ P["Wq"].astype(F64)
    K = xt @ P["Wk"].astype(F64)
    V = xt @ P["Wv"].astype(F64)
    if P.get("bq") is not None:
        Q = Q + P["bq"].astype(F64)
    if P.get("bk") is not None:
        K = K + P["bk"].astype(F64)
    if P.get("bv") is not None:
        V = V + P["bv"].astype(F64)
    Q = Q.reshape(B, T, M)
    K = K.reshape(B, T, M)
    V = V.reshape(B, T, M)
    O = np.zeros((B, T, M), dtype=F64)
    upper = np.triu(np.ones((T, T), dtype=bool), 1) if plan["causal"] else None
    for b in range(B):
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            qh = Q[b, :, sl]
            kh = K[b, :, sl]
            vh = V[b, :, sl]
            scores = (qh @ kh.T) * scale
            if upper is not None:
                scores = np.where(upper, -np.inf, scores)
            attn = _softmax_rows(scores)
            O[b, :, sl] = attn @ vh
    y = O.reshape(B * T, M) @ P["Wo"].astype(F64)
    if P.get("bo") is not None:
        y = y + P["bo"].astype(F64)
    return y


def _pool_attention(y, B, T, plan):
    """(B, T, M) -> the layer's declared output."""
    if plan["pool"] == "none":
        return y.reshape(B, T, plan["M"])
    if plan["pool"] == "last":
        return y.reshape(B, T, plan["M"])[:, -1, :]
    return y.reshape(B, T, plan["M"]).mean(axis=1)


def _sigmoid(x):
    xd = x.astype(F64)
    out = np.empty_like(xd)
    pos = xd >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-xd[pos]))
    e = np.exp(xd[~pos])
    out[~pos] = e / (1.0 + e)
    return out.astype(F32)


def _softmax(x):
    xd = x.astype(F64)
    m = xd.max(axis=-1, keepdims=True)
    ed = np.exp(xd - m)
    s = ed.sum(axis=-1, keepdims=True)
    s = np.where(s == 0, 1e-12, s)
    # NeuronForge stores exp() in float32 before scaling by 1/sum (sum of unrounded values).
    return (ed.astype(F32).astype(F64) * (1.0 / s)).astype(F32)


def _activation(name, x, alpha=0.01):
    if name == "linear":
        return x
    if name == "relu":
        return np.where(x > 0, x, 0).astype(F32)
    if name == "leakyrelu":
        return np.where(x > 0, x.astype(F64), alpha * x.astype(F64)).astype(F32)
    if name == "sigmoid":
        return _sigmoid(x)
    if name == "tanh":
        return np.tanh(x.astype(F64)).astype(F32)
    if name == "gelu":
        c = math.sqrt(2.0 / math.pi)
        v = x.astype(F64)
        return (0.5 * v * (1.0 + np.tanh(c * (v + 0.044715 * v * v * v)))).astype(F32)
    if name == "silu":
        return (x.astype(F64) * _sigmoid(x).astype(F64)).astype(F32)
    if name == "softmax":
        return _softmax(x)
    raise ValueError(f"Unsupported activation '{name}'")


def _compile(d, c, s, out, P):
    t = d["type"]
    name = d.get("name", t)

    if t == "dense":
        din = s[0]
        use_bias = _bool(c.get("useBias"), True)
        act = _act_name(c)
        W = P["W"].astype(F64)
        b = P["b"].astype(F64) if use_bias else None

        def f(x):
            if x.ndim != 2 or x.shape[1] != din:
                raise ValueError(f"Dense '{name}': expected [B, {din}], got {list(x.shape)}")
            y = (x.astype(F64) @ W).astype(F32)
            if use_bias:
                y = (y.astype(F64) + b).astype(F32)
            if act != "none":
                y = _activation(act, y, 0.01)
            return y
        return f

    if t == "flatten":
        return lambda x: x.reshape(x.shape[0], -1)
    if t == "reshape":
        return lambda x: x.reshape((x.shape[0],) + tuple(out))
    if t in ("dropout", "identity"):
        return lambda x: x  # Dropout is disabled at inference, as in NeuronForge evaluation.
    if t in ACTIVATION_LAYERS:
        alpha = _num(c.get("alpha"), 0.01)
        return lambda x: _activation(t, x, alpha)

    if t == "batchnorm":
        eps = _num(c.get("epsilon"), 1e-5)
        g, bt = P["gamma"].astype(F64), P["beta"].astype(F64)
        rm, rv = P["runningMean"].astype(F64), P["runningVar"].astype(F64)
        return lambda x: (g * ((x.astype(F64) - rm) / np.sqrt(rv + eps)) + bt).astype(F32)

    if t == "layernorm":
        eps = _num(c.get("epsilon"), 1e-5)
        g, bt = P["gamma"].astype(F64), P["beta"].astype(F64)

        def f(x):
            xd = x.astype(F64)
            m = xd.mean(axis=-1, keepdims=True)
            v = ((xd - m) ** 2).mean(axis=-1, keepdims=True)
            std = np.sqrt(v + eps).astype(F32).astype(F64)
            return (g * ((xd - m) / std) + bt).astype(F32)
        return f

    if t == "conv2d":
        H, W_, C = s
        oh, ow, F = out
        kH, kW = _kernel(c.get("kernelSize"))
        stride = _int(c.get("stride"), 1)
        _, _, pad_h, pad_w = _conv_geom(H, W_, kH, kW, stride, _padding(c))
        pt, pl = pad_h // 2, pad_w // 2
        K = P["K"].astype(F64)
        b = P["b"].astype(F64) if _bool(c.get("useBias"), True) else None
        act = _act_name(c)

        def f(x):
            B = x.shape[0]
            xp = np.zeros((B, H + pad_h, W_ + pad_w, C), dtype=F64)
            xp[:, pt:pt + H, pl:pl + W_, :] = x
            y = np.zeros((B, oh, ow, F), dtype=F64)
            for kh in range(kH):
                for kw in range(kW):
                    patch = xp[:, kh:kh + stride * (oh - 1) + 1:stride, kw:kw + stride * (ow - 1) + 1:stride, :]
                    y += patch @ K[kh, kw]
            if b is not None:
                y += b
            y = y.astype(F32)
            if act != "none":
                y = _activation(act, y, 0.01)
            return y
        return f

    if t in ("maxpool2d", "avgpool2d"):
        p = _int(c.get("poolSize"), 2)
        st = _int(c.get("stride"), p)
        oh, ow = out[0], out[1]
        is_max = t == "maxpool2d"

        def f(x):
            acc = None
            for kh in range(p):
                for kw in range(p):
                    win = x[:, kh:kh + st * (oh - 1) + 1:st, kw:kw + st * (ow - 1) + 1:st, :].astype(F64)
                    if acc is None:
                        acc = win.copy()
                    elif is_max:
                        acc = np.maximum(acc, win)
                    else:
                        acc = acc + win
            return (acc if is_max else acc / (p * p)).astype(F32)
        return f

    if t == "gap2d":
        return lambda x: x.astype(F64).mean(axis=(1, 2)).astype(F32)

    if t == "attention":
        plan = _attention_plan(c, s, name)
        din = int(s[1])
        m = plan["M"]
        proj = din != m

        def f(x):
            if x.ndim != 3:
                raise ValueError(f"Attention '{name}': expected [B, T, {din}], got {list(x.shape)}")
            B, T, _ = x.shape
            xin = x
            if proj:
                xin = (x.astype(F64).reshape(B * T, din) @ P["Win"].astype(F64) + P["bin_b"].astype(F64)).astype(F32)
            y = _mha_forward(xin, B, T, plan, P)
            return _pool_attention(y, B, T, plan).astype(F32)
        return f

    if t == "transformer_block":
        plan = _attention_plan(c, s, name)
        din = int(s[1])
        m = plan["M"]
        ff = _ff_width(c, m)
        proj = din != m
        eps = _num(c.get("epsilon"), 1e-5)
        act = _act_name(c)
        g1 = P["ln1_gamma"].astype(F64)
        b1 = P["ln1_beta"].astype(F64)
        g2 = P["ln2_gamma"].astype(F64)
        b2 = P["ln2_beta"].astype(F64)

        def f(x):
            if x.ndim != 3:
                raise ValueError(f"Transformer block '{name}': expected [B, T, {din}], got {list(x.shape)}")
            B, T, _ = x.shape
            xin = x
            if proj:
                xin = (x.astype(F64).reshape(B * T, din) @ P["Win"].astype(F64) + P["bin_b"].astype(F64)).astype(F32)
            xf = np.asarray(xin, dtype=F64).reshape(B * T, m)
            ln1 = _layer_norm_rows(xf, g1, b1, eps)
            attn = _mha_forward(ln1, B, T, plan, P)
            h = xf + attn
            ln2 = _layer_norm_rows(h, g2, b2, eps)
            hidden = ln2 @ P["W1"].astype(F64)
            if P.get("b1") is not None:
                hidden = hidden + P["b1"].astype(F64)
            hidden = hidden.astype(F32)
            if act != "none":
                hidden = _activation(act, hidden, 0.01)
            out = hidden.astype(F64) @ P["W2"].astype(F64)
            if P.get("b2") is not None:
                out = out + P["b2"].astype(F64)
            y = h + out
            return _pool_attention(y, B, T, plan).astype(F32)
        return f

    if t == "attention":
        plan = _attention_plan(c, s, name)
        din = int(s[1])
        m = plan["M"]
        proj = din != m

        def f(x):
            if x.ndim != 3:
                raise ValueError(f"Attention '{name}': expected [B, T, {din}], got {list(x.shape)}")
            B, T, _ = x.shape
            xin = x
            if proj:
                xin = (x.astype(F64).reshape(B * T, din) @ P["Win"].astype(F64) + P["bin_b"].astype(F64)).astype(F32)
            y = _mha_forward(xin, B, T, plan, P)
            return _pool_attention(y, B, T, plan).astype(F32)
        return f

    if t == "transformer_block":
        plan = _attention_plan(c, s, name)
        din = int(s[1])
        m = plan["M"]
        ff = _ff_width(c, m)
        proj = din != m
        eps = _num(c.get("epsilon"), 1e-5)
        act = _act_name(c)
        g1 = P["ln1_gamma"].astype(F64)
        b1 = P["ln1_beta"].astype(F64)
        g2 = P["ln2_gamma"].astype(F64)
        b2 = P["ln2_beta"].astype(F64)

        def f(x):
            if x.ndim != 3:
                raise ValueError(f"Transformer block '{name}': expected [B, T, {din}], got {list(x.shape)}")
            B, T, _ = x.shape
            xin = x
            if proj:
                xin = (x.astype(F64).reshape(B * T, din) @ P["Win"].astype(F64) + P["bin_b"].astype(F64)).astype(F32)
            xf = np.asarray(xin, dtype=F64).reshape(B * T, m)
            ln1 = _layer_norm_rows(xf, g1, b1, eps)
            attn = _mha_forward(ln1, B, T, plan, P)
            h = xf + attn
            ln2 = _layer_norm_rows(h, g2, b2, eps)
            hidden = ln2 @ P["W1"].astype(F64)
            if P.get("b1") is not None:
                hidden = hidden + P["b1"].astype(F64)
            hidden = hidden.astype(F32)
            if act != "none":
                hidden = _activation(act, hidden, 0.01)
            out = hidden.astype(F64) @ P["W2"].astype(F64)
            if P.get("b2") is not None:
                out = out + P["b2"].astype(F64)
            y = h + out
            return _pool_attention(y, B, T, plan).astype(F32)
        return f

    if t == "embedding":
        vocab = _int(c.get("vocabSize"), 32)
        Wt = P["W"]

        def f(x):
            B = x.shape[0]
            ids = np.floor(x.astype(F64).reshape(B, -1) + 0.5)  # Math.round semantics
            ids = np.nan_to_num(ids, nan=0.0)
            ids = np.clip(ids, 0, vocab - 1).astype(np.int64)
            return Wt[ids].astype(F32)
        return f

    if t == "rnn":
        units = _int(c.get("units"), 32)
        ret = _bool(c.get("returnSequences"), False)
        Wxh, Whh, b = P["Wxh"].astype(F64), P["Whh"].astype(F64), P["b"].astype(F64)

        def f(x):
            B, T, _ = x.shape
            h = np.zeros((B, units), dtype=F64)
            seq = []
            xd = x.astype(F64)
            for step in range(T):
                h = np.tanh(xd[:, step, :] @ Wxh + h @ Whh + b).astype(F32).astype(F64)
                if ret:
                    seq.append(h)
            return (np.stack(seq, axis=1) if ret else h).astype(F32)
        return f

    if t == "lstm":
        units = _int(c.get("units"), 32)
        ret = _bool(c.get("returnSequences"), False)
        Wx, Wh, b = P["Wx"].astype(F64), P["Wh"].astype(F64), P["b"].astype(F64)

        def f(x):
            B, T, _ = x.shape
            h = np.zeros((B, units), dtype=F64)
            c_state = np.zeros((B, units), dtype=F64)
            seq = []
            xd = x.astype(F64)
            for step in range(T):
                pre = xd[:, step, :] @ Wx + h @ Wh + b
                ig = _sigmoid(pre[:, :units])
                fg = _sigmoid(pre[:, units:2 * units])
                gg = np.tanh(pre[:, 2 * units:3 * units])
                og = _sigmoid(pre[:, 3 * units:4 * units])
                c_state = fg * c_state + ig * gg
                h = og * np.tanh(c_state)
                if ret:
                    seq.append(h)
            return (np.stack(seq, axis=1) if ret else h).astype(F32)
        return f

    if t == "gru":
        units = _int(c.get("units"), 32)
        ret = _bool(c.get("returnSequences"), False)
        Wxz, Wxr, Wxn = P["Wxz"].astype(F64), P["Wxr"].astype(F64), P["Wxn"].astype(F64)
        Whz, Whr, Whn = P["Whz"].astype(F64), P["Whr"].astype(F64), P["Whn"].astype(F64)
        bz, br, bn = P["bz"].astype(F64), P["br"].astype(F64), P["bn"].astype(F64)

        def f(x):
            B, T, _ = x.shape
            h = np.zeros((B, units), dtype=F64)
            seq = []
            xd = x.astype(F64)
            for step in range(T):
                xv = xd[:, step, :]
                z = _sigmoid(xv @ Wxz + h @ Whz + bz)
                r = _sigmoid(xv @ Wxr + h @ Whr + br)
                n = np.tanh(xv @ Wxn + (r * h) @ Whn + bn)
                h = (1.0 - z) * n + z * h
                if ret:
                    seq.append(h)
            return (np.stack(seq, axis=1) if ret else h).astype(F32)
        return f

    if t == "residual_dense":
        din = s[0]
        w = _residual_width(c, din)
        use_bias = _bool(c.get("useBias"), True)
        act = _act_name(c)
        act2 = str(c.get("finalActivation") or "none")
        scale = _num(c.get("scale"), 1.0)
        W1, W2 = P["W1"].astype(F64), P["W2"].astype(F64)
        b1 = P["b1"].astype(F64) if use_bias else None
        b2 = P["b2"].astype(F64) if use_bias else None
        Wp = P["Wp"].astype(F64) if w != din else None

        def f(x):
            if x.ndim != 2 or x.shape[1] != din:
                raise ValueError(f"Residual Dense '{name}': expected [B, {din}], got {list(x.shape)}")
            xd = x.astype(F64)
            h = xd @ W1
            if use_bias:
                h = h + b1
            if act != "none":
                h = _activation(act, h.astype(F32), 0.01).astype(F64)
            main = h @ W2
            if use_bias:
                main = main + b2
            skip = xd @ Wp if Wp is not None else xd
            y = (skip + scale * main).astype(F32)
            if act2 != "none":
                y = _activation(act2, y, 0.01)
            return y
        return f

    if t == "residual_conv":
        plan = _residual_conv_plan(c, s, name)
        H, W_, Cin = plan["H"], plan["W"], plan["Cin"]
        kH, kW = plan["kH"], plan["kW"]
        stride = plan["stride"]
        filters = plan["filters"]
        use_bias = _bool(c.get("useBias"), True)
        act = _act_name(c)
        act2 = str(c.get("finalActivation") or "none")
        scale = _num(c.get("scale"), 1.0)
        K1, K2 = P["K1"].astype(F64), P["K2"].astype(F64)
        b1 = P["b1"].astype(F64) if use_bias else None
        b2 = P["b2"].astype(F64) if use_bias else None
        Kp = P["Kp"].astype(F64) if not plan["identityShortcut"] else None

        def conv(xd, K, bias, oh, ow, st, geom):
            B = xd.shape[0]
            _, _, pad_h, pad_w = geom
            pt, pl = pad_h // 2, pad_w // 2
            xp = np.zeros((B, xd.shape[1] + pad_h, xd.shape[2] + pad_w, xd.shape[3]), dtype=F64)
            xp[:, pt:pt + xd.shape[1], pl:pl + xd.shape[2], :] = xd
            y = np.zeros((B, oh, ow, K.shape[3]), dtype=F64)
            for kh in range(K.shape[0]):
                for kw in range(K.shape[1]):
                    patch = xp[:, kh:kh + st * (oh - 1) + 1:st, kw:kw + st * (ow - 1) + 1:st, :]
                    y += patch @ K[kh, kw]
            if bias is not None:
                y += bias
            return y.astype(F32)

        def f(x):
            xd = x.astype(F64)
            h = conv(xd, K1, b1, plan["g1"][0], plan["g1"][1], stride, plan["g1"])
            if act != "none":
                h = _activation(act, h, 0.01).astype(F64)
            main = conv(h, K2, b2, plan["outH"], plan["outW"], 1, plan["g2"])
            if Kp is not None:
                skip = conv(xd, Kp, None, plan["proj"][0], plan["proj"][1], stride, plan["proj"])
            else:
                skip = x.astype(F32)
            y = (skip.astype(F64) + scale * main.astype(F64)).astype(F32)
            if act2 != "none":
                y = _activation(act2, y, 0.01)
            return y
        return f

    raise ValueError(f"Unsupported layer type '{t}'")


# --------------------------------------------------------------------------- tokenizer + sampling

_WORD_RE = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")


def tokenize_words(text):
    return _WORD_RE.findall(str(text).lower())


def join_words(words):
    if not words:
        return ""
    s = words[0]
    for w in words[1:]:
        if (len(w) == 1 and w in ",.;:!?)]}") or w in ("n't", "'s"):
            s += w
        else:
            s += " " + w
    return s


class RNG:
    """Mulberry32, bit-identical to NeuronForge's generator."""

    def __init__(self, seed):
        self.s = (int(seed) & _M32) or 1

    def next(self):
        self.s = (self.s + 0x6D2B79F5) & _M32
        t = self.s
        t = ((t ^ (t >> 15)) * (t | 1)) & _M32
        t = t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & _M32)) & _M32)
        return ((t ^ (t >> 14)) & _M32) / 4294967296


def _top_k_indices(logits, k):
    n = len(logits)
    stride = max(1, n // 512)
    sample = sorted((logits[i] for i in range(0, n, stride)), reverse=True)
    threshold = sample[min(k - 1, len(sample) - 1)]
    idx = [i for i in range(n) if logits[i] >= threshold]
    if len(idx) > k:
        idx.sort(key=lambda i: -logits[i])
        idx = idx[:k]
    return idx


def sample_from_probs(probs, temperature, top_k, rng):
    """Temperature + top-k sampling, same algorithm as the NeuronForge sampler."""
    n = len(probs)
    if n == 0:
        raise ValueError("Cannot sample from an empty distribution")

    def argmax():
        best = 0
        for i in range(1, n):
            if probs[i] > probs[best]:
                best = i
        return best

    if not math.isfinite(temperature) or temperature <= 0.01:
        return argmax()
    inv_t = 1.0 / temperature
    logits = [math.log(max(float(p), 1e-9)) * inv_t for p in probs]
    k = max(1, min(int(math.floor(top_k)) or n, n))
    cand = list(range(n)) if k >= n else _top_k_indices(logits, k)
    mx = max(logits[i] for i in cand)
    weights = [math.exp(logits[i] - mx) for i in cand]
    total = sum(weights)
    if not total > 0:
        return argmax()
    r = rng.next() * total
    for i, w in zip(cand, weights):
        r -= w
        if r <= 0:
            return i
    return cand[-1]


# --------------------------------------------------------------------------- model

class StandaloneModel:
    def __init__(self, bundle, verify_checksums=True):
        if isinstance(bundle, str):
            bundle = json.loads(bundle)
        if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
            raise ValueError("Not a NeuronForge standalone bundle")
        if bundle.get("formatVersion") != BUNDLE_VERSION:
            raise ValueError(f"Bundle format version {bundle.get('formatVersion')} not supported by runtime {RUNTIME_VERSION}")
        self.bundle = bundle
        arch = bundle["architecture"]
        weights = bundle.get("weights", {})
        self.layers = []
        self.layer_info = []
        self.weights = {}
        used = set()
        shape = [int(v) for v in arch["inputShape"]]
        self.param_count = 0
        for idx, d in enumerate(arch["layers"]):
            t = d["type"]
            if t not in SUPPORTED_LAYERS:
                raise ValueError(f"Unsupported layer type '{t}' (layer {idx + 1}, '{d.get('name')}'). "
                                 f"Runtime {RUNTIME_VERSION} supports: {', '.join(SUPPORTED_LAYERS)}")
            c = d.get("config") or {}
            if t in ("dense", "conv2d") and _act_name(c) not in ("none", "linear") + ACTIVATION_LAYERS:
                raise ValueError(f"Layer '{d.get('name')}' uses unsupported fused activation '{_act_name(c)}'")
            out = _infer_shape(t, c, shape, d.get("name"))
            if "inputShape" in d and [int(v) for v in d["inputShape"]] != shape:
                raise ValueError(f"Shape check failed at '{d.get('name')}': bundle input {d['inputShape']} vs computed {shape}")
            if "outputShape" in d and [int(v) for v in d["outputShape"]] != out:
                raise ValueError(f"Shape check failed at '{d.get('name')}': bundle output {d['outputShape']} vs computed {out}")
            P = {}
            for pname, eshape in _expected_params(t, c, shape, out).items():
                key = f"{d['id']}.{pname}"
                entry = weights.get(key)
                if entry is None:
                    raise ValueError(f"Missing weight tensor '{key}' for layer '{d.get('name')}' ({t})")
                if entry.get("dtype", "float32") != "float32":
                    raise ValueError(f"Weight '{key}' dtype {entry.get('dtype')} unsupported")
                if [int(v) for v in entry["shape"]] != [int(v) for v in eshape]:
                    raise ValueError(f"Weight '{key}' has shape {entry['shape']} but layer needs {eshape}")
                P[pname] = _decode_tensor(entry, key, verify_checksums)
                self.weights[key] = P[pname]
                used.add(key)
                self.param_count += P[pname].size
            self.layers.append(_compile(d, c, shape, out, P))
            self.layer_info.append({"name": d.get("name"), "type": t, "inputShape": list(shape), "outputShape": list(out)})
            shape = out
        unused = sorted(set(weights) - used)
        if unused:
            raise ValueError(f"Weight tensor(s) not used by any layer: {unused}")

        self.input_shape = [int(v) for v in arch["inputShape"]]
        self.output_shape = shape
        self.in_size = _numel(self.input_shape)
        self.task = bundle.get("task", "raw")
        self.output = bundle.get("output") or {}
        self.tokenizer = bundle.get("tokenizer")
        self.preprocessing = bundle.get("preprocessing")
        self.image = bundle.get("image")
        self.sampling = bundle.get("sampling") or {"temperature": 0.7, "topK": 50}
        self._stoi = {}
        self._pad_id = 0
        self._ranks = None
        self._ranks_built = False
        if self.tokenizer:
            for i, tkn in enumerate(self.tokenizer["tokens"]):
                self._stoi[tkn] = i
            self._pad_id = self._stoi.get(self.tokenizer.get("padToken"), 0)
            if self.task == "language" and len(self.tokenizer["tokens"]) != _numel(self.output_shape):
                raise ValueError("Tokenizer size does not match the model's output classes")

    # ---- core ----
    def forward(self, x):
        """Run the network on a batch shaped [B, *input_shape]. Returns float32 ndarray."""
        x = np.asarray(x, dtype=F32)
        if list(x.shape[1:]) != self.input_shape:
            x = x.reshape((-1,) + tuple(self.input_shape))
        for layer in self.layers:
            x = layer(x)
        return x

    def _label(self, i):
        names = self.output.get("classNames")
        return str(names[i]) if names and i < len(names) else str(i)

    # ---- preprocessing ----
    def encode(self, row):
        pp = self.preprocessing
        if not pp or pp.get("kind") != "tabular":
            raise ValueError("This model has no tabular preprocessing; pass numeric features directly")
        cols = pp["columns"]
        if isinstance(row, (list, tuple)) and len(row) != len(cols):
            raise ValueError(f"Expected {len(cols)} raw values ({', '.join(c['name'] for c in cols)}), got {len(row)}")
        vec = []
        for i, col in enumerate(cols):
            v = row[i] if isinstance(row, (list, tuple)) else (row or {}).get(col["name"])
            if col["type"] == "numeric":
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    num = float(v) if math.isfinite(float(v)) else float("nan")
                else:
                    raw = "" if v is None else str(v).strip()
                    try:
                        num = float(raw) if raw != "" else float("nan")
                    except ValueError:
                        num = float("nan")
                    if not math.isfinite(num):
                        num = float("nan")
                if math.isnan(num) and pp["missing"] == "drop":
                    raise ValueError(f"Column '{col['name']}' is missing or not numeric and the model was trained with policy 'drop'")
                vec.append(num)
            else:
                s = "" if v is None else str(v).strip()
                cats = col["categories"]
                if col["encoding"] == "label":
                    vec.append(0 if s == "" else (cats.index(s) if s in cats else 0))
                else:
                    oh = [0.0] * max(1, len(cats))
                    if s in cats:
                        oh[cats.index(s)] = 1.0
                    vec.extend(oh)
        out = []
        for j, v in enumerate(vec):
            x = v if math.isfinite(v) else (0.0 if pp["missing"] == "zero" else pp["mean"][j])
            if pp["normalize"] == "minmax":
                d = (pp["max"][j] - pp["min"][j]) or 1.0
                x = (x - pp["min"][j]) / d
            elif pp["normalize"] == "zscore":
                x = (x - pp["mean"][j]) / (pp["std"][j] or 1.0)
            out.append(x)
        return out

    # ---- tokenizer ----
    def _bpe_ranks(self):
        """Rank lookup for the bundle's merge rules, or None without merges."""
        if self._ranks_built:
            return self._ranks
        self._ranks_built = True
        tok = self.tokenizer or {}
        merges = tok.get("merges")
        if tok.get("level") != "bpe" or not merges:
            self._ranks = None
            return None
        self._ranks = {str(rule[0]) + "\u0000" + str(rule[1]): i for i, rule in enumerate(merges) if len(rule) >= 2}
        return self._ranks

    def _bpe_encode_chunk(self, chunk, ranks):
        """Apply the merge rules to one non-whitespace run (lowest rank first)."""
        symbols = list(chunk)
        while len(symbols) > 1:
            best_rank = None
            best_at = -1
            for i in range(len(symbols) - 1):
                rank = ranks.get(symbols[i] + "\u0000" + symbols[i + 1])
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_at = i
            if best_at < 0:
                break
            symbols[best_at:best_at + 2] = [symbols[best_at] + symbols[best_at + 1]]
        return symbols

    def tokenize(self, text):
        if not self.tokenizer:
            raise ValueError("This model was not exported with a tokenizer")
        if self.tokenizer["level"] == "word":
            return [self._stoi.get(w, 0) for w in tokenize_words(text)]
        ranks = self._bpe_ranks()
        if ranks is not None:
            ids = []
            for chunk in re.findall(r"\s+|\S+", str(text).lower()):
                symbols = list(chunk) if chunk[:1].isspace() else self._bpe_encode_chunk(chunk, ranks)
                for symbol in symbols:
                    ids.append(self._stoi.get(symbol, 0))
            return ids
        return [self._stoi[ch] for ch in str(text).lower() if ch in self._stoi]

    def detokenize(self, ids):
        toks = self.tokenizer["tokens"]
        parts = [toks[i] for i in ids if 0 <= i < len(toks)]
        return join_words(parts) if self.tokenizer["level"] == "word" else "".join(parts)

    def _window(self, ids):
        seq = self.input_shape[0]
        w = list(ids[-seq:])
        return [self._pad_id] * (seq - len(w)) + w

    def _prepare(self, x, preprocessed=False):
        if self.task == "language" and isinstance(x, str):
            ids = self.tokenize(x)
            if not ids:
                raise ValueError("The text contains no tokens from the vocabulary")
            return self._window(ids)
        pp = self.preprocessing
        if pp and pp.get("kind") == "tabular" and not preprocessed and isinstance(x, (dict, list, tuple)):
            return self.encode(x)
        flat = np.asarray(x, dtype=F64).reshape(-1).tolist()
        if len(flat) != self.in_size:
            raise ValueError(f"Expected {self.in_size} values for input shape {self.input_shape}, got {len(flat)}")
        return flat

    def _interpret(self, raw, flat):
        raw = [float(v) for v in raw]
        t = self.task
        if t == "binary":
            p = raw[0]
            cls = 1 if p >= 0.5 else 0
            return {"task": t, "predicted": cls, "label": self._label(cls),
                    "probability": p if self.output.get("outputsAreProbabilities") else None, "score": p, "raw": raw}
        if t == "multiclass":
            best = int(np.argmax(raw))
            order = sorted(range(len(raw)), key=lambda k: -raw[k])[:5]
            return {"task": t, "predicted": best, "label": self._label(best),
                    "probabilities": raw if self.output.get("outputsAreProbabilities") else None, "scores": raw,
                    "top": [{"index": k, "label": self._label(k), "value": raw[k]} for k in order], "raw": raw}
        if t == "regression":
            names = self.output.get("targetNames") or []
            return {"task": t, "value": raw[0], "values": raw, "named": dict(zip(names, raw)), "raw": raw}
        if t == "reconstruction":
            err = float(np.mean((np.asarray(raw) - np.asarray(flat, dtype=F64)) ** 2)) if raw else 0.0
            return {"task": t, "reconstruction": raw, "reconstructionError": err, "raw": raw}
        if t == "language":
            order = sorted(range(len(raw)), key=lambda k: -raw[k])[:10]
            top = [{"id": k, "token": self.tokenizer["tokens"][k], "probability": raw[k]} for k in order]
            return {"task": t, "next": top[0]["token"] if top else None, "top": top, "raw": raw}
        return {"task": "raw", "output": raw, "shape": self.output_shape, "raw": raw}

    def predict(self, x, preprocessed=False):
        flat = self._prepare(x, preprocessed)
        out = self.forward(np.asarray([flat], dtype=F32))
        return self._interpret(out.reshape(-1), flat)

    def predict_batch(self, xs, preprocessed=False):
        flats = [self._prepare(x, preprocessed) for x in xs]
        out = self.forward(np.asarray(flats, dtype=F32)).reshape(len(flats), -1)
        return [self._interpret(o, f) for o, f in zip(out, flats)]

    def next_token(self, text):
        return self.predict(str(text))

    def generate_detailed(self, prompt, max_tokens=60, temperature=None, top_k=None, seed=None, stop=None, stop_at_eos=True):
        if not self.tokenizer or self.task != "language":
            raise ValueError("generate() is only available for language models")
        temperature = self.sampling["temperature"] if temperature is None else float(temperature)
        top_k = self.sampling["topK"] if top_k is None else int(top_k)
        seed = random.getrandbits(32) if seed is None else int(seed)
        if stop is None:
            stop = [self.tokenizer["eos"]] if stop_at_eos and self.tokenizer.get("eos") else []
        prompt_ids = self.tokenize(prompt)
        if not prompt_ids:
            level = self.tokenizer["level"]
            raise ValueError(
                "The prompt contains no word tokens"
                if level == "word"
                else "The prompt produced no subword tokens"
                if level == "bpe"
                else "None of the prompt's characters are in the vocabulary"
            )
        rng = RNG(seed)
        ctx = list(prompt_ids)
        ids = []
        stopped_by = None
        toks = self.tokenizer["tokens"]
        for _ in range(int(max_tokens)):
            probs = self.forward(np.asarray([self._window(ctx)], dtype=F32)).reshape(-1)
            idx = sample_from_probs(probs, temperature, top_k, rng)
            if toks[idx] in stop:
                stopped_by = toks[idx]
                break
            ids.append(idx)
            ctx.append(idx)
        text = self.detokenize(ids)
        if self.tokenizer["level"] == "word":
            full = str(prompt).strip() + ("" if not text or text[0] in ",.;:!?)]}" else " ") + text
        else:
            full = str(prompt) + text
        return {"text": text, "fullText": full, "ids": ids, "tokens": [toks[i] for i in ids], "promptIds": prompt_ids,
                "stoppedBy": stopped_by, "settings": {"maxTokens": max_tokens, "temperature": temperature, "topK": top_k, "seed": seed}}

    def generate(self, prompt, max_tokens=60, temperature=None, top_k=None, seed=None, stop=None, include_prompt=False):
        r = self.generate_detailed(prompt, max_tokens, temperature, top_k, seed, stop)
        return r["fullText"] if include_prompt else r["text"]

    def image_to_input(self, pixels, width=None, height=None, channels=None, scale=None, invert=False):
        spec = self.image
        if not spec:
            raise ValueError("This model was not exported with an image specification")
        W, H, C = spec["width"], spec["height"], spec["channels"]
        src = np.asarray(pixels)
        if src.ndim == 3:
            height, width, channels = src.shape
        sw, sh = width or W, height or H
        flat = src.reshape(-1).astype(F64)
        sc = channels or (4 if flat.size == sw * sh * 4 else 3 if flat.size == sw * sh * 3 else 1 if flat.size == sw * sh else 0)
        if not sc:
            raise ValueError(f"Cannot infer channels: {flat.size} values for a {sw}x{sh} image")
        if scale is None:
            scale = 1 / 255 if src.dtype == np.uint8 else 1.0
        lo, hi = spec["range"]
        out = []
        for y in range(H):
            for x in range(W):
                sx = min(sw - 1, (x * sw) // W)
                sy = min(sh - 1, (y * sh) // H)
                base = (sy * sw + sx) * sc
                for ch in range(C):
                    if C == 1:
                        v = flat[base:base + 3].mean() if sc >= 3 else flat[base]
                    else:
                        v = flat[base + min(ch, sc - 1)]
                    v *= scale
                    if invert:
                        v = 1 - v
                    out.append(lo + v * (hi - lo))
        return out

    def predict_image(self, pixels, **kw):
        flat = self.image_to_input(pixels, **kw)
        return self._interpret(self.forward(np.asarray([flat], dtype=F32)).reshape(-1), flat)

    # ---- verification ----
    def verify(self):
        tests = self.bundle.get("tests") or {}
        # NumPy accumulates in a different order than the JS loops: allow a slightly wider tolerance.
        atol, rtol = 1e-4, 1e-3
        checks = []
        for i, tc in enumerate(tests.get("forward", [])):
            try:
                got = self.forward(np.asarray([tc["input"]], dtype=F32)).reshape(-1).astype(F64)
                exp = np.asarray(tc["output"], dtype=F64)
                if got.shape != exp.shape:
                    checks.append((f"forward #{i + 1}", "fail", f"output length {got.size} != {exp.size}"))
                    continue
                diff = np.abs(got - exp)
                bad = int(np.sum(diff > atol + rtol * np.abs(exp)))
                checks.append((f"forward #{i + 1}", "fail" if bad else "pass",
                               f"max |diff| = {diff.max():.2e}" + (f", {bad} values outside tolerance" if bad else "")))
            except Exception as e:  # noqa: BLE001
                checks.append((f"forward #{i + 1}", "fail", str(e)))
        for i, tc in enumerate(tests.get("tokenizer", [])):
            ok = self.tokenize(tc["text"]) == tc["ids"]
            checks.append((f"tokenizer #{i + 1}", "pass" if ok else "fail", "ids match" if ok else "ids differ"))
        for i, tc in enumerate(tests.get("generation", [])):
            try:
                r = self.generate_detailed(tc["prompt"], tc["maxTokens"], tc["temperature"], tc["topK"], tc["seed"], stop=[])
                same = r["ids"] == tc["ids"]
                checks.append((f"generation #{i + 1}", "pass" if same else "warn",
                               "identical tokens" if same else "sampled tokens diverged (expected occasionally: float differences can flip a sampling tie)"))
            except Exception as e:  # noqa: BLE001
                checks.append((f"generation #{i + 1}", "fail", str(e)))
        for i, tc in enumerate(tests.get("preprocessing", [])):
            try:
                enc = np.asarray(self.encode(tc["raw"]), dtype=F64)
                exp = np.asarray(tc["encoded"], dtype=F64)
                ok = enc.shape == exp.shape and bool(np.all(np.abs(enc - exp) <= 1e-9 + 1e-7 * np.abs(exp)))
                checks.append((f"preprocessing #{i + 1}", "pass" if ok else "fail", "encoding matches" if ok else "encoding differs"))
            except Exception as e:  # noqa: BLE001
                checks.append((f"preprocessing #{i + 1}", "fail", str(e)))
        return {"ok": all(c[1] != "fail" for c in checks), "runtimeVersion": RUNTIME_VERSION,
                "checks": [{"name": n, "status": s, "detail": d} for n, s, d in checks]}

    @property
    def info(self):
        return {"name": self.bundle.get("name"), "task": self.task, "inputShape": self.input_shape,
                "outputShape": self.output_shape, "paramCount": self.param_count, "layers": self.layer_info,
                "tokenizer": None if not self.tokenizer else {"level": self.tokenizer["level"],
                                                              "vocabSize": len(self.tokenizer["tokens"]),
                                                              "seqLen": self.tokenizer["seqLen"],
                                                              "merges": len(self.tokenizer.get("merges") or [])},
                "limitations": self.bundle.get("limitations", []), "runtimeVersion": RUNTIME_VERSION}


def load_model(path=None, verify_checksums=True):
    """Load bundle.json (default: next to this file) and return a StandaloneModel."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundle.json")
    with open(path, "r", encoding="utf-8") as fh:
        return StandaloneModel(json.load(fh), verify_checksums=verify_checksums)


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="NeuronForge standalone model")
    ap.add_argument("--bundle", default=None, help="path to bundle.json")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--verify", action="store_true", help="run the test vectors embedded by NeuronForge")
    ap.add_argument("--predict", help="comma-separated input values")
    ap.add_argument("--generate", help="prompt for language models")
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    m = load_model(a.bundle)
    if a.info:
        print(json.dumps(m.info, indent=2))
    if a.verify:
        rep = m.verify()
        for c in rep["checks"]:
            print(f"[{c['status'].upper():4}] {c['name']}: {c['detail']}")
        print("OK" if rep["ok"] else "FAILED")
        if not rep["ok"]:
            return 1
    if a.predict:
        vals = [v.strip() for v in a.predict.split(",")]
        if not (m.preprocessing and m.preprocessing.get("kind") == "tabular"):
            vals = [float(v) for v in vals]
        print(json.dumps(m.predict(vals), indent=2))
    if a.generate is not None:
        print(m.generate(a.generate, a.max_tokens, a.temperature, a.top_k, a.seed, include_prompt=True))
    if not (a.info or a.verify or a.predict or a.generate is not None):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
