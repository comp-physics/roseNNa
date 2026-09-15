"""Every rejection `validate` promises, exercised.

`validate.py` exists so that a model roseNNa cannot lower is refused by name
rather than silently mis-compiled -- its own docstring says the failure mode it
prevents is "plausible numbers that are wrong". A rejection nobody has run is
not that guarantee: the condition may be inverted, may raise `TypeError` or
`KeyError` before reaching the `raise`, or may name the wrong node.

Coverage of validate.py was 75% before this file, and every uncovered line was
a `raise`.

These build `frontend.Graph` objects directly rather than going through ONNX.
That is deliberate: most of these models are ones `onnx.checker` would reject
too, and the point here is what *roseNNa* says about them, not whether ONNX can
represent them.
"""
import numpy as np
import pytest

from rosenna.errors import UnsupportedModel
from rosenna.frontend import Graph, Node, Tensor
from rosenna.validate import validate

F32 = "f32"


def _t(name, shape):
    return Tensor(name, tuple(shape), F32)


def _graph(node, values, inits=None, inputs=("x",), outputs=("y",)):
    return Graph("t", (node,), dict(values), dict(inits or {}), tuple(inputs), tuple(outputs))


def _reject(graph, fragment):
    with pytest.raises(UnsupportedModel, match=fragment):
        validate(graph)


def _arr(*shape):
    return np.ones(shape, dtype=np.float32)


# --- Gemm / MatMul --------------------------------------------------------

def test_gemm_rejections():
    vals = {"x": _t("x", (1, 3)), "y": _t("y", (1, 2))}
    w, b = _arr(3, 2), _arr(2)
    _reject(_graph(Node("Gemm", "g", ("x",), ("y",), {}), vals, {}),
            "requires at least 2 inputs")
    _reject(_graph(Node("Gemm", "g", ("x", "w"), ("y",), {"transA": 1}), vals, {"w": w}),
            "transA=1")
    for attr in ("alpha", "beta"):
        _reject(_graph(Node("Gemm", "g", ("x", "w"), ("y",), {attr: 2.0}), vals, {"w": w}),
                f"Gemm {attr}")
    _reject(_graph(Node("Gemm", "g", ("x", "runtime"), ("y",), {}),
                   {**vals, "runtime": _t("runtime", (3, 2))}, {}),
            "must be a constant")
    _reject(_graph(Node("Gemm", "g", ("x", "w"), ("y",), {}), vals, {"w": _arr(2, 3, 2)}),
            "has rank 3")
    _reject(_graph(Node("Gemm", "g", ("x", "w", "bad"), ("y",), {}),
                   {**vals, "bad": _t("bad", (2,))}, {"w": w}),
            "bias 'bad' must be a constant")
    _reject(_graph(Node("Gemm", "g", ("x", "w", "b"), ("y",), {}), vals, {"w": w, "b": _arr(1, 2)}),
            "bias has rank 2")
    _reject(_graph(Node("Gemm", "g", ("x", "w", "b"), ("y",), {}), vals, {"w": w, "b": _arr(5)}),
            "values for 2 outputs")


def test_matmul_rejections():
    vals = {"x": _t("x", (1, 3)), "y": _t("y", (1, 2))}
    _reject(_graph(Node("MatMul", "m", ("x",), ("y",), {}), vals, {}),
            "requires at least 2 inputs")
    _reject(_graph(Node("MatMul", "m", ("x", "r"), ("y",), {}),
                   {**vals, "r": _t("r", (3, 2))}, {}),
            "needs a constant second input")
    _reject(_graph(Node("MatMul", "m", ("x", "w"), ("y",), {}), vals, {"w": _arr(2, 3, 2)}),
            "has rank 3")


# --- Conv / pooling -------------------------------------------------------

def _spatial_vals(in_shape=(1, 2, 8, 8), out_shape=(1, 2, 6, 6)):
    return {"x": _t("x", in_shape), "y": _t("y", out_shape)}


def test_spatial_shape_rejections():
    _reject(_graph(Node("MaxPool", "p", ("x",), ("y",), {"kernel_shape": (2, 2)}),
                   {"x": _t("x", (1, 8)), "y": _t("y", (1, 4))}),
            "has rank 2; only rank-4 NCHW")
    _reject(_graph(Node("MaxPool", "p", ("x",), ("y",), {"kernel_shape": (2, 2)}),
                   {"x": _t("x", (1, 2, 8, 8)), "y": _t("y", (1, 4))}),
            "must be a rank-4 value")
    _reject(_graph(Node("MaxPool", "p", ("x",), ("y",), {}), _spatial_vals()),
            "needs kernel_shape")
    _reject(_graph(Node("MaxPool", "p", ("x",), ("y",), {"kernel_shape": (2, 2, 2)}),
                   _spatial_vals()),
            "only 2-D is supported")


def test_spatial_attribute_rejections():
    base = {"kernel_shape": (3, 3)}
    for attr in ("strides", "dilations"):
        _reject(_graph(Node("MaxPool", "p", ("x",), ("y",), {**base, attr: (1, 1, 1)}),
                       _spatial_vals()),
                f"{attr} has 3 entries")
    _reject(_graph(Node("MaxPool", "p", ("x",), ("y",), {**base, "pads": (0, 0)}), _spatial_vals()),
            "pads has 2 entries")
    _reject(_graph(Node("MaxPool", "p", ("x",), ("y",), {**base, "auto_pad": "WEIRD"}),
                   _spatial_vals()),
            "auto_pad='WEIRD'")
    _reject(_graph(Node("MaxPool", "p", ("x",), ("y",),
                        {**base, "auto_pad": "SAME_UPPER", "pads": (1, 1, 1, 1)}), _spatial_vals()),
            "is ambiguous")
    _reject(_graph(Node("MaxPool", "p", ("x",), ("y",), {**base, "ceil_mode": 1}), _spatial_vals()),
            "ceil_mode=1")
    _reject(_graph(Node("MaxPool", "p", ("x",), ("y", "idx"), base), _spatial_vals()),
            "second .indices. output")
    _reject(_graph(Node("MaxPool", "p", ("x",), ("y",), {**base, "storage_order": 1}),
                   _spatial_vals()),
            "storage_order=1")


def test_conv_rejections():
    vals = _spatial_vals()
    good = _arr(2, 2, 3, 3)
    _reject(_graph(Node("Conv", "c", ("x", "rt"), ("y",), {"kernel_shape": (3, 3)}),
                   {**vals, "rt": _t("rt", (2, 2, 3, 3))}, {}),
            "weight must be a constant initializer")
    _reject(_graph(Node("Conv", "c", ("x", "w"), ("y",), {"kernel_shape": (3, 3)}),
                   vals, {"w": _arr(2, 2, 3)}),
            "weight has rank 3")
    _reject(_graph(Node("Conv", "c", ("x", "w"), ("y",), {"group": 2}), vals, {"w": good}),
            "grouped Conv")
    _reject(_graph(Node("Conv", "c", ("x", "w"), ("y",), {}), vals, {"w": _arr(2, 5, 3, 3)}),
            "channels but the weight expects")
    _reject(_graph(Node("Conv", "c", ("x", "w", "rb"), ("y",), {}),
                   {**vals, "rb": _t("rb", (2,))}, {"w": good}),
            "bias 'rb' must be a constant")
    _reject(_graph(Node("Conv", "c", ("x", "w", "b"), ("y",), {}), vals,
                   {"w": good, "b": _arr(9)}),
            "one value per output channel")


# --- Concat ---------------------------------------------------------------

def test_concat_rejections():
    a, b = _t("a", (1, 3)), _t("b", (1, 4))
    vals = {"x": _t("x", (1, 3)), "a": a, "b": b, "y": _t("y", (1, 7))}
    _reject(_graph(Node("Concat", "cat", (), ("y",), {"axis": 1}), vals),
            "at least one input")
    _reject(_graph(Node("Concat", "cat", ("a", "gone"), ("y",), {"axis": 1}), vals),
            "no known shape")
    _reject(_graph(Node("Concat", "cat", ("a", "r3"), ("y",), {"axis": 1}),
                   {**vals, "r3": _t("r3", (1, 2, 3))}),
            "same rank")
    _reject(_graph(Node("Concat", "cat", ("a", "b"), ("y",), {"axis": 5}), vals),
            "out of range")
    _reject(_graph(Node("Concat", "cat", ("a", "wide"), ("y",), {"axis": 1}),
                   {**vals, "wide": _t("wide", (2, 4))}),
            "axis|differ")


# --- Add ------------------------------------------------------------------

def test_add_rejections():
    vals = {"x": _t("x", (1, 4)), "y": _t("y", (1, 4))}
    c = _arr(4)
    _reject(_graph(Node("Add", "a", ("x",), ("y",), {}), vals, {"c": c}),
            "exactly 2 inputs")
    _reject(_graph(Node("Add", "a", ("x", "z"), ("y",), {}),
                   {**vals, "z": _t("z", (1, 4))}, {}),
            "exactly one runtime operand")
    _reject(_graph(Node("Add", "a", ("x", "c"), ("gone",), {}), vals, {"c": c}),
            "must have inferred shapes")
    _reject(_graph(Node("Add", "a", ("x", "c"), ("y",), {}),
                   {"x": _t("x", (1, 2)), "y": _t("y", (1, 4))}, {"c": c}),
            "only the constant operand may broadcast")
    _reject(_graph(Node("Add", "a", ("x", "c"), ("y",), {}), vals, {"c": _arr(1, 1, 4)}),
            "wider than the output")
    _reject(_graph(Node("Add", "a", ("x", "c"), ("y",), {}), vals, {"c": _arr(3)}),
            "neither\n?matches|neither matches")


# --- LSTM -----------------------------------------------------------------

def _lstm_vals(hidden=2, inp=3, seq=2):
    return {"x": _t("x", (seq, 1, inp)), "y": _t("y", (seq, 1, 1, hidden))}


def _lstm(attrs=None, inputs=("x", "W", "R"), outputs=("y",), extra_vals=None, inits=None):
    vals = {**_lstm_vals(), **(extra_vals or {})}
    base = {"W": _arr(1, 8, 3), "R": _arr(1, 8, 2)}
    return _graph(Node("LSTM", "l", inputs, outputs, {"hidden_size": 2, **(attrs or {})}),
                  vals, {**base, **(inits or {})})


def test_lstm_attribute_rejections():
    _reject(_lstm({"direction": "bidirectional"}), "direction=")
    _reject(_lstm({"activations": ("Relu",) * 3}), "custom activations")
    for attr, val in (("clip", 1.0), ("input_forget", 1), ("layout", 1)):
        _reject(_lstm({attr: val}), f"{attr}=")


def test_lstm_input_rejections():
    _reject(_lstm(inputs=("x", "W", "R", "", "lens")), "sequence_lens")
    _reject(_lstm(inputs=("x", "W", "R", "", "", "", "", "P")), "peephole")
    g = _graph(Node("LSTM", "l", ("x", "W", "R"), ("y",), {"hidden_size": 2}),
               {"x": _t("x", (1, 4)), "y": _t("y", (1, 2))},
               {"W": _arr(1, 8, 3), "R": _arr(1, 8, 2)})
    _reject(g, "rank-3")
    _reject(_lstm(inputs=("x", "rt", "R"), extra_vals={"rt": _t("rt", (1, 8, 3))},
                  inits={"W": _arr(1, 8, 3)}), "must be a constant initializer")
    _reject(_lstm(inits={"W": _arr(8, 3)}), r"must have shape \(1, 4\*hidden")
    _reject(_lstm(inputs=("x", "W", "R", "B"), inits={"B": _arr(3, 3)}),
            r"B must be a constant of shape")


def test_lstm_initial_state_rejections():
    _reject(_lstm(inputs=("x", "W", "R", "", "", "h0"),
                  extra_vals={"h0": _t("h0", (1, 1, 2))}),
            "supplied together")
    _reject(_lstm(inputs=("x", "W", "R", "", "", "h0", "c0"),
                  extra_vals={"h0": _t("h0", (1, 2)), "c0": _t("c0", (1, 1, 2))}),
            "must be a rank-3")


# --- the global checks ----------------------------------------------------

def test_rank_and_dtype_rejections():
    _reject(_graph(Node("Relu", "r", ("x",), ("y",), {}),
                   {"x": _t("x", (1, 2, 3, 4, 5)), "y": _t("y", (1, 2, 3, 4, 5))}),
            "this generator handles rank 1 to 4")
    # An initializer no per-op rule claims: the per-op weight checks are
    # tighter and fire before the global one, so this reaches the global check.
    _reject(_graph(Node("Relu", "r", ("x",), ("y",), {}),
                   {"x": _t("x", (1, 3)), "y": _t("y", (1, 3))},
                   {"spare": np.ones((1, 2, 3, 4, 5), np.float32)}),
            "initializer 'spare' has rank 5")
    _reject(_graph(Node("Relu", "r", ("x",), ("y",), {}),
                   {"x": _t("x", (1, 3)), "y": _t("y", (1, 3))},
                   {"spare": np.ones((3, 2), np.int64)}),
            "only floating-point")


def test_an_unsupported_op_names_the_node_and_lists_what_is_supported():
    with pytest.raises(UnsupportedModel) as e:
        validate(_graph(Node("Softmax", "sm0", ("x",), ("y",), {"axis": 1}),
                        {"x": _t("x", (1, 3)), "y": _t("y", (1, 3))}))
    assert "sm0" in str(e.value) and "Softmax is not supported" in str(e.value)
    assert "Gemm" in str(e.value), "the message should list what it does handle"
