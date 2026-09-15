"""The constant-folding evaluators, exercised directly.

`fold.py` runs at generation time on nodes whose inputs are all constants, so
nothing it computes is ever checked against onnxruntime -- a wrong fold is a
wrong *weight*, and every later comparison agrees with it. The golden models
reach only the two shapes a PyTorch export happens to emit (a `Reshape` of a
weight, a `Constant`), which left most of this file untested.

These call the evaluators on arrays whose right answer is obvious by
inspection, and pin the refusals for the shapes it cannot fold.
"""
import numpy as np
import pytest

from rosenna.errors import UnsupportedModel
from rosenna.fold import _evaluate, _resolve_shape, fold_constants
from rosenna.frontend import Graph, Node, Tensor


def _node(op, inputs, outputs=("out",), **attrs):
    return Node(op, f"{op.lower()}0", tuple(inputs), tuple(outputs), attrs)


# --- Reshape target semantics ---------------------------------------------

def test_resolve_shape_copies_a_zero_from_the_input():
    assert _resolve_shape([0, 3], (4, 6)) == (4, 3)


def test_resolve_shape_infers_a_single_minus_one():
    assert _resolve_shape([-1, 3], (4, 6)) == (8, 3)
    assert _resolve_shape([2, -1], (4, 6)) == (2, 12)


def test_resolve_shape_refuses_a_zero_with_no_matching_axis():
    with pytest.raises(UnsupportedModel, match="no matching input axis"):
        _resolve_shape([3, 0], (4,))


def test_resolve_shape_refuses_two_inferred_axes():
    with pytest.raises(UnsupportedModel, match="at most one -1"):
        _resolve_shape([-1, -1], (4, 6))


def test_resolve_shape_refuses_an_indivisible_inference():
    with pytest.raises(UnsupportedModel, match="cannot infer"):
        _resolve_shape([-1, 5], (4, 6))


# --- the evaluators -------------------------------------------------------

def test_identity_returns_its_input():
    a = np.arange(6.0).reshape(2, 3)
    assert np.array_equal(_evaluate(_node("Identity", ["a"]), {"a": a}), a)


def test_reshape_uses_the_constant_shape():
    a = np.arange(12.0).reshape(3, 4)
    inits = {"a": a, "s": np.array([2, 6], np.int64)}
    assert _evaluate(_node("Reshape", ["a", "s"]), inits).shape == (2, 6)


def test_reshape_refuses_a_runtime_shape():
    with pytest.raises(UnsupportedModel, match="needs a constant shape"):
        _evaluate(_node("Reshape", ["a"]), {"a": np.zeros((2, 2))})


def test_transpose_honours_perm_and_defaults_to_reversing():
    a = np.arange(24.0).reshape(2, 3, 4)
    assert _evaluate(_node("Transpose", ["a"], perm=(1, 0, 2)), {"a": a}).shape == (3, 2, 4)
    assert _evaluate(_node("Transpose", ["a"]), {"a": a}).shape == (4, 3, 2)


def test_squeeze_takes_axes_from_the_attribute_or_from_input_one():
    a = np.zeros((1, 3, 1, 4))
    assert _evaluate(_node("Squeeze", ["a"], axes=(0, 2)), {"a": a}).shape == (3, 4)
    inits = {"a": a, "ax": np.array([0], np.int64)}
    assert _evaluate(_node("Squeeze", ["a", "ax"]), inits).shape == (3, 1, 4)
    # No axes at all: every length-1 axis goes.
    assert _evaluate(_node("Squeeze", ["a"]), {"a": a}).shape == (3, 4)


def test_squeeze_refuses_runtime_axes():
    with pytest.raises(UnsupportedModel, match="axes must be constant"):
        _evaluate(_node("Squeeze", ["a", "runtime"]), {"a": np.zeros((1, 3))})


def test_unsqueeze_inserts_each_axis():
    a = np.zeros((3, 4))
    inits = {"a": a, "ax": np.array([0], np.int64)}
    assert _evaluate(_node("Unsqueeze", ["a", "ax"]), inits).shape == (1, 3, 4)


def test_flatten_splits_at_its_axis():
    a = np.arange(24.0).reshape(2, 3, 4)
    assert _evaluate(_node("Flatten", ["a"], axis=1), {"a": a}).shape == (2, 12)
    assert _evaluate(_node("Flatten", ["a"], axis=2), {"a": a}).shape == (6, 4)
    # axis 0 means "one row", not "no rows".
    assert _evaluate(_node("Flatten", ["a"], axis=0), {"a": a}).shape == (1, 24)


def test_concat_of_constants_folds():
    inits = {"a": np.zeros((2, 3)), "b": np.ones((2, 5))}
    assert _evaluate(_node("Concat", ["a", "b"], axis=1), inits).shape == (2, 8)


# --- the pass itself ------------------------------------------------------

def test_folding_removes_the_node_and_its_now_unused_shape_tensor():
    a = np.arange(12.0, dtype=np.float32).reshape(3, 4)
    graph = Graph(
        "g",
        (_node("Reshape", ["a", "s"], ("folded",)),
         Node("Relu", "r0", ("folded",), ("y",), {})),
        {"folded": Tensor("folded", (2, 6), "f32"), "y": Tensor("y", (2, 6), "f32")},
        {"a": a, "s": np.array([2, 6], np.int64)},
        (), ("y",))
    out = fold_constants(graph)
    assert [n.op for n in out.nodes] == ["Relu"], "the Reshape should be gone"
    assert out.initializers["folded"].shape == (2, 6)
    assert "folded" not in out.values, "a folded value is an initializer, not a value"
    assert "s" not in out.initializers, "the int64 shape tensor is unreferenced now"


def test_a_constant_node_without_a_readable_value_is_refused():
    graph = Graph("g", (Node("Constant", "c0", (), ("v",), {}),),
                  {"v": Tensor("v", (1,), "f32")}, {}, (), ("v",))
    with pytest.raises(UnsupportedModel, match="Constant without a value"):
        fold_constants(graph)


def test_a_constant_node_folds_from_each_value_attribute():
    for attr, payload in (("value", np.array([1.0, 2.0], np.float32)),
                          ("value_floats", (1.0, 2.0)),
                          ("value_ints", (1, 2))):
        graph = Graph("g", (Node("Constant", "c0", (), ("v",), {attr: payload}),),
                      {"v": Tensor("v", (2,), "f32")}, {}, (), ("v",))
        out = fold_constants(graph)
        assert out.nodes == (), f"{attr}: the Constant should be folded away"
        assert out.initializers["v"].shape == (2,)


# --- BatchNormalization folding -------------------------------------------
#
# The fold is what makes BatchNormalization work at all: there is no loop nest
# for it in either emitter, so if the pass stops firing the op does not get
# slower, it stops being supported. These check both that it fires and that
# the arithmetic it folds is right.

def _bn(name, x, out, chan, dtype=np.float32, **attrs):
    """A BatchNormalization node plus its four constant parameters."""
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    params = {
        f"{name}_scale": rng.uniform(0.5, 2.0, chan).astype(dtype),
        f"{name}_B": rng.uniform(-1, 1, chan).astype(dtype),
        f"{name}_mean": rng.uniform(-1, 1, chan).astype(dtype),
        f"{name}_var": rng.uniform(0.5, 2.0, chan).astype(dtype),
    }
    node = Node("BatchNormalization", name, (x,) + tuple(params), (out,), attrs)
    return node, params


def _conv_bn_graph(**bn_attrs):
    """Conv(3->4, 3x3) -> BatchNormalization, as a Graph ready for the pass."""
    rng = np.random.default_rng(1)
    w = rng.uniform(-1, 1, (4, 3, 3, 3)).astype(np.float32)
    b = rng.uniform(-1, 1, 4).astype(np.float32)
    conv = Node("Conv", "c0", ("x", "w", "b"), ("h",),
                {"kernel_shape": (3, 3), "pads": (1, 1, 1, 1)})
    bn, params = _bn("bn0", "h", "y", 4, **bn_attrs)
    return Graph("g", (conv, bn),
                 {"x": Tensor("x", (1, 3, 8, 8), "f32"),
                  "h": Tensor("h", (1, 4, 8, 8), "f32"), "y": Tensor("y", (1, 4, 8, 8), "f32")},
                 {"w": w, "b": b, **params}, ("x",), ("y",))


def test_batchnorm_folds_into_a_conv_and_matches_the_reference_arithmetic():
    from rosenna.fold import fold_batchnorm
    g = _conv_bn_graph()
    w0, b0 = g.initializers["w"].copy(), g.initializers["b"].copy()
    scale, shift = g.initializers["bn0_scale"], g.initializers["bn0_B"]
    mean, var = g.initializers["bn0_mean"], g.initializers["bn0_var"]

    out = fold_batchnorm(g)
    assert [n.op for n in out.nodes] == ["Conv"], "the BatchNormalization should be gone"
    assert out.nodes[0].outputs == ("y",), "the Conv takes over the BN's output"
    assert "h" not in out.values, "the value between them stops existing"

    s = scale / np.sqrt(var + 1e-5)
    assert np.allclose(out.initializers["w"], w0 * s.reshape(-1, 1, 1, 1))
    assert np.allclose(out.initializers[out.nodes[0].inputs[2]], (b0 - mean) * s + shift)
    # The BN's own parameters are unreferenced now and must not ship.
    assert not any(k.startswith("bn0_") for k in out.initializers), sorted(out.initializers)


@pytest.mark.parametrize("trans_b", [0, 1])
def test_batchnorm_folds_into_a_gemm_on_either_weight_layout(trans_b):
    """transB decides which axis of the weight the channel scale broadcasts along."""
    from rosenna.fold import fold_batchnorm
    rng = np.random.default_rng(2)
    shape = (5, 6) if trans_b else (6, 5)
    w = rng.uniform(-1, 1, shape).astype(np.float32)
    gemm = Node("Gemm", "g0", ("x", "w"), ("h",), {"transB": trans_b})
    bn, params = _bn("bn1", "h", "y", 5)
    g = Graph("g", (gemm, bn),
              {"h": Tensor("h", (1, 5), "f32"), "y": Tensor("y", (1, 5), "f32")},
              {"w": w, **params}, ("x",), ("y",))
    out = fold_batchnorm(g)
    assert [n.op for n in out.nodes] == ["Gemm"]
    s = params["bn1_scale"] / np.sqrt(params["bn1_var"] + 1e-5)
    want = w * (s.reshape(-1, 1) if trans_b else s.reshape(1, -1))
    assert np.allclose(out.initializers["w"], want)
    # The Gemm had no bias; the shift is not optional, so it gains one.
    assert len(out.nodes[0].inputs) == 3
    assert np.allclose(out.initializers[out.nodes[0].inputs[2]],
                       (0 - params["bn1_mean"]) * s + params["bn1_B"])


@pytest.mark.parametrize("why,mutate", [
    ("training mode", lambda g: _retag(g, {"training_mode": 1})),
    ("a second reader of the intermediate", lambda g: _add_consumer(g)),
    ("the intermediate is a graph output", lambda g: g._replace(outputs=("y", "h"))
     if hasattr(g, "_replace") else _also_output(g)),
])
def test_batchnorm_is_left_in_place_when_folding_would_change_the_model(why, mutate):
    """Each of these makes the fold unsound, so the op survives and is refused."""
    from rosenna.fold import fold_batchnorm
    from rosenna.validate import validate
    g = mutate(_conv_bn_graph())
    out = fold_batchnorm(g)
    assert any(n.op == "BatchNormalization" for n in out.nodes), why
    with pytest.raises(UnsupportedModel, match="could not be folded"):
        validate(out)


def _retag(g, attrs):
    nodes = tuple(Node(n.op, n.name, n.inputs, n.outputs, {**n.attrs, **attrs})
                  if n.op == "BatchNormalization" else n for n in g.nodes)
    return Graph(g.name, nodes, g.values, g.initializers, g.inputs, g.outputs)


def _add_consumer(g):
    extra = Node("Relu", "r0", ("h",), ("z",), {})
    return Graph(g.name, g.nodes + (extra,), {**g.values, "z": Tensor("z", (1, 4, 8, 8), "f32")},
                 g.initializers, g.inputs, g.outputs + ("z",))


def _also_output(g):
    return Graph(g.name, g.nodes, g.values, g.initializers, g.inputs, ("y", "h"))
