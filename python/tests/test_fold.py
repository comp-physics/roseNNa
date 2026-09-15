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
