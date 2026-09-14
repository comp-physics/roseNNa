"""Regressions from the review of the device-library branch (PR #12).

Every model here is built inline with onnx.helper; each one is a shape the
golden set does not contain: a tied initializer, an end-only padded pool, a
real Transpose in a graph with no Add, an Unsqueeze with several negative
axes, a (1,) Gemm bias, a weights file whose table of contents is short, and
a Constant LSTM initial state.
"""
import struct
import subprocess

import numpy as np
import pytest
from onnx import helper, numpy_helper, TensorProto

from rosenna.errors import UnsupportedModel
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from rosenna.weights import write_weights
from tests.conftest import save_model
from tests.test_regressions import _both_backends, _init_status, _reference


def _f32(rng, shape, name):
    return numpy_helper.from_array(rng.uniform(-1, 1, shape).astype(np.float32), name)


def test_tied_initializer_is_loaded_into_every_use(tmp_path):
    # x @ w -> Relu -> @ w: one initializer, two nodes. The plan made one
    # WeightSpec per use; C's loader filled the first and returned, leaving
    # the second all zeros, and Fortran's select-case had two labels for 'w'.
    rng = np.random.default_rng(3)
    w = _f32(rng, (3, 3), "w")
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["g"], name="mm0"),
        helper.make_node("Relu", ["g"], ["r"], name="r0"),
        helper.make_node("MatMul", ["r", "w"], ["y"], name="mm1"),
    ]
    path = save_model(tmp_path, "tied", nodes, [w], (1, 3), (1, 3))
    inputs, expected = _reference(path, [1, 3])
    f_out, c_out = _both_backends(tmp_path, path, "tied", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_average_pool_with_end_only_padding_divides_by_the_window_count(tmp_path):
    # kernel 2, stride 2, pads [0,0,1,1] on a 3x3 input: the last row and
    # column of windows hang off the end by one. ONNX (count_include_pad=0)
    # divides those by the cells that exist; the emitters divided every
    # window by the full kernel because the *begin* pads were zero.
    nodes = [helper.make_node("AveragePool", ["x"], ["y"], name="p0",
                              kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 1, 1])]
    path = save_model(tmp_path, "endpad", nodes, [], (1, 1, 3, 3), (1, 1, 2, 2))
    inputs, expected = _reference(path, [1, 1, 3, 3])
    f_out, c_out = _both_backends(tmp_path, path, "endpad", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_real_transpose_compiles_in_fortran_without_an_add(tmp_path):
    # Transpose(perm=[0,2,1]) on (1,3,4) moves a real axis, so it is a loop
    # over counters c0, c1, ...; the Fortran emitter only declared those
    # when the model also had an Add.
    rng = np.random.default_rng(5)
    w = _f32(rng, (3, 2), "w")
    nodes = [
        helper.make_node("Transpose", ["x"], ["t"], name="t0", perm=[0, 2, 1]),
        helper.make_node("MatMul", ["t", "w"], ["y"], name="mm0"),
    ]
    path = save_model(tmp_path, "tr", nodes, [w], (1, 3, 4), (1, 4, 2))
    inputs, expected = _reference(path, [1, 3, 4])
    f_out, c_out = _both_backends(tmp_path, path, "tr", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_unsqueeze_with_several_negative_axes_folds_to_the_right_shape(tmp_path):
    # Unsqueeze(c[3], axes=[-1, -2]) is (3,1,1) -- negatives count from the
    # OUTPUT rank -- and the fold resolved them against ndim+1, giving
    # (1,1,3). Added to x of shape (1,3,1,3) both broadcast, so a wrong fold
    # is a silently wrong answer, not an error.
    c = numpy_helper.from_array(np.array([1.0, 2.0, 3.0], np.float32), "c")
    axes = numpy_helper.from_array(np.array([-1, -2], np.int64), "axes")
    nodes = [
        helper.make_node("Unsqueeze", ["c", "axes"], ["cu"], name="u0"),
        helper.make_node("Add", ["x", "cu"], ["y"], name="a0"),
    ]
    path = save_model(tmp_path, "unsq", nodes, [c, axes], (1, 3, 1, 3), (1, 3, 1, 3))
    inputs, expected = _reference(path, [1, 3, 1, 3])
    f_out, c_out = _both_backends(tmp_path, path, "unsq", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_gemm_bias_of_length_one_is_rejected(tmp_path):
    # A (1,) bias is a legal ONNX unidirectional broadcast, but the emitters
    # index b[i] for every output, so validation must refuse it rather than
    # let generated code read past a one-element array.
    rng = np.random.default_rng(7)
    w = _f32(rng, (2, 3), "w")
    b = numpy_helper.from_array(np.array([0.5], np.float32), "b")
    nodes = [helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="g0")]
    path = save_model(tmp_path, "b1", nodes, [w, b], (1, 2), (1, 3))
    with pytest.raises(UnsupportedModel, match="bias"):
        build_plan(load_graph(path), dtype="f64", embed=False)


@pytest.mark.parametrize("lang", ["c", "fortran"])
def test_weights_file_missing_a_tensor_returns_status_9(tmp_path, golden_model, lang):
    # A file with the right hash whose table of contents lists fewer tensors
    # than the plan has: init returned 0 and infer ran on zeroed arrays.
    onnx_path = golden_model("gemm_small")
    graph = load_graph(onnx_path)
    plan = build_plan(graph, dtype="f64", embed=False)
    good = tmp_path / "good.rwt"
    write_weights(plan, graph, good)
    data = good.read_bytes()
    # Header layout is write_weights' business; the count is the one
    # little-endian uint32 that equals the number of plan weights and sits
    # in the first 64 bytes.
    n = len(plan.weights)
    off = next(i for i in range(0, 64, 4) if struct.unpack_from("<I", data, i)[0] == n)
    short = bytearray(data)
    struct.pack_into("<I", short, off, 0)
    rc, out = _init_status(tmp_path, lang, onnx_path, "gemm_small", bytes(short))
    assert rc == 0, f"generated init aborted instead of returning a status (exit {rc})"
    assert out.split() == ["9"], out


def test_constant_lstm_initial_state_is_lowered_to_weights(tmp_path):
    # fold.py's docstring says a Constant initial_h/initial_c is folded away
    # and supported; after folding they are initializers, and the plan
    # refused them as "must be a rank-3 value". They are now weights.
    rng = np.random.default_rng(9)
    hidden, n_in = 4, 3
    W = _f32(rng, (1, 4 * hidden, n_in), "W")
    R = _f32(rng, (1, 4 * hidden, hidden), "R")
    B = _f32(rng, (1, 8 * hidden), "B")
    h0 = numpy_helper.from_array(rng.uniform(-1, 1, (1, 1, hidden)).astype(np.float32), "h0")
    c0 = numpy_helper.from_array(rng.uniform(-1, 1, (1, 1, hidden)).astype(np.float32), "c0")
    nodes = [
        helper.make_node("Constant", [], ["h0"], name="k0", value=h0),
        helper.make_node("Constant", [], ["c0"], name="k1", value=c0),
        helper.make_node("LSTM", ["x", "W", "R", "B", "", "h0", "c0"], ["yl"], name="l0",
                         hidden_size=hidden),
        helper.make_node("Flatten", ["yl"], ["y"], name="f0"),
    ]
    path = save_model(tmp_path, "lstmk", nodes, [W, R, B], (1, 1, n_in), (1, hidden))
    plan = build_plan(load_graph(path), dtype="f64", embed=False)
    lstm = next(op for op in plan.ops if op.kind == "lstm")
    assert lstm.init_syms and not lstm.extra_in
    assert {w.name for w in plan.weights} >= {"h0", "c0"}
    inputs, expected = _reference(path, [1, 1, n_in])
    f_out, c_out = _both_backends(tmp_path, path, "lstmk", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_lstm_without_its_y_output_is_refused_not_a_traceback(tmp_path):
    # ONNX lets a graph ask for Y_h alone (outputs ["", "yh"]). The plan reads
    # the first output's shape and raised KeyError('') on the empty name; the
    # answer is an UnsupportedModel naming what is missing.
    rng = np.random.default_rng(9)
    hidden, n_in = 4, 3
    W = _f32(rng, (1, 4 * hidden, n_in), "W")
    R = _f32(rng, (1, 4 * hidden, hidden), "R")
    nodes = [
        helper.make_node("LSTM", ["x", "W", "R"], ["", "yh"], name="l0", hidden_size=hidden),
        helper.make_node("Flatten", ["yh"], ["y"], name="f0"),
    ]
    path = save_model(tmp_path, "lstmy", nodes, [W, R], (1, 1, n_in), (1, hidden))
    with pytest.raises(UnsupportedModel, match="Y"):
        build_plan(load_graph(path), dtype="f64", embed=False)
