"""Several graph outputs, concatenated in y, and the Concat op.

The multi-output contract mirrors the multi-input one: every graph output
lands in the single y buffer, flat, in declaration order, so infer(x, y),
infer_batch, the native kernel and the device contract are unchanged. That
is what lets a recurrent model hand its new hidden and cell state back to
a solver that keeps them resident across time steps.
"""
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper, TensorProto

from rosenna.cli import main
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from tests.test_regressions import _both_backends, _reference


def _f32(rng, shape, name):
    return numpy_helper.from_array(rng.uniform(-1, 1, shape).astype(np.float32), name)


def _save(directory, name, nodes, inits, inputs, outputs):
    """Like conftest.save_model but with any number of inputs and outputs."""
    ins = [helper.make_tensor_value_info(n, TensorProto.FLOAT, list(sh)) for n, sh in inputs]
    outs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, list(sh)) for n, sh in outputs]
    g = helper.make_graph(nodes, name, ins, outs, initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    path = directory / f"{name}.onnx"
    onnx.save(m, path)
    return path


def _ort_concat(session, x_rows):
    """onnxruntime's outputs for each row of x, concatenated flat -- the y layout."""
    names = [i.name for i in session.get_inputs()]
    shapes = [[int(d) for d in i.shape] for i in session.get_inputs()]
    lens = [int(np.prod(sh)) for sh in shapes]
    out = []
    for row in x_rows:
        feed, off = {}, 0
        for n, sh, ln in zip(names, shapes, lens):
            feed[n] = row[off:off + ln].reshape(sh).astype(np.float32)
            off += ln
        out.append(np.concatenate([o.ravel() for o in session.run(None, feed)]))
    return np.array(out)


def test_two_outputs_land_concatenated_in_y(tmp_path):
    # y = [r (1,3), g (1,3)]: the activation's output first, then the
    # pre-activation it was computed from, in declaration order.
    rng = np.random.default_rng(1)
    w = _f32(rng, (2, 3), "w")
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["g"], name="mm0"),
        helper.make_node("Relu", ["g"], ["r"], name="r0"),
    ]
    path = _save(tmp_path, "two", nodes, [w], [("x", (1, 2))], [("r", (1, 3)), ("g", (1, 3))])
    plan = build_plan(load_graph(path), dtype="f64", embed=False)
    assert plan.output.shape == (6,)
    x = np.random.default_rng(2).uniform(-2, 2, (5, 2))
    expected = _ort_concat(ort.InferenceSession(str(path)), x)
    f_out, c_out = _both_backends(tmp_path, path, "two", x)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_recurrent_lstm_round_trips_its_state_through_x_and_y(tmp_path):
    # The model a solver keeps per cell: x = [p, h, c], y = [Y, h', c'].
    # Two steps are chained by feeding y's h'/c' back into x, and the second
    # step must match onnxruntime driven the same way.
    rng = np.random.default_rng(3)
    H, I = 4, 1
    W, R, B = _f32(rng, (1, 4 * H, I), "W"), _f32(rng, (1, 4 * H, H), "R"), _f32(rng, (1, 8 * H), "B")
    nodes = [helper.make_node("LSTM", ["p", "W", "R", "B", "", "h", "c"], ["Y", "hn", "cn"],
                              name="l0", hidden_size=H)]
    path = _save(tmp_path, "cell", nodes, [W, R, B],
                 [("p", (1, 1, I)), ("h", (1, 1, H)), ("c", (1, 1, H))],
                 [("Y", (1, 1, 1, H)), ("hn", (1, 1, H)), ("cn", (1, 1, H))])
    plan = build_plan(load_graph(path), dtype="f64", embed=True)
    assert plan.input.shape == (I + 2 * H,) and plan.output.shape == (3 * H,)
    session = ort.InferenceSession(str(path))
    x0 = np.random.default_rng(4).uniform(-1, 1, (3, I + 2 * H))
    y0 = _ort_concat(session, x0)
    x1 = np.concatenate([x0[:, :I], y0[:, H:2 * H], y0[:, 2 * H:]], axis=1)   # feed h', c' back
    expected = np.concatenate([y0, _ort_concat(session, x1)])
    f_out, c_out = _both_backends(tmp_path, path, "cell", np.concatenate([x0, x1]))
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_info_prints_the_x_and_y_layouts(tmp_path, capsys):
    rng = np.random.default_rng(3)
    H, I = 4, 1
    W, R = _f32(rng, (1, 4 * H, I), "W"), _f32(rng, (1, 4 * H, H), "R")
    nodes = [helper.make_node("LSTM", ["p", "W", "R", "", "", "h", "c"], ["Y", "hn", "cn"],
                              name="l0", hidden_size=H)]
    path = _save(tmp_path, "cell", nodes, [W, R],
                 [("p", (1, 1, I)), ("h", (1, 1, H)), ("c", (1, 1, H))],
                 [("Y", (1, 1, 1, H)), ("hn", (1, 1, H)), ("cn", (1, 1, H))])
    assert main(["info", str(path)]) == 0
    out = capsys.readouterr().out
    for line in ("x: p[0:1] h[1:5] c[5:9]", "y: Y[0:4] hn[4:8] cn[8:12]"):
        assert line in out, out


@pytest.mark.parametrize("axis", [1, 2])
def test_concat_of_runtime_values_along_an_axis(tmp_path, axis):
    # Two branches of x concatenated along a non-leading axis: the emitted
    # loop is outer x (block per input), resolved at generation time.
    rng = np.random.default_rng(5)
    w0, w1 = _f32(rng, (3, 3), "w0"), _f32(rng, (3, 3), "w1")
    nodes = [
        helper.make_node("MatMul", ["x", "w0"], ["a"], name="mm0"),
        helper.make_node("MatMul", ["x", "w1"], ["b"], name="mm1"),
        helper.make_node("Relu", ["b"], ["r"], name="r0"),
        helper.make_node("Concat", ["a", "r"], ["y"], name="c0", axis=axis),
    ]
    out_shape = (1, 4, 3) if axis == 1 else (1, 2, 6)
    path = _save(tmp_path, f"cat{axis}", nodes, [w0, w1], [("x", (1, 2, 3))], [("y", out_shape)])
    inputs, expected = _reference(path, [1, 2, 3])
    f_out, c_out = _both_backends(tmp_path, path, f"cat{axis}", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_concat_with_a_constant_operand(tmp_path):
    # A constant concatenated onto a runtime value (a bias-like feature
    # appended before a dense layer) becomes a weight.
    rng = np.random.default_rng(6)
    k = numpy_helper.from_array(np.array([[0.5, -1.5]], np.float32), "k")
    w = _f32(rng, (4, 2), "w")
    nodes = [
        helper.make_node("Concat", ["x", "k"], ["xk"], name="c0", axis=1),
        helper.make_node("Gemm", ["xk", "w"], ["y"], name="g0", transB=0),
    ]
    path = _save(tmp_path, "catk", nodes, [k, w], [("x", (1, 2))], [("y", (1, 2))])
    inputs, expected = _reference(path, [1, 2])
    f_out, c_out = _both_backends(tmp_path, path, "catk", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)
