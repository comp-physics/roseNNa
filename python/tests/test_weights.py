import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper, TensorProto
from rosenna.frontend import load_graph, UnsupportedModel
from rosenna.plan import build_plan
from rosenna.validate import validate
from rosenna.weights import write_weights, read_weights, MAGIC


def _write(tmp_path, golden_model, name="gemm_small"):
    g = load_graph(golden_model(name))
    p = build_plan(g)
    out = tmp_path / f"{name}.rwt"
    write_weights(p, g, out)
    return p, g, out


def test_round_trips_values(tmp_path, golden_model):
    plan, graph, path = _write(tmp_path, golden_model)
    tensors, header = read_weights(path)
    for w in plan.weights:
        np.testing.assert_allclose(tensors[w.name], graph.initializers[w.name], rtol=0, atol=0)
    assert header["plan_hash"] == plan.hash()
    assert header["count"] == len(plan.weights)


def test_starts_with_magic(tmp_path, golden_model):
    _, _, path = _write(tmp_path, golden_model)
    assert path.read_bytes()[:8] == MAGIC


def test_offsets_match_the_plan(tmp_path, golden_model):
    plan, _, path = _write(tmp_path, golden_model)
    blob = path.read_bytes()
    total = sum(w.nbytes for w in plan.weights)
    assert len(blob) > total
    assert plan.weights[-1].offset + plan.weights[-1].nbytes == total


def test_dtype_is_honoured(tmp_path, golden_model):
    g = load_graph(golden_model("gemm_small"))
    for dtype, itemsize in (("f32", 4), ("f64", 8)):
        p = build_plan(g, dtype=dtype)
        out = tmp_path / f"{dtype}.rwt"
        write_weights(p, g, out)
        tensors, header = read_weights(out)
        assert header["dtype"] == dtype
        assert tensors["linear_relu_stack.0.weight"].itemsize == itemsize


def test_truncated_weights_file_in_toc(tmp_path, golden_model):
    """Truncate mid-table-of-contents and verify descriptive error."""
    plan, graph, path = _write(tmp_path, golden_model)
    blob = path.read_bytes()
    # Truncate in the middle of the TOC (around byte 100, which is well before data_start)
    truncated_path = tmp_path / "truncated_toc.rwt"
    truncated_path.write_bytes(blob[:100])
    with pytest.raises(ValueError) as exc_info:
        read_weights(truncated_path)
    error_msg = str(exc_info.value)
    assert str(truncated_path) in error_msg, f"File path not in error: {error_msg}"


def test_truncated_weights_file_in_data(tmp_path, golden_model):
    """Truncate mid-data section and verify descriptive error."""
    plan, graph, path = _write(tmp_path, golden_model)
    blob = path.read_bytes()
    # Truncate at 80% of file size (well into the data section)
    truncated_path = tmp_path / "truncated_data.rwt"
    truncated_path.write_bytes(blob[:int(len(blob) * 0.8)])
    with pytest.raises(ValueError) as exc_info:
        read_weights(truncated_path)
    error_msg = str(exc_info.value)
    assert str(truncated_path) in error_msg, f"File path not in error: {error_msg}"


def _model_with_init(tmp_path, init_dtype, init_name="w"):
    """Helper to create a model with a specific initializer dtype."""
    w = numpy_helper.from_array(np.zeros((2, 2), dtype=init_dtype), init_name)
    n = helper.make_node("MatMul", ["x", init_name], ["y"], name="mm1")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    g = helper.make_graph([n], "t", [x], [y], initializer=[w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    p = tmp_path / "t.onnx"
    onnx.save(m, p)
    return load_graph(p)


def test_non_float_initializer_rejected(tmp_path):
    """Integer initializer should be rejected at validate time."""
    g = _model_with_init(tmp_path, np.int64)
    with pytest.raises(UnsupportedModel) as exc_info:
        validate(g)
    error_msg = str(exc_info.value)
    assert "w" in error_msg, f"Initializer name not in error: {error_msg}"
    assert "dtype" in error_msg.lower() or "int" in error_msg.lower(), f"dtype info not in error: {error_msg}"


def test_all_golden_models_validate_and_write(tmp_path, golden_model):
    """Guard: all five dense golden models still validate and write successfully."""
    models = ["gemm_small", "gemm_big", "gemm_nobias", "droplet", "batchnet"]
    for model_name in models:
        g = load_graph(golden_model(model_name))
        validate(g)  # Should not raise
        p = build_plan(g)
        out = tmp_path / f"{model_name}.rwt"
        write_weights(p, g, out)  # Should not raise
        tensors, header = read_weights(out)
        assert len(tensors) == len(p.weights)
        assert header["plan_hash"] == p.hash()


def test_unknown_dtype_code_is_a_named_value_error(tmp_path, golden_model):
    """An out-of-range dtype code must name the file and the code, not KeyError."""
    _, _, path = _write(tmp_path, golden_model)
    blob = bytearray(path.read_bytes())
    blob[12:16] = (99).to_bytes(4, "little", signed=True)   # dtype code field
    bad = tmp_path / "bad_dtype.rwt"
    bad.write_bytes(bytes(blob))
    with pytest.raises(ValueError) as exc_info:
        read_weights(bad)
    msg = str(exc_info.value)
    assert str(bad) in msg
    assert "99" in msg
