import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper, TensorProto
from rosenna.frontend import load_graph, UnsupportedModel
from rosenna.validate import validate

def _model(tmp_path, nodes, inits, in_shape=(1, 2), out_shape=(1, 2)):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(in_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))
    g = helper.make_graph(nodes, "t", [x], [y], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    p = tmp_path / "t.onnx"
    onnx.save(m, p)
    return load_graph(p)

def test_accepts_supported_dense_models(golden_model):
    for name in ["gemm_small", "gemm_big", "gemm_nobias", "droplet", "batchnet"]:
        validate(load_graph(golden_model(name)))

def test_rejects_unsupported_op(tmp_path):
    n = helper.make_node("Softmax", ["x"], ["y"], name="soft1")
    g = _model(tmp_path, [n], [])
    with pytest.raises(UnsupportedModel, match="soft1.*Softmax"):
        validate(g)

def test_rejects_gemm_transa(tmp_path):
    w = numpy_helper.from_array(np.zeros((2, 2), np.float32), "w")
    n = helper.make_node("Gemm", ["x", "w"], ["y"], name="g1", transA=1)
    with pytest.raises(UnsupportedModel, match="g1.*transA"):
        validate(_model(tmp_path, [n], [w]))

def test_rejects_gemm_alpha(tmp_path):
    w = numpy_helper.from_array(np.zeros((2, 2), np.float32), "w")
    n = helper.make_node("Gemm", ["x", "w"], ["y"], name="g1", alpha=2.0)
    with pytest.raises(UnsupportedModel, match="g1.*alpha"):
        validate(_model(tmp_path, [n], [w]))

def test_rejects_matmul_with_runtime_rhs(tmp_path):
    n1 = helper.make_node("Relu", ["x"], ["r"], name="r1")
    n2 = helper.make_node("MatMul", ["x", "r"], ["y"], name="mm1")
    with pytest.raises(UnsupportedModel, match="mm1"):
        validate(_model(tmp_path, [n1, n2], []))
