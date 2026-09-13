import numpy as np
import pytest
from rosenna.frontend import load_graph, UnsupportedModel


def test_loads_nodes_in_order(golden_model):
    model_path = golden_model("gemm_small")
    g = load_graph(model_path)
    assert [n.op for n in g.nodes] == ["Gemm", "Relu", "Gemm", "Relu"]
    assert g.name == "gemm_small"

def test_every_shape_is_literal(golden_model):
    model_path = golden_model("gemm_small")
    g = load_graph(model_path)
    for t in g.values.values():
        assert all(isinstance(d, int) for d in t.shape), t

def test_input_and_output_shapes(golden_model):
    model_path = golden_model("gemm_small")
    g = load_graph(model_path)
    assert g.values[g.inputs[0]].shape == (1, 2)
    assert g.values[g.outputs[0]].shape == (1, 3)

def test_initializers_are_arrays(golden_model):
    model_path = golden_model("gemm_small")
    g = load_graph(model_path)
    w = g.initializers["linear_relu_stack.0.weight"]
    assert isinstance(w, np.ndarray) and w.shape == (2, 2)

def test_attrs_decoded(golden_model):
    model_path = golden_model("gemm_small")
    g = load_graph(model_path)
    gemm = g.nodes[0]
    assert gemm.attrs["transB"] == 1
    assert gemm.attrs["alpha"] == pytest.approx(1.0)

def test_symbolic_dim_is_rejected(tmp_path):
    import onnx
    from onnx import helper, TensorProto
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])
    node = helper.make_node("Relu", ["x"], ["y"])
    m = helper.make_model(helper.make_graph([node], "dyn", [x], [y]))
    p = tmp_path / "dyn.onnx"
    onnx.save(m, p)
    with pytest.raises(UnsupportedModel, match="batch"):
        load_graph(p)

def test_unsupported_attribute_names_node(tmp_path):
    import onnx
    from onnx import helper, TensorProto, AttributeProto
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Relu", ["x"], ["y"], name="test_relu")
    # Add an unsupported TENSOR attribute
    tensor_attr = AttributeProto()
    tensor_attr.name = "bad_attr"
    tensor_attr.type = AttributeProto.TENSOR
    node.attribute.append(tensor_attr)
    m = helper.make_model(helper.make_graph([node], "test", [x], [y]))
    p = tmp_path / "bad_attr.onnx"
    onnx.save(m, p)
    with pytest.raises(UnsupportedModel, match="test_relu"):
        load_graph(p)
