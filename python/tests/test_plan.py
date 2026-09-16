import pytest
from rosenna.frontend import load_graph, UnsupportedModel
from rosenna.plan import build_plan


def test_ops_lowered_in_order(golden_model):
    p = build_plan(load_graph(golden_model("gemm_small")))
    assert [o.kind for o in p.ops] == ["gemm", "relu", "gemm", "relu"]


def test_gemm_extents_and_weights(golden_model):
    p = build_plan(load_graph(golden_model("gemm_small")))
    g0 = p.ops[0]
    assert (g0.n_in, g0.n_out) == (2, 2)
    assert g0.weight == "w0" and g0.bias == "b0"
    g1 = p.ops[2]
    assert (g1.n_in, g1.n_out) == (2, 3)


def test_values_flatten_to_rank_one(golden_model):
    p = build_plan(load_graph(golden_model("gemm_small")))
    assert p.input.shape == (2,) and p.output.shape == (3,)


def test_weight_offsets_are_packed_in_order(golden_model):
    p = build_plan(load_graph(golden_model("gemm_small")))
    assert [w.symbol for w in p.weights] == ["w0", "b0", "w1", "b1"]
    offset = 0
    for w in p.weights:
        assert w.offset == offset
        offset += w.nbytes


def test_buffers_are_reused(golden_model):
    p = build_plan(load_graph(golden_model("batchnet")))
    assert len(p.buffers) <= 4     # x, y, and at most two scratch buffers


def test_hash_is_stable_and_shape_sensitive(golden_model):
    a = build_plan(load_graph(golden_model("gemm_small"))).hash()
    b = build_plan(load_graph(golden_model("gemm_small"))).hash()
    c = build_plan(load_graph(golden_model("gemm_big"))).hash()
    assert a == b and a != c
    assert len(a) == 64


def test_nobias_gemm_from_matmul(golden_model):
    p = build_plan(load_graph(golden_model("gemm_nobias")))
    assert [o.kind for o in p.ops] == ["gemm", "gemm"]
    assert all(o.bias is None for o in p.ops)


def test_multi_output_graph_concatenates_its_outputs_in_y(tmp_path):
    # Several graph outputs are the mirror of several inputs: y is the
    # concatenation in declaration order, and each secondary output is a
    # copy into its slice of y after the last op.
    import onnx
    from onnx import helper, TensorProto
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
    n1 = helper.make_node("Relu", ["x"], ["y"], name="r1")
    n2 = helper.make_node("Relu", ["x"], ["z"], name="r2")
    g = helper.make_graph([n1, n2], "multi_output", [x], [y, z])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    p = tmp_path / "multi_output.onnx"
    onnx.save(m, p)
    plan = build_plan(load_graph(p))
    assert plan.output.shape == (4,)
    gather = [op for op in plan.ops if op.kind == "copy" and op.dst_offset]
    assert [(op.inp, op.dst_offset, op.n_out) for op in gather] == [("z", 2, 2)]
    assert plan.assignment[gather[0].out] == "y"
