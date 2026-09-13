from rosenna.frontend import load_graph
from rosenna.plan import build_plan


def _plan(name):
    return build_plan(load_graph(f"../goldenFiles/{name}/{name}.onnx"))


def test_ops_lowered_in_order():
    p = _plan("gemm_small")
    assert [o.kind for o in p.ops] == ["gemm", "relu", "gemm", "relu"]


def test_gemm_extents_and_weights():
    p = _plan("gemm_small")
    g0 = p.ops[0]
    assert (g0.n_in, g0.n_out) == (2, 2)
    assert g0.weight == "w0" and g0.bias == "b0"
    g1 = p.ops[2]
    assert (g1.n_in, g1.n_out) == (2, 3)


def test_values_flatten_to_rank_one():
    p = _plan("gemm_small")
    assert p.input.shape == (2,) and p.output.shape == (3,)


def test_weight_offsets_are_packed_in_order():
    p = _plan("gemm_small")
    assert [w.symbol for w in p.weights] == ["w0", "b0", "w1", "b1"]
    offset = 0
    for w in p.weights:
        assert w.offset == offset
        offset += w.nbytes


def test_buffers_are_reused():
    p = _plan("batchnet")          # 6 Gemm + 5 Relu
    assert len(p.buffers) <= 4     # x, y, and at most two scratch buffers


def test_hash_is_stable_and_shape_sensitive():
    a = _plan("gemm_small").hash()
    b = _plan("gemm_small").hash()
    c = _plan("gemm_big").hash()
    assert a == b and a != c
    assert len(a) == 64


def test_nobias_gemm_from_matmul():
    p = _plan("gemm_nobias")
    assert [o.kind for o in p.ops] == ["gemm", "gemm"]
    assert all(o.bias is None for o in p.ops)
