import numpy as np
import pytest
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
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
