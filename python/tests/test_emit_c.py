import subprocess
import numpy as np
import onnxruntime as ort
import pytest
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from rosenna.weights import write_weights
from rosenna.emit_c import emit_c
from tests.test_emit_fortran import _live_reference, _build_and_run as _fortran_build_and_run

DENSE = ["gemm_small", "gemm_big", "gemm_nobias", "droplet", "batchnet"]


def _build_and_run(tmp_path, onnx_path, name, inputs, dtype="f64"):
    graph = load_graph(onnx_path)
    plan = build_plan(graph, dtype=dtype)
    source, header = emit_c(plan)
    (tmp_path / f"{name}.c").write_text(source)
    (tmp_path / f"{name}.h").write_text(header)
    write_weights(plan, graph, tmp_path / f"{name}.rwt")
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    c_type = "double" if dtype == "f64" else "float"
    fmt = "%lf" if dtype == "f64" else "%f"
    (tmp_path / "main.c").write_text(f"""
#include <stdio.h>
#include "{name}.h"
int main(void) {{
    {c_type} x[{n_in}], y[{n_out}];
    int ncases, status = {name}_init("{name}.rwt");
    if (status != 0) {{ printf("init status %d\\n", status); return 1; }}
    if (scanf("%d", &ncases) != 1) return 1;
    for (int c = 0; c < ncases; ++c) {{
        for (int i = 0; i < {n_in}; ++i) if (scanf("{fmt}", &x[i]) != 1) return 1;
        {name}_infer(x, y);
        for (int i = 0; i < {n_out}; ++i) printf("%.17e ", (double)y[i]);
        printf("\\n");
    }}
    return 0;
}}
""")
    subprocess.run(["gcc", "-O2", "-Wall", "-Wextra", "-std=c11", "-o", "run",
                    f"{name}.c", "main.c", "-lm"], cwd=tmp_path, check=True,
                   capture_output=True, text=True)
    stdin = f"{len(inputs)}\n" + "\n".join(" ".join(repr(float(v)) for v in row) for row in inputs)
    out = subprocess.run(["./run"], cwd=tmp_path, input=stdin, capture_output=True,
                         text=True, check=True).stdout
    return np.array([[float(v) for v in line.split()] for line in out.strip().splitlines()])


@pytest.mark.parametrize("name", DENSE)
def test_matches_onnxruntime(tmp_path, golden_model, name):
    onnx_path = golden_model(name)
    session = ort.InferenceSession(str(onnx_path))
    shape = session.get_inputs()[0].shape
    inputs, expected = _live_reference(session, shape, np.float64)
    if inputs is None:
        pytest.skip(f"{name}: onnxruntime reference is all-zero across 10 resampled "
                    f"batches; its golden-file weights produced a dead model")
    got = _build_and_run(tmp_path, onnx_path, name, inputs)
    for produced, exp_row in zip(got, expected):
        np.testing.assert_allclose(produced, exp_row, rtol=1e-5, atol=1e-6)


def test_matches_onnxruntime_f32(tmp_path, golden_model):
    # See test_emit_fortran.test_matches_onnxruntime_f32: fp32 is the common
    # case (every real ONNX export is float32), and it needs a looser
    # tolerance because the C side itself is now computing in single
    # precision, accumulating the same kind of work-order rounding noise the
    # Fortran backend does.
    name = "gemm_big"
    onnx_path = golden_model(name)
    session = ort.InferenceSession(str(onnx_path))
    shape = session.get_inputs()[0].shape
    inputs, expected = _live_reference(session, shape, np.float32)
    if inputs is None:
        pytest.skip(f"{name}: onnxruntime reference is all-zero across 10 resampled "
                    f"batches; its golden-file weights produced a dead model")
    got = _build_and_run(tmp_path, onnx_path, name, inputs, dtype="f32")
    for produced, exp_row in zip(got, expected):
        np.testing.assert_allclose(produced, exp_row, rtol=1e-3, atol=1e-4)


def test_infer_is_pure_and_has_literal_bounds(golden_model):
    plan = build_plan(load_graph(golden_model("gemm_small")), dtype="f64")
    source, header = emit_c(plan)
    assert "void gemm_small_infer(const double *restrict x, double *restrict y) {" in source
    # The scratch buffers come from plan.buffers now (ruling R13), not from a
    # second allocator private to this emitter: gemm_small's t0 is reused by
    # both gemms, so the plan sizes it at the larger of the two (3), and the
    # Fortran backend declares exactly the same set.
    assert "double t0[3];" in source
    assert "double t1[2];" in source
    assert "malloc" not in source
    assert "restrict" in header


def test_init_rejects_a_foreign_weights_file(tmp_path, golden_model):
    graph = load_graph(golden_model("gemm_small"))
    plan = build_plan(graph, dtype="f64")
    other_graph = load_graph(golden_model("gemm_big"))
    other_plan = build_plan(other_graph, dtype="f64")
    source, header = emit_c(plan)
    (tmp_path / "gemm_small.c").write_text(source)
    (tmp_path / "gemm_small.h").write_text(header)
    write_weights(other_plan, other_graph, tmp_path / "gemm_small.rwt")
    (tmp_path / "main.c").write_text("""
#include <stdio.h>
#include "gemm_small.h"
int main(void) {
    int status = gemm_small_init("gemm_small.rwt");
    printf("%d\\n", status);
    return 0;
}
""")
    subprocess.run(["gcc", "-O2", "-Wall", "-Wextra", "-std=c11", "-o", "run",
                    "gemm_small.c", "main.c", "-lm"], cwd=tmp_path, check=True,
                   capture_output=True, text=True)
    out = subprocess.run(["./run"], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["6"]      # plan hash mismatch


def test_both_backends_agree(tmp_path, golden_model):
    (tmp_path / "c").mkdir()
    (tmp_path / "f").mkdir()
    name = "gemm_small"
    onnx_path = golden_model(name)
    session = ort.InferenceSession(str(onnx_path))
    shape = session.get_inputs()[0].shape
    inputs, expected = _live_reference(session, shape, np.float64, seed=2)
    if inputs is None:
        pytest.skip(f"{name}: onnxruntime reference is all-zero across 10 resampled "
                    f"batches; its golden-file weights produced a dead model")
    c_out = _build_and_run(tmp_path / "c", onnx_path, name, inputs)
    f_out = _fortran_build_and_run(tmp_path / "f", onnx_path, name, inputs)
    np.testing.assert_allclose(c_out, f_out, rtol=1e-12, atol=1e-14)
