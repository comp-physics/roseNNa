import subprocess
import numpy as np
import onnxruntime as ort
import pytest
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from rosenna.weights import write_weights
from rosenna.emit_fortran import emit_fortran

DENSE = ["gemm_small", "gemm_big", "gemm_nobias", "droplet", "batchnet"]


def _build_and_run(tmp_path, onnx_path, name, inputs, dtype="f64"):
    graph = load_graph(onnx_path)
    plan = build_plan(graph, dtype=dtype)
    (tmp_path / f"{name}_model.f90").write_text(emit_fortran(plan))
    write_weights(plan, graph, tmp_path / f"{name}.rwt")
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    real_kind = "real64" if dtype == "f64" else "real32"
    (tmp_path / "main.f90").write_text(f"""
program main
    use {name}_model
    use iso_fortran_env, only: {real_kind}
    implicit none
    real({real_kind}) :: x({n_in}), y({n_out})
    integer :: status, i, ncases
    read(*,*) ncases
    call {name}_init('{name}.rwt', status)
    if (status /= 0) then
        print *, 'init status', status
        stop 1
    end if
    do i = 1, ncases
        read(*,*) x
        call {name}_infer(x, y)
        print '({n_out}(es24.16,1x))', y
    end do
end program
""")
    subprocess.run(["gfortran", "-O2", "-Wall", "-Wextra", "-o", "run",
                    f"{name}_model.f90", "main.f90"], cwd=tmp_path, check=True,
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
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-2, 2, (8, int(np.prod(shape)))).astype(np.float64)
    got = _build_and_run(tmp_path, onnx_path, name, inputs)
    for row, produced in zip(inputs, got):
        expected = session.run(None, {session.get_inputs()[0].name:
                                      row.reshape(shape).astype(np.float32)})[0].ravel()
        np.testing.assert_allclose(produced, expected, rtol=1e-5, atol=1e-6)
        # Guard against a degenerate all-zeros comparison (e.g. a ReLU model
        # whose inputs happened to all die): require every row -- not just
        # somewhere across the whole batch -- to carry real signal. This is
        # deliberately >=1, not a higher fixed count: gemm_small's golden
        # file has no manual_seed, so a freshly regenerated model can
        # legitimately zero two of its three ReLU outputs for a given
        # random input row (observed directly: a cold run produced
        # [0.543, 0.0, 0.0]). Demanding >=1 nonzero per row still forbids
        # the fully-degenerate all-zero row the old whole-batch check let
        # through, without making the test flaky against genuine ReLU
        # sparsity in an unseeded model.
        assert np.count_nonzero(produced) >= 1, f"row output is degenerate: {produced}"


def test_matches_onnxruntime_f32(tmp_path, golden_model):
    # Every real ONNX export is float32, and --precision defaults to the
    # model's own dtype, so the f32 path is the common one, not a corner
    # case. It needs its own (looser) tolerance: with the Fortran side
    # itself computing in single precision, both the reference and the
    # generated code accumulate rounding error from work-order differences
    # in the dot products, on top of the fp32 quantization of intermediate
    # activations. rtol=1e-3 / atol=1e-4 comfortably separates that expected
    # rounding noise from an actual layout or indexing bug (which produces
    # errors many orders of magnitude larger, not a marginal tolerance miss).
    name = "gemm_big"
    onnx_path = golden_model(name)
    session = ort.InferenceSession(str(onnx_path))
    shape = session.get_inputs()[0].shape
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-2, 2, (8, int(np.prod(shape)))).astype(np.float32)
    got = _build_and_run(tmp_path, onnx_path, name, inputs, dtype="f32")
    for row, produced in zip(inputs, got):
        expected = session.run(None, {session.get_inputs()[0].name:
                                      row.reshape(shape).astype(np.float32)})[0].ravel()
        np.testing.assert_allclose(produced, expected, rtol=1e-3, atol=1e-4)
        assert np.count_nonzero(produced) >= 1, f"row output is degenerate: {produced}"


def test_infer_is_pure_and_has_literal_bounds(golden_model):
    plan = build_plan(load_graph(golden_model("gemm_small")), dtype="f64")
    src = emit_fortran(plan)
    assert "pure subroutine gemm_small_infer" in src
    assert "real(wp), protected :: w0(2,2)" in src
    assert "allocatable" not in src


def test_init_rejects_a_foreign_weights_file(tmp_path, golden_model):
    graph = load_graph(golden_model("gemm_small"))
    plan = build_plan(graph, dtype="f64")
    other_graph = load_graph(golden_model("gemm_big"))
    other_plan = build_plan(other_graph, dtype="f64")
    (tmp_path / "gemm_small_model.f90").write_text(emit_fortran(plan))
    write_weights(other_plan, other_graph, tmp_path / "gemm_small.rwt")
    (tmp_path / "main.f90").write_text("""
program main
    use gemm_small_model
    implicit none
    integer :: status
    call gemm_small_init('gemm_small.rwt', status)
    print *, status
end program
""")
    subprocess.run(["gfortran", "-O2", "-o", "run", "gemm_small_model.f90", "main.f90"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    out = subprocess.run(["./run"], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["6"]      # plan hash mismatch
