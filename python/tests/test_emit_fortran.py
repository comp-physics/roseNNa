import subprocess
import numpy as np
import onnxruntime as ort
import pytest
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from rosenna.weights import write_weights
from rosenna.emit_fortran import emit_fortran

DENSE = ["gemm_small", "gemm_big", "gemm_nobias", "droplet", "batchnet"]


def _build_and_run(tmp_path, name, inputs):
    onnx_path = f"../goldenFiles/{name}/{name}.onnx"
    graph = load_graph(onnx_path)
    plan = build_plan(graph, dtype="f64")
    (tmp_path / f"{name}_model.f90").write_text(emit_fortran(plan))
    write_weights(plan, graph, tmp_path / f"{name}.rwt")
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    (tmp_path / "main.f90").write_text(f"""
program main
    use {name}_model
    use iso_fortran_env, only: real64
    implicit none
    real(real64) :: x({n_in}), y({n_out})
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
def test_matches_onnxruntime(tmp_path, name):
    session = ort.InferenceSession(f"../goldenFiles/{name}/{name}.onnx")
    shape = session.get_inputs()[0].shape
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-2, 2, (8, int(np.prod(shape)))).astype(np.float64)
    got = _build_and_run(tmp_path, name, inputs)
    for row, produced in zip(inputs, got):
        expected = session.run(None, {session.get_inputs()[0].name:
                                      row.reshape(shape).astype(np.float32)})[0].ravel()
        np.testing.assert_allclose(produced, expected, rtol=1e-5, atol=1e-6)
    # Guard against a degenerate all-zeros comparison (e.g. a ReLU model
    # whose inputs happened to all die): require some real signal.
    assert np.count_nonzero(got) >= 2


def test_infer_is_pure_and_has_literal_bounds():
    plan = build_plan(load_graph("../goldenFiles/gemm_small/gemm_small.onnx"), dtype="f64")
    src = emit_fortran(plan)
    assert "pure subroutine gemm_small_infer" in src
    assert "real(wp), protected :: w0(2,2)" in src
    assert "allocatable" not in src


def test_init_rejects_a_foreign_weights_file(tmp_path):
    graph = load_graph("../goldenFiles/gemm_small/gemm_small.onnx")
    plan = build_plan(graph, dtype="f64")
    other_graph = load_graph("../goldenFiles/gemm_big/gemm_big.onnx")
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
