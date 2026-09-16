import subprocess
import numpy as np
import onnxruntime as ort
import pytest
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from rosenna.weights import write_weights
from rosenna.emit_fortran import emit_fortran
# _live_reference now lives in rosenna/verify.py (rosenna/gate.py needs it too,
# and cannot import test code); re-exported here under its original name so
# every existing `from tests.test_emit_fortran import _live_reference` keeps
# working unchanged.
from rosenna.verify import _live_reference

DENSE = ["gemm_small", "gemm_big", "gemm_nobias", "droplet", "batchnet"]


def _build_and_run(tmp_path, onnx_path, name, inputs, dtype="f64"):
    # embed=False: this helper's driver always calls `<name>_init` against a
    # written .rwt file, the file-loaded contract. The dedicated embed=True/
    # False matrix lives in tests/test_device_fortran.py and tests/test_embed.py.
    graph = load_graph(onnx_path)
    plan = build_plan(graph, dtype=dtype, embed=False)
    (tmp_path / f"{name}_model.F90").write_text(emit_fortran(plan))
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
                    f"{name}_model.F90", "main.f90"], cwd=tmp_path, check=True,
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
    # Compare every row, including any that individually landed on zero
    # inside an otherwise-live batch -- degenerate rows are not filtered
    # out, only a fully dead batch is resampled away.
    for produced, exp_row in zip(got, expected):
        np.testing.assert_allclose(produced, exp_row, rtol=1e-5, atol=1e-6)


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
    # See _live_reference: non-degeneracy is checked on the onnxruntime
    # reference, with resampling, not on our own output -- a dead model's
    # all-zero reference is a fixture property (gemm_big also has no
    # manual_seed), and correct code reproducing it is also all zero.
    inputs, expected = _live_reference(session, shape, np.float32)
    if inputs is None:
        pytest.skip(f"{name}: onnxruntime reference is all-zero across 10 resampled "
                    f"batches; its golden-file weights produced a dead model")
    got = _build_and_run(tmp_path, onnx_path, name, inputs, dtype="f32")
    for produced, exp_row in zip(got, expected):
        np.testing.assert_allclose(produced, exp_row, rtol=1e-3, atol=1e-4)


def test_infer_is_pure_and_has_literal_bounds(golden_model):
    # embed=False: this test is specifically about the file-loaded contract
    # (`protected` module variables filled by `init`), which an embedded
    # plan's module does not declare.
    plan = build_plan(load_graph(golden_model("gemm_small")), dtype="f64", embed=False)
    src = emit_fortran(plan)
    assert "pure subroutine gemm_small_infer" in src
    assert "real(wp), protected :: w0(2,2)" in src
    assert "allocatable" not in src


def test_init_rejects_a_foreign_weights_file(tmp_path, golden_model):
    # embed=False: this test is specifically about `_init`, which an
    # embedded plan's module does not declare.
    graph = load_graph(golden_model("gemm_small"))
    plan = build_plan(graph, dtype="f64", embed=False)
    other_graph = load_graph(golden_model("gemm_big"))
    other_plan = build_plan(other_graph, dtype="f64", embed=False)
    (tmp_path / "gemm_small_model.F90").write_text(emit_fortran(plan))
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
    subprocess.run(["gfortran", "-O2", "-o", "run", "gemm_small_model.F90", "main.f90"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    out = subprocess.run(["./run"], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["6"]      # plan hash mismatch


def test_case_labels_escape_quotes(tmp_path):
    """A quote in a tensor name must be doubled, not left to break the source."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    quoted = "layer.0'weight"
    w = numpy_helper.from_array(np.zeros((2, 2), np.float32), quoted)
    node = helper.make_node("MatMul", ["x", quoted], ["y"], name="mm")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    g = helper.make_graph([node], "quoted", [x], [y], initializer=[w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "quoted.onnx"
    onnx.save(m, path)
    # embed=False: the case-label select lives in `load_tensor`, which an
    # embedded plan's module does not emit.
    src = emit_fortran(build_plan(load_graph(path), dtype="f64", embed=False))
    assert "case ('layer.0''weight')" in src
    (tmp_path / "quoted_model.F90").write_text(src)
    subprocess.run(["gfortran", "-O2", "-Wall", "-Wextra", "-c", "quoted_model.F90"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
