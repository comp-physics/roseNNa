import os
import platform
import shutil
import subprocess
import numpy as np
import onnxruntime as ort
import pytest
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from rosenna.weights import write_weights
from rosenna.emit_fortran import emit_fortran, emit_fortran_recipe
from tests.conftest import _assert_warning_free
from tests.test_emit_fortran import _live_reference


def _omp_fc():
    fc = shutil.which("gfortran")
    if not fc:
        pytest.skip("no gfortran")
    probe = subprocess.run([fc, "-fopenmp", "-x", "f95", "-", "-o", os.devnull],
                           input="end\n", capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("gfortran without -fopenmp")
    return fc


HOST = """
program host
    use {name}_model
    use iso_fortran_env, only: real64
    implicit none
    real(real64) :: x({n_in}, 64), y({n_out}, 64), yb({n_out}, 64)
    integer :: n, p, status
    {init_lines}
    read(*,*) n
    read(*,*) x(:, 1:n)
    !$omp target enter data map(to: x) map(alloc: y, yb)
    !$omp target teams loop
    do p = 1, n
        call {name}_infer(x(:, p), y(:, p))
    end do
    ! infer_batch's has_device_addr wants the mapped arrays' device addresses,
    ! which use_device_addr supplies (ruling R5; on this host-only build they
    ! are the host addresses, so the omission would have been invisible).
    !$omp target data use_device_addr(x, yb)
    call {name}_infer_batch(n, x, yb, status)
    !$omp end target data
    !$omp target exit data map(from: y, yb) map(delete: x)
    if (status /= 0) stop 4
    ! abs(...) > 0, not /=: an exact-bits comparison without tripping
    ! gfortran's -Wcompare-reals on a bare real (in)equality.
    if (any(abs(y(:, 1:n) - yb(:, 1:n)) > 0.0_real64)) stop 5
    do p = 1, n
        print '({n_out}(es24.16,1x))', y(:, p)
    end do
end program
"""


def _build_and_run(tmp_path, name, embed, inputs, golden_model):
    graph = load_graph(golden_model(name))
    plan = build_plan(graph, dtype="f64", embed=embed)
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    (tmp_path / f"{name}_model.f90").write_text(emit_fortran(plan))
    init_lines = "" if embed else f'call {name}_init("{name}.rwt", status); if (status /= 0) stop 2'
    if not embed:
        write_weights(plan, graph, tmp_path / f"{name}.rwt")
    (tmp_path / "host.f90").write_text(HOST.format(name=name, n_in=n_in, n_out=n_out, init_lines=init_lines))
    fc = _omp_fc()
    flags = ["-O2", "-Wall", "-Wextra", "-std=f2008", "-fopenmp"]
    r = subprocess.run([fc, *flags, "-c", f"{name}_model.f90"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    _assert_warning_free("gfortran", r.stderr)
    subprocess.run(["ar", "rcs", f"lib{name}.a", f"{name}_model.o"], cwd=tmp_path, check=True)
    r = subprocess.run([fc, *flags, "host.f90", f"lib{name}.a", "-o", "host"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    _assert_warning_free("gfortran", r.stderr)
    stdin = f"{len(inputs)}\n" + "\n".join(" ".join(repr(float(v)) for v in row) for row in inputs)
    out = subprocess.run(["./host"], cwd=tmp_path, input=stdin, capture_output=True, text=True, check=True).stdout
    return np.array([[float(v) for v in line.split()] for line in out.strip().splitlines()])


@pytest.mark.parametrize("name", ["gemm_small", "gemm_big", "gemm_nobias", "droplet", "batchnet"])
@pytest.mark.parametrize("embed", [True, False])
def test_host_region_calls_module_infer_and_matches(tmp_path, golden_model, name, embed):
    session = ort.InferenceSession(golden_model(name))
    inputs, expected = _live_reference(session, session.get_inputs()[0].shape, np.float64, seed=8, batch=8)
    if inputs is None:
        pytest.skip(f"{name}: onnxruntime reference is all-zero across 10 resampled "
                    f"batches; its golden-file weights produced a dead model")
    got = _build_and_run(tmp_path, name, embed, inputs, golden_model)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_fortran_target_regions_are_real(tmp_path, golden_model):
    # Two-tier evidence (controller ruling R31). The only evidence here used to be that a
    # host-only libgomp refuses a target region under OMP_TARGET_OFFLOAD=MANDATORY: if the
    # pragmas were missing or ignored the program would succeed. That held locally (gfortran
    # 15) but on the macOS CI runner (Homebrew GCC 13.4, `gfortran` -> `gfortran-13`) this
    # program ran to completion and returned 0 -- not refused -- while the identical C test
    # (tests/test_device_c.py::test_target_regions_are_real) passed on that same runner and
    # compiler. So MANDATORY enforcement is not portable evidence by itself. The primary,
    # platform-independent assertion is instead that the compiled host object references
    # GOMP_target_ext: a real `!$omp target` region cannot be compiled without a call to it.
    # The MANDATORY run is kept as corroborating evidence where libgomp does enforce it, and
    # downgraded to a skip (not a failure) where it does not, naming the toolchain that let it
    # through so the CI log records exactly which combination did this. Neither tier compares
    # against onnxruntime, so a constant input (not _live_reference) is enough, and this test
    # cannot itself be a dead-model false pass/fail like test_host_region_calls_module_infer_
    # and_matches above needs to guard against.
    name = "gemm_small"
    graph = load_graph(golden_model(name)); plan = build_plan(graph, dtype="f64", embed=True)
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    inputs = np.full((1, n_in), 0.5)
    (tmp_path / f"{name}_model.f90").write_text(emit_fortran(plan))
    (tmp_path / "host.f90").write_text(HOST.format(name=name, n_in=n_in, n_out=n_out, init_lines=""))
    fc = _omp_fc()
    flags = ["-O2", "-std=f2008", "-fopenmp"]

    # Compile the module first (for its .mod) and the host as a standalone object, so the
    # symbol check below inspects exactly what the host's own target region compiled to.
    r = subprocess.run([fc, *flags, "-c", f"{name}_model.f90"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([fc, *flags, "-c", "host.f90", "-o", "host.o"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    nm = shutil.which("nm")
    if not nm:
        pytest.skip("no nm")
    nm_out = subprocess.run([nm, "-u", "host.o"], cwd=tmp_path, capture_output=True, text=True).stdout
    undefined = {line.split()[-1] for line in nm_out.splitlines() if line.strip()}
    assert any("GOMP_target_ext" in sym for sym in undefined), nm_out

    r = subprocess.run([fc, *flags, "host.o", f"{name}_model.o", "-o", "host"], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    stdin = "1\n" + " ".join(repr(float(v)) for v in inputs[0])
    r = subprocess.run(["./host"], cwd=tmp_path, input=stdin, capture_output=True, text=True,
                       env={**os.environ, "OMP_TARGET_OFFLOAD": "MANDATORY"})
    if r.returncode == 0:
        version = subprocess.run([fc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
        pytest.skip(f"libgomp did not enforce OMP_TARGET_OFFLOAD=MANDATORY for a Fortran target "
                    f"region on {platform.platform()} with {version}")
    assert "MANDATORY" in r.stderr, r.stderr


def test_generated_fortran_still_fits_in_132_columns(golden_model):
    for name in ["gemm_small", "gemm_big", "gemm_nobias", "droplet", "batchnet"]:
        for embed in (True, False):
            src = emit_fortran(build_plan(load_graph(golden_model(name)), dtype="f64", embed=embed))
            longest = max(len(line) for line in src.splitlines())
            assert longest <= 132, (name, embed, longest)


def test_infer_batch_body_never_transfers(golden_model):
    # Ruling R5, structural check: infer_batch's own body holds no map/copyin/
    # copyout/update/enter-exit-data token -- init is exempt (it is the plan
    # step and the only routine that transfers).
    forbidden = ("map(", "copyin", "copyout", "target update", "update device", "enter data", "exit data")
    for name in ["gemm_small", "gemm_big", "gemm_nobias", "droplet", "batchnet"]:
        for embed in (True, False):
            plan = build_plan(load_graph(golden_model(name)), dtype="f64", embed=embed)
            src = emit_fortran(plan)
            start = src.index(f"subroutine {name}_infer_batch(")
            end = src.index("end subroutine", start)
            body = src[start:end]
            for token in forbidden:
                assert token not in body, (name, embed, token)


def test_embedded_module_has_no_init_and_file_loaded_does(golden_model):
    for embed, expect_init in ((True, False), (False, True)):
        plan = build_plan(load_graph(golden_model("gemm_small")), dtype="f64", embed=embed)
        src = emit_fortran(plan)
        assert ("subroutine gemm_small_init(" in src) == expect_init
        # Both forms are `protected` module arrays; the embedded one carries
        # its initializer (see _emit_embedded_weights for why not `parameter`).
        assert ("real(wp), protected :: w0(2,2) = reshape([" in src) == embed
        assert ("real(wp), protected :: w0(2,2)\n" in src) == (not embed)
        assert ("!$acc declare copyin(w0, b0, w1, b1)" in src) == embed
        assert ("!$acc declare create(w0, b0, w1, b1)" in src) == (not embed)


def test_generated_fortran_is_warning_free_under_openacc(tmp_path, golden_model):
    # Mirrors tests/test_kernel.py::test_generated_c_is_warning_free_under_openacc.
    # gfortran -fopenacc rejects a `routine seq` function reading a module
    # array with no `declare` directive of its own, and refuses a `declare`
    # on a `parameter` array (why _emit_embedded_weights emits initialized
    # `protected` arrays with `declare copyin`). A real compile (-c), not
    # -fsyntax-only: the diagnostic comes after the front end and
    # -fsyntax-only let an uncompilable embedded module through.
    fc = shutil.which("gfortran")
    if not fc:
        pytest.skip("no gfortran")
    probe = subprocess.run([fc, "-fopenacc", "-x", "f95", "-", "-o", os.devnull],
                           input="end\n", capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip(f"{fc} does not accept -fopenacc")
    for name in ["gemm_small", "gemm_big", "gemm_nobias", "droplet", "batchnet"]:
        for embed in (True, False):
            plan = build_plan(load_graph(golden_model(name)), dtype="f64", embed=embed)
            src_path = tmp_path / f"{name}_{embed}_model.f90"
            src_path.write_text(emit_fortran(plan))
            r = subprocess.run(
                [fc, "-O2", "-Wall", "-Wextra", "-std=f2008", "-fopenacc", "-c",
                 src_path.name, "-o", f"{name}_{embed}_model.o"],
                cwd=tmp_path, capture_output=True, text=True)
            assert r.returncode == 0, (name, embed, r.stderr)
            _assert_warning_free("gfortran", r.stderr)


def test_bind_c_interface_targets_the_c_infer_batch_symbol(golden_model):
    plan = build_plan(load_graph(golden_model("gemm_small")), dtype="f64")
    src = emit_fortran(plan)
    assert 'bind(C, name="gemm_small_infer_batch") result(status)' in src
    assert "gemm_small_infer_batch_dev" in src
    assert "type(c_ptr), value :: x, y, stream" in src


def test_fortran_recipe_builds_the_library(tmp_path, golden_model):
    name = "gemm_small"
    plan = build_plan(load_graph(golden_model(name)), dtype="f64")
    (tmp_path / f"{name}_model.f90").write_text(emit_fortran(plan))
    (tmp_path / "Makefile").write_text(emit_fortran_recipe(plan))
    fc = _omp_fc()
    subprocess.run(["make", f"FC={fc}"], cwd=tmp_path, check=True, capture_output=True, text=True)
    # lib<name>_f.a, not lib<name>.a (ruling R13): see
    # tests/test_cli.py::test_both_recipes_build_distinct_archives_in_one_directory
    # for why the two names must never collide.
    assert (tmp_path / f"lib{name}_f.a").exists()
    assert (tmp_path / f"{name}_model.o").exists()
    subprocess.run(["make", "clean"], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert not (tmp_path / f"lib{name}_f.a").exists()
    assert not (tmp_path / f"{name}_model.o").exists()
