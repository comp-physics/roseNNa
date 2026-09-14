"""Regressions from the whole-branch review of the codegen-dense wave.

Every model here is built inline with onnx.helper. The golden set is five
PyTorch exports and structurally cannot express any of these shapes: a
square MatMul weight, a graph whose first node is an activation, an
initializer name longer than the emitters' old fixed buffer, a genuine
float64 model, or a NaN travelling through a Relu.
"""
import re
import struct
import subprocess

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper, TensorProto

from rosenna.cli import main
from rosenna.emit_c import emit_c
from rosenna.emit_fortran import emit_fortran
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from rosenna.weights import write_weights
from tests.conftest import _assert_warning_free, _source_diagnostics, save_model
from tests.test_emit_c import _build_and_run as _c_build_and_run
from tests.test_emit_fortran import _build_and_run as _f_build_and_run
from tests.test_emit_fortran import _live_reference

DENSE = ["gemm_small", "gemm_big", "gemm_nobias", "droplet", "batchnet"]

# A PyTorch export names its initializers after the dotted module path, which
# routinely runs past the 128-character buffer both emitters used to declare.
_LONG_NAME = "model.encoder.layers.0.feedforward.linear_relu_stack." * 3 + "projection.weight"


_save = save_model


def _both_backends(tmp_path, onnx_path, name, inputs, dtype="f64"):
    """Build and run the model through both backends, returning (fortran, c)."""
    fdir, cdir = tmp_path / "f", tmp_path / "c"
    fdir.mkdir(exist_ok=True)
    cdir.mkdir(exist_ok=True)
    return (_f_build_and_run(fdir, onnx_path, name, inputs, dtype=dtype),
            _c_build_and_run(cdir, onnx_path, name, inputs, dtype=dtype))


def _reference(onnx_path, shape, dtype=np.float64, seed=0, batch=6):
    session = ort.InferenceSession(str(onnx_path))
    return _live_reference(session, shape, dtype, seed=seed, batch=batch)


def _init_status(tmp_path, lang, onnx_path, name, rwt_bytes):
    """Emit `name`'s init for `lang`, hand it `rwt_bytes`, return (exit code, stdout)."""
    work = tmp_path / lang
    work.mkdir(exist_ok=True)
    # embed=False: this helper hands `<name>_init` crafted/corrupted .rwt
    # bytes and checks the status it returns, so it always needs the
    # file-loaded contract (and its hash must match the caller's plan, which
    # also builds with embed=False -- see the two call sites below).
    plan = build_plan(load_graph(onnx_path), dtype="f64", embed=False)
    (work / f"{name}.rwt").write_bytes(rwt_bytes)
    if lang == "fortran":
        (work / f"{name}_model.F90").write_text(emit_fortran(plan))
        (work / "main.f90").write_text(f"""
program main
    use {name}_model
    implicit none
    integer :: status
    call {name}_init('{name}.rwt', status)
    print *, status
end program
""")
        subprocess.run(["gfortran", "-O2", "-o", "run", f"{name}_model.F90", "main.f90"],
                       cwd=work, check=True, capture_output=True, text=True)
    else:
        source, header = emit_c(plan)
        (work / f"{name}.c").write_text(source)
        (work / f"{name}.h").write_text(header)
        (work / "main.c").write_text(f"""
#include <stdio.h>
#include "{name}.h"
int main(void) {{
    printf("%d\\n", {name}_init("{name}.rwt"));
    return 0;
}}
""")
        subprocess.run(["gcc", "-O2", "-std=c11", "-o", "run", f"{name}.c", "main.c", "-lm"],
                       cwd=work, check=True, capture_output=True, text=True)
    done = subprocess.run(["./run"], cwd=work, capture_output=True, text=True)
    return done.returncode, done.stdout.strip()


# --- item 1: a square MatMul weight must not be transposed ------------------

def test_square_matmul_weight_is_not_transposed(tmp_path):
    # A non-symmetric 3x3 weight: reading it transposed changes the answer,
    # and n_in == n_out makes the emitters' old shape-sniffing ambiguous.
    w = np.arange(1, 10, dtype=np.float32).reshape(3, 3) * 0.1
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="mm")
    path = _save(tmp_path, "sqmm", [node], [numpy_helper.from_array(w, "w")], (1, 3), (1, 3))
    inputs, expected = _reference(path, [1, 3])
    assert inputs is not None
    f_out, c_out = _both_backends(tmp_path, path, "sqmm", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_square_transb_gemm_weight_is_not_transposed(tmp_path):
    # The mirror image: transB=1 with a square weight, which the same
    # shape-sniffing branch would have got right only by luck.
    w = numpy_helper.from_array(np.arange(1, 10, dtype=np.float32).reshape(3, 3) * 0.1, "w")
    b = numpy_helper.from_array(np.array([0.5, -0.25, 0.125], np.float32), "b")
    node = helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="g", transB=1)
    path = _save(tmp_path, "sqgemm", [node], [w, b], (1, 3), (1, 3))
    inputs, expected = _reference(path, [1, 3])
    assert inputs is not None
    f_out, c_out = _both_backends(tmp_path, path, "sqgemm", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


# --- item 3: the C emitter must consume the plan's own buffer assignment ----

def test_c_emitter_handles_a_leading_activation(tmp_path):
    # x -> Tanh -> MatMul -> Relu -> MatMul -> y. The C emitter's private
    # fusion pass gave the leading activation a length of None.
    rng = np.random.default_rng(7)
    w0 = numpy_helper.from_array(rng.uniform(-1, 1, (3, 4)).astype(np.float32), "w0")
    w1 = numpy_helper.from_array(rng.uniform(-1, 1, (4, 2)).astype(np.float32), "w1")
    nodes = [
        helper.make_node("Tanh", ["x"], ["a"], name="t0"),
        helper.make_node("MatMul", ["a", "w0"], ["b"], name="mm0"),
        helper.make_node("Relu", ["b"], ["c"], name="r0"),
        helper.make_node("MatMul", ["c", "w1"], ["y"], name="mm1"),
    ]
    path = _save(tmp_path, "leadact", nodes, [w0, w1], (1, 3), (1, 2))
    inputs, expected = _reference(path, [1, 3])
    assert inputs is not None
    f_out, c_out = _both_backends(tmp_path, path, "leadact", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_c_emitter_handles_a_gemm_output_with_two_consumers(tmp_path):
    # g feeds both a Relu and a Tanh; the private allocator lost g's buffer
    # to fusion and then raised KeyError looking the second consumer up.
    rng = np.random.default_rng(11)
    w0 = numpy_helper.from_array(rng.uniform(-1, 1, (3, 4)).astype(np.float32), "w0")
    w1 = numpy_helper.from_array(rng.uniform(-1, 1, (4, 2)).astype(np.float32), "w1")
    nodes = [
        helper.make_node("MatMul", ["x", "w0"], ["g"], name="mm0"),
        helper.make_node("Relu", ["g"], ["r"], name="r0"),
        helper.make_node("Tanh", ["g"], ["t"], name="t0"),
        helper.make_node("MatMul", ["t", "w1"], ["y"], name="mm1"),
    ]
    path = _save(tmp_path, "twouse", nodes, [w0, w1], (1, 3), (1, 2))
    inputs, expected = _reference(path, [1, 3])
    assert inputs is not None
    f_out, c_out = _both_backends(tmp_path, path, "twouse", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_activation_after_a_wider_op_reads_its_own_length(tmp_path):
    # The activation loop used to be bounded by a running "length of the
    # previous op's output". Here h is 5 wide, the op emitted before the Relu
    # is 40 wide, and the Relu reads h: with the old bound it ran 40
    # iterations over a double[5], reading and writing 35 elements past a
    # stack array with no diagnostic. The bound now comes from the plan.
    rng = np.random.default_rng(17)
    w0 = numpy_helper.from_array(rng.uniform(-1, 1, (2, 5)).astype(np.float32), "w0")
    w1 = numpy_helper.from_array(rng.uniform(-1, 1, (5, 40)).astype(np.float32), "w1")
    w2 = numpy_helper.from_array(rng.uniform(-1, 1, (5, 2)).astype(np.float32), "w2")
    nodes = [
        helper.make_node("MatMul", ["x", "w0"], ["h"], name="mm0"),
        helper.make_node("MatMul", ["h", "w1"], ["wide"], name="mm1"),
        helper.make_node("Relu", ["h"], ["r"], name="late_relu"),
        helper.make_node("MatMul", ["r", "w2"], ["y"], name="mm2"),
    ]
    path = _save(tmp_path, "latewide", nodes, [w0, w1, w2], (1, 2), (1, 2))
    inputs, expected = _reference(path, [1, 2])
    assert inputs is not None
    f_out, c_out = _both_backends(tmp_path, path, "latewide", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


# --- item 2: names and ranks from the file must not overflow a fixed buffer -

def _long_name_model(tmp_path):
    w = numpy_helper.from_array(np.arange(1, 7, dtype=np.float32).reshape(3, 2) * 0.1, _LONG_NAME)
    node = helper.make_node("MatMul", ["x", _LONG_NAME], ["y"], name="mm")
    return _save(tmp_path, "longname", [node], [w], (1, 3), (1, 2))


def test_long_initializer_name_round_trips(tmp_path):
    assert len(_LONG_NAME) > 128
    path = _long_name_model(tmp_path)
    # embed=False: this test is about the name buffer inside `init`'s table-
    # of-contents reader, which an embedded plan's source does not emit.
    plan = build_plan(load_graph(path), dtype="f64", embed=False)

    fsrc = emit_fortran(plan)
    csrc, _ = emit_c(plan)
    f_cap = int(re.search(r"character\(len=(\d+)\) :: name", fsrc).group(1))
    c_cap = int(re.search(r"char name\[(\d+)\]", csrc).group(1))
    assert f_cap >= len(_LONG_NAME), f"fortran name buffer is {f_cap}"
    assert c_cap >= len(_LONG_NAME), f"c name buffer is {c_cap}"

    inputs, expected = _reference(path, [1, 3])
    assert inputs is not None
    f_out, c_out = _both_backends(tmp_path, path, "longname", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("lang", ["fortran", "c"])
def test_oversized_name_length_in_the_file_is_rejected(tmp_path, golden_model, lang):
    # expected_hash is compiled into the binary, so a crafted file can copy it
    # verbatim and still reach the name read. The table of contents starts at
    # byte 60; overwrite the first tensor's name length with 4000.
    onnx_path = golden_model("gemm_small")
    graph = load_graph(onnx_path)
    # embed=False to match _init_status's own plan -- both build the same
    # model the same way, or their plan hashes (and thus this file's
    # embedded expected_hash) would disagree.
    plan = build_plan(graph, dtype="f64", embed=False)
    good = tmp_path / "good.rwt"
    write_weights(plan, graph, good)
    blob = bytearray(good.read_bytes())
    blob[60:64] = struct.pack("<i", 4000)
    rc, out = _init_status(tmp_path, lang, onnx_path, "gemm_small", bytes(blob))
    assert rc == 0, f"generated init died instead of returning a status (exit {rc})"
    assert out.split() == ["8"], out


# --- item 4: verify must feed a float64 model float64 inputs ----------------

def test_verify_accepts_a_genuine_float64_model(tmp_path, capsys):
    rng = np.random.default_rng(3)
    w = numpy_helper.from_array(rng.uniform(-1, 1, (3, 4)), "w")      # float64
    b = numpy_helper.from_array(rng.uniform(-1, 1, (4,)), "b")
    nodes = [
        helper.make_node("Gemm", ["x", "w", "b"], ["g"], name="g0"),
        helper.make_node("Tanh", ["g"], ["y"], name="t0"),
    ]
    path = _save(tmp_path, "f64model", nodes, [w, b], (1, 3), (1, 4),
                 elem=TensorProto.DOUBLE)
    rc = main(["verify", str(path), "--cases", "4"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "fortran" in out and "c" in out and "FAIL" not in out


# --- item 8: Relu must propagate NaN ---------------------------------------

def _relu_only_model(tmp_path):
    node = helper.make_node("Relu", ["x"], ["y"], name="r0")
    return _save(tmp_path, "relunan", [node], [], (1, 4), (1, 4))


def test_relu_propagates_nan_in_both_backends(tmp_path):
    path = _relu_only_model(tmp_path)
    inputs = np.array([[np.nan, -1.0, 2.0, 0.0]])
    session = ort.InferenceSession(str(path))
    expected = session.run(None, {"x": inputs.astype(np.float32).reshape(1, 4)})[0].ravel()
    assert np.isnan(expected[0])
    f_out, c_out = _both_backends(tmp_path, path, "relunan", inputs)
    for lang, got in (("fortran", f_out), ("c", c_out)):
        assert np.isnan(got[0][0]), f"{lang} laundered NaN into {got[0][0]}"
        np.testing.assert_allclose(got[0][1:], [0.0, 2.0, 0.0])


# --- item 5: a truncated weights file must become a status, not a crash ----

@pytest.mark.parametrize("lang", ["fortran", "c"])
def test_truncated_weights_file_returns_a_status(tmp_path, golden_model, lang):
    onnx_path = golden_model("gemm_small")
    graph = load_graph(onnx_path)
    # embed=False: see the comment in the sibling test above.
    plan = build_plan(graph, dtype="f64", embed=False)
    good = tmp_path / "good.rwt"
    write_weights(plan, graph, good)
    rc, out = _init_status(tmp_path, lang, onnx_path, "gemm_small", good.read_bytes()[:-12])
    assert rc == 0, f"generated init aborted instead of returning a status (exit {rc})"
    assert out.split() == ["9"], out


# --- item 10 / verification 3: generated code must compile warning-free ----
#
# _DRIVER_NOTICE / _source_diagnostics / _assert_warning_free live in
# conftest.py so every test module (device/kernel tests included) can filter
# driver-level toolchain notices, e.g. Apple clang's on the macOS CI runner:
#   clang: warning: overriding deployment version from '16.0' to '26.0' [-Woverriding-deployment-version]
# (ruling R21, R30).


def _compile_warnings(tmp_path, onnx_path, name, dtype="f64"):
    plan = build_plan(load_graph(onnx_path), dtype=dtype)
    work = tmp_path / f"{name}_{dtype}"
    work.mkdir(exist_ok=True)
    (work / f"{name}_model.F90").write_text(emit_fortran(plan))
    source, header = emit_c(plan)
    (work / f"{name}.c").write_text(source)
    (work / f"{name}.h").write_text(header)
    # -std=f2008 makes every gfortran enforce the 132-column limit (ruling R20)
    # and anything else non-standard, rather than only the CI compiler.
    f = subprocess.run(["gfortran", "-std=f2008", "-O2", "-Wall", "-Wextra", "-c",
                        f"{name}_model.F90"],
                       cwd=work, capture_output=True, text=True, check=True)
    c = subprocess.run(["gcc", "-O2", "-Wall", "-Wextra", "-std=c11", "-c", f"{name}.c"],
                       cwd=work, capture_output=True, text=True, check=True)
    return f.stderr, c.stderr


def test_weight_free_model_compiles_without_warnings(tmp_path):
    path = _relu_only_model(tmp_path)
    f_err, c_err = _compile_warnings(tmp_path, path, "relunan")
    _assert_warning_free("gfortran", f_err)
    _assert_warning_free("gcc", c_err)


@pytest.mark.parametrize("name", DENSE)
@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_dense_models_compile_without_warnings(tmp_path, golden_model, name, dtype):
    f_err, c_err = _compile_warnings(tmp_path, golden_model(name), name, dtype=dtype)
    _assert_warning_free("gfortran", f_err)
    _assert_warning_free("gcc", c_err)


def test_source_diagnostics_filter_keeps_real_warnings_and_drops_driver_noise():
    clang_noise = ("clang: warning: overriding deployment version from '16.0' to '26.0' "
                   "[-Woverriding-deployment-version]")
    c_warning = "foo.c:3:5: warning: unused variable 'x' [-Wunused-variable]"
    kept, dropped = _source_diagnostics(clang_noise + "\n" + c_warning + "\n")
    assert kept == c_warning
    assert dropped == clang_noise

    # gfortran's multi-line form: the located header and the bare Warning line
    # both survive, so a real Fortran warning still fails the assertion.
    gfortran_warning = ("m.f90:65:23:\n\n   65 |         integer :: i, j\n"
                        "      |                       1\n"
                        "Warning: Unused variable 'j' declared at (1) [-Wunused-variable]\n")
    kept, dropped = _source_diagnostics(clang_noise + "\n" + gfortran_warning)
    assert "Warning: Unused variable 'j'" in kept and "m.f90:65:23:" in kept
    assert dropped == clang_noise

    # Only the noise: nothing about the source remains.
    assert _source_diagnostics(clang_noise + "\n") == ("", clang_noise)
    with pytest.raises(AssertionError, match=r"(?s)unused variable 'x'.*ignored: clang"):
        _assert_warning_free("gcc", clang_noise + "\n" + c_warning + "\n")
    _assert_warning_free("gcc", clang_noise + "\n")


# --- ruling R20: generated Fortran must respect the 132-column free-form limit

_FORTRAN_MAX_COLS = 132


def _over_long_fortran_lines(src):
    return [(n, len(l)) for n, l in enumerate(src.splitlines(), 1) if len(l) > _FORTRAN_MAX_COLS]


@pytest.mark.parametrize("name", DENSE)
@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_generated_fortran_fits_in_132_columns(golden_model, name, dtype):
    # Free-form Fortran source is limited to 132 columns. gfortran 15 accepts a
    # longer line silently; gfortran 13 (CI) rejects it with -Werror=line-truncation,
    # and so does any gfortran under -std=f2008. The expected_hash constructor
    # alone was 553 columns. This check must not depend on which gfortran is
    # installed, so it is on the emitted text.
    src = emit_fortran(build_plan(load_graph(golden_model(name)), dtype=dtype))
    assert _over_long_fortran_lines(src) == []


def test_generated_fortran_fits_in_132_columns_with_a_long_tensor_name(tmp_path):
    # The name buffer is plan-derived now, so a 176-character initializer name
    # becomes a 176-character case label unless the literal is continued.
    # embed=False: the case label this test asserts on lives in `load_tensor`,
    # which an embedded plan's module does not emit.
    src = emit_fortran(build_plan(load_graph(_long_name_model(tmp_path)), dtype="f64", embed=False))
    assert _over_long_fortran_lines(src) == []
    # Wrapped, not dropped: the literal is continued across lines, so the
    # head of the name is still there and the tail follows a leading `&`.
    assert "case ('" + _LONG_NAME[:20] in src
    assert _LONG_NAME[-12:] + "')" in src
    assert any(l.strip().startswith("&") for l in src.splitlines())
