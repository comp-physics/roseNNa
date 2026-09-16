"""Regressions from the whole-branch review of the codegen-dense wave.

Every model here is built inline with onnx.helper. The golden set is five
PyTorch exports and structurally cannot express any of these shapes: a
square MatMul weight, a graph whose first node is an activation, an
initializer name longer than the emitters' old fixed buffer, a genuine
float64 model, or a NaN travelling through a Relu.
"""
import os
import re
import struct
import subprocess

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper, TensorProto

from rosenna.cli import main
from rosenna.errors import UnsupportedModel
from rosenna.emit_c import emit_c
from rosenna.emit_fortran import emit_fortran
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from rosenna.weights import write_weights
from tests.conftest import _assert_warning_free, _source_diagnostics, save_model
from tests.test_emit_c import _build_and_run as _c_build_and_run
from tests.test_emit_fortran import _build_and_run as _f_build_and_run
from tests.test_emit_fortran import _live_reference
from tests.conftest import save_model

DENSE = ["gemm_small", "gemm_big", "gemm_nobias", "droplet", "batchnet"]
from tests.test_golden_suite import GOLDEN as GOLDEN_NAMES

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
    plan = build_plan(load_graph(onnx_path, name), dtype=dtype)
    work = tmp_path / f"{name}_{dtype}"
    work.mkdir(exist_ok=True)
    (work / f"{name}_model.F90").write_text(emit_fortran(plan))
    source, header = emit_c(plan)
    (work / f"{name}.c").write_text(source)
    (work / f"{name}.h").write_text(header)
    # -std=f2008 makes every gfortran enforce the 132-column limit (ruling R20)
    # and anything else non-standard, rather than only the CI compiler.
    # ROSENNA_CC / ROSENNA_FC let a CI job point this at a second compiler.
    # Without them the clang job would re-run gcc and prove nothing, which is
    # the failure mode the widened parametrization above already fell into.
    cc = os.environ.get("ROSENNA_CC", "gcc")
    fc = os.environ.get("ROSENNA_FC", "gfortran")
    f = subprocess.run([fc, "-std=f2008", "-O2", "-Wall", "-Wextra", "-c",
                        f"{name}_model.F90"],
                       cwd=work, capture_output=True, text=True, check=True)
    c = subprocess.run([cc, "-O2", "-Wall", "-Wextra", "-std=c11", "-c", f"{name}.c"],
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


# Every golden model, not just the dense ones. The dense-only list above dated
# from when dense was all the generator emitted, and it quietly stopped being
# "the generated code" once Conv, pooling, LSTM and the shape ops landed: the
# LSTM emitter was writing buffers for outputs the model never reads, and gcc
# reported it in every CI run that never compiled an LSTM model.
@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_every_golden_model_compiles_without_warnings(tmp_path, golden_model, name):
    safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
    f_err, c_err = _compile_warnings(tmp_path, golden_model(name), safe)
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


def test_a_float64_model_built_single_is_held_to_the_single_tolerance(tmp_path):
    # Tolerance used to follow the model's dtype alone, so a genuine float64
    # model built --precision single was held to the f64 tolerance it had no
    # way of meeting: a false FAIL, and that configuration could not be
    # verified at all. It now follows whichever side rounds more coarsely.
    rng = np.random.default_rng(5)
    w = numpy_helper.from_array(rng.uniform(-1, 1, (4, 3)).astype(np.float64), "w")
    b = numpy_helper.from_array(rng.uniform(-1, 1, (3,)).astype(np.float64), "b")
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="g0")], "f64model",
        [helper.make_tensor_value_info("x", TensorProto.DOUBLE, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [1, 3])], [w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    import onnx as _onnx
    path = tmp_path / "f64model.onnx"
    _onnx.save(model, str(path))
    from rosenna.verify import verify_model
    for precision in ("f64", "f32"):
        r = verify_model(path, "c", precision, 8, tmp_path / precision)[0]
        assert r.ok, f"--precision {precision}: max_abs={r.max_abs:.3e} max_rel={r.max_rel:.3e}"
    # And the f64 build really is the more accurate one, so the looser bar for
    # the single build is not hiding a wrong answer.
    f64 = verify_model(path, "c", "f64", 8, tmp_path / "again64")[0]
    f32 = verify_model(path, "c", "f32", 8, tmp_path / "again32")[0]
    assert f64.max_abs < f32.max_abs


# ===========================================================================
# From the review of the device-library branch (PR #12). Everything above is
# from the earlier codegen-dense review; both are the same kind of test -- a
# shape the golden set cannot express, pinned after it shipped as a bug.
#
# Regressions from the review of the device-library branch (PR #12).
#
# Every model here is built inline with onnx.helper; each one is a shape the
# golden set does not contain: a tied initializer, an end-only padded pool, a
# real Transpose in a graph with no Add, an Unsqueeze with several negative
# axes, a (1,) Gemm bias, a weights file whose table of contents is short, and
# a Constant LSTM initial state.
# ===========================================================================

def _f32(rng, shape, name):
    return numpy_helper.from_array(rng.uniform(-1, 1, shape).astype(np.float32), name)


def test_tied_initializer_is_loaded_into_every_use(tmp_path):
    # x @ w -> Relu -> @ w: one initializer, two nodes. The plan made one
    # WeightSpec per use; C's loader filled the first and returned, leaving
    # the second all zeros, and Fortran's select-case had two labels for 'w'.
    rng = np.random.default_rng(3)
    w = _f32(rng, (3, 3), "w")
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["g"], name="mm0"),
        helper.make_node("Relu", ["g"], ["r"], name="r0"),
        helper.make_node("MatMul", ["r", "w"], ["y"], name="mm1"),
    ]
    path = save_model(tmp_path, "tied", nodes, [w], (1, 3), (1, 3))
    inputs, expected = _reference(path, [1, 3])
    f_out, c_out = _both_backends(tmp_path, path, "tied", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_average_pool_with_end_only_padding_divides_by_the_window_count(tmp_path):
    # kernel 2, stride 2, pads [0,0,1,1] on a 3x3 input: the last row and
    # column of windows hang off the end by one. ONNX (count_include_pad=0)
    # divides those by the cells that exist; the emitters divided every
    # window by the full kernel because the *begin* pads were zero.
    nodes = [helper.make_node("AveragePool", ["x"], ["y"], name="p0",
                              kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 1, 1])]
    path = save_model(tmp_path, "endpad", nodes, [], (1, 1, 3, 3), (1, 1, 2, 2))
    inputs, expected = _reference(path, [1, 1, 3, 3])
    f_out, c_out = _both_backends(tmp_path, path, "endpad", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_real_transpose_compiles_in_fortran_without_an_add(tmp_path):
    # Transpose(perm=[0,2,1]) on (1,3,4) moves a real axis, so it is a loop
    # over counters c0, c1, ...; the Fortran emitter only declared those
    # when the model also had an Add.
    rng = np.random.default_rng(5)
    w = _f32(rng, (3, 2), "w")
    nodes = [
        helper.make_node("Transpose", ["x"], ["t"], name="t0", perm=[0, 2, 1]),
        helper.make_node("MatMul", ["t", "w"], ["y"], name="mm0"),
    ]
    path = save_model(tmp_path, "tr", nodes, [w], (1, 3, 4), (1, 4, 2))
    inputs, expected = _reference(path, [1, 3, 4])
    f_out, c_out = _both_backends(tmp_path, path, "tr", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_unsqueeze_with_several_negative_axes_folds_to_the_right_shape(tmp_path):
    # Unsqueeze(c[3], axes=[-1, -2]) is (3,1,1) -- negatives count from the
    # OUTPUT rank -- and the fold resolved them against ndim+1, giving
    # (1,1,3). Added to x of shape (1,3,1,3) both broadcast, so a wrong fold
    # is a silently wrong answer, not an error.
    c = numpy_helper.from_array(np.array([1.0, 2.0, 3.0], np.float32), "c")
    axes = numpy_helper.from_array(np.array([-1, -2], np.int64), "axes")
    nodes = [
        helper.make_node("Unsqueeze", ["c", "axes"], ["cu"], name="u0"),
        helper.make_node("Add", ["x", "cu"], ["y"], name="a0"),
    ]
    path = save_model(tmp_path, "unsq", nodes, [c, axes], (1, 3, 1, 3), (1, 3, 1, 3))
    inputs, expected = _reference(path, [1, 3, 1, 3])
    f_out, c_out = _both_backends(tmp_path, path, "unsq", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_gemm_bias_of_length_one_is_rejected(tmp_path):
    # A (1,) bias is a legal ONNX unidirectional broadcast, but the emitters
    # index b[i] for every output, so validation must refuse it rather than
    # let generated code read past a one-element array.
    rng = np.random.default_rng(7)
    w = _f32(rng, (2, 3), "w")
    b = numpy_helper.from_array(np.array([0.5], np.float32), "b")
    nodes = [helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="g0")]
    path = save_model(tmp_path, "b1", nodes, [w, b], (1, 2), (1, 3))
    with pytest.raises(UnsupportedModel, match="bias"):
        build_plan(load_graph(path), dtype="f64", embed=False)


@pytest.mark.parametrize("lang", ["c", "fortran"])
def test_weights_file_missing_a_tensor_returns_status_9(tmp_path, golden_model, lang):
    # A file with the right hash whose table of contents lists fewer tensors
    # than the plan has: init returned 0 and infer ran on zeroed arrays.
    onnx_path = golden_model("gemm_small")
    graph = load_graph(onnx_path)
    plan = build_plan(graph, dtype="f64", embed=False)
    good = tmp_path / "good.rwt"
    write_weights(plan, graph, good)
    data = good.read_bytes()
    # Header layout is write_weights' business; the count is the one
    # little-endian uint32 that equals the number of plan weights and sits
    # in the first 64 bytes.
    n = len(plan.weights)
    off = next(i for i in range(0, 64, 4) if struct.unpack_from("<I", data, i)[0] == n)
    short = bytearray(data)
    struct.pack_into("<I", short, off, 0)
    rc, out = _init_status(tmp_path, lang, onnx_path, "gemm_small", bytes(short))
    assert rc == 0, f"generated init aborted instead of returning a status (exit {rc})"
    assert out.split() == ["9"], out


def test_constant_lstm_initial_state_is_lowered_to_weights(tmp_path):
    # fold.py's docstring says a Constant initial_h/initial_c is folded away
    # and supported; after folding they are initializers, and the plan
    # refused them as "must be a rank-3 value". They are now weights.
    rng = np.random.default_rng(9)
    hidden, n_in = 4, 3
    W = _f32(rng, (1, 4 * hidden, n_in), "W")
    R = _f32(rng, (1, 4 * hidden, hidden), "R")
    B = _f32(rng, (1, 8 * hidden), "B")
    h0 = numpy_helper.from_array(rng.uniform(-1, 1, (1, 1, hidden)).astype(np.float32), "h0")
    c0 = numpy_helper.from_array(rng.uniform(-1, 1, (1, 1, hidden)).astype(np.float32), "c0")
    nodes = [
        helper.make_node("Constant", [], ["h0"], name="k0", value=h0),
        helper.make_node("Constant", [], ["c0"], name="k1", value=c0),
        helper.make_node("LSTM", ["x", "W", "R", "B", "", "h0", "c0"], ["yl"], name="l0",
                         hidden_size=hidden),
        helper.make_node("Flatten", ["yl"], ["y"], name="f0"),
    ]
    path = save_model(tmp_path, "lstmk", nodes, [W, R, B], (1, 1, n_in), (1, hidden))
    plan = build_plan(load_graph(path), dtype="f64", embed=False)
    lstm = next(op for op in plan.ops if op.kind == "lstm")
    assert lstm.init_syms and not lstm.extra_in
    assert {w.name for w in plan.weights} >= {"h0", "c0"}
    inputs, expected = _reference(path, [1, 1, n_in])
    f_out, c_out = _both_backends(tmp_path, path, "lstmk", inputs)
    np.testing.assert_allclose(f_out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-6)


def test_lstm_without_its_y_output_is_refused_not_a_traceback(tmp_path):
    # ONNX lets a graph ask for Y_h alone (outputs ["", "yh"]). The plan reads
    # the first output's shape and raised KeyError('') on the empty name; the
    # answer is an UnsupportedModel naming what is missing.
    rng = np.random.default_rng(9)
    hidden, n_in = 4, 3
    W = _f32(rng, (1, 4 * hidden, n_in), "W")
    R = _f32(rng, (1, 4 * hidden, hidden), "R")
    nodes = [
        helper.make_node("LSTM", ["x", "W", "R"], ["", "yh"], name="l0", hidden_size=hidden),
        helper.make_node("Flatten", ["yh"], ["y"], name="f0"),
    ]
    path = save_model(tmp_path, "lstmy", nodes, [W, R], (1, 1, n_in), (1, hidden))
    with pytest.raises(UnsupportedModel, match="Y"):
        build_plan(load_graph(path), dtype="f64", embed=False)


# --- Softmax ---------------------------------------------------------------

def _softmax_model(path, shape, axis=None, out_shape=None):
    """A Softmax-only graph, so nothing upstream can mask what it computes."""
    attrs = {} if axis is None else {"axis": axis}
    graph = helper.make_graph(
        [helper.make_node("Softmax", ["x"], ["y"], name="s0", **attrs)], "smx",
        [helper.make_tensor_value_info("x", TensorProto.DOUBLE, list(shape))],
        [helper.make_tensor_value_info("y", TensorProto.DOUBLE, list(out_shape or shape))], [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    import onnx as _onnx
    _onnx.save(model, str(path))
    return path


def test_softmax_matches_onnxruntime_on_both_backends(tmp_path):
    path = _softmax_model(tmp_path / "smx.onnx", (1, 7))
    rng = np.random.default_rng(11)
    x = rng.uniform(-4, 4, (1, 7))
    f, c = _both_backends(tmp_path, path, "smx", x)
    session = ort.InferenceSession(str(path))
    want = session.run(None, {"x": x})[0].ravel()
    for got, lang in ((f, "fortran"), (c, "c")):
        assert np.allclose(got, want, rtol=1e-12, atol=1e-14), f"{lang}: {got} != {want}"
    # Whatever else it is, a softmax is a distribution.
    assert abs(float(np.sum(c)) - 1.0) < 1e-12


def test_softmax_over_the_last_axis_of_a_rank4_value(tmp_path):
    """Each of the 2*3*4 rows normalises independently, not the whole buffer."""
    path = _softmax_model(tmp_path / "smx4.onnx", (2, 3, 4, 5), axis=-1)
    rng = np.random.default_rng(12)
    x = rng.uniform(-3, 3, (2, 3, 4, 5))
    # The driver takes a list of cases, each a flat n_in vector: one case here.
    f, c = _both_backends(tmp_path, path, "smx4", x.reshape(1, -1))
    want = ort.InferenceSession(str(path)).run(None, {"x": x})[0].ravel()
    assert np.allclose(c, want, rtol=1e-12, atol=1e-14)
    assert np.allclose(f, want, rtol=1e-12, atol=1e-14)
    # 24 rows each summing to 1, which a single global softmax would fail.
    assert np.allclose(np.asarray(c).reshape(2, 3, 4, 5).sum(axis=-1), 1.0)


def test_softmax_does_not_overflow_on_a_large_logit(tmp_path):
    """exp(800) is inf; subtracting the row maximum is what avoids it.

    Without the max-subtraction pass this returns nan (inf/inf), which is why
    the pass is there rather than being an optimisation.
    """
    path = _softmax_model(tmp_path / "big.onnx", (1, 4))
    x = np.array([[800.0, 799.0, -800.0, 0.0]])
    f, c = _both_backends(tmp_path, path, "big", x)
    want = ort.InferenceSession(str(path)).run(None, {"x": x})[0].ravel()
    for got, lang in ((f, "fortran"), (c, "c")):
        assert np.all(np.isfinite(got)), f"{lang}: {got}"
        assert np.allclose(got, want, rtol=1e-12, atol=1e-14), f"{lang}: {got} != {want}"


def test_softmax_propagates_a_nan_through_the_row_sum(tmp_path):
    """A NaN anywhere in a row makes that row all-NaN, and leaves others alone.

    The maximum is written `v > mx`, so a NaN loses it -- the propagation comes
    from the row sum instead (exp(nan - mx) is nan, so the sum is nan and every
    ratio in the row is nan). A NaN-sticky maximum would give nan - nan for
    every element and lose the reason. See _emit_softmax_c.
    """
    path = _softmax_model(tmp_path / "nan.onnx", (2, 3), axis=-1)
    x = np.array([[1.0, np.nan, 2.0], [1.0, 2.0, 3.0]])
    f, c = _both_backends(tmp_path, path, "nan", x.reshape(1, -1))
    for got, lang in ((np.asarray(f), "fortran"), (np.asarray(c), "c")):
        row0, row1 = got.reshape(2, 3)
        assert np.all(np.isnan(row0)), f"{lang}: the NaN row should be all NaN, got {row0}"
        assert np.all(np.isfinite(row1)), f"{lang}: the clean row should survive, got {row1}"
        assert abs(float(row1.sum()) - 1.0) < 1e-12, f"{lang}: {row1}"


@pytest.mark.parametrize("shape,axis,fragment", [
    ((2, 3, 4), 1, "only the last axis"),
    ((2, 3, 4), -2, "only the last axis"),
    ((2, 3, 4), 0, "only the last axis"),
    ((2, 3, 4), None, "ambiguous across opsets"),
])
def test_softmax_refuses_what_its_loop_does_not_compute(tmp_path, shape, axis, fragment):
    from rosenna.validate import validate
    out = (1,) if axis is None else shape
    path = _softmax_model(tmp_path / "bad.onnx", shape, axis=axis, out_shape=shape)
    with pytest.raises(UnsupportedModel, match=fragment):
        validate(load_graph(path))


# --- BatchNormalization ----------------------------------------------------

def test_conv_batchnorm_folds_away_and_still_matches_onnxruntime(tmp_path):
    """End to end: the fold is only correct if the generated code agrees with ORT.

    The unit tests in test_fold.py check the arithmetic of the rewrite; this
    checks that the rewritten graph, compiled, computes what the original
    model means -- which is the claim that matters, and the one a sign error
    in the broadcast axis would break.
    """
    # float32: onnxruntime has no float64 Conv kernel, so a f64 model cannot be
    # given a reference at all.
    rng = np.random.default_rng(21)
    ini = [numpy_helper.from_array(a.astype(np.float32), n) for a, n in (
        (rng.uniform(-1, 1, (4, 3, 3, 3)), "w"),
        (rng.uniform(-1, 1, 4), "b"),
        (rng.uniform(0.5, 2.0, 4), "scale"),
        (rng.uniform(-1, 1, 4), "B"),
        (rng.uniform(-1, 1, 4), "mean"),
        (rng.uniform(0.5, 2.0, 4), "var"))]
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w", "b"], ["h"], name="c0",
                          kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
         helper.make_node("BatchNormalization", ["h", "scale", "B", "mean", "var"],
                          ["bn"], name="n0", epsilon=1e-5),
         helper.make_node("Relu", ["bn"], ["y"], name="r0")], "convbn",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 6, 6])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 6, 6])], ini)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    import onnx as _onnx
    path = tmp_path / "convbn.onnx"
    _onnx.save(model, str(path))

    # The op is gone by the time anything downstream sees the graph.
    assert not any(n.op == "BatchNormalization" for n in load_graph(path).nodes)

    x = rng.uniform(-2, 2, (1, 3, 6, 6)).astype(np.float32)
    f, c = _both_backends(tmp_path, path, "convbn", x.reshape(1, -1), dtype="f32")
    want = ort.InferenceSession(str(path)).run(None, {"x": x})[0].ravel()
    assert np.allclose(c, want, rtol=1e-5, atol=1e-6), f"c: {np.max(np.abs(c - want)):.3e}"
    assert np.allclose(f, want, rtol=1e-5, atol=1e-6), f"fortran: {np.max(np.abs(f - want)):.3e}"


@pytest.mark.parametrize("c_in,c_out,group,label", [
    (4, 4, 4, "depthwise"),          # one input channel per output channel
    (4, 6, 2, "uneven groups"),      # c_in_per_group 2, c_out_per_group 3
    (6, 6, 3, "square groups"),
])
def test_grouped_conv_reads_only_its_own_group(tmp_path, c_in, c_out, group, label):
    """A grouped Conv is wrong in a way that still runs: it reads the neighbouring
    group's channels. Only a reference catches that, so compare to onnxruntime.

    The uneven case matters most: with c_in_per_group != c_out_per_group an
    off-by-one in the group offset lands inside the buffer and returns
    plausible numbers.
    """
    rng = np.random.default_rng(31 + group)
    w = numpy_helper.from_array(
        rng.uniform(-1, 1, (c_out, c_in // group, 3, 3)).astype(np.float32), "w")
    b = numpy_helper.from_array(rng.uniform(-1, 1, c_out).astype(np.float32), "b")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w", "b"], ["y"], name="c0",
                          kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=group)], "grp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, c_in, 5, 5])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, c_out, 5, 5])], [w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    import onnx as _onnx
    path = tmp_path / f"grp{group}.onnx"
    _onnx.save(model, str(path))

    x = rng.uniform(-2, 2, (1, c_in, 5, 5)).astype(np.float32)
    f, c = _both_backends(tmp_path, path, f"grp{group}", x.reshape(1, -1), dtype="f32")
    want = ort.InferenceSession(str(path)).run(None, {"x": x})[0].ravel()
    for got, lang in ((f, "fortran"), (c, "c")):
        assert np.allclose(got, want, rtol=1e-5, atol=1e-6), \
            f"{label} {lang}: max |diff| {np.max(np.abs(np.asarray(got).ravel() - want)):.3e}"


# --- Pad -------------------------------------------------------------------

def _pad_model(path, in_shape, pads, value=0.0, mode="constant", runtime_pads=False, axes=None):
    rank = len(in_shape)
    if axes is None:
        out = tuple(int(d) + pads[k] + pads[k + rank] for k, d in enumerate(in_shape))
    else:
        norm = [a + rank if a < 0 else a for a in axes]
        out = list(int(d) for d in in_shape)
        for k, a in enumerate(norm):
            out[a] += pads[k] + pads[k + len(axes)]
        out = tuple(out)
    ini = [] if runtime_pads else [
        numpy_helper.from_array(np.array(pads, np.int64), "p"),
        numpy_helper.from_array(np.array(value, np.float64), "v")]
    if axes is not None:
        ini.append(numpy_helper.from_array(np.array(axes, np.int64), "a"))
    ins = [helper.make_tensor_value_info("x", TensorProto.DOUBLE, list(in_shape))]
    if runtime_pads:
        ins.append(helper.make_tensor_value_info("p", TensorProto.INT64, [2 * rank]))
    graph = helper.make_graph(
        [helper.make_node("Pad", ["x", "p"] + ([] if runtime_pads else ["v"])
                          + ([] if axes is None else ["a"]),
                          ["y"], name="p0", mode=mode)], "pad", ins,
        [helper.make_tensor_value_info("y", TensorProto.DOUBLE, list(out))], ini)
    # `axes` only exists from opset 18; onnxruntime rejects a 4-input Pad
    # against the opset-13 schema, so the reference could not even be built.
    opset = 18 if axes is not None else 13
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 9 if axes is not None else 8
    import onnx as _onnx
    _onnx.save(model, str(path))
    return path, out


def test_pad_places_the_input_block_and_fills_the_rest(tmp_path):
    """Asymmetric pads on two axes, so a begins/ends mix-up cannot pass."""
    path, out = _pad_model(tmp_path / "pad.onnx", (1, 2, 4, 5),
                           [0, 0, 1, 2, 0, 0, 3, 1], value=-1.5)
    rng = np.random.default_rng(41)
    x = rng.uniform(-2, 2, (1, 2, 4, 5))
    f, c = _both_backends(tmp_path, path, "pad", x.reshape(1, -1))
    want = ort.InferenceSession(str(path)).run(None, {"x": x})[0].ravel()
    assert np.allclose(c, want) and np.allclose(f, want)
    # The fill value really is the constant, and the block really moved.
    got = np.asarray(c).reshape(out)
    assert got[0, 0, 0, 0] == -1.5
    assert np.allclose(got[0, :, 1:5, 2:7], x)


def test_pad_feeding_a_conv_is_the_shape_that_turns_up(tmp_path):
    """An explicit Pad before a Conv: what an export emits for asymmetric padding."""
    rng = np.random.default_rng(42)
    w = numpy_helper.from_array(rng.uniform(-1, 1, (3, 2, 3, 3)).astype(np.float32), "w")
    pads = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "p")
    graph = helper.make_graph(
        [helper.make_node("Pad", ["x", "p"], ["xp"], name="p0", mode="constant"),
         helper.make_node("Conv", ["xp", "w"], ["y"], name="c0", kernel_shape=[3, 3])],
        "padconv",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 6, 6])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 6, 6])], [w, pads])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    import onnx as _onnx
    path = tmp_path / "padconv.onnx"
    _onnx.save(model, str(path))
    x = rng.uniform(-2, 2, (1, 2, 6, 6)).astype(np.float32)
    f, c = _both_backends(tmp_path, path, "padconv", x.reshape(1, -1), dtype="f32")
    want = ort.InferenceSession(str(path)).run(None, {"x": x})[0].ravel()
    assert np.allclose(c, want, rtol=1e-5, atol=1e-6)
    assert np.allclose(f, want, rtol=1e-5, atol=1e-6)


def test_pad_with_negative_pads_crops(tmp_path):
    """A negative pad removes elements, and the same nest already computes it.

    The output reads FURTHER into the input, so the shift is an addition. It
    was refused at first out of caution; the only thing actually wrong was the
    spelling -- `(c - -1)` is legal C and a Fortran syntax error. The mixed
    case is the one worth testing: cropping the front of an axis while padding
    its back means the read can still run off the end, so the upper bounds
    test has to survive.
    """
    path, out = _pad_model(tmp_path / "crop.onnx", (1, 2, 6, 6),
                           [0, 0, -1, 2, 0, 0, -2, 1], value=7.0)
    assert out == (1, 2, 3, 9), out
    rng = np.random.default_rng(43)
    x = rng.uniform(-2, 2, (1, 2, 6, 6))
    f, c = _both_backends(tmp_path, path, "crop", x.reshape(1, -1))
    want = ort.InferenceSession(str(path)).run(None, {"x": x})[0].ravel()
    assert np.allclose(c, want) and np.allclose(f, want)
    got = np.asarray(c).reshape(out)
    # Cropped off the front of h, and the far end of w is past the input.
    assert np.allclose(got[0, :, 0, 2:8], x[0, :, 1, 0:6])
    assert got[0, 0, 0, 8] == 7.0


@pytest.mark.parametrize("kwargs,fragment", [
    (dict(pads=[0, 0, 1, 1, 0, 0, 1, 1], mode="wrap"), "only 'constant', 'edge' and 'reflect'"),
    # One reflection only: a pad as wide as the axis would need repeated
    # reflection, and the index map would fold to the wrong element.
    (dict(pads=[0, 0, 4, 0, 0, 0, 0, 0], mode="reflect"), "more than one reflection"),
    # A runtime `pads` is an int64 graph input, which the frontend refuses on
    # dtype before validate sees the node at all. Still named, still refused,
    # just earlier -- pinning the message that actually fires rather than the
    # one _validate_pad would have given.
    (dict(pads=[0, 0, 1, 1, 0, 0, 1, 1], runtime_pads=True), "only float32 and float64"),
])
def test_pad_refuses_what_its_loop_does_not_compute(tmp_path, kwargs, fragment):
    from rosenna.validate import validate
    pads = kwargs.pop("pads")
    # A negative pad's output shape is smaller, which the helper computes, so
    # the graph is well-formed and only validate should object.
    path, _ = _pad_model(tmp_path / "bad.onnx", (1, 2, 4, 5), pads, **kwargs)
    with pytest.raises(UnsupportedModel, match=fragment):
        validate(load_graph(path))


@pytest.mark.parametrize("mode", ["edge", "reflect"])
def test_pad_edge_and_reflect_land_on_real_elements(tmp_path, mode):
    """Neither mode ever writes a pad value: both are index maps.

    Asymmetric pads on both spatial axes, so an edge/reflect mix-up or an
    off-by-one in the mirror shows up as a mismatch rather than cancelling.
    """
    path, out = _pad_model(tmp_path / f"{mode}.onnx", (1, 2, 5, 6),
                           [0, 0, 2, 3, 0, 0, 1, 2], mode=mode)
    rng = np.random.default_rng(51)
    x = rng.uniform(-2, 2, (1, 2, 5, 6))
    f, c = _both_backends(tmp_path, path, mode, x.reshape(1, -1))
    want = ort.InferenceSession(str(path)).run(None, {"x": x})[0].ravel()
    assert np.allclose(c, want) and np.allclose(f, want)
    got = np.asarray(c).reshape(out)
    if mode == "edge":
        # The first two rows are copies of the input's first row.
        assert np.allclose(got[0, :, 0, 3:9], x[0, :, 0, :])
        assert np.allclose(got[0, :, 1, 3:9], x[0, :, 0, :])
    else:
        # Mirrored without repeating the edge: row 2 is the input's row 0,
        # so rows 1 and 0 are its rows 1 and 2.
        assert np.allclose(got[0, :, 1, 3:9], x[0, :, 1, :])
        assert np.allclose(got[0, :, 0, 3:9], x[0, :, 2, :])


def test_pad_axes_operand_expands_to_a_full_rank_pads(tmp_path):
    """opset-18 `axes` names which axes `pads` counts; a negative axis counts back."""
    path, out = _pad_model(tmp_path / "axes.onnx", (1, 2, 4, 5),
                           [1, 2, 3, 1], value=-3.0, axes=[2, -1])
    assert out == (1, 2, 8, 8), out
    rng = np.random.default_rng(52)
    x = rng.uniform(-2, 2, (1, 2, 4, 5))
    f, c = _both_backends(tmp_path, path, "axes", x.reshape(1, -1))
    want = ort.InferenceSession(str(path)).run(None, {"x": x})[0].ravel()
    assert np.allclose(c, want) and np.allclose(f, want)
    # Axes 0 and 1 were not named, so they are unpadded.
    assert np.asarray(c).reshape(out).shape[:2] == (1, 2)


# --- 1-D spatial ops -------------------------------------------------------

@pytest.mark.parametrize("op,attrs,c_in,c_out,w_in,w_out", [
    ("Conv", dict(kernel_shape=[3], pads=[1, 1]), 3, 5, 16, 16),
    ("Conv", dict(kernel_shape=[3], pads=[2, 2], strides=[2], dilations=[2]), 3, 5, 16, 8),
    ("Conv", dict(kernel_shape=[3], pads=[1, 1], group=2), 4, 6, 16, 16),
    ("MaxPool", dict(kernel_shape=[3], strides=[2], pads=[1, 1]), 3, 3, 20, 10),
    ("AveragePool", dict(kernel_shape=[2], strides=[2]), 3, 3, 20, 10),
])
def test_one_dimensional_spatial_ops_match_onnxruntime(tmp_path, op, attrs, c_in, c_out,
                                                       w_in, w_out):
    """A 1-D op is the 2-D nest with a height of 1.

    On a flat row-major buffer (N,C,W) and (N,C,1,W) are the same bytes, so
    this needs no loop nest of its own -- but that equivalence is exactly the
    kind of claim that is either right or silently off by a stride, so each
    variant is compared against onnxruntime.
    """
    rng = np.random.default_rng(61)
    ini = []
    inputs = ["x"]
    if op == "Conv":
        group = attrs.get("group", 1)
        ini = [numpy_helper.from_array(
            rng.uniform(-1, 1, (c_out, c_in // group, attrs["kernel_shape"][0])).astype(np.float32), "w"),
            numpy_helper.from_array(rng.uniform(-1, 1, c_out).astype(np.float32), "b")]
        inputs += ["w", "b"]
    graph = helper.make_graph(
        [helper.make_node(op, inputs, ["y"], name="s0", **attrs)], "d1",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, c_in, w_in])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, c_out, w_out])], ini)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    import onnx as _onnx
    path = tmp_path / "d1.onnx"
    _onnx.save(model, str(path))

    x = rng.uniform(-2, 2, (1, c_in, w_in)).astype(np.float32)
    f, c = _both_backends(tmp_path, path, "d1", x.reshape(1, -1), dtype="f32")
    want = ort.InferenceSession(str(path)).run(None, {"x": x})[0].ravel()
    for got, lang in ((f, "fortran"), (c, "c")):
        assert np.allclose(got, want, rtol=1e-5, atol=1e-6), \
            f"{op} {lang}: max |diff| {np.max(np.abs(np.asarray(got).ravel() - want)):.3e}"


@pytest.mark.parametrize("attrs,fragment", [
    (dict(kernel_shape=[3, 3], pads=[1, 1]), "they must agree"),
    (dict(kernel_shape=[3], strides=[1, 1], pads=[1, 1]), "a 1-D op takes 1"),
    (dict(kernel_shape=[3], pads=[1, 1, 1, 1]), "a 1-D op takes 2"),
])
def test_a_1d_op_refuses_attributes_of_the_wrong_arity(tmp_path, attrs, fragment):
    from rosenna.validate import validate
    graph = helper.make_graph(
        [helper.make_node("MaxPool", ["x"], ["y"], name="s0", **attrs)], "bad",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 16])], [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    import onnx as _onnx
    path = tmp_path / "bad.onnx"
    _onnx.save(model, str(path))
    with pytest.raises(UnsupportedModel, match=fragment):
        validate(load_graph(path))


# --- GRU -------------------------------------------------------------------

def _gru_model(path, lbr, bias, init, T=5, B=1, I=4, H=6, seed=31):
    rng = np.random.default_rng(seed)
    ini = [numpy_helper.from_array(rng.uniform(-1, 1, (1, 3 * H, I)).astype(np.float32), "W"),
           numpy_helper.from_array(rng.uniform(-1, 1, (1, 3 * H, H)).astype(np.float32), "R")]
    inputs = ["x", "W", "R"]
    if bias:
        ini.append(numpy_helper.from_array(
            rng.uniform(-1, 1, (1, 6 * H)).astype(np.float32), "B"))
        inputs.append("B")
    else:
        inputs.append("")
    if init:
        inputs += ["", "h0"]
        ini.append(numpy_helper.from_array(
            rng.uniform(-1, 1, (1, B, H)).astype(np.float32), "h0"))
    graph = helper.make_graph(
        [helper.make_node("GRU", inputs, ["Y", "Yh"], name="g0",
                          hidden_size=H, linear_before_reset=lbr)], "gru",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [T, B, I])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [T, 1, B, H]),
         helper.make_tensor_value_info("Yh", TensorProto.FLOAT, [1, B, H])], ini)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    import onnx as _onnx
    _onnx.save(model, str(path))
    return path, (T, B, I)


@pytest.mark.parametrize("lbr", [0, 1])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("init", [True, False])
def test_gru_matches_onnxruntime_on_both_backends(tmp_path, lbr, bias, init):
    """Both readings of linear_before_reset, with and without B and initial_h.

    linear_before_reset changes the ARITHMETIC of the h gate, not its spelling:
    the reset gate multiplies the state before the recurrent matmul when it is
    0 and the matmul's result when it is 1, and the two agree only where r is
    1. PyTorch exports 1 while the ONNX default is 0, so picking one would
    have been wrong half the time -- hence both, and hence this matrix.
    """
    path, (T, B, I) = _gru_model(tmp_path / f"gru{lbr}{int(bias)}{int(init)}.onnx",
                                 lbr, bias, init)
    rng = np.random.default_rng(77)
    x = rng.uniform(-1.5, 1.5, (T, B, I)).astype(np.float32)
    f, c = _both_backends(tmp_path, path, path.stem, x.reshape(1, -1), dtype="f32")
    want = np.concatenate([a.ravel() for a in
                           ort.InferenceSession(str(path)).run(None, {"x": x})])
    for got, lang in ((f, "fortran"), (c, "c")):
        assert np.allclose(np.asarray(got).ravel(), want, rtol=1e-4, atol=1e-5), \
            f"lbr={lbr} bias={bias} init={init} {lang}: " \
            f"max |diff| {np.max(np.abs(np.asarray(got).ravel() - want)):.3e}"


def test_the_two_linear_before_reset_readings_actually_differ(tmp_path):
    """Guards the matrix above: if they agreed, it would be testing one thing twice."""
    rng = np.random.default_rng(78)
    x = rng.uniform(-1.5, 1.5, (5, 1, 4)).astype(np.float32)
    outs = []
    for lbr in (0, 1):
        path, _ = _gru_model(tmp_path / f"d{lbr}.onnx", lbr, True, True)
        outs.append(ort.InferenceSession(str(path)).run(None, {"x": x})[0].ravel())
    assert not np.allclose(outs[0], outs[1], rtol=1e-3), \
        "the two readings gave the same answer; this model does not distinguish them"


@pytest.mark.parametrize("attrs,fragment", [
    (dict(direction="reverse"), "only 'forward'"),
    (dict(clip=1.0), "clip"),
    (dict(activations=["Sigmoid", "Tanh", "Tanh"]), "custom activations"),
])
def test_gru_refuses_what_changes_the_recurrence(tmp_path, attrs, fragment):
    from rosenna.validate import validate
    T, B, I, H = 3, 1, 4, 5
    rng = np.random.default_rng(79)
    ini = [numpy_helper.from_array(rng.uniform(-1, 1, (1, 3 * H, I)).astype(np.float32), "W"),
           numpy_helper.from_array(rng.uniform(-1, 1, (1, 3 * H, H)).astype(np.float32), "R")]
    graph = helper.make_graph(
        [helper.make_node("GRU", ["x", "W", "R"], ["Y", "Yh"], name="g0",
                          hidden_size=H, **attrs)], "bad",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [T, B, I])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [T, 1, B, H]),
         helper.make_tensor_value_info("Yh", TensorProto.FLOAT, [1, B, H])], ini)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    import onnx as _onnx
    path = tmp_path / "bad.onnx"
    _onnx.save(model, str(path))
    with pytest.raises(UnsupportedModel, match=fragment):
        validate(load_graph(path))
