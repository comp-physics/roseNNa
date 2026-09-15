"""Register-blocked dense layers, and the layer-parallel infer_one.

A dense layer computes GEMM_BLOCK output columns per pass over its input
vector, in every backend and both languages. infer_one runs ONE sample with
a launch per op and the thread index over the op's output elements, the
intermediate activations in static device buffers rather than per-thread
locals -- the form a whole-field model (a conv net over a grid) needs. It
shares the op bodies with the header's per-point infer.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest

from rosenna.emit_c import GEMM_BLOCK, emit_c, emit_c_recipe, large_locals
from rosenna.emit_fortran import emit_fortran
from rosenna.emit_kernel import emit_kernel
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from rosenna.rt_header import rt_header
from rosenna.weights import write_weights
from tests.conftest import _assert_warning_free
from tests.test_device_c import _omp_cc

DENSE = "gemm_big"
CONV = "conv_padding"
LSTM = "lstm_gemm_hid"


def test_wide_dense_layers_are_register_blocked_in_both_languages(tmp_path, golden_model):
    # GEMM_BLOCK output columns per pass over the input vector, in the header's
    # infer (which the batched kernel calls per point) and in the Fortran
    # module, for a layer at least GEMM_BLOCK_MIN_IN wide; narrower layers
    # (all of gemm_big's) keep the one-column loop.
    plan = _wide_plan(tmp_path)
    source, header = emit_c(plan)
    assert f"a{GEMM_BLOCK - 1} += sj *" in header and "const double sj = " in header
    assert f"a{GEMM_BLOCK - 1} = a{GEMM_BLOCK - 1} + sj *" in emit_fortran(plan)
    cu = emit_kernel(plan)
    assert "wide_infer(" in cu and "__shared__" not in cu
    narrow = build_plan(load_graph(golden_model(DENSE)), dtype="f64", embed=True)
    assert "sj" not in emit_c(narrow)[1] and "sj" not in emit_fortran(narrow)


def _wide_plan(tmp_path):
    from onnx import helper, numpy_helper
    from tests.conftest import save_model
    rng = np.random.default_rng(2)
    w = lambda shape, name: numpy_helper.from_array(rng.uniform(-0.1, 0.1, shape).astype(np.float32), name)
    nodes = [helper.make_node("Gemm", ["x", "w0", "b0"], ["a"], name="g0", transB=1),
             helper.make_node("Tanh", ["a"], ["b"], name="t0"),
             helper.make_node("Gemm", ["b", "w1", "b1"], ["y"], name="g1", transB=1)]
    path = save_model(tmp_path, "wide", nodes, [w((64, 128), "w0"), w((64,), "b0"), w((3, 64), "w1"), w((3,), "b1")],
                      (1, 128), (1, 3))
    return build_plan(load_graph(path), dtype="f64", embed=True)


def test_infer_one_is_declared_and_defined_unless_the_plan_has_an_lstm(golden_model):
    for name in (DENSE, CONV):
        plan = build_plan(load_graph(golden_model(name)), dtype="f64", embed=True)
        source, header = emit_c(plan)
        assert f"int {name}_infer_one(const double *ROSENNA_RESTRICT x, double *ROSENNA_RESTRICT y, void *stream);" in header
        assert f"int {name}_infer_one(" in source            # the omp fallback
        assert f'extern "C" int {name}_infer_one(' in emit_kernel(plan)
        assert f'bind(C, name="{name}_infer_one")' in emit_fortran(plan)
    plan = build_plan(load_graph(golden_model(LSTM)), dtype="f64", embed=True)
    source, header = emit_c(plan)
    assert "_infer_one" not in header and "_infer_one" not in source
    assert "_infer_one" not in emit_kernel(plan) and "_infer_one" not in emit_fortran(plan)


_HOST = """
#include <math.h>
#include <stdio.h>
#include "{name}.h"
int main(void) {{
    {init}
    double x[{n_in}], y[{n_out}], y1[{n_out}];
    for (int i = 0; i < {n_in}; ++i) x[i] = sin(0.37 * i) - 0.2;
    {name}_infer(x, y);
    if ({name}_infer_one(x, y1, 0)) return 3;
    for (int i = 0; i < {n_out}; ++i)
        if (!(fabs(y[i] - y1[i]) <= 1e-12 + 1e-12 * fabs(y[i]))) {{ printf("differ at %d: %.17g %.17g\\n", i, y[i], y1[i]); return 4; }}
    printf("agree\\n");
    return 0;
}}
"""


@pytest.mark.parametrize("name,embed", [(DENSE, True), (DENSE, False), (CONV, True), ("batchnet", True),
                                        ("maxpool_padding", True), ("avgpool_basic", True)])
def test_omp_infer_one_matches_the_per_point_infer_on_the_host(tmp_path, golden_model, name, embed):
    # The omp fallback of infer_one: one target loop per op over the static
    # buffers. On a host-only build the loops are plain loops, so the
    # layer-parallel form is checked numerically here, without a GPU.
    graph = load_graph(golden_model(name)); plan = build_plan(graph, dtype="f64", embed=embed)
    source, header = emit_c(plan)
    (tmp_path / f"{name}.c").write_text(source); (tmp_path / f"{name}.h").write_text(header)
    (tmp_path / "Makefile").write_text(emit_c_recipe(plan))
    init = ""
    if not embed:
        write_weights(plan, graph, tmp_path / f"{name}.rwt")
        init = f'if ({name}_init("{name}.rwt")) return 2;'
    cc = _omp_cc()
    r = subprocess.run(["make", f"CC={cc}", "ROSENNA_BACKEND=omp", "ROSENNA_OFFLOAD_FLAGS=-fopenmp"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    _assert_warning_free("gcc", r.stderr)
    (tmp_path / "host.c").write_text(_HOST.format(name=name, n_in=plan.input.shape[0],
                                                   n_out=plan.output.shape[0], init=init))
    r = subprocess.run([cc, "-O2", "-Wall", "-Wextra", "-std=c11", "-fopenmp", "host.c", f"lib{name}.a",
                        "-lm", "-o", "host"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    _assert_warning_free("gcc", r.stderr)
    r = subprocess.run(["./host"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "agree", (r.returncode, r.stdout, r.stderr)


# --- on a GPU: hipcc-built archive, infer_batch (staged) and infer_one against the host infer ---

_DEV = """
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "rosenna_rt.h"
#include "{name}.h"
#define NPTS 1000
int main(void) {{
    {init}
    double *hx = (double*)malloc(sizeof(double) * NPTS * {n_in}), *hy = (double*)malloc(sizeof(double) * NPTS * {n_out});
    double *ref = (double*)malloc(sizeof(double) * NPTS * {n_out}), *one = (double*)malloc(sizeof(double) * {n_out});
    for (int c = 0; c < NPTS * {n_in}; ++c) hx[c] = sin(0.37 * c) - 0.2;
    double *dx, *dy;
    ROSENNA_MALLOC(&dx, sizeof(double) * NPTS * {n_in}); ROSENNA_MALLOC(&dy, sizeof(double) * NPTS * {n_out});
    ROSENNA_MEMCPY_H2D(dx, hx, sizeof(double) * NPTS * {n_in});
    if ({name}_infer_batch(NPTS, dx, dy, 0)) return 3;
    if ({name}_sync(0)) return 3;
    {memcpy_d2h}(hy, dy, sizeof(double) * NPTS * {n_out});
    {infer_one}
    /* reference: the host infer, on the host (file-loaded) or via a one-thread kernel (embedded) */
    {reference}
    int bad = 0;
    for (int c = 0; c < NPTS * {n_out}; ++c) if (!(fabs(hy[c] - ref[c]) <= 1e-9 + 1e-9 * fabs(ref[c]))) ++bad;
    printf("infer_batch: %d of %d mismatch\\n", bad, NPTS * {n_out});
    {check_one}
    return bad ? 4 : 0;
}}
"""


def _gpu_available():
    return shutil.which("hipcc") and shutil.which("rocminfo") and \
        "gfx" in subprocess.run(["rocminfo"], capture_output=True, text=True).stdout


@pytest.mark.parametrize("name", [DENSE, CONV])
def test_hip_kernels_match_the_host_infer(tmp_path, golden_model, name):
    if not _gpu_available():
        pytest.skip("no hipcc + AMD GPU")
    graph = load_graph(golden_model(name)); plan = build_plan(graph, dtype="f64", embed=False)
    source, header = emit_c(plan)
    for fn, text in ((f"{name}.c", source), (f"{name}.h", header), (f"{name}_kernel.cu", emit_kernel(plan)),
                     ("rosenna_rt.h", rt_header()), ("Makefile", emit_c_recipe(plan))):
        (tmp_path / fn).write_text(text)
    write_weights(plan, graph, tmp_path / f"{name}.rwt")
    r = subprocess.run(["make", "ROSENNA_BACKEND=hip"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    (tmp_path / "driver.cu").write_text(_DEV.format(
        name=name, n_in=n_in, n_out=n_out, init=f'if ({name}_init("{name}.rwt")) return 2;',
        memcpy_d2h="hipMemcpy_d2h",
        infer_one=f"if ({name}_infer_one(dx + 3 * {n_in}, dy, 0)) return 5; if ({name}_sync(0)) return 5; "
                  f"hipMemcpy_d2h(one, dy, sizeof(double) * {n_out});",
        reference=f"for (int p = 0; p < NPTS; ++p) {name}_infer(hx + p * {n_in}, ref + p * {n_out});",
        check_one=f"int bad1 = 0; for (int i = 0; i < {n_out}; ++i) if (!(fabs(one[i] - ref[3 * {n_out} + i]) <= 1e-9 + 1e-9 * fabs(ref[3 * {n_out} + i]))) ++bad1; "
                  f'printf("infer_one: %d of %d mismatch\\n", bad1, {n_out}); bad += bad1;'))
    (tmp_path / "driver.cu").write_text(
        "#define hipMemcpy_d2h(h, d, n) hipMemcpy((h), (d), (n), hipMemcpyDeviceToHost)\n"
        + (tmp_path / "driver.cu").read_text())
    r = subprocess.run(["hipcc", "-O2", "driver.cu", "-L.", f"-l{name}", "-o", "driver"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run(["./driver"], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)


def _whole_field_plan(tmp_path):
    """A conv net over a 64x64 field: 8 x 72 x 72 + 8 x 68 x 68 doubles of activations, 660 KB."""
    from onnx import helper, numpy_helper
    from tests.conftest import save_model
    rng = np.random.default_rng(1)
    w = lambda shape, name: numpy_helper.from_array(rng.uniform(-0.1, 0.1, shape).astype(np.float32), name)
    nodes = [helper.make_node("Conv", ["x", "w0"], ["a"], name="c0", kernel_shape=[5, 5]),
             helper.make_node("Tanh", ["a"], ["b"], name="t0"),
             helper.make_node("Conv", ["b", "w1"], ["c"], name="c1", kernel_shape=[5, 5]),
             helper.make_node("Tanh", ["c"], ["d"], name="t1"),
             helper.make_node("Conv", ["d", "w2"], ["y"], name="c2", kernel_shape=[5, 5])]
    path = save_model(tmp_path, "field", nodes, [w((8, 1, 5, 5), "w0"), w((8, 8, 5, 5), "w1"), w((1, 8, 5, 5), "w2")],
                      (1, 1, 76, 76), (1, 1, 64, 64))
    return build_plan(load_graph(path), dtype="f64", embed=True)


def test_infer_batch_of_a_whole_field_model_runs_infer_one_per_point(tmp_path):
    # The per-point infer of this plan would hold 660 KB of locals, more than a
    # device thread's stack (hipcc: "stack frame size exceeds limit"), so the
    # archive's infer_batch must not instantiate it: it loops infer_one over the
    # points instead, and the omp fallback does the same.
    plan = _whole_field_plan(tmp_path)
    assert large_locals(plan)
    cu = emit_kernel(plan)
    assert "field_kernel" not in cu and "field_infer_one(x + (size_t)p * 5776" in cu
    source, _ = emit_c(plan)
    batch = source[source.index("int field_infer_batch("):]
    assert "field_infer_one(x + (size_t)p * 5776" in batch and "field_infer(x" not in batch.split("\n}")[0]
    small = build_plan(load_graph(_golden(tmp_path)), dtype="f64", embed=True)
    assert not large_locals(small)


def _golden(tmp_path):
    from tests.conftest import save_model
    from onnx import helper, numpy_helper
    w = numpy_helper.from_array(np.eye(2, dtype=np.float32), "w")
    return save_model(tmp_path, "tiny", [helper.make_node("MatMul", ["x", "w"], ["y"], name="m")], [w], (1, 2), (1, 2))
