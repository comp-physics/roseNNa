import os
import shutil
import subprocess
import numpy as np
import onnxruntime as ort
import pytest
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from rosenna.weights import write_weights
from rosenna.emit_c import emit_c
from tests.test_emit_fortran import _live_reference


def _omp_cc():
    for cand in ("gcc-15", "gcc-14", "gcc-13", "gcc"):
        path = shutil.which(cand)
        if not path:
            continue
        probe = subprocess.run([cand, "-fopenmp", "-x", "c", "-", "-o", os.devnull],
                               input="int main(void){return 0;}", capture_output=True, text=True)
        if probe.returncode == 0:
            return cand
    pytest.skip("no C compiler with -fopenmp found")


def _write(tmp_path, name, plan, graph):
    source, header = emit_c(plan)
    (tmp_path / f"{name}.c").write_text(source)
    (tmp_path / f"{name}.h").write_text(header)
    if not plan.embed:
        write_weights(plan, graph, tmp_path / f"{name}.rwt")


HOST = """
#include <stdio.h>
#include "{name}.h"
int main(void) {{
    int n; {init}
    if (scanf("%d", &n) != 1) return 1;
    double x[64 * {n_in}], y[64 * {n_out}];
    for (int c = 0; c < n * {n_in}; ++c) if (scanf("%lf", &x[c]) != 1) return 1;
#ifdef _OPENMP
    #pragma omp target teams loop map(to: x[0:n*{n_in}]) map(from: y[0:n*{n_out}])
#endif
    for (int p = 0; p < n; ++p) {name}_infer(x + p * {n_in}, y + p * {n_out});   /* the host's own region calls the header inline */
    for (int p = 0; p < n; ++p) {{ for (int i = 0; i < {n_out}; ++i) printf("%.17e ", y[p * {n_out} + i]); printf("\\n"); }}
    return 0;
}}
"""


def _build_and_run(tmp_path, name, plan, graph, cc, flags, inputs, env=None):
    _write(tmp_path, name, plan, graph)
    init = "" if plan.embed else f'if ({name}_init("{name}.rwt")) return 2;'
    (tmp_path / "host.c").write_text(HOST.format(name=name, n_in=plan.input.shape[0], n_out=plan.output.shape[0], init=init))
    objs = ["host.c"]
    if not plan.embed:
        r = subprocess.run([cc, *flags, "-c", f"{name}.c"], cwd=tmp_path, capture_output=True, text=True)
        assert r.returncode == 0 and r.stderr == "", r.stderr
        subprocess.run(["ar", "rcs", f"lib{name}.a", f"{name}.o"], cwd=tmp_path, check=True)
        objs.append(f"lib{name}.a")
    r = subprocess.run([cc, *flags, *objs, "-lm", "-o", "host"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0 and r.stderr == "", r.stderr
    stdin = f"{len(inputs)}\n" + " ".join(repr(float(v)) for v in inputs.ravel())
    return subprocess.run(["./host"], cwd=tmp_path, input=stdin, capture_output=True, text=True, env=env)


@pytest.mark.parametrize("name", ["gemm_small", "gemm_big", "gemm_nobias", "droplet", "batchnet"])
@pytest.mark.parametrize("embed", [True, False])
def test_host_region_calls_header_inline_and_matches(tmp_path, golden_model, name, embed):
    graph = load_graph(golden_model(name))
    plan = build_plan(graph, dtype="f64", embed=embed)
    session = ort.InferenceSession(golden_model(name))
    inputs, expected = _live_reference(session, session.get_inputs()[0].shape, np.float64, seed=5, batch=8)
    r = _build_and_run(tmp_path, name, plan, graph, _omp_cc(), ["-O2", "-Wall", "-Wextra", "-std=c11", "-fopenmp"], inputs)
    assert r.returncode == 0, r.stderr
    got = np.array([[float(v) for v in line.split()] for line in r.stdout.strip().splitlines()])
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_target_regions_are_real(tmp_path, golden_model):
    # A host-only libgomp refuses a target region under OMP_TARGET_OFFLOAD=MANDATORY. If the pragmas
    # were missing or ignored the program would succeed; this is the cheapest evidence without a GPU.
    name = "gemm_small"
    graph = load_graph(golden_model(name)); plan = build_plan(graph, dtype="f64")
    inputs = np.full((1, plan.input.shape[0]), 0.5)
    r = _build_and_run(tmp_path, name, plan, graph, _omp_cc(), ["-O2", "-std=c11", "-fopenmp"], inputs,
                       env={**os.environ, "OMP_TARGET_OFFLOAD": "MANDATORY"})
    assert r.returncode != 0 and "MANDATORY" in r.stderr


def test_plain_compiler_without_openmp_still_matches(tmp_path, golden_model):
    # Every decoration is inert under a compiler with no -fopenmp and no CUDA; the numbers must not change.
    name = "gemm_small"
    graph = load_graph(golden_model(name)); plan = build_plan(graph, dtype="f64")
    cc = shutil.which("clang") or shutil.which("cc") or _omp_cc()
    session = ort.InferenceSession(golden_model(name))
    inputs, expected = _live_reference(session, session.get_inputs()[0].shape, np.float64, seed=6, batch=4)
    r = _build_and_run(tmp_path, name, plan, graph, cc, ["-O2", "-Wall", "-Wextra", "-std=c11"], inputs)
    assert r.returncode == 0, r.stderr
    got = np.array([[float(v) for v in line.split()] for line in r.stdout.strip().splitlines()])
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_header_carries_exactly_the_two_macros(golden_model):
    from rosenna.emit_c import emit_c
    from rosenna.frontend import load_graph as lg
    # A CUDA/HIP host must see __host__ __device__ and __constant__; nothing else in the header may mention CUDA.
    # Controller ruling P2: take the golden path through the golden_model fixture (tests.conftest has no
    # standalone golden_path function). Controller ruling P3: a third macro, ROSENNA_RESTRICT, sits next to
    # the two above so `restrict` -- not a keyword once this header reaches a C++ (nvcc) translation unit --
    # never appears bare in a signature; assert all three macro names are present.
    plan = build_plan(lg(golden_model("gemm_small")), dtype="f64")
    _, header = emit_c(plan)
    assert header.count("__CUDACC__") == 1 and "__host__ __device__" in header and "__constant__" in header
    assert "ROSENNA_DEVICE_FN" in header and "ROSENNA_CONST" in header and "ROSENNA_RESTRICT" in header
