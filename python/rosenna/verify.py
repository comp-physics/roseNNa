"""Compile the generated code and compare its output against onnxruntime."""
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .emit_c import emit_c
from .emit_fortran import emit_fortran
from .frontend import load_graph
from .plan import build_plan
from .weights import write_weights

_TOL = {"f32": (1e-5, 1e-6), "f64": (1e-9, 1e-12)}
_SEED = 0
_MAX_ATTEMPTS = 10


class VerificationError(RuntimeError):
    """A comparison would not mean anything (e.g. the reference is dead)."""


@dataclass(frozen=True)
class VerifyResult:
    lang: str          # "fortran" | "c"
    cases: int
    max_abs: float
    max_rel: float
    ok: bool


def verify_model(model_path, lang: str, dtype: str | None, cases: int, workdir) -> list:
    """Generate, compile and run `lang` backend(s) for `model_path`, and compare to onnxruntime.

    Draws `cases` random inputs from a fixed seed and compares every backend's output
    against the same onnxruntime reference, at rtol/atol keyed by the plan's dtype.

    Non-degeneracy: the reference model's own weights can be dead (every output zero,
    e.g. a final ReLU whose pre-activations are all negative), most likely on one of the
    unseeded golden generators. A verification run that "passes" by reproducing an
    all-zero reference proves nothing, so the input batch is resampled (fixed seed,
    incrementing draw) until the onnxruntime reference itself has at least two non-zero
    values, and a VerificationError is raised -- not silently swallowed into ok=True --
    if no resampled batch is alive.
    """
    workdir = Path(workdir)
    graph = load_graph(model_path)
    plan = build_plan(graph, dtype=dtype)

    session = ort.InferenceSession(str(model_path))
    shape = session.get_inputs()[0].shape
    inputs, expected = _live_inputs(session, shape, cases, model_path)

    backends = ["fortran", "c"] if lang == "both" else [lang]
    rtol, atol = _TOL[plan.dtype]
    results = []
    for backend in backends:
        backend_dir = workdir / backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        write_weights(plan, graph, backend_dir / f"{plan.model}.rwt")
        got = _run_backend(backend, plan, backend_dir, inputs)
        abs_err = np.abs(got - expected)
        denom = np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
        rel_err = abs_err / denom
        ok = bool(np.all(abs_err <= atol + rtol * np.abs(expected)))
        results.append(VerifyResult(backend, cases, float(abs_err.max()), float(rel_err.max()), ok))
    return results


def _live_inputs(session, shape, cases: int, model_path):
    """Resample input batches (fixed seed) until the onnxruntime reference is alive.

    See verify_model's docstring: a dead reference is a property of the model's own
    (possibly unseeded) weights, not of the code under test, and no comparison against
    it can distinguish "correct" from "also dead".
    """
    input_name = session.get_inputs()[0].name
    n = int(np.prod(shape))
    for attempt in range(_MAX_ATTEMPTS):
        rng = np.random.default_rng(_SEED + attempt)
        inputs = rng.uniform(-2, 2, (cases, n)).astype(np.float32)
        expected = np.array([
            session.run(None, {input_name: row.reshape(shape).astype(np.float32)})[0].ravel()
            for row in inputs
        ])
        if np.count_nonzero(expected) >= 2:
            return inputs, expected
    raise VerificationError(
        f"{model_path}: onnxruntime reference is all-zero across {_MAX_ATTEMPTS} resampled "
        f"batches of {cases} cases; this model's weights compute a dead network, so a "
        f"passing comparison here would not demonstrate anything")


def _fortran_driver(name: str, n_in: int, n_out: int, dtype: str) -> str:
    real_kind = "real64" if dtype == "f64" else "real32"
    return f"""
program verify_main
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
"""


def _c_driver(name: str, n_in: int, n_out: int, dtype: str) -> str:
    c_type = "double" if dtype == "f64" else "float"
    fmt = "%lf" if dtype == "f64" else "%f"
    return f"""
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
"""


def _run_backend(backend: str, plan, workdir: Path, inputs):
    name = plan.model
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    dtype = plan.dtype
    if backend == "fortran":
        (workdir / f"{name}_model.f90").write_text(emit_fortran(plan))
        (workdir / "verify_main.f90").write_text(_fortran_driver(name, n_in, n_out, dtype))
        subprocess.run(
            ["gfortran", "-O2", "-Wall", "-Wextra", "-o", "verify_run",
             f"{name}_model.f90", "verify_main.f90"],
            cwd=workdir, check=True, capture_output=True, text=True)
    elif backend == "c":
        source, header = emit_c(plan)
        (workdir / f"{name}.c").write_text(source)
        (workdir / f"{name}.h").write_text(header)
        (workdir / "verify_main.c").write_text(_c_driver(name, n_in, n_out, dtype))
        subprocess.run(
            ["gcc", "-O2", "-Wall", "-Wextra", "-std=c11", "-o", "verify_run",
             f"{name}.c", "verify_main.c", "-lm"],
            cwd=workdir, check=True, capture_output=True, text=True)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    stdin = f"{len(inputs)}\n" + "\n".join(" ".join(repr(float(v)) for v in row) for row in inputs)
    out = subprocess.run(["./verify_run"], cwd=workdir, input=stdin,
                         capture_output=True, text=True, check=True).stdout
    return np.array([[float(v) for v in line.split()] for line in out.strip().splitlines()])
