"""Compile the generated code and compare its output against onnxruntime."""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .emit_c import emit_c
from .emit_fortran import emit_fortran
from .frontend import load_graph
from .plan import build_plan, validate_model_name
from .weights import write_weights

_TOL = {"f32": (1e-5, 1e-6), "f64": (1e-9, 1e-12)}
_NUMPY = {"f32": np.float32, "f64": np.float64}
_SEED = 0
_MAX_ATTEMPTS = 10


class VerificationError(RuntimeError):
    """A comparison would not mean anything (e.g. the reference is dead)."""


def _live_reference(session, shape, dtype, seed=0, batch=8, max_attempts=10):
    """Resample input batches until the onnxruntime reference itself is alive.

    Non-degeneracy is a property of the randomly generated fixture, not of
    the code under test: several golden models (e.g. gemm_small) have no
    manual_seed, so their weights differ on every regeneration, and an
    all-zero reference (a dead model, e.g. every pre-activation negative
    into a final ReLU) is a property of that draw of weights -- correct
    generated code reproducing a dead model must *also* be all zero, so no
    assertion on our own output can tell the two cases apart. The fix
    belongs here, on the reference, before we ever build or run anything.

    `dtype` is a numpy dtype (e.g. np.float64), not a plan dtype string
    ("f32"/"f64") -- this helper draws and feeds inputs at that numpy dtype
    directly.

    Returns (inputs, expected) for the first batch whose reference has at
    least two non-zero values across the whole batch, or (None, None) if
    max_attempts batches all came back dead.

    Moved here (from tests/test_emit_fortran.py) so that `rosenna/gate.py`
    can reuse it without importing test code; tests/test_emit_fortran.py
    re-exports the same name so every existing `from tests.test_emit_fortran
    import _live_reference` keeps working unchanged.
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        inputs = rng.uniform(-2, 2, (batch, int(np.prod(shape)))).astype(dtype)
        expected = np.array([
            np.concatenate([o.ravel() for o in session.run(
                None, {session.get_inputs()[0].name: row.reshape(shape).astype(np.float32)})])
            for row in inputs
        ])
        if np.count_nonzero(expected) >= 2:
            return inputs, expected
    return None, None


@dataclass(frozen=True)
class VerifyResult:
    lang: str          # "fortran" | "c"
    cases: int
    max_abs: float
    max_rel: float
    ok: bool


# ROSENNA_CC / ROSENNA_FC point verify at a different compiler. The generated
# code is plain C11 and Fortran 2008, so any conforming compiler should build
# and run it and reach the same numbers; a CI job or a developer checking a new
# toolchain needs a way to say so without editing this file.
#
# -Wall -Wextra -std= are added only for a compiler whose basename says it
# takes them; flang, ifx and nvfortran get -O2 and nothing else.
_GNU_STYLE = ("gcc", "gfortran", "cc", "clang")


def _compiler(role: str) -> str:
    return os.environ.get(f"ROSENNA_{role}", {"CC": "gcc", "FC": "gfortran"}[role])


def _warn_flags(tool: str, std: str) -> list:
    from pathlib import Path as _P
    return ["-Wall", "-Wextra", std] if _P(tool).name.startswith(_GNU_STYLE) else []


def verify_model(model_path, lang: str, dtype: str | None, cases: int, workdir,
                 embed: bool | None = None, name: str | None = None) -> list:
    """Generate, compile and run `lang` backend(s) for `model_path`, and compare to onnxruntime.

    Draws `cases` random inputs from a fixed seed and compares every backend's output
    against the same onnxruntime reference.

    Tolerance (controller ruling R11): keyed on the ONNX model's OWN dtype -- the
    precision onnxruntime actually computes the reference in -- not on `dtype`/the
    plan's dtype (i.e. not on `--precision`). Generating f64 code from a float32 model
    makes our own arithmetic more precise, but it cannot make onnxruntime's float32
    reference any more accurate, so a float32 model is always compared at the
    fp32-appropriate tolerance, even when `--precision double` asked for an f64 build:
    the tighter tolerance is reserved for models whose reference computation is itself
    float64.

    Non-degeneracy: the reference model's own weights can be dead (every output zero,
    e.g. a final ReLU whose pre-activations are all negative), most likely on one of the
    unseeded golden generators. A verification run that "passes" by reproducing an
    all-zero reference proves nothing, so the input batch is resampled (fixed seed,
    incrementing draw) until the onnxruntime reference itself has at least two non-zero
    values, and a VerificationError is raised -- not silently swallowed into ok=True --
    if no resampled batch is alive.
    """
    workdir = Path(workdir)
    graph = load_graph(model_path, name)
    plan = build_plan(graph, dtype=dtype, embed=embed)
    validate_model_name(plan.model)

    # Model's own dtype, read before --precision is applied: this is what onnxruntime
    # actually computes the reference in, regardless of what precision we generate.
    # It also fixes the element type onnxruntime will accept in the input feed: a
    # genuine float64 model rejects a float32 array outright, so the drawn inputs are
    # cast to the model's dtype rather than unconditionally to float32.
    model_dtype = graph.values[graph.inputs[0]].dtype

    session = ort.InferenceSession(str(model_path))
    shapes = [i.shape for i in session.get_inputs()]
    inputs, expected = _live_inputs(session, shapes, cases, model_path, _NUMPY[model_dtype])

    backends = ["fortran", "c"] if lang == "both" else [lang]
    # Tolerance follows the COARSER of the reference's precision and the build's.
    # The reference is computed at the model's own dtype, so f64 code compared
    # against an f32 reference is still held to the f32 tolerance (generating
    # f64 cannot make onnxruntime's f32 answer more accurate). The converse was
    # a false FAIL: a genuine float64 model built `--precision single` was held
    # to the f64 tolerance it had no way of meeting, so that configuration could
    # not be verified at all. Whichever side rounds more coarsely sets the bar.
    compare_dtype = "f32" if "f32" in (model_dtype, plan.dtype) else "f64"
    rtol, atol = _TOL[compare_dtype]
    atol = atol + _cancellation_atol(plan, compare_dtype, expected)
    results = []
    for backend in backends:
        backend_dir = workdir / backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        # Both backends now embed by default: a plan that embeds has no
        # weights file to load in either language (every weight is baked into
        # the generated source -- an initialized `protected` module array in
        # Fortran, deliberately not `parameter`, and ROSENNA_CONST in C).
        if not plan.embed:
            write_weights(plan, graph, backend_dir / f"{plan.model}.rwt")
        got = _run_backend(backend, plan, backend_dir, inputs)
        abs_err = np.abs(got - expected)
        denom = np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
        rel_err = abs_err / denom
        ok = bool(np.all(abs_err <= atol + rtol * np.abs(expected)))
        results.append(VerifyResult(backend, cases, float(abs_err.max()), float(rel_err.max()), ok))
    return results


def _live_inputs(session, shapes, cases: int, model_path, np_dtype):
    """Resample input batches (fixed seed) until the onnxruntime reference is alive.

    See verify_model's docstring: a dead reference is a property of the model's own
    (possibly unseeded) weights, not of the code under test, and no comparison against
    it can distinguish "correct" from "also dead".

    A model with several graph inputs (an LSTM's initial hidden and cell state)
    gets one drawn row per case holding all of them concatenated in declaration
    order -- exactly the layout the generated infer(x, y) expects -- and the row
    is split back up to feed onnxruntime.
    """
    names = [i.name for i in session.get_inputs()]
    lens = [int(np.prod(sh)) for sh in shapes]
    n = sum(lens)
    for attempt in range(_MAX_ATTEMPTS):
        rng = np.random.default_rng(_SEED + attempt)
        inputs = rng.uniform(-2, 2, (cases, n)).astype(np_dtype)
        expected = []
        for row in inputs:
            feed, off = {}, 0
            for name, sh, ln in zip(names, shapes, lens):
                feed[name] = row[off:off + ln].reshape(sh).astype(np_dtype)
                off += ln
            # Every graph output, flat, in declaration order: the y layout.
            expected.append(np.concatenate([o.ravel() for o in session.run(None, feed)]))
        expected = np.array(expected)
        if np.count_nonzero(expected) >= 2:
            return inputs, expected
    raise VerificationError(
        f"{model_path}: onnxruntime reference is all-zero across {_MAX_ATTEMPTS} resampled "
        f"batches of {cases} cases; this model's weights compute a dead network, so a "
        f"passing comparison here would not demonstrate anything")


def _fortran_driver(name: str, n_in: int, n_out: int, dtype: str, embed: bool) -> str:
    real_kind = "real64" if dtype == "f64" else "real32"
    # An embedded plan has no `_init`: every weight is already an initialized
    # `protected` array in the generated module, resident from program load.
    # (`protected`, not `parameter`: gfortran -fopenacc will not take a
    # `declare` on a named constant. See _emit_embedded_weights.)
    init = "" if embed else f"""
    call {name}_init('{name}.rwt', status)
    if (status /= 0) then
        print *, 'init status', status
        stop 1
    end if"""
    return f"""
program verify_main
    use {name}_model
    use iso_fortran_env, only: {real_kind}
    implicit none
    real({real_kind}) :: x({n_in}), y({n_out})
    integer :: status, i, ncases
    read(*,*) ncases{init}
    do i = 1, ncases
        read(*,*) x
        call {name}_infer(x, y)
        print '({n_out}(es24.16,1x))', y
    end do
end program
"""


def _c_driver(name: str, n_in: int, n_out: int, dtype: str, embed: bool) -> str:
    c_type = "double" if dtype == "f64" else "float"
    fmt = "%lf" if dtype == "f64" else "%f"
    # An embedded plan has no `_init`: every weight is already a ROSENNA_CONST
    # array in the header, resident from program load.
    init = "" if embed else (
        f'int status = {name}_init("{name}.rwt");\n'
        f'    if (status != 0) {{ printf("init status %d\\n", status); return 1; }}\n    ')
    return f"""
#include <stdio.h>
#include "{name}.h"
int main(void) {{
    {c_type} x[{n_in}], y[{n_out}];
    {init}int ncases;
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


def _run(step: str, backend: str, args: list, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess step, turning a failure into a VerificationError with full context.

    A raw CalledProcessError reaches the user as an opaque Python traceback, with the
    compiler's own diagnostic buried inside `.stderr` where nothing prints it. For a
    command whose entire job is compiling and running generated code against a user's
    own model, that diagnostic -- not a traceback -- is the useful thing on screen. Wrap
    the call here, where the step name and backend are known, so `verify_model`'s caller
    (`cli.main`) can report it the same way as `UnsupportedModel`: `rosenna: <message>`,
    non-zero exit, no traceback.
    """
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True, **kwargs)
    except subprocess.CalledProcessError as e:
        raise VerificationError(
            f"{backend}: {step} failed running `{' '.join(args)}` "
            f"(exit {e.returncode}):\n{e.stderr}") from e
    except FileNotFoundError as e:
        # A compiler that is simply not installed, or not on PATH, is the most
        # common first-run failure of all; without this arm it escapes as a
        # traceback naming a file the user never asked about.
        raise VerificationError(
            f"{backend}: {step} could not run `{args[0]}`: {e.strerror}. "
            f"Is it installed and on PATH?") from e


def _run_backend(backend: str, plan, workdir: Path, inputs):
    name = plan.model
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    dtype = plan.dtype
    if backend == "fortran":
        (workdir / f"{name}_model.F90").write_text(emit_fortran(plan))
        (workdir / "verify_main.f90").write_text(_fortran_driver(name, n_in, n_out, dtype, plan.embed))
        # Compile the module to an object, archive it, and link the driver
        # against the archive -- the library form -- rather than compiling
        # both sources together, mirroring the C backend below.
        fc = _compiler("FC")
        fw = _warn_flags(fc, "-std=f2008")
        _run("compile", backend,
             [fc, "-O2", *fw, "-c", f"{name}_model.F90"],
             cwd=workdir)
        # lib<name>_f.a, not lib<name>.a (ruling R13): the C backend's own
        # archive is lib<name>.a, and although verify's fortran/c backends
        # build in separate directories (no collision here), the two names
        # must never be the same anywhere a caller might build both recipes
        # in one place (cli.py's `generate --lang both`, in particular).
        _run("archive", backend,
             ["ar", "rcs", f"lib{name}_f.a", f"{name}_model.o"],
             cwd=workdir)
        _run("compile/link", backend,
             [fc, "-O2", *fw, "-o", "verify_run",
              "verify_main.f90", f"lib{name}_f.a"],
             cwd=workdir)
    elif backend == "c":
        source, header = emit_c(plan)
        (workdir / f"{name}.c").write_text(source)
        (workdir / f"{name}.h").write_text(header)
        (workdir / "verify_main.c").write_text(_c_driver(name, n_in, n_out, dtype, plan.embed))
        # Compile the generated source to an object, archive it, and link the
        # driver against the archive -- the library form -- rather than
        # compiling both sources together, so `verify` exercises the same
        # delivery shape a downstream host build uses.
        cc = _compiler("CC")
        cw = _warn_flags(cc, "-std=c11")
        _run("compile", backend,
             [cc, "-O2", *cw, "-c", f"{name}.c", "-o", f"{name}.o"],
             cwd=workdir)
        _run("archive", backend,
             ["ar", "rcs", f"lib{name}.a", f"{name}.o"],
             cwd=workdir)
        _run("compile/link", backend,
             [cc, "-O2", *cw, "-o", "verify_run",
              "verify_main.c", f"lib{name}.a", "-lm"],
             cwd=workdir)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    stdin = f"{len(inputs)}\n" + "\n".join(" ".join(repr(float(v)) for v in row) for row in inputs)
    out = _run("run", backend, ["./verify_run"], cwd=workdir, input=stdin).stdout
    return np.array([[float(v) for v in line.split()] for line in out.strip().splitlines()])


def _reduction_depth(op) -> int:
    """How many terms the longest single summation inside this op adds up."""
    if op.kind == "gemm":
        return op.n_in
    if op.kind == "conv":
        return op.spatial.c_in * op.spatial.kh * op.spatial.kw
    if op.kind == "avgpool":
        return op.spatial.kh * op.spatial.kw
    if op.kind == "lstm":
        return op.lstm.input_size + op.lstm.hidden
    return 1


def _cancellation_atol(plan, model_dtype: str, expected) -> float:
    """Slack for the one error a relative-to-output tolerance cannot express.

    onnxruntime and the generated code compute in the same precision but not in
    the same order -- ORT blocks and vectorises its convolutions and GEMMs. The
    classical bound on summing n terms is n * eps * sum|terms|, and when the sum
    cancels, sum|terms| is far larger than |result|: the error is then large
    relative to the output while both implementations are perfectly correct.
    A tolerance written as rtol * |expected| cannot see that and will reject a
    correct implementation.

    sum|terms| is not observable from here, so the batch's largest |expected|
    stands in for the scale of the computation. That is a proxy, and a
    deliberately generous one -- it is the only quantity available that tracks
    the magnitude the accumulation actually works at.

    mnist is the case that forced this: a 256-term MatMul, an output of
    magnitude 0.05 carrying 4.6e-6 of absolute error, and the same plan built
    in double matching an independent float64 reference to 2.4e-15. Models
    whose deepest reduction is short get a negligible bump and keep the flat
    tolerance in practice.
    """
    depth = max((_reduction_depth(op) for op in plan.ops), default=1)
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    return depth * float(np.finfo(_NUMPY[model_dtype]).eps) * scale
