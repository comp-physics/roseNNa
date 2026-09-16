"""Every golden model, run under AddressSanitizer and UndefinedBehaviorSanitizer.

A compiler warning cannot see the bug class this catches. The generated code
is loop nests over fixed-size local arrays whose bounds come from the plan, so
the way it goes wrong is an index computed from the wrong extent -- which reads
and writes past a stack array and produces plausible numbers. That is exactly
what happened once already: an activation's loop was bounded by the previous
op's output length, and a branching graph ran 40 iterations over a double[5].
gcc compiled it silently, and it matched onnxruntime on every model that did
not branch.

ASan sees the access itself, so it does not depend on anyone having thought of
the shape of graph that triggers it.
"""
import os
import re
import shutil
import subprocess

import pytest

from rosenna.emit_c import emit_c
from rosenna.emit_fortran import emit_fortran
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from rosenna.weights import write_weights
from tests.test_golden_suite import GOLDEN

SAN = ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-g", "-O1"]
# ASan reports on stderr and, with the flag above, exits non-zero. Both are
# checked: a report that somehow did not set the exit status still fails.
_REPORT = re.compile(r"AddressSanitizer|runtime error|LeakSanitizer|SEGV", re.I)

_C_DRIVER = """
#include <stdio.h>
#include <stdlib.h>
#include "{name}.h"

#define NPTS 3

int main(void) {{
    static double x[NPTS * {n_in}], y[NPTS * {n_out}], yb[NPTS * {n_out}];
    for (int i = 0; i < NPTS * {n_in}; ++i) x[i] = 0.25 + 0.01 * (double)(i % 17);
    {init}
    for (int p = 0; p < NPTS; ++p) {name}_infer(x + p * {n_in}, y + p * {n_out});
    if ({name}_infer_batch(NPTS, x, yb, NULL) != 0) return 2;
    double s = 0.0;
    for (int i = 0; i < NPTS * {n_out}; ++i) s += y[i] + yb[i];
    printf("%g\\n", s);
    return 0;
}}
"""

_F_DRIVER = """
program san
    use {name}_model
    use iso_fortran_env, only: real64
    implicit none
    real(real64) :: x({n_in}), y({n_out})
    integer :: i
    {init}
    do i = 1, {n_in}
        x(i) = 0.25_real64 + 0.01_real64 * real(mod(i - 1, 17), real64)
    end do
    call {name}_infer(x, y)
    print '(es16.8)', sum(y)
end program
"""


def _sanitize(work, name, plan, graph, lang):
    """Build one backend under the sanitizers, run it, and return its output."""
    env = {**os.environ, "UBSAN_OPTIONS": "print_stacktrace=1", "ASAN_OPTIONS": "detect_leaks=0"}
    if lang == "c":
        source, header = emit_c(plan)
        (work / f"{name}.c").write_text(source)
        (work / f"{name}.h").write_text(header)
        init = "" if plan.embed else f'if ({name}_init("{name}.rwt") != 0) return 1;'
        (work / "drv.c").write_text(_C_DRIVER.format(
            name=name, n_in=plan.input.shape[0], n_out=plan.output.shape[0], init=init))
        build = ["gcc", *SAN, "-std=c11", "-I.", "drv.c", f"{name}.c", "-lm", "-o", "run"]
    else:
        (work / f"{name}_model.F90").write_text(emit_fortran(plan))
        init = "" if plan.embed else f'call {name}_init("{name}.rwt", i)'
        (work / "drv.f90").write_text(_F_DRIVER.format(
            name=name, n_in=plan.input.shape[0], n_out=plan.output.shape[0], init=init))
        build = ["gfortran", *SAN, f"{name}_model.F90", "drv.f90", "-o", "run"]
    b = subprocess.run(build, cwd=work, capture_output=True, text=True, env=env)
    assert b.returncode == 0, b.stderr[-2500:]
    r = subprocess.run(["./run"], cwd=work, capture_output=True, text=True, env=env, timeout=600)
    return r


def _can_sanitize(tool, source, tmp_path):
    """Can this toolchain actually build a sanitized binary?

    Probed, not assumed: Homebrew's gcc on macOS accepts -fsanitize=address
    and then fails at link with `ld: library 'asan' not found`, because the
    runtime ships with Apple's clang rather than with gcc. Checking the
    compiler exists is not the same question.
    """
    if not shutil.which(tool):
        return False
    d = tmp_path / f"probe_{tool}"
    d.mkdir(exist_ok=True)
    (d / source).write_text("int main(void){return 0;}\n" if source.endswith(".c")
                            else "program p\nend program\n")
    r = subprocess.run([tool, *SAN, source, "-o", "probe"], cwd=d, capture_output=True, text=True)
    return r.returncode == 0


@pytest.mark.parametrize("lang", ["c", "fortran"])
@pytest.mark.parametrize("name", GOLDEN)
def test_golden_model_is_clean_under_asan_and_ubsan(name, lang, tmp_path, golden_model):
    tool, probe = ("gcc", "p.c") if lang == "c" else ("gfortran", "p.f90")
    if not _can_sanitize(tool, probe, tmp_path):
        pytest.skip(f"{tool} cannot link a sanitized binary here")
    safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
    graph = load_graph(golden_model(name), safe)
    plan = build_plan(graph, dtype="f64")
    if not plan.embed:
        write_weights(plan, graph, tmp_path / f"{safe}.rwt")
    r = _sanitize(tmp_path, safe, plan, graph, lang)
    assert not _REPORT.search(r.stderr), f"{name}/{lang}:\n{r.stderr[-3000:]}"
    assert r.returncode == 0, f"{name}/{lang} exited {r.returncode}:\n{r.stderr[-3000:]}"
