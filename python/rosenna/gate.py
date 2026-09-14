"""rosenna gpu-gate: the script a user runs on a real GPU machine.

None of the CUDA/HIP path had ever been compiled or run on the machine that
wrote it (no nvcc, hipcc, or GPU). This script is the evidence that fact
cannot produce: it generates the gemm_big plan embedded and file-loaded, in
both languages, builds the omp-backend C archive and the Fortran library
with the host compiler and, under --backend cuda|hip, the native-kernel
archive with the device compiler as well (ruling R21: the per-point host
harness links the host compiler's own archive in every backend, the
infer_batch driver links the device compiler's), then runs three harnesses
-- a microfd-shaped per-point host in C, the same in Fortran, and a host
that hands device-resident data to infer_batch -- each compared against
onnxruntime and timed per point. Every command, every line of its output,
the compiler versions and the timings go into gate-report.md; a failure at
any step still writes the report and the process exits 1.

`--host-fallback` drops the OMP_TARGET_OFFLOAD=MANDATORY requirement so the
omp backend can be exercised end to end on a machine with no accelerator
(this is how the test suite runs this script); without it, a machine with no
working offload device fails loudly instead of silently falling back to the
host, which is the whole point of running under MANDATORY in the first
place.

Device residency is not claimed anywhere until this script has actually run
on a GPU machine and its report recorded. For CUDA that has now happened:
`--backend cuda` PASSes on an A100 under NVIDIA HPC SDK 25.11 (nvc,
nvfortran -mp=gpu -gpu=cc80, nvcc 13.0), with zero cudaMemcpy inside the
timed infer_batch call. For HIP too: `--backend hip` PASSes on an MI210
(gfx90a) under ROCm 7.2.0 (amdclang, amdflang -fopenmp --offload-arch=gfx90a,
hipcc) and under the TheRock AFAR 23.2.1 drop, with zero transfers inside the
roctx-scoped 4-step loop of every harness (rocprofv3, the nsys check's twin
over both the HIP API trace and the memory-copy trace).
"""
import csv
import io
import os
import platform
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .emit_c import emit_c, emit_c_recipe
from .emit_fortran import emit_fortran, emit_fortran_recipe
from .emit_kernel import emit_kernel
from .frontend import load_graph
from .golden import golden_generator_run, golden_model_path
from .plan import build_plan, validate_model_name
from .rt_header import rt_header
from .verify import _live_reference
from .weights import write_weights

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL = "gemm_big"
_TIMED_ITERS = 1_000_000
# Every timed harness runs this many time steps over the same device-resident
# points, the way a solver calls the model once per step. The transfer check
# brackets the whole step loop, so a copy that recurred per step -- the
# weights re-mapped on each target region entry, say -- would be counted
# NSTEPS times, not hidden inside a single call.
_NSTEPS = 4
_RTOL, _ATOL = 1e-5, 1e-6
_RUN_TIMEOUT = 300
# Ruling R24: -Wall -Wextra -std=c11|f2008 are added only for a compiler whose
# basename says it takes them; any other --cc/--fc (nvc, nvfortran, amdclang,
# amdflang, flang, icx, ifx) gets -O2 and the user's --flags, nothing else.
_GNU_STYLE_PREFIXES = ("gcc", "gfortran", "cc", "clang")
# The generated C sources one configuration directory holds; the device
# compiler's archive is built from a copy of them in its own subdirectory.
_C_SOURCE_FILES = ("{name}.c", "{name}.h", "rosenna_rt.h", "{name}_kernel.cu", "{name}.mk")


class _Report:
    """Accumulates gate-report.md; every command and every output line goes in."""

    def __init__(self):
        self.lines = []

    def h(self, text: str, level: int = 2) -> None:
        self.lines.append(f"\n{'#' * level} {text}\n")

    def p(self, text: str) -> None:
        self.lines.append(text)

    def block(self, label: str, text: str) -> None:
        self.lines.append(f"{label}:\n```\n{text}\n```")

    def command(self, label: str, args: list, cwd=None) -> None:
        """Log one command, shell-quoted so a multi-word argument reads back as one."""
        where = f" (in {cwd})" if cwd is not None else ""
        self.lines.append(f"\n**{label}**{where}\n\n```\n$ {shlex.join(str(a) for a in args)}\n```")

    def outcome(self, proc) -> None:
        self.lines.append(f"exit status: {proc.returncode}")
        if proc.stdout:
            self.block("stdout", proc.stdout)
        if proc.stderr:
            self.block("stderr", proc.stderr)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


class _FakeProc:
    """Stands in for subprocess.CompletedProcess when the executable itself is missing."""

    def __init__(self, returncode, stderr):
        self.returncode = returncode
        self.stdout = ""
        self.stderr = stderr


def _sh(report: _Report, label: str, args: list, cwd=None, env=None, input_text=None,
        timeout=_RUN_TIMEOUT):
    """Run one subprocess step, logging the command and its full output either way."""
    report.command(label, args, cwd)
    try:
        proc = subprocess.run(args, cwd=cwd, env=env, input=input_text,
                              capture_output=True, text=True, timeout=timeout)
    except OSError as e:
        # A missing executable, or one the previous step left unrunnable:
        # recorded as a failed step, so the remaining configurations still run.
        proc = _FakeProc(127, f"{args[0]}: cannot run ({e.strerror})")
    except subprocess.TimeoutExpired as e:
        proc = _FakeProc(124, f"timed out after {timeout}s\nstdout so far:\n{e.stdout}\nstderr so far:\n{e.stderr}")
    report.outcome(proc)
    return proc


def _link_archive(lib: Path, cwd: Path) -> list:
    """`-L<dir> -l<name>` for lib<name>.a, relative to cwd (see the hipcc note at its use)."""
    assert lib.name.startswith("lib") and lib.suffix == ".a", lib
    return [f"-L{lib.parent.relative_to(cwd)}", f"-l{lib.name[3:-2]}"]


def _gnu_style(compiler: str) -> bool:
    return Path(shlex.split(compiler)[0]).name.startswith(_GNU_STYLE_PREFIXES)


def _c_flags(cc: str) -> list:
    """The C flags the gate adds before the user's --flags (ruling R24)."""
    return ["-O2", "-Wall", "-Wextra", "-std=c11"] if _gnu_style(cc) else ["-O2"]


def _f_flags(fc: str) -> list:
    """The Fortran flags the gate adds before the user's --flags (ruling R24)."""
    return ["-O2", "-Wall", "-Wextra", "-std=f2008"] if _gnu_style(fc) else ["-O2"]


def _record_versions(report: _Report, cc: str, fc: str, devcc) -> None:
    report.h("toolchain", 3)
    report.p(f"platform: {platform.platform()}")
    for label, exe in (("cc", cc), ("fc", fc), ("devcc", devcc)):
        if not exe:
            continue
        # --devcc may carry its own arguments ("nvcc -ccbin nvc++"), so it is
        # split like a shell word list wherever it becomes argv.
        try:
            proc = subprocess.run([*shlex.split(exe), "--version"], capture_output=True, text=True)
            text = proc.stdout or proc.stderr or "(no output)"
        except FileNotFoundError as e:
            text = f"{exe}: not found ({e.strerror})"
        report.block(f"{label} ({exe}) --version", text)


def _ensure_model(report: _Report) -> Path:
    onnx_path = golden_model_path(_REPO_ROOT, _MODEL)
    if not onnx_path.exists():
        with golden_generator_run(_REPO_ROOT, _MODEL) as (argv, cwd, env):
            _sh(report, "generate the golden gemm_big model", argv, cwd=cwd, env=env)
    return onnx_path


def _generate(outdir: Path, onnx_path: Path, embed: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    graph = load_graph(str(onnx_path))
    plan = build_plan(graph, dtype="f64", embed=embed)
    validate_model_name(plan.model)
    name = plan.model
    (outdir / f"{name}_model.F90").write_text(emit_fortran(plan))
    (outdir / f"{name}_fortran.mk").write_text(emit_fortran_recipe(plan))
    source, header = emit_c(plan)
    (outdir / f"{name}.c").write_text(source)
    (outdir / f"{name}.h").write_text(header)
    (outdir / f"{name}.mk").write_text(emit_c_recipe(plan))
    (outdir / f"{name}_kernel.cu").write_text(emit_kernel(plan))
    (outdir / "rosenna_rt.h").write_text(rt_header())
    if not plan.embed:
        write_weights(plan, graph, outdir / f"{name}.rwt")
    return plan


@dataclass(frozen=True)
class _CLibs:
    """The C archives one configuration builds (ruling R21).

    `host`: lib<name>.a built by the HOST compiler (ROSENNA_BACKEND=omp with
    the host offload flags) in the configuration directory, for the per-point
    C harness in every backend. A file-loaded model's weight arrays reach the
    host compiler's offload region only through the `declare target` device
    copies that this build makes and that init's `target update` fills; an
    nvcc/hipcc build of <name>.c has neither (_OPENMP is not defined there),
    so linking that archive into a per-point OpenMP/OpenACC host leaves the
    offload loop reading weights that do not exist on the device.
    `dev`: lib<name>.a built by the device compiler (ROSENNA_BACKEND=cuda|hip)
    from a copy of the sources in <backend>_lib/, for the .cu driver of the
    infer_batch harness. None under --backend omp, where that harness links
    `host` instead. Either is None when its build failed.
    """
    host: Path | None
    dev: Path | None


def _build_c_libs(report: _Report, outdir: Path, plan, cc, flags, backend, devcc, devflags) -> _CLibs:
    name = plan.model
    label = "embedded" if plan.embed else "file-loaded"
    # CFLAGS is passed explicitly (ruling R24): the recipe's own default is the
    # gcc-style set, which must never reach a vendor host compiler.
    args = ["make", "-f", f"{name}.mk", "ROSENNA_BACKEND=omp", f"CC={cc}",
            f"CFLAGS={' '.join(_c_flags(cc))}", f"ROSENNA_OFFLOAD_FLAGS={flags}"]
    proc = _sh(report, f"build c library ({label}, backend=omp, host compiler: "
                       "serves the per-point harness)", args, cwd=outdir)
    host = outdir / f"lib{name}.a"
    if not (proc.returncode == 0 and host.exists()):
        host = None
    if backend == "omp":
        return _CLibs(host, None)
    dev_dir = outdir / f"{backend}_lib"
    dev_dir.mkdir(exist_ok=True)
    for f in _C_SOURCE_FILES:
        shutil.copyfile(outdir / f.format(name=name), dev_dir / f.format(name=name))
    args = ["make", "-f", f"{name}.mk", f"ROSENNA_BACKEND={backend}", f"DEVCC={devcc}"]
    if devflags:
        args.append(f"DEVFLAGS={devflags}")
    proc = _sh(report, f"build c library ({label}, backend={backend}, device compiler, in "
                       f"{dev_dir.name}/: serves the infer_batch harness)", args, cwd=dev_dir)
    dev = dev_dir / f"lib{name}.a"
    if not (proc.returncode == 0 and dev.exists()):
        dev = None
    return _CLibs(host, dev)


def _build_fortran_lib(report: _Report, outdir: Path, plan, fc, flags) -> bool:
    name = plan.model
    label = "embedded" if plan.embed else "file-loaded"
    # FFLAGS explicitly, for the same reason as CFLAGS above (ruling R24):
    # flang rejects the recipe's default -std=f2008.
    args = ["make", "-f", f"{name}_fortran.mk", f"FC={fc}", f"FFLAGS={' '.join(_f_flags(fc))}",
            f"ROSENNA_OFFLOAD_FLAGS={flags}"]
    proc = _sh(report, f"build fortran library ({label})", args, cwd=outdir)
    return proc.returncode == 0 and (outdir / f"lib{name}_f.a").exists()


def _stdin_for(inputs) -> str:
    return f"{len(inputs)}\n" + "\n".join(" ".join(repr(float(v)) for v in row) for row in inputs)


def _check_output(report: _Report, stdout: str, expected) -> tuple:
    lines = [l for l in stdout.strip().splitlines() if l.strip()]
    data_lines = [l for l in lines if not l.startswith("TIMING")]
    timing_lines = [l for l in lines if l.startswith("TIMING")]
    if len(data_lines) != len(expected):
        report.p(f"FAIL: expected {len(expected)} output rows from onnxruntime's own batch, "
                 f"got {len(data_lines)}")
        return False, None
    got = np.array([[float(v) for v in l.split()] for l in data_lines])
    try:
        np.testing.assert_allclose(got, expected, rtol=_RTOL, atol=_ATOL)
    except AssertionError as e:
        report.block("FAIL: output does not match the onnxruntime reference", str(e))
        return False, None
    report.p(f"matches the onnxruntime reference (rtol={_RTOL}, atol={_ATOL})")
    ns = None
    if timing_lines:
        ns = float(timing_lines[-1].split()[-1])
        report.p(f"{ns:.3f} ns per point")
    else:
        report.p("FAIL: no TIMING line in the harness output")
        return False, None
    return True, ns


_C_HARNESS1 = """/* rosenna gpu-gate: microfd-shaped per-point host, C. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "{name}.h"
#ifdef ROSENNA_GATE_MARKERS
#include {marker_header}
#endif
int main(void) {{
    {init}
    int n;
    if (scanf("%d", &n) != 1) return 1;
    double *x = malloc(sizeof(double) * (size_t)n * {n_in});
    double *y = malloc(sizeof(double) * (size_t)n * {n_out});
    for (int c = 0; c < n * {n_in}; ++c) if (scanf("%lf", &x[c]) != 1) return 1;
#ifdef _OPENMP
    #pragma omp target teams distribute parallel for map(to: x[0:n*{n_in}]) map(from: y[0:n*{n_out}])
#endif
    for (int p = 0; p < n; ++p) {name}_infer(x + p * {n_in}, y + p * {n_out});   /* microfd's own target loop calling the header inline */
    for (int p = 0; p < n; ++p) {{
        for (int i = 0; i < {n_out}; ++i) printf("%.17e ", y[p * {n_out} + i]);
        printf("\\n");
    }}
    /* Timing: one offload region over ntime points, each reading its own
       input slot and writing its own output slot, with the points tiled on
       the host and mapped BEFORE the clock starts (ruling R16, as in the
       infer_batch driver). Both of those matter for the comparison this
       gate's report invites: a timing loop that cycles a handful of points
       measures cached reads and a million-way store collision on a few
       slots, not the per-point cost of the same distinct-point work
       infer_batch does, and a map inside the timed window charges this path
       for a transfer the other one makes outside it. */
    long ntime = {ntime}L;
    double *xt = malloc(sizeof(double) * (size_t)ntime * {n_in});
    double *yt = malloc(sizeof(double) * (size_t)ntime * {n_out});
    for (long t = 0; t < ntime; ++t) {{
        long b = t % n;
        for (int i = 0; i < {n_in}; ++i) xt[t * {n_in} + i] = x[b * {n_in} + i];
    }}
#ifdef _OPENMP
    #pragma omp target enter data map(to: xt[0:ntime*{n_in}]) map(alloc: yt[0:ntime*{n_out}])
#endif
    /* {nsteps} time steps over the resident data, as a solver would call the
       model once per step; the transfer check brackets the whole loop. */
#ifdef ROSENNA_GATE_MARKERS
    {marker_push}("rosenna_timed");
#endif
    double t0 = omp_get_wtime();
    for (int step = 0; step < {nsteps}; ++step) {{
#ifdef _OPENMP
        #pragma omp target teams distribute parallel for
#endif
        for (long p = 0; p < ntime; ++p) {name}_infer(xt + p * {n_in}, yt + p * {n_out});
    }}
    double t1 = omp_get_wtime();
#ifdef ROSENNA_GATE_MARKERS
    {marker_pop}();
#endif
#ifdef _OPENMP
    #pragma omp target exit data map(from: yt[0:ntime*{n_out}]) map(delete: xt[0:ntime*{n_in}])
#endif
    /* The timed loop's own results are checked, not just the correctness
       loop's: xt/yt that failed to map would leave this timing a loop over
       garbage, and nothing else here would notice. Every timed point is a
       tile of one of the n correctness points, so yt[t] must reproduce
       y[t % n] -- to a tolerance, since the two loops are separate regions
       the compiler may schedule (and contract) differently. */
    for (long t = 0; t < ntime; ++t) {{
        long b = t % n;
        for (int i = 0; i < {n_out}; ++i) {{
            double got = yt[t * {n_out} + i], want = y[b * {n_out} + i];
            if (!(fabs(got - want) <= 1e-9 + 1e-9 * fabs(want))) {{
                printf("TIMED MISMATCH at point %ld slot %d: %.17e vs %.17e\\n",
                       t, i, got, want);
                return 5;
            }}
        }}
    }}
    printf("TIMING %.6f\\n", (t1 - t0) * 1.0e9 / ((double)ntime * {nsteps}));
    free(x); free(y); free(xt); free(yt);
    return 0;
}}
"""

# The Fortran harnesses are .F90 so the marker bracket can be preprocessed
# in the same way; the push/pop are bound to the C symbols directly.
_F_MARKERS = """
#ifdef ROSENNA_GATE_MARKERS
    interface
        function rosenna_range_push(msg) bind(C, name="{marker_push}") result(r)
            use iso_c_binding, only: c_char, c_int
            character(kind=c_char), dimension(*), intent(in) :: msg
            integer(c_int) :: r
        end function
        function rosenna_range_pop() bind(C, name="{marker_pop}") result(r)
            use iso_c_binding, only: c_int
            integer(c_int) :: r
        end function
    end interface
    integer(c_int) :: irange
#endif
"""

_F_HARNESS2 = """
program host
    use {name}_model
    use iso_fortran_env, only: real64
    use iso_c_binding, only: c_int, c_null_char
    implicit none
    real(real64), allocatable :: x(:,:), y(:,:), xt(:,:), yt(:,:)
    integer :: n, p, status, b, ntime, t
    integer(8) :: c0, c1, crate
    real(real64) :: ns_per_point
{f_markers}
    status = 0
    {init_lines}
    read(*,*) n
    allocate(x({n_in}, n), y({n_out}, n))
    read(*,*) x
    !$omp target teams distribute parallel do map(to: x) map(from: y)
    do p = 1, n
        call {name}_infer(x(:, p), y(:, p))
    end do
    do p = 1, n
        print '({n_out}(es24.16,1x))', y(:, p)
    end do
    ! Timing: one offload region over ntime points, each reading its own
    ! input column and writing its own output column, with the points tiled
    ! on the host and mapped BEFORE the clock starts (ruling R16, as in the
    ! infer_batch driver). Both of those matter for the comparison this
    ! gate's report invites: a timing loop that cycles a handful of points
    ! measures cached reads and a million-way store collision on a few
    ! columns, not the per-point cost of the same distinct-point work
    ! infer_batch does, and a map inside the timed window charges this path
    ! for a transfer the other one makes outside it.
    ntime = {ntime}
    allocate(xt({n_in}, ntime), yt({n_out}, ntime))
    do t = 1, ntime
        b = mod(t - 1, n) + 1
        xt(:, t) = x(:, b)
    end do
    !$omp target enter data map(to: xt) map(alloc: yt)
    ! {nsteps} time steps over the resident data, as a solver would call the
    ! model once per step; the transfer check brackets the whole loop.
#ifdef ROSENNA_GATE_MARKERS
    irange = rosenna_range_push("rosenna_timed" // c_null_char)
#endif
    call system_clock(count=c0, count_rate=crate)
    call step_loop(xt, yt, ntime)
    call system_clock(count=c1)
#ifdef ROSENNA_GATE_MARKERS
    irange = rosenna_range_pop()
#endif
    !$omp target exit data map(from: yt) map(delete: xt)
    ! The timed loop's own results are checked, not just the correctness
    ! loop's: xt/yt that failed to map would leave this timing a loop over
    ! garbage, and nothing else here would notice. Every timed point is a
    ! tile of one of the n correctness points, so yt(:, t) must reproduce
    ! y(:, mod(t-1,n)+1) -- to a tolerance, since the two loops are separate
    ! regions the compiler may schedule (and contract) differently.
    do t = 1, ntime
        b = mod(t - 1, n) + 1
        if (any(abs(yt(:, t) - y(:, b)) > 1.0e-9_real64 + 1.0e-9_real64 * abs(y(:, b)))) then
            print '(A, I0)', 'TIMED MISMATCH at point ', t
            stop 5
        end if
    end do
    ns_per_point = real(c1 - c0, real64) / real(crate, real64) * 1.0e9_real64 &
                   / (real(ntime, real64) * {nsteps}.0_real64)
    print '(A, ES24.16)', 'TIMING ', ns_per_point
contains
    ! The step loop sees the arrays as explicit-shape dummies, not as the
    ! allocatables they are in the caller: an allocatable carries a
    ! descriptor, and flang's OpenMP re-maps that descriptor (two small
    ! host-to-device copies here, one per array) on EVERY target-region
    ! entry, i.e. once per time step -- seen on an MI210 with amdflang. An
    ! explicit-shape dummy has no descriptor, so the resident data is
    ! reached with no transfer at all, which is what a solver's own step
    ! loop should do too.
    subroutine step_loop(xt, yt, ntime)
        integer, intent(in) :: ntime
        real(real64), intent(in) :: xt({n_in}, ntime)
        real(real64), intent(inout) :: yt({n_out}, ntime)
        integer :: step, t
        do step = 1, {nsteps}
            !$omp target teams distribute parallel do
            do t = 1, ntime
                call {name}_infer(xt(:, t), yt(:, t))
            end do
        end do
    end subroutine
end program
"""

_C_HARNESS3_OMP = """/* rosenna gpu-gate: infer_batch over device-resident data (omp backend), C. */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "{name}.h"
int main(void) {{
    {init}
    int n;
    if (scanf("%d", &n) != 1) return 1;
    double *x = malloc(sizeof(double) * (size_t)n * {n_in});
    double *y = malloc(sizeof(double) * (size_t)n * {n_out});
    for (int c = 0; c < n * {n_in}; ++c) if (scanf("%lf", &x[c]) != 1) return 1;
    int status;
#ifdef _OPENMP
    #pragma omp target enter data map(to: x[0:n*{n_in}]) map(alloc: y[0:n*{n_out}])
    #pragma omp target data use_device_ptr(x, y)
#endif
    {{
        status = {name}_infer_batch(n, x, y, 0);
    }}
#ifdef _OPENMP
    #pragma omp target exit data map(from: y[0:n*{n_out}]) map(delete: x[0:n*{n_in}])
#endif
    if (status != 0) return 20 + status;
    for (int p = 0; p < n; ++p) {{
        for (int i = 0; i < {n_out}; ++i) printf("%.17e ", y[p * {n_out} + i]);
        printf("\\n");
    }}
    /* Timing: ONE call over ntime points (a batch is meant to be called once
       over many points, not called many times over a small batch -- the
       latter pays a construct-entry cost per call and times that instead of
       the kernel). Values are the correctness batch tiled; only infer_batch's
       own cost is timed, not the tiling. */
    long ntime = {ntime}L;
    double *xt = malloc(sizeof(double) * (size_t)ntime * {n_in});
    double *yt = malloc(sizeof(double) * (size_t)ntime * {n_out});
    for (long t = 0; t < ntime; ++t) {{
        int b = (int)(t % n);
        for (int i = 0; i < {n_in}; ++i) xt[t * {n_in} + i] = x[b * {n_in} + i];
    }}
    /* Ruling R16: the mapping (a real transfer, exempt from R5 since it is
       the harness's own setup, not inside infer_batch) happens before t0
       and is undone after t1, so the timed window holds only the call --
       matching what the cuda/hip .cu driver already does. */
#ifdef _OPENMP
    #pragma omp target enter data map(to: xt[0:ntime*{n_in}]) map(alloc: yt[0:ntime*{n_out}])
#endif
    double t0 = omp_get_wtime();
#ifdef _OPENMP
    #pragma omp target data use_device_ptr(xt, yt)
#endif
    {{
        for (int step = 0; step < {nsteps} && status == 0; ++step)
            status = {name}_infer_batch((int)ntime, xt, yt, 0);
    }}
    double t1 = omp_get_wtime();
#ifdef _OPENMP
    #pragma omp target exit data map(from: yt[0:ntime*{n_out}]) map(delete: xt[0:ntime*{n_in}])
#endif
    if (status != 0) return 30 + status;
    printf("TIMING %.6f\\n", (t1 - t0) * 1.0e9 / ((double)ntime * {nsteps}));
    free(x); free(y); free(xt); free(yt);
    return 0;
}}
"""

_F_HARNESS3_OMP = """
program host
    use {name}_model
    use iso_fortran_env, only: real64
    implicit none
    real(real64), allocatable :: x(:,:), y(:,:), xt(:,:), yt(:,:)
    integer :: n, p, status, ntime, t, b, step
    integer(8) :: c0, c1, crate
    real(real64) :: ns_per_point
    status = 0
    {init_lines}
    read(*,*) n
    allocate(x({n_in}, n), y({n_out}, n))
    read(*,*) x
    ! x and y are mapped first, and the call sees their device addresses
    ! (use_device_addr): infer_batch's has_device_addr clause needs those,
    ! not the host addresses (ruling R5).
    !$omp target enter data map(to: x) map(alloc: y)
    !$omp target data use_device_addr(x, y)
    call {name}_infer_batch(n, x, y, status)
    !$omp end target data
    !$omp target exit data map(from: y) map(delete: x)
    if (status /= 0) stop 20
    do p = 1, n
        print '({n_out}(es24.16,1x))', y(:, p)
    end do
    ! Timing: ONE call over ntime points (a batch is meant to be called once
    ! over many points, not called many times over a small batch), values
    ! tiled from the correctness batch above; only infer_batch itself is timed.
    ntime = {ntime}
    allocate(xt({n_in}, ntime), yt({n_out}, ntime))
    do t = 1, ntime
        b = mod(t - 1, n) + 1
        xt(:, t) = x(:, b)
    end do
    ! Ruling R16: map before c0 and unmap after c1, so the timed window
    ! holds only the infer_batch call, matching the cuda/hip .cu driver.
    !$omp target enter data map(to: xt) map(alloc: yt)
    !$omp target data use_device_addr(xt, yt)
    call system_clock(count=c0, count_rate=crate)
    do step = 1, {nsteps}
        if (status == 0) call {name}_infer_batch(ntime, xt, yt, status)
    end do
    call system_clock(count=c1)
    !$omp end target data
    !$omp target exit data map(from: yt) map(delete: xt)
    if (status /= 0) stop 21
    ns_per_point = real(c1 - c0, real64) / real(crate, real64) * 1.0e9_real64 &
                   / (real(ntime, real64) * {nsteps}.0_real64)
    print '(A, ES24.16)', 'TIMING ', ns_per_point
end program
"""

_DEV_HARNESS3 = """/* rosenna gpu-gate: infer_batch over raw device pointers ({backend}), written by the gate. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/* rosenna_rt.h includes cuda_runtime.h or hip/hip_runtime.h for whichever
   compiler this is: nvcc includes its runtime implicitly, hipcc does not. */
#include "rosenna_rt.h"
#include "{name}.h"
#ifdef ROSENNA_GATE_MARKERS
/* Ruling R15: only defined (via -DROSENNA_GATE_MARKERS=1) for the separate
   build the profiler check compiles, so the ordinary timed run never needs
   this header: nvtx3 for nsys (documented as header-only; _run_nsys_check
   retries the link with -lnvToolsExt if the no-link form fails), roctx for
   rocprofv3 (linked with -lrocprofiler-sdk-roctx). */
#include {marker_header}
#endif
int main(void) {{
    {init}
    int n;
    if (scanf("%d", &n) != 1) return 1;
    double *hx = (double*)malloc(sizeof(double) * (size_t)n * {n_in});
    double *hy = (double*)malloc(sizeof(double) * (size_t)n * {n_out});
    for (int c = 0; c < n * {n_in}; ++c) if (scanf("%lf", &hx[c]) != 1) return 1;
    double *dx = 0, *dy = 0;
    if ({p}Malloc((void**)&dx, sizeof(double) * (size_t)n * {n_in}) != {p}Success) return 2;
    if ({p}Malloc((void**)&dy, sizeof(double) * (size_t)n * {n_out}) != {p}Success) return 2;
    if ({p}Memcpy(dx, hx, sizeof(double) * (size_t)n * {n_in}, {p}MemcpyHostToDevice) != {p}Success) return 2;
    int status = {name}_infer_batch(n, dx, dy, 0);
    if (status != 0) return 20 + status;
    if ({p}Memcpy(hy, dy, sizeof(double) * (size_t)n * {n_out}, {p}MemcpyDeviceToHost) != {p}Success) return 3;
    for (int p2 = 0; p2 < n; ++p2) {{
        for (int i = 0; i < {n_out}; ++i) printf("%.17e ", hy[p2 * {n_out} + i]);
        printf("\\n");
    }}
    /* Timing: ONE call over ntime points (a batch is meant to be called once
       over many points), values tiled from the correctness batch above on
       the host, copied to the device once, before the clock starts. */
    long ntime = {ntime}L;
    double *hxt = (double*)malloc(sizeof(double) * (size_t)ntime * {n_in});
    for (long t = 0; t < ntime; ++t) {{
        int b = (int)(t % n);
        for (int i = 0; i < {n_in}; ++i) hxt[t * {n_in} + i] = hx[b * {n_in} + i];
    }}
    double *dxt = 0, *dyt = 0;
    if ({p}Malloc((void**)&dxt, sizeof(double) * (size_t)ntime * {n_in}) != {p}Success) return 4;
    if ({p}Malloc((void**)&dyt, sizeof(double) * (size_t)ntime * {n_out}) != {p}Success) return 4;
    if ({p}Memcpy(dxt, hxt, sizeof(double) * (size_t)ntime * {n_in}, {p}MemcpyHostToDevice) != {p}Success) return 4;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    /* No cudaMemcpy/hipMemcpy in this call (ruling R5). Ruling R15: the
       profiler check brackets ONLY this call with a named range (nvtx under
       nsys, which captures just that range; roctx under rocprofv3, whose
       trace is then cut to the range's timestamps), so the memcpy count is
       scoped to the call itself, not to this driver's untimed setup above
       (which legitimately memcpys) -- counting across the whole profile
       would fail an R5-compliant infer_batch. */
#ifdef ROSENNA_GATE_MARKERS
    {marker_push}("rosenna_timed");
#endif
    for (int step = 0; step < {nsteps} && status == 0; ++step)
        status = {name}_infer_batch((int)ntime, dxt, dyt, 0);
    {p}DeviceSynchronize();
#ifdef ROSENNA_GATE_MARKERS
    {marker_pop}();
#endif
    if (status != 0) return 30 + status;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("TIMING %.6f\\n", secs * 1.0e9 / ((double)ntime * {nsteps}));
    {p}Free(dx); {p}Free(dy); {p}Free(dxt); {p}Free(dyt);
    free(hx); free(hy); free(hxt);
    return 0;
}}
"""


def _c_harness1(plan, backend: str) -> str:
    name = plan.model
    header, push, pop = _MARKERS.get(backend, ("<stddef.h>", "", ""))
    return _C_HARNESS1.format(
        name=name, n_in=plan.input.shape[0], n_out=plan.output.shape[0],
        init="" if plan.embed else f'if ({name}_init("{name}.rwt")) return 2;',
        ntime=_TIMED_ITERS, nsteps=_NSTEPS, marker_header=header, marker_push=push, marker_pop=pop)


def _f_harness2(plan, backend: str) -> str:
    name = plan.model
    _, push, pop = _MARKERS.get(backend, ("", "", ""))
    return _F_HARNESS2.format(
        name=name, n_in=plan.input.shape[0], n_out=plan.output.shape[0],
        init_lines="" if plan.embed else f'call {name}_init("{name}.rwt", status); if (status /= 0) stop 2',
        ntime=_TIMED_ITERS, nsteps=_NSTEPS,
        f_markers=_F_MARKERS.format(marker_push=push, marker_pop=pop))


def _run_c_harness1(report, cfg_dir, plan, cc, flags, host_lib, inputs, expected, env, backend) -> bool:
    (cfg_dir / "gate_harness1.c").write_text(_c_harness1(plan, backend))
    # Rulings R21/R22: the HOST compiler, with its own offload flags (nvc's
    # -mp=gpu -gpu=cc80, amdclang's -fopenmp --offload-arch=..., or plain
    # -fopenmp), both compiles and links this harness in every backend. The
    # archive it links (file-loaded plans only) is the omp-backend one the
    # same host compiler built (_CLibs.host), so no CUDA/HIP runtime is
    # involved and nothing is forwarded to a device compiler: nvcc's default
    # host compiler is g++, which rejects -mp=gpu, so a device-compiler link
    # of this object cannot work, and the omp archive is the only one whose
    # weight arrays have the declare-target copies this offload loop reads.
    cc_proc = _sh(report, "compile c per-point harness (host compiler, host offload flags)",
                 [cc, *_c_flags(cc), *flags.split(),
                  "-c", "gate_harness1.c", "-o", "gate_harness1.o"], cwd=cfg_dir)
    if cc_proc.returncode != 0:
        return False
    objs = ["gate_harness1.o"]
    if not plan.embed:
        objs.append(str(host_lib.relative_to(cfg_dir)))
    link_proc = _sh(report, "link c per-point harness (host compiler, host offload flags, "
                            "omp-backend archive)",
                    [cc, *flags.split(), *objs, "-lm", "-o", "gate_harness1"], cwd=cfg_dir)
    if link_proc.returncode != 0:
        return False
    run_proc = _sh(report, "run c per-point harness", ["./gate_harness1"], cwd=cfg_dir,
                   env=env, input_text=_stdin_for(inputs))
    if run_proc.returncode != 0:
        return False
    ok, _ = _check_output(report, run_proc.stdout, expected)
    return ok


def _run_fortran_harness2(report, cfg_dir, plan, fc, flags, inputs, expected, env, backend) -> bool:
    name = plan.model
    (cfg_dir / "gate_harness2.F90").write_text(_f_harness2(plan, backend))
    fc_proc = _sh(report, "compile fortran per-point harness",
                 [fc, *_f_flags(fc), *flags.split(),
                  "gate_harness2.F90", f"lib{name}_f.a", "-o", "gate_harness2"], cwd=cfg_dir)
    if fc_proc.returncode != 0:
        return False
    run_proc = _sh(report, "run fortran per-point harness", ["./gate_harness2"], cwd=cfg_dir,
                   env=env, input_text=_stdin_for(inputs))
    if run_proc.returncode != 0:
        return False
    ok, _ = _check_output(report, run_proc.stdout, expected)
    return ok


def _run_c_harness3_omp(report, cfg_dir, plan, cc, flags, host_lib, inputs, expected, env) -> bool:
    name = plan.model
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    init = "" if plan.embed else f'if ({name}_init("{name}.rwt")) return 2;'
    (cfg_dir / "gate_harness3.c").write_text(_C_HARNESS3_OMP.format(
        name=name, n_in=n_in, n_out=n_out, init=init, ntime=_TIMED_ITERS, nsteps=_NSTEPS))
    cc_proc = _sh(report, "compile c infer_batch harness (omp)",
                 [cc, *_c_flags(cc), *flags.split(), "gate_harness3.c",
                  str(host_lib.relative_to(cfg_dir)), "-lm", "-o", "gate_harness3"], cwd=cfg_dir)
    if cc_proc.returncode != 0:
        return False
    run_proc = _sh(report, "run c infer_batch harness (omp)", ["./gate_harness3"], cwd=cfg_dir,
                   env=env, input_text=_stdin_for(inputs))
    if run_proc.returncode != 0:
        return False
    ok, _ = _check_output(report, run_proc.stdout, expected)
    return ok


def _run_fortran_harness3_omp(report, cfg_dir, plan, fc, flags, inputs, expected, env) -> bool:
    name = plan.model
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    init_lines = "" if plan.embed else (
        f'call {name}_init("{name}.rwt", status); if (status /= 0) stop 2')
    (cfg_dir / "gate_harness3.f90").write_text(_F_HARNESS3_OMP.format(
        name=name, n_in=n_in, n_out=n_out, init_lines=init_lines, ntime=_TIMED_ITERS, nsteps=_NSTEPS))
    fc_proc = _sh(report, "compile fortran infer_batch harness (omp)",
                 [fc, *_f_flags(fc), *flags.split(),
                  "gate_harness3.f90", f"lib{name}_f.a", "-o", "gate_harness3_f"], cwd=cfg_dir)
    if fc_proc.returncode != 0:
        return False
    run_proc = _sh(report, "run fortran infer_batch harness (omp)", ["./gate_harness3_f"],
                   cwd=cfg_dir, env=env, input_text=_stdin_for(inputs))
    if run_proc.returncode != 0:
        return False
    ok, _ = _check_output(report, run_proc.stdout, expected)
    return ok


def _run_dev_harness3(report, cfg_dir, plan, devcc, devflags, backend, dev_lib, inputs, expected) -> bool:
    name = plan.model
    (cfg_dir / "gate_harness3.cu").write_text(_dev_harness3(plan, backend))
    # Ruling R22: the device compiler compiles AND links this driver (it
    # supplies its own runtime), against the archive it built itself. The
    # archive goes to the linker as -L/-l rather than as a bare path: hipcc
    # injects `-x hip` ahead of a .cu input, and that applies to every input
    # after it, so a bare libfoo.a after the .cu is compiled as HIP source
    # ("!<arch>: expected unqualified-id"). nvcc dispatches by extension and
    # takes either form. Seen on an MI210 with ROCm 7.2.
    proc = _sh(report, f"compile and link {backend} infer_batch harness (device compiler)",
              [*shlex.split(devcc), *devflags.split(),
               "gate_harness3.cu", *_link_archive(dev_lib, cfg_dir), "-o", "gate_harness3_dev"],
              cwd=cfg_dir)
    if proc.returncode != 0:
        return False
    run_proc = _sh(report, f"run {backend} infer_batch harness", ["./gate_harness3_dev"],
                   cwd=cfg_dir, input_text=_stdin_for(inputs))
    if run_proc.returncode != 0:
        return False
    ok, _ = _check_output(report, run_proc.stdout, expected)
    return ok


# Per backend: the marker header the profiler check's harness includes, and
# the range push/pop it calls. nvtx3 is what nsys captures on; roctx (the
# rocprofiler-sdk one, ROCm >= 6.2) is what rocprofv3 --marker-trace records.
_MARKERS = {
    "cuda": ("<nvtx3/nvToolsExt.h>", "nvtxRangePushA", "nvtxRangePop"),
    "hip": ("<rocprofiler-sdk-roctx/roctx.h>", "roctxRangePushA", "roctxRangePop"),
}


def _dev_harness3(plan, backend: str) -> str:
    """Render the infer_batch driver; the marker bracket is inert without -DROSENNA_GATE_MARKERS."""
    name = plan.model
    header, push, pop = _MARKERS[backend]
    return _DEV_HARNESS3.format(
        name=name, n_in=plan.input.shape[0], n_out=plan.output.shape[0],
        init="" if plan.embed else f'if ({name}_init("{name}.rwt")) return 2;',
        ntime=_TIMED_ITERS, nsteps=_NSTEPS, p=backend, backend=backend,
        marker_header=header, marker_push=push, marker_pop=pop)


def _probe_marker_header(report: _Report, cfg_dir: Path, devcc: str, devflags: str,
                         header: str) -> bool:
    """Compile-only probe for the marker header with the device compiler.

    Ruling R15: if the header is not found, the profiler check is skipped
    with a named reason rather than failing the gate or attempting to build
    the instrumented variant anyway.
    """
    (cfg_dir / "gate_marker_probe.cu").write_text(
        f"#include {header}\nint main(void){{return 0;}}\n")
    proc = _sh(report, f"probe for {header}",
              [*shlex.split(devcc), *devflags.split(), "-c", "gate_marker_probe.cu",
               "-o", "gate_marker_probe.o"], cwd=cfg_dir)
    return proc.returncode == 0


@dataclass(frozen=True)
class NsysParseResult:
    """Ruling R18: a structured parse result, not a bare int.

    `parsed` is True only when the Name/Num Calls columns were both
    recognised AND at least one row was actually read -- proving the CSV
    was genuinely parsed, not just that an empty or unrelated header
    happened to match nothing. `count` (the sum of Num Calls over every row
    whose Name starts with "cudaMemcpy" or "cuMemcpy") is meaningful only when `parsed`
    is True: an unparsed 0 must never be read as a passing zero, since the
    whole point of this check is R5 evidence.
    """
    parsed: bool
    count: int


def _sum_cudamemcpy_calls(csv_text: str) -> NsysParseResult:
    """Parse `nsys stats --report cuda_api_sum --format csv` output.

    On stdout the CSV comes after a preamble ("Generating SQLite file ...",
    "Processing ...", a "** CUDA API Summary" title); the gate asks for -q
    to drop it, but does not rely on that: the header is the first line
    naming both a "Num Calls" and a "Name" column, case-insensitively, and
    parsing starts there. Column names/casing can drift slightly across
    Nsight Systems versions, so the columns are then matched by substring
    rather than exact string.
    """
    lines = csv_text.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if "num calls" in line.lower() and "name" in line.lower()), None)
    if start is None:
        return NsysParseResult(False, 0)
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    if not reader.fieldnames:
        return NsysParseResult(False, 0)
    name_col = next((f for f in reader.fieldnames if "name" in f.lower()), None)
    calls_col = next((f for f in reader.fieldnames
                      if "num calls" in f.lower() or "numcalls" in f.lower().replace(" ", "")),
                     None)
    if not name_col or not calls_col:
        return NsysParseResult(False, 0)
    rows_read = 0
    total = 0
    for row in reader:
        rows_read += 1
        name = (row.get(name_col) or "").strip()
        # Runtime API (cudaMemcpy*) and driver API (cuMemcpy*, which is what
        # nvc's OpenMP offload calls) alike.
        if name.startswith("cudaMemcpy") or name.startswith("cuMemcpy"):
            total += int(float(row.get(calls_col) or 0))
    if rows_read == 0:
        # Header recognised but no data rows: still inconclusive, not a
        # genuine (parsed) zero -- an empty cuda_api_sum table is at least
        # as likely to mean "nsys produced nothing useful" as "zero calls".
        return NsysParseResult(False, 0)
    return NsysParseResult(True, total)


@dataclass(frozen=True)
class _Instrumented:
    """One harness rebuilt with its marker bracket, for the profiler.

    `sources` are the inputs the ordinary build used (the same files: the
    bracket is inert without -DROSENNA_GATE_MARKERS); `compiler` is the
    command that built them, and `env` what the run needs (MANDATORY for the
    OpenMP harnesses). Every harness a configuration ran is profiled: the
    per-point C and Fortran hosts are the shape a solver actually has -- the
    model called from its own target loop, once per time step -- and the
    infer_batch driver is the native-kernel path.
    """
    label: str
    exe: str
    compiler: list
    sources: list
    env: dict | None


def _rocm_root() -> Path | None:
    """The ROCm install rocprofv3 came from; where its roctx library lives."""
    rocprof = shutil.which("rocprofv3")
    return Path(rocprof).resolve().parent.parent if rocprof else None


def _marker_link(backend: str) -> list:
    """Linker arguments for the marker library, for a host compiler.

    hipcc knows its own lib dir; amdclang/amdflang do not, so the roctx
    library is named with its directory. nvtx3 is header-only for C (it
    loads libnvToolsExt at runtime); the Fortran bracket binds the symbols
    directly and so needs the library -- _build_instrumented retries with
    -lnvToolsExt and then nvfortran's -cudalib=nvtx.
    """
    if backend == "hip":
        root = _rocm_root()
        return ([f"-L{root / 'lib'}", f"-I{root / 'include'}"] if root else []) + ["-lrocprofiler-sdk-roctx"]
    return []


def _build_instrumented(report: _Report, cfg_dir: Path, h: _Instrumented, backend: str) -> bool:
    base = [*h.compiler, "-DROSENNA_GATE_MARKERS=1", *h.sources, *_marker_link(backend)]
    attempts = [([], "")]
    if backend == "cuda":
        attempts += [(["-lnvToolsExt"], " (retry: -lnvToolsExt)"),
                     (["-cudalib=nvtx"], " (retry: -cudalib=nvtx)")]
    for extra, note in attempts:
        proc = _sh(report, f"compile marker-bracketed {h.label} harness{note}",
                   [*base, *extra, "-o", h.exe], cwd=cfg_dir)
        if proc.returncode == 0:
            return True
    return False


def _run_nsys_check(report: _Report, cfg_dir: Path, h: _Instrumented, inputs) -> bool:
    """Ruling R15 under nsys: zero cudaMemcpy/cuMemcpy inside the nvtx-scoped step loop.

    Profiling the whole harness and counting memcpys would fail an
    R5-compliant model: the driver's own untimed setup (H2D copies before
    the clock starts) legitimately copies, and so does init. Scoped instead
    with an nvtx range around only the timed step loop, `nsys profile
    --capture-range=nvtx --nvtx-capture=rosenna_timed`, and the count read
    from `nsys stats --report cuda_api_sum --format csv`, which lists the
    runtime API (cudaMemcpy*) and the driver API (cuMemcpy*, what nvc's
    OpenMP offload uses) alike. Validated on an A100 for the infer_batch
    driver; the per-point harnesses go through the same path unexercised.
    """
    nsys = shutil.which("nsys")
    stats_base = (cfg_dir / f"gate_nsys_{h.exe}").resolve()
    # --capture-range-end=stop: profiling stops when the nvtx range closes
    # and the harness runs on to completion (the default for an nvtx
    # capture range shuts the application down instead).
    # -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0: nsys only honours a capture
    # range named by a REGISTERED nvtx string unless this is off, and the
    # harness pushes a plain nvtxRangePushA. Without it the capture range
    # never opens, nothing is collected, and nsys exits 0 having written no
    # .nsys-rep at all ("No reports were generated") -- verified on nsys
    # 2025.5 (HPC SDK 25.11) against an A100. The report path is absolute:
    # nsys runs with cwd=cfg_dir and would resolve a relative one twice.
    profile_proc = _sh(
        report, f"nsys profile --capture-range=nvtx --nvtx-capture=rosenna_timed ({h.label})",
        [nsys, "profile", "-e", "NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
         "--capture-range=nvtx", "--nvtx-capture=rosenna_timed",
         "--capture-range-end=stop", "--stats=true", "--force-overwrite=true",
         "-o", str(stats_base), f"./{h.exe}"], cwd=cfg_dir, env=h.env,
        input_text=_stdin_for(inputs))
    if profile_proc.returncode != 0:
        report.p("FAIL: nsys profile did not complete successfully")
        return False
    report_file = stats_base.with_suffix(".nsys-rep")
    if not report_file.exists():
        # nsys exits 0 having written nothing both when the capture range
        # never opened and when it could not create the file.
        report.p(f"FAIL: nsys profile exited 0 but wrote no {report_file} -- either the "
                 "nvtx capture range never opened or the report could not be created; "
                 "the stderr above says which.")
        return False
    # --force-export=true: `nsys profile --stats=true` above already wrote a
    # .sqlite beside the report, and nsys refuses to read one it considers
    # older than the .nsys-rep (which its own finalization makes it).
    stats_proc = _sh(report, "nsys stats -q --force-export=true --report cuda_api_sum --format csv",
                     [nsys, "stats", "-q", "--force-export=true",
                      "--report", "cuda_api_sum", "--format", "csv",
                      str(report_file)], cwd=cfg_dir)
    if stats_proc.returncode != 0:
        report.p("FAIL: nsys stats did not complete successfully")
        return False
    result = _sum_cudamemcpy_calls(stats_proc.stdout)
    if not result.parsed:
        # Ruling R18: an unparseable export must never read as a passing zero.
        report.p("nsys check inconclusive: could not parse cuda_api_sum "
                 "(no recognised Name/Num Calls columns, or no data rows read).")
        raw_lines = stats_proc.stdout.splitlines()[:8]
        report.block("first lines of `nsys stats --report cuda_api_sum --format csv`",
                     "\n".join(raw_lines) if raw_lines else "(empty output)")
        report.p("FAIL: an inconclusive parse counts as a gate failure, not a pass.")
        return False
    report.p(f"{h.label}: Memcpy API calls (cudaMemcpy*, cuMemcpy*) inside the nvtx-scoped "
             f"{_NSTEPS}-step loop, from cuda_api_sum: {result.count}")
    if result.count != 0:
        report.p("FAIL: nothing in the step loop may transfer (ruling R5)")
        return False
    return True


def _scope_hip_transfers(marker_csv: str, api_csv, copy_csv) -> NsysParseResult:
    """Count HIP transfers whose whole interval lies inside the rosenna_timed range.

    rocprofv3 has no nsys-style capture range, but `--hip-trace
    --marker-trace --memory-copy-trace -f csv` writes every HIP API call,
    every copy the runtime performed and every roctx range with
    Start_Timestamp/End_Timestamp on one clock, so the scoping is done here.
    Both traces count: a small hipMemcpy is staged by the host and never
    appears as a MEMORY_COPY, and OpenMP offload's copies go over HSA and
    never appear as a HIP API call -- an OpenMP harness makes no HIP API
    calls at all, so its api_csv is None. Ruling R18 as for nsys: `parsed`
    is True only when exactly one rosenna_timed range was found AND at
    least one trace had recognised columns and at least one row; otherwise
    the count is not evidence.
    """
    def rows(text, name_col):
        if text is None:
            return None
        reader = csv.DictReader(io.StringIO(text))
        fields = reader.fieldnames or []
        need = (name_col, "Start_Timestamp", "End_Timestamp")
        if not all(any(n.lower() == f.lower() for f in fields) for n in need):
            return None
        col = {n: next(f for f in fields if f.lower() == n.lower()) for n in need}
        return [(r[col[name_col]], int(r[col["Start_Timestamp"]]), int(r[col["End_Timestamp"]]))
                for r in reader]

    markers = rows(marker_csv, "Function")
    if markers is None:
        return NsysParseResult(False, 0)
    ranges = [(t0, t1) for fn, t0, t1 in markers if fn == "rosenna_timed"]
    if len(ranges) != 1:
        return NsysParseResult(False, 0)
    t0, t1 = ranges[0]
    calls, copies = rows(api_csv, "Function"), rows(copy_csv, "Kind")
    if not calls and not copies:
        return NsysParseResult(False, 0)
    inside = lambda a, b: a >= t0 and b <= t1
    count = (sum(1 for fn, a, b in (calls or []) if fn.startswith("hipMemcpy") and inside(a, b))
             + sum(1 for _, a, b in (copies or []) if inside(a, b)))
    return NsysParseResult(True, count)


def _run_rocprof_check(report: _Report, cfg_dir: Path, h: _Instrumented, inputs) -> bool:
    """The HIP twin of _run_nsys_check: zero transfers inside the roctx-scoped step loop.

    Validated on an MI210 under ROCm 7.2.0 and AFAR 23.2.1 for all three
    harnesses, embedded and file-loaded.
    """
    rocprof = shutil.which("rocprofv3")
    prof_dir = (cfg_dir / f"gate_rocprof_{h.exe}").resolve()
    shutil.rmtree(prof_dir, ignore_errors=True)
    profile_proc = _sh(
        report, f"rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv ({h.label})",
        [rocprof, "--hip-trace", "--marker-trace", "--memory-copy-trace", "-f", "csv",
         "-d", str(prof_dir), "-o", "prof", "--", f"./{h.exe}"], cwd=cfg_dir, env=h.env,
        input_text=_stdin_for(inputs))
    if profile_proc.returncode != 0:
        report.p("FAIL: rocprofv3 did not complete successfully")
        return False
    marker_csv = prof_dir / "prof_marker_api_trace.csv"
    if not marker_csv.exists():
        report.p(f"FAIL: rocprofv3 exited 0 but wrote no {marker_csv.name} in {prof_dir}")
        return False
    # rocprofv3 writes a trace file only for a domain that had events: an
    # OpenMP harness has no HIP API trace, and a harness with no copies at
    # all would have no memory-copy trace (every one here copies in setup).
    read = lambda name: (prof_dir / name).read_text() if (prof_dir / name).exists() else None
    result = _scope_hip_transfers(marker_csv.read_text(), read("prof_hip_api_trace.csv"),
                                  read("prof_memory_copy_trace.csv"))
    if not result.parsed:
        # Ruling R18: an unparseable trace is a gate failure, not a pass.
        report.p("rocprof check inconclusive: no single rosenna_timed range in the marker "
                 "trace, or no trace rows at all.")
        report.block(f"first lines of {marker_csv.name}",
                     "\n".join(marker_csv.read_text().splitlines()[:8]) or "(empty)")
        report.p("FAIL: an inconclusive parse counts as a gate failure, not a pass.")
        return False
    report.p(f"{h.label}: hipMemcpy* API calls plus MEMORY_COPY operations inside the "
             f"roctx-scoped {_NSTEPS}-step loop: {result.count}")
    if result.count != 0:
        report.p("FAIL: nothing in the step loop may transfer (ruling R5)")
        return False
    return True


def _run_transfer_check(report: _Report, cfg_dir: Path, backend: str, devcc: str,
                        devflags: str, harnesses: list, inputs) -> bool:
    """Ruling R15 for every harness this configuration ran, with the backend's profiler.

    The profiler and the marker header are looked for once; if either is
    missing the check is skipped with a named reason (that does not fail
    the gate on its own). A harness whose instrumented build fails is
    likewise recorded and skipped -- loudly, since it is the evidence.
    """
    profiler, header = (("rocprofv3", _MARKERS["hip"][0]) if backend == "hip"
                        else (("nsys", _MARKERS["cuda"][0])))
    report.h(f"{profiler} check: transfers inside the marker-scoped {_NSTEPS}-step loop, "
             f"every harness (ruling R15)", 4)
    if not shutil.which(profiler):
        report.p(f"{profiler} not found on PATH; the transfer-count check was NOT run "
                 "(recorded here rather than silently skipped).")
        return True
    if not _probe_marker_header(report, cfg_dir, devcc, devflags, header):
        report.p(f"{profiler} check skipped: marker header {header} did not compile "
                 "with the device compiler.")
        return True
    check = _run_rocprof_check if backend == "hip" else _run_nsys_check
    ok = True
    for h in harnesses:
        if not _build_instrumented(report, cfg_dir, h, backend):
            report.p(f"{profiler} check NOT run for the {h.label} harness: its "
                     "marker-bracketed build failed (see above). This harness has no "
                     "transfer evidence in this report.")
            continue
        if not check(report, cfg_dir, h, inputs):
            ok = False
    return ok


def run_gate(*, cc: str, fc: str, flags: str, backend: str, devcc=None, devflags: str = "",
            out: str = ".", host_fallback: bool = False) -> int:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "gate-report.md"
    report = _Report()
    ok = True
    try:
        report.h("rosenna gpu-gate report", 1)
        report.p(f"- model: {_MODEL}")
        report.p(f"- backend: {backend}")
        report.p(f"- host-fallback: {host_fallback}")
        report.p(f"- cc: {cc}")
        report.p(f"- fc: {fc}")
        report.p(f"- flags: {flags!r}")
        if backend != "omp":
            report.p(f"- devcc: {devcc}")
            report.p(f"- devflags: {devflags!r}")
        if host_fallback:
            report.p("- OMP_TARGET_OFFLOAD=MANDATORY is NOT set (host-fallback mode): "
                     "this run exercises the omp-backend contract end to end on a machine "
                     "with no accelerator, the same host-fallback contract "
                     "tests/test_device_c.py and tests/test_device_fortran.py already cover.")
        else:
            report.p("- OMP_TARGET_OFFLOAD=MANDATORY is set for every omp-backend harness: "
                     "a machine with no working offload device must fail here, loudly, "
                     "rather than silently pass by falling back to the host.")

        # Resolved once, used everywhere a device compiler command is needed:
        # --devcc has a default (it is not required, see --help), so every
        # call site uses this instead of repeating the fallback logic.
        resolved_devcc = devcc or ("nvcc" if backend == "cuda" else "hipcc")

        _record_versions(report, cc, fc, resolved_devcc if backend != "omp" else None)

        onnx_path = _ensure_model(report)
        session = ort.InferenceSession(str(onnx_path))
        shape = session.get_inputs()[0].shape
        inputs, expected = _live_reference(session, shape, np.float64, seed=42, batch=8)
        if inputs is None:
            report.p("FATAL: the onnxruntime reference for gemm_big is dead across every "
                     "resampled batch; nothing here would demonstrate anything.")
            report_path.write_text(report.text())
            return 1

        env = dict(os.environ)
        if not host_fallback:
            env["OMP_TARGET_OFFLOAD"] = "MANDATORY"

        for embed in (True, False):
            label = "embedded" if embed else "file-loaded"
            report.h(f"{_MODEL}: {label}", 2)
            cfg_dir = out_dir / ("embedded" if embed else "file_loaded")
            plan = _generate(cfg_dir, onnx_path, embed)

            libs = _build_c_libs(report, cfg_dir, plan, cc, flags, backend, resolved_devcc, devflags)
            f_built = _build_fortran_lib(report, cfg_dir, plan, fc, flags)

            harnesses = []
            report.h("c harness: per-point infer via target teams distribute parallel for", 3)
            if libs.host is not None:
                if not _run_c_harness1(report, cfg_dir, plan, cc, flags, libs.host, inputs,
                                       expected, env, backend):
                    ok = False
                else:
                    harnesses.append(_Instrumented(
                        "c per-point", "gate_harness1_prof", [cc, *_c_flags(cc), *flags.split()],
                        ["gate_harness1.c"] + ([] if plan.embed else [str(libs.host.relative_to(cfg_dir))])
                        + ["-lm"], env))
            else:
                report.p("skipped: c library build (omp backend, host compiler) failed")
                ok = False

            report.h("fortran harness: per-point infer via target teams distribute parallel do", 3)
            if f_built:
                if not _run_fortran_harness2(report, cfg_dir, plan, fc, flags, inputs, expected,
                                             env, backend):
                    ok = False
                else:
                    harnesses.append(_Instrumented(
                        "fortran per-point", "gate_harness2_prof", [fc, *_f_flags(fc), *flags.split()],
                        ["gate_harness2.F90", f"lib{plan.model}_f.a"], env))
            else:
                report.p("skipped: fortran library build failed")
                ok = False

            report.h("infer_batch harness: device-resident data", 3)
            if backend == "omp":
                if libs.host is not None:
                    if not _run_c_harness3_omp(report, cfg_dir, plan, cc, flags, libs.host,
                                               inputs, expected, env):
                        ok = False
                else:
                    report.p("skipped: c library build failed")
                    ok = False
                if f_built:
                    if not _run_fortran_harness3_omp(report, cfg_dir, plan, fc, flags, inputs, expected, env):
                        ok = False
                else:
                    report.p("skipped: fortran library build failed")
                    ok = False
            else:
                if libs.dev is not None:
                    dev_ok = _run_dev_harness3(report, cfg_dir, plan, resolved_devcc,
                                               devflags, backend, libs.dev, inputs, expected)
                    if not dev_ok:
                        ok = False
                    else:
                        harnesses.append(_Instrumented(
                            f"{backend} infer_batch", "gate_harness3_prof",
                            [*shlex.split(resolved_devcc), *devflags.split()],
                            ["gate_harness3.cu", *_link_archive(libs.dev, cfg_dir)], None))
                else:
                    report.p(f"skipped: c library build (backend={backend}, device compiler) failed")
                    ok = False
                # Ruling R15: nsys for cuda, rocprofv3 for hip, over every
                # harness that ran; the check skips itself (with a named
                # reason) when the profiler or the marker header is
                # unavailable, which does not fail the gate on its own.
                if not _run_transfer_check(report, cfg_dir, backend, resolved_devcc, devflags,
                                           harnesses, inputs):
                    ok = False

        report.h("result", 2)
        report.p("PASS: every configuration matched." if ok else
                 "FAIL: at least one configuration above did not match or did not run.")
        report_path.write_text(report.text())
        return 0 if ok else 1
    except Exception as e:  # noqa: BLE001 -- the report must still be written
        report.h("FATAL", 2)
        report.p(f"gate raised an unexpected exception: {type(e).__name__}: {e}")
        report_path.write_text(report.text())
        return 1
