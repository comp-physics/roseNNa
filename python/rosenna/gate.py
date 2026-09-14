"""rosenna gpu-gate: the script a user runs on a real GPU machine.

None of the CUDA/HIP path has ever been compiled or run on the machine that
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
on a GPU machine and its report recorded. Until then: compiles and runs on
the host; device path unvalidated.
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
from .plan import build_plan, validate_model_name
from .rt_header import rt_header
from .verify import _live_reference
from .weights import write_weights

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL = "gemm_big"
_TIMED_ITERS = 1_000_000
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
    onnx_path = _REPO_ROOT / "goldenFiles" / _MODEL / f"{_MODEL}.onnx"
    if not onnx_path.exists():
        gen = _REPO_ROOT / "goldenFiles" / _MODEL / f"{_MODEL}.py"
        _sh(report, "generate the golden gemm_big model", [sys.executable, str(gen)],
           cwd=_REPO_ROOT / "test")
    return onnx_path


def _generate(outdir: Path, onnx_path: Path, embed: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    graph = load_graph(str(onnx_path))
    plan = build_plan(graph, dtype="f64", embed=embed)
    validate_model_name(plan.model)
    name = plan.model
    (outdir / f"{name}_model.f90").write_text(emit_fortran(plan))
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
#ifdef _OPENMP
    #pragma omp target teams loop map(to: x[0:n*{n_in}]) map(from: y[0:n*{n_out}])
#endif
    for (int p = 0; p < n; ++p) {name}_infer(x + p * {n_in}, y + p * {n_out});   /* microfd's own target loop calling the header inline */
    for (int p = 0; p < n; ++p) {{
        for (int i = 0; i < {n_out}; ++i) printf("%.17e ", y[p * {n_out} + i]);
        printf("\\n");
    }}
    /* Timing loop: b cycles through the n (= 8) correctness points, so the
       iterations sharing a b all write the same eight output slots. That
       race is benign and intentional: every writer of a slot stores the
       same value, and y is not read after the loop. */
    long ntime = {ntime}L;
    double t0 = omp_get_wtime();
#ifdef _OPENMP
    #pragma omp target teams loop map(to: x[0:n*{n_in}]) map(from: y[0:n*{n_out}])
#endif
    for (long p = 0; p < ntime; ++p) {{
        int b = (int)(p % n);
        {name}_infer(x + b * {n_in}, y + b * {n_out});
    }}
    double t1 = omp_get_wtime();
    printf("TIMING %.6f\\n", (t1 - t0) * 1.0e9 / (double)ntime);
    free(x); free(y);
    return 0;
}}
"""

_F_HARNESS2 = """
program host
    use {name}_model
    use iso_fortran_env, only: real64
    implicit none
    real(real64), allocatable :: x(:,:), y(:,:)
    integer :: n, p, status, b, ntime, t
    integer(8) :: c0, c1, crate
    real(real64) :: ns_per_point
    status = 0
    {init_lines}
    read(*,*) n
    allocate(x({n_in}, n), y({n_out}, n))
    read(*,*) x
    !$omp target teams loop map(to: x) map(from: y)
    do p = 1, n
        call {name}_infer(x(:, p), y(:, p))
    end do
    do p = 1, n
        print '({n_out}(es24.16,1x))', y(:, p)
    end do
    ! Timing loop: b cycles through the n (= 8) correctness points, so the
    ! iterations sharing a b all write the same eight output columns. That
    ! race is benign and intentional: every writer of a column stores the
    ! same value, and y is not read after the loop.
    ntime = {ntime}
    call system_clock(count=c0, count_rate=crate)
    !$omp target teams loop map(to: x) map(from: y)
    do t = 1, ntime
        b = mod(t - 1, n) + 1
        call {name}_infer(x(:, b), y(:, b))
    end do
    call system_clock(count=c1)
    ns_per_point = real(c1 - c0, real64) / real(crate, real64) * 1.0e9_real64 / real(ntime, real64)
    print '(A, ES24.16)', 'TIMING ', ns_per_point
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
        status = {name}_infer_batch((int)ntime, xt, yt, 0);
    }}
    double t1 = omp_get_wtime();
#ifdef _OPENMP
    #pragma omp target exit data map(from: yt[0:ntime*{n_out}]) map(delete: xt[0:ntime*{n_in}])
#endif
    if (status != 0) return 30 + status;
    printf("TIMING %.6f\\n", (t1 - t0) * 1.0e9 / (double)ntime);
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
    integer :: n, p, status, ntime, t, b
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
    call {name}_infer_batch(ntime, xt, yt, status)
    call system_clock(count=c1)
    !$omp end target data
    !$omp target exit data map(from: yt) map(delete: xt)
    if (status /= 0) stop 21
    ns_per_point = real(c1 - c0, real64) / real(crate, real64) * 1.0e9_real64 / real(ntime, real64)
    print '(A, ES24.16)', 'TIMING ', ns_per_point
end program
"""

_DEV_HARNESS3 = """/* rosenna gpu-gate: infer_batch over raw device pointers ({backend}), written by the gate. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "{name}.h"
#ifdef ROSENNA_GATE_NVTX
/* Ruling R15: only defined (via -DROSENNA_GATE_NVTX=1) for the separate
   build the nsys check compiles, so the ordinary timed run above never
   needs this header. nvtx3 is documented as header-only (it loads
   libnvToolsExt itself at runtime); _run_nsys_check retries the link with
   -lnvToolsExt if the no-link form fails, since that has not been verified
   against every toolkit version here. */
#include <nvtx3/nvToolsExt.h>
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
    /* No cudaMemcpy/hipMemcpy in this call (ruling R5). Ruling R15: the nsys
       check brackets ONLY this call with an nvtx range and profiles with
       --capture-range=nvtx, so its cudaMemcpy count is scoped to the call
       itself, not to this driver's untimed setup above (which legitimately
       memcpys) -- counting across the whole profile would fail an
       R5-compliant infer_batch. */
#ifdef ROSENNA_GATE_NVTX
    nvtxRangePushA("rosenna_timed");
#endif
    status = {name}_infer_batch((int)ntime, dxt, dyt, 0);
#ifdef ROSENNA_GATE_NVTX
    nvtxRangePop();
#endif
    if (status != 0) return 30 + status;
    {p}DeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("TIMING %.6f\\n", secs * 1.0e9 / (double)ntime);
    {p}Free(dx); {p}Free(dy); {p}Free(dxt); {p}Free(dyt);
    free(hx); free(hy); free(hxt);
    return 0;
}}
"""


def _run_c_harness1(report, cfg_dir, plan, cc, flags, host_lib, inputs, expected, env) -> bool:
    name = plan.model
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    init = "" if plan.embed else f'if ({name}_init("{name}.rwt")) return 2;'
    (cfg_dir / "gate_harness1.c").write_text(_C_HARNESS1.format(
        name=name, n_in=n_in, n_out=n_out, init=init, ntime=_TIMED_ITERS))
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


def _run_fortran_harness2(report, cfg_dir, plan, fc, flags, inputs, expected, env) -> bool:
    name = plan.model
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    init_lines = "" if plan.embed else (
        f'call {name}_init("{name}.rwt", status); if (status /= 0) stop 2')
    (cfg_dir / "gate_harness2.f90").write_text(_F_HARNESS2.format(
        name=name, n_in=n_in, n_out=n_out, init_lines=init_lines, ntime=_TIMED_ITERS))
    fc_proc = _sh(report, "compile fortran per-point harness",
                 [fc, *_f_flags(fc), *flags.split(),
                  "gate_harness2.f90", f"lib{name}_f.a", "-o", "gate_harness2"], cwd=cfg_dir)
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
        name=name, n_in=n_in, n_out=n_out, init=init, ntime=_TIMED_ITERS))
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
        name=name, n_in=n_in, n_out=n_out, init_lines=init_lines, ntime=_TIMED_ITERS))
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
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    prefix = "hip" if backend == "hip" else "cuda"
    init = "" if plan.embed else f'if ({name}_init("{name}.rwt")) return 2;'
    (cfg_dir / "gate_harness3.cu").write_text(_DEV_HARNESS3.format(
        name=name, n_in=n_in, n_out=n_out, init=init, ntime=_TIMED_ITERS, p=prefix,
        backend=backend))
    # Ruling R22: the device compiler compiles AND links this driver (it
    # supplies its own runtime), against the archive it built itself. No
    # `-x cu|hip`: the driver is a .cu, which both compilers take as device
    # source by extension, and a -x before the archive would make
    # clang-based hipcc compile the archive as source too.
    proc = _sh(report, f"compile and link {backend} infer_batch harness (device compiler)",
              [*shlex.split(devcc), *devflags.split(),
               "gate_harness3.cu", str(dev_lib.relative_to(cfg_dir)), "-o", "gate_harness3_dev"],
              cwd=cfg_dir)
    if proc.returncode != 0:
        return False
    run_proc = _sh(report, f"run {backend} infer_batch harness", ["./gate_harness3_dev"],
                   cwd=cfg_dir, input_text=_stdin_for(inputs))
    if run_proc.returncode != 0:
        return False
    ok, _ = _check_output(report, run_proc.stdout, expected)
    return ok


def _probe_nvtx_header(report: _Report, cfg_dir: Path, devcc: str, devflags: str) -> bool:
    """Compile-only probe for <nvtx3/nvToolsExt.h> with the device compiler.

    Ruling R15: if the header is not found, the nsys check is skipped with
    a named reason rather than failing the gate or attempting to build the
    nvtx-instrumented variant anyway.
    """
    (cfg_dir / "gate_nvtx_probe.cu").write_text(
        "#include <nvtx3/nvToolsExt.h>\nint main(void){return 0;}\n")
    proc = _sh(report, "probe for <nvtx3/nvToolsExt.h>",
              [*shlex.split(devcc), *devflags.split(), "-c", "gate_nvtx_probe.cu",
               "-o", "gate_nvtx_probe.o"], cwd=cfg_dir)
    return proc.returncode == 0


@dataclass(frozen=True)
class NsysParseResult:
    """Ruling R18: a structured parse result, not a bare int.

    `parsed` is True only when the Name/Num Calls columns were both
    recognised AND at least one row was actually read -- proving the CSV
    was genuinely parsed, not just that an empty or unrelated header
    happened to match nothing. `count` (the sum of Num Calls over every row
    whose Name starts with "cudaMemcpy") is meaningful only when `parsed`
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
        if name.startswith("cudaMemcpy"):
            total += int(float(row.get(calls_col) or 0))
    if rows_read == 0:
        # Header recognised but no data rows: still inconclusive, not a
        # genuine (parsed) zero -- an empty cuda_api_sum table is at least
        # as likely to mean "nsys produced nothing useful" as "zero calls".
        return NsysParseResult(False, 0)
    return NsysParseResult(True, total)


def _run_nsys_check(report: _Report, cfg_dir: Path, plan, devcc: str, devflags: str,
                    backend: str, dev_lib: Path, inputs) -> bool:
    """Ruling R15: assert zero cudaMemcpy calls inside the timed infer_batch call only.

    Profiling the whole harness and counting "cudaMemcpy" across nsys's
    free-text output (the previous approach) would fail an R5-compliant
    infer_batch: the driver's own untimed setup (H2D copies before the
    clock starts) legitimately calls cudaMemcpy. Scoped instead with an
    nvtx range around only the timed call, `nsys profile
    --capture-range=nvtx --nvtx-capture=rosenna_timed`, and the count read
    from `nsys stats --report cuda_api_sum --format csv` on the resulting
    report. None of this has ever run (no nvcc/nsys here); see the task
    report for what remains unexercised.
    """
    report.h("nsys check: cudaMemcpy count inside the nvtx-scoped infer_batch call (ruling R15)", 4)
    if backend == "hip":
        report.p("nsys check skipped: --backend hip (rocprof scoping of the call is a "
                 "follow-up; nsys/nvtx are CUDA-only).")
        return True
    nsys = shutil.which("nsys")
    if not nsys:
        report.p("nsys not found on PATH; the cudaMemcpy-count check was NOT run "
                 "(recorded here rather than silently skipped).")
        return True
    if not _probe_nvtx_header(report, cfg_dir, devcc, devflags):
        report.p("nsys check skipped: nvtx header not found "
                 "(<nvtx3/nvToolsExt.h> did not compile with this device compiler).")
        return True

    name = plan.model
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    init = "" if plan.embed else f'if ({name}_init("{name}.rwt")) return 2;'
    (cfg_dir / "gate_harness3.cu").write_text(_DEV_HARNESS3.format(
        name=name, n_in=n_in, n_out=n_out, init=init, ntime=_TIMED_ITERS, p="cuda",
        backend=backend))
    # nvtx3 (<nvtx3/nvToolsExt.h>) is documented as header-only: it loads
    # libnvToolsExt itself at runtime rather than needing it at link time.
    # Some toolkit versions still expect an explicit link; try without
    # -lnvToolsExt first (the documented form) and retry once with it if
    # linking fails, noting which form was needed. Neither path has run here.
    base_cmd = [*shlex.split(devcc), *devflags.split(), "-DROSENNA_GATE_NVTX=1",
                "gate_harness3.cu", str(dev_lib.relative_to(cfg_dir))]
    proc = _sh(report, "compile nvtx-bracketed infer_batch harness (no explicit -lnvToolsExt)",
              [*base_cmd, "-o", "gate_harness3_nvtx"], cwd=cfg_dir)
    if proc.returncode != 0:
        proc = _sh(report, "compile nvtx-bracketed infer_batch harness (retry: -lnvToolsExt)",
                  [*base_cmd, "-lnvToolsExt", "-o", "gate_harness3_nvtx"], cwd=cfg_dir)
        if proc.returncode != 0:
            report.p("nsys check skipped: the nvtx-bracketed driver did not link, "
                     "with or without -lnvToolsExt.")
            return True

    stats_base = cfg_dir / "gate_nsys_profile"
    # --capture-range-end=stop: profiling stops when the nvtx range closes
    # and the harness runs on to completion (the default for an nvtx
    # capture range shuts the application down instead).
    profile_proc = _sh(
        report, "nsys profile --capture-range=nvtx --nvtx-capture=rosenna_timed "
                "--capture-range-end=stop --stats=true",
        [nsys, "profile", "--capture-range=nvtx", "--nvtx-capture=rosenna_timed",
         "--capture-range-end=stop", "--stats=true", "--force-overwrite=true",
         "-o", str(stats_base), "./gate_harness3_nvtx"], cwd=cfg_dir,
        input_text=_stdin_for(inputs))
    if profile_proc.returncode != 0:
        report.p("FAIL: nsys profile did not complete successfully")
        return False

    report_file = stats_base.with_suffix(".nsys-rep")
    stats_proc = _sh(report, "nsys stats -q --report cuda_api_sum --format csv",
                     [nsys, "stats", "-q", "--report", "cuda_api_sum", "--format", "csv",
                      str(report_file)], cwd=cfg_dir)
    if stats_proc.returncode != 0:
        report.p("FAIL: nsys stats did not complete successfully")
        return False

    result = _sum_cudamemcpy_calls(stats_proc.stdout)
    if not result.parsed:
        # Ruling R18: an unparseable export must never read as a passing
        # zero -- it is treated as a gate failure, since the check's whole
        # purpose is R5 evidence and an inconclusive parse provides none.
        report.p("nsys check inconclusive: could not parse cuda_api_sum "
                 "(no recognised Name/Num Calls columns, or no data rows read).")
        raw_lines = stats_proc.stdout.splitlines()[:8]
        report.block("first lines of `nsys stats --report cuda_api_sum --format csv`",
                     "\n".join(raw_lines) if raw_lines else "(empty output)")
        report.p("FAIL: an inconclusive parse counts as a gate failure, not a pass.")
        return False
    report.p(f"cudaMemcpy* Num Calls inside the nvtx-scoped infer_batch call, from "
             f"cuda_api_sum: {result.count}")
    if result.count != 0:
        report.p("FAIL: infer_batch's timed call must never call cudaMemcpy (ruling R5)")
        return False
    return True


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

            report.h("c harness: per-point infer via target teams loop", 3)
            if libs.host is not None:
                if not _run_c_harness1(report, cfg_dir, plan, cc, flags, libs.host, inputs, expected, env):
                    ok = False
            else:
                report.p("skipped: c library build (omp backend, host compiler) failed")
                ok = False

            report.h("fortran harness: per-point infer via target teams loop", 3)
            if f_built:
                if not _run_fortran_harness2(report, cfg_dir, plan, fc, flags, inputs, expected, env):
                    ok = False
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
                        # Ruling R15: invoked for both backends; it skips
                        # itself (with a named reason) for hip, and for cuda
                        # when nsys or the nvtx header is unavailable, none
                        # of which fails the gate on its own.
                        if not _run_nsys_check(report, cfg_dir, plan, resolved_devcc, devflags,
                                               backend, libs.dev, inputs):
                            ok = False
                else:
                    report.p(f"skipped: c library build (backend={backend}, device compiler) failed")
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
