"""rosenna gpu-gate: the script a user runs on a real GPU machine.

None of the CUDA/HIP path has ever been compiled or run on the machine that
wrote it (no nvcc, hipcc, or GPU). This script is the evidence that fact
cannot produce: it generates the gemm_big plan embedded and file-loaded, in
both languages, builds the C library with the chosen batched backend and the
Fortran library with the host compiler, then runs three harnesses -- a
microfd-shaped per-point host in C, the same in Fortran, and a host that
hands device-resident data to infer_batch -- each compared against
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
import os
import platform
import shutil
import subprocess
import sys
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
        where = f" (in {cwd})" if cwd is not None else ""
        self.lines.append(f"\n**{label}**{where}\n\n```\n$ {' '.join(str(a) for a in args)}\n```")

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
    except FileNotFoundError as e:
        proc = _FakeProc(127, f"{args[0]}: not found ({e.strerror})")
    except subprocess.TimeoutExpired as e:
        proc = _FakeProc(124, f"timed out after {timeout}s\nstdout so far:\n{e.stdout}\nstderr so far:\n{e.stderr}")
    report.outcome(proc)
    return proc


def _record_versions(report: _Report, cc: str, fc: str, devcc) -> None:
    report.h("toolchain", 3)
    report.p(f"platform: {platform.platform()}")
    for label, exe in (("cc", cc), ("fc", fc), ("devcc", devcc)):
        if not exe:
            continue
        proc = subprocess.run([exe, "--version"], capture_output=True, text=True)
        report.block(f"{label} ({exe}) --version", proc.stdout or proc.stderr or "(no output)")


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


def _build_c_lib(report: _Report, outdir: Path, plan, cc, flags, backend, devcc, devflags) -> bool:
    name = plan.model
    label = "embedded" if plan.embed else "file-loaded"
    args = ["make", "-f", f"{name}.mk", f"ROSENNA_BACKEND={backend}"]
    if backend == "omp":
        args += [f"CC={cc}", f"ROSENNA_OFFLOAD_FLAGS={flags}"]
    else:
        args += [f"DEVCC={devcc or ('nvcc' if backend == 'cuda' else 'hipcc')}"]
        if devflags:
            args.append(f"DEVFLAGS={devflags}")
    proc = _sh(report, f"build c library ({label}, backend={backend})", args, cwd=outdir)
    return proc.returncode == 0 and (outdir / f"lib{name}.a").exists()


def _build_fortran_lib(report: _Report, outdir: Path, plan, fc, flags) -> bool:
    name = plan.model
    label = "embedded" if plan.embed else "file-loaded"
    args = ["make", "-f", f"{name}_fortran.mk", f"FC={fc}", f"ROSENNA_OFFLOAD_FLAGS={flags}"]
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
    double t0 = omp_get_wtime();
#ifdef _OPENMP
    #pragma omp target enter data map(to: xt[0:ntime*{n_in}]) map(alloc: yt[0:ntime*{n_out}])
    #pragma omp target data use_device_ptr(xt, yt)
#endif
    {{
        status = {name}_infer_batch((int)ntime, xt, yt, 0);
    }}
#ifdef _OPENMP
    #pragma omp target exit data map(from: yt[0:ntime*{n_out}]) map(delete: xt[0:ntime*{n_in}])
#endif
    double t1 = omp_get_wtime();
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
    !$omp target enter data map(to: x) map(alloc: y)
    call {name}_infer_batch(n, x, y, status)          ! x, y already on the device (R5)
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
    call system_clock(count=c0, count_rate=crate)
    !$omp target enter data map(to: xt) map(alloc: yt)
    call {name}_infer_batch(ntime, xt, yt, status)
    !$omp target exit data map(from: yt) map(delete: xt)
    call system_clock(count=c1)
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
    /* No cudaMemcpy/hipMemcpy in this call (ruling R5): it only launches. The
       nsys check (cuda backend, when nsys is on PATH) asserts that
       structurally from the profile, not just by inspection of this source. */
    status = {name}_infer_batch((int)ntime, dxt, dyt, 0);
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


def _run_c_harness1(report, cfg_dir, plan, cc, flags, inputs, expected, env) -> bool:
    name = plan.model
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    init = "" if plan.embed else f'if ({name}_init("{name}.rwt")) return 2;'
    (cfg_dir / "gate_harness1.c").write_text(_C_HARNESS1.format(
        name=name, n_in=n_in, n_out=n_out, init=init, ntime=_TIMED_ITERS))
    objs = ["gate_harness1.c"]
    if not plan.embed:
        objs.append(f"lib{name}.a")
    cc_proc = _sh(report, "compile c per-point harness",
                 [cc, "-O2", "-Wall", "-Wextra", "-std=c11", *flags.split(), *objs, "-lm",
                  "-o", "gate_harness1"], cwd=cfg_dir)
    if cc_proc.returncode != 0:
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
                 [fc, "-O2", "-Wall", "-Wextra", "-std=f2008", *flags.split(),
                  "gate_harness2.f90", f"lib{name}_f.a", "-o", "gate_harness2"], cwd=cfg_dir)
    if fc_proc.returncode != 0:
        return False
    run_proc = _sh(report, "run fortran per-point harness", ["./gate_harness2"], cwd=cfg_dir,
                   env=env, input_text=_stdin_for(inputs))
    if run_proc.returncode != 0:
        return False
    ok, _ = _check_output(report, run_proc.stdout, expected)
    return ok


def _run_c_harness3_omp(report, cfg_dir, plan, cc, flags, inputs, expected, env) -> bool:
    name = plan.model
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    init = "" if plan.embed else f'if ({name}_init("{name}.rwt")) return 2;'
    (cfg_dir / "gate_harness3.c").write_text(_C_HARNESS3_OMP.format(
        name=name, n_in=n_in, n_out=n_out, init=init, ntime=_TIMED_ITERS))
    cc_proc = _sh(report, "compile c infer_batch harness (omp)",
                 [cc, "-O2", "-Wall", "-Wextra", "-std=c11", *flags.split(),
                  "gate_harness3.c", f"lib{name}.a", "-lm", "-o", "gate_harness3"], cwd=cfg_dir)
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
                 [fc, "-O2", "-Wall", "-Wextra", "-std=f2008", *flags.split(),
                  "gate_harness3.f90", f"lib{name}_f.a", "-o", "gate_harness3_f"], cwd=cfg_dir)
    if fc_proc.returncode != 0:
        return False
    run_proc = _sh(report, "run fortran infer_batch harness (omp)", ["./gate_harness3_f"],
                   cwd=cfg_dir, env=env, input_text=_stdin_for(inputs))
    if run_proc.returncode != 0:
        return False
    ok, _ = _check_output(report, run_proc.stdout, expected)
    return ok


def _run_dev_harness3(report, cfg_dir, plan, devcc, devflags, backend, inputs, expected) -> bool:
    name = plan.model
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    prefix = "hip" if backend == "hip" else "cuda"
    init = "" if plan.embed else f'if ({name}_init("{name}.rwt")) return 2;'
    (cfg_dir / "gate_harness3.cu").write_text(_DEV_HARNESS3.format(
        name=name, n_in=n_in, n_out=n_out, init=init, ntime=_TIMED_ITERS, p=prefix,
        backend=backend))
    x_flag = "hip" if backend == "hip" else "cu"
    proc = _sh(report, f"compile {backend} infer_batch harness",
              [devcc, *devflags.split(), "-x", x_flag,
               "gate_harness3.cu", f"lib{name}.a", "-o", "gate_harness3_dev"], cwd=cfg_dir)
    if proc.returncode != 0:
        return False
    run_proc = _sh(report, f"run {backend} infer_batch harness", ["./gate_harness3_dev"],
                   cwd=cfg_dir, input_text=_stdin_for(inputs))
    if run_proc.returncode != 0:
        return False
    ok, _ = _check_output(report, run_proc.stdout, expected)
    return ok


def _run_nsys_check(report: _Report, cfg_dir: Path) -> bool:
    report.h("nsys check: cudaMemcpy count inside the timed infer_batch loop (ruling R5)", 4)
    nsys = shutil.which("nsys")
    if not nsys:
        report.p("nsys not found on PATH; the cudaMemcpy-count check was NOT run "
                 "(recorded here rather than silently skipped).")
        return True
    stats_base = cfg_dir / "gate_nsys_profile"
    proc = _sh(report, "nsys profile --stats=true",
              [nsys, "profile", "--stats=true", "--force-overwrite=true",
               "-o", str(stats_base), "./gate_harness3_dev"], cwd=cfg_dir)
    combined = (proc.stdout or "") + (proc.stderr or "")
    count = combined.count("cudaMemcpy")
    report.p(f"cudaMemcpy occurrences reported by nsys: {count}")
    if count != 0:
        report.p("FAIL: infer_batch's timed loop must never call cudaMemcpy (ruling R5)")
        return False
    return proc.returncode == 0


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

        _record_versions(report, cc, fc, devcc if backend != "omp" else None)

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

            c_built = _build_c_lib(report, cfg_dir, plan, cc, flags, backend, devcc, devflags)
            f_built = _build_fortran_lib(report, cfg_dir, plan, fc, flags)

            report.h("c harness: per-point infer via target teams loop", 3)
            if c_built:
                if not _run_c_harness1(report, cfg_dir, plan, cc, flags, inputs, expected, env):
                    ok = False
            else:
                report.p("skipped: c library build failed")
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
                if c_built:
                    if not _run_c_harness3_omp(report, cfg_dir, plan, cc, flags, inputs, expected, env):
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
                if c_built:
                    dev_ok = _run_dev_harness3(report, cfg_dir, plan, devcc or
                                               ("nvcc" if backend == "cuda" else "hipcc"),
                                               devflags, backend, inputs, expected)
                    if not dev_ok:
                        ok = False
                    if backend == "cuda" and dev_ok:
                        if not _run_nsys_check(report, cfg_dir):
                            ok = False
                else:
                    report.p(f"skipped: c library build failed (backend={backend})")
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
