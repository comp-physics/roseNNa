import os
import re
import shutil
import subprocess
import numpy as np
import pytest
from onnx import helper, numpy_helper
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from rosenna.emit_kernel import emit_kernel
from rosenna.rt_header import rt_header
from rosenna.emit_c import emit_c, emit_c_recipe, CONSTANT_MEMORY_LIMIT
from tests.conftest import save_model


def test_kernel_source_uses_only_the_rt_macros(golden_model):
    plan = build_plan(load_graph(golden_model("gemm_small")), dtype="f64")
    cu = emit_kernel(plan)
    assert "rosenna_rt.h" in cu and "__global__" in cu and 'extern "C" int gemm_small_infer_batch(' in cu
    for forbidden in ("cudaMalloc", "hipMalloc", "cudaMemcpy", "hipMemcpy", "<<<"):
        assert forbidden not in cu, forbidden   # only rosenna_rt.h may name the runtime


def test_kernel_source_names_no_runtime_symbol_for_a_file_loaded_plan(golden_model):
    # The file-loaded kernel file also holds the plan-step bind of the device
    # weight pointers (called by init); that too must go through
    # rosenna_rt.h, never a cuda*/hip* name.
    plan = build_plan(load_graph(golden_model("gemm_big")), dtype="f64", embed=False)
    cu = emit_kernel(plan)
    assert re.search(r"\b(cuda|hip)[A-Z]", cu) is None
    assert 'extern "C" int gemm_big_device_bind(void) {' in cu
    assert "return gemm_big_device_bind_here();" in cu
    assert "ROSENNA_LAUNCH(gemm_big_kernel" in cu
    # Ruling R10: the launch is checked with a peek (never a sync), status 11.
    assert "if (ROSENNA_LAUNCH_STATUS() != ROSENNA_OK) return 11;" in cu
    # An embedded plan reads its ROSENNA_CONST arrays directly and binds nothing.
    cu_e = emit_kernel(build_plan(load_graph(golden_model("gemm_big")), dtype="f64", embed=True))
    assert "ROSENNA_MEMCPY_TO_SYMBOL" not in cu_e and "device_bind" not in cu_e


def _function_body(text: str, signature_start: str) -> str:
    """The text of one top-level C function, from its signature to its closing brace."""
    start = text.index(signature_start)
    end = text.index("\n}\n", start) + 3
    return text[start:end]


_LOOP_PATH_FORBIDDEN = ("map(to", "map(from", "map(tofrom", "copyin", "copyout",
                        "ROSENNA_MALLOC", "ROSENNA_MEMCPY", "ROSENNA_SYNC",
                        "cudaMemcpy", "hipMemcpy", "cudaMalloc", "hipMalloc",
                        "cudaDeviceSynchronize", "hipDeviceSynchronize",
                        "target update", "update device", "enter data", "exit data",
                        "omp_target_memcpy", "acc_memcpy")


def test_loop_path_never_transfers(golden_model):
    # Controller rulings R5/R6: init is the plan step and the only routine that
    # allocates, transfers or synchronizes; x and y are device-resident in
    # every backend. No transfer can be observed on a host-only build, so
    # this structural check is the CI-able guarantee. Exempt: the body of
    # `<name>_init` and of `<name>_upload`, the static tail of init that
    # holds the cuda/hip allocation and copies (init calls it and nothing
    # else does). Everything else in the .c, and the whole kernel file, must
    # be free of every transfer token. The kernel file's `<name>_device_bind`
    # only forwards to the header's `<name>_device_bind_here`, which is where
    # the one symbol copy of the plan step lives; that header function is
    # checked here to be the sole holder of ROSENNA_MEMCPY_TO_SYMBOL.
    for embed in (True, False):
        name = "gemm_big"; plan = build_plan(load_graph(golden_model(name)), dtype="f64", embed=embed)
        source, header = emit_c(plan)
        cu = emit_kernel(plan)
        rest_c = source
        if not embed:
            init = _function_body(source, f"int {name}_init(")
            upload = _function_body(source, f"static int {name}_upload(")
            assert f"return {name}_upload();" in init and "ROSENNA_MALLOC" in upload
            rest_c = rest_c.replace(init, "").replace(upload, "")
        for forbidden in _LOOP_PATH_FORBIDDEN:
            assert forbidden not in rest_c, (embed, forbidden)
            assert forbidden not in cu, (embed, forbidden)
        if not embed:
            bind_here = _function_body(header, f"static inline int {name}_device_bind_here(")
            assert "ROSENNA_MEMCPY_TO_SYMBOL" in bind_here
            assert header.count("ROSENNA_MEMCPY_TO_SYMBOL") == 1
            assert f"return {name}_device_bind_here();" in _function_body(cu, f'extern "C" int {name}_device_bind(')


def test_rt_header_maps_both_runtimes():
    h = rt_header()
    assert "__HIPCC__" in h and "__CUDACC__" in h and "ROSENNA_LAUNCH" in h
    # Every macro the generated sources use is defined once per runtime.
    for macro in ("ROSENNA_STREAM_T", "ROSENNA_MALLOC", "ROSENNA_MEMCPY_H2D", "ROSENNA_FREE",
                  "ROSENNA_OK", "ROSENNA_SYNC", "ROSENNA_LAUNCH", "ROSENNA_MEMCPY_TO_SYMBOL",
                  "ROSENNA_LAUNCH_STATUS"):
        assert h.count(f"#define {macro}(") + h.count(f"#define {macro} ") == 2, macro


def test_header_declares_infer_batch_with_c_linkage_on_both_forms(golden_model):
    for embed in (True, False):
        plan = build_plan(load_graph(golden_model("gemm_small")), dtype="f64", embed=embed)
        source, header = emit_c(plan)
        # Ruling R7: the batch's pointers are restrict-qualified.
        assert ("int gemm_small_infer_batch(int n, const double *ROSENNA_RESTRICT x, "
                "double *ROSENNA_RESTRICT y, void *stream);") in header
        assert 'extern "C" {' in header and "__cplusplus" in header
        assert "x and y must already be on\n   the device; init is the only routine that transfers." in header
        # The OpenMP fallback lives in the .c under the negation of the CUDA/HIP
        # guard, over device pointers (ruling R5), each pragma under its own guard.
        assert ("int gemm_small_infer_batch(int n, const double *ROSENNA_RESTRICT x, "
                "double *ROSENNA_RESTRICT y, void *stream) {") in source
        assert ('extern "C" int gemm_small_infer_batch(int n, const double *__restrict__ x, '
                "double *__restrict__ y, void *stream) {") in emit_kernel(plan)
        assert "#if !defined(__CUDACC__) && !defined(__HIPCC__)" in source
        assert "#if defined(_OPENMP)\n#pragma omp target teams loop is_device_ptr(x, y)\n" in source
        assert "#elif defined(_OPENACC)\n#pragma acc parallel loop deviceptr(x, y)\n#endif" in source


def test_file_loaded_source_copies_to_the_device_under_the_cuda_guard(golden_model):
    plan = build_plan(load_graph(golden_model("gemm_small")), dtype="f64", embed=False)
    source, header = emit_c(plan)
    assert "double *gemm_small_w0_dev = 0;" in source
    assert "ROSENNA_MALLOC(&gemm_small_w0_dev, sizeof gemm_small_w0)" in source
    assert "ROSENNA_MEMCPY_H2D(gemm_small_w0_dev, gemm_small_w0, sizeof gemm_small_w0)" in source
    assert "return 10;" in source and '#include "rosenna_rt.h"' not in source   # the header includes it
    # init ends by publishing the copies to the kernel's translation unit,
    # through the header's per-translation-unit bind (ruling R8), which any
    # user kernel's translation unit must call as well.
    assert "return gemm_small_device_bind();" in source
    assert "int gemm_small_device_bind(void);" in header
    assert "static inline int gemm_small_device_bind_here(void) {" in header
    assert ("call gemm_small_device_bind_here() once after gemm_small_init()\n"
            "   in every translation unit whose kernels call gemm_small_infer. Embedded\n"
            "   models need nothing.") in header
    assert '#include "rosenna_rt.h"' in header
    # Under an offloading OpenMP/OpenACC build the host arrays have device
    # copies that init updates in the same call (the plan step); each
    # directive under its own guard.
    weights = "gemm_small_w0, gemm_small_b0, gemm_small_w1, gemm_small_b1"
    assert f"#ifdef _OPENMP\n#pragma omp target update to({weights})\n#endif" in source
    assert f"#ifdef _OPENACC\n#pragma acc update device({weights})\n#endif" in source
    assert f"#pragma acc declare create({weights})" in header
    assert "#pragma omp declare target\n#endif\nextern double gemm_small_w0[4];" in header


def test_generated_c_is_warning_free_under_openacc(tmp_path, golden_model):
    # gcc -fopenacc rejects a `routine seq` function reading a file-scope
    # array with no `declare` directive (HEAD before this task failed here);
    # both forms must now compile clean, including a host calling the
    # `parallel loop deviceptr(x, y)` fallback.
    from tests.test_device_c import _omp_cc
    cc = _omp_cc()
    probe = subprocess.run([cc, "-fopenacc", "-x", "c", "-", "-o", os.devnull],
                           input="int main(void){return 0;}", capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip(f"{cc} does not accept -fopenacc")
    for embed in (True, False):
        name = "gemm_small"; plan = build_plan(load_graph(golden_model(name)), dtype="f64", embed=embed)
        d = tmp_path / ("e" if embed else "f"); d.mkdir()
        source, header = emit_c(plan)
        (d / f"{name}.c").write_text(source); (d / f"{name}.h").write_text(header)
        (d / "host.c").write_text(f"""
#include "{name}.h"
int main(void) {{ double x[2] = {{0.5, 0.5}}, y[3]; return {name}_infer_batch(1, x, y, 0); }}
""")
        r = subprocess.run([cc, "-O2", "-Wall", "-Wextra", "-std=c11", "-fopenacc", f"{name}.c", "host.c", "-lm", "-o", "host"],
                           cwd=d, capture_output=True, text=True)
        assert r.returncode == 0 and r.stderr == "", r.stderr
        if embed:
            assert subprocess.run(["./host"], cwd=d, capture_output=True).returncode == 0
    # Status 10 is in the emitted legend, next to the routine that returns it.
    assert "10  device allocation or copy failed in init" in source
    # Device code reads the per-translation-unit __constant__ pointer table;
    # host code, under any compiler, reads the host arrays.
    assert "static __constant__ const double *gemm_small_devw[4];" in header
    assert "#define ROSENNA_REF_gemm_small_w0 gemm_small_devw[0]" in header
    assert "#define ROSENNA_REF_gemm_small_w0 gemm_small_w0" in header
    assert "__CUDA_ARCH__" in header and "__HIP_DEVICE_COMPILE__" in header


def test_embedded_infer_is_a_stub_in_the_host_pass_of_a_device_build(golden_model):
    # Ruling R9: the host instantiation of __host__ __device__ infer must not
    # read the __constant__/__device__ arrays (nvcc diagnoses it; hip-clang's
    # host shadow is undefined). It asserts instead, a no-op under NDEBUG.
    # The literals are emitted once; there is no host twin.
    plan = build_plan(load_graph(golden_model("gemm_small")), dtype="f64", embed=True)
    _, header = emit_c(plan)
    assert "#include <assert.h>" in header
    assert "#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)\n#define ROSENNA_INFER_HOST_STUB 0\n#else\n#define ROSENNA_INFER_HOST_STUB 1\n#endif" in header
    assert ('#if ROSENNA_INFER_HOST_STUB\n    (void)x;\n    (void)y;\n    assert(0 && "rosenna: gemm_small_infer '
            'is device-only in a CUDA/HIP build; call it from a kernel or use gemm_small_infer_batch");\n#else') in header
    assert header.count("gemm_small_w0[4] = {") == 1
    assert "host instantiation\n   of infer is a stub" in header
    # A file-loaded plan computes on the host in every pass (it reads the host arrays).
    _, header_f = emit_c(build_plan(load_graph(golden_model("gemm_small")), dtype="f64", embed=False))
    assert "ROSENNA_INFER_HOST_STUB 1" not in header_f and "assert(" not in header_f
    # Both headers can share one translation unit: every macro is #undef'd first.
    for macro in ("ROSENNA_DEVICE_FN", "ROSENNA_CONST", "ROSENNA_RESTRICT", "ROSENNA_INFER_HOST_STUB"):
        assert f"#undef {macro}" in header and f"#undef {macro}" in header_f


def _embedded_plan(tmp_path, name, n_in, n_out):
    w = numpy_helper.from_array(np.random.default_rng(1).uniform(-1, 1, (n_in, n_out)).astype(np.float32), "w")
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="m0")
    path = save_model(tmp_path, name, [node], [w], (1, n_in), (1, n_out))
    return build_plan(load_graph(path), dtype="f64", embed=True)


def test_constant_memory_limit_selects_the_device_qualifier(tmp_path):
    # Controller ruling R4: CUDA __constant__ memory is 64 KB per module and
    # the embed threshold is 1M parameters, so an embedded model past the
    # constant budget must go to __device__ const instead. The cut is 48 KB of
    # weight bytes; a header states which it chose.
    assert CONSTANT_MEMORY_LIMIT == 48 * 1024
    small = _embedded_plan(tmp_path, "under", 64, 90)       # 5760 f64 = 46080 B < 48 KB
    big = _embedded_plan(tmp_path, "over", 64, 100)         # 6400 f64 = 51200 B > 48 KB
    _, h_small = emit_c(small)
    _, h_big = emit_c(big)
    assert "#define ROSENNA_CONST static __constant__" in h_small
    assert "__device__ const" not in h_small
    assert "#define ROSENNA_CONST static __device__ const" in h_big
    assert "__constant__" not in h_big
    assert "46080" in h_small and "51200" in h_big       # the comment names the byte count it judged


def _omp_host(name, n_in, n_out, npts, init):
    """A host that maps its data first and hands infer_batch device pointers (R5).

    Under the gcc host fallback every step is the identity and the batch must
    agree bit for bit with the host's own calls of the header inline.
    """
    return f"""
#include <stdio.h>
#include <stdlib.h>
#include "{name}.h"
int main(void) {{
  const int nx = {npts} * {n_in}, ny = {npts} * {n_out};
  double *x = malloc(nx * sizeof *x), *y = malloc(ny * sizeof *y), *yb = malloc(ny * sizeof *yb);
  int status = 0;
  {init}
  for (int c = 0; c < nx; ++c) x[c] = 0.01 * c - 0.3;
  for (int p = 0; p < {npts}; ++p) {name}_infer(x + p * {n_in}, y + p * {n_out});
#ifdef _OPENMP
  #pragma omp target enter data map(to: x[0:nx]) map(alloc: yb[0:ny])
  #pragma omp target data use_device_ptr(x, yb)
#endif
  {{
    status = {name}_infer_batch({npts}, x, yb, 0);
    if (status == 0) status = {name}_infer_batch(0, x, yb, 0) ? 4 : 0;
  }}
#ifdef _OPENMP
  #pragma omp target exit data map(from: yb[0:ny]) map(delete: x[0:nx])
#endif
  if (status) return status;
  for (int c = 0; c < ny; ++c) if (y[c] != yb[c]) return 5;
  puts("agree"); return 0; }}
"""


def _omp_build_and_run(tmp_path, name, plan, cc, init):
    source, header = emit_c(plan)
    (tmp_path / f"{name}.c").write_text(source); (tmp_path / f"{name}.h").write_text(header)
    (tmp_path / "Makefile").write_text(emit_c_recipe(plan))
    r = subprocess.run(["make", f"CC={cc}", "ROSENNA_BACKEND=omp", "ROSENNA_OFFLOAD_FLAGS=-fopenmp"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "warning" not in r.stderr, r.stderr
    (tmp_path / "host.c").write_text(_omp_host(name, plan.input.shape[0], plan.output.shape[0], 16, init))
    r = subprocess.run([cc, "-O2", "-Wall", "-Wextra", "-std=c11", "-fopenmp", "host.c", f"lib{name}.a", "-lm", "-o", "host"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0 and r.stderr == "", r.stderr
    return subprocess.run(["./host"], cwd=tmp_path, capture_output=True, text=True)


def test_omp_backend_infer_batch_matches_inline(tmp_path, golden_model):
    # The fallback backend: a library-side target loop over device pointers
    # (is_device_ptr) must agree exactly with the host's own loop over the
    # header inline. The host maps x and y first and passes what
    # use_device_ptr yields, per ruling R5.
    from tests.test_device_c import _omp_cc
    from rosenna.weights import write_weights
    name = "gemm_big"; graph = load_graph(golden_model(name)); plan = build_plan(graph, dtype="f64", embed=False)
    write_weights(plan, graph, tmp_path / f"{name}.rwt")
    r = _omp_build_and_run(tmp_path, name, plan, _omp_cc(), f'if ({name}_init("{name}.rwt")) return 2;')
    assert r.returncode == 0 and r.stdout.strip() == "agree", (r.returncode, r.stdout, r.stderr)


def test_omp_backend_builds_an_embedded_plan_too(tmp_path, golden_model):
    # An embedded plan's .c holds only the fallback infer_batch; the recipe
    # must still produce a library for it, and the batch must agree with the
    # inline.
    from tests.test_device_c import _omp_cc
    name = "gemm_small"; graph = load_graph(golden_model(name)); plan = build_plan(graph, dtype="f64", embed=True)
    r = _omp_build_and_run(tmp_path, name, plan, _omp_cc(), "")
    assert r.returncode == 0 and r.stdout.strip() == "agree", (r.returncode, r.stdout, r.stderr)


def test_omp_target_loop_in_infer_batch_is_real(tmp_path, golden_model):
    # As in test_device_c: a host-only libgomp refuses a target region under
    # OMP_TARGET_OFFLOAD=MANDATORY, so the library's own loop must fail there
    # -- proof the pragma is a real target construct and not ignored.
    from tests.test_device_c import _omp_cc
    name = "gemm_small"; plan = build_plan(load_graph(golden_model(name)), dtype="f64", embed=True)
    source, header = emit_c(plan)
    (tmp_path / f"{name}.c").write_text(source); (tmp_path / f"{name}.h").write_text(header)
    (tmp_path / "host.c").write_text(f"""
#include "{name}.h"
int main(void) {{ double x[2] = {{0.5, 0.5}}, y[3]; return {name}_infer_batch(1, x, y, 0); }}
""")
    cc = _omp_cc()
    r = subprocess.run([cc, "-O2", "-std=c11", "-fopenmp", "host.c", f"{name}.c", "-lm", "-o", "host"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run(["./host"], cwd=tmp_path, capture_output=True, text=True,
                       env={**os.environ, "OMP_TARGET_OFFLOAD": "MANDATORY"})
    assert r.returncode != 0 and "MANDATORY" in r.stderr


def test_generated_c_is_warning_free_under_a_plain_compiler(tmp_path, golden_model):
    # clang without -fopenmp and without CUDA: every guard is false and the
    # fallback still has to compile clean under -Wall -Wextra.
    cc = shutil.which("clang") or shutil.which("cc")
    if cc is None:
        pytest.skip("no plain C compiler found")
    for embed in (True, False):
        name = "gemm_small"; plan = build_plan(load_graph(golden_model(name)), dtype="f64", embed=embed)
        d = tmp_path / ("e" if embed else "f"); d.mkdir()
        source, header = emit_c(plan)
        (d / f"{name}.c").write_text(source); (d / f"{name}.h").write_text(header)
        r = subprocess.run([cc, "-O2", "-Wall", "-Wextra", "-std=c11", "-c", f"{name}.c"], cwd=d, capture_output=True, text=True)
        assert r.returncode == 0 and r.stderr == "", r.stderr


def test_generated_c_compiles_as_cpp_with_the_rt_header_stubbed(tmp_path, golden_model):
    # nvcc and hipcc compile <name>.c as C++ (-x cu / -x hip). No such compiler
    # runs here, so this is the nearest local check: a host C++ compiler with
    # __CUDACC__ forced on and the CUDA keywords and runtime replaced by inert
    # stand-ins. It catches C-only constructs in the .c, linkage mismatches
    # between the header's extern "C" block and the definitions, and any use
    # of a runtime name outside rosenna_rt.h. It proves nothing about device
    # code generation; that waits for the nvcc CI job.
    #
    # Two variants of the kernel translation unit. With __CUDA_ARCH__ defined
    # (as in nvcc's device pass) infer reads the file-loaded weights through
    # the per-translation-unit table that device_bind_here fills, and the
    # embedded body is the real one: init -> upload -> bind -> launch must
    # agree bit for bit with the inline, and an unbound table is status 10.
    # Without it (nvcc's host pass) the file-loaded kernel reads the host
    # arrays and the embedded kernel is the R9 stub, so that variant is
    # compiled and linked, and run only for the file-loaded plan.
    cxx = shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no C++ compiler found")
    stub = """
#ifndef ROSENNA_RT_H
#define ROSENNA_RT_H
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
typedef void *ROSENNA_STREAM_T_;
static inline int rosenna_stub_malloc(void **p, size_t n) { *p = malloc(n); return *p ? 0 : 1; }
static inline int rosenna_stub_h2d(void *d, const void *h, size_t n) { memcpy(d, h, n); return 0; }
static inline int rosenna_stub_free(void *p) { free(p); return 0; }
static inline int rosenna_stub_sync(void *s) { (void)s; return 0; }
#define ROSENNA_STREAM_T ROSENNA_STREAM_T_
#define ROSENNA_MALLOC(p, n) rosenna_stub_malloc((void **)(p), (n))
#define ROSENNA_MEMCPY_H2D(d, h, n) rosenna_stub_h2d((d), (h), (n))
#define ROSENNA_MEMCPY_TO_SYMBOL(sym, src, n) rosenna_stub_h2d((sym), (src), (n))
#define ROSENNA_FREE(p) rosenna_stub_free(p)
#define ROSENNA_OK 0
#define ROSENNA_SYNC(s) rosenna_stub_sync(s)
#define ROSENNA_LAUNCH(k, g, b, s, ...) \\
    do { for (blockIdx.x = 0; blockIdx.x < (unsigned)((g) * (b)); ++blockIdx.x) k(__VA_ARGS__); } while (0)
#define ROSENNA_LAUNCH_STATUS() 0
#endif
"""
    from rosenna.weights import write_weights
    for embed in (True, False):
        for arch in (None, "800"):
            name = "gemm_small"; graph = load_graph(golden_model(name)); plan = build_plan(graph, dtype="f64", embed=embed)
            d = tmp_path / f"{'e' if embed else 'f'}{arch or ''}"; d.mkdir()
            source, header = emit_c(plan)
            (d / f"{name}.c").write_text(source); (d / f"{name}.h").write_text(header)
            (d / "rosenna_rt.h").write_text(stub)
            if not embed:
                write_weights(plan, graph, d / f"{name}.rwt")
            (d / f"{name}_kernel.cu").write_text(emit_kernel(plan))
            # blockIdx/blockDim/threadIdx are CUDA builtins; give the stub the
            # three as plain objects so the kernel body parses, with one thread
            # per block so the stub launch's loop over blockIdx.x walks the points.
            (d / "builtins.h").write_text(
                "struct rosenna_dim3 { unsigned int x, y, z; };\n"
                "static struct rosenna_dim3 blockIdx = {0, 0, 0};\n"
                "static const struct rosenna_dim3 blockDim = {1, 0, 0}, threadIdx = {0, 0, 0};\n")
            # -ffp-contract=off on every translation unit: the host may be built
            # by a different compiler than the library, and clang contracts
            # `acc += a * b` to an FMA by default where g++ in ISO mode does not,
            # which breaks the bit-for-bit comparison below for no real reason.
            common = [cxx, "-x", "c++", "-std=c++11", "-ffp-contract=off", "-Wall", "-Wextra", "-c",
                      "-D__CUDACC__=1", "-D__host__=", "-D__device__=", "-D__constant__=", "-D__global__=",
                      "-include", "builtins.h"]
            r = subprocess.run(common + [f"{name}.c", "-o", f"{name}.o"], cwd=d, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            arch_flag = [f"-D__CUDA_ARCH__={arch}"] if arch else []
            r = subprocess.run(common + arch_flag + [f"{name}_kernel.cu", "-o", f"{name}_kernel.o"], cwd=d, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            # A plain C host TU links against the C++-compiled objects: the API
            # has C linkage. With the stubbed runtime, init's upload and bind run
            # on host memory and the "launch" is a serial call of the kernel body.
            n_in, n_out = plan.input.shape[0], plan.output.shape[0]
            unbound = "" if embed else f"if ({name}_infer_batch(4, x, yb, 0) != 10) return 6;"
            init = "" if embed else f'if ({name}_init("none.rwt") != 1) return 2; if ({name}_init("{name}.rwt") != 0) return 3;'
            (d / "host.c").write_text(f"""
#include "{name}.h"
int main(void) {{ double x[4 * {n_in}], y[4 * {n_out}], yb[4 * {n_out}];
  for (int c = 0; c < 4 * {n_in}; ++c) x[c] = 0.1 * c - 0.2;
  {unbound}
  {init}
  for (int p = 0; p < 4; ++p) {name}_infer(x + p * {n_in}, y + p * {n_out});
  if ({name}_infer_batch(4, x, yb, 0)) return 4;
  for (int c = 0; c < 4 * {n_out}; ++c) if (y[c] != yb[c]) return 5;
  return {name}_infer_batch(0, x, yb, 0); }}
""")
            cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
            r = subprocess.run([cc, "-std=c11", "-ffp-contract=off", "-c", "host.c", "-o", "host.o"], cwd=d, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            r = subprocess.run([cxx, "host.o", f"{name}.o", f"{name}_kernel.o", "-lm", "-o", "host"], cwd=d, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            if embed and arch is None:
                continue        # the kernel would hit the R9 host stub's assert
            r = subprocess.run(["./host"], cwd=d, capture_output=True, text=True)
            assert r.returncode == 0, (embed, arch, r.returncode, r.stderr)


def test_recipe_selects_the_backend(golden_model):
    plan = build_plan(load_graph(golden_model("gemm_small")), dtype="f64", embed=False)
    mk = emit_c_recipe(plan)
    assert "ROSENNA_BACKEND ?= omp" in mk
    assert "ifeq ($(ROSENNA_BACKEND),cuda)" in mk and "else ifeq ($(ROSENNA_BACKEND),hip)" in mk
    assert "DEVCC ?= nvcc" in mk and "DEVCC ?= hipcc" in mk
    assert "-x cu -c $< -o $@" in mk and "-x hip -c $< -o $@" in mk
    assert "gemm_small_kernel.o: gemm_small_kernel.cu gemm_small.h rosenna_rt.h" in mk
    assert "$(CC) $(CFLAGS) $(ROSENNA_OFFLOAD_FLAGS) -c $< -o $@" in mk


def test_generate_writes_the_kernel_and_rt_header(tmp_path, golden_model):
    from rosenna.cli import main
    rc = main(["generate", str(golden_model("gemm_small")), "--lang", "c", "--out", str(tmp_path)])
    assert rc == 0
    for f in ["gemm_small.c", "gemm_small.h", "gemm_small.mk", "gemm_small_kernel.cu", "rosenna_rt.h"]:
        assert (tmp_path / f).exists(), f
    assert (tmp_path / "rosenna_rt.h").read_text() == rt_header()
    rc = main(["generate", str(golden_model("gemm_small")), "--lang", "fortran", "--out", str(tmp_path / "f")])
    assert rc == 0
    assert not (tmp_path / "f" / "rosenna_rt.h").exists()


def _nvcc_build(d, name, plan):
    source, header = emit_c(plan)
    (d / f"{name}.c").write_text(source); (d / f"{name}.h").write_text(header)
    (d / f"{name}_kernel.cu").write_text(emit_kernel(plan)); (d / "rosenna_rt.h").write_text(rt_header())
    (d / "Makefile").write_text(emit_c_recipe(plan))
    r = subprocess.run(["make", "ROSENNA_BACKEND=cuda", "DEVFLAGS=-O2 -arch=sm_80"], cwd=d, capture_output=True, text=True)
    # nvcc's own diagnostics (warnings included) go to the CI log under -s:
    # they are the only view of this code a CUDA compiler gives us.
    print(f"\n--- nvcc {name} embed={plan.embed} dtype={plan.dtype} ---\n{r.stdout}{r.stderr}")
    assert r.returncode == 0, r.stderr
    assert (d / f"lib{name}.a").exists()


@pytest.mark.skipif(not shutil.which("nvcc"), reason="nvcc not installed")
def test_cuda_backend_compiles(tmp_path, golden_model):
    # Compile only: nvcc builds device code with no GPU present. Running needs the GPU gate.
    for embed in (True, False):
        name = "gemm_small"; graph = load_graph(golden_model(name)); plan = build_plan(graph, dtype="f64", embed=embed)
        d = tmp_path / ("e" if embed else "f"); d.mkdir()
        _nvcc_build(d, name, plan)


@pytest.mark.skipif(not shutil.which("nvcc"), reason="nvcc not installed")
@pytest.mark.parametrize("name", ["gemm_big", "gemm_nobias", "droplet", "batchnet"])
@pytest.mark.parametrize("dtype", ["f32", "f64"])
@pytest.mark.parametrize("embed", [True, False])
def test_cuda_backend_compiles_every_dense_model(tmp_path, golden_model, name, dtype, embed):
    # The activations (tanhf/expf and their double forms) and every weight
    # layout the plan can produce must also pass nvcc, not just gemm_small.
    plan = build_plan(load_graph(golden_model(name)), dtype=dtype, embed=embed)
    _nvcc_build(tmp_path, name, plan)


@pytest.mark.skipif(not shutil.which("nvcc"), reason="nvcc not installed")
def test_cuda_backend_compiles_past_the_constant_memory_budget(tmp_path):
    # An embedded model over CONSTANT_MEMORY_LIMIT takes the __device__ const
    # path (ruling R4); nvcc must accept that header too.
    plan = _embedded_plan(tmp_path, "over", 64, 100)
    assert "__device__ const" in emit_c(plan)[1]
    d = tmp_path / "b"; d.mkdir()
    _nvcc_build(d, "over", plan)
