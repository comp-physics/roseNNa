"""Render a plan as a self-contained C source/header pair."""
from .abi import name_capacity, rank_capacity, status_code_comment
import re

from .plan import Plan, lstm_initial_state

_CTYPE = {"f32": "float", "f64": "double"}
_DTYPE_CODE = {"f32": 0, "f64": 1}
_ITEMSIZE = {"f32": 4, "f64": 8}
# The CUDA/HIP compiler guard. __HIPCC__ is defined by hip-clang itself for
# any HIP compilation (and by the hipcc wrapper besides), so it is the one
# HIP macro to test. __HIP__ is NOT a HIP-compilation signal: clang's OpenMP
# AMDGPU device pass defines it from openmp_wrappers/math.h to borrow HIP's
# device math, so a header that accepted it took the __device__ branch under
# `amdclang -fopenmp --offload-arch=gfx90a` (ruling R23, HIP twin).
_IS_CUDA = "defined(__CUDACC__)"
_IS_HIP = "defined(__HIPCC__)"
_CUDA_GUARD = f"#if {_IS_CUDA} || {_IS_HIP}"
_NOT_CUDA_GUARD = "#if !defined(__CUDACC__) && !defined(__HIPCC__)"
# Which half of the library this translation unit provides, set by the recipe
# rather than sniffed from the compiler. <name>.c is ALWAYS built by the host
# compiler with its offload flags -- it owns the declare-target weights and the
# `target update` that puts them on the device, which only that compiler can
# make sense of -- and <name>_kernel.cu owns every CUDA/HIP symbol. Detecting
# __CUDACC__ here instead is what forced the recipe to hand the whole of
# <name>.c to nvcc, silently disabling the declare-target on the weights, so a
# per-point OpenMP/OpenACC host had to link a second omp-backend archive.
_NATIVE_GUARD = "#ifdef ROSENNA_NATIVE_KERNEL"
_NOT_NATIVE_GUARD = "#ifndef ROSENNA_NATIVE_KERNEL"
# Device-pass guard: nvcc defines __CUDA_ARCH__ and hipcc __HIP_DEVICE_COMPILE__
# only while compiling for the device, so a header-inline function can read one
# storage in its host instantiation and another in its device instantiation.
# Each arch macro is tested together with its compiler macro (ruling R23):
# clang's OpenMP nvptx device pass defines __CUDA_ARCH__ without __CUDACC__,
# and there the host arrays, not the __constant__ table (which only the
# CUDA/HIP guard declares), are the storage that exists.
_DEVICE_PASS_GUARD = (f"#if ({_IS_CUDA} && defined(__CUDA_ARCH__)) || "
                      f"({_IS_HIP} && defined(__HIP_DEVICE_COMPILE__))")

# Controller ruling R4. CUDA __constant__ memory is 64 KB per module, while
# a model embeds by default below EMBED_THRESHOLD (1M parameters, up to 8 MB
# of f64), so an embedded model whose weights exceed the constant budget must
# be placed in ordinary device memory or nvcc rejects the header. The decision
# is per model, made once at generation time, and the header records which it
# took.
#
# The cut is 2 KB of weight bytes, not the 48 KB the 64-KB-per-module bank
# would allow, and the reason is the constant *cache*, not the bank. Constant
# memory is fast only while the working set fits a per-SM cache of a couple of
# KB; past that every weight read misses. Measured on an A100 (ncu,
# smsp__warp_issue_stalled_imc_miss_per_warp_active), infer_batch spends 71% of
# its warp-issue stalls on constant-cache misses for gemm_big (23 KB of
# weights) and 90% for batchnet (16 KB), against 0.09% for the same models
# reading the same weights from ordinary device memory:
#
#   model        weight bytes   __constant__   __device__ const
#   gemm_small            120      0.027 ns/pt      0.028 ns/pt
#   gemm_nobias           160      0.027            0.028
#   droplet               344      0.041            0.041
#   batchnet           15,904      6.731            2.384
#   gemm_big           23,208      3.574            1.423
#
# so 2 KB keeps the models that measure the same and moves out the ones that
# pay 2.5-2.8x. This is calibrated for the shape this library targets: a
# per-point closure with a handful of inputs, where the weights dominate the
# cache. A model with a wide input streams enough of x through L1 to change
# the balance -- synthetic 16-input models measure ~1.3x the other way -- so
# if one of those ever turns up, this wants to become a generate-time flag
# rather than a different constant.
CONSTANT_MEMORY_LIMIT = 2 * 1024
# The one-thread-per-point kernel's block size (emit_kernel).
KERNEL_TILE = 128
# Output columns a dense layer computes per pass over its input vector. Each
# element of the input is loaded once per pass, so the loads of a per-thread
# activation vector -- scratch memory on a GPU, and the bound on a kernel
# with wide layers -- are amortised over GEMM_BLOCK dot products. On an
# MI210 (reaction_patch, 18-128-128-2) this is 3.2x over one column per
# pass; staging the weights in shared memory instead was slower than either.
# A layer narrower than GEMM_BLOCK_MIN_IN keeps one column per pass: there
# the extra accumulators cost registers and buy little or nothing (gemm_big,
# 2-40 wide, ran 3x slower blocked at a million points; at 64 wide C gained
# 10% and Fortran lost 20%; at 128 both gained 3.5x).
GEMM_BLOCK = 8
GEMM_BLOCK_MIN_IN = 96


def gemm_block(op) -> int:
    return GEMM_BLOCK if op.n_in >= GEMM_BLOCK_MIN_IN else 1
# Embedding must be lossless: %.17g round-trips any f64, %.9g any f32
# (Steele & White / Ryu-style shortest-exact-decimal bounds).
_EMBED_FMT = {"f32": "%.9g", "f64": "%.17g"}

# Activation expressions, per plan dtype. An f32 plan must call the float
# intrinsics: tanh/exp on a float promote the whole expression to double, so
# the C backend computed f32 models in double precision while Fortran's f32
# path used the single-precision intrinsics, and the inner loop lost its
# vectorization. relu is written `v < 0 ? 0 : v`, not `v > 0 ? v : 0`, so
# that a NaN -- whose comparisons are all false -- comes out as itself
# instead of being laundered into a plausible zero.
_ACT_C = {
    "f32": {"relu": "{v} < 0.0f ? 0.0f : {v}", "tanh": "tanhf({v})",
            "sigmoid": "1.0f / (1.0f + expf(-({v})))"},
    "f64": {"relu": "{v} < 0.0 ? 0.0 : {v}", "tanh": "tanh({v})",
            "sigmoid": "1.0 / (1.0 + exp(-({v})))"},
}
_ZERO = {"f32": "0.0f", "f64": "0.0"}


def _c_string(s: str) -> str:
    """Escape a plan-supplied (ASCII) name as a C string literal body."""
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _weight_size(shape) -> int:
    size = 1
    for d in shape:
        size *= d
    return size


def _c_weight_symbol(model: str, symbol: str) -> str:
    """The C external identifier for a plan weight symbol (controller ruling R1).

    plan.py names every model's weights `w0`, `b0`, `w1`, ... uniformly:
    those symbols enter the plan hash and Fortran's module scope, where the
    `module` keyword already isolates them, so plan.py stays as it is.
    Dropping `static` from the C weight definitions (this task) gives them
    external linkage, though, and two different models linked into one host
    would then collide on `_w0`/`_b0`. Every C site that names a weight
    array goes through this one helper, prefixed with the model name, so no
    site can drift out of sync with another.
    """
    return f"{model}_{symbol}"


def _c_weight_ref_macro(model: str, symbol: str) -> str:
    """The identifier `infer` uses to read a file-loaded weight.

    A file-loaded plan's header holds two different storages for the same
    weight -- the host array `<symbol>` and, in the device pass of nvcc or
    hipcc, an entry of the per-translation-unit __constant__ pointer table
    (see _emit_device_weight_table) -- and `infer` is emitted exactly once,
    so it cannot spell either name directly. This macro (itself prefixed with
    the model-qualified symbol, so it carries ruling R1's collision safety
    same as every other external name here) is `#define`d to whichever of
    the two the device-pass guard selects; `infer`'s body reads only this
    name. An embedded plan has no such split -- its weights are one
    ROSENNA_CONST array reachable in either pass -- so `infer` reads the
    plain symbol directly there and never goes through this macro.
    """
    return f"ROSENNA_REF_{_c_weight_symbol(model, symbol)}"


def _weight_ref(plan: Plan, model: str, symbol: str) -> str:
    return _c_weight_symbol(model, symbol) if plan.embed else _c_weight_ref_macro(model, symbol)


def _format_embedded_value(v: float, dtype: str) -> str:
    """Format one embedded weight, losslessly, as a C literal of the plan's dtype.

    %g drops the decimal point for an exact integer (`1.0` -> `"1"`), and `1f`
    is not a floating-constant in C -- the `f` suffix is only legal directly
    after a decimal point or an exponent -- so an f32 value with neither gets
    one inserted before the suffix is appended.
    """
    s = _EMBED_FMT[dtype] % v
    if dtype == "f32":
        if not any(c in s for c in ".eE"):
            s += ".0"
        s += "f"
    return s


def _weight_index_c(weight_by_symbol: dict, op) -> str:
    """Decide the accumulation index order from the op's own transB flag.

    A transB=1 Gemm weight has ONNX shape (n_out, n_in), stored row-major and
    indexed w[i * n_in + j]. A transB=0 weight, or any MatMul weight, has
    ONNX shape (n_in, n_out), indexed w[j * n_out + i]. Unlike the Fortran
    side, C never reverses the dimensions, so no transposition happens
    anywhere.

    The flag is authoritative. The shape check below only makes a plan that
    disagrees with its own weights fail loudly; sniffing the convention back
    out of the shape, which is what this replaces, is ambiguous exactly when
    n_in == n_out and silently transposes a square weight.
    """
    spec = weight_by_symbol[op.weight]
    expected = (op.n_out, op.n_in) if op.trans_b else (op.n_in, op.n_out)
    if tuple(spec.shape) != expected:
        raise AssertionError(
            f"weight {op.weight}: shape {tuple(spec.shape)} does not match the layout "
            f"the plan claims (transB={int(op.trans_b)} implies {expected})")
    return f"i * {op.n_in} + j" if op.trans_b else f"j * {op.n_out} + i"


def emit_c(plan: Plan) -> tuple:
    ctype = _CTYPE[plan.dtype]
    header = _emit_header(plan, ctype)
    # nvcc and hipcc compile this file as C++ (-x cu / -x hip in the recipe)
    # so that init can call the runtime, so every external definition sits
    # in an extern "C" block matching the header's declarations; under a C
    # compiler the block is not there.
    lines = _emit_source_head(plan, ctype)
    if plan.embed:
        # Every weight is a ROSENNA_CONST array in the header, so the source
        # has nothing to define or load; it holds only the OpenMP-fallback
        # infer_batch. (Under nvcc/hipcc that macro is __constant__ or
        # __device__ const; the nvcc form is validated on an A100 by the GPU
        # gate, the hipcc form is not.)
        pass
    else:
        lines += _emit_load(plan)
        lines += _emit_upload(plan)
        lines += _emit_init(plan)
    lines += _emit_fallback_infer_batch(plan, ctype)
    lines += ["#if defined(__cplusplus)", "}", "#endif"]
    source = "\n".join(lines) + "\n"
    return source, header


def emit_c_recipe(plan: Plan) -> str:
    """A Makefile fragment that builds lib<name>.a for one of three backends.

    <name>.c is built by the HOST compiler with the host's offload flags under
    every backend, because it owns the declare-target weights and the `target
    update` that puts them on the device -- only that compiler can act on
    those. ROSENNA_BACKEND=cuda|hip adds <name>_kernel.cu, built by nvcc or
    hipcc, which owns every CUDA/HIP symbol, and defines ROSENNA_NATIVE_KERNEL
    so <name>.c yields the batched entry points to it.

    That is what lets ONE archive serve both call paths. Before, the cuda/hip
    recipe handed <name>.c to nvcc as well, which silently disabled the
    declare-target on the weights it defines -- so a per-point OpenMP/OpenACC
    host could not use a file-loaded cuda/hip archive at all and had to link a
    second omp-backend one built by its own compiler.
    """
    n = plan.model
    return f"""# Generated by rosenna. Builds lib{n}.a; ROSENNA_BACKEND selects the batched path.
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
ROSENNA_OFFLOAD_FLAGS ?=
DEVFLAGS ?= -O2
# cuda | hip | omp
ROSENNA_BACKEND ?= omp

ifeq ($(ROSENNA_BACKEND),cuda)
DEVCC ?= nvcc
lib{n}.a: {n}.o {n}_kernel.o
\tar rcs $@ $^
{n}_kernel.o: {n}_kernel.cu {n}.h rosenna_rt.h
\t$(DEVCC) $(DEVFLAGS) -c $< -o $@
{n}.o: {n}.c {n}.h
\t$(CC) $(CFLAGS) $(ROSENNA_OFFLOAD_FLAGS) -DROSENNA_NATIVE_KERNEL -c $< -o $@
else ifeq ($(ROSENNA_BACKEND),hip)
DEVCC ?= hipcc
lib{n}.a: {n}.o {n}_kernel.o
\tar rcs $@ $^
{n}_kernel.o: {n}_kernel.cu {n}.h rosenna_rt.h
\t$(DEVCC) $(DEVFLAGS) -x hip -c $< -o $@
{n}.o: {n}.c {n}.h
\t$(CC) $(CFLAGS) $(ROSENNA_OFFLOAD_FLAGS) -DROSENNA_NATIVE_KERNEL -c $< -o $@
else
lib{n}.a: {n}.o
\tar rcs $@ $^
{n}.o: {n}.c {n}.h
\t$(CC) $(CFLAGS) $(ROSENNA_OFFLOAD_FLAGS) -c $< -o $@
endif
clean:
\trm -f {n}.o {n}_kernel.o lib{n}.a
.PHONY: clean
"""


def _embedded_weight_bytes(plan: Plan) -> int:
    return plan.n_params * _ITEMSIZE[plan.dtype] if plan.embed else 0


def _emit_header(plan: Plan, ctype: str) -> str:
    m = plan.model
    guard = f"ROSENNA_{m.upper()}_H"
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "/* Generated by rosenna. Do not edit. */",
        "",
        "#include <math.h>",
    ]
    if plan.embed:
        lines.append("#include <assert.h>")
    else:
        lines += [
            "/* The runtime macros the per-translation-unit bind below needs; a",
            "   host C compiler never sees this include. */",
            _CUDA_GUARD,
            '#include "rosenna_rt.h"',
            "#endif",
        ]
    lines.append("")
    lines += _emit_device_macros(plan)
    lines += [
        "#if defined(__cplusplus)",
        'extern "C" {',
        "#endif",
        "",
    ]
    if not plan.embed:
        lines += _emit_weight_declarations(plan, ctype)
        lines += [
            f"int {m}_init(const char *path);",
            "",
        ]
    lines += [
        "/* Batched inference over n points stored contiguously: point p reads",
        f"   x + p * {n_in} and writes y + p * {n_out}. x and y must already be on",
        "   the device; init is the only routine that transfers. This call",
        "   allocates nothing, copies nothing and never synchronizes; the caller",
        f"   owns the stream. Which implementation the library holds is fixed when",
        f"   lib{m}.a is built (ROSENNA_BACKEND in {m}.mk); the two are never",
        "   linked together.",
        "   cuda/hip backend: x and y are raw device pointers holding n * n_in",
        "     and n * n_out values; stream is a cudaStream_t / hipStream_t, or",
        "     NULL for the default stream; the launch is asynchronous on it.",
        "   omp backend: x and y are device pointers the host obtained from",
        "     omp_target_alloc or from use_device_ptr on data it mapped (on a",
        "     host-only build those are the host pointers and the loop runs on",
        "     the CPU); stream is ignored.",
        f"   Returns 0; 10 if a device allocation or copy failed in {m}_init",
        "   (cuda/hip backend of a file-loaded model; init never having been",
        "   called counts as that); 11 if the kernel launch failed (cuda/hip). */",
        f"int {m}_infer_batch(int n, const {ctype} *ROSENNA_RESTRICT x, {ctype} *ROSENNA_RESTRICT y, void *stream);",
        "",
        f"/* Wait for every {m}_infer_batch launched on `stream` to finish. The",
        "   backend-agnostic way for a host that has no stream of its own -- an",
        "   OpenMP host whose next target region would otherwise race a cuda/hip",
        "   launch -- to order the two: StreamSynchronize in the cuda/hip",
        "   archive, a no-op in the omp one, whose loop is synchronous. Returns",
        "   0, or 11 if the wait reported an error (an asynchronous fault in the",
        "   kernel surfaces here). */",
        f"int {m}_sync(void *stream);",
        "",
        *([f"/* One sample, x and y device-resident as for infer_batch, one launch per op",
           "   with the thread index over the op's output elements and the intermediate",
           "   activations in static device buffers: the form for a model over a whole",
           "   field, whose activations are too large for a thread's locals. The buffers",
           "   are shared by every call, so calls on different streams must not overlap.",
           "   Asynchronous like infer_batch; wait with " + m + "_sync. Same return codes. */",
           f"int {m}_infer_one(const {ctype} *ROSENNA_RESTRICT x, {ctype} *ROSENNA_RESTRICT y, void *stream);",
           ""] if has_infer_one(plan) else []),
        "#if defined(__cplusplus)",
        "}",
        "#endif",
        "",
    ]
    if not plan.embed:
        lines += _emit_device_weight_table(plan, ctype)
    lines += _emit_device_region(plan, ctype)
    lines.append("#endif")
    return "\n".join(lines) + "\n"


def _emit_device_macros(plan: Plan) -> list:
    """The macros at the top of the header, and nothing else defines them.

    ROSENNA_CONST (controller ruling R4): the embedded weights go to
    __constant__ only while their total size stays under
    CONSTANT_MEMORY_LIMIT; a larger embedded model reads them from
    __device__ const global memory instead, which is both what the 64 KB
    per-module bank requires above 64 KB and, well below that, what the
    per-SM constant cache makes faster -- see CONSTANT_MEMORY_LIMIT for the
    measurements behind the 2 KB cut. Both are `static` so that each translation unit that includes
    the header -- <name>.c compiled as C++, <name>_kernel.cu, and any host
    .cu -- gets its own copy with internal linkage: a namespace-scope
    __constant__ definition with external linkage in a header is a duplicate
    symbol the moment two objects include it.
    """
    nbytes = _embedded_weight_bytes(plan)
    if plan.embed and nbytes < CONSTANT_MEMORY_LIMIT:
        const_qual = "static __constant__"
        const_note = (f"static __constant__: the {nbytes} bytes of embedded weights",
                      f"   fit the {CONSTANT_MEMORY_LIMIT}-byte constant-memory budget.")
    elif plan.embed:
        const_qual = "static __device__ const"
        const_note = (f"static __device__ const: the {nbytes} bytes of embedded weights",
                      f"   exceed the {CONSTANT_MEMORY_LIMIT}-byte constant-memory budget.")
    else:
        const_qual = "static __constant__"
        const_note = ("static __constant__ (unused: this model loads its weights",
                      "   from a file).")
    if plan.embed:
        stub_note = [
            "   ROSENNA_INFER_HOST_STUB is 1 in the host pass of a CUDA/HIP build: the",
            "   embedded weights are device storage there, so the host instantiation",
            "   of infer is a stub that asserts (a no-op under NDEBUG); call infer from",
            "   a kernel, or use infer_batch. Any other compiler computes on the host.",
        ]
    else:
        stub_note = []
    return [
        "/* Under nvcc/hipcc: ROSENNA_DEVICE_FN = __host__ __device__ (infer is",
        "   callable from device code), ROSENNA_RESTRICT = __restrict__, and",
        f"   ROSENNA_CONST = {const_note[0]}",
        const_note[1],
        "   Otherwise (plain C, or a host OpenMP/OpenACC build): ROSENNA_DEVICE_FN",
        "   is empty, ROSENNA_CONST = static const, and ROSENNA_RESTRICT is",
        "   __restrict__ in C++ or restrict in C. infer is separately wrapped in a",
        "   guarded OpenMP declare-target region with a guarded OpenACC routine-seq",
        "   pragma below; both are no-ops unless that compiler defines",
        "   _OPENMP/_OPENACC.",
        *stub_note,
        "   The macros are redefined (#undef first) so that headers of several",
        "   models can share one translation unit without redefinition warnings. */",
        "#undef ROSENNA_DEVICE_FN",
        "#undef ROSENNA_CONST",
        "#undef ROSENNA_RESTRICT",
        "#undef ROSENNA_INFER_HOST_STUB",
        "#undef ROSENNA_UNROLL",
        "/* A dense layer's dot-product loop, unrolled where the compiler takes the",
        "   hint: clang keeps the loop rolled otherwise and each weight load's",
        "   latency is exposed before its FMA (3x on a small kernel, MI210). nvcc",
        "   and nvc unroll on their own and get no hint. */",
        "#if defined(__clang__)",
        f'#define ROSENNA_UNROLL _Pragma("unroll {GEMM_BLOCK}")',
        "#elif defined(__GNUC__) && !defined(__NVCOMPILER) && !defined(__NVCC__)",
        f'#define ROSENNA_UNROLL _Pragma("GCC unroll {GEMM_BLOCK}")',
        "#else",
        "#define ROSENNA_UNROLL",
        "#endif",
        _CUDA_GUARD,
        "#define ROSENNA_DEVICE_FN __host__ __device__",
        f"#define ROSENNA_CONST {const_qual}",
        "#define ROSENNA_RESTRICT __restrict__",
        *(["#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)",
           "#define ROSENNA_INFER_HOST_STUB 0",
           "#else",
           "#define ROSENNA_INFER_HOST_STUB 1",
           "#endif"] if plan.embed else ["#define ROSENNA_INFER_HOST_STUB 0"]),
        "#else",
        "#define ROSENNA_DEVICE_FN",
        "#define ROSENNA_CONST static const",
        "#define ROSENNA_INFER_HOST_STUB 0",
        "#if defined(__cplusplus)",
        "#define ROSENNA_RESTRICT __restrict__",
        "#else",
        "#define ROSENNA_RESTRICT restrict",
        "#endif",
        "#endif",
        "",
    ]


def _emit_weight_declarations(plan: Plan, ctype: str) -> list:
    """A file-loaded plan's weights, inside the header's extern "C" block.

    The host arrays are declared under every compiler: <name>_init fills
    them from the file, and the host instantiation of infer reads them even
    when nvcc or hipcc is the compiler. The `<symbol>_dev` pointers exist
    only under nvcc/hipcc: <name>_init allocates and fills each one (status
    10 on failure) and <name>_infer_batch passes them to the kernel. Under
    a plain or OpenMP host build the guard is false and no device pointer
    is even declared.
    """
    m = plan.model
    if not plan.weights:
        return []
    lines = [
        "/* Host arrays filled by init from the weights file. Under an offloading",
        "   OpenMP or OpenACC build they also have device copies, which init",
        "   updates once the file is read (the plan step, ruling R5). */",
    ]
    lines += _omp_declare_target_begin()
    for w in plan.weights:
        sym = _c_weight_symbol(m, w.symbol)
        lines.append(f"extern {ctype} {sym}[{_weight_size(w.shape)}];")
    lines += _omp_declare_target_end()
    lines += _acc_declare(plan, "create")
    lines += [
        "",
        "/* Device copies of the arrays above: allocated and filled by",
        f"   {m}_init on a cuda/hip build, read by the batched kernel; not",
        "   referenced under a plain or OpenMP host build. */",
        _CUDA_GUARD,
    ]
    for w in plan.weights:
        sym = _c_weight_symbol(m, w.symbol)
        lines.append(f"extern {ctype} *{sym}_dev;")
    lines += [
        f"/* Called by {m}_init once the copies above exist: publishes them to",
        f"   the kernel's translation unit ({m}_kernel.cu, where it is defined).",
        "   Part of the plan step, not of the host API. */",
        f"int {_device_bind(m)}(void);",
        f"/* The cuda/hip half of init, defined in {m}_kernel.cu: allocates the",
        "   device copies of the weights and binds them. Called by",
        f"   {m}_init when {m}.c is built with -DROSENNA_NATIVE_KERNEL. */",
        f"int {m}_upload_device(void);",
        "#endif",
        "",
    ]
    return lines


def _omp_declare_target_begin() -> list:
    return ["#ifdef _OPENMP", "#pragma omp declare target", "#endif"]


def _omp_declare_target_end() -> list:
    return ["#ifdef _OPENMP", "#pragma omp end declare target", "#endif"]


def _weight_symbol_list(plan: Plan) -> str:
    return ", ".join(_c_weight_symbol(plan.model, w.symbol) for w in plan.weights)


def _acc_declare(plan: Plan, clause: str) -> list:
    """`#pragma acc declare <clause>(every weight)`, guarded by _OPENACC.

    gcc -fopenacc rejects a `routine seq` function that reads a file-scope
    array with no `declare` directive, so every weight array infer reads
    carries one: `create` for the file-loaded host arrays (init then does
    `update device`), `copyin` for the embedded constants.
    """
    if not plan.weights:
        return []
    return ["#ifdef _OPENACC", f"#pragma acc declare {clause}({_weight_symbol_list(plan)})", "#endif"]


def _device_bind(model: str) -> str:
    """The plan-step function in <name>_kernel.cu that binds the weight table."""
    return f"{model}_device_bind"


def _device_weight_table(model: str) -> str:
    """The per-translation-unit __constant__ table of device weight pointers."""
    return f"{model}_devw"


def _emit_device_weight_table(plan: Plan, ctype: str) -> list:
    """How infer reaches a file-loaded weight from device code.

    Device code cannot read a host global, and without relocatable device
    code a __device__ or __constant__ variable cannot be declared extern
    across translation units, so the device addresses the host holds in
    `<symbol>_dev` reach the kernel through this `static __constant__` table,
    one copy per translation unit, which <name>_device_bind (in the same
    translation unit as the kernel) fills once, at the end of init, through
    ROSENNA_MEMCPY_TO_SYMBOL. The ROSENNA_REF_<symbol> macros then
    select the table entry in the device pass and the host array otherwise,
    so infer's one body reads the right storage in each of its two
    instantiations without ever naming host storage from device code.
    """
    m = plan.model
    if not plan.weights:
        return []
    table = _device_weight_table(m)
    lines = [
        "/* Device-side view of the file-loaded weights: a per-translation-unit",
        f"   __constant__ table of the device pointers, filled once, after {m}_init,",
        f"   by {_device_bind(m)}_here below. Device code reads its own translation",
        "   unit's table; host code, under any compiler, reads the host arrays. */",
        _CUDA_GUARD,
        f"static __constant__ const {ctype} *{table}[{len(plan.weights)}];",
        "#endif",
        _DEVICE_PASS_GUARD,
    ]
    for k, w in enumerate(plan.weights):
        lines.append(f"#define {_c_weight_ref_macro(m, w.symbol)} {table}[{k}]")
    lines.append("#else")
    for w in plan.weights:
        lines.append(f"#define {_c_weight_ref_macro(m, w.symbol)} {_c_weight_symbol(m, w.symbol)}")
    lines += ["#endif", ""]
    nw = len(plan.weights)
    lines += [
        f"/* In a CUDA/HIP build, call {m}_device_bind_here() after EVERY call to {m}_init()",
        f"   in every translation unit whose kernels call {m}_infer. Embedded",
        "   models need nothing. Each translation unit holds its own copy of the",
        "   table above; this fills the including translation unit's copy from",
        f"   the device addresses init made (lib{m}.a does it for its own kernel).",
        "   Part of the plan step: it transfers, so never call it from a loop.",
        "   Returns 0, or 10 if init has not made the device copies. */",
        _CUDA_GUARD,
        f"static inline int {_device_bind(m)}_here(void) {{",
        f"    const {ctype} *table[{nw}] = {{",
    ]
    for w in plan.weights:
        lines.append(f"        {_c_weight_symbol(m, w.symbol)}_dev,")
    lines += [
        "    };",
        f"    for (int k = 0; k < {nw}; ++k) if (table[k] == 0) return 10;",
        f"    if (ROSENNA_MEMCPY_TO_SYMBOL({table}, table, sizeof table) != ROSENNA_OK) return 10;",
        "    return 0;",
        "}",
        "#endif",
        "",
    ]
    return lines


def _emit_device_region(plan: Plan, ctype: str) -> list:
    """The declare-target region: embedded weight consts (if any) plus infer.

    Every OpenMP pragma is guarded by `_OPENMP` and every OpenACC pragma by
    `_OPENACC` -- gcc -Wall warns `ignoring '#pragma acc ...'
    [-Wunknown-pragmas]` on an unguarded one under a compiler without
    OpenACC. Under nvcc/hipcc both guards are false and ROSENNA_DEVICE_FN
    supplies the CUDA decoration instead.
    """
    lines = ["#ifdef _OPENMP", "#pragma omp declare target", "#endif", ""]
    if plan.embed:
        lines += _emit_embedded_weights(plan, ctype)
    lines += ["#ifdef _OPENACC", "#pragma acc routine seq", "#endif"]
    lines += _emit_infer(plan, ctype)
    lines += _emit_elem_functions(plan, ctype)
    lines += ["#ifdef _OPENMP", "#pragma omp end declare target", "#endif", ""]
    return lines


def _emit_embedded_weights(plan: Plan, ctype: str) -> list:
    m = plan.model
    lines = []
    for w in plan.weights:
        sym = _c_weight_symbol(m, w.symbol)
        values = ", ".join(_format_embedded_value(v, plan.dtype) for v in w.values)
        lines.append(f"ROSENNA_CONST {ctype} {sym}[{_weight_size(w.shape)}] = {{ {values} }};")
    lines += _acc_declare(plan, "copyin")
    if plan.weights:
        lines.append("")
    return lines


def _emit_source_head(plan: Plan, ctype: str) -> list:
    m = plan.model
    lines = [
        "/* Generated by rosenna. Do not edit. */",
        f'#include "{m}.h"',
        "",
        "#include <stddef.h>",
    ]
    if not plan.embed:
        lines += [
            "#include <stdint.h>",
            "#include <stdio.h>",
            "#include <string.h>",
        ]
    lines += [
        "",
        "/* nvcc and hipcc compile this file as C++ (-x cu / -x hip, see the",
        "   recipe); the definitions below then carry C linkage to match the",
        "   header's declarations, so C and Fortran hosts link unchanged. */",
        "#if defined(__cplusplus)",
        'extern "C" {',
        "#endif",
        "",
    ]
    if plan.embed:
        return lines
    hash_bytes = ", ".join(f"0x{b:02x}" for b in bytes.fromhex(plan.hash()))
    lines += [
        f"static const unsigned char expected_hash[32] = {{ {hash_bytes} }};",
        "",
    ]
    # The header's `acc declare create` on the extern declarations already
    # covers these definitions (gcc rejects a second declare for the same
    # variable); the OpenMP declare-target region is repeated, which is
    # allowed and keeps the definition self-describing.
    lines += _omp_declare_target_begin()
    for w in plan.weights:
        lines.append(f"{ctype} {_c_weight_symbol(m, w.symbol)}[{_weight_size(w.shape)}];")
    lines += _omp_declare_target_end()
    # The <symbol>_dev pointers are DEFINED in <name>_kernel.cu, beside the
    # runtime calls that fill them; the header declares them extern so any
    # translation unit's device_bind_here can read them. They used to be
    # defined here, which is why this file needed nvcc.
    lines.append("")
    return lines


def _emit_upload(plan: Plan) -> list:
    """Nothing: the cuda/hip half of init lives in <name>_kernel.cu now.

    Every line of it is a CUDA/HIP runtime call and rosenna_rt.h is #error for
    any other compiler, so keeping it here is what forced the recipe to build
    <name>.c with nvcc -- which silently turned off the declare-target on the
    weights this same file defines. <name>.c now only calls
    <name>_upload_device(); see emit_kernel._emit_upload_device.
    """
    return []


def _emit_fallback_infer_batch(plan: Plan, ctype: str) -> list:
    """The OpenMP-target infer_batch: the omp backend, over device pointers.

    Controller ruling R5: x and y are already on the device (is_device_ptr
    / deviceptr), so the loop path maps, allocates and synchronizes nothing.
    Compiled only when the compiler is not nvcc or hipcc; the cuda/hip
    backends define the same function in <name>_kernel.cu instead. Under a
    host compiler without -fopenmp/-fopenacc the pragmas are inert and this
    is a plain loop over host pointers, which is also what use_device_ptr
    yields on a host-only build. The loop is `teams distribute parallel for`,
    one point per thread: `teams loop` maps one point per TEAM under nvc (the
    ~30x cliff python/README.md describes) and under amdclang (3.5 us per
    point on an MI210, seen on examples/surrogates/B), and the bias
    reordering that lets nvc compile the per-point harnesses' distribute
    parallel for applies to this loop too.
    """
    m = plan.model
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    return [
        _NOT_NATIVE_GUARD,
        f"int {m}_infer_batch(int n, const {ctype} *ROSENNA_RESTRICT x, {ctype} *ROSENNA_RESTRICT y, void *stream) {{",
        "    (void)stream;",
        "    if (n <= 0) return 0;",
        *([f"    /* {m}_infer's locals exceed a device thread's stack: one sample at a time,",
           "       each a launch per op over the static buffers. */",
           "    int status = 0;",
           f"    for (int p = 0; p < n && status == 0; ++p)",
           f"        status = {m}_infer_one(x + (size_t)p * {n_in}, y + (size_t)p * {n_out}, stream);",
           "    return status;",
           "}"] if large_locals(plan) else [
           "#if defined(_OPENMP)",
           "#pragma omp target teams distribute parallel for is_device_ptr(x, y)",
           "#elif defined(_OPENACC)",
           "#pragma acc parallel loop deviceptr(x, y)",
           "#endif",
           f"    for (int p = 0; p < n; ++p) {m}_infer(x + (size_t)p * {n_in}, y + (size_t)p * {n_out});",
           "    return 0;",
           "}"]),
        "",
        "/* The loop above is synchronous, so there is nothing to wait for. */",
        f"int {m}_sync(void *stream) {{",
        "    (void)stream;",
        "    return 0;",
        "}",
        *_emit_fallback_infer_one(plan, ctype),
        "#endif",
        "",
    ]


def _emit_fallback_infer_one(plan: Plan, ctype: str) -> list:
    """infer_one for the omp backend: one target loop per op over static declare-target buffers."""
    if not has_infer_one(plan):
        return []
    m = plan.model
    lines = ["", "/* infer_one's activations: device-resident globals, not a thread's locals. */"]
    lines += _omp_declare_target_begin()
    lines += [f"static {ctype} {field_buffer(m, sym)}[{plan.buffers[sym]}];" for sym in scratch_symbols(plan)]
    lines += _omp_declare_target_end()
    lines += [f"int {m}_infer_one(const {ctype} *ROSENNA_RESTRICT x, {ctype} *ROSENNA_RESTRICT y, void *stream) {{",
              "    (void)stream;"]
    for k, op in enumerate(plan.ops):
        if op.kind == "alias":
            continue
        lines += ["#if defined(_OPENMP)",
                  "#pragma omp target teams distribute parallel for is_device_ptr(x, y)",
                  "#elif defined(_OPENACC)",
                  "#pragma acc parallel loop deviceptr(x, y)",
                  "#endif",
                  f"    for (int e = 0; e < {elem_length(op)}; ++e) {elem_call(plan, k, op)};"]
    lines += ["    return 0;", "}"]
    return lines


def _emit_load(plan: Plan) -> list:
    if not plan.weights:
        return []
    m = plan.model
    lines = [
        "/* seen[k] is set when plan weight k has been filled: a table of contents",
        "   that names a tensor twice, or not at all, is status 9 (inconsistent),",
        "   never a zero array that infer then runs on. */",
        "static int load_tensor(FILE *f, const char *name, int32_t namelen, long pos,",
        "                       int64_t length, unsigned char *seen) {",
    ]
    for k, w in enumerate(plan.weights):
        c_sym = _c_weight_symbol(m, w.symbol)
        # `length` comes from the file's own table of contents; a tensor whose
        # declared byte count disagrees with the array it is about to fill
        # means a corrupt file, so reject it rather than short-read into the
        # array and infer on half-loaded weights.
        lines.append(
            f"    if (namelen == {len(w.name)} && "
            f"memcmp(name, {_c_string(w.name)}, {len(w.name)}) == 0) {{")
        lines.append(f"        if (seen[{k}]) return 9;")
        lines.append(f"        seen[{k}] = 1;")
        lines.append(f"        if (length != (int64_t)sizeof {c_sym}) return 9;")
        lines.append("        if (fseek(f, pos, SEEK_SET) != 0) return 9;")
        lines.append(f"        if (fread({c_sym}, sizeof {c_sym}, 1, f) != 1) return 9;")
        lines.append("        return 0;")
        lines.append("    }")
    lines += ["    return 7;", "}", ""]
    return lines


def _emit_init(plan: Plan) -> list:
    m = plan.model
    dtype_code = _DTYPE_CODE[plan.dtype]
    lines = ["/*"] + [" " + line for line in status_code_comment("*", m)] + [" */"]
    lines += [
        f"int {m}_init(const char *path) {{",
        '    FILE *f = fopen(path, "rb");',
        "    if (!f) return 1;",
        "    unsigned char magic[8];",
        "    if (fread(magic, 1, 8, f) != 8) { fclose(f); return 9; }",
        '    if (memcmp(magic, "ROSENNA1", 8) != 0) { fclose(f); return 2; }',
        "    int32_t version, dtype, endian, ntensors;",
        "    if (fread(&version, sizeof version, 1, f) != 1) { fclose(f); return 9; }",
        "    if (fread(&dtype, sizeof dtype, 1, f) != 1) { fclose(f); return 9; }",
        "    if (fread(&endian, sizeof endian, 1, f) != 1) { fclose(f); return 9; }",
        "    if (fread(&ntensors, sizeof ntensors, 1, f) != 1) { fclose(f); return 9; }",
        "    if (version != 1) { fclose(f); return 3; }",
        f"    if (dtype != {dtype_code}) {{ fclose(f); return 4; }}",
        "    if (endian != 0x01020304) { fclose(f); return 5; }",
        "    unsigned char file_hash[32];",
        "    if (fread(file_hash, 1, 32, f) != 32) { fclose(f); return 9; }",
        "    if (memcmp(file_hash, expected_hash, 32) != 0) { fclose(f); return 6; }",
    ]
    if not plan.weights:
        # This model has no weights, so a well-formed file for it holds no
        # tensors. Skipping the table of contents keeps the generated source
        # free of an unreachable loop and of the unused parameters a
        # weight-free load_tensor would carry.
        lines += [
            "    if (ntensors != 0) { fclose(f); return 7; }",
            "    fclose(f);",
            "    return 0;",
            "}",
            "",
        ]
        return lines
    name_cap, rank_cap = name_capacity(plan), rank_capacity(plan)
    lines += [
        "    int32_t toclen;",
        "    if (fread(&toclen, sizeof toclen, 1, f) != 1) { fclose(f); return 9; }",
        "    long data_start = 60L + (long)toclen;",
        f"    unsigned char seen[{len(plan.weights)}] = {{0}};",
        "    for (int32_t k = 0; k < ntensors; ++k) {",
        "        int32_t namelen;",
        "        if (fread(&namelen, sizeof namelen, 1, f) != 1) { fclose(f); return 9; }",
        f"        if (namelen < 0 || namelen > {name_cap}) {{ fclose(f); return 8; }}",
        f"        char name[{name_cap}] = {{0}};",
        "        if (fread(name, 1, (size_t)namelen, f) != (size_t)namelen)",
        "            { fclose(f); return 9; }",
        "        int32_t rank;",
        "        if (fread(&rank, sizeof rank, 1, f) != 1) { fclose(f); return 9; }",
        f"        if (rank < 0 || rank > {rank_cap}) {{ fclose(f); return 8; }}",
        f"        int64_t dims[{rank_cap}];",
        "        if (fread(dims, sizeof(int64_t), (size_t)rank, f) != (size_t)rank)",
        "            { fclose(f); return 9; }",
        "        (void)dims;",
        "        int64_t off, length;",
        "        if (fread(&off, sizeof off, 1, f) != 1) { fclose(f); return 9; }",
        "        if (fread(&length, sizeof length, 1, f) != 1) { fclose(f); return 9; }",
        "        long tocpos = ftell(f);",
        "        int status = load_tensor(f, name, namelen, data_start + (long)off, length, seen);",
        "        if (status != 0) { fclose(f); return status; }",
        "        if (fseek(f, tocpos, SEEK_SET) != 0) { fclose(f); return 9; }",
        "    }",
        "    fclose(f);",
        f"    for (int k = 0; k < {len(plan.weights)}; ++k) if (!seen[k]) return 9;",
        "    /* The plan step's transfer: the device copies of the arrays, under",
        "       whichever offload family this build has. */",
        "#ifdef _OPENMP",
        f"#pragma omp target update to({_weight_symbol_list(plan)})",
        "#endif",
        "#ifdef _OPENACC",
        f"#pragma acc update device({_weight_symbol_list(plan)})",
        "#endif",
        _NATIVE_GUARD,
        f"    return {m}_upload_device();",
        "#else",
        "    return 0;",
        "#endif",
        "}",
        "",
    ]
    return lines


def _nest(loops, body, indent="    "):
    """Wrap body lines in for-loops over `loops`, a list of (counter, extent)."""
    L = []
    for k, (nm, ext) in enumerate(loops):
        tail = " {" if k == len(loops) - 1 else ""
        L.append(f"{indent}for (int {nm} = 0; {nm} < {ext}; ++{nm}){tail}")
    L += [indent + "    " + b for b in body]
    L.append(indent + "}")
    return L


def _decode(loops, body, indent="    "):
    """Recover the counters of `loops` from the flat row-major element index e, then the body."""
    L = [f"{indent}int rem = e;"]
    for nm, ext in reversed(loops):
        L.append(f"{indent}const int {nm} = rem % {ext}; rem /= {ext};")
    L.append(f"{indent}(void)rem;")
    L += [indent + b for b in body]
    return L


def _flat_index(names, shape, base=""):
    """Row-major flat index from per-axis counters, as a C/Fortran expression."""
    expr = names[0]
    for k in range(1, len(shape)):
        expr = f"({expr} * {shape[k]} + {names[k]})"
    return expr + base


def _emit_transpose_c(op, dst, src):
    """A real axis permutation: one loop per output axis, gathering from the source.

    Only reached when the permutation moves an axis with extent > 1 -- plan.py
    turns the rest into buffer aliases, since those move no bytes.
    """
    names = [f"c{k}" for k in range(len(op.out_shape))]
    terms = [nm if st == 1 else f"{nm} * {st}" for nm, st in zip(names, op.perm_strides) if st]
    body = [f"{dst}[{_flat_index(names, op.out_shape)}] = {src}[{' + '.join(terms) if terms else '0'}];"]
    return list(zip(names, op.out_shape)), body


def _emit_lstm_c(op, ctype, act, dst, src, h0, c0, wsym, rsym, bsym, outs, zero):
    """One forward LSTM, ONNX default activations, as a plain sequential loop.

    Gate order in W/R/B is ONNX's i, o, f, c -- not the i, f, c, o most
    references use -- so the four blocks are read at 0H, 1H, 2H, 3H in that
    order. B holds Wb and Rb back to back, both of which are added.
    """
    sp = op.lstm
    H, I, B, T = sp.hidden, sp.input_size, sp.batch, sp.seq
    h, c, g = sp.h_sym, sp.c_sym, sp.g_sym
    L = [f"    for (int i = 0; i < {B * H}; ++i) {h}[i] = {h0 + '[i]' if h0 else zero};",
         f"    for (int i = 0; i < {B * H}; ++i) {c}[i] = {c0 + '[i]' if c0 else zero};",
         f"    for (int t = 0; t < {T}; ++t)",
         f"    for (int b = 0; b < {B}; ++b) {{",
         f"        for (int k = 0; k < {4 * H}; ++k) {{",
         f"            {ctype} acc = {zero};",
         f"            for (int j = 0; j < {I}; ++j) "
         f"acc += {src}[(t * {B} + b) * {I} + j] * {wsym}[k * {I} + j];",
         f"            for (int j = 0; j < {H}; ++j) "
         f"acc += {h}[b * {H} + j] * {rsym}[k * {H} + j];"]
    if bsym:
        L.append(f"            acc += {bsym}[k] + {bsym}[{4 * H} + k];")
    L += [f"            {g}[k] = acc;",
          "        }",
          f"        for (int j = 0; j < {H}; ++j) {{",
          f"            const {ctype} gi = {act['sigmoid'].format(v=f'{g}[j]')};",
          f"            const {ctype} go = {act['sigmoid'].format(v=f'{g}[{H} + j]')};",
          f"            const {ctype} gf = {act['sigmoid'].format(v=f'{g}[{2 * H} + j]')};",
          f"            const {ctype} gc = {act['tanh'].format(v=f'{g}[{3 * H} + j]')};",
          f"            const {ctype} cn = gf * {c}[b * {H} + j] + gi * gc;",
          f"            {c}[b * {H} + j] = cn;",
          f"            {h}[b * {H} + j] = go * {act['tanh'].format(v='cn')};",
          *([f"            {dst}[(t * {B} + b) * {H} + j] = {h}[b * {H} + j];"]
            if sp.emit_y else []),
          "        }",
          "    }"]
    # outs is positional: [0] is Y_h and [1] is Y_c, "" for one nothing reads.
    for sym, state in zip(outs, (h, c)):
        if sym:
            L.append(f"    for (int i = 0; i < {B * H}; ++i) {sym}[i] = {state}[i];")
    return L


def _concat_sources(op, runtime_names, const_names) -> list:
    """The operands in ONNX order, each as (array name, block length)."""
    rt, ct, out = iter(runtime_names), iter(const_names), []
    for is_const, block in zip(op.concat.consts, op.concat.blocks):
        out.append((next(ct) if is_const else next(rt), block))
    return out


def _emit_concat_c(op, dst, runtime_names, const_names):
    """Concat along an axis: output element (o, i) comes from the operand whose block holds i."""
    cc = op.concat
    srcs = _concat_sources(op, runtime_names, const_names)
    row = sum(cc.blocks)
    body, off = [], 0
    for k, (name, block) in enumerate(srcs):
        cond = f"{'if' if k == 0 else 'else if'} (i < {off + block})" if k < len(srcs) - 1 else "else"
        body.append(f"{cond} {dst}[o * {row} + i] = {name}[o * {block} + i - {off}];")
        off += block
    return [("o", cc.outer), ("i", row)], body


def _emit_add_c(op, dst, src, wsym):
    """Elementwise add of a broadcast constant: one loop per output axis.

    Iterating the axes rather than the flat extent is what makes the constant's
    index affine -- `c1 * stride1 + ...` with the broadcast axes contributing
    nothing -- instead of a decomposition with divisions inside the loop.
    """
    bc = op.bcast
    names = [f"c{k}" for k in range(len(bc.out_shape))]
    flat = _flat_index(names, bc.out_shape)
    terms = [nm if st == 1 else f"{nm} * {st}" for nm, st in zip(names, bc.strides) if st]
    widx = " + ".join(terms) if terms else "0"
    return list(zip(names, bc.out_shape)), [f"{dst}[{flat}] = {src}[{flat}] + {wsym}[{widx}];"]


def _literal_c(value: float, ctype: str) -> str:
    """The pad value as a C literal, at the generated code's precision."""
    return f"{value!r}f" if ctype == "float" else repr(float(value))


def _emit_pad_c(op, ctype: str, dst: str, src: str):
    """Constant Pad: loop the output, read the input where the shift is in range.

    Only axes that are actually padded get a bounds test -- on an unpadded axis
    the output index IS the input index, so a test there would always pass and
    would only make the generated nest harder to read.
    """
    pd = op.pad
    names = [f"c{k}" for k in range(len(pd.out_shape))]
    shifted, checks, pre = [], [], []
    for k, (nm, b) in enumerate(zip(names, pd.begins)):
        if b == 0 and pd.in_shape[k] == pd.out_shape[k]:
            shifted.append(nm)
            continue
        # A negative begin is a crop: the output reads FURTHER into the input,
        # so it is spelled as an addition. `(c - -1)` is legal C and a syntax
        # error in Fortran, which is what made this worth spelling out rather
        # than letting the sign fall out of the arithmetic.
        expr = f"({nm} - {b})" if b > 0 else f"({nm} + {-b})" if b < 0 else nm
        n_in = pd.in_shape[k]
        if pd.mode in ("edge", "reflect"):
            # Into a named local, not inlined: the expression appears three
            # times in the ternary, and four of them in one subscript ran past
            # Fortran's 132-column limit in the twin emitter. One name per
            # padded axis keeps both readable and evaluates it once.
            m = (f"{expr} < 0 ? 0 : ({expr} >= {n_in} ? {n_in - 1} : {expr})"
                 if pd.mode == "edge" else
                 f"{expr} < 0 ? -{expr} : ({expr} >= {n_in} ? {2 * (n_in - 1)} - {expr} : {expr})")
            pre.append(f"const int pi{k} = {m};")
            shifted.append(f"pi{k}")
        else:
            shifted.append(expr)
            # Only a positive begin can put the read before the input; a crop
            # cannot, so that half of the test would always pass.
            if b > 0:
                checks.append(f"{expr} >= 0")
            checks.append(f"{expr} < {n_in}")
    out_idx = _flat_index(names, pd.out_shape)
    in_idx = _flat_index(shifted, pd.in_shape)
    val = _literal_c(pd.value, ctype)
    if not checks:
        # edge and reflect always land on a real element, so there is no test
        # and no pad value -- the index map is the whole of the operator.
        return list(zip(names, pd.out_shape)), pre + [f"{dst}[{out_idx}] = {src}[{in_idx}];"]
    return list(zip(names, pd.out_shape)), pre + [
        f"{dst}[{out_idx}] = ({' && '.join(checks)}) ? {src}[{in_idx}] : {val};"]


def _emit_softmax_c(op, ctype: str, dtype: str, dst: str, src: str, zero: str):
    """Last-axis Softmax: max, then exp into the destination, then normalise.

    Subtracting the row maximum before exponentiating is what keeps a logit of
    +800 from overflowing to inf; it cancels exactly in the ratio, so it costs
    only the extra pass.

    The maximum uses `v > mx`, which the NaN rule in doc/adding-an-operator.md tells
    you not to write -- deliberately, and this is the one op where it is
    right. A NaN must LOSE the maximum here: a NaN-sticky maximum would make
    every exponent NaN - NaN, whereas letting the NaN lose keeps `mx` a real
    number, so `exp(NaN - mx)` is NaN, the row sum is NaN, and every output in
    that row is NaN. The NaN propagates through the sum instead of through the
    maximum, and no finite input can overflow on the way.
    """
    sm = op.softmax
    expf = "expf" if dtype == "f32" else "exp"
    c = sm.axis_len
    at = f"n * {c}"
    return [("n", sm.outer)], [
        f"{ctype} mx = {src}[{at}];",
        f"for (int j = 1; j < {c}; ++j) {{",
        f"    const {ctype} v = {src}[{at} + j];",
        "    if (v > mx) mx = v;",
        "}",
        f"{ctype} s = {zero};",
        f"for (int j = 0; j < {c}; ++j) {{",
        f"    const {ctype} e = {expf}({src}[{at} + j] - mx);",
        f"    {dst}[{at} + j] = e;",
        "    s += e;",
        "}",
        f"for (int j = 0; j < {c}; ++j) {dst}[{at} + j] = {dst}[{at} + j] / s;",
    ]


def _emit_spatial_c(op, ctype, dst, src, weight_sym, bias_sym, zero):
    """A 2-D Conv / MaxPool / AveragePool as an explicit loop nest over flat buffers.

    Buffers stay rank 1 whatever the value's logical rank: NCHW is flattened
    row-major and the index arithmetic is written out, which keeps one buffer
    model for dense and spatial ops alike and keeps every bound a literal.

    The `continue` on an out-of-range (ih, iw) is what implements padding:
    nothing is materialised, a pad cell simply contributes nothing. That is
    exactly right for Conv (pad = 0 contributes 0) and for MaxPool (ONNX pads
    with -inf, i.e. a pad cell never wins); AveragePool needs to know how many
    cells were real, which is what `cnt` counts.
    """
    sp = op.spatial
    # AveragePool divides by the count of in-bounds cells only when a window
    # can actually reach outside the input and the caller did not ask for the
    # full-kernel divisor; otherwise the divisor is a literal.
    needs_count = (op.kind == "avgpool"
                   and not (sp.every_window_is_inside or sp.count_include_pad))
    L = []
    # Grouped Conv: output channel oc belongs to group oc/c_out_per_group and
    # reads only that group's c_in_per_group input channels, so the loop bound
    # is the per-group count and the input channel is offset by the group. For
    # group=1 c_in_per_group == c_in and every expression below collapses to
    # exactly what it was, so an ordinary convolution emits identical code.
    cpg = sp.c_in_per_group or sp.c_in
    # The group's first input channel, hoisted: it depends only on oc, so
    # recomputing it per element would put an integer division in the
    # innermost index expression (and gfortran warns about the division under
    # -Winteger-division there, which is a fair complaint about the shape of
    # the code rather than a false positive).
    in_c = "(icg + ic)" if sp.grouped else "ic"
    idx_in = f"((n * {sp.c_in} + {in_c}) * {sp.h_in} + ih) * {sp.w_in} + iw"
    idx_out = f"((n * {sp.c_out} + oc) * {sp.h_out} + oh) * {sp.w_out} + ow"
    loops = [("n", sp.n), ("oc", sp.c_out), ("oh", sp.h_out), ("ow", sp.w_out)]

    if op.kind == "conv":
        L.append(f"{ctype} acc = {zero};")
        if sp.grouped:
            L.append(f"const int icg = oc / {sp.c_out_per_group} * {cpg};")
        L.append(f"for (int ic = 0; ic < {cpg}; ++ic)")
    elif op.kind == "maxpool":
        # The first in-range cell seeds the running maximum; `seen` makes that
        # independent of any sentinel value, so a window of all -inf inputs
        # still yields -inf rather than a made-up number.
        L.append(f"{ctype} best = {zero};")
        L.append("int seen = 0;")
        L.append("const int ic = oc;")
    else:
        L.append(f"{ctype} acc = {zero};")
        # Only when the divisor is the count of cells that fell inside. With a
        # constant divisor nothing reads it, and a counter that is incremented
        # and never read is a warning clang reports and gcc does not.
        if needs_count:
            L.append("int cnt = 0;")
        L.append("const int ic = oc;")

    L.append(f"for (int kh = 0; kh < {sp.kh}; ++kh)")
    L.append(f"for (int kw = 0; kw < {sp.kw}; ++kw) {{")
    L.append(f"    const int ih = oh * {sp.sh} - {sp.ph} + kh * {sp.dh};")
    L.append(f"    const int iw = ow * {sp.sw} - {sp.pw} + kw * {sp.dw};")
    L.append(f"    if (ih < 0 || ih >= {sp.h_in} || iw < 0 || iw >= {sp.w_in}) continue;")
    if op.kind == "conv":
        widx = f"((oc * {cpg} + ic) * {sp.kh} + kh) * {sp.kw} + kw"
        L.append(f"    acc += {src}[{idx_in}] * {weight_sym}[{widx}];")
        L.append("}")
    elif op.kind == "maxpool":
        L.append(f"    const {ctype} v = {src}[{idx_in}];")
        # !(v <= best), not (v > best): a NaN loses every comparison, so the
        # naive form drops it. This library is linked into solvers where a NaN
        # out of a diverged run is the signal, so it has to survive a pool.
        L.append("    if (!seen || !(v <= best)) { best = v; seen = 1; }")
        L.append("}")
    else:
        L.append(f"    acc += {src}[{idx_in}];")
        if needs_count:
            L.append("    ++cnt;")
        L.append("}")

    if op.kind == "conv":
        if bias_sym:
            L.append(f"acc += {bias_sym}[oc];")
        L.append(f"{dst}[{idx_out}] = acc;")
    elif op.kind == "maxpool":
        L.append(f"{dst}[{idx_out}] = best;")
    else:
        full = sp.kh * sp.kw
        if not needs_count:
            # No pad cell can fall in a window, or the caller asked for the
            # full-kernel divisor: a literal either way.
            L.append(f"{dst}[{idx_out}] = acc / ({ctype}){full};")
        else:
            L.append(f"{dst}[{idx_out}] = cnt ? acc / ({ctype})cnt : {zero};")
    return loops, L


def _emit_gemm_c(op, ctype, dst, src, weight_sym, idx_expr, bias_sym, zero):
    """A dense layer: output (r, i) is the dot product of input row r with weight column i.

    The bias is added AFTER the dot product, not used to seed the
    accumulator. Seeding it from a declare-target array is what makes nvc
    refuse to generate a `distribute parallel for` body (it emits a kernel
    that traps); adding it afterwards compiles, and unlocks a ~19x faster
    per-point offload loop. See doc/nvhpc_teams_mapping/. emit_fortran
    does the same, so the two backends stay bit-comparable. `r` indexes
    independent rows sharing one weight (1 for a dense per-point model); it
    is only emitted when there is more than one.
    """
    ri, ro = (f"r * {op.n_in} + ", f"r * {op.n_out} + ") if op.rows > 1 else ("", "")
    loops = ([("r", op.rows)] if op.rows > 1 else []) + [("i", op.n_out)]
    body = [f"{ctype} acc = {zero};",
            f"for (int j = 0; j < {op.n_in}; ++j) acc += {src}[{ri}j] * {weight_sym}[{idx_expr}];"]
    if bias_sym:
        body.append(f"acc += {bias_sym}[i];")
    body.append(f"{dst}[{ro}i] = acc;")
    return loops, body


def _emit_gemm_blocked_c(op, ctype, dst, src, weight_sym, idx_expr, bias_sym, zero, block=None):
    """The nested-loop form of a dense layer, `block` output columns per pass over the input."""
    block = block or gemm_block(op)
    ri, ro = (f"r * {op.n_in} + ", f"r * {op.n_out} + ") if op.rows > 1 else ("", "")
    col = lambda k: re.sub(r"\bi\b", f"(i + {k})", idx_expr)
    L = []
    if op.rows > 1:
        L.append(f"    for (int r = 0; r < {op.rows}; ++r) {{")
    ind = "        " if op.rows > 1 else "    "
    full = (op.n_out // block) * block if block > 1 else 0
    if full:
        L += [f"{ind}for (int i = 0; i < {full}; i += {block}) {{",
              f"{ind}    {ctype} " + ", ".join(f"a{k} = {zero}" for k in range(block)) + ";",
              f"{ind}    ROSENNA_UNROLL",
              f"{ind}    for (int j = 0; j < {op.n_in}; ++j) {{",
              f"{ind}        const {ctype} sj = {src}[{ri}j];",
              *[f"{ind}        a{k} += sj * {weight_sym}[{col(k)}];" for k in range(block)],
              f"{ind}    }}",
              *[f"{ind}    {dst}[{ro}i + {k}] = a{k}" + (f" + {bias_sym}[i + {k}];" if bias_sym else ";")
                for k in range(block)],
              f"{ind}}}"]
    if full < op.n_out:
        L += [f"{ind}for (int i = {full}; i < {op.n_out}; ++i) {{",
              f"{ind}    {ctype} acc = {zero};",
              f"{ind}    for (int j = 0; j < {op.n_in}; ++j) acc += {src}[{ri}j] * {weight_sym}[{idx_expr}];",
              f"{ind}    {dst}[{ro}i] = acc" + (f" + {bias_sym}[i];" if bias_sym else ";"),
              f"{ind}}}"]
    if op.rows > 1:
        L.append("    }")
    return L


def _op_pieces(plan: Plan, ctype: str, op, dst, src, extra_srcs=None):
    """(loops, body) for one op, or None for an alias / an LSTM (which has no per-element form).

    `dst`/`src` are the array names the body reads and writes; `extra_srcs`
    names a Concat's further runtime operands.
    """
    m = plan.model
    act = _ACT_C[plan.dtype]
    zero = _ZERO[plan.dtype]
    weight_by_symbol = {w.symbol: w for w in plan.weights}
    if op.kind == "gemm":
        return _emit_gemm_c(op, ctype, dst, src, _weight_ref(plan, m, op.weight),
                            _weight_index_c(weight_by_symbol, op),
                            _weight_ref(plan, m, op.bias) if op.bias else None, zero)
    if op.kind == "transpose":
        return _emit_transpose_c(op, dst, src)
    if op.kind == "add":
        return _emit_add_c(op, dst, src, _weight_ref(plan, m, op.weight))
    if op.kind == "concat":
        return _emit_concat_c(op, dst, [src] + list(extra_srcs or []),
                              [_weight_ref(plan, m, sym) for sym in op.concat_syms])
    if op.kind == "copy":
        soff = f"{op.src_offset} + " if op.src_offset else ""
        doff = f"{op.dst_offset} + " if op.dst_offset else ""
        return [("i", op.n_out)], [f"{dst}[{doff}i] = {src}[{soff}i];"]
    if op.kind == "pad":
        return _emit_pad_c(op, ctype, dst, src)
    if op.kind == "softmax":
        return _emit_softmax_c(op, ctype, plan.dtype, dst, src, zero)
    if op.kind in ("conv", "maxpool", "avgpool"):
        return _emit_spatial_c(op, ctype, dst, src,
                               _weight_ref(plan, m, op.weight) if op.weight else None,
                               _weight_ref(plan, m, op.bias) if op.bias else None, zero)
    if op.kind in act:
        return [("i", op.n_out)], [f"{dst}[i] = {act[op.kind].format(v=f'{src}[i]')};"]
    if op.kind in ("alias", "lstm"):
        return None
    raise AssertionError(f"unhandled op kind {op.kind!r}")


def _emit_op_sequence(plan: Plan, ctype: str) -> list:
    """The op sequence over plan.assignment's buffers, as infer runs it."""
    m = plan.model
    act = _ACT_C[plan.dtype]
    lines = []
    for op in plan.ops:
        if op.kind == "alias":
            continue
        dst, src = plan.assignment.get(op.out), plan.assignment[op.inp]
        if op.kind == "lstm":
            h0, c0 = lstm_initial_state(op, lambda sym: _weight_ref(plan, m, sym), plan.assignment)
            lines += _emit_lstm_c(
                op, ctype, act, dst, src, h0, c0,
                _weight_ref(plan, m, op.weight), _weight_ref(plan, m, op.weight2),
                _weight_ref(plan, m, op.bias) if op.bias else None,
                [plan.assignment[o] if o else "" for o in op.outs], _ZERO[plan.dtype])
            continue
        if op.kind == "gemm":
            weight_by_symbol = {w.symbol: w for w in plan.weights}
            lines += _emit_gemm_blocked_c(op, ctype, dst, src, _weight_ref(plan, m, op.weight),
                                          _weight_index_c(weight_by_symbol, op),
                                          _weight_ref(plan, m, op.bias) if op.bias else None,
                                          _ZERO[plan.dtype])
            continue
        loops, body = _op_pieces(plan, ctype, op, dst, src, [plan.assignment[n] for n in op.extra_in])
        lines += _nest(loops, body)
    return lines


def has_infer_one(plan: Plan) -> bool:
    """infer_one runs one op per launch; an LSTM is a sequence, so a plan with one has no infer_one."""
    return not any(op.kind == "lstm" for op in plan.ops)


def _elem_fn(model: str, k: int) -> str:
    return f"{model}_op{k}"


def _elem_params(op, ctype: str) -> str:
    extra = "".join(f", const {ctype} *ROSENNA_RESTRICT s{j + 1}" for j in range(len(op.extra_in)))
    return f"int e, const {ctype} *ROSENNA_RESTRICT src, {ctype} *ROSENNA_RESTRICT dst{extra}"


def _emit_elem_functions(plan: Plan, ctype: str) -> list:
    """One device-decorated function per op computing output element e: what infer_one launches over."""
    if not has_infer_one(plan):
        return []
    m = plan.model
    lines = [f"/* {m}_infer_one runs one op per launch, the thread index over the op's output",
             "   elements, through these; the same op bodies infer runs in nested loops. */"]
    for k, op in enumerate(plan.ops):
        if op.kind == "alias":
            continue
        pieces = _op_pieces(plan, ctype, op, "dst", "src", [f"s{j + 1}" for j in range(len(op.extra_in))])
        lines.append(f"static inline ROSENNA_DEVICE_FN void {_elem_fn(m, k)}({_elem_params(op, ctype)}) {{")
        if plan.embed:
            lines += ["#if ROSENNA_INFER_HOST_STUB", "    (void)e; (void)src; (void)dst;",
                      *[f"    (void)s{j + 1};" for j in range(len(op.extra_in))],
                      f'    assert(0 && "rosenna: {m}_infer_one is device-only in a CUDA/HIP build");',
                      "#else"]
        lines += _decode(*pieces)
        if plan.embed:
            lines.append("#endif")
        lines += ["}", ""]
    return lines


def elem_length(op) -> int:
    """How many output elements op has: the launch/loop extent of its per-element function."""
    return op.n_out * op.rows if op.kind == "gemm" else op.n_out


def elem_call(plan: Plan, k: int, op) -> str:
    """`<name>_op<k>(e, src, dst, ...)` over infer_one's buffers (x and y are its arguments)."""
    m = plan.model
    args = [field_buffer(m, plan.assignment[op.inp]), field_buffer(m, plan.assignment[op.out])]
    args += [field_buffer(m, plan.assignment[n]) for n in op.extra_in]
    return f"{_elem_fn(m, k)}(e, {', '.join(args)})"


def scratch_symbols(plan: Plan) -> list:
    return sorted((s for s in plan.buffers if s not in ("x", "y")), key=lambda s: int(s[1:]))


# Bytes of per-point locals above which infer must not be instantiated as a
# device thread's body: a thread's stack is 128 KB on AMD (hipcc refuses the
# kernel) and less on NVIDIA. A whole-field model (a conv net over a grid)
# is the case; its infer_batch runs infer_one per point instead.
LOCALS_LIMIT = 64 * 1024


def large_locals(plan: Plan) -> bool:
    return sum(plan.buffers[sym] for sym in scratch_symbols(plan)) * _ITEMSIZE[plan.dtype] > LOCALS_LIMIT


def field_buffer(model: str, sym: str) -> str:
    """The static device buffer infer_one uses for scratch symbol `sym` (x and y are its arguments)."""
    return sym if sym in ("x", "y") else f"{model}_f_{sym}"


def _emit_infer(plan: Plan, ctype: str) -> list:
    """Emit the inference loop nest straight from plan.buffers / plan.assignment.

    There is deliberately no fusion pass and no private allocator here. The
    plan already owns the buffer assignment, one allocator serves both
    backends, and the plan hash covers what it produced; a second allocator
    with narrower assumptions (ruling R13) crashed on models validate accepts
    and the Fortran backend handles.
    """
    m = plan.model
    scratch = sorted((s for s in plan.buffers if s not in ("x", "y")),
                     key=lambda s: int(s[1:]))

    # Storage class first, then the attribute macro: the order CUDA's own
    # headers use for `static inline __host__ __device__`.
    lines = [f"static inline ROSENNA_DEVICE_FN void {m}_infer("
             f"const {ctype} *ROSENNA_RESTRICT x, {ctype} *ROSENNA_RESTRICT y) {{"]
    if plan.embed:
        # Controller ruling R9: in the host pass of a CUDA/HIP build the
        # embedded arrays are device storage (nvcc diagnoses a direct read,
        # hip-clang's host shadow is undefined), so that instantiation must
        # not touch them. The literals are not duplicated for a host twin.
        lines += [
            "#if ROSENNA_INFER_HOST_STUB",
            "    (void)x;",
            "    (void)y;",
            f'    assert(0 && "rosenna: {m}_infer is device-only in a CUDA/HIP build; '
            f'call it from a kernel or use {m}_infer_batch");',
            "#else",
        ]
    for sym in scratch:
        lines.append(f"    {ctype} {sym}[{plan.buffers[sym]}];")
    lines += _emit_op_sequence(plan, ctype)

    if plan.embed:
        lines.append("#endif")
    lines.append("}")
    lines.append("")
    return lines
