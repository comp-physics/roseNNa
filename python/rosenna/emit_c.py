"""Render a plan as a self-contained C source/header pair."""
from .abi import name_capacity, rank_capacity, status_code_comment
from .plan import Plan

_CTYPE = {"f32": "float", "f64": "double"}
_DTYPE_CODE = {"f32": 0, "f64": 1}
_ITEMSIZE = {"f32": 4, "f64": 8}
_CUDA_GUARD = "#if defined(__CUDACC__) || defined(__HIPCC__)"
_NOT_CUDA_GUARD = "#if !defined(__CUDACC__) && !defined(__HIPCC__)"
# Device-pass guard: nvcc defines __CUDA_ARCH__ and hipcc __HIP_DEVICE_COMPILE__
# only while compiling for the device, so a header-inline function can read one
# storage in its host instantiation and another in its device instantiation.
_DEVICE_PASS_GUARD = "#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)"

# Controller ruling R4. CUDA __constant__ memory is 64 KB per module, while
# a model embeds by default below EMBED_THRESHOLD (1M parameters, up to 8 MB
# of f64), so an embedded model whose weights exceed the constant budget must
# be placed in ordinary device memory or nvcc rejects the header. The cut is
# 48 KB of weight bytes, leaving the remaining 16 KB for anything else the
# translation unit puts in constant memory; the decision is per model, made
# once at generation time, and the header records which it took.
CONSTANT_MEMORY_LIMIT = 48 * 1024
# The one-thread-per-point kernel's block size (emit_kernel).
KERNEL_TILE = 128
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
        # __device__ const; compiles and runs on the host, device path
        # unvalidated until the GPU gate.)
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

    ROSENNA_BACKEND=omp (the default) compiles <name>.c with the host C
    compiler and the host's offload flags; its infer_batch is the OpenMP
    target loop over host pointers. ROSENNA_BACKEND=cuda|hip compiles both
    <name>.c (as C++, since init then calls the runtime) and <name>_kernel.cu
    with nvcc or hipcc; that infer_batch launches the native kernel over
    device pointers. The two are never linked together.
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
{n}.o: {n}.c {n}.h rosenna_rt.h
\t$(DEVCC) $(DEVFLAGS) -x cu -c $< -o $@
else ifeq ($(ROSENNA_BACKEND),hip)
DEVCC ?= hipcc
lib{n}.a: {n}.o {n}_kernel.o
\tar rcs $@ $^
{n}_kernel.o: {n}_kernel.cu {n}.h rosenna_rt.h
\t$(DEVCC) $(DEVFLAGS) -x hip -c $< -o $@
{n}.o: {n}.c {n}.h rosenna_rt.h
\t$(DEVCC) $(DEVFLAGS) -x hip -c $< -o $@
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
    CONSTANT_MEMORY_LIMIT, since CUDA constant memory is 64 KB per module;
    a larger embedded model reads them from __device__ const global memory
    instead. Both are `static` so that each translation unit that includes
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
    if plan.weights:
        lines.append(_CUDA_GUARD)
        for w in plan.weights:
            lines.append(f"{ctype} *{_c_weight_symbol(m, w.symbol)}_dev = 0;")
        lines.append("#endif")
    lines.append("")
    return lines


def _emit_upload(plan: Plan) -> list:
    """The cuda/hip half of init: copy the freshly loaded host arrays to the device.

    Only compiled under nvcc/hipcc, where the runtime is reachable through
    rosenna_rt.h (included by the header under the same guard). A repeated init frees the previous copies first (freeing a
    null pointer is a no-op in both runtimes); a failed allocation or copy
    leaves that pointer null and returns 10, so a later infer_batch refuses
    to launch rather than read an unfilled buffer. The last step publishes
    the new addresses to the kernel's translation unit (<name>_device_bind,
    in <name>_kernel.cu); after init returns, the loop path transfers
    nothing (controller ruling R5).
    """
    m = plan.model
    if not plan.weights:
        return []
    lines = [
        _CUDA_GUARD,
        f"static int {m}_upload(void) {{",
    ]
    for w in plan.weights:
        sym = _c_weight_symbol(m, w.symbol)
        lines += [
            f"    (void)ROSENNA_FREE({sym}_dev);",
            f"    {sym}_dev = 0;",
            f"    if (ROSENNA_MALLOC(&{sym}_dev, sizeof {sym}) != ROSENNA_OK) return 10;",
            f"    if (ROSENNA_MEMCPY_H2D({sym}_dev, {sym}, sizeof {sym}) != ROSENNA_OK) {{",
            f"        (void)ROSENNA_FREE({sym}_dev);",
            f"        {sym}_dev = 0;",
            f"        return 10;",
            f"    }}",
        ]
    lines += [f"    return {_device_bind(m)}();", "}", "#endif", ""]
    return lines


def _emit_fallback_infer_batch(plan: Plan, ctype: str) -> list:
    """The OpenMP-target infer_batch: the omp backend, over device pointers.

    Controller ruling R5: x and y are already on the device (is_device_ptr
    / deviceptr), so the loop path maps, allocates and synchronizes nothing.
    Compiled only when the compiler is not nvcc or hipcc; the cuda/hip
    backends define the same function in <name>_kernel.cu instead. Under a
    host compiler without -fopenmp/-fopenacc the pragmas are inert and this
    is a plain loop over host pointers, which is also what use_device_ptr
    yields on a host-only build. Compiles and runs on the host; device path
    unvalidated.
    """
    m = plan.model
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    return [
        _NOT_CUDA_GUARD,
        f"int {m}_infer_batch(int n, const {ctype} *ROSENNA_RESTRICT x, {ctype} *ROSENNA_RESTRICT y, void *stream) {{",
        "    (void)stream;",
        "    if (n <= 0) return 0;",
        "#if defined(_OPENMP)",
        "#pragma omp target teams loop is_device_ptr(x, y)",
        "#elif defined(_OPENACC)",
        "#pragma acc parallel loop deviceptr(x, y)",
        "#endif",
        f"    for (int p = 0; p < n; ++p) {m}_infer(x + (size_t)p * {n_in}, y + (size_t)p * {n_out});",
        "    return 0;",
        "}",
        "#endif",
        "",
    ]


def _emit_load(plan: Plan) -> list:
    if not plan.weights:
        return []
    m = plan.model
    lines = [
        "static int load_tensor(FILE *f, const char *name, int32_t namelen, long pos,",
        "                       int64_t length) {",
    ]
    for w in plan.weights:
        c_sym = _c_weight_symbol(m, w.symbol)
        # `length` comes from the file's own table of contents; a tensor whose
        # declared byte count disagrees with the array it is about to fill
        # means a corrupt file, so reject it rather than short-read into the
        # array and infer on half-loaded weights.
        lines.append(
            f"    if (namelen == {len(w.name)} && "
            f"memcmp(name, {_c_string(w.name)}, {len(w.name)}) == 0) {{")
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
        "        int status = load_tensor(f, name, namelen, data_start + (long)off, length);",
        "        if (status != 0) { fclose(f); return status; }",
        "        if (fseek(f, tocpos, SEEK_SET) != 0) { fclose(f); return 9; }",
        "    }",
        "    fclose(f);",
        "    /* The plan step's transfer: the device copies of the arrays, under",
        "       whichever offload family this build has. */",
        "#ifdef _OPENMP",
        f"#pragma omp target update to({_weight_symbol_list(plan)})",
        "#endif",
        "#ifdef _OPENACC",
        f"#pragma acc update device({_weight_symbol_list(plan)})",
        "#endif",
        _CUDA_GUARD,
        f"    return {m}_upload();",
        "#else",
        "    return 0;",
        "#endif",
        "}",
        "",
    ]
    return lines


def _emit_infer(plan: Plan, ctype: str) -> list:
    """Emit the inference loop nest straight from plan.buffers / plan.assignment.

    There is deliberately no fusion pass and no private allocator here. The
    plan already owns the buffer assignment, one allocator serves both
    backends, and the plan hash covers what it produced; a second allocator
    with narrower assumptions (ruling R13) crashed on models validate accepts
    and the Fortran backend handles.
    """
    m = plan.model
    n_in = plan.input.shape[0]
    act = _ACT_C[plan.dtype]
    weight_by_symbol = {w.symbol: w for w in plan.weights}
    scratch = sorted((s for s in plan.buffers if s not in ("x", "y")),
                     key=lambda s: int(s[1:]))

    lines = [f"ROSENNA_DEVICE_FN static inline void {m}_infer("
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

    cur_len = n_in
    for op in plan.ops:
        dst, src = plan.assignment[op.out], plan.assignment[op.inp]
        if op.kind == "gemm":
            idx_expr = _weight_index_c(weight_by_symbol, op)
            weight_sym = _weight_ref(plan, m, op.weight)
            bias_init = f"{_weight_ref(plan, m, op.bias)}[i]" if op.bias else _ZERO[plan.dtype]
            lines.append(f"    for (int i = 0; i < {op.n_out}; ++i) {{")
            lines.append(f"        {ctype} acc = {bias_init};")
            lines.append(
                f"        for (int j = 0; j < {op.n_in}; ++j) "
                f"acc += {src}[j] * {weight_sym}[{idx_expr}];")
            lines.append(f"        {dst}[i] = acc;")
            lines.append("    }")
            cur_len = op.n_out
        elif op.kind in act:
            expr = act[op.kind].format(v=f"{src}[i]")
            lines.append(f"    for (int i = 0; i < {cur_len}; ++i) {dst}[i] = {expr};")
        else:
            raise AssertionError(f"unhandled op kind {op.kind!r}")

    if plan.embed:
        lines.append("#endif")
    lines.append("}")
    lines.append("")
    return lines
