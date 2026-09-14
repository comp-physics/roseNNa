"""Render a plan as a self-contained C source/header pair."""
from .abi import name_capacity, rank_capacity, status_code_comment
from .plan import Plan

_CTYPE = {"f32": "float", "f64": "double"}
_DTYPE_CODE = {"f32": 0, "f64": 1}
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

    A file-loaded plan's header declares two different C names for the same
    weight -- the host array `<symbol>` and, under __CUDACC__/__HIPCC__, the
    device pointer `<symbol>_dev` (controller ruling P4; the pointer is
    declared-only until Task 3) -- and `infer` is emitted exactly once, so it
    cannot spell either name directly. This macro (itself prefixed with the
    model-qualified symbol, so it carries ruling R1's collision safety same
    as every other external name here) is `#define`d to whichever of the two
    the same __CUDACC__/__HIPCC__ guard selects; `infer`'s body reads only
    this name. An embedded plan has no such split -- its weights are one
    ROSENNA_CONST array reachable under any guard -- so `infer` reads the
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
    if plan.embed:
        # Every weight is a ROSENNA_CONST array in the header, so the source
        # has nothing left to define. (Under __CUDACC__/__HIPCC__ that macro
        # is __constant__; compiles and runs on the host, device path
        # unvalidated until the GPU gate.)
        source = f'/* Generated by rosenna. Do not edit. */\n#include "{plan.model}.h"\n'
    else:
        lines = []
        lines += _emit_source_head(plan, ctype)
        lines += _emit_load(plan)
        lines += _emit_init(plan)
        source = "\n".join(lines) + "\n"
    return source, header


def emit_c_recipe(plan: Plan) -> str:
    """A Makefile fragment that builds lib<name>.a with the host's offload flags."""
    n = plan.model
    return f"""# Generated by rosenna. Build lib{n}.a with the same offload flags as the host.
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
ROSENNA_OFFLOAD_FLAGS ?=

lib{n}.a: {n}.o
\tar rcs $@ $<
{n}.o: {n}.c {n}.h
\t$(CC) $(CFLAGS) $(ROSENNA_OFFLOAD_FLAGS) -c $< -o $@
clean:
\trm -f {n}.o lib{n}.a
.PHONY: clean
"""


def _emit_header(plan: Plan, ctype: str) -> str:
    m = plan.model
    guard = f"ROSENNA_{m.upper()}_H"
    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "/* Generated by rosenna. Do not edit. */",
        "",
        "#include <math.h>",
        "",
        # Every device decoration in this header goes through exactly these
        # three macros (controller rulings: brief + P3). Under nvcc/hipcc,
        # `infer` is callable from device code and embedded weights live in
        # __constant__ memory; under any other compiler both are inert. C has
        # no `restrict` keyword once this header is pulled into a C++ (or
        # CUDA/HIP, which is always C++) translation unit, so ROSENNA_RESTRICT
        # picks the compiler-correct spelling instead of `infer` hardcoding one.
        "#if defined(__CUDACC__) || defined(__HIPCC__)",
        "#define ROSENNA_DEVICE_FN __host__ __device__",
        "#define ROSENNA_CONST __constant__",
        "#define ROSENNA_RESTRICT __restrict__",
        "#else",
        "#define ROSENNA_DEVICE_FN",
        "#define ROSENNA_CONST static const",
        "#if defined(__cplusplus)",
        "#define ROSENNA_RESTRICT __restrict__",
        "#else",
        "#define ROSENNA_RESTRICT restrict",
        "#endif",
        "#endif",
        "",
    ]
    if not plan.embed:
        lines += _emit_weight_declarations(plan, ctype)
    lines += _emit_device_region(plan, ctype)
    if not plan.embed:
        lines += [
            f"int {m}_init(const char *path);",
            "",
        ]
    lines.append("#endif")
    return "\n".join(lines) + "\n"


def _emit_weight_declarations(plan: Plan, ctype: str) -> list:
    """A file-loaded plan's weights: a host array, or (Task 3) a device pointer.

    Controller ruling P4: the `_dev` pointers are declared, not defined --
    nothing fills them in until Task 3's CUDA/HIP init exists. Under a plain
    or OpenMP host build the __CUDACC__/__HIPCC__ guard is false, so this
    branch is never even compiled; no build in this task can reach it.
    Compiles and runs on the host; device path unvalidated.
    """
    m = plan.model
    if not plan.weights:
        return []
    lines = ["#if defined(__CUDACC__) || defined(__HIPCC__)"]
    for w in plan.weights:
        sym = _c_weight_symbol(m, w.symbol)
        lines.append(f"extern {ctype} *{sym}_dev;")
        lines.append(f"#define {_c_weight_ref_macro(m, w.symbol)} {sym}_dev")
    lines.append("#else")
    for w in plan.weights:
        sym = _c_weight_symbol(m, w.symbol)
        lines.append(f"extern {ctype} {sym}[{_weight_size(w.shape)}];")
        lines.append(f"#define {_c_weight_ref_macro(m, w.symbol)} {sym}")
    lines += ["#endif", ""]
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
    if plan.weights:
        lines.append("")
    return lines


def _emit_source_head(plan: Plan, ctype: str) -> list:
    m = plan.model
    hash_bytes = ", ".join(f"0x{b:02x}" for b in bytes.fromhex(plan.hash()))
    lines = [
        "/* Generated by rosenna. Do not edit. */",
        f'#include "{m}.h"',
        "",
        "#include <stdint.h>",
        "#include <stdio.h>",
        "#include <string.h>",
        "",
        f"static const unsigned char expected_hash[32] = {{ {hash_bytes} }};",
        "",
    ]
    for w in plan.weights:
        lines.append(f"{ctype} {_c_weight_symbol(m, w.symbol)}[{_weight_size(w.shape)}];")
    lines.append("")
    return lines


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
        "    return 0;",
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

    lines.append("}")
    lines.append("")
    return lines
