"""Render a plan as a self-contained Fortran module."""
from .abi import name_capacity, rank_capacity, status_code_comment
from .plan import Plan

_KIND = {"f32": "real32", "f64": "real64"}
# relu is written as merge, not max: max(v, 0) returns 0 for a NaN input, and
# this library is linked into solvers where a NaN out of a diverged run is the
# signal. merge(0, v, v < 0) has a false mask for NaN and so returns v.
_ACT = {"relu": "merge(0.0_wp, {v}, {v} < 0.0_wp)", "tanh": "tanh({v})",
        "sigmoid": "1.0_wp / (1.0_wp + exp(-({v})))"}
_DTYPE_CODE = {"f32": 0, "f64": 1}

# Embedding must be lossless: %.16e (17 significant digits: one before the
# point, sixteen after) round-trips any f64, %.8e (9 significant digits) any
# f32 -- the same Steele & White / Ryu-style bounds emit_c's %.17g/%.9g rely
# on, spelled as a fixed-width Fortran-legal exponential literal instead of
# C's shortest-form %g (which drops trailing digits %e always keeps).
_EMBED_FMT = {"f32": "%.8e", "f64": "%.16e"}

# Free-form Fortran source is limited to 132 columns (F2008 3.3.2.1). gfortran
# 15 accepts a longer line silently, gfortran 13 rejects it with
# -Werror=line-truncation, and every gfortran rejects it under -std=f2008, so
# nothing this emitter writes may rely on the compiler being lenient. The two
# places a line can grow without bound -- the 32-term hash constructor and a
# case label carrying a plan-derived tensor name -- go through the two wrappers
# below.
_MAX_COLS = 132


def _wrap_items(head: str, items: list, tail: str, indent: str) -> list:
    """Lay `head item, item, ... tail` out over continuation lines.

    Every emitted line, including the last one carrying `tail`, stays within
    _MAX_COLS. A continued line keeps its separating comma and ends with the
    free-form trailing `&`.
    """
    lines, cur = [], head + items[0]
    for item in items[1:]:
        # Either ", item" joins this line, which must then still have room
        # for whichever of ", &" and tail is longer, or the line closes here.
        if len(cur) + len(", " + item) + max(len(", &"), len(tail)) > _MAX_COLS:
            lines.append(cur + ", &")
            cur = indent + item
        else:
            cur += ", " + item
    lines.append(cur + tail)
    return lines


def _wrap_literal(prefix: str, s: str, suffix: str, indent: str) -> list:
    """Render `prefix'<s>'suffix`, continuing the character literal if needed.

    A character context is continued with a trailing `&` and a leading `&` on
    the next line; the text joins at the character level, so the split may
    fall anywhere -- except that this splits on the *unescaped* characters, so
    a doubled quote is never torn across two lines.
    """
    pieces = ["''" if c == "'" else c for c in s]
    single = prefix + "'" + "".join(pieces) + "'" + suffix
    if len(single) <= _MAX_COLS:
        return [single]
    lines, cur = [], prefix + "'"
    for piece in pieces:
        if len(cur) + len(piece) + len("&") > _MAX_COLS:
            lines.append(cur + "&")
            cur = indent + "&"
        cur += piece
    if len(cur) + len("'" + suffix) > _MAX_COLS:
        lines.append(cur + "&")
        cur = indent + "&"
    lines.append(cur + "'" + suffix)
    return lines


def _weight_dims(plan: Plan, symbol: str) -> str:
    spec = next(w for w in plan.weights if w.symbol == symbol)
    return "(" + ",".join(str(d) for d in reversed(spec.shape)) + ")"


def _weight_dims_list(plan: Plan, symbol: str) -> str:
    spec = next(w for w in plan.weights if w.symbol == symbol)
    return "[" + ",".join(str(d) for d in reversed(spec.shape)) + "]"


def _format_embedded_value(v: float, dtype: str) -> str:
    return (_EMBED_FMT[dtype] % v) + "_wp"


def _weight_symbol_list(plan: Plan) -> str:
    return ", ".join(w.symbol for w in plan.weights)


# F2008 free-form source caps a single statement at 255 continuation lines
# (3.3.2.4), independently of the 132-column limit each of those lines
# obeys: gfortran diagnoses an over-long array constructor with "Warning:
# Limit of 255 continuations exceeded", not a column complaint, so
# _wrap_items alone cannot keep a several-hundred-element weight (e.g.
# gemm_big's 40x30 = 1200-element layer) legal. Above _EMBED_CHUNK elements,
# the flat literal list is split into several small `parameter` arrays
# (each well under the continuation limit) and reassembled by the weight's
# own initializer over their names -- a statement with a handful of short
# identifiers, never close to either limit itself.
_EMBED_CHUNK = 500


def _emit_embedded_weights(plan: Plan) -> list:
    """`plan.embed`'s weights, as initialized `protected` module arrays.

    Not `parameter`: gfortran -fopenacc materializes a named-constant array
    read inside a `routine seq` as a static and then demands an OpenACC
    `declare` for it, which it refuses on a named constant ("not a
    variable"), so an embedded module could not be compiled (-c; the error
    is raised after the front end, so -fsyntax-only does not see it). An
    initialized `protected` module variable takes `declare copyin` and
    `declare target`, gets its device copy at program start under either
    offload family, and is as read-only outside the module as a constant.
    The chunk arrays a long literal list is split into (see _EMBED_CHUNK)
    stay `parameter`: only the initializer names them, never the routine.

    A rank-1 array (every bias) is a plain bracketed list. A rank-2 array
    (every Gemm/MatMul weight) is `reshape([flat values], [dims])`: `values`
    is already the C-order (row-major) flattening of the ONNX-shaped tensor
    (plan.py's `np.ravel(order="C")`), and Fortran's default array
    constructor fills its target in column-major order, so reshaping that
    same flat sequence into the *reversed* shape reproduces exactly the
    memory layout `_weight_dims` already declares for the file-loaded form
    -- the same raw-bytes-in trick `load_tensor` relies on, done here at
    compile time instead of at file-read time.
    """
    lines = []
    for w in plan.weights:
        dims = _weight_dims(plan, w.symbol)
        items = [_format_embedded_value(v, plan.dtype) for v in w.values]
        if len(items) <= _EMBED_CHUNK:
            flat = items
        else:
            chunks = [items[i:i + _EMBED_CHUNK] for i in range(0, len(items), _EMBED_CHUNK)]
            chunk_names = [f"{w.symbol}_c{k}" for k in range(len(chunks))]
            for cname, chunk in zip(chunk_names, chunks):
                lines += _wrap_items(f"    real(wp), parameter :: {cname}({len(chunk)}) = [ ",
                                     chunk, " ]", " " * 8)
            flat = chunk_names
        if len(w.shape) <= 1:
            head, tail = f"    real(wp), protected :: {w.symbol}{dims} = [ ", " ]"
        else:
            head = f"    real(wp), protected :: {w.symbol}{dims} = reshape([ "
            tail = f" ], {_weight_dims_list(plan, w.symbol)})"
        lines += _wrap_items(head, flat, tail, " " * 8)
    return lines


def emit_fortran(plan: Plan) -> str:
    m, wp = plan.model, _KIND[plan.dtype]
    public = [f"{m}_infer", f"{m}_infer_batch", f"{m}_infer_batch_dev"]
    if not plan.embed:
        public.insert(0, f"{m}_init")
    lines = [
        f"module {m}_model",
        "    ! Generated by rosenna. Do not edit.",
        "    use iso_fortran_env, only: int8, int32, int64, " + wp,
        "    use iso_c_binding, only: c_int, c_ptr",
        "    implicit none",
        "    private",
        "    public :: " + ", ".join(public),
        "",
        f"    integer, parameter :: wp = {wp}",
    ]
    if not plan.embed:
        # Wrap each literal in an explicit int(..., int8) conversion: an
        # unsuffixed literal is default INTEGER(4), and initializing an
        # INTEGER(1) parameter array from those trips gfortran's -Wconversion
        # under -Wextra. A plain "_int8" kind suffix does not work either --
        # gfortran checks a literal's unsigned magnitude against the kind's
        # range before applying unary minus, so "-128_int8" is rejected as
        # "Integer too big for its kind" even though -128 is in range.
        hash_terms = [f"int({b if b < 128 else b - 256}, int8)" for b in bytes.fromhex(plan.hash())]
        lines += _wrap_items("    integer(int8), parameter :: expected_hash(32) = [ ", hash_terms, " ]",
                             " " * 8)
        for w in plan.weights:
            lines.append(f"    real(wp), protected :: {w.symbol}{_weight_dims(plan, w.symbol)}")
    else:
        lines += _emit_embedded_weights(plan)
    if plan.weights:
        lines.append(f"    !$omp declare target({_weight_symbol_list(plan)})")
        # gfortran rejects a `routine seq` function that reads a module array
        # with no OpenACC `declare` directive of its own: `create` for the
        # file-loaded arrays (init then does `update device`), `copyin` for
        # the embedded ones, whose initializer is the device copy's value.
        clause = "copyin" if plan.embed else "create"
        lines.append(f"    !$acc declare {clause}({_weight_symbol_list(plan)})")
    lines += [
        "",
        "    interface",
        f"        function {m}_infer_batch_dev(n, x, y, stream) &",
        f'                bind(C, name="{m}_infer_batch") result(status)',
        "            import :: c_int, c_ptr",
        "            integer(c_int), value :: n",
        "            type(c_ptr), value :: x, y, stream",
        "            integer(c_int) :: status",
        "        end function",
        "    end interface",
        "",
        "contains",
        "",
    ]
    if not plan.embed:
        lines += _emit_init(plan)
        lines += _emit_load(plan)
    lines += _emit_infer(plan)
    lines += _emit_infer_batch(plan)
    lines.append(f"end module {m}_model")
    return "\n".join(lines) + "\n"


def _emit_init(plan: Plan) -> list:
    m = plan.model
    dtype_code = _DTYPE_CODE[plan.dtype]
    lines = ["    " + line for line in status_code_comment("!", m)]
    lines += [
        f"    subroutine {m}_init(path, status)",
        "        character(*), intent(in) :: path",
        "        integer, intent(out) :: status",
        "        integer :: u, ios, ntensors",
        "        integer(int32) :: version, dtype, endian",
        "        character(len=8) :: magic",
        "        integer(int8) :: file_hash(32)",
    ]
    if plan.weights:
        lines += [
            "        integer :: toclen, k, namelen, rank",
            "        integer(int64) :: dims(%d), off, length, data_start, tocpos" % rank_capacity(plan),
            "        character(len=%d) :: name" % name_capacity(plan),
        ]
    lines += [
        "        status = 0",
        "        open(newunit=u, file=path, status='old', action='read', access='stream', &",
        "             form='unformatted', iostat=ios)",
        "        if (ios /= 0) then; status = 1; return; end if",
        "        read(u, iostat=ios) magic",
        "        if (ios /= 0) then; status = 9; close(u); return; end if",
        "        if (magic /= 'ROSENNA1') then; status = 2; close(u); return; end if",
        "        read(u, iostat=ios) version, dtype, endian, ntensors",
        "        if (ios /= 0) then; status = 9; close(u); return; end if",
        "        if (version /= 1) then; status = 3; close(u); return; end if",
        f"        if (dtype /= {dtype_code}) then; status = 4; close(u); return; end if",
        "        if (endian /= 16909060) then; status = 5; close(u); return; end if",
        "        read(u, iostat=ios) file_hash",
        "        if (ios /= 0) then; status = 9; close(u); return; end if",
        "        if (any(file_hash /= expected_hash)) then; status = 6; close(u); return; end if",
    ]
    if not plan.weights:
        # This model has no weights at all, so a well-formed file for it holds
        # no tensors. Skipping the table of contents entirely keeps the
        # generated module free of an unreachable loop and of the unused
        # locals and dummy arguments that loop would need.
        lines += [
            "        if (ntensors /= 0) then; status = 7; close(u); return; end if",
            "        close(u)",
            "    end subroutine",
            "",
        ]
        return lines
    lines += [
        "        read(u, iostat=ios) toclen",
        "        if (ios /= 0) then; status = 9; close(u); return; end if",
        "        data_start = int(60, int64) + int(toclen, int64) + 1_int64",
        "        do k = 1, ntensors",
        "            read(u, iostat=ios) namelen",
        "            if (ios /= 0) then; status = 9; close(u); return; end if",
        f"            if (namelen < 0 .or. namelen > {name_capacity(plan)}) then",
        "                status = 8; close(u); return",
        "            end if",
        "            name = ''",
        "            read(u, iostat=ios) name(1:namelen)",
        "            if (ios /= 0) then; status = 9; close(u); return; end if",
        "            read(u, iostat=ios) rank",
        "            if (ios /= 0) then; status = 9; close(u); return; end if",
        f"            if (rank < 0 .or. rank > {rank_capacity(plan)}) then",
        "                status = 8; close(u); return",
        "            end if",
        "            read(u, iostat=ios) dims(1:rank)",
        "            if (ios /= 0) then; status = 9; close(u); return; end if",
        "            read(u, iostat=ios) off, length",
        "            if (ios /= 0) then; status = 9; close(u); return; end if",
        "            inquire(unit=u, pos=tocpos)",
        "            call load_tensor(u, name(1:namelen), data_start + off, length, status)",
        "            if (status /= 0) then; close(u); return; end if",
        "            read(u, pos=tocpos, iostat=ios)",
        "            if (ios /= 0) then; status = 9; close(u); return; end if",
        "        end do",
        "        close(u)",
        "        ! The plan step's transfer (ruling R5): make the freshly loaded",
        "        ! weights device-resident. Fortran has no runtime-API path without",
        "        ! CUDA Fortran, so this directive form is the whole of init's device",
        "        ! copy; one host call both loads and uploads.",
        f"        !$omp target update to({_weight_symbol_list(plan)})",
        f"        !$acc update device({_weight_symbol_list(plan)})",
        "    end subroutine",
        "",
    ]
    return lines


def _emit_load(plan: Plan) -> list:
    if not plan.weights:
        return []
    lines = [
        "    subroutine load_tensor(u, name, pos, length, status)",
        "        integer, intent(in) :: u",
        "        character(*), intent(in) :: name",
        "        integer(int64), intent(in) :: pos, length",
        "        integer, intent(out) :: status",
        "        integer :: ios",
        "        status = 0",
        "        ios = 0",
        "        select case (name)",
    ]
    for w in plan.weights:
        # `length` comes from the file's own table of contents; a tensor whose
        # declared byte count disagrees with the array it is about to fill
        # means a corrupt file, not a short read, so reject before reading.
        lines += _wrap_literal("        case (", w.name, ")", " " * 12)
        lines.append(f"            if (length /= {w.nbytes}_int64) then; status = 9; return; end if")
        lines.append(f"            read(u, pos=pos, iostat=ios) {w.symbol}")
    lines += [
        "        case default; status = 7; return",
        "        end select",
        "        if (ios /= 0) status = 9",
        "    end subroutine",
        "",
    ]
    return lines


def _weight_index(weight_by_symbol: dict, op) -> str:
    """Decide the accumulation index order from the op's own transB flag.

    A transB=1 Gemm weight has ONNX shape (n_out, n_in): Fortran dims are
    (n_in, n_out) and the accumulation reads w(j, i). A transB=0 weight, or
    any MatMul weight, has ONNX shape (n_in, n_out): Fortran dims are
    (n_out, n_in) and the accumulation reads w(i, j).

    The flag is authoritative; the shape check below only makes a plan that
    disagrees with its own weights fail loudly instead of silently reading a
    transposed array. Sniffing the convention back out of the shape is what
    this replaces: when n_in == n_out the two cases are indistinguishable.
    """
    spec = weight_by_symbol[op.weight]
    expected = (op.n_out, op.n_in) if op.trans_b else (op.n_in, op.n_out)
    if tuple(spec.shape) != expected:
        raise AssertionError(
            f"weight {op.weight}: shape {tuple(spec.shape)} does not match the layout "
            f"the plan claims (transB={int(op.trans_b)} implies {expected})")
    return "j, i" if op.trans_b else "i, j"


def _emit_infer(plan: Plan) -> list:
    m = plan.model
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    weight_by_symbol = {w.symbol: w for w in plan.weights}
    scratch = sorted((s for s in plan.buffers if s not in ("x", "y")),
                      key=lambda s: int(s[1:]))

    lines = [
        f"    pure subroutine {m}_infer(x, y)",
        "        !$omp declare target",
        "        !$acc routine seq",
        f"        real(wp), intent(in) :: x({n_in})",
        f"        real(wp), intent(out) :: y({n_out})",
    ]
    for sym in scratch:
        lines.append(f"        real(wp) :: {sym}({plan.buffers[sym]})")
    # Declare only the loop variables some op actually uses: a weight-free
    # model (x -> Relu -> y) has no inner accumulation loop, and an unused
    # 'j' is a warning in any tree built with -Werror.
    loop_vars = [v for v, used in (("i", bool(plan.ops)),
                                   ("j", any(op.kind == "gemm" for op in plan.ops))) if used]
    if loop_vars:
        lines.append("        integer :: " + ", ".join(loop_vars))

    # Fusing an activation into its preceding gemm's loop would require the
    # activation's output buffer to equal the gemm's output buffer. Task 4's
    # _assign_buffers can never produce that: an op's output symbol is
    # assigned before its input's symbol is returned to the free list, so no
    # op -- activation or otherwise -- can ever share a buffer with its own
    # input. That makes the fusion case unreachable by construction, not by
    # coincidence, so there is no such branch here: every op gets its own
    # loop, straight from plan.assignment and plan.buffers.
    ops = plan.ops
    cur_len = n_in
    for op in ops:
        if op.kind == "gemm":
            dst = plan.assignment[op.out]
            src = plan.assignment[op.inp]
            idx_expr = _weight_index(weight_by_symbol, op)

            lines.append(f"        do i = 1, {op.n_out}")
            if op.bias:
                lines.append(f"            {dst}(i) = {op.bias}(i)")
            else:
                lines.append(f"            {dst}(i) = 0.0_wp")
            lines.append(f"            do j = 1, {op.n_in}")
            lines.append(f"                {dst}(i) = {dst}(i) + {src}(j) * {op.weight}({idx_expr})")
            lines.append("            end do")
            lines.append("        end do")
            cur_len = op.n_out
        elif op.kind in _ACT:
            dst = plan.assignment[op.out]
            src = plan.assignment[op.inp]
            lines.append(f"        do i = 1, {cur_len}")
            lines.append(f"            {dst}(i) = {_ACT[op.kind].format(v=f'{src}(i)')}")
            lines.append("        end do")
        else:
            raise AssertionError(f"unhandled op kind {op.kind!r}")

    lines.append("    end subroutine")
    lines.append("")
    return lines


def _emit_infer_batch(plan: Plan) -> list:
    """The OpenMP-target fallback infer_batch: over device-resident arrays.

    Ruling R5: x and y are already on the device (has_device_addr / the
    OpenACC deviceptr twin), so the loop path allocates, maps and transfers
    nothing -- the host maps them itself (`!$omp target enter data`) once,
    outside this call. `has_device_addr` on this explicit-shape dummy
    (rather than the assumed-shape form the spec flags as a compiler risk)
    is accepted, without warning, under gfortran 15 -fopenmp -std=f2008; see
    task-4-report.md for which gfortran this was verified against.
    """
    m = plan.model
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    return [
        f"    subroutine {m}_infer_batch(n, x, y, status)",
        "        integer, intent(in) :: n",
        f"        real(wp), intent(in) :: x({n_in}, n)",
        f"        real(wp), intent(out) :: y({n_out}, n)",
        "        integer, intent(out) :: status",
        "        integer :: p",
        "        ! Ruling R5: x and y are already device-resident. No data clause and",
        "        ! nothing else here transfers, allocates or synchronizes.",
        "        !$omp target teams loop has_device_addr(x, y)",
        "        !$acc parallel loop deviceptr(x, y)",
        "        do p = 1, n",
        f"            call {m}_infer(x(:, p), y(:, p))",
        "        end do",
        "        status = 0",
        "    end subroutine",
        "",
    ]


def emit_fortran_recipe(plan: Plan) -> str:
    """A Makefile fragment that builds lib<name>_f.a from the generated module.

    Mirrors emit_c_recipe's shape: FC/FFLAGS/ROSENNA_OFFLOAD_FLAGS are
    override points for the host's own compiler and offload flags. gfortran
    drops <name>_model.mod alongside the object as a side effect of
    compiling; nothing in the recipe needs to name it, but a host module
    that `use`s this one needs that .mod on its include path.

    Controller ruling R13: the archive is lib<name>_f.a, not lib<name>.a --
    `generate --lang both` writes this recipe and emit_c_recipe's into ONE
    output directory, and `ar rcs` APPENDS to an existing archive, so two
    recipes sharing one archive name silently merge their objects into it
    the moment both are built there (and either recipe's `clean` then
    deletes the other's artifacts too). A Fortran host that also links the
    native CUDA/HIP kernel links both archives: `-l<name>_f -l<name>`.
    """
    n = plan.model
    return f"""# Generated by rosenna. Builds lib{n}_f.a from {n}_model.f90 (and {n}_model.mod).
FC ?= gfortran
FFLAGS ?= -O2 -Wall -Wextra -std=f2008
ROSENNA_OFFLOAD_FLAGS ?=

lib{n}_f.a: {n}_model.o
\tar rcs $@ $^
{n}_model.o: {n}_model.f90
\t$(FC) $(FFLAGS) $(ROSENNA_OFFLOAD_FLAGS) -c $< -o $@
clean:
\trm -f {n}_model.o {n}_model.mod lib{n}_f.a
.PHONY: clean
"""
