"""Render a plan as a self-contained Fortran module.

Written to <name>_model.F90, with a capital F: infer_batch's device-pointer
clause is spelled one way for nvfortran and another for everyone else (see
_emit_infer_batch), and a capital-F suffix is the one way to ask for the
preprocessor that every Fortran compiler honours without a flag.
"""
from .abi import name_capacity, rank_capacity, status_code_comment
from .emit_c import GEMM_BLOCK, gemm_block, has_infer_one
from .plan import Plan, lstm_initial_state

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
    public = [f"{m}_infer", f"{m}_infer_batch", f"{m}_infer_batch_dev", f"{m}_sync_dev"]
    if not plan.embed:
        public.insert(0, f"{m}_init")
        public.append(f"{m}_init_dev")
    if has_infer_one(plan):
        public.append(f"{m}_infer_one_dev")
    lines = [
        f"module {m}_model",
        "    ! Generated by rosenna. Do not edit.",
        "    use iso_fortran_env, only: int8, int32, int64, " + wp,
        "    use iso_c_binding, only: c_int, c_ptr, c_char",
        "    implicit none",
        "    private",
        # One name per line: a long model name times five symbols runs past
        # the 132-column limit (ruling R20).
        *[f"    public :: {name}" for name in public],
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
        "        ! Waits for every infer_batch launched on `stream` (a no-op for an",
        "        ! omp-backend libmodel.a); see the C header.",
        f"        function {m}_sync_dev(stream) &",
        f'                bind(C, name="{m}_sync") result(status)',
        "            import :: c_int, c_ptr",
        "            type(c_ptr), value :: stream",
        "            integer(c_int) :: status",
        "        end function",
    ]
    if has_infer_one(plan):
        lines += [
            f"        ! One sample over device pointers, one launch per op (see the C header).",
            f"        function {m}_infer_one_dev(x, y, stream) &",
            f'                bind(C, name="{m}_infer_one") result(status)',
            "            import :: c_int, c_ptr",
            "            type(c_ptr), value :: x, y, stream",
            "            integer(c_int) :: status",
            "        end function",
        ]
    if not plan.embed:
        lines += [
            f"        ! The C archive's own init (lib{m}.a): a host calling {m}_infer_batch_dev",
            "        ! on a cuda/hip archive must run this, since only it uploads the archive's",
            f"        ! device copies -- {m}_init above fills this module's arrays for the",
            "        ! Fortran per-point path. Pass a C string (path // c_null_char).",
            f"        function {m}_init_dev(path) &",
            f'                bind(C, name="{m}_init") result(status)',
            "            import :: c_int, c_char",
            "            character(kind=c_char), dimension(*), intent(in) :: path",
            "            integer(c_int) :: status",
            "        end function",
        ]
    lines += [
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
            "        logical :: seen(%d)" % len(plan.weights),
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
        "        seen = .false.",
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
        "            call load_tensor(u, name(1:namelen), data_start + off, length, seen, status)",
        "            if (status /= 0) then; close(u); return; end if",
        "            read(u, pos=tocpos, iostat=ios)",
        "            if (ios /= 0) then; status = 9; close(u); return; end if",
        "        end do",
        "        close(u)",
        "        if (.not. all(seen)) then; status = 9; return; end if",
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
        "    ! seen(k) is set when plan weight k has been filled: a table of",
        "    ! contents that names a tensor twice, or not at all, is status 9",
        "    ! (inconsistent), never a zero array that infer then runs on.",
        "    subroutine load_tensor(u, name, pos, length, seen, status)",
        "        integer, intent(in) :: u",
        "        character(*), intent(in) :: name",
        "        integer(int64), intent(in) :: pos, length",
        "        logical, intent(inout) :: seen(:)",
        "        integer, intent(out) :: status",
        "        integer :: ios",
        "        status = 0",
        "        ios = 0",
        "        select case (name)",
    ]
    for k, w in enumerate(plan.weights, start=1):
        # `length` comes from the file's own table of contents; a tensor whose
        # declared byte count disagrees with the array it is about to fill
        # means a corrupt file, not a short read, so reject before reading.
        lines += _wrap_literal("        case (", w.name, ")", " " * 12)
        lines.append(f"            if (seen({k})) then; status = 9; return; end if")
        lines.append(f"            seen({k}) = .true.")
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


def _gemm_blocked_f(op, dst, src, ri, ro, idx_expr) -> list:
    """A dense layer, gemm_block(op) columns per pass, scalar accumulators (a0.. and acc are declared by infer)."""
    block = gemm_block(op)
    col = lambda k: idx_expr.replace("i", f"i + {k}")
    bias = lambda k: f" + {op.bias}(i + {k})" if op.bias else ""
    L = []
    full = (op.n_out // block) * block if block > 1 else 0
    if full:
        L += [f"        do i = 1, {full}, {block}",
              "            " + "; ".join(f"a{k} = 0.0_wp" for k in range(block)),
              f"            do j = 1, {op.n_in}",
              f"                sj = {src}({ri}j)",
              *[f"                a{k} = a{k} + sj * {op.weight}({col(k)})" for k in range(block)],
              "            end do",
              *[f"            {dst}({ro}i + {k}) = a{k}{bias(k)}" for k in range(block)],
              "        end do"]
    if full < op.n_out:
        L += [f"        do i = {full + 1}, {op.n_out}",
              "            acc = 0.0_wp",
              f"            do j = 1, {op.n_in}",
              f"                acc = acc + {src}({ri}j) * {op.weight}({idx_expr})",
              "            end do",
              f"            {dst}({ro}i) = acc{bias(0).replace('i + 0', 'i')}",
              "        end do"]
    return L


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
    # 'j' is a warning in any tree built with -Werror. The spatial nest brings
    # its own set, and its accumulator has to be a named local because Fortran
    # has no statement-scoped declarations the way the C nest does.
    kinds = {op.kind for op in plan.ops}
    spatial = kinds & {"conv", "maxpool", "avgpool"}
    loop_vars = [v for v, used in (
        ("i", any(k in ("gemm", "copy", "lstm", "concat") or k in _ACT for k in kinds)),
        ("o", "concat" in kinds),
        ("j", "lstm" in kinds and "gemm" not in kinds),
        ("j", "gemm" in kinds),
        ("r", any(op.kind == "gemm" and op.rows > 1 for op in plan.ops)),
        ("n, oc, oh, ow, ic, kh, kw, ih, iw", bool(spatial)),
        ("icg", any(op.kind == "conv" and op.spatial.grouped for op in plan.ops)),
        ("smn, smj", "softmax" in kinds),
        ("seen", "maxpool" in kinds),
        ("cnt", any(_avgpool_needs_count(op) for op in plan.ops)),
        ("lt, lb, lk", "lstm" in kinds),
        (", ".join(f"c{k}" for k in range(_max_counter_rank(plan))),
         bool(kinds & {"add", "transpose", "pad"}))) if used]
    if loop_vars:
        lines.append("        integer :: " + ", ".join(loop_vars))
    gemms = [op for op in plan.ops if op.kind == "gemm"]
    blocked = any(gemm_block(op) > 1 and op.n_out >= GEMM_BLOCK for op in gemms)
    remainder = any(gemm_block(op) == 1 or op.n_out % GEMM_BLOCK for op in gemms)
    reals = ["acc"] if (spatial or "lstm" in kinds or remainder) else []
    if "maxpool" in kinds:
        reals.append("v")
    if "softmax" in kinds:
        reals += ["smx", "ssum", "sexp"]
    if "lstm" in kinds:
        reals += ["lgi", "lgo", "lgf", "lgc", "lcn"]
    if blocked:
        reals += ["sj"] + [f"a{k}" for k in range(GEMM_BLOCK)]
    if reals:
        lines.append("        real(wp) :: " + ", ".join(reals))

    # Fusing an activation into its preceding gemm's loop would require the
    # activation's output buffer to equal the gemm's output buffer. Task 4's
    # _assign_buffers can never produce that: an op's output symbol is
    # assigned before its input's symbol is returned to the free list, so no
    # op -- activation or otherwise -- can ever share a buffer with its own
    # input. That makes the fusion case unreachable by construction, not by
    # coincidence, so there is no such branch here: every op gets its own
    # loop, straight from plan.assignment and plan.buffers.
    ops = plan.ops
    for op in ops:
        if op.kind == "gemm":
            dst = plan.assignment[op.out]
            src = plan.assignment[op.inp]
            idx_expr = _weight_index(weight_by_symbol, op)

            # The bias is added AFTER the dot product, not used to seed the
            # accumulator. Seeding it from a declare-target array is what makes
            # nvc/nvfortran refuse to generate a `distribute parallel for` body
            # (it emits a kernel that traps); adding it afterwards compiles, and
            # unlocks a ~19x faster per-point offload loop. See
            # examples/nvhpc_teams_mapping/. Both emitters do this identically,
            # so the C and Fortran backends stay bit-comparable.
            # See emit_c: `r` indexes independent rows sharing one weight, and
            # is only emitted when a model actually has more than one.
            ri, ro = ((f"(r - 1) * {op.n_in} + ", f"(r - 1) * {op.n_out} + ")
                      if op.rows > 1 else ("", ""))
            if op.rows > 1:
                lines.append(f"        do r = 1, {op.rows}")
            lines += _gemm_blocked_f(op, dst, src, ri, ro, idx_expr)
            if op.rows > 1:
                lines.append("        end do")
        elif op.kind == "alias":
            continue
        elif op.kind == "transpose":
            lines += _emit_transpose_f(op, plan.assignment[op.out], plan.assignment[op.inp])
        elif op.kind == "lstm":
            h0, c0 = lstm_initial_state(op, lambda sym: sym, plan.assignment)
            lines += _emit_lstm_f(
                op, plan.assignment.get(op.out), plan.assignment[op.inp], h0, c0,
                [plan.assignment[o] if o else "" for o in op.outs])
        elif op.kind == "add":
            lines += _emit_add_f(op, plan.assignment[op.out], plan.assignment[op.inp])
        elif op.kind == "concat":
            lines += _emit_concat_f(op, plan.assignment[op.out],
                                    [plan.assignment[op.inp]] + [plan.assignment[n] for n in op.extra_in])
        elif op.kind == "copy":
            dst, src = plan.assignment.get(op.out), plan.assignment[op.inp]
            soff = f"{op.src_offset} + " if op.src_offset else ""
            doff = f"{op.dst_offset} + " if op.dst_offset else ""
            lines.append(f"        do i = 1, {op.n_out}")
            lines.append(f"            {dst}({doff}i) = {src}({soff}i)")
            lines.append("        end do")
        elif op.kind == "pad":
            lines += _emit_pad_f(op, plan.assignment[op.out], plan.assignment[op.inp])
        elif op.kind == "softmax":
            lines += _emit_softmax_f(op, plan.assignment[op.out], plan.assignment[op.inp])
        elif op.kind in ("conv", "maxpool", "avgpool"):
            lines += _emit_spatial_f(op, plan.assignment[op.out], plan.assignment[op.inp])
        elif op.kind in _ACT:
            dst = plan.assignment[op.out]
            src = plan.assignment[op.inp]
            lines.append(f"        do i = 1, {op.n_out}")
            lines.append(f"            {dst}(i) = {_ACT[op.kind].format(v=f'{src}(i)')}")
            lines.append("        end do")
        else:
            raise AssertionError(f"unhandled op kind {op.kind!r}")

    lines.append("    end subroutine")
    lines.append("")
    return lines


def _emit_pad_f(op, dst: str, src: str) -> list:
    """The Fortran twin of _emit_pad_c: same nest, same bounds, 1-based subscripts."""
    pd = op.pad
    names = [f"c{k}" for k in range(len(pd.out_shape))]
    shifted, checks = [], []
    for k, (nm, b) in enumerate(zip(names, pd.begins)):
        if b == 0 and pd.in_shape[k] == pd.out_shape[k]:
            shifted.append(nm)
            continue
        # A negative begin is a crop: the output reads FURTHER into the input,
        # so it is spelled as an addition. `(c - -1)` is legal C and a syntax
        # error in Fortran, which is what made this worth spelling out rather
        # than letting the sign fall out of the arithmetic.
        expr = f"({nm} - {b})" if b > 0 else f"({nm} + {-b})" if b < 0 else nm
        shifted.append(expr)
        # Only a positive begin can put the read before the input; a crop
        # cannot, so that half of the test would always pass.
        if b > 0:
            checks.append(f"{expr} >= 0")
        checks.append(f"{expr} < {pd.in_shape[k]}")
    L = []
    for nm, extent in zip(names, pd.out_shape):
        L.append(f"        do {nm} = 0, {extent - 1}")
    out_idx = _flat_index_f(names, pd.out_shape)
    in_idx = _flat_index_f(shifted, pd.in_shape)
    if checks:
        L.append(f"            if ({' .and. '.join(checks)}) then")
        L.append(f"                {dst}({out_idx}) = {src}({in_idx})")
        L.append("            else")
        L.append(f"                {dst}({out_idx}) = {pd.value!r}_wp")
        L.append("            end if")
    else:
        L.append(f"            {dst}({out_idx}) = {src}({in_idx})")
    L += ["        end do"] * len(names)
    return L


def _emit_softmax_f(op, dst: str, src: str) -> list:
    """The Fortran twin of _emit_softmax_c, line for line.

    Counters stay 0-based so the index arithmetic reads the same as C's; only
    the subscript gains the `+ 1`. See _emit_softmax_c for why the maximum is
    written `>` here rather than the NaN-preserving form the other reductions
    use.
    """
    sm = op.softmax
    c = sm.axis_len
    at = f"smn * {c}"
    return [
        f"        do smn = 0, {sm.outer - 1}",
        f"            smx = {src}({at} + 1)",
        f"            do smj = 1, {c - 1}",
        f"                if ({src}({at} + smj + 1) > smx) smx = {src}({at} + smj + 1)",
        "            end do",
        "            ssum = 0.0_wp",
        f"            do smj = 0, {c - 1}",
        f"                sexp = exp({src}({at} + smj + 1) - smx)",
        f"                {dst}({at} + smj + 1) = sexp",
        "                ssum = ssum + sexp",
        "            end do",
        f"            do smj = 0, {c - 1}",
        f"                {dst}({at} + smj + 1) = {dst}({at} + smj + 1) / ssum",
        "            end do",
        "        end do",
    ]


def _flat_index_f(names, shape):
    """Row-major flat index, 1-based: the C expression plus one."""
    expr = names[0]
    for k in range(1, len(shape)):
        expr = f"({expr} * {shape[k]} + {names[k]})"
    return f"{expr} + 1"


def _max_counter_rank(plan) -> int:
    """How many c-counters the widest Add or Transpose nest in this model needs."""
    return max((len(op.bcast.out_shape) if op.kind == "add"
                else len(op.pad.out_shape) if op.kind == "pad"
                else len(op.out_shape)
                for op in plan.ops if op.kind in ("add", "transpose", "pad")), default=0)


def _emit_concat_f(op, dst, runtime_names):
    """The Fortran twin of _emit_concat_c; weights are module arrays, named by symbol."""
    from .emit_c import _concat_sources
    cc = op.concat
    srcs = _concat_sources(op, runtime_names, list(op.concat_syms))
    row = sum(cc.blocks)
    L = [f"        do o = 0, {cc.outer - 1}"]
    off = 0
    for name, block in srcs:
        L.append(f"            do i = 1, {block}")
        L.append(f"                {dst}(o * {row} + {off} + i) = {name}(o * {block} + i)")
        L.append("            end do")
        off += block
    L.append("        end do")
    return L


def _emit_transpose_f(op, dst, src):
    """The Fortran twin of _emit_transpose_c."""
    names = [f"c{k}" for k in range(len(op.out_shape))]
    L = [f"        do {nm} = 0, {ext - 1}" for nm, ext in zip(names, op.out_shape)]
    terms = [nm if st == 1 else f"{nm} * {st}" for nm, st in zip(names, op.perm_strides) if st]
    rhs = (" + ".join(terms) + " + 1") if terms else "1"
    L.append(f"            {dst}({_flat_index_f(names, op.out_shape)}) = {src}({rhs})")
    L += ["        end do"] * len(names)
    return L


def _emit_lstm_f(op, dst, src, h0, c0, outs):
    """The Fortran twin of _emit_lstm_c: same recurrence, same ONNX i,o,f,c order."""
    sp = op.lstm
    H, I, B, T = sp.hidden, sp.input_size, sp.batch, sp.seq
    h, c, g = sp.h_sym, sp.c_sym, sp.g_sym
    L = [f"        do i = 1, {B * H}",
         f"            {h}(i) = " + (f"{h0}(i)" if h0 else "0.0_wp"),
         "        end do",
         f"        do i = 1, {B * H}",
         f"            {c}(i) = " + (f"{c0}(i)" if c0 else "0.0_wp"),
         "        end do",
         f"        do lt = 0, {T - 1}",
         f"        do lb = 0, {B - 1}",
         f"            do lk = 0, {4 * H - 1}",
         "                acc = 0.0_wp",
         f"                do j = 0, {I - 1}",
         f"                    acc = acc + {src}((lt * {B} + lb) * {I} + j + 1) * "
         f"{op.weight}(lk * {I} + j + 1)",
         "                end do",
         f"                do j = 0, {H - 1}",
         f"                    acc = acc + {h}(lb * {H} + j + 1) * {op.weight2}(lk * {H} + j + 1)",
         "                end do"]
    if op.bias:
        L.append(f"                acc = acc + {op.bias}(lk + 1) + {op.bias}({4 * H} + lk + 1)")
    L += [f"                {g}(lk + 1) = acc",
          "            end do",
          f"            do j = 0, {H - 1}",
          f"                lgi = {_ACT['sigmoid'].format(v=f'{g}(j + 1)')}",
          f"                lgo = {_ACT['sigmoid'].format(v=f'{g}({H} + j + 1)')}",
          f"                lgf = {_ACT['sigmoid'].format(v=f'{g}({2 * H} + j + 1)')}",
          f"                lgc = {_ACT['tanh'].format(v=f'{g}({3 * H} + j + 1)')}",
          f"                lcn = lgf * {c}(lb * {H} + j + 1) + lgi * lgc",
          f"                {c}(lb * {H} + j + 1) = lcn",
          f"                {h}(lb * {H} + j + 1) = lgo * {_ACT['tanh'].format(v='lcn')}",
          *([f"                {dst}((lt * {B} + lb) * {H} + j + 1) = {h}(lb * {H} + j + 1)"]
            if sp.emit_y else []),
          "            end do",
          "        end do",
          "        end do"]
    # outs is positional: [0] is Y_h and [1] is Y_c, "" for one nothing reads.
    for sym, state in zip(outs, (h, c)):
        if sym:
            L += [f"        do i = 1, {B * H}",
                  f"            {sym}(i) = {state}(i)",
                  "        end do"]
    return L


def _emit_add_f(op, dst, src):
    """The Fortran twin of _emit_add_c. Counters stay 0-based; only the
    subscript gains the +1, so the two emitters compute the same index."""
    bc = op.bcast
    names = [f"c{k}" for k in range(len(bc.out_shape))]
    L = []
    for nm, ext in zip(names, bc.out_shape):
        L.append(f"        do {nm} = 0, {ext - 1}")
    flat = _flat_index_f(names, bc.out_shape)
    terms = [nm if st == 1 else f"{nm} * {st}" for nm, st in zip(names, bc.strides) if st]
    widx = (" + ".join(terms) + " + 1") if terms else "1"
    L.append(f"            {dst}({flat}) = {src}({flat}) + {op.weight}({widx})")
    L += ["        end do"] * len(names)
    return L


def _avgpool_needs_count(op) -> bool:
    """True when AveragePool's divisor is the count of in-bounds cells.

    Otherwise it is a literal and nothing reads the counter -- and a counter
    set but never read is a warning (clang reports it on the C twin; keeping
    the two emitters in step keeps the Fortran one quiet too).
    """
    if op.kind != "avgpool":
        return False
    sp = op.spatial
    return not (sp.every_window_is_inside or sp.count_include_pad)


def _emit_spatial_f(op, dst, src):
    """The Fortran twin of _emit_spatial_c: same loop nest, same arithmetic.

    Buffers are rank-1 here too, so the NCHW index is spelled out. The flat
    index is the C one plus 1 (Fortran is 1-based); the loop counters stay
    0-based so the two emitters read as the same code and so the stride/pad
    arithmetic is identical character for character.

    The weight is the exception: emit_fortran declares it with the ONNX shape
    REVERSED (see _weight_dims), and Fortran fills a column-major array from
    the same C-order value list, so w(kw, kh, ic, oc) -- 1-based -- addresses
    the very element C reaches as w[((oc*IC+ic)*KH+kh)*KW+kw].
    """
    sp = op.spatial
    L = []
    # Grouped Conv: output channel oc belongs to group oc/c_out_per_group and
    # reads only that group's c_in_per_group input channels, so the loop bound
    # is the per-group count and the input channel is offset by the group. For
    # group=1 c_in_per_group == c_in and every expression below collapses to
    # exactly what it was, so an ordinary convolution emits identical code.
    cpg = sp.c_in_per_group or sp.c_in
    # See _emit_spatial_c: hoisted out of the index expression, both because
    # it is invariant there and because gfortran warns about the division.
    in_c = "(icg + ic)" if sp.grouped else "ic"
    idx_in = f"((n * {sp.c_in} + {in_c}) * {sp.h_in} + ih) * {sp.w_in} + iw + 1"
    idx_out = f"((n * {sp.c_out} + oc) * {sp.h_out} + oh) * {sp.w_out} + ow + 1"
    L.append(f"        do n = 0, {sp.n - 1}")
    L.append(f"        do oc = 0, {sp.c_out - 1}")
    L.append(f"        do oh = 0, {sp.h_out - 1}")
    L.append(f"        do ow = 0, {sp.w_out - 1}")
    if op.kind == "conv":
        L.append("            acc = 0.0_wp")
    elif op.kind == "maxpool":
        L.append("            acc = 0.0_wp")
        L.append("            seen = 0")
        L.append("            ic = oc")
    else:
        L.append("            acc = 0.0_wp")
        if _avgpool_needs_count(op):
            L.append("            cnt = 0")
        L.append("            ic = oc")
    if op.kind == "conv":
        if sp.grouped:
            L.append(f"            icg = oc / {sp.c_out_per_group} * {cpg}")
        L.append(f"            do ic = 0, {cpg - 1}")
    L.append(f"            do kh = 0, {sp.kh - 1}")
    L.append(f"            do kw = 0, {sp.kw - 1}")
    L.append(f"                ih = oh * {sp.sh} - {sp.ph} + kh * {sp.dh}")
    L.append(f"                iw = ow * {sp.sw} - {sp.pw} + kw * {sp.dw}")
    L.append(f"                if (ih >= 0 .and. ih < {sp.h_in} .and. "
             f"iw >= 0 .and. iw < {sp.w_in}) then")
    if op.kind == "conv":
        L.append(f"                    acc = acc + {src}({idx_in}) * "
                 f"{op.weight}(kw + 1, kh + 1, ic + 1, oc + 1)")
    elif op.kind == "maxpool":
        L.append(f"                    v = {src}({idx_in})")
        # .not. (v <= acc), not (v > acc): a NaN loses every comparison, so
        # the naive form would drop it. Matches emit_c's !(v <= best).
        L.append("                    if (seen == 0 .or. .not. (v <= acc)) then")
        L.append("                        acc = v")
        L.append("                        seen = 1")
        L.append("                    end if")
    else:
        L.append(f"                    acc = acc + {src}({idx_in})")
        if _avgpool_needs_count(op):
            L.append("                    cnt = cnt + 1")
    L.append("                end if")
    L.append("            end do")
    L.append("            end do")
    if op.kind == "conv":
        L.append("            end do")
        if op.bias:
            L.append(f"            acc = acc + {op.bias}(oc + 1)")
        L.append(f"            {dst}({idx_out}) = acc")
    elif op.kind == "maxpool":
        L.append(f"            {dst}({idx_out}) = acc")
    else:
        full = sp.kh * sp.kw
        if sp.every_window_is_inside or sp.count_include_pad:
            L.append(f"            {dst}({idx_out}) = acc / real({full}, wp)")
        else:
            L.append("            if (cnt > 0) then")
            L.append(f"                {dst}({idx_out}) = acc / real(cnt, wp)")
            L.append("            else")
            L.append(f"                {dst}({idx_out}) = 0.0_wp")
            L.append("            end if")
    L.append("        end do")
    L.append("        end do")
    L.append("        end do")
    L.append("        end do")
    return L


def _emit_infer_batch(plan: Plan) -> list:
    """The OpenMP-target fallback infer_batch: over device-resident arrays.

    Ruling R5: x and y are already on the device (has_device_addr / the
    OpenACC deviceptr twin), so the loop path allocates, maps and transfers
    nothing -- the host maps them itself (`!$omp target enter data`) once,
    outside this call. `has_device_addr` on this explicit-shape dummy
    (rather than the assumed-shape form the spec flags as a compiler risk)
    is accepted, without warning, under gfortran 15 -fopenmp -std=f2008; see
    task-4-report.md for which gfortran this was verified against.

    nvfortran does not implement `has_device_addr` at all -- through 25.11 it
    is a syntax error, not a diagnostic about an unsupported clause -- so the
    module is emitted as .F90 and this one directive is chosen by the
    preprocessor. `is_device_ptr` is what nvfortran accepts for a Fortran
    array holding a device address (its pre-5.1 spelling); it was verified to
    give correct values under nvfortran 25.11 -mp=gpu -gpu=cc80 on an A100,
    via `rosenna gpu-gate`. gfortran keeps the standard 5.1 clause, since
    OpenMP 5.1 restricts Fortran `is_device_ptr` to TYPE(C_PTR) and a future
    gfortran is entitled to reject an array there.
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
        "        ! distribute parallel do, one point per thread: teams loop maps one",
        "        ! point per TEAM under nvfortran and amdflang (see emit_c).",
        "#ifdef __NVCOMPILER",
        "        !$omp target teams distribute parallel do is_device_ptr(x, y)",
        "#else",
        "        !$omp target teams distribute parallel do has_device_addr(x, y)",
        "#endif",
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
    return f"""# Generated by rosenna. Builds lib{n}_f.a from {n}_model.F90 (and {n}_model.mod).
FC ?= gfortran
FFLAGS ?= -O2 -Wall -Wextra -std=f2008
ROSENNA_OFFLOAD_FLAGS ?=

lib{n}_f.a: {n}_model.o
\tar rcs $@ $^
{n}_model.o: {n}_model.F90
\t$(FC) $(FFLAGS) $(ROSENNA_OFFLOAD_FLAGS) -c $< -o $@
clean:
\trm -f {n}_model.o {n}_model.mod lib{n}_f.a
.PHONY: clean
"""
