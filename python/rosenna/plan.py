"""Lower a validated graph into an explicit plan: ops, buffers, weight layout."""
import hashlib
import json
import re
from dataclasses import dataclass, asdict, replace

import numpy as np

from .frontend import Graph, Tensor, UnsupportedModel
from .validate import validate

_ACTIVATIONS = {"Relu": "relu", "Tanh": "tanh", "Sigmoid": "sigmoid"}
_ITEMSIZE = {"f32": 4, "f64": 8}
_NUMPY_DTYPE = {"f32": np.float32, "f64": np.float64}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

# Measured on this machine (gcc-15 -O2 -c on an empty translation unit that
# only #includes the generated header; see tests/measure_embed_threshold.py
# and task-2-report.md for the full table, including an extended sweep past
# 1e6 that locates where the real 5-second crossover falls):
#
#      params    compile time (s)
#        1056           0.05
#       10100           0.06
#      100172           0.12
#      300852           0.25
#     1001000           0.80
#
# Every one of the five measured sizes compiles in under a second, so the
# largest of them -- already a round number -- is the threshold: a model
# under 1,000,000 parameters embeds by default. (The crossover past 5s does
# not occur until several million parameters; see the report for that
# supporting data point.)
EMBED_THRESHOLD = 1_000_000


def validate_model_name(name: str) -> None:
    """Reject a model name that cannot be interpolated into a Fortran/C identifier.

    The name reaches the emitters inside `module <name>_model`, `#ifndef
    ROSENNA_<NAME>_H` and every public function name, and it defaults to the
    ONNX file stem -- which routinely carries dots and hyphens
    (`my-model.v2.onnx`). Catching it here turns a gfortran syntax error on
    generated code into one sentence naming the remedy.
    """
    if not _IDENTIFIER.match(name):
        raise UnsupportedModel(
            f"model name '{name}' is not a valid Fortran/C identifier; it is interpolated "
            f"into module and function names, so it must match [A-Za-z_][A-Za-z0-9_]* -- "
            f"pass --name with a usable name")


@dataclass(frozen=True)
class WeightSpec:
    name: str
    symbol: str
    shape: tuple
    offset: int
    nbytes: int
    # Populated only when the owning Plan embeds its weights (Plan.embed):
    # the flattened values, at the plan's own dtype, that emit_c prints as a
    # ROSENNA_CONST array literal. A file-loaded plan leaves this None; the
    # values live in the .rwt file instead, and this field enters the hash
    # only for an embedded plan (embed is itself part of the hash, so the
    # two forms of the same model are already distinct artifacts).
    values: tuple | None = None


@dataclass(frozen=True)
class Spatial:
    """Everything a 2-D Conv/pool loop nest needs, resolved at generation time.

    The emitters never see an ONNX attribute: auto_pad is already turned into
    begin-pads here (it depends on the input shape, which is literal), and
    every extent is a plain int the emitted loop bounds interpolate directly.
    End-pads are not carried because nothing reads them -- the output extent
    they would determine is taken from ONNX shape inference instead, so a
    window that would run off the end simply never exists.
    """
    n: int
    c_in: int
    h_in: int
    w_in: int
    c_out: int
    h_out: int
    w_out: int
    kh: int
    kw: int
    sh: int
    sw: int
    ph: int
    pw: int
    dh: int
    dw: int
    # AveragePool only: divide by the full kernel (True) or by the count of
    # cells that actually fell inside the input (False, the ONNX default).
    count_include_pad: bool = False
    # Conv only. group=1 is an ordinary convolution; group=c_in with
    # c_out=c_in is a depthwise one. The weight's channel axis is c_in/group,
    # and output channel oc reads only its own group's input channels, so the
    # loop bound and the input-channel offset both change. Carried as the two
    # derived extents the loop actually needs rather than as `group`.
    c_in_per_group: int = 0
    c_out_per_group: int = 0

    @property
    def grouped(self) -> bool:
        return self.c_in_per_group not in (0, self.c_in)

    @property
    def every_window_is_inside(self) -> bool:
        """True when no window reaches past the input on any side.

        The begin pads say whether the first window starts early; the last
        window's reach says whether it runs off the end -- an end-only pad
        (pads=[0,0,1,1]) is exactly that case, and it is not carried here, so
        it has to be read off the output extent. An AveragePool divides every
        window by the full kernel only when this holds (or count_include_pad).
        """
        return (self.ph == 0 and self.pw == 0
                and (self.h_out - 1) * self.sh + (self.kh - 1) * self.dh + 1 <= self.h_in
                and (self.w_out - 1) * self.sw + (self.kw - 1) * self.dw + 1 <= self.w_in)


@dataclass(frozen=True)
class Softmax:
    """A last-axis Softmax, as `outer` independent rows of `axis_len` each.

    The axis is resolved to the trailing one in validate.py, so the emitters
    see a flat [outer, axis_len] view of a buffer that is already row-major
    and never learn that `axis` existed.
    """
    outer: int
    axis_len: int


@dataclass(frozen=True)
class Pad:
    """Constant-mode Pad: where the input block sits inside the output.

    `begins` is one offset per axis; the emitters loop over the output and
    read the input where the shifted index is in range, writing `value`
    everywhere else.
    """
    in_shape: tuple
    out_shape: tuple
    begins: tuple
    value: float
    # "constant", "edge" or "reflect". constant tests the bounds and writes
    # `value` outside them; the other two transform the index instead, so
    # every output element reads some input element and there is no test.
    mode: str = "constant"


@dataclass(frozen=True)
class Broadcast:
    """How a constant operand maps onto the output of an elementwise op.

    `strides` is one entry per output axis: the step to take in the constant's
    flat layout when that axis advances, and 0 where the constant is broadcast
    along it. Resolved here so the emitters write plain affine arithmetic and
    never reason about ranks or alignment.
    """
    out_shape: tuple
    strides: tuple


@dataclass(frozen=True)
class Concat:
    """A Concat resolved to copy extents.

    Row-major, concatenating along `axis` means: for each of `outer` index
    tuples over the axes before it, the output row is the inputs' blocks laid
    end to end, block j being input j's extent along the axis times the
    inner size. `blocks` is one entry per operand, in order; `consts` marks
    which operands are weights (read through their symbol) rather than
    runtime buffers. The emitters write two loops and no shape arithmetic.
    """
    outer: int
    blocks: tuple
    consts: tuple


@dataclass(frozen=True)
class Lstm:
    """A forward LSTM with the ONNX default activations, resolved to extents.

    ONNX orders the gates i, o, f, c in W, R and B -- not the i, f, c, o that
    most papers and most other runtimes use -- and the emitters read the gate
    blocks at those offsets directly, so the order is recorded here once rather
    than rediscovered in two emitters.
    """
    seq: int
    batch: int
    input_size: int
    hidden: int
    has_bias: bool
    has_initial: bool
    # False when nothing reads Y: the recurrence then keeps its state
    # but never stores the per-timestep output, and Y gets no buffer.
    emit_y: bool = True
    # Buffer symbols for the carried state and the per-step gate vector.
    h_sym: str = ""
    c_sym: str = ""
    g_sym: str = ""


@dataclass(frozen=True)
class Op:
    kind: str
    out: str
    inp: str
    weight: str | None
    bias: str | None
    n_in: int
    n_out: int
    # The weight's ONNX layout, straight from the node's transB attribute
    # (MatMul is always 0). Both emitters branch on this. Re-deriving it by
    # comparing the weight's shape against n_out is ambiguous whenever
    # n_in == n_out, and a square weight then gets read transposed.
    trans_b: bool = False
    # Set for kind in ("conv", "maxpool", "avgpool"); None for everything else.
    spatial: Spatial | None = None
    # Set for kind == "add".
    bcast: Broadcast | None = None
    # Set for kind == "softmax".
    softmax: "Softmax | None" = None
    # Set for kind == "pad".
    pad: "Pad | None" = None
    # kind == "concat": the operands are inp (the first runtime one) plus
    # extra_in (the remaining runtime ones) and the weight symbols in
    # concat_syms, interleaved in ONNX input order as concat.consts says.
    concat: "Concat | None" = None
    concat_syms: tuple = ()
    # kind == "lstm": the recurrence weight R (weight/bias carry W and B).
    weight2: str | None = None
    # kind == "lstm": the recurrent shape, and the names of the extra operands
    # and results an LSTM has beyond the single in/out every other op uses.
    lstm: "Lstm | None" = None
    extra_in: tuple = ()
    outs: tuple = ()
    # kind == "lstm": weight symbols holding a constant (initializer) initial
    # hidden and cell state -- the case a folded `Constant` node leaves behind.
    # Exclusive with extra_in, which names them when they are graph values.
    init_syms: tuple = ()
    # kind == "copy": read the source starting this far into its buffer (a
    # secondary graph input's slice of the concatenated x), or write the
    # destination starting this far into its buffer (a secondary graph
    # output's slice of the concatenated y).
    src_offset: int = 0
    dst_offset: int = 0
    # kind == "transpose": per-output-axis stride into the source buffer.
    perm_strides: tuple = ()
    out_shape: tuple = ()
    # kind == "gemm": how many independent rows share the weight. 1 for a dense
    # per-point model; an LSTM whose sequence output feeds a Gemm applies it
    # once per timestep, and the leading axis carries that count.
    rows: int = 1


@dataclass(frozen=True)
class Plan:
    model: str
    dtype: str
    input: Tensor
    output: Tensor
    ops: tuple
    buffers: dict
    assignment: dict
    weights: tuple
    embed: bool
    n_params: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()


def _length(t: Tensor) -> int:
    return int(np.prod(t.shape)) if t.shape else 1


def _weight_elems(shape: tuple) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


_POOL_KIND = {"MaxPool": "maxpool", "AveragePool": "avgpool"}
_RELABEL = {"Reshape", "Squeeze", "Unsqueeze", "Flatten", "Identity"}


def _pair(value, default):
    """An ONNX 2-D attribute as (h, w), defaulting when the attribute is absent."""
    if value is None:
        return default, default
    return int(value[0]), int(value[1])


def _begin_pads(node, in_hw, out_hw, k_hw, s_hw, d_hw):
    """Resolve pads -- explicit, VALID, or auto_pad SAME -- to (begin_h, begin_w).

    Only the begin-pads reach the emitted index arithmetic: `ih = oh*s - ph +
    kh*d`. SAME_UPPER/SAME_LOWER depend on the input extent, which is literal
    here, so the whole auto_pad concept is resolved now and never appears in
    generated code. The total padding SAME needs is derived from the output
    extent ONNX shape inference already computed, so this agrees with the
    reference by construction rather than by re-deriving the formula.
    """
    auto_pad = node.attrs.get("auto_pad", "NOTSET")
    if auto_pad in ("NOTSET", ""):
        pads = node.attrs.get("pads")
        return (0, 0) if pads is None else (int(pads[0]), int(pads[1]))
    if auto_pad == "VALID":
        return 0, 0
    begin = []
    for i in (0, 1):
        span = (k_hw[i] - 1) * d_hw[i] + 1
        total = max(0, (out_hw[i] - 1) * s_hw[i] + span - in_hw[i])
        # SAME_UPPER puts the odd pad at the end, SAME_LOWER at the beginning.
        begin.append(total // 2 if auto_pad == "SAME_UPPER" else (total + 1) // 2)
    return begin[0], begin[1]


def _spatial(graph: Graph, node) -> Spatial:
    """Lower one Conv/MaxPool/AveragePool node to literal loop extents."""
    x = graph.values[node.inputs[0]]
    out = graph.values[node.outputs[0]]
    n, c_in, h_in, w_in = (int(d) for d in x.shape)
    _, c_out, h_out, w_out = (int(d) for d in out.shape)
    if node.op == "Conv":
        w = graph.initializers[node.inputs[1]]
        kh, kw = int(w.shape[2]), int(w.shape[3])
    else:
        kh, kw = _pair(node.attrs.get("kernel_shape"), 1)
    sh, sw = _pair(node.attrs.get("strides"), 1)
    dh, dw = _pair(node.attrs.get("dilations"), 1)
    ph, pw = _begin_pads(node, (h_in, w_in), (h_out, w_out), (kh, kw), (sh, sw), (dh, dw))
    group = int(node.attrs.get("group", 1)) if node.op == "Conv" else 1
    return Spatial(n=n, c_in=c_in, h_in=h_in, w_in=w_in,
                   c_out=c_out, h_out=h_out, w_out=w_out,
                   kh=kh, kw=kw, sh=sh, sw=sw, ph=ph, pw=pw, dh=dh, dw=dw,
                   count_include_pad=bool(int(node.attrs.get("count_include_pad", 0))),
                   c_in_per_group=c_in // group, c_out_per_group=c_out // group)


def _pad_mode(node) -> str:
    mode = node.attrs.get("mode", "constant")
    return mode.decode() if isinstance(mode, bytes) else str(mode)


def _broadcast(out_shape, const_shape) -> Broadcast:
    """Right-align the constant against the output and give each axis a stride."""
    out_shape = tuple(int(d) for d in out_shape)
    const_shape = tuple(int(d) for d in const_shape)
    pad = len(out_shape) - len(const_shape)
    aligned = (1,) * pad + const_shape
    strides, step = [], 1
    for dim in reversed(aligned):
        strides.append(0 if dim == 1 else step)
        step *= dim
    return Broadcast(out_shape, tuple(reversed(strides)))


def _flat_preserving(in_shape, perm) -> bool:
    """True when a Transpose only moves size-1 axes, so the flat bytes are unchanged.

    Row-major order is decided by the axes that actually have extent, in the
    order they appear. Moving a length-1 axis past them changes the shape and
    nothing else -- which is every Transpose a PyTorch LSTM export emits, since
    those only swap the batch axis of a batch-1 model.
    """
    kept = [a for a in perm if in_shape[a] != 1]
    return kept == sorted(kept)


def lstm_initial_state(op, weight_ref, assignment) -> tuple:
    """(h0, c0) array names for an LSTM op, for either emitter.

    `weight_ref` renders a weight symbol the way that emitter spells it;
    `assignment` maps a value name to its buffer. Constant states are weights,
    caller-supplied ones are buffers, and an LSTM without them gets None.
    """
    if op.init_syms:
        return tuple(weight_ref(sym) for sym in op.init_syms)
    return (assignment[op.extra_in[0]] if op.extra_in else None,
            assignment[op.extra_in[1]] if len(op.extra_in) > 1 else None)


def _transpose_strides(in_shape, perm) -> tuple:
    """Per-output-axis stride into the source's flat layout."""
    src_stride, step = [0] * len(in_shape), 1
    for axis in reversed(range(len(in_shape))):
        src_stride[axis] = step
        step *= in_shape[axis]
    return tuple(src_stride[a] for a in perm)


def _lstm_spec(graph: Graph, node) -> Lstm:
    x = graph.values[node.inputs[0]]
    seq, batch, input_size = (int(d) for d in x.shape)
    hidden = int(node.attrs["hidden_size"]) if "hidden_size" in node.attrs else \
        int(graph.initializers[node.inputs[2]].shape[2])
    has_bias = len(node.inputs) > 3 and bool(node.inputs[3])
    has_initial = len(node.inputs) > 5 and bool(node.inputs[5])
    return Lstm(seq=seq, batch=batch, input_size=input_size, hidden=hidden,
                has_bias=has_bias, has_initial=has_initial)


def build_plan(graph: Graph, dtype: str | None = None, embed: bool | None = None) -> Plan:
    validate(graph)
    if len(graph.inputs) < 1 or len(graph.outputs) < 1:
        raise UnsupportedModel(
            f"this generator needs at least one input and one output; "
            f"got {len(graph.inputs)} inputs and {len(graph.outputs)} outputs")
    dtype = dtype or graph.values[graph.inputs[0]].dtype
    if dtype not in _ITEMSIZE:
        raise UnsupportedModel(f"dtype {dtype} is not supported")

    consumed = {i for n in graph.nodes for i in n.inputs if i} | set(graph.outputs)
    ops, weights, offset, widx = [], [], 0, 0
    by_name = {}

    def weight(name: str, symbol: str, shape: tuple) -> str:
        """Register initializer `name` once and return its symbol.

        A second node using the same initializer (a tied weight) gets the
        first node's symbol: one WeightSpec, one file entry, one array. The
        loaders match file entries by name, so a duplicate spec was filled
        once in C (the other stayed zero) and was a duplicate CASE in Fortran.
        """
        nonlocal offset
        if name in by_name:
            prior = by_name[name]
            if prior.shape != shape:
                raise UnsupportedModel(
                    f"initializer '{name}' is used by two nodes that need it declared "
                    f"with different shapes ({prior.shape} and {shape})")
            return prior.symbol
        a = graph.initializers[name]
        spec = WeightSpec(name, symbol, shape, offset, a.size * _ITEMSIZE[dtype])
        weights.append(spec)
        by_name[name] = spec
        offset += spec.nbytes
        return spec.symbol
    for node in graph.nodes:
        if node.op in _ACTIVATIONS:
            # n_in/n_out carry the activation's OWN length, taken from the
            # value it produces. The emitters used to bound an activation's
            # loop by a running "length of the previous op's output", which is
            # the same number only in an unbranched chain: give a value a
            # second consumer and the later read gets the wider op's bound,
            # running off the end of a fixed-size local array in both
            # directions. plan.py knows the real length here, so it says it.
            act_len = _length(graph.values[node.outputs[0]])
            ops.append(Op(_ACTIVATIONS[node.op], node.outputs[0], node.inputs[0],
                          None, None, act_len, act_len))
            continue
        if node.op == "Pad":
            in_shape = tuple(int(d) for d in graph.values[node.inputs[0]].shape)
            out_shape = tuple(int(d) for d in graph.values[node.outputs[0]].shape)
            pads = tuple(int(v) for v in node.attrs["pads"])
            ops.append(Op("pad", node.outputs[0], node.inputs[0], None, None,
                          _length(graph.values[node.inputs[0]]),
                          _length(graph.values[node.outputs[0]]),
                          pad=Pad(in_shape, out_shape, pads[:len(in_shape)],
                                  float(node.attrs.get("value", 0.0)),
                                  _pad_mode(node))))
            continue
        if node.op == "Softmax":
            shape = tuple(int(d) for d in graph.values[node.outputs[0]].shape)
            axis_len = shape[-1]
            n = _length(graph.values[node.outputs[0]])
            ops.append(Op("softmax", node.outputs[0], node.inputs[0], None, None, n, n,
                          softmax=Softmax(outer=n // axis_len, axis_len=axis_len)))
            continue
        if node.op in _RELABEL or (node.op == "Transpose" and _flat_preserving(
                graph.values[node.inputs[0]].shape,
                node.attrs.get("perm", tuple(reversed(range(len(graph.values[node.inputs[0]].shape))))))):
            # Relabels the axes without moving a byte. It becomes a buffer
            # alias -- no code, no copy -- unless it produces the graph output,
            # which has to land in the caller's own y.
            n = _length(graph.values[node.outputs[0]])
            kind = "copy" if node.outputs[0] in graph.outputs else "alias"
            ops.append(Op(kind, node.outputs[0], node.inputs[0], None, None, n, n))
            continue
        if node.op == "Transpose":
            in_t = graph.values[node.inputs[0]]
            out_t = graph.values[node.outputs[0]]
            perm = node.attrs.get("perm", tuple(reversed(range(len(in_t.shape)))))
            ops.append(Op("transpose", node.outputs[0], node.inputs[0], None, None,
                          _length(in_t), _length(out_t),
                          perm_strides=_transpose_strides(
                              tuple(int(d) for d in in_t.shape), tuple(int(p) for p in perm)),
                          out_shape=tuple(int(d) for d in out_t.shape)))
            continue
        if node.op == "LSTM":
            spec = _lstm_spec(graph, node)
            syms = {}
            for role, idx in (("weight", 1), ("weight2", 2), ("bias", 3)):
                if idx < len(node.inputs) and node.inputs[idx]:
                    a = graph.initializers[node.inputs[idx]]
                    sym = f"{'w' if role != 'bias' else 'b'}{widx}{'r' if role == 'weight2' else ''}"
                    syms[role] = weight(node.inputs[idx], sym, (int(a.size),))
            states = tuple(i for i in node.inputs[5:7] if i) if len(node.inputs) > 5 else ()
            if states and states[0] in graph.initializers:
                # Constant initial state (validate checked both are): a weight
                # each, flat, read like any other by the emitters.
                init_syms = tuple(weight(nm, f"{role}{widx}", (int(graph.initializers[nm].size),))
                                  for role, nm in zip(("h", "c"), states))
                extra = ()
            else:
                init_syms, extra = (), states
            # Only the outputs something downstream reads. An LSTM always
            # produces Y, Y_h and Y_c, and a model typically wants one of
            # them; carrying the others cost a buffer and a dead copy per
            # call -- per thread, on a device -- and left the generated code
            # warning on any compiler asked to look.
            #
            # POSITIONAL: index 0 is always Y_h and index 1 always Y_c, and a
            # dropped one is "" rather than absent. Compacting the tuple moves
            # Y_c into Y_h's slot, and the emitters copy h into it.
            tail = tuple(node.outputs[1:3]) + ("",) * (2 - len(node.outputs[1:3]))
            outs = tuple(o if (o and o in consumed) else "" for o in tail)
            spec = replace(spec, emit_y=node.outputs[0] in consumed)
            ops.append(Op("lstm", node.outputs[0], node.inputs[0],
                          syms.get("weight"), syms.get("bias"),
                          _length(graph.values[node.inputs[0]]),
                          _length(graph.values[node.outputs[0]]),
                          weight2=syms.get("weight2"), lstm=spec,
                          extra_in=extra, outs=outs, init_syms=init_syms))
            widx += 1
            continue
        if node.op == "Concat":
            shapes = [tuple(int(d) for d in (graph.initializers[i].shape if i in graph.initializers
                                              else graph.values[i].shape)) for i in node.inputs]
            rank = len(shapes[0])
            axis = int(node.attrs.get("axis", 0))
            axis = axis + rank if axis < 0 else axis
            outer = int(np.prod(shapes[0][:axis])) if axis else 1
            inner = int(np.prod(shapes[0][axis + 1:])) if axis + 1 < rank else 1
            blocks = tuple(sh[axis] * inner for sh in shapes)
            consts = tuple(i in graph.initializers for i in node.inputs)
            syms = []
            for i in node.inputs:
                if i in graph.initializers:
                    syms.append(weight(i, f"w{widx}_{len(syms)}", (int(graph.initializers[i].size),)))
            runtime = [i for i in node.inputs if i not in graph.initializers]
            if not runtime:
                raise UnsupportedModel(f"node '{node.name}': all-constant Concat should have folded")
            out_t = graph.values[node.outputs[0]]
            ops.append(Op("concat", node.outputs[0], runtime[0], None, None,
                          _length(graph.values[runtime[0]]), _length(out_t),
                          extra_in=tuple(runtime[1:]),
                          concat=Concat(outer=outer, blocks=blocks, consts=consts),
                          concat_syms=tuple(syms)))
            if syms:
                widx += 1
            continue
        if node.op == "Add":
            const_name = next(i for i in node.inputs if i in graph.initializers)
            src_name = next(i for i in node.inputs if i not in graph.initializers)
            c = graph.initializers[const_name]
            # Declared flat, not with the ONNX shape: the broadcast strides are
            # offsets into the constant's row-major flat layout, so a rank-1
            # declaration is what both emitters subscript. (emit_fortran would
            # otherwise declare a rank-3 (1,1,8) array and reject the single
            # subscript the stride arithmetic produces.)
            csym = weight(const_name, f"w{widx}", (int(c.size),))
            out_t = graph.values[node.outputs[0]]
            ops.append(Op("add", node.outputs[0], src_name, csym, None,
                          _length(out_t), _length(out_t),
                          bcast=_broadcast(out_t.shape, c.shape)))
            widx += 1
            continue
        if node.op in _POOL_KIND:
            sp = _spatial(graph, node)
            ops.append(Op(_POOL_KIND[node.op], node.outputs[0], node.inputs[0], None, None,
                          _length(graph.values[node.inputs[0]]),
                          _length(graph.values[node.outputs[0]]), spatial=sp))
            continue
        if node.op == "Conv":
            sp = _spatial(graph, node)
            w = graph.initializers[node.inputs[1]]
            wsym = weight(node.inputs[1], f"w{widx}", tuple(int(d) for d in w.shape))
            bsym = None
            if len(node.inputs) > 2 and node.inputs[2]:
                b = graph.initializers[node.inputs[2]]
                bsym = weight(node.inputs[2], f"b{widx}", tuple(int(d) for d in b.shape))
            ops.append(Op("conv", node.outputs[0], node.inputs[0], wsym, bsym,
                          _length(graph.values[node.inputs[0]]),
                          _length(graph.values[node.outputs[0]]), spatial=sp))
            widx += 1
            continue
        w = graph.initializers[node.inputs[1]]
        trans_b = int(node.attrs.get("transB", 0)) if node.op == "Gemm" else 0
        n_out, n_in = (w.shape[0], w.shape[1]) if trans_b else (w.shape[1], w.shape[0])
        wsym = weight(node.inputs[1], f"w{widx}", tuple(int(d) for d in w.shape))
        bsym = None
        if node.op == "Gemm" and len(node.inputs) > 2:
            b = graph.initializers[node.inputs[2]]
            bsym = weight(node.inputs[2], f"b{widx}", tuple(int(d) for d in b.shape))
        in_len = _length(graph.values[node.inputs[0]])
        if in_len % int(n_in):
            raise UnsupportedModel(
                f"node '{node.name}': input holds {in_len} values, not a whole number of "
                f"rows of {n_in}")
        ops.append(Op("gemm", node.outputs[0], node.inputs[0], wsym, bsym,
                      int(n_in), int(n_out), bool(trans_b), rows=in_len // int(n_in)))
        widx += 1

    # One entry point, one input buffer, one output buffer: a model with
    # several graph inputs (an LSTM's initial hidden and cell state, say)
    # takes them concatenated in declaration order, each secondary input
    # copied out of its slice of x below; a model with several graph outputs
    # (that LSTM's Y, Y_h and Y_c) writes them concatenated in declaration
    # order, each secondary output copied into its slice of y after the last
    # op. Keeping infer(x, y) intact is what keeps infer_batch, the native
    # kernel, the weights ABI and the whole device contract unchanged, and
    # is what lets a solver keep a recurrent model's state resident: y's
    # h'/c' slices go straight back into x's h/c slices next step.
    in_lens = [_length(graph.values[n]) for n in graph.inputs]
    out_lens = [_length(graph.values[n]) for n in graph.outputs]
    in_t = graph.values[graph.inputs[0]]
    out_t = graph.values[graph.outputs[0]]
    flat_in = Tensor(in_t.name, (sum(in_lens),), dtype)
    flat_out = Tensor(out_t.name, (sum(out_lens),), dtype)
    slice_ops, off = [], in_lens[0]
    for name, n in zip(graph.inputs[1:], in_lens[1:]):
        slice_ops.append(Op("copy", name, graph.inputs[0], None, None, n, n, src_offset=off))
        off += n
    gather_ops, off = [], out_lens[0]
    for name, n in zip(graph.outputs[1:], out_lens[1:]):
        gather_ops.append(Op("copy", f"{name}->y", name, None, None, n, n, dst_offset=off))
        off += n
    ops = slice_ops + ops + gather_ops
    buffers, assignment = _assign_buffers(graph, ops, flat_in, flat_out)

    n_params = sum(_weight_elems(w.shape) for w in weights)
    if embed is None:
        embed = n_params < EMBED_THRESHOLD
    if embed:
        np_dtype = _NUMPY_DTYPE[dtype]
        weights = [
            WeightSpec(w.name, w.symbol, w.shape, w.offset, w.nbytes,
                       values=tuple(np.asarray(graph.initializers[w.name], dtype=np_dtype)
                                    .ravel(order="C").tolist()))
            for w in weights
        ]

    return Plan(graph.name, dtype, flat_in, flat_out, tuple(ops), buffers, assignment,
                tuple(weights), embed, n_params)


def op_outputs(op) -> tuple:
    """The values this op actually writes: no dead LSTM Y, no dropped Y_h/Y_c."""
    head = () if (op.kind == "lstm" and not op.lstm.emit_y) else (op.out,)
    return head + tuple(o for o in op.outs if o)


def _assign_buffers(graph: Graph, ops, flat_in: Tensor, flat_out: Tensor):
    """Give the input and output dedicated buffers; rotate intermediates through a pool.

    An "alias" op contributes no buffer of its own: it hands its output the
    symbol its input already holds, because a relabelling op moves no bytes.
    Liveness is therefore tracked on the *root* of an alias chain, so a buffer
    is only returned to the free list after the last read of anything that
    shares it -- reading through an alias counts.

    Multiple consumers of one value are fine: last_use records the last op that
    reads it, not the first.
    """
    alias = {op.out: op.inp for op in ops if op.kind == "alias"}

    def root(name):
        seen = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    buffers = {"x": flat_in.shape[0], "y": flat_out.shape[0]}
    assignment = {flat_in.name: "x", flat_out.name: "y"}
    # A secondary graph output's gather copy writes into y at its offset.
    for op in ops:
        if op.kind == "copy" and op.dst_offset:
            assignment[op.out] = "y"
    last_use = {}
    for i, op in enumerate(ops):
        for src in (op.inp,) + tuple(op.extra_in):
            last_use[root(src)] = i
    free, pool = [], 0

    def take(length):
        nonlocal pool
        if free:
            return free.pop()
        sym = f"t{pool}"
        pool += 1
        return sym

    for i, op in enumerate(ops):
        if op.kind == "alias":
            assignment[op.out] = assignment[root(op.inp)]
            continue
        if op.kind == "copy" and op.dst_offset:
            # A gather into y: its destination is y itself (assigned above), and
            # y's length is already the whole concatenated output.
            continue
        for out in op_outputs(op):
            if out not in assignment:
                sym = take(0)
                assignment[out] = sym
            length = _length(graph.values[out])
            buffers[assignment[out]] = max(buffers.get(assignment[out], 0), length)
        if op.kind == "lstm":
            # Carried state and the per-step gate vector: internal to the op,
            # so they get their own buffers rather than sharing the pool (they
            # stay live across the whole sequence loop).
            sp = op.lstm
            for role, size in (("h_sym", sp.batch * sp.hidden),
                               ("c_sym", sp.batch * sp.hidden),
                               ("g_sym", 4 * sp.hidden)):
                sym = f"t{pool}"
                pool += 1
                buffers[sym] = size
                object.__setattr__(sp, role, sym)
        for src in (op.inp,) + tuple(op.extra_in):
            r = root(src)
            if r in assignment and assignment[r].startswith("t") and last_use.get(r) == i:
                free.append(assignment[r])
    return buffers, assignment
