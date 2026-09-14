"""Reject anything this generator cannot lower, naming the node."""
import numpy as np
from .frontend import Graph, UnsupportedModel

# Dense ops (rank 1-2), the 2-D spatial ops (rank 4), LSTM, and the
# relabelling ops plan.py turns into buffer aliases. Everything here is
# lowered by plan.py into explicit loop nests over flat buffers; an op that is
# not here is refused by name rather than silently mis-lowered.
SUPPORTED = {"Gemm", "MatMul", "Relu", "Tanh", "Sigmoid",
             "Conv", "MaxPool", "AveragePool", "Add", "Transpose", "LSTM",
             # Relabelling ops: fold.resolve_shape_ops deletes these outright
             # unless one produces the graph output, where it becomes a copy.
             "Reshape", "Squeeze", "Unsqueeze", "Flatten", "Identity"}

# The spatial ops are 2-D only: kernel_shape, strides, pads and dilations all
# have to describe exactly two spatial axes, which is what a rank-4 NCHW value
# carries. A 1-D or 3-D convolution would need a different loop nest.
_SPATIAL = {"Conv", "MaxPool", "AveragePool"}
MAX_RANK = 4


def validate(graph: Graph) -> None:
    for node in graph.nodes:
        if node.op not in SUPPORTED:
            raise UnsupportedModel(
                f"node '{node.name}': {node.op} is not supported; "
                f"this generator handles {sorted(SUPPORTED)}")
        if node.op == "Gemm":
            _validate_gemm(graph, node)
        if node.op == "MatMul":
            if len(node.inputs) < 2:
                raise UnsupportedModel(
                    f"node '{node.name}': MatMul requires at least 2 inputs")
            if node.inputs[1] not in graph.initializers:
                raise UnsupportedModel(
                    f"node '{node.name}': MatMul needs a constant second input; "
                    f"'{node.inputs[1]}' is computed at runtime")
            rhs = graph.initializers[node.inputs[1]]
            if rhs.ndim != 2:
                raise UnsupportedModel(
                    f"node '{node.name}': MatMul weight '{node.inputs[1]}' has rank "
                    f"{rhs.ndim}; only rank 2 is supported")
        if node.op == "Add":
            _validate_add(graph, node)
        if node.op == "LSTM":
            _validate_lstm(graph, node)
        if node.op in _SPATIAL:
            _validate_spatial(graph, node)
    for name, t in graph.values.items():
        if not 1 <= len(t.shape) <= MAX_RANK:
            raise UnsupportedModel(
                f"value '{name}' has rank {len(t.shape)}; "
                f"this generator handles rank 1 to {MAX_RANK}")
    for name, init in graph.initializers.items():
        if not 1 <= init.ndim <= MAX_RANK:
            raise UnsupportedModel(
                f"initializer '{name}' has rank {init.ndim}; "
                f"this generator handles rank 1 to {MAX_RANK}")
        if not np.issubdtype(init.dtype, np.floating):
            raise UnsupportedModel(
                f"initializer '{name}' has dtype {init.dtype}; only floating-point types are supported")


def _validate_gemm(graph: Graph, node) -> None:
    if len(node.inputs) < 2:
        raise UnsupportedModel(
            f"node '{node.name}': Gemm requires at least 2 inputs")
    if int(node.attrs.get("transA", 0)) != 0:
        raise UnsupportedModel(f"node '{node.name}': Gemm transA=1 is not supported")
    for attr in ("alpha", "beta"):
        value = float(node.attrs.get(attr, 1.0))
        if abs(value - 1.0) > 1e-12:
            raise UnsupportedModel(f"node '{node.name}': Gemm {attr}={value} is not supported; only 1.0")
    if node.inputs[1] not in graph.initializers:
        raise UnsupportedModel(
            f"node '{node.name}': Gemm weight '{node.inputs[1]}' must be a constant")
    # Rank is checked per op, not globally: the global initializer bound had to
    # widen to 4 for Conv's OCxICxKHxKW kernels, and a rank-3 Gemm weight would
    # otherwise slip through and be read as if it were a matrix.
    gw = graph.initializers[node.inputs[1]]
    if gw.ndim != 2:
        raise UnsupportedModel(
            f"node '{node.name}': Gemm weight '{node.inputs[1]}' has rank {gw.ndim}; "
            f"only rank 2 is supported")
    if len(node.inputs) > 2:
        bias = graph.initializers.get(node.inputs[2])
        if bias is None:
            raise UnsupportedModel(f"node '{node.name}': Gemm bias '{node.inputs[2]}' must be a constant")
        if bias.ndim != 1:
            raise UnsupportedModel(
                f"node '{node.name}': Gemm bias has rank {bias.ndim}; only rank 1 is supported")
        # The emitters index b[i] for every output: a (1,) bias, though a
        # legal ONNX broadcast, would be read past its end.
        n_out = gw.shape[0] if int(node.attrs.get("transB", 0)) else gw.shape[1]
        if bias.shape[0] != n_out:
            raise UnsupportedModel(
                f"node '{node.name}': Gemm bias has {bias.shape[0]} values for {n_out} outputs; "
                f"a broadcast bias is not supported")


def _validate_spatial(graph: Graph, node) -> None:
    """Conv, MaxPool and AveragePool: 2-D spatial, rank-4 NCHW, no exotic attributes.

    Every attribute this refuses is one whose meaning the emitted loop nest
    does not implement -- so a model using it would otherwise get plausible
    numbers that are wrong, which is the failure mode this whole file exists
    to prevent.
    """
    where = f"node '{node.name}'"
    x = graph.values.get(node.inputs[0])
    if x is None:
        raise UnsupportedModel(f"{where}: input '{node.inputs[0]}' has no inferred shape")
    if len(x.shape) != 4:
        raise UnsupportedModel(
            f"{where}: {node.op} input has rank {len(x.shape)}; only rank-4 NCHW is supported")
    out = graph.values.get(node.outputs[0])
    if out is None or len(out.shape) != 4:
        raise UnsupportedModel(f"{where}: {node.op} output must be a rank-4 value with an inferred shape")

    kernel = node.attrs.get("kernel_shape")
    if node.op == "Conv" and kernel is None:
        w = graph.initializers.get(node.inputs[1])
        kernel = tuple(int(d) for d in w.shape[2:]) if w is not None else None
    if kernel is None:
        raise UnsupportedModel(f"{where}: {node.op} needs kernel_shape")
    if len(kernel) != 2:
        raise UnsupportedModel(
            f"{where}: kernel_shape has {len(kernel)} spatial axes; only 2-D is supported")

    for attr in ("strides", "dilations"):
        v = node.attrs.get(attr)
        if v is not None and len(v) != 2:
            raise UnsupportedModel(f"{where}: {attr} has {len(v)} entries; only 2-D is supported")
    pads = node.attrs.get("pads")
    if pads is not None and len(pads) != 4:
        raise UnsupportedModel(
            f"{where}: pads has {len(pads)} entries; a 2-D op takes 4 "
            f"(begin_h, begin_w, end_h, end_w)")
    auto_pad = node.attrs.get("auto_pad", "NOTSET")
    if auto_pad not in ("NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"):
        raise UnsupportedModel(f"{where}: auto_pad='{auto_pad}' is not supported")
    if auto_pad != "NOTSET" and pads is not None and any(pads):
        raise UnsupportedModel(
            f"{where}: auto_pad='{auto_pad}' with an explicit non-zero pads is ambiguous")
    if int(node.attrs.get("ceil_mode", 0)) != 0:
        raise UnsupportedModel(f"{where}: ceil_mode=1 is not supported")

    if node.op == "Conv":
        if len(node.inputs) < 2 or node.inputs[1] not in graph.initializers:
            raise UnsupportedModel(f"{where}: Conv weight must be a constant initializer")
        w = graph.initializers[node.inputs[1]]
        if w.ndim != 4:
            raise UnsupportedModel(f"{where}: Conv weight has rank {w.ndim}; only rank 4 is supported")
        group = int(node.attrs.get("group", 1))
        if group != 1:
            raise UnsupportedModel(f"{where}: grouped Conv (group={group}) is not supported")
        if x.shape[1] != w.shape[1]:
            raise UnsupportedModel(
                f"{where}: Conv input has {x.shape[1]} channels but the weight expects {w.shape[1]}")
        if len(node.inputs) > 2 and node.inputs[2]:
            b = graph.initializers.get(node.inputs[2])
            if b is None:
                raise UnsupportedModel(f"{where}: Conv bias '{node.inputs[2]}' must be a constant")
            if b.ndim != 1 or b.shape[0] != w.shape[0]:
                raise UnsupportedModel(
                    f"{where}: Conv bias must be rank 1 with one value per output channel")
    else:
        if len(node.outputs) > 1 and node.outputs[1]:
            raise UnsupportedModel(
                f"{where}: {node.op} with a second (indices) output is not supported")
        if int(node.attrs.get("storage_order", 0)) != 0:
            raise UnsupportedModel(f"{where}: MaxPool storage_order=1 (column major) is not supported")


def _validate_add(graph: Graph, node) -> None:
    """Add of a runtime value and a constant, broadcast right-aligned.

    Two runtime operands would need two live buffers reaching one op, which the
    single-input Op model does not carry; a constant operand is the shape that
    actually turns up (a per-channel bias a Conv export did not fold in).
    """
    where = f"node '{node.name}'"
    if len(node.inputs) != 2:
        raise UnsupportedModel(f"{where}: Add takes exactly 2 inputs")
    runtime = [i for i in node.inputs if i not in graph.initializers]
    if len(runtime) != 1:
        raise UnsupportedModel(
            f"{where}: Add needs exactly one runtime operand and one constant; "
            f"got {len(runtime)} runtime")
    out = graph.values.get(node.outputs[0])
    x = graph.values.get(runtime[0])
    if out is None or x is None:
        raise UnsupportedModel(f"{where}: Add operands must have inferred shapes")
    if tuple(out.shape) != tuple(x.shape):
        raise UnsupportedModel(
            f"{where}: Add broadcasts its runtime operand from {tuple(x.shape)} to "
            f"{tuple(out.shape)}; only the constant operand may broadcast")
    const = graph.initializers[[i for i in node.inputs if i in graph.initializers][0]]
    if const.ndim > len(out.shape):
        raise UnsupportedModel(
            f"{where}: Add constant has rank {const.ndim}, wider than the output's "
            f"{len(out.shape)}")
    for axis, (o, c) in enumerate(zip(out.shape[len(out.shape) - const.ndim:], const.shape)):
        if c not in (1, o):
            raise UnsupportedModel(
                f"{where}: Add constant axis {axis} has extent {c}, which neither "
                f"matches the output's {o} nor broadcasts")


def _validate_lstm(graph: Graph, node) -> None:
    """Forward-direction LSTM with the ONNX default activations and no clipping.

    Everything refused here changes the recurrence itself, so a model using it
    would run and return confident nonsense.
    """
    where = f"node '{node.name}'"
    direction = node.attrs.get("direction", "forward")
    if direction != "forward":
        raise UnsupportedModel(f"{where}: direction='{direction}'; only 'forward' is supported")
    if "activations" in node.attrs:
        raise UnsupportedModel(
            f"{where}: custom activations are not supported; only the defaults "
            f"(sigmoid on the gates, tanh on the cell and the output)")
    for attr in ("clip", "input_forget", "layout"):
        if node.attrs.get(attr):
            raise UnsupportedModel(f"{where}: {attr}={node.attrs[attr]} is not supported")
    if len(node.inputs) > 4 and node.inputs[4]:
        raise UnsupportedModel(f"{where}: a sequence_lens input is not supported")
    if len(node.inputs) > 7 and node.inputs[7]:
        raise UnsupportedModel(f"{where}: peephole weights (input P) are not supported")
    x = graph.values.get(node.inputs[0])
    if x is None or len(x.shape) != 3:
        raise UnsupportedModel(
            f"{where}: LSTM input must be a rank-3 (seq, batch, input_size) value")
    for idx, role in ((1, "W"), (2, "R")):
        if idx >= len(node.inputs) or node.inputs[idx] not in graph.initializers:
            raise UnsupportedModel(f"{where}: LSTM {role} must be a constant initializer")
        a = graph.initializers[node.inputs[idx]]
        if a.ndim != 3 or a.shape[0] != 1:
            raise UnsupportedModel(
                f"{where}: LSTM {role} must have shape (1, 4*hidden, k); got {tuple(a.shape)}")
    if len(node.inputs) > 3 and node.inputs[3]:
        b = graph.initializers.get(node.inputs[3])
        if b is None or b.ndim != 2 or b.shape[0] != 1:
            raise UnsupportedModel(f"{where}: LSTM B must be a constant of shape (1, 8*hidden)")
    # The initial state is either a graph value (the caller supplies it, in
    # x) or an initializer (a folded Constant: it becomes a weight); either
    # way rank 3, and both of the pair the same way.
    kinds = set()
    for idx, role in ((5, "initial_h"), (6, "initial_c")):
        if len(node.inputs) > idx and node.inputs[idx]:
            name = node.inputs[idx]
            if name in graph.initializers:
                shape, kinds = graph.initializers[name].shape, kinds | {"initializer"}
            else:
                v = graph.values.get(name)
                shape, kinds = (v.shape if v is not None else ()), kinds | {"value"}
            if len(shape) != 3:
                raise UnsupportedModel(
                    f"{where}: {role} must be a rank-3 (num_directions, batch, hidden) "
                    f"value or initializer")
    if (len(node.inputs) > 5 and bool(node.inputs[5])) != (len(node.inputs) > 6 and bool(node.inputs[6])):
        raise UnsupportedModel(
            f"{where}: initial_h and initial_c must be supplied together or not at all")
    if len(kinds) > 1:
        raise UnsupportedModel(
            f"{where}: initial_h and initial_c must both be values or both be initializers")
