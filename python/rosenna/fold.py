"""Evaluate at generation time everything that does not depend on the input.

A real ONNX export is full of nodes that only ever see constants: the shape
tensor of a Reshape, a `Constant` holding an LSTM's initial state, a weight
that is transposed once on the way in. None of that belongs in a per-point
inference loop, and some of it could not be emitted at all -- the shape
tensors are int64, a dtype the generated code never carries.

So this pass runs them now. What is left is a graph whose every remaining node
genuinely depends on the input, and whose initializers are all floating-point
arrays the emitters know how to lay out.
"""
import numpy as np

from .errors import UnsupportedModel

# Nodes that are pure functions of their inputs and cheap to evaluate here.
# Anything not listed is left for the emitters, even if its inputs happen to
# all be constant: this list is "what fold knows how to compute", not "what is
# foldable in principle".
FOLDABLE = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Identity", "Concat"}


def _resolve_shape(target, source_shape) -> tuple:
    """ONNX Reshape target semantics: 0 copies the input's dim, -1 is inferred."""
    dims = [int(d) for d in target]
    out = []
    for i, d in enumerate(dims):
        if d == 0:
            if i >= len(source_shape):
                raise UnsupportedModel(f"Reshape: a 0 at axis {i} has no matching input axis")
            out.append(int(source_shape[i]))
        else:
            out.append(d)
    if out.count(-1) > 1:
        raise UnsupportedModel("Reshape: at most one -1 may be inferred")
    if -1 in out:
        known = 1
        for d in out:
            if d != -1:
                known *= d
        total = int(np.prod(source_shape)) if len(source_shape) else 1
        if known == 0 or total % known:
            raise UnsupportedModel(f"Reshape: cannot infer -1 for {tuple(target)} from {tuple(source_shape)}")
        out[out.index(-1)] = total // known
    return tuple(out)


def _axes(node, inits, rank) -> tuple:
    """Squeeze/Unsqueeze axes, from the attribute (opset < 13) or input 1 (>= 13)."""
    if "axes" in node.attrs:
        axes = node.attrs["axes"]
    elif len(node.inputs) > 1 and node.inputs[1]:
        if node.inputs[1] not in inits:
            raise UnsupportedModel(
                f"node '{node.name}': {node.op} axes must be constant")
        axes = inits[node.inputs[1]].tolist()
    else:
        return ()
    return tuple(int(a) % rank if int(a) >= 0 else int(a) + rank for a in axes)


def _evaluate(node, inits):
    """Compute one foldable node's output from constant inputs."""
    a = inits[node.inputs[0]]
    if node.op == "Identity":
        return a
    if node.op == "Concat":
        axis = int(node.attrs.get("axis", 0))
        return np.concatenate([inits[i] for i in node.inputs], axis=axis)
    if node.op == "Reshape":
        if len(node.inputs) < 2 or node.inputs[1] not in inits:
            raise UnsupportedModel(f"node '{node.name}': Reshape needs a constant shape input")
        return a.reshape(_resolve_shape(inits[node.inputs[1]], a.shape))
    if node.op == "Transpose":
        perm = node.attrs.get("perm")
        return np.transpose(a, tuple(int(p) for p in perm) if perm else None)
    if node.op == "Squeeze":
        axes = _axes(node, inits, a.ndim)
        return np.squeeze(a, axis=axes or None)
    if node.op == "Unsqueeze":
        # Negative axes count from the OUTPUT rank (ONNX): with two axes to
        # add, -1 is the last of ndim + 2, not of ndim + 1. Resolve against
        # the output rank, then insert in ascending order so each position
        # is final when it is written.
        n_new = len(_axes(node, inits, a.ndim + 1))
        out = a
        for ax in sorted(_axes(node, inits, a.ndim + n_new)):
            out = np.expand_dims(out, ax)
        return out
    if node.op == "Flatten":
        axis = int(node.attrs.get("axis", 1))
        rows = int(np.prod(a.shape[:axis])) if axis else 1
        return a.reshape(rows, -1)
    raise AssertionError(f"not foldable: {node.op}")


def _constant_value(node):
    """The array a Constant node produces, from whichever attribute carries it."""
    for key in ("value", "value_float", "value_floats", "value_int", "value_ints"):
        if key in node.attrs:
            return np.asarray(node.attrs[key])
    raise UnsupportedModel(f"node '{node.name}': Constant without a value this generator reads")


def fold_constants(graph):
    """Return a graph with every constant-only node evaluated away."""
    from .frontend import Graph          # deferred: frontend imports this module
    inits = dict(graph.initializers)
    values = dict(graph.values)
    nodes = []
    for node in graph.nodes:
        if node.op == "Constant":
            inits[node.outputs[0]] = _constant_value(node)
            values.pop(node.outputs[0], None)
            continue
        if node.op in FOLDABLE and all(i in inits for i in node.inputs if i):
            inits[node.outputs[0]] = np.ascontiguousarray(_evaluate(node, inits))
            values.pop(node.outputs[0], None)
            continue
        nodes.append(node)

    # Drop initializers nothing refers to any more -- the int64 shape tensors
    # a folded Reshape consumed, above all, which no later pass could lay out.
    used = {i for n in nodes for i in n.inputs if i} | set(graph.outputs)
    inits = {k: v for k, v in inits.items() if k in used}
    for name in list(values):
        if name in inits:
            values.pop(name)
    return Graph(graph.name, tuple(nodes), values, inits, graph.inputs, graph.outputs)


def _bn_foldable_into(graph, bn, consumers):
    """The Conv/Gemm this BatchNormalization can be folded into, or None.

    Every condition here is a reason the fold would change the model's meaning
    rather than preserve it, so failing one leaves the BatchNormalization in
    the graph, where validate.py refuses it by name.
    """
    inits = graph.initializers
    if int(bn.attrs.get("training_mode", 0)) != 0:
        return None
    # The running-stat outputs only exist in training mode; a graph that reads
    # one is not doing inference, whatever training_mode says.
    if any(o for o in bn.outputs[1:]):
        return None
    if len(bn.inputs) < 5 or not all(i in inits for i in bn.inputs[1:5]):
        return None

    src = bn.inputs[0]
    producers = [n for n in graph.nodes if src in n.outputs]
    if len(producers) != 1:
        return None
    prod = producers[0]
    if prod.op not in ("Conv", "Gemm"):
        return None
    # Folding rewrites the producer's weights, so anything else reading the
    # pre-BN value would silently start seeing post-BN numbers.
    if len(consumers[src]) != 1 or src in graph.outputs:
        return None
    if len(prod.inputs) < 2 or prod.inputs[1] not in inits:
        return None
    if len(prod.inputs) > 2 and prod.inputs[2] and prod.inputs[2] not in inits:
        return None
    if prod.op == "Gemm":
        # alpha/beta != 1 would not compose with the scale this way. validate
        # refuses them anyway; not folding keeps the message about the real
        # problem instead of about a weight this pass had already rewritten.
        if any(abs(float(prod.attrs.get(a, 1.0)) - 1.0) > 1e-12 for a in ("alpha", "beta")):
            return None
        if int(prod.attrs.get("transA", 0)) != 0:
            return None

    w = inits[prod.inputs[1]]
    n_out = (w.shape[0] if prod.op == "Conv" or int(prod.attrs.get("transB", 0))
             else w.shape[-1])
    if any(inits[i].ndim != 1 or inits[i].shape[0] != n_out for i in bn.inputs[1:5]):
        return None
    return prod


def fold_batchnorm(graph):
    """Fold an inference BatchNormalization into the Conv or Gemm that feeds it.

    At inference a BatchNormalization is an affine map per channel:

        y = scale * (x - mean) / sqrt(var + eps) + B

    and a Conv or Gemm already applies an affine map, so the two compose into
    one: multiply the per-output-channel factor into the weight and push the
    shift through the bias. The operator then disappears before validate.py
    ever sees it, which is why there is no BatchNormalization loop nest in
    either emitter -- the alternative, a runtime op, would read five extra
    arrays per channel to compute what is by then a constant.

        s  = scale / sqrt(var + eps)
        W' = W * s            (broadcast along the output-channel axis)
        b' = (b - mean) * s + B

    A producer with no bias gains one: the shift is not optional, and a new
    initializer is cheaper than a second op.
    """
    from .frontend import Graph, Node
    consumers = {}
    for n in graph.nodes:
        for i in n.inputs:
            if i:
                consumers.setdefault(i, []).append(n)

    folded = {}                     # producer name -> rewritten Node
    drop = set()                    # BatchNormalization nodes removed
    inits = dict(graph.initializers)
    values = dict(graph.values)
    for bn in graph.nodes:
        if bn.op != "BatchNormalization":
            continue
        prod = _bn_foldable_into(graph, bn, consumers)
        if prod is None or prod.name in folded:
            continue

        eps = float(bn.attrs.get("epsilon", 1e-5))
        scale, shift, mean, var = (inits[bn.inputs[k]] for k in (1, 2, 3, 4))
        w = inits[prod.inputs[1]]
        s = (scale / np.sqrt(var + eps)).astype(w.dtype)

        has_bias = len(prod.inputs) > 2 and prod.inputs[2]
        b = inits[prod.inputs[2]] if has_bias else np.zeros(s.shape, w.dtype)
        # Conv weights are [OC, IC, KH, KW] and a transB=1 Gemm's are
        # [OUT, IN]: the output channel leads, so s broadcasts along the
        # trailing axes. A transB=0 Gemm's are [IN, OUT], so it broadcasts
        # along the last.
        lead = prod.op == "Conv" or int(prod.attrs.get("transB", 0))
        s_w = s.reshape((-1,) + (1,) * (w.ndim - 1)) if lead else s.reshape((1,) * (w.ndim - 1) + (-1,))
        inits[prod.inputs[1]] = np.ascontiguousarray(w * s_w)
        bias_name = prod.inputs[2] if has_bias else f"{prod.name}_bn_bias"
        inits[bias_name] = np.ascontiguousarray(
            ((b - mean) * s + shift).astype(w.dtype))

        # The producer takes over the BN's output, so the value between them
        # stops existing.
        values.pop(prod.outputs[0], None)
        folded[prod.name] = Node(prod.op, prod.name,
                                 tuple(prod.inputs[:2]) + (bias_name,),
                                 (bn.outputs[0],) + tuple(prod.outputs[1:]),
                                 prod.attrs)
        drop.add(bn.name)

    if not drop:
        return graph
    nodes = tuple(folded.get(n.name, n) for n in graph.nodes if n.name not in drop)
    used = {i for n in nodes for i in n.inputs if i} | set(graph.outputs)
    inits = {k: v for k, v in inits.items() if k in used}
    return Graph(graph.name, nodes, values, inits, graph.inputs, graph.outputs)


# Shape ops that only relabel axes: on a flat row-major buffer the bytes are
# unchanged, so the value they "produce" is the value they were given. They are
# NOT removed from the graph -- a later Transpose's perm counts axes, so the
# relabelled value has to keep its own shape. plan.py turns them into buffer
# aliases instead, which costs no code and no copy.
RELABEL = {"Reshape", "Squeeze", "Unsqueeze", "Flatten", "Identity"}


def strip_shape_inputs(graph):
    """Drop the metadata operands of relabelling ops, and anything left unused.

    A Reshape whose data is a runtime value cannot be folded, but its shape
    operand is still pure metadata: plan.py turns the node into a buffer alias
    and never reads it. Left in place it would be an int64 initializer nothing
    downstream can lay out, so it goes here -- along with any other initializer
    that no surviving node reads any more.
    """
    from .frontend import Graph, Node

    nodes = tuple(
        Node(n.op, n.name, n.inputs[:1], n.outputs, n.attrs) if n.op in RELABEL else n
        for n in graph.nodes)
    used = {i for n in nodes for i in n.inputs if i} | set(graph.outputs)
    inits = {k: v for k, v in graph.initializers.items() if k in used}
    return Graph(graph.name, nodes, graph.values, inits, graph.inputs, graph.outputs)
