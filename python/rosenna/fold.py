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
FOLDABLE = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Identity"}


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
