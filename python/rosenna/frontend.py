"""Load an ONNX model into a graph whose shapes are all literal."""
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import numpy_helper, shape_inference

from .errors import UnsupportedModel
from .fold import fold_constants, strip_shape_inputs

_DTYPES = {onnx.TensorProto.FLOAT: "f32", onnx.TensorProto.DOUBLE: "f64"}


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class Node:
    op: str
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict


@dataclass(frozen=True)
class Graph:
    name: str
    nodes: tuple
    values: dict
    initializers: dict
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


def _attr_value(a, node_name: str):
    if a.type == onnx.AttributeProto.INT:
        return int(a.i)
    if a.type == onnx.AttributeProto.FLOAT:
        return float(a.f)
    if a.type == onnx.AttributeProto.STRING:
        try:
            return a.s.decode("ascii")
        except UnicodeDecodeError:
            raise UnsupportedModel(f"node '{node_name}' attribute '{a.name}': non-ASCII string")
    if a.type == onnx.AttributeProto.INTS:
        return tuple(int(v) for v in a.ints)
    if a.type == onnx.AttributeProto.FLOATS:
        return tuple(float(v) for v in a.floats)
    if a.type == onnx.AttributeProto.STRINGS:
        try:
            return tuple(s.decode("ascii") for s in a.strings)
        except UnicodeDecodeError:
            raise UnsupportedModel(f"node '{node_name}' attribute '{a.name}': non-ASCII string")
    if a.type == onnx.AttributeProto.TENSOR:
        # A Constant node's payload. Kept as an array so fold.py can evaluate
        # the node away; nothing downstream of folding ever sees one.
        return numpy_helper.to_array(a.t)
    raise UnsupportedModel(f"node '{node_name}' attribute '{a.name}': unsupported attribute type {a.type}")


def _shape(vi):
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        if not d.HasField("dim_value"):
            raise UnsupportedModel(
                f"value '{vi.name}' has symbolic dimension '{d.dim_param}'; "
                f"roseNNa fixes every shape at generation")
        dims.append(int(d.dim_value))
    return tuple(dims)


def _dtype(vi):
    code = vi.type.tensor_type.elem_type
    if code not in _DTYPES:
        raise UnsupportedModel(f"value '{vi.name}' has element type {code}; only float32 and float64")
    return _DTYPES[code]


def load_graph(path, name: str | None = None) -> Graph:
    path = Path(path)
    model = shape_inference.infer_shapes(onnx.load(path))
    g = model.graph
    initializers = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    values, unsupported_dtype = {}, {}
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.name in initializers:
            continue
        code = vi.type.tensor_type.elem_type
        if code not in _DTYPES:
            # Deferred, not refused: an int64 value is almost always a shape
            # tensor that fold_constants is about to evaluate away. If one
            # survives folding it is a real dtype the emitters cannot carry,
            # and the check below says so then.
            unsupported_dtype[vi.name] = code
            continue
        values[vi.name] = Tensor(vi.name, _shape(vi), _dtype(vi))
    nodes = []
    for i, n in enumerate(g.node):
        node_name = n.name or f"{n.op_type}#{i}"
        nodes.append(Node(
            op=n.op_type,
            name=node_name,
            inputs=tuple(n.input),
            outputs=tuple(n.output),
            attrs={a.name: _attr_value(a, node_name) for a in n.attribute},
        ))
    inputs = tuple(vi.name for vi in g.input if vi.name not in initializers)
    outputs = tuple(vi.name for vi in g.output)
    graph = strip_shape_inputs(fold_constants(
        Graph(name or path.stem, tuple(nodes), values, initializers, inputs, outputs)))
    for n in graph.nodes:
        for v in tuple(n.inputs) + tuple(n.outputs):
            if v in unsupported_dtype:
                raise UnsupportedModel(
                    f"value '{v}' has element type {unsupported_dtype[v]}; "
                    f"only float32 and float64")
    # Defensive check: catch outputs whose names collide with initializers.
    missing = [v for v in graph.inputs + graph.outputs if v not in graph.values]
    if missing:
        raise UnsupportedModel(f"shape inference produced no shape for {missing}")
    return graph
