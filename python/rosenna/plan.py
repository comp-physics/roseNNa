"""Lower a validated graph into an explicit plan: ops, buffers, weight layout."""
import hashlib
import json
from dataclasses import dataclass, asdict

import numpy as np

from .frontend import Graph, Tensor, UnsupportedModel
from .validate import validate

_ACTIVATIONS = {"Relu": "relu", "Tanh": "tanh", "Sigmoid": "sigmoid"}
_ITEMSIZE = {"f32": 4, "f64": 8}


@dataclass(frozen=True)
class WeightSpec:
    name: str
    symbol: str
    shape: tuple
    offset: int
    nbytes: int


@dataclass(frozen=True)
class Op:
    kind: str
    out: str
    inp: str
    weight: str | None
    bias: str | None
    n_in: int
    n_out: int


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

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=list)

    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()


def _length(t: Tensor) -> int:
    return int(np.prod(t.shape)) if t.shape else 1


def build_plan(graph: Graph, dtype: str | None = None) -> Plan:
    validate(graph)
    dtype = dtype or graph.values[graph.inputs[0]].dtype
    if dtype not in _ITEMSIZE:
        raise UnsupportedModel(f"dtype {dtype} is not supported")
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise UnsupportedModel(
            f"this generator handles one input and one output; "
            f"got {len(graph.inputs)} and {len(graph.outputs)}")

    ops, weights, offset, widx = [], [], 0, 0
    for node in graph.nodes:
        if node.op in _ACTIVATIONS:
            ops.append(Op(_ACTIVATIONS[node.op], node.outputs[0], node.inputs[0], None, None, 0, 0))
            continue
        w = graph.initializers[node.inputs[1]]
        trans_b = int(node.attrs.get("transB", 0)) if node.op == "Gemm" else 0
        n_out, n_in = (w.shape[0], w.shape[1]) if trans_b else (w.shape[1], w.shape[0])
        wsym = f"w{widx}"
        weights.append(WeightSpec(node.inputs[1], wsym, tuple(int(d) for d in w.shape),
                                  offset, w.size * _ITEMSIZE[dtype]))
        offset += weights[-1].nbytes
        bsym = None
        if node.op == "Gemm" and len(node.inputs) > 2:
            b = graph.initializers[node.inputs[2]]
            bsym = f"b{widx}"
            weights.append(WeightSpec(node.inputs[2], bsym, (int(b.size),), offset,
                                      b.size * _ITEMSIZE[dtype]))
            offset += weights[-1].nbytes
        ops.append(Op("gemm", node.outputs[0], node.inputs[0], wsym, bsym, int(n_in), int(n_out)))
        widx += 1

    in_t = graph.values[graph.inputs[0]]
    out_t = graph.values[graph.outputs[0]]
    flat_in = Tensor(in_t.name, (_length(in_t),), dtype)
    flat_out = Tensor(out_t.name, (_length(out_t),), dtype)
    buffers, assignment = _assign_buffers(graph, ops, flat_in, flat_out)
    return Plan(graph.name, dtype, flat_in, flat_out, tuple(ops), buffers, assignment, tuple(weights))


def _assign_buffers(graph: Graph, ops, flat_in: Tensor, flat_out: Tensor):
    """Give the input and output dedicated buffers; rotate intermediates through a pool."""
    buffers = {"x": flat_in.shape[0], "y": flat_out.shape[0]}
    assignment = {flat_in.name: "x", flat_out.name: "y"}
    last_use = {}
    for i, op in enumerate(ops):
        last_use[op.inp] = i
    free, pool = [], 0
    for i, op in enumerate(ops):
        if op.out not in assignment:
            if free:
                sym = free.pop()
            else:
                sym = f"t{pool}"
                pool += 1
            assignment[op.out] = sym
            length = _length(graph.values[op.out])
            buffers[sym] = max(buffers.get(sym, 0), length)
        if op.inp in assignment and assignment[op.inp].startswith("t") and last_use.get(op.inp) == i:
            free.append(assignment[op.inp])
    return buffers, assignment
