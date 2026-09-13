"""Reject anything this generator cannot lower, naming the node."""
from .frontend import Graph, UnsupportedModel

SUPPORTED = {"Gemm", "MatMul", "Relu", "Tanh", "Sigmoid"}


def validate(graph: Graph) -> None:
    for node in graph.nodes:
        if node.op not in SUPPORTED:
            raise UnsupportedModel(
                f"node '{node.name}': {node.op} is not supported; "
                f"this generator handles {sorted(SUPPORTED)}")
        if node.op == "Gemm":
            _validate_gemm(graph, node)
        if node.op == "MatMul" and node.inputs[1] not in graph.initializers:
            raise UnsupportedModel(
                f"node '{node.name}': MatMul needs a constant second input; "
                f"'{node.inputs[1]}' is computed at runtime")
    for name, t in graph.values.items():
        if len(t.shape) not in (1, 2):
            raise UnsupportedModel(
                f"value '{name}' has rank {len(t.shape)}; this generator handles rank 1 and 2")
        if len(t.shape) == 2 and t.shape[0] != 1:
            raise UnsupportedModel(
                f"value '{name}' has leading dimension {t.shape[0]}; "
                f"this generator infers one point per call")


def _validate_gemm(graph: Graph, node) -> None:
    if int(node.attrs.get("transA", 0)) != 0:
        raise UnsupportedModel(f"node '{node.name}': Gemm transA=1 is not supported")
    for attr in ("alpha", "beta"):
        value = float(node.attrs.get(attr, 1.0))
        if abs(value - 1.0) > 1e-12:
            raise UnsupportedModel(f"node '{node.name}': Gemm {attr}={value} is not supported; only 1.0")
    if node.inputs[1] not in graph.initializers:
        raise UnsupportedModel(
            f"node '{node.name}': Gemm weight '{node.inputs[1]}' must be a constant")
    if len(node.inputs) > 2:
        bias = graph.initializers.get(node.inputs[2])
        if bias is None:
            raise UnsupportedModel(f"node '{node.name}': Gemm bias '{node.inputs[2]}' must be a constant")
        if bias.ndim != 1:
            raise UnsupportedModel(
                f"node '{node.name}': Gemm bias has rank {bias.ndim}; only rank 1 is supported")
