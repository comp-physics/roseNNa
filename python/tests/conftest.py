"""Fixtures for golden file models and for inline models built with onnx.helper."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper, TensorProto


def save_model(directory, name, nodes, inits, in_shape, out_shape, elem=TensorProto.FLOAT):
    """Save a one-input, one-output ONNX graph as <directory>/<name>.onnx and return the path."""
    x = helper.make_tensor_value_info("x", elem, list(in_shape))
    y = helper.make_tensor_value_info("y", elem, list(out_shape))
    g = helper.make_graph(nodes, name, [x], [y], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    path = Path(directory) / f"{name}.onnx"
    onnx.save(m, path)
    return path


@pytest.fixture
def live_gemm_model(tmp_path):
    """A tiny deterministic model whose onnxruntime reference can never be all-zero.

    One Gemm with fixed non-zero weights and a non-zero bias, and no activation,
    so every sampled input produces a non-zero output. Use it for tests whose
    subject is an error path (a compile failure, a missing compiler) rather than
    a numerical comparison: the golden generators are unseeded, and a regenerated
    gemm_small can be a dead model on which verify's non-degeneracy check fires
    -- returning 1 with its all-zero message -- before the code under test is
    ever reached (ruling R22).
    """
    w = numpy_helper.from_array(np.array([[0.5, -0.25], [1.0, 0.75], [-0.5, 2.0]], np.float32), "w")
    b = numpy_helper.from_array(np.array([0.125, -0.375], np.float32), "b")
    node = helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="g0")
    return save_model(tmp_path, "livegemm", [node], [w, b], (1, 3), (1, 2))


@pytest.fixture(scope="session")
def golden_model():
    """Return a helper that generates a golden ONNX model by running its generator script.

    The helper takes a model name (e.g. "gemm_small"), returns the path to the ONNX file
    (../goldenFiles/<name>/<name>.onnx), and generates it if it does not exist.
    Runs the generator script from test/ as the working directory so filePath resolution
    and side effects (inputs.fpp) stay in a disposable directory.
    """
    generated = {}

    def _get_model_path(name: str) -> Path:
        if name not in generated:
            model_path = Path(f"../goldenFiles/{name}/{name}.onnx")
            if not model_path.exists():
                subprocess.run(
                    [sys.executable, f"../goldenFiles/{name}/{name}.py"],
                    cwd="../test",
                    check=True,
                )
            generated[name] = model_path
        return generated[name]

    return _get_model_path
