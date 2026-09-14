"""Fixtures for golden file models and for inline models built with onnx.helper."""
import re
import subprocess
import tempfile
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper, TensorProto

# A diagnostic about the generated source carries a <file>:<line>: location.
# A driver-level notice instead names the tool as its "location" -- for
# example Apple clang on the macOS CI runner prints, on every invocation and
# whatever the source,
#   clang: warning: overriding deployment version from '16.0' to '26.0' [-Woverriding-deployment-version]
# which is about the SDK versus the deployment target and nothing to do with
# our C (ruling R21). gfortran's own multi-line diagnostics keep their
# `<file>:<line>:<col>:` header and a bare `Warning: ...` line, neither of
# which this pattern matches, so they survive.
_DRIVER_NOTICE = re.compile(r"^[^\s:]+: (warning|note): ")


def _source_diagnostics(stderr: str):
    """Split compiler stderr into (about the source, driver-level noise)."""
    kept, dropped = [], []
    for line in stderr.splitlines():
        (dropped if _DRIVER_NOTICE.match(line) else kept).append(line)
    return "\n".join(kept).strip(), "\n".join(dropped).strip()


def _assert_warning_free(lang: str, stderr: str) -> None:
    kept, dropped = _source_diagnostics(stderr)
    assert kept == "", (f"{lang}: diagnostics about the generated source:\n{kept}\n"
                        f"(driver-level notices ignored: {dropped or 'none'})")


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
    (goldenFiles/<name>/<name>.onnx), and generates it if it does not exist.

    Each generator script writes its ONNX to a hard-coded `../goldenFiles/<name>/`
    and drops an `inputs.fpp` beside itself, so it is run from a throwaway
    directory holding a `goldenFiles` symlink: the model lands in the real tree
    and the scratch output is discarded with the temp directory. (It used to run
    in `test/`, which existed to serve the old runtime library's shell suite.)
    """
    root = Path(__file__).resolve().parents[2]
    generated = {}
    tmp = tempfile.TemporaryDirectory()
    cwd = Path(tmp.name) / "run"
    cwd.mkdir()
    (Path(tmp.name) / "goldenFiles").symlink_to(root / "goldenFiles", target_is_directory=True)

    def _get_model_path(name: str) -> Path:
        if name not in generated:
            model_path = root / "goldenFiles" / name / f"{name}.onnx"
            if not model_path.exists():
                subprocess.run(
                    [sys.executable, str(root / "goldenFiles" / name / f"{name}.py")],
                    cwd=cwd, check=True,
                )
            generated[name] = model_path
        return generated[name]

    yield _get_model_path
    tmp.cleanup()


@pytest.fixture
def repo_root():
    """The repository root, from this file's location rather than the cwd."""
    return Path(__file__).resolve().parents[2]
