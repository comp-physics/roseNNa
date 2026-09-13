"""Fixtures for golden file models."""
import subprocess
import sys
from pathlib import Path

import pytest


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
