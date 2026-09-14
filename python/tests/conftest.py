"""Fixtures for golden file models and for inline models built with onnx.helper."""
import os
import platform
import re
import subprocess
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper, TensorProto

from rosenna.golden import golden_generator_run, golden_model_path

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


def skip_unless_libgomp_enforces_mandatory(cc: str, tmp_path: Path) -> None:
    """Skip when this libgomp runs a target region to completion under MANDATORY.

    A libgomp built with no offload plugins (a plain distro gcc < 13, say)
    ignores OMP_TARGET_OFFLOAD=MANDATORY and falls back to the host, so a test
    whose evidence is "the program was refused" cannot run there. Skip, naming
    the toolchain, rather than fail; the caller keeps a platform-independent
    assertion (the object references GOMP_target_ext) as its primary evidence.
    """
    probe = tmp_path / "mandatory_probe.c"
    probe.write_text("int main(void) {\n    int v = 0;\n"
                     "    #pragma omp target map(tofrom: v)\n    v = 1;\n    return v ? 0 : 3;\n}\n")
    build = subprocess.run([cc, "-fopenmp", str(probe), "-o", str(tmp_path / "mandatory_probe")],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(tmp_path / "mandatory_probe")], capture_output=True, text=True,
                         env={**os.environ, "OMP_TARGET_OFFLOAD": "MANDATORY"})
    if run.returncode == 0:
        version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
        pytest.skip(f"libgomp did not enforce OMP_TARGET_OFFLOAD=MANDATORY for a C target region "
                    f"on {platform.platform()} with {version}")


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
    (goldenFiles/<name>/<name>.onnx), and generates it if it does not exist. How the
    generator is run is rosenna.golden's business, shared with the gpu-gate.
    """
    root = Path(__file__).resolve().parents[2]
    generated = {}

    def _get_model_path(name: str) -> Path:
        if name not in generated:
            model_path = golden_model_path(root, name)
            if not model_path.exists():
                with golden_generator_run(root, name) as (argv, cwd, env):
                    subprocess.run(argv, cwd=cwd, check=True, env=env)
            if not model_path.exists():
                # goldenFiles/mnist/mnist.py reads its .onnx rather than
                # writing one -- that model is checked in. Say so, instead of
                # handing back a path that does not exist.
                raise FileNotFoundError(
                    f"{model_path} is missing and {name}.py did not create it; "
                    f"if it is a checked-in model, restore it with git checkout")
            generated[name] = model_path
        return generated[name]

    return _get_model_path


@pytest.fixture
def repo_root():
    """The repository root, from this file's location rather than the cwd."""
    return Path(__file__).resolve().parents[2]
