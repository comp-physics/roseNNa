"""Run a goldenFiles/<name>/<name>.py generator so its ONNX lands in the tree.

Each generator writes its model to a hard-coded `../goldenFiles/<name>/` and
drops scratch output (an `inputs.fpp`) beside its cwd, so it is run from a
throwaway directory holding a `goldenFiles` symlink: the model lands in the
real tree and the scratch output is discarded with the temp directory. (The
generators used to run in `test/`, which served the retired runtime library's
shell suite and no longer exists.) The LSTM generators `import nnLSTM`, a
helper that lives beside them in goldenFiles/, so that directory goes on
PYTHONPATH. Shared by the test fixture and the gpu-gate so both run the
generators the same way.
"""
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


def golden_model_path(root: Path, name: str) -> Path:
    return root / "goldenFiles" / name / f"{name}.onnx"


@contextmanager
def golden_generator_run(root: Path, name: str):
    """Yield (argv, cwd, env) that runs the generator; cwd exists only inside the block."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp) / "run"
        cwd.mkdir()
        (Path(tmp) / "goldenFiles").symlink_to(root / "goldenFiles", target_is_directory=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root / "goldenFiles")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        yield [sys.executable, str(root / "goldenFiles" / name / f"{name}.py")], cwd, env
