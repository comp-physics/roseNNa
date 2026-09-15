"""Measure gcc -O2 compile time for a header embedding N weight scalars.

Not a test: run manually --

    cd python && python3 tests/measure_embed_threshold.py

-- to reproduce the table cited in the comment next to EMBED_THRESHOLD in
rosenna/plan.py (and pasted into the Task 2 report). Builds a single-Gemm
ONNX model sized so its embedded weight count is close to each of
1e3/1e4/1e5/3e5/1e6 parameters, emits it with `embed=True`, and times
`gcc -O2 -c` compiling an otherwise-empty translation unit that only
`#include`s the generated header (the header, not the source, holds the
embedded ROSENNA_CONST arrays -- the source is nearly empty for an
embedded plan).
"""
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rosenna.emit_c import emit_c
from rosenna.frontend import load_graph
from rosenna.plan import build_plan

SIZES = [1_000, 10_000, 100_000, 300_000, 1_000_000]


def _cc() -> str:
    for cand in ("gcc-15", "gcc-14", "gcc-13", "gcc"):
        path = shutil.which(cand)
        if path:
            probe = subprocess.run([cand, "-fopenmp", "-x", "c", "-", "-o", "/dev/null"],
                                   input="int main(void){return 0;}", capture_output=True, text=True)
            if probe.returncode == 0:
                return cand
    raise SystemExit("no C compiler with -fopenmp found")


def _make_model(path: Path, target_params: int) -> None:
    """A single Gemm layer whose weight+bias element count is close to target_params."""
    n = max(2, round((target_params) ** 0.5))
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.uniform(-1, 1, (n, n)).astype(np.float32), "w")
    b = numpy_helper.from_array(rng.uniform(-1, 1, (n,)).astype(np.float32), "b")
    node = helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="g0")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, n])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, n])
    g = helper.make_graph([node], "measuretmp", [x], [y], initializer=[w, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(m, path)


def main() -> None:
    cc = _cc()
    rows = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for target in SIZES:
            onnx_path = td / f"m{target}.onnx"
            _make_model(onnx_path, target)
            graph = load_graph(onnx_path)
            plan = build_plan(graph, dtype="f64", embed=True)
            _, header = emit_c(plan)
            (td / f"m{target}.h").write_text(header)
            tu = td / f"m{target}.c"
            tu.write_text(f'#include "m{target}.h"\n')
            start = time.perf_counter()
            subprocess.run([cc, "-O2", "-c", str(tu), "-o", str(td / f"m{target}.o")],
                           check=True, capture_output=True, text=True)
            elapsed = time.perf_counter() - start
            rows.append((plan.n_params, elapsed))

    print(f"compiler: {cc}")
    print(f"{'params':>10}  {'compile s':>10}")
    for n_params, elapsed in rows:
        print(f"{n_params:>10}  {elapsed:>10.3f}")


if __name__ == "__main__":
    main()
