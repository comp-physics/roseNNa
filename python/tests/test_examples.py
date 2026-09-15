"""The surrogate examples build and run on the host toolchain, at their small sizes.

Each example is a solver plus a surrogate that prints an error and exits
non-zero if the surrogate failed to do what its README claims, so `make`
succeeding is the assertion. The trained models are checked in; this
generates, compiles (gcc / gfortran, -fopenmp on the host) and runs.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_device_c import _omp_cc

EXAMPLES = ["burgers_closure", "reaction_patch", "bubble_lstm", "poisson_guess"]
ROOT = Path(__file__).resolve().parents[2] / "examples" / "surrogates"


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_builds_and_runs_on_the_host(example, tmp_path):
    cc = _omp_cc()                       # a C compiler that takes -fopenmp (gcc-15.. on macOS)
    for tool in ("gfortran", "make"):
        if not shutil.which(tool):
            pytest.skip(f"no {tool}")
    work = tmp_path / example
    shutil.copytree(ROOT / example, work, ignore=shutil.ignore_patterns("gen", "*.rwt"))
    shutil.copy(ROOT / "common.mk", tmp_path / "common.mk")
    env = {**os.environ, "OMP_NUM_THREADS": "4"}
    # NSTEPS=3 for poisson_guess: 20 steps is ~100k OpenMP regions, which a
    # macOS runner's libgomp takes minutes to wake threads for.
    r = subprocess.run(["make", "-s", "TOOLCHAIN=gnu", "NB=4", "NX=64", "NSTEPS=3", f"CC={cc}",
                        f"ROSENNA={sys.executable} -m rosenna"],
                       cwd=work, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "OK" in r.stdout, r.stdout
