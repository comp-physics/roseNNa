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
    # One thread, deliberately. These examples step a small field, so each
    # OpenMP region is tiny and thread coordination dominates: on this machine
    # poisson_guess takes 406 ms on one thread and 6.4 s on four, and with
    # OMP_NUM_THREADS unset (all cores) 145 s. macOS runners have the same
    # shape of problem worse -- Homebrew libgomp wakes threads slowly -- and
    # four different examples have timed out there in turn, each "fixed" by
    # shrinking its step count, which was treating the symptom.
    #
    # The test asserts the surrogate computes what its README claims, not that
    # it is fast, so the thread count is free to be whatever runs cleanest.
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    # NSTEPS=3 for poisson_guess: 20 steps is ~100k OpenMP regions, which a
    # macOS runner's libgomp takes minutes to wake threads for.
    r = subprocess.run(["make", "-s", "TOOLCHAIN=gnu", "NB=4", "NX=64", "NSTEPS=3", f"CC={cc}",
                        f"ROSENNA={sys.executable} -m rosenna"],
                       cwd=work, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "OK" in r.stdout, r.stdout


# --- the GPU toolchains ---------------------------------------------------
#
# TOOLCHAIN=nvidia and amd were not covered by anything until three of the four
# examples turned out not to link under nvidia at all: the branch had never
# been run. The host test above cannot catch that -- it builds with gcc.
#
# These run the same examples through the real offload compilers, and under
# OMP_TARGET_OFFLOAD=MANDATORY (common.mk sets it for every non-gnu
# toolchain), so a silent fall back to the host fails instead of passing.

def _nvidia_arch():
    """`ccXY` for the installed GPU, or None if there is no usable NVIDIA GPU."""
    if not all(shutil.which(t) for t in ("nvc", "nvfortran", "nvcc", "nvidia-smi")):
        return None
    r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                       capture_output=True, text=True)
    caps = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    if r.returncode != 0 or not caps:
        return None
    return "cc" + caps[0].replace(".", "")


def _amd_arch():
    """The first `gfx...` target, or None if there is no usable AMD GPU."""
    if not all(shutil.which(t) for t in ("amdclang", "amdflang", "hipcc", "rocminfo")):
        return None
    r = subprocess.run(["rocminfo"], capture_output=True, text=True)
    for tok in r.stdout.split():
        if tok.startswith("gfx"):
            return tok.strip(":")
    return None


GPU_TOOLCHAINS = [
    pytest.param("nvidia", _nvidia_arch,
                 marks=pytest.mark.skipif(_nvidia_arch() is None,
                                          reason="no nvc/nvfortran/nvcc + NVIDIA GPU")),
    pytest.param("amd", _amd_arch,
                 marks=pytest.mark.skipif(_amd_arch() is None,
                                          reason="no amdclang/amdflang/hipcc + AMD GPU")),
]


@pytest.mark.parametrize("toolchain,arch", GPU_TOOLCHAINS)
@pytest.mark.parametrize("example", EXAMPLES)
def test_example_builds_and_runs_on_a_gpu(example, toolchain, arch, tmp_path):
    if not shutil.which("make"):
        pytest.skip("no make")
    work = tmp_path / example
    shutil.copytree(ROOT / example, work, ignore=shutil.ignore_patterns("gen", "*.rwt"))
    shutil.copy(ROOT / "common.mk", tmp_path / "common.mk")
    # Small sizes: this is a "does the toolchain link and run" check, not a
    # benchmark. The sizes the READMEs quote are what run_all.sh uses.
    r = subprocess.run(["make", "-s", f"TOOLCHAIN={toolchain}", f"ARCH={arch()}",
                        "NB=4", "NX=64", "NSTEPS=3",
                        f"ROSENNA={sys.executable} -m rosenna"],
                       cwd=work, env=os.environ, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "OK" in r.stdout, r.stdout
