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


# --- the hello-world path -------------------------------------------------
#
# `examples/run_basic.sh` is the shortest path through the whole tool and the
# first thing a reader runs: export gemm_small, generate both backends, call
# the result from C (`cAPI.c`) and from Fortran (`capiTester.f90`), then verify
# against onnxruntime. Nothing ran it. Both callers are hand-written against
# the generated API -- the header's signature, the module name, whether an
# `_init` is needed -- so a change to what `generate` emits would leave them
# stale with only a reader to notice.
#
# The test drives the script itself rather than a copy of its commands, which
# is the point: a copy would keep passing after the script rotted.

def test_the_basic_example_script_runs_both_callers():
    cc = _omp_cc()
    for tool in ("gfortran", "bash"):
        if not shutil.which(tool):
            pytest.skip(f"no {tool}")
    script = ROOT.parent / "run_basic.sh"
    env = {**os.environ, "PYTHON": sys.executable,
           "ROSENNA": f"{sys.executable} -m rosenna", "CC": cc}
    r = subprocess.run(["bash", str(script)], env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]

    # Both callers feed the same input to the same model, so their output is
    # the same three numbers. The script's own `verify` step is what checks
    # those numbers against onnxruntime; this checks that the two hand-written
    # callers agree, which verify never sees.
    printed = {}
    for line in r.stdout.splitlines():
        for label in ("C:", "Fortran:"):
            if line.startswith(label):
                printed[label] = [float(v) for v in line[len(label):].split()]
    assert set(printed) == {"C:", "Fortran:"}, r.stdout
    assert len(printed["C:"]) == 3, r.stdout
    # Fortran's f0.6 drops the leading zero (".535685"); float() reads both.
    assert printed["C:"] == printed["Fortran:"], r.stdout

    # And the script's last step, which is the one that cannot come out
    # vacuous: gemm_small is re-exported unseeded every run and sometimes
    # returns all zeros, so "the two callers agree" can be 0 == 0. `verify`
    # reports a line per language ending in `ok` or `FAIL`.
    verdicts = {line.split()[0]: line.split()[-1]
                for line in r.stdout.splitlines()
                if line.split()[:1] in (["c"], ["fortran"])}
    assert verdicts == {"c": "ok", "fortran": "ok"}, r.stdout
