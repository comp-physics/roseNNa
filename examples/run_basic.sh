#!/usr/bin/env bash
# End-to-end: build a model, generate code for it, compile and run both backends.
set -euo pipefail
cd "$(dirname "$0")"

# Overridable so the test suite can drive this script itself rather than a copy
# of it, and so a machine whose Fortran is not gfortran can still run it. The
# defaults are what a reader would type; PYTHON matters when the interpreter
# holding torch is a virtualenv's rather than whatever `python3` resolves to.
# ROSENNA is left unquoted on purpose: it may be a multi-word command such as
# `python3 -m rosenna`.
py=${PYTHON:-python3}
# Where the golden model lives. Overridable because this script REGENERATES
# gemm_small.onnx, and the generator is unseeded: pointed at the repository's
# own tree it rewrites a file other tests read, which is an order dependency
# that only shows up when they run concurrently.
golden=${GOLDEN_DIR:-$PWD/../goldenFiles}
rosenna=${ROSENNA:-rosenna}
cc=${CC:-cc}
fc=${FC:-gfortran}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# 1. Build the PyTorch model and export it to ONNX (writes gemm_small.onnx).
#    The script writes its ONNX to a path relative to the working directory, so
#    it is run from a scratch directory holding a link to the real tree.
ln -sfn "$golden" "$work/goldenFiles"
mkdir -p "$work/run"
(cd "$work/run" && "$py" ../goldenFiles/gemm_small/gemm_small.py >/dev/null)

# 2. Generate Fortran and C for it.
$rosenna generate "$golden/gemm_small/gemm_small.onnx" --lang both --out "$work"

# 3. Build and run the C caller.
"$cc" -O2 -I"$work" cAPI.c -lm -o "$work/capi"
echo -n "C:       "; "$work/capi"

# 4. Build and run the Fortran caller.
# Where to put (and find) the .mod: -J is gfortran's and flang's spelling,
# -module is nvfortran's and ifx's. On a machine with an HPC SDK loaded, FC is
# often already nvfortran, so guessing wrong here is the common case, not the
# exotic one.
# (the whole output, not its first line: nvfortran prints a blank one first)
case "$("$fc" --version 2>&1)" in
    *nvfortran*|*NVIDIA*|*ifx*|*IFX*) moddir=(-module "$work") ;;
    *)                                moddir=("-J$work") ;;
esac
"$fc" -O2 -c "$work/gemm_small_model.F90" -o "$work/gemm_small_model.o" "${moddir[@]}"
"$fc" -O2 -I"$work" capiTester.f90 "$work/gemm_small_model.o" -o "$work/fapi"
echo -n "Fortran: "; "$work/fapi"

# 5. Check both against onnxruntime.
$rosenna verify "$golden/gemm_small/gemm_small.onnx" --lang both --cases 16
