#!/usr/bin/env bash
# End-to-end: build a model, generate code for it, compile and run both backends.
set -euo pipefail
cd "$(dirname "$0")"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# 1. Build the PyTorch model and export it to ONNX (writes gemm_small.onnx).
#    The script writes its ONNX to a path relative to the working directory, so
#    it is run from a scratch directory holding a link to the real tree.
ln -sfn "$PWD/../goldenFiles" "$work/goldenFiles"
mkdir -p "$work/run"
(cd "$work/run" && python3 ../goldenFiles/gemm_small/gemm_small.py >/dev/null)

# 2. Generate Fortran and C for it.
rosenna generate ../goldenFiles/gemm_small/gemm_small.onnx --lang both --out "$work"

# 3. Build and run the C caller.
cc -O2 -I"$work" cAPI.c -lm -o "$work/capi"
echo -n "C:       "; "$work/capi"

# 4. Build and run the Fortran caller.
gfortran -O2 -c "$work/gemm_small_model.F90" -o "$work/gemm_small_model.o" -J"$work"
gfortran -O2 -I"$work" capiTester.f90 "$work/gemm_small_model.o" -o "$work/fapi"
echo -n "Fortran: "; "$work/fapi"

# 5. Check both against onnxruntime.
rosenna verify ../goldenFiles/gemm_small/gemm_small.onnx --lang both --cases 16
