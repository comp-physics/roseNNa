#!/usr/bin/env bash
# Build and run every example with one toolchain: ./run_all.sh [amd|nvidia|gnu]
set -euo pipefail
cd "$(dirname "$0")"
tc=${1:-gnu}
extra=()
[ "$tc" = gnu ] && extra=(NB=4 NX=64)
for d in burgers_closure reaction_patch bubble_lstm poisson_guess; do
    echo "== $d ($tc)"
    make -s -C "$d" TOOLCHAIN="$tc" "${extra[@]}"
done
echo "all four passed"
