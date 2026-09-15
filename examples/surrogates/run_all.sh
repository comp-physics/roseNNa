#!/usr/bin/env bash
# Build and run every example with one toolchain: ./run_all.sh [amd|nvidia|gnu]
set -euo pipefail
cd "$(dirname "$0")"
tc=${1:-gnu}
extra=()
[ "$tc" = gnu ] && extra=(NB=4 NX=64)
for d in A_burgers_closure B_reaction_patch C_bubble_lstm D_poisson_guess; do
    echo "== $d ($tc)"
    make -s -C "$d" TOOLCHAIN="$tc" "${extra[@]}"
done
echo "all four passed"
