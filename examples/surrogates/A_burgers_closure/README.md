# A. Coarse-grid Burgers with a learned subgrid closure

**PDE class**: hyperbolic (viscous Burgers, `u_t + (u²/2)_x = ν u_xx`).
**Where the NN sits**: a per-cell closure, called from the solver's own
offload loop on every cell of every time step.
**Code structure it shows**: the simplest one -- an embedded model,
header-inline `closure_infer`, no `init`, nothing to link; the solver's own
arrays mapped once before the time loop and nothing moving inside it.

```
make TOOLCHAIN=amd        # MI-series GPU: amdclang / amdflang / hipcc
make TOOLCHAIN=nvidia     # A100-class GPU: nvc / nvfortran / nvcc
make                      # any machine: gcc / gfortran on the host (slow; use NB=4)
```

## What it does

An ensemble of `NB` independent realizations of periodic Burgers, each a sum
of three random sine modes. For every realization the program runs

1. the **fine reference** (2048 cells, 64 sub-steps per coarse step),
   box-filtered onto the coarse grid;
2. the **coarse scheme alone** (128 cells: Godunov flux, central viscous
   term, forward Euler);
3. the **coarse scheme plus the closure**: the same scheme, with
   `closure_infer(stencil, corr)` adding a learned correction to each
   cell's right-hand side.

and prints the mean relative L2 error of 2 and 3 against 1 after 200
steps. It exits 0 if the closure helps. On an MI210 (both languages give
the same numbers to every digit):

```
ensemble of 64 realizations, 128 coarse cells, 200 steps; mean relative L2 error vs the filtered fine reference:
  coarse scheme alone     4.3532e-02   (47.4 ms)
  coarse + NN closure     3.1346e-02   (315.5 ms, 163.63 ns per cell-step for the closure)
  fine reference, 2048 cells               (3179.8 ms)
OK: closure reduces the error by 1.4x
```

## The model

`train.py` builds the closure: an MLP from the 7-point stencil
`ū_{i-3..i+3}` to a per-cell correction, 7 → 64 → 64 → 1 with `tanh`. The
recipe is the part worth reading. A closure fitted *a priori* -- on states
taken from the filtered fine solution -- explains most of the residual's
variance and then makes the coarse run *worse*, because at run time it
sees its own drifting coarse state. So it is trained *a posteriori*, with
the coarse solver in the loop: the coarse scheme is written once more in
torch (`rhs_torch`, the same arithmetic as the C and Fortran solvers), a
full 200-step rollout with the closure inside is unrolled from each
training initial condition, and the loss is the relative error of that
trajectory against the filtered-fine one. A local 7-point closure cannot
recover sub-cell structure, so ~1.3-1.4x is roughly the ceiling at this
coarsening; the point of the example is the wiring, not the closure.

`closure.onnx` is checked in because training takes a few minutes on a
CPU; `make train` regenerates it.

## The wiring

```c
static void step(const double *u, double *unew, int n, double dx, double dt, int use_nn) {
#pragma omp target teams distribute parallel for collapse(2)
    for (int b = 0; b < NB; ++b)
        for (int i = 0; i < n; ++i) {
            ...
            if (use_nn) {
                const double stencil[7] = {r[im3], r[im2], r[im1], r[i], r[ip1], r[ip2], r[ip3]};
                double corr[1];
                closure_infer(stencil, corr);      /* the surrogate, per cell, on the device */
                rhs += corr[0];
            }
            unew[(size_t)b * n + i] = r[i] + dt * rhs;
        }
}
```

`closure_infer` is `static inline` in `gen/closure.h` with the weights baked
in as `static const` arrays inside a `declare target` region, so the model
is part of the device image: nothing is uploaded, ever. The time loop maps
`u`/`tmp` once (`target enter data`) and swaps the two device buffers by
pointer; `rosenna gpu-gate` is the check that this kind of loop moves
nothing (see `python/README.md`).

The Fortran twin, `burgers.F90`, `use`s the generated module and is the same
program line for line, with one Fortran-specific point: the step loop
reaches its arrays as **explicit-shape dummies** (`u(n, nb_)`), never as
allocatables. An allocatable referenced inside a target region carries a
descriptor, and `amdflang`'s OpenMP re-maps that descriptor on every region
entry -- a copy per array per step in a loop whose data is fully resident.

## Two things to know about the numbers

- One realization (128 cells) is far too small to occupy a GPU, which is
  why the example is an ensemble: a kernel that small is latency-bound
  whatever it computes, and its per-cell cost says nothing about the model.
  Even at 64 × 128 cells the closure's ~160 ns per cell-step is mostly
  latency; `rosenna gpu-gate` measures ~2 ns per point for a larger model
  over a million points.
- `NB=4` (`make NB=4`) is the CI-sized run for a host build; the fine
  reference dominates its time.
