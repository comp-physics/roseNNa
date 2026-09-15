# Surrogate models inside PDE solvers

Four self-contained solvers, each in C and Fortran, with a network called
inside the time-step loop. They differ in where the network sits and what
code structure that forces:

| | PDE | Where the network sits | Structure |
|---|---|---|---|
| [A](A_burgers_closure/) | coarse-grid Burgers | per-cell closure in the flux loop | embedded model, header-inline `infer` from the solver's own offload loop |
| [B](B_reaction_patch/) | 2-D FitzHugh-Nagumo | the time step, on a 3×3 patch | batched: gather → one `infer_batch` → scatter; file-loaded weights, `init`, an archive for the native backend, `<name>_sync` |
| [C](C_bubble_lstm/) | acoustics through bubbles | recurrent model per cell replacing a bubble population | stateful: an LSTM whose `(h, c)` stays on the device; several inputs and outputs concatenated in `x` and `y` |
| [D](D_poisson_guess/) | periodic Poisson | initial guess for the iterative solve | whole-field: the entire RHS as one input, one call per step; runs on the host, see its README |

Each runs its reference physics and its surrogate from the same held-out
initial condition, prints an error and both timings, and exits 0 only if
the surrogate did what its README says. The C and Fortran twins print the
same numbers on an MI210.

```
cd A_burgers_closure
make TOOLCHAIN=amd          # amdclang / amdflang / hipcc
make TOOLCHAIN=nvidia       # nvc / nvfortran / nvcc
make                        # gcc / gfortran, host only; NB=4 NX=64 for speed
../run_all.sh amd           # all four
```

`common.mk` holds the toolchain selection and the build rules; each
example's Makefile names its model and program. The trained `.onnx` files
are checked in; `make train` rebuilds one (a minute to a few minutes).

Common to all four: the model goes to the device once (embedded, or
uploaded by `init`); the solver's arrays are mapped once before the loop
and swapped by pointer; Fortran passes arrays into target regions as
explicit-shape dummies, since `amdflang` re-maps an allocatable's
descriptor on every region entry; A and C are ensembles of 1-D problems,
because one is too small to occupy a GPU.

Two generator limits show up here and are described where they do: the
per-thread `infer_batch` kernel reads the whole weight matrix from every
thread (B), and a whole-field model's activations do not fit a device
thread (D). A batched GEMM kernel and a tiled conv kernel are on the
TODO list.
