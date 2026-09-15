# B. Reaction-diffusion with a learned time-stepper on patches

**PDE class**: parabolic, reactive (2-D FitzHugh-Nagumo).
**Where the NN sits**: it *is* the time step -- from each cell's 3×3 patch
of `(u, v)` it predicts that cell's `(u, v)` ten fine steps later.
**Code structure it shows**: the batched one. Every big step the solver
gathers all cells' patches into one feature array, makes **one** call to
`stepper_infer_batch` over the whole field on device-resident data, and
scatters the result back. The model is generated **file-loaded**
(`--no-embed`), so this is also the example with a real `init` -- the one
upload, before the loop -- and with an archive (`libstepper.a`) built for
the toolchain's native backend and linked into the solver.

```
make TOOLCHAIN=amd        # hipcc builds libstepper.a; the solver links -lamdhip64
make TOOLCHAIN=nvidia     # nvcc builds it; -lcudart
make                      # gcc/gfortran; the archive holds the OpenMP fallback
```

## What it does

The fine scheme (explicit Euler, 5-point Laplacian, `dt = 0.1`, periodic)
is the reference; the surrogate takes `K = 10` of its steps at once. Both
run from the same held-out initial field for 100 big steps; the program
prints the relative L2 error of the surrogate at the end and the time of
both, and exits 0 if the error is under 5%. On an MI210:

```
256x256 grid, 100 surrogate steps of 10 fine steps each:
  fine reference     24.0 ms  (0.24 ms per big step)
  surrogate         767.1 ms  (7.67 ms per big step: gather, infer_batch, scatter)
  relative L2 error of the surrogate vs the reference: 1.999e-03
OK
```

Read the timing honestly: this surrogate is 30× *slower* than the physics
it replaces, because an explicit FitzHugh-Nagumo step costs a handful of
flops per cell and the stepper costs 19k. A learned stepper pays off when
the fine step is expensive (stiff chemistry, an implicit solve); this
example is about the wiring, and the wiring is the same either way.

## The model

`train.py`: 18 → 128 → 128 → 2, `tanh`, fitted to the one-step map on every
(patch, centre-10-steps-later) pair from 32 random fields. The recipe's one
essential ingredient is **noise on the input patches** during the fit.
Fitted on clean inputs the map is 1% accurate per step and blows up after
~40 of its own steps, because it never learned to contract the errors it
introduces; fitted with 3% Gaussian noise it tracks the fine solution for
100 steps (1000 fine steps) to 0.2%, with its largest error (~8%) in the
fast early transient. Unrolled a-posteriori training, which A needs, was
tried here and made this map worse; noise does the same job for a stepper.

## The wiring

```c
static int big_step(double *u, double *v, double *feat, double *out) {
    gather(u, v, feat);                       /* a target loop: 18 values per cell */
    int status;
#pragma omp target data use_device_ptr(feat, out)
    status = stepper_infer_batch(NCELL, feat, out, 0);   /* one launch over the field */
    if (status) return status;
    status = stepper_sync(0);                 /* see below */
    if (status) return status;
    scatter(out, u, v);                       /* a target loop */
    return 0;
}
```

- `stepper_init("stepper.rwt")` runs once, first: it reads the weights and
  uploads them. Nothing else transfers.
- `feat` and `out` are mapped once (`target enter data`); `use_device_ptr`
  hands `infer_batch` their device addresses. `infer_batch` allocates,
  copies and synchronizes nothing.
- **`stepper_sync(0)`** is the one line that is not obvious. With a cuda/hip
  archive, `infer_batch` launches on the null stream and returns; the
  solver's next target region runs on the OpenMP runtime's own queue, and
  nothing orders the two -- without the wait, the scatter read stale output
  on the MI210. `<name>_sync` is the backend-agnostic wait: a stream
  synchronize in the cuda/hip archive, a no-op in the omp one (whose loop
  is synchronous), so this source links against any `ROSENNA_BACKEND`.

The Fortran twin reaches the archive through the module's `bind(C)`
interfaces: `stepper_init_dev` (the **archive's** init -- the module's own
`stepper_init` fills the module's arrays for the Fortran per-point path,
which this program does not use), `stepper_infer_batch_dev` with `c_loc`
of the arrays inside `target data use_device_addr`, and `stepper_sync_dev`.

## What this example found

Three things in the generator came out of building it, all now fixed:
`infer_batch`'s OpenMP fallback used `target teams loop`, which `amdclang`
maps one point per *team* (3.5 µs per point; `distribute parallel for` is
28× faster); there was no backend-agnostic wait (`<name>_sync`); and the
Fortran module had no route to the archive's init (`<name>_init_dev`).

And one thing it shows that is not fixed: at this model size the per-thread
kernel is bound by every thread streaming the whole 150 KB weight matrix,
~160 GMAC/s on the MI210. A fused batched GEMM, which reuses weights
across points, is the fix, and is on the PR's TODO list.
