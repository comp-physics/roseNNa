# B. Reaction-diffusion with a learned time-stepper on patches

Parabolic, reactive PDE (2-D FitzHugh-Nagumo); the network is the time
step: from each cell's 3×3 patch of `(u, v)` it predicts that cell's
`(u, v)` ten fine steps later. The batched pattern: gather all patches
into one array, one `stepper_infer_batch` over the field on device
pointers, scatter back. The model is file-loaded (`--no-embed`), so
`stepper_init` uploads the weights once before the loop, and the solver
links `libstepper.a` built for the toolchain's native backend.

```
make TOOLCHAIN=amd | nvidia | gnu      # gnu: host only, use NX=64
```

Reference: explicit Euler, 5-point Laplacian, `dt = 0.1`, periodic.
Surrogate: `K = 10` of those per call, 100 calls from the same initial
field. Exits 0 if the final relative L2 error is under 5%. MI210:

```
256x256 grid, 100 surrogate steps of 10 fine steps each:
  fine reference     24.0 ms  (0.24 ms per big step)
  surrogate         767.1 ms  (7.67 ms per big step)
  relative L2 error of the surrogate: 1.999e-03
OK
```

The surrogate is 30× slower than the physics it replaces here: an
explicit FitzHugh-Nagumo step is a few flops per cell, the stepper 19k.
The pattern pays when the fine step is expensive (stiff chemistry, an
implicit solve).

## Model

`train.py`: 18 → 128 → 128 → 2, `tanh`, fitted to the one-step map with 3%
Gaussian noise on the input patches. Fitted on clean inputs the map is
1% accurate per step and diverges after ~40 of its own steps; with the
noise it tracks 1000 fine steps to 0.2%. Unrolled training (as in A) made
it worse.

## Wiring

```c
gather(u, v, feat);
#pragma omp target data use_device_ptr(feat, out)
status = stepper_infer_batch(NCELL, feat, out, 0);
status = stepper_sync(0);
scatter(out, u, v);
```

`feat` and `out` are mapped once. `stepper_sync` is needed with a
cuda/hip archive: `infer_batch` launches on the null stream and returns,
and the OpenMP scatter runs on the runtime's own queue with no ordering
between them. With an omp archive it is a no-op.

`react.F90` uses the module's `bind(C)` routes: `stepper_init_dev` (the
archive's init; the module's own `stepper_init` serves the Fortran
per-point path, not used here), `stepper_infer_batch_dev` with `c_loc`
inside `target data use_device_addr`, `stepper_sync_dev`.

## Notes

- Building this example changed the generator: `infer_batch`'s OpenMP
  fallback used `target teams loop`, which `amdclang` maps one point per
  team (3.5 µs per point; `distribute parallel for` is 28× faster);
  `<name>_sync` and `<name>_init_dev` did not exist.
- At this model size the per-thread kernel is bound by every thread
  reading the whole 150 KB weight matrix, about 160 GMAC/s on the MI210.
  A batched GEMM kernel would fix that; it is on the PR's TODO list.
