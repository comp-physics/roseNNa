# Poisson solves with a conv-net initial guess

Elliptic PDE (periodic Poisson, the pressure solve of a projection
method). The network does not replace the solver; it starts it. A small
conv net maps the right-hand side to an initial guess and Jacobi iterates
from there. The whole-field pattern: the entire field is one model input
(NCHW `1×1×76×76`, `f` with the 6-cell periodic halo the solver builds) and
one output (`1×1×64×64`), one call per step: `poisson_guess_infer_one`,
which runs the model on the device as one launch per layer over device
pointers, its activations in static device buffers.

```
make TOOLCHAIN=amd | nvidia | gnu
```

Twenty steps of a rotating right-hand side. Each step Jacobi runs to
`|lap(φ) − f| / |f| < 1e-3` from three starts: zero, the previous step's
solution, the NN guess. MI210, both languages:

```
64x64 periodic Poisson, 20 steps of a rotating right-hand side, Jacobi to 1e-03:
  iterations per step from zero                  560
  iterations per step from the last solution     446
  iterations per step from the NN guess          460   (guess: 3.0 ms per step)
OK: NN guess saves 18% of the zero-start iterations
```

## Model

`train.py`: three 5×5 convolutions, 1 → 8 → 8 → 1, `tanh` between, no
padding; the receptive field is 13 cells. The loss is the residual of the
guess, `|lap(NN(f)) − f|²`, not the distance to `φ`: Jacobi stops on the
residual, and a guess fitted to `φ` carries high-mode error that the
Laplacian amplifies by k². Fitted that way this net doubled the iteration
count; on the residual it leaves 8% of the zero guess's.

## Wiring

```c
#pragma omp target teams distribute parallel for collapse(2)  /* periodic halo */
for (int i = 0; i < NP; ++i)
    for (int j = 0; j < NP; ++j)
        fp[i * NP + j] = f[wrap(i - HALO) * N + wrap(j - HALO)];
#pragma omp target data use_device_ptr(fp, phi_nn)
status = poisson_guess_infer_one(fp, phi_nn, 0);      /* the whole field, on the device */
status = poisson_guess_sync(0);
it_nn += jacobi(phi_nn, tmp, f, fnorm);
```

The per-point `infer` keeps its intermediate activations as locals, and
for a whole-field model those are the field's: `8×72² + 8×68²` doubles,
660 KB, more than a device thread's stack (`amdclang`: "stack frame size
(663568) exceeds limit (131056)"). `infer_one` is the form for this
model: a launch per layer with the thread index over the layer's output
elements, the activations in static device buffers. Nothing is copied
inside the step loop but the new right-hand side. The archive
(`libpoisson_guess.a`, built for the toolchain's backend) provides it;
with the gnu toolchain it is one OpenMP target loop per layer.

Two notes on `poisson.F90`: the generated module's own `infer_batch`
would instantiate the per-point `infer` on the device, so `Makefile`
builds the module host-only (`MODULE_OFFLOAD :=`) and the program calls
the archive through `poisson_guess_infer_one_dev`; and `fp` is indexed
`(j, i)`, since the model's NCHW input is row-major and Fortran is
column-major.
