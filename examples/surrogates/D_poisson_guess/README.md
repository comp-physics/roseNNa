# D. Poisson solves with a conv-net initial guess

Elliptic PDE (periodic Poisson, the pressure solve of a projection
method). The network does not replace the solver; it starts it. A small
conv net maps the right-hand side to an initial guess and Jacobi iterates
from there. The whole-field pattern: the entire field is one model input
(NCHW `1×1×76×76`, `f` with the 6-cell periodic halo the solver builds) and
one output (`1×1×64×64`), one `infer` call per step.

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
  iterations per step from the NN guess          460   (guess: 7.7 ms per step)
OK: NN guess saves 18% of the zero-start iterations
```

## Model

`train.py`: three 5×5 convolutions, 1 → 8 → 8 → 1, `tanh` between, no
padding; the receptive field is 13 cells. The loss is the residual of the
guess, `|lap(NN(f)) − f|²`, not the distance to `φ`: Jacobi stops on the
residual, and a guess fitted to `φ` carries high-mode error that the
Laplacian amplifies by k². Fitted that way this net doubled the iteration
count; on the residual it leaves 8% of the zero guess's.

## Wiring, and the limit it runs into

```c
for (int i = 0; i < NP; ++i)                 /* periodic halo */
    for (int j = 0; j < NP; ++j)
        fp[i * NP + j] = f[wrap(i - HALO) * N + wrap(j - HALO)];
poisson_guess_infer(fp, phi_nn);             /* the whole field, on the host */
#pragma omp target update to(phi_nn[0:N * N]);
it_nn += jacobi(phi_nn, tmp, f, fnorm);      /* on the device */
```

This is the one copy inside a step loop in these examples. The generated
`infer` keeps its intermediate activations as locals, and for a
whole-field model those are the field's activations: `8×72² + 8×68²`
doubles, 660 KB. A device thread cannot hold that (`amdclang`: "stack
frame size (663568) exceeds limit (131056)"), so the guess runs on the
host and the solver uploads 32 KB per step, next to the `target update
to(f)` it does anyway. The weights are embedded and never move.

Two consequences in `poisson.F90`: the generated module always contains
an `infer_batch` target region, which instantiates `infer` on the device
whether or not it is called, so `Makefile` builds the module host-only
(`MODULE_OFFLOAD :=`); and `fp` is indexed `(j, i)`, since the model's
NCHW input is row-major and Fortran is column-major.

A conv net over a field on the device needs a tiled kernel with
activations in shared memory, which the generator does not have. At this
size the host guess costs 8 ms per step, about what the 100 Jacobi
iterations it saves cost.
