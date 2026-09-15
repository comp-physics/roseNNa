# D. Poisson solves with a conv-net initial guess

**PDE class**: elliptic (periodic Poisson, `lap(φ) = f`, the pressure
solve of a projection method).
**Where the NN sits**: it does not replace the solver, it *starts* it: a
small conv net maps the right-hand side to an initial guess, and Jacobi
iterates from there to a residual tolerance.
**Code structure it shows**: a **whole-field** surrogate. The entire field
is one model input (NCHW `1×1×76×76`: `f` with the 6-cell periodic halo
that three 5×5 valid convolutions consume, which the solver builds) and one
output (`1×1×64×64`) -- one `infer` call per step, not one per cell.

```
make TOOLCHAIN=amd | nvidia | gnu
```

## What it does

Twenty steps of a rotating right-hand side (six Fourier modes whose phases
advance with the step), as a projection RHS would. Each step, Jacobi runs
to `|lap(φ) − f| / |f| < 1e-3` from three starts -- zero, the previous
step's solution, and the NN guess -- and the iteration counts are compared.
On an MI210 (both languages, to the iteration):

```
64x64 periodic Poisson, 20 steps of a rotating right-hand side, Jacobi to 1e-03:
  iterations per step, from a zero guess                 560
  iterations per step, from the previous solution        446
  iterations per step, from the NN guess                 460   (guess: 7.7 ms per step)
OK: the NN guess saves 18% of the zero-start iterations
```

## The model, and the one training detail that matters

`train.py`: three 5×5 convolutions, 1 → 8 → 8 → 1 channels, `tanh` between,
no padding. The receptive field is 13 cells, so it cannot represent the
inverse Laplacian's long tail; it is a guess. The detail: it is trained on
the **residual** of its guess, `|lap(NN(f)) − f|²`, not on the distance to
the exact `φ`. Jacobi stops on the residual, and a guess fitted to `φ` in
L2 carries high-mode error that the Laplacian amplifies by k²: the first
version of this model, trained that way, made Jacobi take *twice* as many
iterations as a zero guess. Trained on the residual, the same net leaves 8%
of the zero guess's residual (and, as it happens, a better `φ`).

## The wiring -- and the limit this pattern runs into

```c
for (int i = 0; i < NP; ++i)                 /* the periodic halo, on the host */
    for (int j = 0; j < NP; ++j)
        fp[i * NP + j] = f[wrap(i - HALO) * N + wrap(j - HALO)];
poisson_guess_infer(fp, phi_nn);             /* one call, the whole field, on the host */
#pragma omp target update to(phi_nn[0:N * N]) /* 32 KB, once per step */
it_nn += jacobi(phi_nn, tmp, f, fnorm);      /* on the device, from the guess */
```

This is the one place in these examples where a copy sits inside the step
loop, and it is there because of how the generated code is shaped. `infer`
is a per-point routine whose intermediate activations are locals of the
call; for a whole-field model those are the whole field's activations --
`8×72×72 + 8×68×68` doubles, 660 KB -- and a device thread cannot hold that
(`amdclang`: "stack frame size (663568) exceeds limit (131056)"). So the
guess runs on the host and the solver uploads it: an explicit
`target update to` of 32 KB per step, the solver's own choice, next to the
`target update to(f)` it does anyway for the new right-hand side. The
weights themselves are embedded and never move.

The Fortran twin has a second consequence: the generated module always
contains an `infer_batch` target region, which instantiates `infer` on the
device whether or not the program calls it, so the module cannot be built
with offload flags at all for this model. `Makefile` builds it host-only
and offloads only the program's own loops. Also note `fp(np_, np_)` indexed
`(j, i)`: the model's NCHW input is row-major with the row index slowest,
and Fortran is column-major.

Both are real limits of the per-point design for whole-field models. A
conv net over a field on the device wants a tiled kernel with activations
in shared memory, which the generator does not have; that, and the fused
batched GEMM that example B wants, are the two kernels on the TODO list.
For the sizes here the host guess costs 8 ms per step, about what the 100
Jacobi iterations it saves cost, so the example is a wash on time -- which
is the honest answer for a 13-cell guess on a 64² grid.
