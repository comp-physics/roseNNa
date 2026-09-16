# Coarse-grid Burgers with a learned subgrid closure

Hyperbolic PDE; the network is a per-cell closure called from the solver's
own offload loop. Embedded model, header-inline `closure_infer`, no `init`,
nothing to link. The solver's arrays are mapped once; nothing moves inside
the time loop.

```
make TOOLCHAIN=amd | nvidia | gnu      # gnu: host only, use NB=4
```

An ensemble of `NB` periodic Burgers realizations (three random sine
modes each). For each: the fine reference (2048 cells, 64 sub-steps per
coarse step) box-filtered to the coarse grid; the coarse scheme alone
(128 cells, Godunov flux, central viscous term, forward Euler); the coarse
scheme plus `closure_infer(stencil, corr)` on every cell. Exits 0 if the
closure reduces the mean error after 200 steps. MI210, both languages:

```
64 realizations, 128 coarse cells, 200 steps; mean relative L2 error vs filtered fine:
  coarse             4.3532e-02  (47.7 ms)
  coarse + closure   3.1346e-02  (299.6 ms, 153.7 ns per cell-step for the closure)
  fine, 2048 cells   (3234.7 ms)
OK: error reduced 1.4x
```

## Model

`train.py`: 7-point stencil → 64 → 64 → 1, `tanh`. Trained with the coarse
solver in the loop (a 200-step rollout in torch, relative trajectory
error). An a-priori fit to the residual on filtered-fine states made the
coarse run worse, since at run time the closure sees its own drifting
state. `closure.onnx` is checked in (training takes a few minutes);
`make train` rebuilds it.

## Notes

- One 128-cell realization cannot occupy a GPU and a kernel that small is
  latency-bound; the ensemble is what makes the timing mean anything. Even
  so the closure's ~150 ns per cell-step here is mostly latency; `rosenna
  gpu-gate` measures ~2 ns per point over a million points.
- `burgers.F90` passes arrays into target regions as explicit-shape
  dummies. `amdflang` re-maps an allocatable's descriptor on every region
  entry, two small copies per array per step.
