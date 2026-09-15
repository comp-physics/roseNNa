# cns_closure: a learned closure inside a compressible Navier-Stokes solver

`cns.c` is a compact 3D compressible Navier-Stokes solver in one file, with a
roseNNa-generated MLP called once per cell inside its own offloaded loop. It
builds, runs and checks itself, on the host and on a GPU.

This example used to be `patch.md`: a documented diff against **microfd**, a
compact compressible solver that is not part of this repository. A patch
against a file we do not control cannot be compiled, run or tested, so
everything about it was unverifiable by construction -- it described how the
closure would plug in and said so honestly ("device path unvalidated"). The
solver here is our own implementation, so the example is now an ordinary
program that either works or fails the build. microfd is still what gave it
its shape: padded blocks, one array per quantity, a `g` struct of pointers,
and `LOCALS`/`IDX` macros. No microfd source is used.

## What this is

- **`cns.c`** -- finite volume on a uniform periodic box: MUSCL reconstruction
  with a minmod limiter, HLLC flux, full Newtonian viscous stress plus Fourier
  conduction, SSP-RK3 in time. Initial condition is a compressible
  Taylor-Green vortex, which is smooth and produces a full velocity-gradient
  tensor for the closure to consume. No MPI: like the other examples here it
  is serial plus OpenMP target offload.
- **`closure.py`** -- builds and exports `closure.onnx`: a 9-input, 16-hidden
  (Tanh), 1-output MLP, deterministically initialized.
- The closure maps the nine components of the local velocity-gradient tensor
  to a turbulent viscosity, which is added to the molecular viscosity at every
  face: `mu_eff = mu + rho * nut`. So the network's output feeds real
  numerics rather than being computed and discarded.

**The network is deterministically initialized, not trained.** Its output is a
fixed but arbitrary function of the gradients, so `cns.c` floors it at zero and
scales it to stay comparable to the molecular viscosity. This example is about
the plumbing and the numerics around a learned closure, not about a physical
model; a trained closure would need neither the floor nor the scale.

## Build and run

```
make                                  # host: gcc, -fopenmp
make TOOLCHAIN=nvidia ARCH=cc80       # nvc, offloaded, OMP_TARGET_OFFLOAD=MANDATORY
make TOOLCHAIN=amd    ARCH=gfx90a     # amdclang
make TOOLCHAIN=nvidia BATCHED=1       # the batched path, linking libclosure.a
make NX=16 NSTEPS=5                   # smaller: NX is cells per direction, so NX^3
```

The Makefile borrows the surrogates' toolchain block
(`../surrogates/common.mk`) rather than restating it, so the offload flags,
the archive rules and the NVIDIA link flags are the same ones the other four
examples use.

Two ways of calling the model are built from the same source:

- **per-point** (default): `closure_infer` from inside the solver's
  `target teams distribute parallel for`. The plan embeds (177 parameters), so
  `closure_infer` is `static inline` in `closure.h` with the weights baked in
  -- no `closure_init`, nothing to link, and nothing allocated, transferred or
  synchronised anywhere in the loop path.
- **batched** (`BATCHED=1`): gather every cell's nine features into one device
  array, then one `closure_infer_batch` call over the whole field. That is the
  right trade for a larger network, where the per-call overhead starts to
  matter. `closure_infer_batch` is not header-inline, so this path links
  `libclosure.a` -- built by `closure.mk` with whichever `ROSENNA_BACKEND` the
  toolchain selects (`cuda`, `hip` or `omp`).

## What the run asserts

It prints `OK` and exits 0 only if all of the following hold, so `make`
succeeding is the assertion:

1. **Mass and total energy are conserved** to round-off, which is what a
   conservative flux-difference update on a periodic box owes you. This is a
   check on the solver, not the network: it is what tells you the closure was
   wired into the viscous flux without breaking conservation.
2. **Density and pressure stay positive** and nothing goes non-finite.
3. **The `nut` field the offloaded solver computed equals a host-side
   evaluation of the same model on the same primitives.** This is the roseNNa
   claim, checked inside a real solver rather than a harness.
4. With `BATCHED=1`, **the batched path agrees with the per-point path** --
   `closure_infer_batch` (the native CUDA/HIP kernel, or the OpenMP fallback)
   against the header-inline `closure_infer`, compared directly rather than
   each against the host.

## Measured

On an A100 80GB with NVIDIA HPC SDK 25.11, `NX=16 NSTEPS=5`, and on the same
machine's host toolchain (gcc 13.3):

| | mass drift | energy drift | nut vs host | batched vs per-point |
|---|---|---|---|---|
| `TOOLCHAIN=gnu` | 4.2e-15 | 2.2e-14 | 0 | -- |
| `TOOLCHAIN=nvidia` | 1.1e-16 | 1.1e-15 | 1.4e-19 | -- |
| `TOOLCHAIN=nvidia BATCHED=1` | 1.1e-16 | 1.1e-15 | 1.4e-19 | 0 |

Both toolchains reach the same minimum density and pressure to every printed
digit, and the native CUDA `infer_batch` kernel and the header-inline `infer`
agree exactly. `TOOLCHAIN=amd` builds from the same source but has not been
run here; the test suite runs it wherever an AMD GPU and `amdclang` are
present.

### One thing worth knowing about nvc -O2

`hllc()` copies the chosen star state into a local array instead of selecting
it through a pointer:

```c
/* not: const double *S = SM >= 0 ? L : R; */
double S[NV], Uk[NV], Fk[NV], Sk;
if (SM >= 0) { ... } else { ... }
```

Selecting between two local arrays through a pointer miscompiles under
`nvc -O2 -mp=gpu`: the kernel dies with `CUDA_ERROR_LAUNCH_FAILED` at the
first write to a mapped array, while `-O1` is correct and
`compute-sanitizer` reports no out-of-bounds access. Copying five doubles
costs nothing next to the flux arithmetic. This is the second nvc codegen
bailout this branch has had to work around -- see
[`doc/nvhpc_teams_mapping/`](../../doc/nvhpc_teams_mapping/) for the first,
which was about where a dense layer adds its bias.

## Validating the device path more broadly

This example checks its own closure against a host evaluation. To check the
whole generated device contract on your hardware:

```
rosenna gpu-gate --cc nvc --fc nvfortran --flags "-mp=gpu -gpu=cc80" \
    --backend cuda --devcc nvcc --out /tmp/rosenna-gate
```

That records `gate-report.md`: every command it ran, every line of output, the
compiler versions, and nanoseconds per point for each harness. `--cc`/`--fc`
must be a HOST compiler capable of OpenMP target offload (the pairing above,
not a plain `gcc` -- `gcc-15` from Homebrew has no offload device to target
and would silently run every per-point harness on the host even though
`--backend cuda` asks for the native kernel). `rosenna gpu-gate --help` lists
the AMD (`amdclang`/`amdflang`/`hip`) and no-GPU pairings too.

Under `--backend cuda|hip` the gate builds two C archives per configuration
(rulings R21/R22): the per-point C harness is compiled and linked by the HOST
compiler with its offload flags against an `omp`-backend archive that the same
host compiler built, and the `.cu` driver of the `infer_batch` harness is
compiled and linked by the device compiler against the `cuda|hip` archive.
That is the same rule a solver has to follow: a per-point OpenMP/OpenACC host
calling `<name>_infer` on a file-loaded model links the `omp`-backend archive
its own compiler built, never the cuda/hip one (see the `Call it from C`
section of `python/README.md`). This closure embeds, so it is not affected --
but the `BATCHED=1` path does link the archive, and does follow that rule.
