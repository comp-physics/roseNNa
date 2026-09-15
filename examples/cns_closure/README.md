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
make TOOLCHAIN=nvidia NO_CLOSURE=1    # no network at all: the plain NS solver
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
5. **Kinetic energy decreases.** A periodic box with no forcing can only lose
   kinetic energy, to viscous and numerical dissipation. This is the one
   assertion about the flow rather than about the code, and it is what would
   catch a closure wired in with the wrong sign -- negative turbulent
   viscosity adds energy.

The conservation bound grows with the step count (`1e-12 + 1e-15 * NSTEPS`)
rather than sitting at a fixed value a long run would trip for no reason:
drift is 2.2e-14 at 5 steps on the host and 1.3e-13 at 2000 steps at 128^3,
while a genuinely non-conservative update is wrong by many orders more.

## Measured

Correctness, on an A100 80GB with NVIDIA HPC SDK 25.11 at `NX=16 NSTEPS=5`,
and on the same machine's host toolchain (gcc 13.3):

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

### The flow

`NX=128 NSTEPS=2000` takes the Taylor-Green vortex to t = 1.19 (Re = 1000,
M = 0.1):

```
128x128x128, 2000 steps, dt 5.950e-04, t 1.1901
  mass drift      1.276e-13 (relative)
  energy drift    9.619e-14 (relative)
  min rho / p     0.997583 / 71.187575   non-finite cells 0
  kinetic energy  3.100628e+01 -> 3.037493e+01  (-2.036%, dissipating)
  closure mu_t    mean 7.350e-05, max 3.101e-04   (molecular mu 1.000e-03)
  closure nut vs host evaluation: worst |device-host| 2.168e-19
OK
```

The closure contributes about 7% of the molecular viscosity on average and up
to 31% at its peak, so it is changing the dissipation rather than rounding off
under it -- which is the point of adding a closure at all. (At `NX=64` the
timestep is twice as large, so the same 2000 steps reach t = 2.38 and lose
16.6% of the kinetic energy: coarser grid, longer time, much more numerical
dissipation. Not a resolution study.)

### Speed of the solver itself, and what the closure adds

`NO_CLOSURE=1` drops the network entirely -- `nut` stays zero, `mu_eff = mu`,
and the program is a plain compressible Navier-Stokes solver. That is the
baseline the closure's cost is measured against, rather than inferred from its
share of a run that always includes it.

A100 80GB, 100 steps, fp64, on an otherwise idle card:

| | plain solver | with the closure | | |
|---|---|---|---|---|
| | ms/step (Mcell-updates/s) | ms/step (Mcell-updates/s) | slowdown | closure cost |
| 64^3 (0.26 M cells) | 1.565 (168) | 1.833 (143) | 1.17x | 1.02 ns/cell-step |
| 128^3 (2.10 M) | 6.941 (302) | 8.680 (242) | 1.25x | 0.83 ns/cell-step |
| 256^3 (16.8 M) | 54.50 (308) | 68.00 (247) | 1.25x | 0.81 ns/cell-step |

The plain solver saturates at about **308 Mcell-updates per second**, 3.25 ns
per cell per timestep -- where a step is SSP-RK3, so three full RHS
evaluations, each with MUSCL reconstruction, HLLC, the full Newtonian stress
and Fourier conduction. Adding the closure costs 25% of wall-clock and
0.81 ns per cell-step, which is three `closure_infer` calls at the 0.28 ns
measured below. 64^3 is too small to fill the card, which is why it is both
slower per cell and cheaper in relative terms.

For contrast, the same plain solver on this machine's host toolchain at 64^3:
281 ms/step on one thread, and **411 ms/step on all 128** -- slower with more
threads. These loops are written for offload, and gcc's host fallback
oversubscribes them exactly as `test_examples.py` documents for the
surrogates. Treat the host path as a correctness fallback, not a CPU baseline;
a CPU-tuned version of this solver would look nothing like it.

### Speed of the closure call, A100 80GB, 20 steps

The run prints this itself. Each step is SSP-RK3, so three closure calls;
"per cell per call" divides by the range the closure covers.

These come from an otherwise-idle GPU. **The numbers move by more than an
order of magnitude on a shared machine** -- the same 64^3 case measured 2.0 ms
per step on a quiet card and 63 ms per step with four other jobs saturating
all four A100s, and the giveaway was 64^3 and 128^3 reporting the same total
time, which is per-launch stall and not compute. Check `nvidia-smi` before
trusting any of this, and pin with `CUDA_VISIBLE_DEVICES`.

| | ns per cell per call | closure share of the step |
|---|---|---|
| per-point, 64^3 | 0.30 | 14.4% |
| per-point, 128^3 | **0.28** | 21.7% |
| batched, 64^3 | 0.56 | 23.9% |
| batched, 128^3 | 0.43 | 29.4% |
| host (gcc, 128 threads), 64^3 | 65 | 10.7% |
| host (gcc, 1 thread), 64^3 | 305 | 51.6% |

At 128^3 (2.2M cells) a closure call is 0.6 ms and the whole step 8.5 ms, so
the learned closure costs about a fifth of a compressible Navier-Stokes step
that is already doing MUSCL, HLLC and a full viscous stress. The A100
per-point path is ~220x the host's 128 threads and ~1100x one thread.

**The per-point path is faster than the batched one here, and the reason is
worth knowing.** Splitting the batched path at 128^3 gives gather 42.7 ms,
`infer_batch` plus its sync 13.4 ms, rescale 1.6 ms. So the native batched
kernel is the cheapest part -- 0.097 ns per cell, about a third of A100 fp64
peak for a network with 16 `tanh` -- and the gather that feeds it costs three
times the inference, because it writes nine doubles per cell and reads them
straight back. The fused per-point path never materialises the features: it
reads the primitives it needs and keeps the nine gradients in registers.

Batching wins when the network is large enough that per-call overhead
dominates that extra traffic. For 177 parameters it does not, and this is the
measurement to repeat before choosing the batched path for a bigger closure --
not a reason to avoid it.

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

### `infer_batch` is asynchronous

`closure_batched()` calls `closure_sync(0)` after `closure_infer_batch`, before
anything reads `nut`. `closure.h` says the launch "is asynchronous on it", and
the rescale loop right after it reads what the kernel wrote.

Without that wait the example still printed the right answer, every time,
because nvc's OpenMP target regions happen to serialize against the CUDA
default stream. That is an implementation accident, not a guarantee, and it is
the kind of thing that works until it is someone else's compiler. `closure_sync`
is the backend-agnostic wait: a stream synchronize in the cuda/hip archive, a
no-op in the omp one, whose loop is already synchronous.

It also made the timing lie. With no sync, `infer_batch` measured 0.65 ms for
60 launches over 2.2M cells -- 68 TFLOP/s of fp64, seven times what the card
can do -- because the cost was landing in the next synchronizing region and
showing up as an absurdly slow rescale.

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
