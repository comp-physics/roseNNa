# microfd closure: a worked example

The device path this repository generates (OpenMP-target, OpenACC, CUDA and
HIP hosts; a native batched kernel; the same contract from C and Fortran) is
validated by running `rosenna gpu-gate` on a machine with a real accelerator
and recording the report it writes. Until that run has happened and its
`gate-report.md` is on record, everything below -- including this example --
is host-validated only: it compiles and runs on the host; the device path is
unvalidated.

## What this is

microfd is a compact, single-file 3D compressible Navier-Stokes solver
(finite volume, WENO5-Z + HLLC, SSP-RK3, MPI + OpenMP target offload). This
directory shows how a rosenna model plugs into it as a per-cell turbulence
closure: a small MLP that maps the nine components of the local
velocity-gradient tensor to a turbulent-viscosity correction, evaluated once
per cell inside microfd's own offloaded loop. `patch.md` is written against
the reference copy of `microfd.c` used to design it; nothing here compiles
microfd itself, and no microfd source is vendored into this repository.

- `closure.py` -- builds and exports `closure.onnx`: a 9-input, 16-hidden
  (Tanh), 1-output MLP, deterministically initialized.
- `patch.md` -- the exact edits to `microfd.c` (as a documented diff, not
  applied to any repository -- nothing here compiles microfd) that add the
  closure to the solver's viscous flux, plus the alternative call convention
  for a larger network.

## Generating the code

```
cd python
python3 examples/microfd_closure/closure.py            # writes closure.onnx
python3 -c "
from rosenna.cli import main
main(['generate', 'examples/microfd_closure/closure.onnx',
      '--lang', 'c', '--precision', 'double', '--out', 'examples/microfd_closure/gen'])
"
```

`--precision double` matches microfd's own arithmetic, which is `double`
throughout; the default (the model's own dtype, float32) would otherwise
force a cast at every call site. The model is tiny (177 parameters, well
under the 1,000,000-parameter embed threshold), so it embeds by default:
`generate` writes `closure.h` with the weights baked in as `ROSENNA_CONST`
arrays, `closure.c` (the OpenMP-fallback `infer_batch`, needed only for the
batched alternative in `patch.md`), `closure_kernel.cu` (the native CUDA/HIP
`infer_batch`, likewise), `closure.mk`, and no `.rwt` file -- an embedded
plan has nothing to load at runtime.

Because the plan embeds, the per-point path in `patch.md` needs only
`#include "closure.h"`: `closure_infer` is `static inline` and fully
resident in the header, so there is no `closure_init` to call and nothing to
link into microfd's own build. The batched alternative (`closure_infer_batch`)
is not header-inline -- it is always defined in `closure.c` / the kernel's
`.cu` file, whichever `ROSENNA_BACKEND` `closure.mk` was built with -- so
that path does link `libclosure.a`.

## Validating on a GPU machine

```
rosenna gpu-gate --cc nvc --fc nvfortran --flags "-mp=gpu -gpu=cc80" \
    --backend cuda --devcc nvcc --out /tmp/rosenna-gate
```

records `gate-report.md`: every command it ran, every line of output, the
compiler versions, and nanoseconds per point for each of the three
harnesses. `--cc`/`--fc` must be a HOST compiler capable of OpenMP target
offload (the pairing above, NVIDIA HPC SDK's `nvc`/`nvfortran`, not a plain
`gcc` -- `gcc-15` from Homebrew, for instance, has no offload device to
target and would silently run every per-point harness on the host even
though `--backend cuda` asks for the native kernel); `rosenna gpu-gate
--help` lists the AMD (`amdclang`/`amdflang`/`hip`) and no-GPU
(`gcc`/`gfortran`/`omp --host-fallback`) pairings too. Run it with `--backend
hip --devcc hipcc` on an AMD GPU, or `--backend omp` to check the
OpenMP-target fallback on either. Only after that report exists for the
backend and hardware you actually run microfd on should the closure above be
described as device-validated rather than host-validated.
