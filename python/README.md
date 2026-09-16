# rosenna: ONNX to a GPU-callable Fortran/C library

This is the roseNNa code generator: it reads an ONNX model and emits a
small, self-contained Fortran module and/or C library that a solver written
in C or Fortran links directly, and calls per point inside its own compute
loop -- on the host, or on a GPU under OpenMP target offload, OpenACC, CUDA
or HIP.

## What you get

`rosenna generate model.onnx` writes the sources and build recipes for a C
library and a Fortran module; `make -f <name>.mk` and `make -f
<name>_fortran.mk` then build `lib<name>.a` (C) and `lib<name>_f.a`
(Fortran, a module in the archive). `<name>_infer` is a plain per-point
function you call inside your own GPU loop, exactly like any other
device-callable routine in your solver. Its weights are baked into the
generated source as constants for a small model, or loaded once at startup
by `<name>_init` for a large one, and the generated code is written so that
they live on the device in either case -- with the caveat, stated under
[Verify](#verify), that the device path has not yet been run on a GPU.

## Install

```sh
pip install -e python
```

The generator itself only needs Python (`onnx`, `numpy`, `onnxruntime` for
`verify`). Building generated code needs a compiler:

- host path (no accelerator, or the OpenMP-target host fallback): `gcc`/`gfortran`
  with `-fopenmp` (on macOS, Homebrew's `gcc-15`/`gfortran-15` -- Apple's
  `clang`-based `gcc` has no `-fopenmp`).
- GPU path: `nvc`/`nvfortran` (NVIDIA HPC SDK) for OpenMP-target or OpenACC on
  an NVIDIA GPU; `amdclang`/`amdflang` for OpenMP-target on an AMD GPU; `icx`/`ifx`
  for OpenMP-target on an Intel GPU. `nvcc` or `hipcc` if you also want the
  native batched kernel: `ROSENNA_BACKEND=cuda|hip` when you run the C recipe
  (and `--backend cuda|hip` to `rosenna gpu-gate`); `generate` itself has no
  backend flag and always writes every file (see [Call it from C](#call-it-from-c)).

## Generate

```sh
rosenna generate model.onnx --lang both --precision single --out build/
```

`--lang` selects `fortran`, `c`, or `both` (default); `--precision` selects
`single` or `double` and defaults to the model's own dtype (see
[Precision](#precision)); `--name` sets the symbol prefix and defaults to the
model file's stem. `generate` prints every file it wrote:

| File | Written when | What it is |
|---|---|---|
| `<name>.h` | `--lang c\|both` | the header: `<name>_infer` (per-point, device-decorated), `<name>_infer_batch` |
| `<name>.c` | `--lang c\|both` | weight loading and `<name>_init` (file-loaded models only), the OpenMP-fallback `<name>_infer_batch` |
| `<name>_kernel.cu` | `--lang c\|both` | the native CUDA/HIP batched kernel; inert unless built with `ROSENNA_BACKEND=cuda\|hip` |
| `rosenna_rt.h` | `--lang c\|both` | the CUDA/HIP runtime macro mapping; identical for every model |
| `<name>.mk` | `--lang c\|both` | the C build recipe: builds `lib<name>.a` for `ROSENNA_BACKEND=cuda\|hip\|omp` |
| `<name>_model.F90` | `--lang fortran\|both` | the Fortran module (capital `.F90`: `infer_batch`'s device-pointer clause is chosen by the preprocessor, since nvfortran does not implement `has_device_addr`) |
| `<name>_fortran.mk` | `--lang fortran\|both` | the Fortran build recipe: builds `lib<name>_f.a` |
| `<name>.rwt` | file-loaded weights only | the weights file `<name>_init` reads |

Both recipes write into the same output directory and build there: a
`--lang both` run gives you one directory holding both archives.

A model embeds its weights as constants (`ROSENNA_CONST` in C, an
initialized `protected` module array in Fortran, since gfortran's OpenACC
cannot read a `parameter` array from a device routine) automatically when
it has fewer than `EMBED_THRESHOLD`
(1,000,000) parameters; above that it is file-loaded by default. `--embed-weights`
forces embedding regardless of size; `--no-embed` forces a `.rwt` file
regardless of size. An embedded model has no `<name>_init` at all -- there is
nothing to load -- and no `.rwt` file is written for it.

## Call it from C

Two paths call the same generated code. This example is generated from
`gemm_small` with `--name model`; it embeds by default, so it has no
`model_init` to call (the commented-out line below shows the file-loaded
form). It reads its inputs from a fixed array, calls `model_infer` in its
own offload loop, calls `model_infer_batch` once, and exits non-zero if the
two disagree:

```c
#include <math.h>
#include <stdio.h>
#include "model.h"

#define NPTS 4

int main(void) {
    /* Fixed inputs: NPTS points of n_in=2 values each. */
    double x[NPTS * 2] = {
        0.10, 0.20,
        0.30, -0.10,
        -0.20, 0.50,
        1.00, -1.00,
    };
    double y_loop[NPTS * 3];
    double y_batch[NPTS * 3];
    int status = 0;

    /* File-loaded models only: gemm_small embeds by default, so this
       generated header has no model_init to call.
       if (model_init("model.rwt") != 0) return 1; */

    /* (a) The per-point path: model_infer inside your own offload loop. */
#if defined(_OPENMP)
    #pragma omp target teams distribute parallel for map(to: x[0:NPTS * 2]) map(from: y_loop[0:NPTS * 3])
#endif
    for (int p = 0; p < NPTS; ++p)
        model_infer(x + p * 2, y_loop + p * 3);

    /* (b) The batched path: model_infer_batch takes device-resident data in
       every backend and never allocates, transfers or synchronizes itself.
       Under the omp backend the host maps its own arrays and hands
       infer_batch the mapped device pointers (use_device_ptr needs a
       pointer variable, not an array, hence xp/yp); a cuda/hip caller
       passes raw device pointers here instead and skips this mapping. */
#if defined(_OPENMP)
    {
        double *xp = x, *yp = y_batch;
        #pragma omp target data map(to: x[0:NPTS * 2]) map(from: y_batch[0:NPTS * 3]) \
                                 use_device_ptr(xp, yp)
        {
            status = model_infer_batch(NPTS, xp, yp, NULL);
        }
    }
#else
    status = model_infer_batch(NPTS, x, y_batch, NULL);
#endif
    if (status != 0) return 1;

    for (int i = 0; i < NPTS * 3; ++i) {
        if (fabs(y_loop[i] - y_batch[i]) > 1e-9) {
            fprintf(stderr, "mismatch at %d: %.17g vs %.17g\n", i, y_loop[i], y_batch[i]);
            return 1;
        }
    }
    return 0;
}
```

Generate and build it (`--precision double` here only to keep the example's
own arithmetic in `double` throughout; see [Precision](#precision)):

```sh
rosenna generate model.onnx --lang c --precision double --out build/ --name model
```
```sh
make -f model.mk ROSENNA_BACKEND=omp CC=gcc-15 ROSENNA_OFFLOAD_FLAGS=-fopenmp
```

`ROSENNA_BACKEND` selects which `model_infer_batch` the archive holds --
`cuda`/`hip` build `model_kernel.cu` with `DEVCC` (default `nvcc`/`hipcc`)
and launch the native kernel over raw device pointers; `omp` (the default)
builds only `model.c` with the host compiler and runs the OpenMP-target
fallback shown above. The two are never linked together. A cuda/hip build
of a *file-loaded* model needs one more call: after every `model_init`, call
`model_device_bind_here()` in every translation unit whose kernels call
`model_infer` (an embedded model needs neither).

A cuda/hip archive does not serve the per-point path of a file-loaded
model. `nvcc`/`hipcc` compile `model.c` with `_OPENMP` and `_OPENACC`
undefined, so the weight arrays get no `declare target` device copies and
`model_init`'s `target update` is not compiled; a host translation unit
compiled by `nvc -mp=gpu` (or `amdclang`, or `gcc` with offload) that calls
`model_infer` inside its own offload loop then reads device copies that do
not exist. So a per-point OpenMP or OpenACC host calling `model_infer` on a
file-loaded model must link the `omp`-backend archive built by that same
host compiler with its offload flags:

```sh
make -f model.mk ROSENNA_BACKEND=omp CC=nvc ROSENNA_OFFLOAD_FLAGS="-mp=gpu -gpu=cc80"
```

The cuda/hip archive serves `model_infer_batch` and your own CUDA/HIP
kernels that call `model_infer` after `model_device_bind_here()`. Build
both archives in separate directories if one program needs both. Embedded
models are unaffected: every translation unit holds its own copy of the
constants, so they work with every backend. See [Limits](#limits) for the
planned resolution.

Embedded weights on a CUDA/HIP build go to one of two storage classes,
decided per model at generate time, not at build time: under 48 KB (12,288
float32 or 6,144 float64 parameters) `ROSENNA_CONST` is `__constant__`
(cached, broadcast to every thread reading the same address in a warp); at
or over 48 KB it is `__device__ const` (ordinary global memory), because
CUDA constant memory is 64 KB per module and the cut leaves 16 KB of that
for anything else the translation unit puts there. Both storage classes
compute the same result -- a model over the threshold still runs correctly,
just without the constant-cache broadcast -- and the header's own comment
on `ROSENNA_CONST` states which one a given model got, so check it there if
a per-grid-point call's throughput is on the critical path.

### Whole-field models: infer_one

`<name>_infer_one(x, y, stream)` runs one sample over device pointers as
one launch per op, the thread index over the op's output elements, with
the intermediate activations in static device buffers. It is for a model
whose activations are too large for a thread's locals -- a conv net over
a whole field: `infer` would hold 660 KB of locals for a 3-layer, 8-channel
net on a 64x64 grid, more than a device thread's stack, and for such a
plan `infer_batch` runs `infer_one` per point. The buffers are shared by
every call, so calls on different streams must not overlap. Absent when
the plan has an LSTM (a sequence, not a launch per op). The `omp` archive
provides it as one target loop per op; Fortran reaches it through
`<name>_infer_one_dev`. `examples/surrogates/poisson_guess` uses it.

### No transfers in the loop

`<name>_init` is the plan step and the only routine that allocates or
transfers. Nothing in the loop path -- `<name>_infer` or
`<name>_infer_batch` -- allocates, transfers or synchronizes; the caller
owns the stream (`model_infer_batch`'s last argument), and `infer_batch`
never even looks at it beyond passing it to the launch. An embedded model's
`<name>_infer` is also device-only under `nvcc`/`hipcc` -- its host
instantiation asserts -- so on those compilers call it from a kernel, or use
`infer_batch`.

A solver's time-step loop is what the contract is for: the weights go to
the device once, in `init` or as constants in the device image, and a
loop calling `infer` from its own target region or `infer_batch` once per
step moves no model data. `rosenna gpu-gate` measures that: every harness
runs a 4-step loop over resident data inside a profiler range, and the
transfer count inside the range must be zero. It is, for all three
harnesses, embedded and file-loaded, on the MI210 (`rocprofv3`) and, for
the `infer_batch` driver, on the A100 (`nsys`).

For Fortran solvers: an `allocatable` referenced inside a target region
carries a descriptor, and `amdflang` re-maps that descriptor on every
region entry, two small copies per step for two arrays whose data is
resident. The gate's Fortran harness reaches its arrays through
explicit-shape dummies instead; a solver's step loop should do the same,
or pass `c_ptr`s as the C path does.

Host offload flags, for the per-point path and the `omp` backend:

| Host compiler | Host flags (the per-point path and the `omp` backend) |
|---|---|
| nvc / nvfortran (NVIDIA, OpenMP) | `-mp=gpu -gpu=cc80` (or your `-gpu=` target) |
| nvc / nvfortran (NVIDIA, OpenACC) | `-acc -gpu=cc80` |
| amdclang / amdflang (AMD) | `-fopenmp --offload-arch=gfx90a` (or your arch) |
| icx / ifx (Intel) | `-fopenmp -fopenmp-targets=spir64` |
| gcc / gfortran, host fallback | `-fopenmp` |

`DEVFLAGS`, for the batched backend:

| Batched backend | `ROSENNA_BACKEND` | `DEVFLAGS` |
|---|---|---|
| CUDA | `cuda` | `-O2 -arch=sm_80` (or your arch) |
| HIP | `hip` | `-O2 --offload-arch=gfx90a` (or your arch) |
| OpenMP fallback | `omp` | none; uses the host flags |

## Call it from Fortran

The same two paths, through `use <name>_model`. This is the same
`gemm_small` model as above (`--name model`), built with `--lang fortran`,
so it also embeds and has no `model_init`:

```fortran
program host
    use model_model
    use iso_fortran_env, only: real64
    implicit none
    integer, parameter :: npts = 4
    real(real64) :: x(2, npts), y_loop(3, npts), y_batch(3, npts)
    integer :: p, status

    x(:, 1) = [ 0.10_real64,  0.20_real64]
    x(:, 2) = [ 0.30_real64, -0.10_real64]
    x(:, 3) = [-0.20_real64,  0.50_real64]
    x(:, 4) = [ 1.00_real64, -1.00_real64]

    ! File-loaded models only: gemm_small embeds by default, so this
    ! generated module has no model_init to call.
    ! call model_init('model.rwt', status)
    ! if (status /= 0) stop 1

    ! (a) The per-point path: model_infer inside your own offload loop.
    !$omp target teams distribute parallel do map(to: x) map(from: y_loop)
    do p = 1, npts
        call model_infer(x(:, p), y_loop(:, p))
    end do

    ! (b) The batched path: model_infer_batch takes device-resident arrays.
    ! The host maps its own arrays and hands infer_batch the mapped device
    ! addresses (use_device_addr); a cuda/hip caller reaches the same
    ! contract through model_infer_batch_dev and c_loc of device memory.
    !$omp target data map(to: x) map(from: y_batch) use_device_addr(x, y_batch)
    call model_infer_batch(npts, x, y_batch, status)
    !$omp end target data
    if (status /= 0) stop 1

    if (maxval(abs(y_loop - y_batch)) > 1.0e-9_real64) stop 1
end program
```

`model_infer` is `pure`; `model_infer_batch(n, x, y, status)` returns its
status (0, 10 or 11 -- see [Status codes](#status-codes)) as an `intent(out)`
argument rather than a function result, so it can be called from inside a
plain (non-`pure`) host subroutine. Build and run it:

```sh
rosenna generate model.onnx --lang fortran --precision double --out build/ --name model
```
```sh
make -f model_fortran.mk FC=gfortran ROSENNA_OFFLOAD_FLAGS=-fopenmp
```
```sh
gfortran -O2 -std=f2008 -fopenmp -I. host.f90 -L. -lmodel_f -o host
```

`gfortran` drops `model_model.mod` next to the object it compiles; `-J DIR`
during the library build sends it to `DIR` instead of the current
directory, and a host that `use`s the module then needs `-I DIR` on its own
compile line to find it (`-I.` above, since the example builds both in the
same directory).

A Fortran host reaches the batched path two ways. `model_infer_batch` as
shown above is always Fortran's own OpenMP-target fallback, compiled
straight into `lib<name>_f.a`, so it links nothing else. The module also
declares a second route straight to the native kernel: the `bind(C)`
interface `model_infer_batch_dev`, bound to the plain C symbol
`model_infer_batch` that `lib<name>.a` provides -- whichever kernel its
`ROSENNA_BACKEND` was built with (see [Call it from C](#call-it-from-c)).
That route needs `c_ptr`s to device-resident memory, which OpenACC's
`host_data use_device` produces from a mapped Fortran array:

```fortran
use iso_c_binding, only: c_loc, c_null_ptr
integer :: status
!$acc host_data use_device(x, y_batch)
status = model_infer_batch_dev(npts, c_loc(x), c_loc(y_batch), c_null_ptr)
!$acc end host_data
```

and links both archives plus the runtime the cuda/hip archive was built
against, which a host that is not itself linked by `nvcc`/`hipcc` has to
name explicitly: `-lmodel_f -lmodel -L$CUDA_HOME/lib64 -lcudart` for CUDA
(or `nvfortran -cuda`, which links it for you), `-lmodel_f -lmodel
-L$ROCM_PATH/lib -lamdhip64` for HIP. With an `omp`-backend `libmodel.a`
nothing extra is needed.

`model_infer` is not itself inlined across the `use model_model` boundary by
every compiler, so a Fortran host's own offload loop generally gets a real
call per point, not an inlined one, unless the build enables cross-module
inlining (`gfortran -flto`, nvfortran `-Minline`).

## Precision

`--precision` defaults to the model's own dtype -- `float32` for a PyTorch
export via `torch.onnx.export`, since that is what PyTorch trains and
exports in. A double-precision host can still call single-precision
generated code: `model_infer`'s `x`/`y` are the plan's own C `float` /
Fortran `real(real32)`, so the host converts at the call site -- an
implicit narrowing conversion for a C `double` array passed element by
element, or an explicit `real(x, real32)` going in and `real(y_f32, real64)`
coming back out in Fortran. `--precision single` is the usual GPU choice
regardless of the host's own precision: consumer and even most datacenter
GPUs run FP64 at a small fraction of their FP32 throughput, so a solver
whose accuracy budget tolerates it gets a substantial speedup from
generating (and calling) the single-precision code even from a
double-precision caller.

## Status codes

`<name>_init` and `<name>_infer_batch` return one of these (rendered here
from `rosenna.abi.STATUS_CODES`, the one place the table is defined):

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | cannot open the weights file |
| 2 | not a roseNNa weights file (bad magic) |
| 3 | weights file version is not supported |
| 4 | weights file dtype does not match this generated code |
| 5 | weights file endianness does not match this machine |
| 6 | weights file plan hash does not match this generated code |
| 7 | weights file holds a tensor this model does not declare |
| 8 | a name or rank in the weights file exceeds this model's capacity |
| 9 | a read failed: the weights file is truncated or inconsistent |
| 10 | device allocation or copy failed in init |
| 11 | kernel launch failed |

Codes 0-9 are `<name>_init`'s; `<name>_infer_batch` only ever returns 0, 10
or 11 (10 and 11 are cuda/hip only -- the `omp` backend's fallback loop
cannot itself fail once its arguments are device-resident, so it always
returns 0).

## Verify

```sh
rosenna verify model.onnx --lang both --cases 16
```

`verify` generates, compiles and runs the per-point `<name>_infer` path on
the host, for one or both languages, and compares its output against
onnxruntime running the same model over the same random inputs. It proves
the generated arithmetic is correct on the host; it never builds or runs the
batched device path (`<name>_infer_batch`, the native kernel, or the
`omp`/`acc` fallbacks under a real offload device), because that needs a GPU
this machine may not have.

The comparison tolerance is keyed on the ONNX model's own dtype, not on
`--precision`: onnxruntime always computes a float32 model's reference in
float32, so `rosenna verify --precision double` on a float32 PyTorch export
is still compared at float32 tolerance (`rtol=1e-5`, `atol=1e-6`), not
float64, however precisely the generated code itself computes. A genuinely
float64 ONNX model is compared at the tight tolerance (`rtol=1e-9`,
`atol=1e-12`) regardless of `--precision`.

```sh
rosenna gpu-gate --help
```
```sh
rosenna gpu-gate --cc gcc --fc gfortran --flags=-fopenmp --backend omp --host-fallback --out gate-report/
```

`gpu-gate` is the check that does exercise the device path: on a machine
with a real accelerator (and the matching compilers -- `--help` lists the
NVIDIA, AMD and no-GPU pairings), it generates a model, builds it for the
chosen `--backend`, and runs three harnesses -- a per-point C host, a
per-point Fortran host, and a host that hands device-resident data to
`infer_batch` -- each compared against onnxruntime and timed, writing every
command and its output to `gate-report.md`.

The CUDA device path has been validated. `gpu-gate` was run on an NVIDIA
A100 80GB (driver 590.48.01) with NVIDIA HPC SDK 25.11 -- `nvc`/`nvfortran`
`-mp=gpu -gpu=cc80` as the host compilers, `nvcc` 13.0 as the device
compiler -- and reported `PASS: every configuration matched`: all six
harnesses (embedded and file-loaded x per-point C, per-point Fortran,
`infer_batch`) matched onnxruntime, and the `nsys` capture scoped to the
timed `infer_batch` call recorded **zero** `cudaMemcpy` calls in both
configurations. Per point, over the same million distinct points with the
data mapped outside the timed window in every harness, every route through
the library lands within noise of every other: 1.7 ns calling `infer` from
a C `target teams distribute parallel for`, 1.7-1.8 ns from the Fortran
equivalent, and 1.5 ns through the native batched kernel, embedded and
file-loaded alike. That is what should be expected of the same arithmetic
over the same data, and two changes were needed to get there.

The first is how a dense layer is written. Write it the obvious way --
seed the accumulator with the bias, `acc = b[i]`, then
add the dot product -- and nvc refuses to generate a `distribute parallel
for` body at all: it emits a kernel that traps at runtime. Add the bias
*after* the dot product instead and the same loop compiles and runs. Both
emitters do it that way, so the two backends stay bit-comparable. Without
that workaround the only form nvc accepts is `target teams loop`, which
maps one point to one *team* -- 1,000,000 blocks of 32 threads with a
single active lane each -- and costs ~47-49 ns per point, some 30x more.

The second is where the embedded weights live. `__constant__` memory is
fast only while the working set fits a per-SM cache of a couple of KB; past
that every read misses, and `ncu` showed embedded `infer_batch` spending
71% of its warp-issue stalls on constant-cache misses. Embedded weights now
go to `__constant__` only below 2 KB and to `__device__ const` above it,
which took embedded `infer_batch` from 4.7 to 1.5 ns per point.

See [`doc/nvhpc_teams_mapping/`](../doc/nvhpc_teams_mapping/) for the
PTX, the `ncu` geometry and stall counters, and a self-contained
reproducer.

The HIP path was validated the same way: `gpu-gate --backend hip` on an
AMD Instinct MI210 (gfx90a), under ROCm 7.2.0 (`amdclang` / `amdflang`
`-fopenmp --offload-arch=gfx90a`, `hipcc`) and under the TheRock AFAR
23.2.1 drop, `PASS` both times. Per
point: 2.0-3.1 ns for the C per-point host, 4.6-4.8 ns Fortran, 4.0 ns
through the native HIP kernel embedded and 6.6-6.9 ns file-loaded. Three
changes were needed, none in the generated arithmetic:

- `__HIP__` is defined by clang's OpenMP AMDGPU device pass (from
  `openmp_wrappers/math.h`), so a header that accepted it took the
  `__device__` branch inside a plain OpenMP host build. hip-clang defines
  `__HIPCC__` for any HIP compilation; the guards test only that.
- `hipcc` does not include its runtime implicitly as `nvcc` does; the
  gate's device harness includes `rosenna_rt.h`.
- `hipcc` puts `-x hip` ahead of a `.cu` input and it applies to every
  later input, so a bare `lib.a` after the `.cu` was compiled as source.
  The gate links the archive as `-L`/`-l`.

Transfer evidence on HIP: `rocprofv3` has no capture range, so the gate
rebuilds each harness with a roctx range around its 4-step loop, runs it
under `--hip-trace --marker-trace --memory-copy-trace`, and cuts both the
HIP API trace and the memory-copy trace to the range (a small `hipMemcpy`
is host-staged and never a copy operation; OpenMP offload's copies go over
HSA and are never an API call). Zero transfers inside the loop for all
three harnesses, embedded and file-loaded; the drivers' setup copies and
`init`'s upload are in the same traces outside the range. On CUDA the
same loops are bracketed with nvtx and `cudaMemcpy*` plus `cuMemcpy*`
counted in `nsys`'s `cuda_api_sum`; run on an A100 for the `infer_batch`
driver, not yet for the per-point harnesses. A compile-only `nvcc` job
exists in CI (`.github/workflows/CI.yml`, `nvcc_compile`).

For runnable examples of a model inside a solver's time loop, C and
Fortran, see `examples/surrogates/`; `examples/cns_closure/` is a compressible
Navier-Stokes solver with a learned per-cell closure.

## Limits

- The suite parallelises: `python3 -m pytest tests -n auto` runs it across
  every core, which is a 5.5x cut here (264s -> 47s on 16 workers) and gives
  byte-identical coverage. Most of it is compiling and running generated code,
  so it scales with cores rather than having any one hot test. Golden-model
  generation takes a lock, so a cold tree is safe too.
- Supported ops: `Gemm`, `MatMul`, `Conv` (grouped/depthwise too), `Pad`, `Softmax`,
  `BatchNormalization` (folded into the preceding `Conv`/`Gemm`), `MaxPool`, `AveragePool`, `LSTM`,
  `Add`, `Concat`, `Reshape`, `Transpose`, `Squeeze`, `Unsqueeze`, `Flatten`,
  `Identity`, `Relu`, `Tanh`, `Sigmoid`. Values may be rank 1 to 4; the
  spatial ops are 2-D (rank-4 NCHW) only, `Conv` must be ungrouped, and
  `LSTM` must be forward-direction with the default activations; its
  initial state may be a graph input (it arrives in `x`) or a constant (it
  becomes a weight). A `Gemm` bias must have one value per output, not a
  broadcast `(1,)`. `Concat` joins runtime values and constants along one
  axis. No batch norm, no `Softmax`, no `Pad` node, no GRU.
- Several inputs and several outputs are fine. Inputs arrive concatenated
  in `x` in declaration order and outputs leave concatenated in `y`, so
  `infer(x, y)`, `infer_batch`, the native kernel and the device contract
  are unchanged. `rosenna info` prints both layouts (`x: p[0:1] h[1:5]
  c[5:9]`, `y: Y[0:4] hn[4:8] cn[8:12]`). A recurrent model with
  `initial_h`/`initial_c` as graph inputs and `Y_h`/`Y_c` as graph outputs
  is called once per cell per step, and `y`'s state slices go back into
  `x` for the next step on the device.
- Everything constant is folded away at generation time, so a `Reshape` of a
  weight or an int64 shape tensor never reaches the generated code. A
  relabelling op on a runtime value (`Reshape`, `Squeeze`, `Flatten`, and any
  `Transpose` that only moves size-1 axes) becomes a buffer alias: no code,
  no copy.
- A file-loaded model's `<name>_infer` reads unset (zero-initialized static)
  weights if `<name>_init` was never called, or failed, before it. Nothing
  in the loop path checks this -- checking it there would be the transfer
  and synchronization ruled out under [No transfers in the loop](#no-transfers-in-the-loop).
  The `omp` backend's `<name>_infer_batch` fallback calls `<name>_infer` per
  point and has the same silent behavior; only the cuda/hip path's
  `<name>_infer_batch` catches this, returning status 10.
- A cuda/hip archive of a file-loaded model serves `<name>_infer_batch` and
  CUDA/HIP kernels only; a per-point OpenMP/OpenACC host must link the
  `omp`-backend archive built by its own compiler (see [Call it from
  C](#call-it-from-c)). The planned resolution is that the host compiler
  always compiles `<name>.c` and `<name>_kernel.cu` owns every CUDA/HIP
  symbol behind `-DROSENNA_NATIVE_KERNEL`, so one archive serves both paths.
- The native batched kernel (`ROSENNA_BACKEND=cuda|hip`) launches one thread
  per point. Dense layers at least 96 wide compute 8 output columns per
  pass over the input vector (`GEMM_BLOCK`), which is what made a
  128-wide MLP 3.4x faster on an MI210; staging weights in shared memory
  was measured slower. An LSTM's layers are not blocked.
- There is no SYCL backend. An Intel GPU is reached through the `omp`
  fallback (`icx`/`ifx` with `-fopenmp -fopenmp-targets=spir64`), not a
  native kernel.
