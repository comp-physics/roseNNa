# Two performance cliffs on the NVIDIA path, and what was behind them

Measured on an NVIDIA A100 80GB (driver 590.48.01) with NVIDIA HPC SDK
25.11 (nvc/nvfortran 25.11-0, nvcc 13.0.88), via `rosenna gpu-gate
--backend cuda`, over a million distinct points with the data mapped
outside the timed window in every harness.

| ns per point | before | after |
|---|---|---|
| `infer` from a C per-point offload loop | 49.5 | **1.7** |
| `infer` from a Fortran per-point offload loop | 47.3 | **1.8** |
| `infer_batch`, native CUDA kernel, file-loaded | 1.54 | 1.49 |
| `infer_batch`, native CUDA kernel, embedded | 3.3 | **1.46** |

Every route through the library now lands within noise of every other,
which is what the same arithmetic over the same data should cost. Three
changes got there, and the first two are entangled:

1. The generated dense layer adds the bias *after* the dot product instead
   of seeding the accumulator with it -- without which nvc will not compile
   the loop below at all.
2. The host loop uses `target teams distribute parallel for` rather than
   `target teams loop`, which is what puts all 32 lanes of a warp to work.
3. Embedded weights go to `__constant__` only below 2 KB, not below 48 KB.

The rest of this file is the evidence for each.

## Why `teams loop` cost 30x

It is not code quality, inlining or LTO. `ptxas -v` on the two kernels:

| | registers/thread | stack frame | spill st/ld |
|---|---|---|---|
| nvc `nvkernel_main_F1L41_4` | 140 | 0 B | 0 / 0 |
| nvcc `gemm_big_kernel` | 96 | 320 B | 0 / 0 |

nvc's per-thread code is the better of the two -- it keeps the body in
registers where nvcc spends a 320-byte frame -- and neither spills.
`-Minline`, `-Minline=maxsize:2000,levels:5` and `-Mnoinline` change
nothing.

It is the loop-to-hardware mapping. `ncu` launch geometry:

```
nvc    nvkernel_main_F1L41_4   (1000000, 1, 1) x (32, 1, 1)
nvcc   gemm_big_kernel         (   7813, 1, 1) x (128, 1, 1)
```

nvc maps one loop iteration to one **team** -- one point per thread block,
32 threads per block, and no inner `parallel` for the other 31 lanes to do.
**31 of every 32 lanes idle**, which is the factor observed (49.5 / 1.54 =
32.1).

The PTX says why the block cannot be used. nvc outlines `gemm_big_infer`
and places its two 40-double locals in *dynamic shared memory*, at fixed
offsets, with no per-thread indexing:

```ptx
.extern .shared .align 8 .b8 S52_1[];
...
st.shared.f64   [S52_1],    %fd50;
st.shared.f64   [S52_1+8],  %fd57;
```

`ncu` confirms 896 bytes of dynamic shared per block -- one point's worth.
That storage is team-shared, correct only while a single thread per team
runs the body, which is exactly what `teams loop` arranges. Self-consistent,
and it costs a factor of 32.

### Confirmed with no compiler in the way

`mapping_emulation.cu` runs the *same* generated `infer` body as a
hand-written CUDA kernel, two ways -- no OpenMP anywhere:

```
128 thr/block, all lanes active :    1.49 ns/point
32 thr/block, lane 0 only (nvc) :   67.80 ns/point
```

The mapping alone reproduces the gap.

## Why `distribute parallel for` did not simply work

It is the idiom that puts every lane to work, and under nvc it used to
abort:

```
Fatal error: expression 'HX_CU_CALL_CHECK(__hx_cuStreamSynchronize(stream))'
(value 1) is not equal to expression 'HX_SUCCESS' (value 0)
```

`compute-sanitizer` reported 208,769 `Trace/breakpoint trap`s inside the
kernel. Not a race and not a resource limit: nvc declines to generate the
loop, and the whole kernel is a 62-line PTX stub that computes the trip
count and traps if any thread has work:

```ptx
        setp.lt.s64     %p5, %rd12, 1;
        @%p5 bra        $L__BB1_6;      // no iterations: return
$L__BB1_6:
        ret;
$L__BB1_5:
        trap;                           // any iterations: trap
```

It survived every obvious remedy: `declare simd`; a manually blocked
`teams distribute` over tiles with an inner `parallel for`; scratch passed
in as parameters; scratch declared inside the loop body; the body fully
inlined with no device routine at all; `thread_limit` 32, 64, 128 and 256;
`-Minline` and `-Mnoinline`. Embedded and file-loaded alike.

### The actual trigger

Bisecting a standalone reproducer down to the line, what nvc cannot
generate is **an accumulator initialised directly from a declare-target
array element** inside a `distribute parallel for` region:

```c
double s = bb[0]; for (int i = 0; i < 40; ++i) s += t[i];   /* traps  */
double s = 0.0;   for (int i = 0; i < 40; ++i) s += t[i]; s += bb[0];   /* fine */
double s = 0.0;   for (int i = 0; i < 8;  ++i) s += bb[i];             /* fine */
```

Which is exactly the shape a dense layer is written in, once per layer:

```c
for (int i = 0; i < 20; ++i) {
    double acc = gemm_big_b0[i];                  /* <- the trigger */
    for (int j = 0; j < 2; ++j) acc += x[j] * gemm_big_w0[i * 2 + j];
    t0[i] = acc;
}
```

`emit_c.py` and `emit_fortran.py` now emit the bias afterwards instead:

```c
for (int i = 0; i < 20; ++i) {
    double acc = 0.0;
    for (int j = 0; j < 2; ++j) acc += x[j] * gemm_big_w0[i * 2 + j];
    acc += gemm_big_b0[i];
    t0[i] = acc;
}
```

Both emitters changed together, so the C and Fortran backends still agree
to 1e-12, and both still match onnxruntime. It does reassociate the sum by
one term, so results can differ from the old code in the last ulp.

`teams_mapping_repro.c` is a 45-line self-contained reproducer -- no
roseNNa headers, no library:

```
nvc -O2 -mp=gpu -gpu=cc80 teams_mapping_repro.c -lm -o repro && ./repro
  teams loop                                  -> OK
  teams distribute parallel for               -> Aborted (core dumped)
  teams distribute parallel for, bias after   -> OK
```

The bailout is a compiler defect and should go to NVIDIA; the bias
reordering is a workaround, not a fix.

## What this means for a solver

Use `target teams distribute parallel for` (C) or `target teams distribute
parallel do` (Fortran) around your per-point `infer` call, as
`microfd_closure/patch.md` and the examples in `python/README.md` now do.
`target teams loop` still compiles and still gives correct answers -- it
just runs about 30x slower, because it leaves 31 of every 32 lanes idle.

Both paths transfer nothing in the loop and both match onnxruntime, so the
choice between per-point and `infer_batch` is now about which shape fits
your solver, not about speed.

## The second cliff: where embedded weights live

With the per-point path fixed, embedded `infer_batch` still cost 4.7
ns/point against file-loaded's 1.7. Same arithmetic, same kernel, different
weight storage: embedded weights went to `__constant__`, file-loaded ones to
ordinary device memory behind a `__constant__` pointer table.

`ncu` on the two kernels:

| | duration | `imc_miss` stall | `long_scoreboard` stall |
|---|---|---|---|
| embedded (`__constant__`) | 4.26 ms | **71.4%** | 0.7% |
| file-loaded (device memory) | 1.92 ms | 0.09% | 28.6% |

`imc_miss` is the immediate-constant-cache miss. Constant memory is fast
only while the working set fits a per-SM cache of a couple of KB; gemm_big
embeds 23 KB of weights, so nearly every read misses. The file-loaded path
reads the same values through L1/L2, and its 28.6% `long_scoreboard` is
ordinary, well-hidden memory latency.

Forcing each dense golden model to the other qualifier, one thread per
point, a million points, identical outputs throughout:

| model | weight bytes | `__constant__` | `__device__ const` |
|---|---|---|---|
| gemm_small | 120 | 0.027 ns/pt | 0.028 ns/pt |
| gemm_nobias | 160 | 0.027 | 0.028 |
| droplet | 344 | 0.041 | 0.041 |
| batchnet | 15,904 | 6.731 | **2.384** |
| gemm_big | 23,208 | 3.574 | **1.423** |

`CONSTANT_MEMORY_LIMIT` was 48 KB -- chosen against the 64 KB per-module
bank, which is a correctness bound, not a performance one. It is now 2 KB:
the three models that measure the same keep `__constant__`, and the two that
pay 2.5-2.8x move to `__device__ const`.

### The caveat on that threshold

A byte count is not really the right control. Synthetic models with a
*wide* input (16 values per point rather than 2) thrash the constant cache
just as hard -- 67% `imc_miss` at 18 KB of weights -- and yet still come out
~1.3x faster in `__constant__` than in device memory, because streaming a
wide `x` puts enough pressure on L1 to change the balance:

| weight bytes | `__constant__` | `__device__ const` |
|---|---|---|
| 2,312 | 0.130 | 0.118 |
| 4,616 | 0.338 | 0.431 |
| 18,440 | 1.623 | 2.025 |
| 36,872 | 3.921 | 5.417 |

2 KB is calibrated for the shape this library targets: a per-point closure
with a handful of inputs, where the weights dominate the cache. If a
wide-input model ever turns up, this should become a generate-time flag
rather than a different constant.
