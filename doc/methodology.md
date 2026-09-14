# Pipeline

roseNNa turns an ONNX model into Fortran and C source at *generation* time. The
generated code contains no parser, no shape logic, no allocation and no mutable
global state: every loop bound is a literal, so it inlines into a solver's own
compute kernel, including a device kernel.

The pipeline is a chain of graph-to-graph passes in `python/rosenna/`, each of
which either resolves something or refuses the model by name.

## 1. Load — `frontend.py`

`onnx.shape_inference` first, so every value has a literal shape. The result is
a `Graph` of `Node`s, `Tensor` values, and initializer arrays. A symbolic
dimension is refused here: roseNNa fixes every shape at generation.

## 2. Fold — `fold.py`

Two passes run before anything else looks at the graph.

`fold_constants` evaluates every node whose inputs are all constants and turns
the result into an initializer. A real export is full of these: the shape
tensor of a `Reshape`, a `Constant` holding an LSTM's initial state, a weight
transposed once on the way in. Running them now is also what removes the int64
tensors the generated code could never carry.

`strip_shape_inputs` then drops the metadata operands of relabelling ops, and
any initializer nothing reads any more.

## 3. Validate — `validate.py`

Refuses, by node name, anything the emitters cannot lower: an unsupported op, a
rank the loop nests do not implement, a `Conv` with `group > 1`, an `LSTM` with
custom activations. Every rule here exists because the alternative is a model
that runs and returns confident nonsense — which is the failure mode this file
exists to prevent.

## 4. Plan — `plan.py`

Lowers the graph to an explicit `Plan`: a list of `Op`s, a set of flat rank-1
buffers, and a weight layout.

- **Shapes become arithmetic.** Buffers stay rank 1 and row-major whatever the
  value's logical rank; a `Spatial` spec carries the literal extents a Conv or
  pool loop nest needs, and `auto_pad` is resolved to begin-pads here, because
  it depends on the input shape and the input shape is known.
- **Relabelling is free.** `Reshape`, `Squeeze`, `Unsqueeze`, `Flatten`,
  `Identity`, and any `Transpose` that only moves size-1 axes move no bytes, so
  they become buffer aliases: no code, no copy. Liveness is tracked on the root
  of an alias chain, so a buffer is only reused after the last read of anything
  sharing it.
- **Buffers are recycled.** Input and output get dedicated buffers; every
  intermediate rotates through a free list.
- **Several inputs, one buffer.** A model with more than one graph input takes
  them concatenated in `x` in declaration order, and each secondary input is
  copied out of its slice. That is what keeps `infer(x, y)` — and with it
  `infer_batch`, the native kernel and the device contract — unchanged.

The plan carries a sha256 of itself, which the weights file records and the
generated reader checks.

## 5. Emit — `emit_c.py`, `emit_fortran.py`, `emit_kernel.py`

Both emitters render the *same* plan, so the two backends agree to 1e-12 and
emit identical buffer structure. `emit_kernel.py` writes the native CUDA/HIP
batched kernel, which calls the same header-inline body.

One detail is not cosmetic: a dense layer adds its bias **after** the dot
product rather than seeding the accumulator with it. Seeding an accumulator
from a declare-target array is what makes nvc refuse to generate a
`distribute parallel for` body at all — it emits a kernel that traps — and the
reordering unlocks a ~30x faster per-point offload loop. See
[`python/examples/nvhpc_teams_mapping/`](../python/examples/nvhpc_teams_mapping/).

## 6. Verify — `verify.py`

`rosenna verify` generates, compiles and runs both backends and compares them
against onnxruntime on random inputs drawn from a fixed seed.

Two things it deliberately does. It resamples until the reference is *alive*: a
model whose own weights compute all zeros would otherwise "pass" by reproducing
a dead network. And it allows a cancellation term in the tolerance — the
classical `n * eps * sum|terms|` bound for a summation — because onnxruntime
blocks and vectorises its convolutions and GEMMs, so two correct
implementations legitimately differ by more than `rtol * |expected|` when the
sum cancels.

`python/tests/test_golden_suite.py` runs every model in `goldenFiles/` through
this, on both backends. That replaced the old shell suite, which compared
against recorded output — a recorded file pins whatever the library did the day
it was recorded, so a wrong-but-stable implementation records its own error as
the expectation.

## 7. Gate — `gate.py`

`rosenna gpu-gate` is the check that exercises the device path on real
hardware: it builds the model embedded and file-loaded, in both languages, and
runs three harnesses — a per-point C host, a per-point Fortran host, and a host
that hands device-resident data to `infer_batch` — each compared against
onnxruntime and timed. An `nsys` capture scoped to the timed call asserts zero
`cudaMemcpy` inside it. Every command and its output goes into
`gate-report.md`.
