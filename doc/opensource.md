# Adding an operator

roseNNa does not implement every ONNX operator. Adding one means teaching four
places about it, in this order. The order matters: each step is refused loudly
by the one before it until you get there, so you are never debugging generated
code that should not have been generated.

Work through it with `rosenna verify` after every step. A new op is done when
the model it unblocks matches onnxruntime on **both** backends.

## 0. Decide whether it is really an operator

Before writing a loop nest, check whether the op belongs in one of the two
categories that cost nothing:

- **Constant-only.** If every input is an initializer, add it to `FOLDABLE` in
  `fold.py` and give `_evaluate` a numpy one-liner. It is then computed at
  generation time and never reaches the emitters. Most `Reshape`s of weights,
  and every `Constant`, land here.
- **Relabelling.** If it only renames axes — it moves no bytes in a flat
  row-major buffer — add it to `RELABEL`. `plan.py` turns it into a buffer
  alias: no code, no copy, no extra buffer. `Reshape`, `Squeeze`, `Unsqueeze`,
  `Flatten` and `Identity` are all in this class, and so is any `Transpose`
  whose permutation only moves size-1 axes (`_flat_preserving` decides).

Only what survives both of those needs real generated code.

## 1. `validate.py` — refuse what you will not implement

Add the op to `SUPPORTED`, then write a `_validate_<op>` that rejects every
attribute your loop nest does **not** honour, naming the node.

This is the most important step and the easiest to under-do. Every rule here
exists because the alternative is not a crash but a model that runs and returns
plausible, wrong numbers. If your Conv ignores `dilations`, refuse a non-unit
`dilations` — do not quietly compute something else.

```python
def _validate_mything(graph: Graph, node) -> None:
    where = f"node '{node.name}'"
    if int(node.attrs.get("some_mode", 0)) != 0:
        raise UnsupportedModel(f"{where}: some_mode=1 is not supported")
```

## 2. `plan.py` — lower it to literal extents

Two parts: a frozen spec dataclass carrying whatever the loop nest needs, and a
branch in `build_plan` that fills it.

Resolve everything shape-dependent **here**, not in the emitters. `auto_pad` is
the worked example: it depends on the input extent, the input extent is
literal, so `_begin_pads` turns it into two integers and the emitted code never
learns that `auto_pad` exists. The emitters should only ever interpolate
numbers.

```python
@dataclass(frozen=True)
class MyThing:
    n: int
    extent: int
```

Add the field to `Op` (default `None`), and append your op in `build_plan`.
`n_in`/`n_out` are the flat element counts — `_length(graph.values[name])`.

If your op has extra operands or results beyond the single in/out every other
op uses, put them in `extra_in` / `outs`; `_assign_buffers` already tracks
liveness across both. If it needs scratch that lives across its own internal
loop, allocate it there too, the way `lstm` does for its carried state.

## 3. The emitters — one loop nest each

`emit_c.py` and `emit_fortran.py` render the same plan, and the golden suite
asserts they agree. Write them together and keep them line-for-line parallel;
it is the only practical way to keep them in step.

Buffers are flat and row-major in both languages. In Fortran the counters stay
0-based and only the subscript gains the `+ 1`, so the two emitters compute
visibly the same index:

```python
idx = f"((n * {c} + ic) * {h} + ih) * {w} + iw"        # C
idx = f"((n * {c} + ic) * {h} + ih) * {w} + iw + 1"    # Fortran
```

Three rules the existing ops follow:

- **Add a bias after the accumulation, never as the seed.** `acc = 0`, sum,
  then `acc += b[i]`. Seeding from a declare-target array makes nvc refuse to
  compile a `distribute parallel for` body at all. See
  [`python/examples/nvhpc_teams_mapping/`](../python/examples/nvhpc_teams_mapping/).
- **Propagate NaN.** `max(v, 0)` returns 0 for a NaN, and `v > best` drops one.
  Write `merge(0, v, v < 0)` and `!(v <= best)`. This library is linked into
  solvers where a NaN out of a diverged run is the signal.
- **Declare Fortran locals.** Fortran has no statement-scoped declarations, so
  any new counter or accumulator has to be added to the `loop_vars` list in
  `emit_fortran.py`, and only when an op actually uses it — an unused variable
  is a warning in any tree built with `-Werror`.

Weight layout differs between the backends: `emit_c` indexes a weight flat,
while `emit_fortran` declares it with the ONNX shape **reversed** and fills it
from the same C-order value list, so `w(kw, kh, ic, oc)` in Fortran addresses
exactly what `w[((oc*IC+ic)*KH+kh)*KW+kw]` reaches in C. A weight whose index
arithmetic is genuinely flat (a broadcast `Add` constant, an LSTM's `W`) is
registered with a flat shape instead.

## 4. Tests

- Add a golden model under `goldenFiles/<name>/<name>.py` if the op needs one,
  and add its name to `GOLDEN` in `python/tests/test_golden_suite.py` — the
  suite asserts that list is exactly the set on disk, so it cannot drift.
- Add a rejection test for each attribute `validate.py` refuses.
- Build a regression test from an inline `onnx.helper` graph for anything the
  golden models do not exercise. `python/tests/test_regressions.py` has the
  pattern; `_both_backends` compiles and runs both and compares to onnxruntime.

Run `cd python && python3 -m pytest tests`. If you have an NVIDIA GPU, run
`rosenna gpu-gate` too — the per-point path is compiled by a different compiler
than the tests use, and it has caught real codegen problems.
