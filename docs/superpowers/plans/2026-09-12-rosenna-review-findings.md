# roseNNa Correctness Review — 2026-09-12

Reviewed at `7e5a4f9` on macOS (arm64) with gfortran 15.2.0, Python 3.11, torch 2.12.0,
onnx 1.22.0, numpy 2.4.6, fypp 3.2. Every defect below was reproduced; the reproduction is
recorded with it. Findings are grouped by severity, not by file.

Baseline at review time: `cd test && ./run.sh` reports **1 out of 17**.

---

## A. Confirmed numerical defects

### A1. MaxPool reads its kernel size from the wrong attribute index

`fLibrary/modelParserONNX.py:423` — `f.write(str(node.attribute[1].ints[0]))`

Assumes `kernel_shape` sits at attribute index 1. Current torch exports MaxPool attributes as
`[ceil_mode, dilations, kernel_shape, pads, strides]`, so index 1 is `dilations`. The Fortran
side then pools with kernel 1, which is strided subsampling rather than max-pooling.

Reproduction:
```
$ cd test && make ex1 case=maxpool_basic && cat onnxModel.txt
1
MaxPool
1            <- kernel size, should be 3
```
ONNX node attributes: `[('ceil_mode',0), ('dilations',[1,1]), ('kernel_shape',[3,3]),
('pads',[0,0,0,0]), ('strides',[3,3])]`.

Output shapes still agree by coincidence ((6-1)/3+1 == 2), so only values are wrong. Three
golden cases fail: `maxpool_basic`, `maxpool_padding`, `maxpool_strides`.

The same block indexes `attributes['pads']` and `attributes['strides']` directly; both are
optional in ONNX, so a model omitting either raises `KeyError`.

### A2. Golden scripts cannot export on current PyTorch

All 16 of `goldenFiles/*/*.py` that call `torch.onnx.export`.

`torch.onnx.export` now defaults to the dynamo exporter, which requires `onnxscript`. Every
golden script fails at export, `make ex1` returns 1, and the suite reports 1/17 (only `mnist`
passes, because it ships its `.onnx`). This is why A1 and everything below went unnoticed.

Adding `dynamo=False` to each export call restores the suite to 14/17 — the three remaining
failures being A1.

### A3. Pooling mixes row and column extents

`fLibrary/layers.f90:262-265` (`max_pool`) and `:301-304` (`avgpool`)

The write index uses `MODULO(overImage, outColDim)` while the read window uses
`MODULO(overImage, outRowDim)`. `conv` (`:225`) uses the same variable for both and is correct.
Whenever the output is non-square the two disagree.

Reproduction — 2x4 input, kernel 1, stride 1, an identity pool:
```
input:  11 12 13 14 / 21 22 23 24
output: 13 14  0  0 / 23 24  0  0
```

Note `conv`'s variable names are themselves backwards: `outRowDim = size(out,4)` is the column
count. It is used consistently, so `conv` is correct despite the naming.

### A4. Pooling silently drops batches

`fLibrary/layers.f90:255, 263` (`max_pool`) and `:294, 302` (`avgpool`)

`out(1,...)` and `padded(1,...)` hardcode batch index 1. A `(2,1,2,2)` input returns shape
`(1,1,2,2)`. `conv` loops over batches correctly, so `Conv -> MaxPool` with batch > 1 loses all
but the first sample.

Every existing golden pooling case uses a square image and batch 1, so A3 and A4 are both
invisible to the suite.

### A5. `tanhh` overflows to NaN

`fLibrary/activation_funcs.f90:106-116`

Computes `(exp(x)-exp(-x))/(exp(x)+exp(-x))`. For `|x| > ~710` both `exp` calls overflow to
`Inf` and the result is `Inf/Inf`.

Reproduction:
```
tanhh   :  0.76159415595576485    NaN    NaN
intrinsic: 0.76159415595576485    1.0   -1.0
```
for `x = [1.0, 720.0, -720.0]`. LSTM cell states are unbounded, so this is reachable.

### A6. Gemm `transB=0` silently corrupts

`fLibrary/layers.f90:11-23`, called from `fLibrary/modelCreator.fpp:89` with
`transInp = 1 - transB`.

The `transInp /= 0` branch computes `matmul(lin%weights, inp)` — `B x A`, not `A x B`. ONNX
Gemm with `transB=0` means `Y = A(m,k) x B(k,n)`.

Reproduction — `A(1,2) x B(2,3)` should give `[5, 11, 17]`:
```
expected matmul(A,B) = 5.0  11.0  17.0
linear_layer gave shape  2  2
```
No error, no warning, wrong shape and wrong values.

`alpha`, `beta`, and `transA` are read by neither the parser nor the layer.

Not currently firing: every golden case comes from `nn.Linear`, which exports `transB=1`, and
the MatMul fallback (`modelParserONNX.py:468`) hardcodes `1`.

### A7. Uninitialized loop variable in the test driver

`test/userTesting.fpp:30, 52`

`integer :: time` is declared and `times(time) = T2-T1` executes, but the loop that assigned
`time` is commented out (`:45`, `:53`). Writes to `times(<undefined>)` — out-of-bounds
undefined behavior that happens not to crash. `bubble_sort` (`:66-82`) is dead along with it.

---

## B. Silent-wrong-answer risks

### B1. LSTM weights are streamed positionally from a second export

`fLibrary/modelParserONNX.py` — `true_weights[true_index]` with `true_index += 1` at eight
sites (191, 204, 259, 274, 368, 383, 451, 475).

Three consequences:

1. **`-w` is mandatory but documented "(Optional)"** (`:14`). Running the parser with only
   `-f` on `lstm_cell` produces `Fail!! / Incorrect outputs.`
2. **Argument order is silently load-bearing.** `test/Makefile:40` passes
   `-f model.onnx -w model_weights.onnx`; `examples/run_basic_maclinux.sh:5` and `README.md:63`
   pass them **swapped**. For `gemm_small` both exports are identical so it works; for an LSTM
   it would be catastrophic.
3. **Any node reordering between the two exports attaches weights to the wrong layer**, with
   nothing to detect it.

Root cause: ONNX constant-folds LSTM weights into `onnx::LSTM_85`-style initializers in a
different gate order. Measured against `lstm_cell.onnx`, the folded `W` blocks map to torch
gates `i, o, f, g` — ONNX's documented `iofc` — while `lstm_cell` (`layers.f90:90-96`) consumes
`i, f, g, o`. Remap index list: `[0, 2, 3, 1]`.

Initializer **names are identical** between the folded and unfolded exports for Gemm, Conv, and
Add; only LSTM is renamed. So name-based lookup works everywhere and LSTM needs only the gate
remap — `-w` can be removed outright.

### B2. Broadcast axis chosen by value instead of position

`fLibrary/modelParserONNX.py:110-119` (`fourDTransform`)

Locates each broadcast axis with `trueshape.index(dim)` — a value search returning the first
match. ONNX broadcasts by right-aligning shapes.

| target | added tensor | gives | correct |
|---|---|---|---|
| `[1,3,3,3]` | `(3,3)` | `[1,3,1,1]` | `[1,1,3,3]` |
| `[1,3,3,3]` | `(3,)` | `[1,3,1,1]` | ok by luck |
| `[1,4,3,3]` | `(1,1,1)` | `[1,1,1,1]` | ok |

`spreadInfo` returns the identical `[3,3,4,3]` for the `(3,)` and `(3,3)` cases, so the two are
indistinguishable downstream.

### B3. Ten bare `except:` clauses hide parse failures

`fLibrary/modelParserONNX.py` — 10 occurrences.

They convert `KeyError`/`IndexError`, the signature of an unrecognized graph, into "quietly take
another path". A1 is exactly the class of defect this hides.

Worst case is `Squeeze` (`:291-304`): three sequential `try` blocks, each of which can append to
`modelArch`, so one Squeeze node can emit up to three layer entries. The first passes
`ioMap[node.input[1]]` — a **string** — as the axes list, which `modelCreator.fpp:135` then
tests with `num not in tup[3][0]`, matching characters instead of axes. Negative axis indices
are passed through unnormalized.

The `else:` branch (`:519-523`) prints `NOT SUPPORTED` and then aliases the output to the input,
so the model runs and produces a wrong answer.

### B4. Layer attributes parsed and then ignored

- **`dilations`** — parsed (`:360`), passed to `conv` (`layers.f90:188`), never read.
- **`ceil_mode`** — passed to both pooling subroutines (`:240`, `:278`), never read; extents
  always floor.
- **Asymmetric pads** — `padding` (`:134-181`) applies `arr(1)`/`arr(2)` symmetrically, using
  only the first two of ONNX's four pad values. The parser's own SAME_UPPER/SAME_LOWER path
  (`:354`, `:356`) computes asymmetric pads that are then applied symmetrically.
- **Non-square kernels** — `conv` derives `kernel_size = SIZE(convWeights, dim=3)`, one scalar
  for both extents.
- **Conv `auto_pad`** — `auto_pad = True` is set whenever the attribute is *present*
  (`:342-344`), including `NOTSET` and `VALID`, which both mean "do not pad like SAME". MaxPool
  guards with `!= "NOTSET"` (`:407`) — still wrong for `VALID`, but inconsistently so. Conv also
  dereferences `attributes['pads']` unconditionally at `:360`, raising `KeyError` when a Conv
  node has neither `auto_pad` nor explicit `pads`.
- **AveragePool `kernel_shape`** defaults to `0` (`:435`), so `attributes[-1][0]` raises
  `TypeError` when the attribute is absent.

### B5. ONNX tensor names are emitted as raw Fortran identifiers

`fLibrary/modelCreator.fpp:46, 52, 63, 69, 73`

Torch's default names are not valid Fortran identifiers. Exporting without `input_names=`
produces a graph input named `onnx::Gemm_0` and a node output named `3`.

Every golden case passes `input_names=['input']`, so generated code compiles only because the
suite avoids the problem. A user following the README's own export snippet without
`input_names` gets a compile error in code they never wrote.

### B6. C API signature mismatch and unchecked file I/O

`fLibrary/reader.f90:38` defines `initialize()` as `bind(c)` with **no arguments**;
`examples/cAPI.c:4` declares `void initialize(char*, char*)` and calls it with two paths. The
paths are silently discarded.

`reader.f90:49-50` hardcodes `onnxModel.txt`/`onnxWeights.txt` on units 10 and 11 with no
`status='old'` and no `iostat`, so a missing file is a runtime crash with no message.

Callers must supply column-major data, documented nowhere; `examples/capiTester.f90:8` does the
`RESHAPE(..., order=[2,1])` dance without explanation.

---

## C. Performance and robustness

### C1. Three copies of the padded tensor per layer

`fLibrary/layers.f90:134-181, 196`

`padding` declares both its result and a local `formatted` at full padded extent, fills
`formatted`, then copies it to the result. The caller receives it into another full-extent
automatic array. Two of the three are stack arrays.

Plausibly why `vgg16`, `gemm_huge`, and `turbulentShear` are excluded in `test/run.sh:8`.

### C2. `sumini` has an implicit SAVE

`fLibrary/layers.f90:205` — `REAL (c_double) :: sumini = 0`. Initialization in a declaration
implies `SAVE`. Works only because the code resets it after each use (`:226`); not reentrant.

### C3. Weights round-trip through ASCII

`modelParserONNX.py` writes space-separated decimal; `reader.f90` reads with list-directed
`read`. Dominates file size and load time for large models and loses bits to decimal formatting.

### C4. `reader.f90` uses module-level scratch state

`:17-34` declares `weights`, `midWeights`, `largeWeights`, `biases`, `w_dim1..4`, `layerName`,
and `i` at module scope, mutated by every `read_*` subroutine. `activation_func` (`:17`) is
never used.

---

## D. Hygiene

### D1. `test/modelCreator.fpp` duplicates `fLibrary/modelCreator.fpp`

Byte-identical (`diff` confirms). Two copies of the file that generates the entire model.

### D2. Makefile defects

`test/Makefile`: `FFLAGS=-O2` (line 4) immediately overwritten by `FFLAGS=-O3` (line 5); the
`capi:` target (33-36) references `capi.c`, which does not exist (the real file is
`examples/cAPI.c`); `clean` (50-51) misses the `$(DIR)/%.o` objects written into `fLibrary/`.

`fLibrary/Makefile`: `clean` (22-25) uses bare `rm`, failing the target when any pattern is
already absent. Dead commented `testing:` block at 26-31.

### D3. Every documentation link is dead

Three independent rots, all needed to fix any one link:

1. The repo was renamed `FyeNNa` -> `roseNNa` (confirmed: `gh repo view comp-physics/FyeNNa`
   follows the redirect and returns `name: roseNNa`).
2. The `develop` branch no longer exists — upstream has only `master`.
3. Paths moved into `fLibrary/` and `test/`, and `readTester.f90` became `reader.f90`.

Verified 404s: `comp-physics/FyeNNa/tree/develop/goldenFiles`,
`comp-physics/roseNNa/blob/develop/derived_types.f90`,
`comp-physics/roseNNa/blob/develop/readTester.f90`.

`doc/opensource.md:60` uses "reader.f90" as the link text while pointing at `readTester.f90`.
`doc/methodology.md:2` names `readTester.f90` in prose as a core source file.

### D4. README markup

`README.md:2` has a stray `</center>` closing a tag never opened (the wrapper is
`<p align="center">`). The DOI badge link and image (`:8-9`) both use `http://` while the
adjacent License badge uses `https://`, so the badge works only via an insecure redirect and is
blocked as mixed content by any https renderer outside GitHub's camo proxy.

### D5. CI is entirely commented out

`.github/workflows/CI.yml` has zero non-comment lines and is the only file in
`.github/workflows/`. GitHub still scans it, so it surfaces as an invalid workflow.

It was commented out (27de216) rather than repaired; the actual break was
`ln -s /opt/homebrew/bin/gfortran-14 /usr/local/bin/gfortran`, a hardcoded version already
re-patched twice as Homebrew's gcc drifted. The README badge was then deleted (7e5a4f9), so
nothing signals the absence.

---

## E. Environment traps

### E1. `.gitignore` is deny-all with an allowlist

Line 1 is `*`, followed by `!*.f90`, `!*.py`, `!*.sh`, `!*.yml`, `!*.c`, `!Makefile`, `!*.fpp`.
Any new file with a different extension is **silently skipped by `git add`**. Verified:
`requirements.txt` and anything under `docs/` are ignored.

### E2. fypp resolves `#:include` against the include path

Not against the source file's directory. `fypp -I<dir>` is required whenever a `.fpp` and its
`variables.fpp` live in different directories. Verified: without `-I`, fypp 3.2 fails with
`include file 'variables.fpp' not found`.
