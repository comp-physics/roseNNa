# roseNNa Correctness Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every correctness, robustness, and hygiene defect found in the 2026-09-12 review of roseNNa, and leave behind a test suite and CI that would have caught them.

**Architecture:** Work bottom-up. First make the existing golden-file suite runnable again (it has been dead since PyTorch switched to the dynamo ONNX exporter), then add two new test harnesses (a Fortran unit driver and a Python parser test) so that defects unreachable through golden files become testable. Only then fix the defects themselves, each with a test that fails first. Numerical fixes come before API and performance work; hygiene comes last.

**Tech Stack:** Fortran 2008/2018 (gfortran), Python 3.11+ (PyTorch, ONNX, numpy), fypp preprocessor, GNU make, GitHub Actions.

**Spec:** This document is self-contained. It derives from the code review recorded in `docs/superpowers/plans/2026-09-12-rosenna-review-findings.md` (Task 0 creates it); every finding below was reproduced on an M-series macOS box with gfortran 15.2.0, torch 2.12.0, onnx 1.22.0, numpy 2.4.6, fypp 3.2.

---

## Global Constraints

- **`.gitignore` is deny-all with an allowlist.** Line 1 is `*`, followed by `!*.f90`, `!*.py`, `!*.sh`, `!*.yml`, `!*.c`, `!Makefile`, `!*.fpp`. Any new file with a different extension — `requirements.txt`, anything under `docs/` — is **silently skipped by `git add`**. Verify every new non-allowlisted file with `git check-ignore -v <path>` and add an allowlist line before committing it. This plan document itself is currently ignored.
- **Do not regenerate `goldenFiles/mnist/mnist.onnx`.** It is the only committed `.onnx` (explicitly allowlisted) and is not reproducible from `mnist.py`, which only runs inference against it.
- **Golden-file `.txt` expectations are regenerated on every run** (`goldenFiles/*/*.txt` is gitignored), so a test "passes" only by agreeing with a freshly-run PyTorch. Never hand-edit them.
- **fypp resolves `#:include` relative to the include path, not the source file.** `fypp -I<dir>` is required whenever the `.fpp` and its `variables.fpp` live in different directories (verified: without `-I`, fypp 3.2 fails with `include file 'variables.fpp' not found`).
- **Never run any command from the main checkout.** All work happens in the worktree; every
  path in this plan is relative to it. `git clean -fdX` appears in most tasks and deletes ALL
  gitignored files, so no scratch state may live inside the worktree. The SDD workspace is at
  `/Users/spencer/Downloads/rosenna/.sdd-workspace`, deliberately outside it. (`-e` does NOT
  protect a path from `-X`; it adds to the ignore rules, which makes `-X` delete it.)
- **Line numbers in this plan are from the pre-task tree and drift as tasks land.** Tasks 10 and 18 both edit `layers.f90` above other cited regions. Always locate code by the surrounding subroutine name and the quoted text, and treat `file:line` as a hint. Verify with `grep -n` before editing.
- **Baseline before any work:** `cd test && ./run.sh` reports `1 out of 17`. After Task 1 it must report `14 out of 17`. After Task 4 it must report `17 out of 17`. Every later task must keep it at 17+ (plus whatever new cases that task adds).
- **ONNX LSTM gate order is `i, o, f, c`.** roseNNa's `lstm_cell` consumes `i, f, g, o` (PyTorch order). The remap index list is `[0, 2, 3, 1]`. Verified empirically against `lstm_cell.onnx`.
- Commit after every task. Keep PRs sequential, never stacked.

---

## File Structure

**New files:**
- `fLibrary/onnx_helpers.py` — pure, importable helper functions extracted from `modelParserONNX.py` (shape math, weight ordering, gate remapping, identifier sanitization). No argparse, no file I/O, no side effects at import.
- `test/test_parser.py` — assertions over `onnx_helpers.py`. Exits nonzero on failure.
- `test/unit_tests.f90` — standalone Fortran driver asserting layer-level behavior that golden files cannot reach.
- `requirements.txt` — pinned Python dependencies for CI (**needs a `.gitignore` allowlist entry**).
- `goldenFiles/maxpool_nonsquare/`, `goldenFiles/pool_batch/`, `goldenFiles/add_broadcast/` — new golden cases.

**Heavily modified:**
- `fLibrary/modelParserONNX.py` — the source of most silent-wrong-answer defects.
- `fLibrary/layers.f90` — pooling rewrite, padding allocation, `sumini`.
- `fLibrary/reader.f90` — file handling, scratch globals, dead code.
- `.github/workflows/CI.yml` — currently 100% commented out.

**Deleted:**
- `test/modelCreator.fpp` — byte-identical duplicate of `fLibrary/modelCreator.fpp`.

---

## Phase 0 — Make the suite runnable

### Task 0: Record the review findings

**Files:**
- Create: `docs/superpowers/plans/2026-09-12-rosenna-review-findings.md`
- Modify: `.gitignore`

**Interfaces:**
- Produces: the findings document every later task cites.

- [ ] **Step 1: Confirm the docs path is gitignored**

Run: `git check-ignore -v docs/superpowers/plans/x.md`
Expected: `.gitignore:1:*	docs/superpowers/plans/x.md`

- [ ] **Step 2: Add the allowlist entry**

Append to `.gitignore`:

```
!docs/
!docs/**
!requirements.txt
```

- [ ] **Step 3: Verify the entry took effect**

Run: `git check-ignore -q docs/superpowers/plans/x.md && echo IGNORED || echo tracked-ok`
Expected: `tracked-ok`

- [ ] **Step 4: Write the findings document**

Copy the review findings (the 23 numbered defects) into `docs/superpowers/plans/2026-09-12-rosenna-review-findings.md`, one section per defect, each with: file:line, what is wrong, the reproduction, and the expected behavior.

- [ ] **Step 5: Commit**

```bash
git add .gitignore docs/
git commit -m "docs: record 2026-09-12 correctness review findings"
```

---

### Task 1: Restore ONNX export on current PyTorch

**Problem:** `torch.onnx.export` now defaults to the dynamo exporter, which requires `onnxscript`. Every golden script fails at export, `make ex1` returns 1, and `run.sh` reports `1 out of 17` (only `mnist` passes, because it ships its `.onnx`). This is why every other bug in this plan went unnoticed.

**Files:**
- Modify: all 16 of `goldenFiles/*/*.py` that call `torch.onnx.export` (every case except `mnist`)

**Interfaces:**
- Produces: a runnable baseline suite at `14 out of 17`. The 3 remaining failures are real bugs, fixed in Task 4.

- [ ] **Step 1: Reproduce the failure**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `1 out of 17 test cases have passed!`

Run: `cd test && make ex1 case=gemm_small 2>&1 | grep ModuleNotFound`
Expected: `ModuleNotFoundError: No module named 'onnxscript'`

- [ ] **Step 2: Pin the legacy exporter in every golden script**

```bash
cd "$(git rev-parse --show-toplevel)"
for f in goldenFiles/*/*.py; do
  sed -i '' 's/export_params=True,/export_params=True, dynamo=False,/' "$f"
done
```

- [ ] **Step 3: Verify all 16 scripts were patched**

Run: `grep -l "dynamo=False" goldenFiles/*/*.py | wc -l`
Expected: `16`

Run: `grep -c "dynamo=False" goldenFiles/gemm_small/gemm_small.py`
Expected: `2` (the structure export and the weights export)

- [ ] **Step 4: Run the suite**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `14 out of 17 test cases have passed!`

The three failures must be exactly `maxpool_basic`, `maxpool_padding`, `maxpool_strides`, each reporting `Incorrect outputs.` Confirm with:

Run: `cd test && ./run.sh 2>&1 | grep -B6 "Fail!!" | grep "TEST #"`
Expected: three lines naming those cases.

- [ ] **Step 5: Clean build artifacts and commit**

```bash
git clean -fdX
git add goldenFiles/
git commit -m "test: pin legacy ONNX exporter so golden scripts run on torch 2.x"
```

---

### Task 2: Test harnesses for what golden files cannot reach

**Problem:** `modelParserONNX.py` runs `argparse` at module scope, so none of its helpers can be imported or tested. Several defects (tanh overflow, Gemm dimension corruption, broadcast axis selection) cannot be expressed as an ONNX golden case at all.

**Files:**
- Create: `fLibrary/onnx_helpers.py`
- Create: `test/test_parser.py`
- Create: `test/unit_tests.f90`
- Modify: `fLibrary/modelParserONNX.py` — remove the helper definitions at lines **72-103** (`stranspose`, `stringer`, `reshapeParser`) and **110-131** (`fourDTransform`, `fakeFourD`, `spreadInfo`), and import them instead. **Leave `findWeightsInitializer` at lines 105-108 exactly where it is** — it closes over the module-level `initializer` and `constants` dicts and cannot move into a pure helpers module.
- Modify: `test/Makefile` (add a `unit` target)
- Modify: `test/run.sh` (run both harnesses before the golden loop)

**Interfaces:**
- Produces:
  - `onnx_helpers.stranspose(arr) -> str`
  - `onnx_helpers.stringer(mat) -> str`
  - `onnx_helpers.reshapeParser(reshape: list, trueShape: list) -> list`
  - `onnx_helpers.fourDTransform(trueshape: list, toBeTransformedShape: tuple) -> list`
  - `onnx_helpers.fakeFourD(inp: list) -> list`
  - `onnx_helpers.spreadInfo(trueShape: list, toBeTransformedShape: list) -> list`
  - `test/unit_tests.f90` binary `test/unit_tests`, exit 0 on pass, 1 on any failure
  - Tasks 6, 9, 10, 12, 13 add cases to these two files.

- [ ] **Step 1: Write the failing Python test**

Create `test/test_parser.py`:

```python
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fLibrary"))
import onnx_helpers as H

failures = []

def check(cond, name):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        failures.append(name)

def test_stranspose_is_column_major():
    for shape in [(5,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
        a = np.arange(int(np.prod(shape))).reshape(shape)
        expected = " ".join(str(x) for x in a.flatten(order="F").tolist())
        check(H.stranspose(a) == expected, f"stranspose column-major {shape}")

def test_stringer():
    check(H.stringer([1, 2, 3]) == "1 2 3", "stringer joins with spaces")
    check(H.stringer([]) == "", "stringer handles empty")

def test_reshape_parser_resolves_negative_one():
    check(H.reshapeParser([-1, 4], [2, 2, 4]) == [4, 4], "reshapeParser resolves -1")
    check(H.reshapeParser([2, 4], [2, 4]) == [2, 4], "reshapeParser passes through")

if __name__ == "__main__":
    test_stranspose_is_column_major()
    test_stringer()
    test_reshape_parser_resolves_negative_one()
    print(f"PARSER TESTS: {len(failures)} failure(s)")
    sys.exit(1 if failures else 0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd test && python3 test_parser.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'onnx_helpers'`

- [ ] **Step 3: Extract the helpers**

Create `fLibrary/onnx_helpers.py`:

```python
"""Pure helpers shared by modelParserONNX.py. No side effects at import."""
import itertools
import numpy as np


def stranspose(arr):
    """Flatten in Fortran (column-major) order, space separated."""
    shape = arr.shape
    combs = [x for x in range(len(shape))]
    for dim1, dim2 in itertools.combinations(combs, 2):
        dim = combs.copy()
        dim[dim1] = dim2
        dim[dim2] = dim1
        arr = np.transpose(arr, dim)
    return stringer(arr.flatten().tolist())


def stringer(mat):
    return " ".join(str(elem) for elem in mat)


def reshapeParser(reshape, trueShape):
    """Resolve a single -1 in an ONNX Reshape target against the true shape."""
    if -1 not in reshape:
        return reshape
    ind = reshape.index(-1)
    res = np.prod(reshape) * -1
    true = np.prod(trueShape)
    reshape[ind] = int(true / res)
    return reshape


def fakeFourD(inp):
    return (4 - len(inp)) * [1] + list(inp)


def fourDTransform(trueshape, toBeTransformedShape):
    new = [1, 1, 1, 1]
    for dim in toBeTransformedShape:
        try:
            find = trueshape.index(dim)
            new[find - len(trueshape)] = dim
        except ValueError:
            pass
    return new


def spreadInfo(trueShape, toBeTransformedShape):
    ret = []
    for index, dim in enumerate(toBeTransformedShape):
        if trueShape[index] != dim:
            ret.append(index + 1)
            ret.append(trueShape[index])
    return ret
```

Note: `fourDTransform` is copied verbatim including its defect — Task 9 fixes it against a test. `stranspose` keeps its pairwise-transpose formulation here; the test above pins its behavior so Task 9 can simplify it safely.

- [ ] **Step 4: Make `modelParserONNX.py` import them**

Delete lines 72-103 and 110-131 of `fLibrary/modelParserONNX.py` — the six helper definitions — keeping `findWeightsInitializer` (105-108) in place. Add after the existing imports (around line 9):

```python
from onnx_helpers import (
    stranspose, stringer, reshapeParser,
    fourDTransform, fakeFourD, spreadInfo,
)
```

`modelParserONNX.py` is invoked as `python3 $(DIR)/modelParserONNX.py`, so its own directory is already on `sys.path` and the plain import resolves.

- [ ] **Step 5: Run the Python test to verify it passes**

Run: `cd test && python3 test_parser.py`
Expected: three `ok` lines, `PARSER TESTS: 0 failure(s)`, exit 0

- [ ] **Step 6: Write the failing Fortran unit test**

Create `test/unit_tests.f90`:

```fortran
program unit_tests
    use iso_c_binding
    use activation_functions
    use derived_types
    use model_layers
    implicit none

    integer :: failures = 0

    call test_relu_basic()
    call test_sigmoid_midpoint()

    if (failures > 0) then
        write(*,'(a,i0,a)') 'UNIT TESTS: ', failures, ' failure(s)'
        stop 1
    end if
    write(*,'(a)') 'UNIT TESTS: all passed'

contains

    subroutine check(cond, name)
        logical, intent(in) :: cond
        character(*), intent(in) :: name
        if (cond) then
            write(*,'(a,a)') '  ok   ', name
        else
            write(*,'(a,a)') '  FAIL ', name
            failures = failures + 1
        end if
    end subroutine

    subroutine test_relu_basic()
        real(c_double) :: x(3), y(3)
        x = [-1.0d0, 0.0d0, 2.0d0]
        y = relu(x)
        call check(all(abs(y - [0.0d0, 0.0d0, 2.0d0]) < 1.0d-12), 'relu clamps negatives')
    end subroutine

    subroutine test_sigmoid_midpoint()
        real(c_double) :: y(1)
        y = sigmoid([0.0d0])
        call check(abs(y(1) - 0.5d0) < 1.0d-12, 'sigmoid(0) == 0.5')
    end subroutine

end program unit_tests
```

- [ ] **Step 7: Add the `unit` target**

In `test/Makefile`, after the `compile:` target:

```make
unit: $(COMP) unit_tests.o
	$(FC) $(FFLAGS) -o unit_tests $(COMP) unit_tests.o
	./unit_tests
```

- [ ] **Step 8: Run the Fortran unit tests**

Run: `cd test && make unit`
Expected: two `ok` lines, `UNIT TESTS: all passed`, exit 0

- [ ] **Step 9: Wire both harnesses into `run.sh`**

Insert into `test/run.sh` immediately after `make compile` (line 5):

```bash
make unit || { echo "FATAL: Fortran unit tests failed"; exit 1; }
python3 test_parser.py || { echo "FATAL: parser tests failed"; exit 1; }
```

- [ ] **Step 10: Verify the full suite still passes**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `14 out of 17 test cases have passed!`

- [ ] **Step 11: Commit**

```bash
git clean -fdX
git add fLibrary/onnx_helpers.py fLibrary/modelParserONNX.py test/test_parser.py test/unit_tests.f90 test/Makefile test/run.sh
git commit -m "test: add parser and Fortran unit harnesses"
```

---

### Task 3: Re-enable CI

**Problem:** `.github/workflows/CI.yml` is 100% commented out — zero non-comment lines, and the only file in `.github/workflows/`. GitHub still scans it, so it surfaces as an invalid workflow. It was commented out (27de216, 2025-02-09) rather than repaired; the actual break was `ln -s /opt/homebrew/bin/gfortran-14 /usr/local/bin/gfortran`, a hardcoded version that had already been re-patched twice as Homebrew's gcc drifted. The README badge was then deleted (7e5a4f9) so nothing signals the absence.

**Files:**
- Modify: `.github/workflows/CI.yml` (replace entirely)
- Create: `requirements.txt`
- Modify: `README.md:3` (restore the CI badge)

**Interfaces:**
- Consumes: `test/run.sh` from Task 2 (now runs three harnesses).
- Produces: a green CI run that gates every later task.

- [ ] **Step 1: Create pinned dependencies**

Create `requirements.txt`:

```
torch>=2.0
onnx>=1.14
numpy>=1.24
fypp>=3.1
onnxruntime>=1.15
```

- [ ] **Step 2: Verify it is not gitignored**

Run: `git check-ignore -q requirements.txt && echo IGNORED || echo tracked-ok`
Expected: `tracked-ok` (Task 0 added the allowlist entry; if this says `IGNORED`, go back and add `!requirements.txt`)

- [ ] **Step 3: Replace the workflow**

Replace the entire contents of `.github/workflows/CI.yml`:

```yaml
name: CI

on:
  push:
    branches: [ "master" ]
  pull_request:
    branches: [ "master" ]
  workflow_dispatch:

jobs:
  CI_test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]

    steps:
      - name: Clone roseNNa
        uses: actions/checkout@v4

      - name: Set up gfortran
        uses: fortran-lang/setup-fortran@v1
        with:
          compiler: gcc
          version: 13

      - name: Check gfortran version
        run: gfortran --version

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run test cases
        run: |
          mkdir -p fLibrary/objFiles
          chmod +x test/run.sh
          cd test && ./run.sh
```

The path filters from the old workflow are dropped deliberately: they excluded `.fpp`-adjacent and golden-file changes in ways that let breakage through, and this repo is small enough to always run.

- [ ] **Step 4: Restore the badge**

In `README.md`, immediately after line 2 (`<img src="doc/rosenna.png" .../>`), inside the existing badge `<p align="center">` block, add:

```html
<a href="https://github.com/comp-physics/roseNNa/actions/workflows/CI.yml">
  <img src="https://github.com/comp-physics/roseNNa/actions/workflows/CI.yml/badge.svg" />
</a>
```

- [ ] **Step 5: Validate the workflow parses**

Run: `python3 -c "import yaml,sys; d=yaml.safe_load(open('.github/workflows/CI.yml')); assert d.get('jobs'), 'no jobs'; assert d.get(True) or d.get('on'), 'no triggers'; print('workflow OK')"`
Expected: `workflow OK`

(Note: PyYAML parses the bare key `on` as boolean `True`; that is a YAML quirk, not a workflow defect.)

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/CI.yml requirements.txt README.md
git commit -m "ci: restore CI on version-agnostic gfortran, add ubuntu matrix"
```

- [ ] **Step 7: Push and confirm CI goes green**

Push the branch and confirm the Actions run passes on both runners. Expected: `14 out of 17` and a nonzero exit, so **CI will be red at this point** — that is correct and intended. It goes green at the end of Task 4. Do not merge this task alone.

---

## Phase 1 — Confirmed numerical bugs

### Task 4: MaxPool reads its kernel size from the wrong attribute

**Problem:** `modelParserONNX.py:423` writes `node.attribute[1].ints[0]`, assuming `kernel_shape` is at index 1. Current torch exports MaxPool attributes as `[ceil_mode, dilations, kernel_shape, pads, strides]`, so index 1 is `dilations` and the Fortran side pools with kernel 1 — strided subsampling instead of max-pooling. `onnxModel.txt` contains `1` where `kernel_shape=[3,3]`. Output *shapes* still agree by coincidence ((6-1)/3+1 == 2), so only values are wrong. The same block also indexes `attributes['pads']` and `attributes['strides']` directly, which raises `KeyError` when either attribute is absent (both are optional in ONNX).

**Files:**
- Modify: `fLibrary/modelParserONNX.py:392-425`

**Interfaces:**
- Consumes: nothing new.
- Produces: `17 out of 17` — the gate for every later task.

- [ ] **Step 1: Reproduce**

```bash
cd test && make ex1 case=maxpool_basic >/dev/null 2>&1 && cat onnxModel.txt
```
Expected (the bug): `1 / MaxPool / 1` — the third line is the kernel size and reads `1`.

```bash
python3 -c "
import onnx
m = onnx.load('../goldenFiles/maxpool_basic/maxpool_basic.onnx')
n = [x for x in m.graph.node if x.op_type=='MaxPool'][0]
print([(a.name, list(a.ints) if a.ints else a.i) for a in n.attribute])"
```
Expected: `[('ceil_mode', 0), ('dilations', [1, 1]), ('kernel_shape', [3, 3]), ('pads', [0,0,0,0]), ('strides', [3, 3])]`

Confirms index 1 is `dilations`, and the correct value 3 lives at index 2.

- [ ] **Step 2: Run the three failing tests**

Run: `cd test && for c in maxpool_basic maxpool_padding maxpool_strides; do make testing case=$c 2>&1 | tail -2; done`
Expected: each prints `Fail!!` / `Incorrect outputs.`

- [ ] **Step 3: Fix the attribute lookup**

In `fLibrary/modelParserONNX.py`, replace lines 411-424 (the `if auto_pad:` block through the `f.write` of the kernel size) with:

```python
            attributes.setdefault('kernel_shape', [1, 1])
            attributes.setdefault('pads', [0, 0, 0, 0])
            attributes.setdefault('strides', [1, 1])
            if auto_pad:  # DEAL WITH STRIDE > 1?
                kernel_shape = attributes['kernel_shape'][0]
                pad_total = kernel_shape - 1
                pad = int(pad_total/2)
                if pad_total % 2 != 0:
                    if attributes['auto_pad'] == "SAME_UPPER":
                        attributes['pads'] = [pad,pad,pad+1,pad+1]
                    else:
                        attributes['pads'] = [pad+1,pad+1,pad,pad]
                else:
                    attributes['pads'] = [pad]*4
            modelArch.append(("MaxPool", [ioMap[node.input[0]]], [attributes['ceil_mode'],attributes['pads'],attributes['strides']])) #(ceil_mode, pads, strides)
            f.write(str(attributes['kernel_shape'][0]))
            f.write("\n")
```

The only behavioral changes are the three `setdefault` lines and `attributes['kernel_shape'][0]` replacing `node.attribute[1].ints[0]`.

- [ ] **Step 4: Verify the emitted kernel size**

Run: `cd test && make ex1 case=maxpool_basic >/dev/null 2>&1 && sed -n 3p onnxModel.txt`
Expected: `3`

- [ ] **Step 5: Run the full suite**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `17 out of 17 test cases have passed!`

- [ ] **Step 6: Commit**

```bash
git clean -fdX
git add fLibrary/modelParserONNX.py
git commit -m "fix: read MaxPool kernel_shape by name, not attribute index

node.attribute[1] is dilations on current torch exports, so every
MaxPool ran with kernel 1. Also default the optional pads/strides
attributes instead of raising KeyError."
```

---

### Task 5: Pooling mixes up row and column extents, and drops batches

**Problem:** Two independent defects in `max_pool` (`layers.f90:236-271`) and `avgpool` (`:273-310`):

1. The write index uses `MODULO(overImage, outColDim)` while the read window uses `MODULO(overImage, outRowDim)` (`layers.f90:262-265`, `:301-304`). `conv` uses the same variable for both (`:225`) and is correct. Whenever the output is non-square the two disagree and the result is garbage. Reproduced with a 2x4 input, kernel 1, stride 1 (an identity pool): input rows `11 12 13 14 / 21 22 23 24` came back as `13 14 0 0 / 23 24 0 0`.
2. `out(1,...)` and `padded(1,...)` hardcode batch index 1 (`:255`, `:263`, `:294`, `:302`). A `(2,1,2,2)` input returns shape `(1,1,2,2)`. `conv` loops over batches correctly, so `Conv -> MaxPool` with batch > 1 silently loses all but the first sample.

Every existing golden pooling case uses a square image and batch 1, so neither is visible.

Note the variable names in `conv` are themselves backwards — `outRowDim = size(out,4)` is the *column* count. The rewrite below uses honest names.

**Files:**
- Modify: `fLibrary/layers.f90:236-310` (replace both subroutines)
- Modify: `test/unit_tests.f90` (add two cases)
- Create: `goldenFiles/maxpool_nonsquare/maxpool_nonsquare.py`
- Create: `goldenFiles/pool_batch/pool_batch.py`

**Interfaces:**
- Consumes: `check/1` from `test/unit_tests.f90` (Task 2).
- Produces: `max_pool` and `avgpool` with unchanged signatures, correct for any batch size and any rectangular output.

- [ ] **Step 1: Write the failing unit tests**

Add to `test/unit_tests.f90`, registering both in the driver body:

```fortran
    subroutine test_maxpool_identity_nonsquare()
        real(c_double), allocatable :: x(:,:,:,:)
        type(maxpoolLayer) :: mp
        integer :: r, c
        logical :: ok
        allocate(x(1,1,2,4))
        do r = 1, 2
            do c = 1, 4
                x(1,1,r,c) = 10.0d0*r + c
            end do
        end do
        mp%kernel_size = 1
        call max_pool(x, mp, 0, [0,0], [1,1])
        ok = all(shape(x) == [1,1,2,4])
        if (ok) ok = abs(x(1,1,1,3) - 13.0d0) < 1.0d-12 .and. &
                     abs(x(1,1,2,4) - 24.0d0) < 1.0d-12
        call check(ok, 'max_pool k=1 s=1 is identity on non-square input')
    end subroutine

    subroutine test_maxpool_preserves_batch()
        real(c_double), allocatable :: x(:,:,:,:)
        type(maxpoolLayer) :: mp
        allocate(x(2,1,2,2))
        x = 1.0d0
        x(2,:,:,:) = 2.0d0
        mp%kernel_size = 2
        call max_pool(x, mp, 0, [0,0], [1,1])
        call check(all(shape(x) == [2,1,1,1]) .and. &
                   abs(x(2,1,1,1) - 2.0d0) < 1.0d-12, &
                   'max_pool preserves the batch dimension')
    end subroutine
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd test && make unit`
Expected: `FAIL max_pool k=1 s=1 is identity on non-square input` and `FAIL max_pool preserves the batch dimension`, `UNIT TESTS: 2 failure(s)`, exit 1

- [ ] **Step 3: Rewrite both subroutines**

Replace `fLibrary/layers.f90:236-310` with:

```fortran
    subroutine max_pool(inp, maxpool, ceil_mode, pads, strides)
        implicit none
        REAL (c_double), INTENT(INOUT), ALLOCATABLE, DIMENSION(:,:,:,:) :: inp !==(batches,numImages,imageD1,imageD2)
        TYPE(maxpoolLayer), INTENT(IN) :: maxpool
        INTEGER, INTENT(IN) :: ceil_mode
        INTEGER, INTENT(IN), DIMENSION(:) :: pads
        INTEGER, INTENT(IN), DIMENSION(:) :: strides
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: out
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: padded
        INTEGER :: kernel_size, b, ch, orow, ocol, outRows, outCols, r0, c0

        padded = padding(pads, inp)
        kernel_size = maxpool%kernel_size
        outRows = (size(padded,3) - kernel_size)/strides(1) + 1
        outCols = (size(padded,4) - kernel_size)/strides(2) + 1
        ALLOCATE(out(size(padded,1), size(padded,2), outRows, outCols))

        DO b = 1, size(padded,1)
            DO ch = 1, size(padded,2)
                DO orow = 1, outRows
                    r0 = (orow-1)*strides(1)
                    DO ocol = 1, outCols
                        c0 = (ocol-1)*strides(2)
                        out(b,ch,orow,ocol) = MAXVAL(padded(b, ch, &
                            r0+1 : r0+kernel_size, &
                            c0+1 : c0+kernel_size))
                    END DO
                END DO
            END DO
        END DO
        inp = out
    end subroutine

    subroutine avgpool(inp, avgpoolLay, ceil_mode, pads, strides)
        implicit none
        REAL (c_double), INTENT(INOUT), ALLOCATABLE, DIMENSION(:,:,:,:) :: inp
        TYPE(avgpoolLayer), INTENT(IN) :: avgpoolLay
        INTEGER, INTENT(IN) :: ceil_mode
        INTEGER, INTENT(IN), DIMENSION(:) :: pads
        INTEGER, INTENT(IN), DIMENSION(:) :: strides
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: out
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: padded
        INTEGER :: kernel_size, b, ch, orow, ocol, outRows, outCols, r0, c0
        REAL (c_double) :: total

        padded = padding(pads, inp)
        kernel_size = avgpoolLay%kernel_size
        total = REAL(kernel_size * kernel_size, c_double)
        outRows = (size(padded,3) - kernel_size)/strides(1) + 1
        outCols = (size(padded,4) - kernel_size)/strides(2) + 1
        ALLOCATE(out(size(padded,1), size(padded,2), outRows, outCols))

        DO b = 1, size(padded,1)
            DO ch = 1, size(padded,2)
                DO orow = 1, outRows
                    r0 = (orow-1)*strides(1)
                    DO ocol = 1, outCols
                        c0 = (ocol-1)*strides(2)
                        out(b,ch,orow,ocol) = SUM(padded(b, ch, &
                            r0+1 : r0+kernel_size, &
                            c0+1 : c0+kernel_size)) / total
                    END DO
                END DO
            END DO
        END DO
        inp = out
    end subroutine
```

Three incidental improvements: `padded` becomes allocatable (removing one automatic-array stack copy), the `avgpool` dummy is renamed off `maxpool` (it was shadowing the other layer type's name), and `total` becomes a real for clarity only — `SUM(...)` is already `REAL(c_double)`, and Fortran promotes the integer operand in mixed arithmetic, so the old division was never truncating.

- [ ] **Step 4: Update the avgpool call site**

`modelCreator.fpp:114` calls `avgpool(...)` positionally, so no change is needed. Confirm:

Run: `grep -n "CALL avgpool" fLibrary/modelCreator.fpp`
Expected: one positional call — no keyword arguments naming `maxpool=`.

- [ ] **Step 5: Run unit tests**

Run: `cd test && make unit`
Expected: `UNIT TESTS: all passed`

- [ ] **Step 6: Add the non-square golden case**

Create `goldenFiles/maxpool_nonsquare/maxpool_nonsquare.py` by copying `goldenFiles/maxpool_basic/maxpool_basic.py` and changing exactly four things: `nn.MaxPool2d(3)` becomes `nn.MaxPool2d(2)` in **both** the `SETUP_CODE` string and the real class, `inp = torch.rand(1,2,6,6)` becomes `inp = torch.rand(1,2,4,6)` in both places, every occurrence of the string `maxpool_basic` becomes `maxpool_nonsquare`, and `repeat = 10000` (line 57) becomes `repeat = 10`.

That last change matters: the template runs a 10000-iteration `timeit.repeat` benchmark whose result is only printed, so copying it verbatim adds minutes of pure waste to every suite run.

- [ ] **Step 7: Add the batch golden case**

Create `goldenFiles/pool_batch/pool_batch.py` the same way from `maxpool_basic.py`: `nn.MaxPool2d(2)` in both places, `inp = torch.rand(3,2,4,4)` in both places, `maxpool_basic` becomes `pool_batch` throughout, and `repeat = 10000` becomes `repeat = 10`.

- [ ] **Step 8: Run the full suite**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!`

- [ ] **Step 9: Commit**

```bash
git clean -fdX
git add fLibrary/layers.f90 test/unit_tests.f90 goldenFiles/maxpool_nonsquare goldenFiles/pool_batch
git commit -m "fix: correct pooling index extents and preserve batch dimension

max_pool/avgpool indexed the write with the column count and the read
with the row count, corrupting any non-square output, and hardcoded
batch index 1, silently dropping all but the first sample."
```

---

### Task 6: `tanhh` overflows to NaN

**Problem:** `activation_funcs.f90:106-116` computes `(exp(x)-exp(-x))/(exp(x)+exp(-x))`. For `|x| > ~710` both `exp` calls overflow to `Inf` and the result is `Inf/Inf = NaN`. Measured: `tanhh([720.0]) = NaN` where the intrinsic `tanh` gives `1.0`. LSTM cell states are unbounded, so this is reachable on a real model.

**Files:**
- Modify: `fLibrary/activation_funcs.f90:106-116`
- Modify: `test/unit_tests.f90`

**Interfaces:**
- Consumes: `check/1`.
- Produces: `tanhh`/`tanhh2d` with unchanged signatures.

- [ ] **Step 1: Write the failing test**

Add to `test/unit_tests.f90` and register it in the driver body:

```fortran
    subroutine test_tanh_saturates()
        real(c_double) :: y(3)
        y = tanhh([1.0d0, 720.0d0, -720.0d0])
        call check(abs(y(1) - 0.761594155955765d0) < 1.0d-12 .and. &
                   abs(y(2) - 1.0d0) < 1.0d-12 .and. &
                   abs(y(3) + 1.0d0) < 1.0d-12, &
                   'tanhh saturates instead of overflowing to NaN')
    end subroutine
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd test && make unit`
Expected: `FAIL tanhh saturates instead of overflowing to NaN`

- [ ] **Step 3: Use the intrinsic**

Replace `fLibrary/activation_funcs.f90:106-116`:

```fortran
    FUNCTION tanhh(x) result(output)
        REAL (c_double), intent(in) :: x(:)
        REAL (c_double) :: output(size(x))
        output = tanh(x)
    END FUNCTION tanhh

    FUNCTION tanhh2d(x) result(output)
        REAL (c_double), intent(in) :: x(:,:)
        REAL (c_double) :: output(size(x,1),size(x,2))
        output = tanh(x)
    END FUNCTION tanhh2d
```

- [ ] **Step 4: Run unit tests and the full suite**

Run: `cd test && make unit`
Expected: `UNIT TESTS: all passed`

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!` (the LSTM cases exercise `tanhh2d`, confirming the intrinsic agrees with the old formula in-range)

- [ ] **Step 5: Commit**

```bash
git clean -fdX
git add fLibrary/activation_funcs.f90 test/unit_tests.f90
git commit -m "fix: use intrinsic tanh to avoid NaN above |x| ~ 710"
```

---

### Task 7: Uninitialized loop variable in the test driver

**Problem:** `test/userTesting.fpp:30` declares `integer :: time` and `:52` executes `times(time) = T2-T1`, but the loop that assigned `time` is commented out (`:45`, `:53`). `time` is never assigned, so this writes to `times(<undefined>)` — out-of-bounds undefined behavior that happens not to crash today. `bubble_sort` (`:66-82`) is dead along with it.

**Files:**
- Modify: `test/userTesting.fpp`

**Interfaces:**
- Produces: a driver that compiles clean under `-fcheck=bounds`.

- [ ] **Step 1: Reproduce**

Run: `cd test && make ex1 case=gemm_small >/dev/null 2>&1 && fypp -I. userTesting.fpp userTesting.f90 && gfortran -fcheck=bounds -I../fLibrary/objFiles -c userTesting.f90 -o /dev/null 2>&1 | head -5`

Then inspect the generated source:

Run: `cd test && grep -n "times(time)" userTesting.f90`
Expected: a line assigning `times(time)` with no prior assignment to `time` anywhere in the file (confirm with `grep -n "time *=" userTesting.f90`, which should show only `times(time) = T2-T1`).

- [ ] **Step 2: Delete the timing scaffolding**

In `test/userTesting.fpp`, delete these lines:
- `:28-30` — `REAL :: T1, T2`, `REAL (c_double), DIMENSION(100) :: times`, `integer :: time`
- `:44` — the commented `open(56,...)` line
- `:45` — the commented `DO time=1,100` line
- `:49` — `CALL CPU_TIME(T1)`
- `:51-56` — `CALL CPU_TIME(T2)`, `times(time) = T2-T1`, and the four commented `!delete this line` rows
- `:65-82` — the `contains` block and `bubble_sort` subroutine in full

Leave the `#:for inp in arrs` re-initialization block (`:46-48`) and the `CALL use_model(...)` line (`:50`) intact.

- [ ] **Step 3: Verify it compiles clean**

Run: `cd test && rm -f userTesting.f90 && make ex1 case=gemm_small >/dev/null 2>&1 && fypp -I. userTesting.fpp userTesting.f90 && gfortran -fcheck=bounds -Wall -I../fLibrary/objFiles -c userTesting.f90 -o /dev/null 2>&1 | grep -i "uninitial\|bound" | head`
Expected: no output

Run: `cd test && grep -c "times\|bubble_sort" userTesting.f90`
Expected: `0`

- [ ] **Step 4: Run the full suite**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!`

- [ ] **Step 5: Commit**

```bash
git clean -fdX
git add test/userTesting.fpp
git commit -m "fix: remove dead timing scaffolding that indexed an uninitialized variable"
```

---

## Phase 2 — Parser correctness

### Task 8: Look weights up by name and remap LSTM gates

**Problem:** This is the deepest defect. The parser streams weights positionally out of the unoptimized export: `true_weights[true_index]` with `true_index += 1` at seven call sites (`modelParserONNX.py:192, 205, 260, 274, 369, 384, 452, 476`). Three consequences:

1. **`-w` is mandatory but documented as optional.** `--weights` is described as "(Optional)". Without it, `lstm_cell` fails — confirmed by running the parser with only `-f` and getting `Fail!! / Incorrect outputs.`
2. **Argument order is silently load-bearing.** `test/Makefile:40` passes `-f model.onnx -w model_weights.onnx`; `examples/run_basic_maclinux.sh:5` and `README.md:63` pass them **swapped**. For `gemm_small` both exports are identical so it works; for an LSTM it would be catastrophic.
3. **Any node reordering between the two exports attaches weights to the wrong layer.** Nothing detects this.

The underlying reason for the dual-export dance is that ONNX constant-folds LSTM weights into `onnx::LSTM_85`-style initializers in a different gate order. Verified: the folded model's `W` blocks map to torch gates `i, o, f, g` (ONNX's documented `iofc`), while `lstm_cell` (`layers.f90:90-96`) consumes `i, f, g, o`. So the remap index list is `[0, 2, 3, 1]`.

Verified that for Gemm, Conv, and Add, initializer **names are identical** between the folded and unfolded exports — only LSTM is renamed. So name-based lookup works everywhere, and LSTM needs only the gate remap. **`-w` becomes unnecessary entirely.**

**Files:**
- Modify: `fLibrary/onnx_helpers.py` (add `regateLSTM`)
- Modify: `fLibrary/modelParserONNX.py` (remove positional streaming at all sites)
- Modify: `test/test_parser.py`
- Modify: `test/Makefile:40`, `examples/run_basic_maclinux.sh:5`, `README.md:63` and `README.md:122`

**Interfaces:**
- Consumes: `onnx_helpers` from Task 2.
- Produces: `onnx_helpers.regateLSTM(arr, axis=0) -> np.ndarray`, and a parser whose `-w` flag is accepted-but-ignored.

- [ ] **Step 1: Write the failing parser test**

Add to `test/test_parser.py` and register it in `__main__`:

```python
def test_regate_lstm_reorders_iofc_to_ifgo():
    h = 2
    # blocks tagged by gate: ONNX order is i, o, f, c
    onnx_w = np.concatenate([
        np.full((h, 3), 0.0),   # i
        np.full((h, 3), 1.0),   # o
        np.full((h, 3), 2.0),   # f
        np.full((h, 3), 3.0),   # c/g
    ], axis=0)
    got = H.regateLSTM(onnx_w, axis=0)
    check(np.all(got[0*h:1*h] == 0.0), "regateLSTM keeps i first")
    check(np.all(got[1*h:2*h] == 2.0), "regateLSTM moves f second")
    check(np.all(got[2*h:3*h] == 3.0), "regateLSTM moves g third")
    check(np.all(got[3*h:4*h] == 1.0), "regateLSTM moves o last")

def test_regate_lstm_handles_direction_axis():
    h = 2
    onnx_w = np.arange(1 * 4 * h * 3, dtype=float).reshape(1, 4 * h, 3)
    got = H.regateLSTM(onnx_w, axis=1)
    check(got.shape == (1, 4 * h, 3), "regateLSTM preserves shape with direction axis")
    check(np.all(got[0, 1 * h:2 * h] == onnx_w[0, 2 * h:3 * h]), "regateLSTM remaps along axis 1")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd test && python3 test_parser.py`
Expected: `AttributeError: module 'onnx_helpers' has no attribute 'regateLSTM'`

- [ ] **Step 3: Implement `regateLSTM`**

Add to `fLibrary/onnx_helpers.py`:

```python
# ONNX stores LSTM gates as (input, output, forget, cell); roseNNa's
# lstm_cell consumes PyTorch order (input, forget, gate/cell, output).
ONNX_TO_ROSENNA_GATES = [0, 2, 3, 1]


def regateLSTM(arr, axis=0):
    """Reorder the 4 gate blocks of an ONNX LSTM W/R/B tensor along `axis`."""
    n = arr.shape[axis]
    if n % 4 != 0:
        raise ValueError(f"LSTM gate axis {axis} has length {n}, not a multiple of 4")
    h = n // 4
    blocks = [
        np.take(arr, range(g * h, (g + 1) * h), axis=axis)
        for g in ONNX_TO_ROSENNA_GATES
    ]
    return np.concatenate(blocks, axis=axis)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd test && python3 test_parser.py`
Expected: all `ok`, exit 0

- [ ] **Step 5: Replace positional streaming with name lookup**

In `fLibrary/modelParserONNX.py`:

(a) Replace `findWeightsInitializer` (lines 105-108) with a raising version:

```python
def findWeightsInitializer(input_name):
    if input_name in initializer:
        return initializer[input_name][1]
    if input_name in constants:
        return constants[input_name]
    raise KeyError(
        f"no weights found for '{input_name}'; it is neither an initializer "
        f"nor a Constant node output"
    )
```

(b) There are eight `if externalWeightsFile:` sites — lines 191, 204, 259, 274, 368, 383, 451, 475. **Two of them (191, 204) are inside the LSTM block and are handled by step (c) below; do not edit them here.** At the other six, replace the whole five-line block of the form:

```python
                    if externalWeightsFile:
                        f2.write(stranspose(numpy_helper.to_array(true_weights[true_index])))
                        true_index+=1
                    else:
                        f2.write(stranspose(findWeightsInitializer(inp)))
```

with:

```python
                    f2.write(stranspose(findWeightsInitializer(inp)))
```

preserving each site's own indentation and its own variable name (`inp` in the Gemm/Conv/MatMul loops, `node.input[1]` at the Add site on line 455).

(c) Replace the LSTM weight block (lines 188-209) with:

```python
            for inp in node.input[1:3]: #represents ONNX's locations of weights
                for dim in initializer[inp][0]:
                    f.write(str(dim)+" ")
                f2.write(stranspose(regateLSTM(findWeightsInitializer(inp), axis=1)))
                f2.write("\n")
                f.write("\n")
            # ONNX packs Wb and Rb into one (num_directions, 8*hidden) tensor.
            bias = regateLSTM(findWeightsInitializer(node.input[3]), axis=1)
            split = np.split(bias, 2, axis=1)
            for x in range(2):
                f.write(str(int(initializer[node.input[3]][0][1]/2)))
                f.write("\n")
                f2.write(stranspose(split[x]))
                f2.write("\n")
```

Careful: the bias tensor is `(num_directions, 8*hidden)` — the concatenation of `Wb[iofc]` and `Rb[iofc]`. Splitting **before** regating is what the old code did and is correct; regate each half separately:

```python
            wb, rb = np.split(findWeightsInitializer(node.input[3]), 2, axis=1)
            for half in (wb, rb):
                f.write(str(int(initializer[node.input[3]][0][1]/2)))
                f.write("\n")
                f2.write(stranspose(regateLSTM(half, axis=1)))
                f2.write("\n")
```

Use this second form; delete the first. Also delete the now-dead `if not externalWeightsFile: split = np.split(...)` line at 199-200.

(d) Add the import at the top: extend the Task 2 import line with `regateLSTM`.

(e) Delete the now-unused `true_index = 0` (line 147), `true_weights = ...` (line 148), and the `externalWeightsFile` flag logic (lines 28-33), replacing the weights-load block with:

```python
if weights is not None:
    print("note: --weights/-w is no longer needed and is ignored; "
          "weights are now read by name from the structure file.")
```

- [ ] **Step 6: Verify LSTM works with only `-f`**

Run:
```bash
cd test && python3 ../goldenFiles/lstm_cell/lstm_cell.py >/dev/null 2>&1 && \
  python3 ../fLibrary/modelParserONNX.py -f ../goldenFiles/lstm_cell/lstm_cell.onnx \
    -i ../goldenFiles/lstm_cell/lstm_cell_inferred.onnx >/dev/null 2>&1 && \
  rm -f modelCreator.f90 userTesting.f90 && make output >/dev/null 2>&1 && \
  ./output 2>outputCase.txt >/dev/null; python3 -Wi testChecker.py lstm_cell
```
Expected: `Outputs match! Pass!` (this exact command printed `Fail!! / Incorrect outputs.` before the fix)

- [ ] **Step 7: Drop `-w` from all call sites**

- `test/Makefile:40`: delete ` -w $(MAIN)goldenFiles/$(case)/$(case)_weights.onnx`
- `examples/run_basic_maclinux.sh:5`: replace with `python3 modelParserONNX.py -f ../goldenFiles/gemm_small/gemm_small.onnx`
- `README.md:63`: same replacement
- `README.md:122`: change to ``Run `python modelParserONNX.py -f path/to/model.onnx` to reconstruct the model.``
- `README.md`: in the "Converting an LSTM?" section (around line 88), replace the two-export instructions with a note that only the `do_constant_folding=True` export is needed, since the parser now remaps ONNX's `iofc` gate order internally.

- [ ] **Step 8: Run the full suite**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!`

- [ ] **Step 9: Commit**

```bash
git clean -fdX
git add fLibrary/onnx_helpers.py fLibrary/modelParserONNX.py test/test_parser.py test/Makefile examples/run_basic_maclinux.sh README.md
git commit -m "fix: look weights up by name and remap ONNX iofc gate order

Replaces positional streaming from a second unoptimized export, which
made -w mandatory-but-documented-optional, made -f/-w order silently
load-bearing (two call sites had them swapped), and attached weights to
the wrong layer under any node reordering."
```

---

### Task 9: Broadcast axis chosen by value instead of position

**Problem:** `fourDTransform` (`onnx_helpers.py`, originally `modelParserONNX.py:110-119`) locates each broadcast axis with `trueshape.index(dim)` — a **value** search that returns the first match. ONNX and numpy broadcast by right-aligning shapes. Verified failures:

| target shape | added tensor | `fourDTransform` gives | correct |
|---|---|---|---|
| `[1,3,3,3]` | `(3,3)` | `[1,3,1,1]` | `[1,1,3,3]` |
| `[1,4,3,3]` | `(1,1,1)` | `[1,1,1,1]` | `[1,1,1,1]` (ok) |
| `[1,3,3,3]` | `(3,)` | `[1,3,1,1]` | `[1,3,1,1]` (ok by luck) |

The first row is wrong, and `spreadInfo` returns the *identical* `[3,3,4,3]` for the `(3,)` and `(3,3)` cases, so the two are indistinguishable downstream.

**Files:**
- Modify: `fLibrary/onnx_helpers.py`
- Modify: `test/test_parser.py`

**Interfaces:**
- Consumes: `check/1` from `test_parser.py`.
- Produces: `fourDTransform` with an unchanged signature and corrected semantics.

- [ ] **Step 1: Write the failing test**

Add to `test/test_parser.py` and register it:

```python
def test_four_d_transform_right_aligns():
    check(H.fourDTransform([1,4,3,3], (4,)) == [1,4,1,1] or
          H.fourDTransform([1,4,3,3], (4,)) == [1,1,1,4],
          "fourDTransform handles a trailing-axis vector")
    check(H.fourDTransform([1,3,3,3], (3,3)) == [1,1,3,3],
          "fourDTransform right-aligns a 2D add against a 4D target")
    check(H.fourDTransform([1,4,3,3], (3,3)) == [1,1,3,3],
          "fourDTransform right-aligns regardless of channel count")
    check(H.fourDTransform([1,4,3,3], (1,4,1,1)) == [1,4,1,1],
          "fourDTransform passes through an already-4D shape")
```

Note the first case: ONNX right-alignment puts a bare `(4,)` at the **last** axis, which for a channel bias is `[1,1,1,4]`, not `[1,4,1,1]`. PyTorch emits channel biases pre-shaped as `(C,1,1)`, so right-alignment is still correct — the assertion accepts either to document the ambiguity, and Task 13 adds the validation that catches a genuine mismatch.

- [ ] **Step 2: Run to verify it fails**

Run: `cd test && python3 test_parser.py`
Expected: `FAIL fourDTransform right-aligns a 2D add against a 4D target`

- [ ] **Step 3: Right-align instead of value-matching**

Replace `fourDTransform` in `fLibrary/onnx_helpers.py`:

```python
def fourDTransform(trueshape, toBeTransformedShape):
    """Right-align `toBeTransformedShape` into 4 dimensions, per ONNX broadcasting.

    Every axis of the result must be either 1 or equal to the corresponding
    axis of `trueshape`, otherwise the two do not broadcast.
    """
    t = list(toBeTransformedShape)
    if len(t) > 4:
        raise ValueError(f"cannot broadcast a {len(t)}-D tensor into 4 dimensions")
    new = [1, 1, 1, 1]
    for i, d in enumerate(reversed(t)):
        new[3 - i] = d
    true4d = fakeFourD(list(trueshape))
    for i, (a, b) in enumerate(zip(true4d, new)):
        if b != 1 and b != a:
            raise ValueError(
                f"axis {i}: cannot broadcast {toBeTransformedShape} "
                f"against {trueshape} ({b} vs {a})"
            )
    return new
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd test && python3 test_parser.py`
Expected: all `ok`, exit 0

- [ ] **Step 5: Run the full suite**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!` — but note this is a **regression check, not coverage**. Verified: on mnist's actual shapes (`[1,8,28,28]` vs `(8,1,1)` and `[1,16,14,14]` vs `(16,1,1)`) the old value-matching code and the new right-aligning code return identical results, because a length-3 tuple's unique nonzero dim lands at the same position either way. The golden suite fails to contradict this fix; it does not corroborate it. The two discriminating assertions in `test/test_parser.py` are the only real evidence.

- [ ] **Step 6: Commit**

```bash
git clean -fdX
git add fLibrary/onnx_helpers.py test/test_parser.py
git commit -m "fix: right-align broadcast axes instead of matching by value"
```

---

### Task 10: Gemm silently corrupts when `transB=0`

**Problem:** `linear_layer` (`layers.f90:11-23`) takes `transInp = 1 - transB` from `modelCreator.fpp:89`. The `transInp /= 0` branch computes `matmul(lin%weights, inp)` — `B × A`, not `A × B`. ONNX Gemm with `transB=0` means `Y = A(m,k) × B(k,n)`. Verified: `A(1,2) × B(2,3)` should give `[5, 11, 17]`; `linear_layer(a, L, 1)` returned an array of shape **(2,2)** with no error or warning. `alpha`, `beta`, and `transA` are read from neither the parser nor the layer.

No golden case exercises it — every case comes from `nn.Linear`, which exports `transB=1`, and the `MatMul` fallback (`modelParserONNX.py:468`) hardcodes `1`. So this is latent, not currently firing.

**Files:**
- Modify: `fLibrary/layers.f90:11-23`
- Modify: `fLibrary/modelParserONNX.py:245-281`
- Modify: `test/unit_tests.f90`

**Interfaces:**
- Consumes: `check/1`.
- Produces: `linear_layer(inp, lin, transInp)` — signature unchanged, `transInp /= 0` branch corrected.

- [ ] **Step 1: Write the failing test**

Add to `test/unit_tests.f90` and register it:

```fortran
    subroutine test_linear_layer_untransposed()
        real(c_double), allocatable :: a(:,:)
        type(linLayer) :: L
        logical :: ok
        allocate(a(1,2))
        a = reshape([1.0d0, 2.0d0], [1,2])
        allocate(L%weights(2,3))
        L%weights = reshape([1.0d0,2.0d0,3.0d0,4.0d0,5.0d0,6.0d0], [2,3])
        allocate(L%biases(3))
        L%biases = 0.0d0
        call linear_layer(a, L, 1)
        ok = all(shape(a) == [1,3])
        if (ok) ok = all(abs(reshape(a, [3]) - [5.0d0, 11.0d0, 17.0d0]) < 1.0d-12)
        call check(ok, 'linear_layer computes A*B when transB=0')
    end subroutine
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd test && make unit`
Expected: `FAIL linear_layer computes A*B when transB=0`

- [ ] **Step 3: Fix the branch**

Replace `fLibrary/layers.f90:11-23`:

```fortran
    !=== for Gemm operations ======
    subroutine linear_layer(inp, lin, transInp)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: transInp
        REAL (c_double), ALLOCATABLE, intent(inout) :: inp(:,:)
        TYPE(linLayer), INTENT(IN) :: lin
        REAL (c_double), ALLOCATABLE :: bias_broadcast(:,:)

        if (transInp == 0) THEN
            !== weights are (out,in): Y = X * W^T + b
            bias_broadcast = SPREAD(lin%biases, 1, size(inp,1))
            inp = matmul(inp, TRANSPOSE(lin%weights)) + bias_broadcast
        ELSE
            !== weights are (in,out): Y = X * W + b
            bias_broadcast = SPREAD(lin%biases, 1, size(inp,1))
            inp = matmul(inp, lin%weights) + bias_broadcast
        END IF
    end subroutine
```

Both branches now broadcast the bias across rows with `SPREAD(..., 1, size(inp,1))`, giving `(batch, out)` directly and removing the `TRANSPOSE(bias_broadcast)` round-trip. `bias_broadcast` becomes allocatable so its shape follows the branch.

- [ ] **Step 4: Reject the Gemm attributes that are still unsupported**

In `fLibrary/modelParserONNX.py`, inside the `elif layer == "Gemm":` block after `names` is built (line 249), insert:

```python
            if names.get('transA', 0):
                raise NotImplementedError(
                    "Gemm transA=1 is not supported by roseNNa")
            for attr in node.attribute:
                if attr.name in ('alpha', 'beta') and abs(attr.f - 1.0) > 1e-12:
                    raise NotImplementedError(
                        f"Gemm {attr.name}={attr.f} is not supported by roseNNa "
                        f"(only 1.0)")
```

- [ ] **Step 5: Run unit tests and the full suite**

Run: `cd test && make unit`
Expected: `UNIT TESTS: all passed`

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!` (every Gemm case goes through the `transInp == 0` branch, confirming the bias rework is equivalent)

- [ ] **Step 6: Commit**

```bash
git clean -fdX
git add fLibrary/layers.f90 fLibrary/modelParserONNX.py test/unit_tests.f90
git commit -m "fix: correct Gemm transB=0 branch and reject alpha/beta/transA

The transB=0 path computed B*A instead of A*B, producing a wrongly
shaped result with no error."
```

---

### Task 11: Bare excepts hide parse failures; Squeeze appends up to three times

**Problem:** `modelParserONNX.py` has 10 bare `except:` clauses. They convert `KeyError`/`IndexError` — the signature of an unrecognized graph — into "quietly take another path". The MaxPool defect in Task 4 is exactly the class of thing they hide.

Worst case is `Squeeze` (`:291-304`): three sequential `try` blocks, each of which can append to `modelArch`, so a single Squeeze node can emit up to three layer entries. The first block passes `ioMap[node.input[1]]` — a **string** — as the axes list, which `modelCreator.fpp:135` then tests with `num not in tup[3][0]`, silently matching characters instead of axes.

**Files:**
- Modify: `fLibrary/modelParserONNX.py` (all 10 sites)

**Interfaces:**
- Produces: a parser that raises a named exception rather than emitting a wrong model.

- [ ] **Step 1: Inventory the bare excepts**

Run: `grep -n "except:" fLibrary/modelParserONNX.py`
Expected: 10 line numbers — record them before editing.

- [ ] **Step 2: Rewrite the Squeeze block**

Replace `fLibrary/modelParserONNX.py:284-308` with a single-append version:

```python
        elif layer == "Squeeze": #changes shape
            f.write(layer)
            f.write("\n")
            rank = len(intermediateShapes[node.input[0]])
            axes = None
            if len(node.input) > 1:
                axes = findWeightsInitializer(node.input[-1]).tolist()
            else:
                for attr in node.attribute:
                    if attr.name == "axes":
                        axes = list(attr.ints)
                        break
            if axes is None:
                # ONNX default: squeeze every axis of extent 1
                axes = [i for i, d in enumerate(intermediateShapes[node.input[0]]) if d == 1]
            axes = [a if a >= 0 else a + rank for a in axes]
            modelArch.append(("Squeeze", (ioMap[node.input[0]], rank),
                              ["output" + extra], [axes]))
            inputs.append(["output"+extra, len(intermediateShapes[node.output[0]])])
            ioMap[node.output[0]] = "output" + extra
            extra = str(int(extra)+1)
```

This also fixes negative axis indices, which the old code passed through unnormalized.

- [ ] **Step 3: Narrow the remaining excepts**

For each remaining bare `except:`, replace with the specific exception the `try` can raise and re-raise anything else. The three shape-inference fallbacks near the top (lines 29-40) are legitimately optional and become:

```python
try:
    inferred = onnx.load(inferred)
    value_info = inferred.graph.value_info
except (TypeError, FileNotFoundError, onnx.checker.ValidationError):
    value_info = onnx.shape_inference.infer_shapes(onnxModel).graph.value_info
```

The `Transpose` default-perm fallback (`:165-167`) becomes `except KeyError:`. The `LSTM` initial-state fallback (`:179-186`) becomes `except (KeyError, IndexError):`. The `Reshape` fallback (`:316-328`) becomes `except KeyError:`. The `MatMul`-to-`Gemm` fallback (`:461-486`) becomes `except KeyError:`.

- [ ] **Step 4: Verify none remain**

Run: `grep -c "except:" fLibrary/modelParserONNX.py`
Expected: `0`

- [ ] **Step 5: Verify unsupported layers still fail loudly**

The `else:` branch (`:519-523`) prints `NOT SUPPORTED` and then does `ioMap[node.output[0]] = ioMap[node.input[0]]`, silently aliasing the output to the input — the model then runs and produces a wrong answer. Replace lines 519-523 with:

```python
        else:
            raise NotImplementedError(
                f"{layer} is not supported by roseNNa. "
                f"Model architecture parsed so far: {modelArch}")
```

- [ ] **Step 6: Run the full suite**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!`

The `Pad` and `Constant` branches must still be reached as no-ops — `mnist` has neither, but `lstm_gemm` has `Constant`. Confirm `lstm_gemm` still passes specifically:

Run: `cd test && make testing case=lstm_gemm 2>&1 | tail -1`
Expected: `Outputs match! Pass!`

- [ ] **Step 7: Commit**

```bash
git clean -fdX
git add fLibrary/modelParserONNX.py
git commit -m "fix: narrow bare excepts, emit one Squeeze layer, fail on unsupported ops"
```

---

### Task 12: ONNX tensor names are emitted as raw Fortran identifiers

**Problem:** `modelCreator.fpp:46, 52, 63, 69, 73` interpolate ONNX tensor names directly into Fortran declarations. Torch's default names are not valid Fortran identifiers — verified that exporting without `input_names=` produces a graph input named `onnx::Gemm_0` and a node output named `3`. Every golden case passes `input_names=['input']`, so the generated code compiles only because the test suite avoids the problem. A user following the README's own export snippet without `input_names` gets a compile error from generated code they never wrote.

**Files:**
- Modify: `fLibrary/onnx_helpers.py`
- Modify: `fLibrary/modelParserONNX.py`
- Modify: `test/test_parser.py`

**Interfaces:**
- Consumes: `check/1`.
- Produces: `onnx_helpers.sanitize(name: str) -> str`, applied at every point a name enters `ioMap`, `inputs`, `outputs`, `outShape`, or `trueInputs`.

- [ ] **Step 1: Write the failing test**

Add to `test/test_parser.py` and register it:

```python
def test_sanitize_produces_fortran_identifiers():
    import re
    ident = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")
    for raw in ["onnx::Gemm_0", "3", "/layer1/Gemm_output_0", "input", "a.b.c", ""]:
        got = H.sanitize(raw)
        check(bool(ident.match(got)), f"sanitize({raw!r}) -> {got!r} is a valid identifier")
    check(H.sanitize("input") == "input", "sanitize leaves valid names alone")
    check(H.sanitize("onnx::Gemm_0") != H.sanitize("onnx::Gemm_1"),
          "sanitize keeps distinct names distinct")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd test && python3 test_parser.py`
Expected: `AttributeError: module 'onnx_helpers' has no attribute 'sanitize'`

- [ ] **Step 3: Implement `sanitize`**

Add `import hashlib` and `import re` to the top of `fLibrary/onnx_helpers.py` alongside the existing `import itertools`, then add:

```python
_VALID = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")


def sanitize(name):
    """Map an ONNX tensor name onto a valid, collision-free Fortran identifier."""
    if _VALID.match(name):
        return name
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "t_" + cleaned
    # A short digest of the original keeps distinct names distinct after cleaning.
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
    return f"{cleaned[:48]}_{digest}"
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd test && python3 test_parser.py`
Expected: all `ok`, exit 0

- [ ] **Step 5: Apply it at the boundary**

In `fLibrary/modelParserONNX.py`, apply `sanitize` where names first enter the emitted namespace, so downstream lookups stay consistent:

- line 51: `ioMap[inp.name] = sanitize(inp.name)`
- line 61: `out[x.name] = ...` stays keyed on the raw name (it is matched against `node.output`), but line 159 becomes `outShape.append([sanitize(x), out[x]])`
- line 526: `outputs[sanitize(x)] = ioMap[x]`
- line 527: `trueInputs = [[sanitize(x.name), [...]] for x in ...]`

Every `ioMap[node.output[0]] = "output" + extra` site already generates a safe name and needs no change. Add `sanitize` to the import line.

- [ ] **Step 6: Verify against an unnamed export**

```bash
cd /private/tmp && python3 -c "
import torch
m = torch.nn.Sequential(torch.nn.Linear(2,3), torch.nn.ReLU())
torch.onnx.export(m, torch.rand(1,2), 'unnamed.onnx', opset_version=10, dynamo=False)
print('exported')"
cd fLibrary && python3 modelParserONNX.py -f /private/tmp/unnamed.onnx >/dev/null && \
  fypp -I. modelCreator.fpp /private/tmp/mc.f90 && \
  grep -E "onnx::|:: 3$" /private/tmp/mc.f90 | head
```
Expected: no output from the final `grep` — no raw ONNX punctuation survives into the generated Fortran.

Then confirm it actually compiles:

Run: `cd fLibrary && mkdir -p objFiles && make output 2>&1 | tail -3`
Expected: no compile errors

- [ ] **Step 7: Run the full suite**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!`

- [ ] **Step 8: Commit**

```bash
git clean -fdX
git add fLibrary/onnx_helpers.py fLibrary/modelParserONNX.py test/test_parser.py
git commit -m "fix: sanitize ONNX tensor names into valid Fortran identifiers"
```

---

### Task 13: Silently-ignored layer attributes

**Problem:** Conv and the pooling layers accept parameters they then ignore, producing a confident wrong answer:

- **`dilations`** is parsed (`modelParserONNX.py:360`), passed to `conv` (`layers.f90:188`), and never read.
- **`ceil_mode`** is passed to both pooling subroutines (`layers.f90:240`, `:278`) and never read; output extents always floor.
- **Asymmetric pads**: `padding` (`layers.f90:134-181`) applies `arr(1)` and `arr(2)` symmetrically, using only the first two of ONNX's four pad values — so the `SAME_UPPER`/`SAME_LOWER` asymmetric pads the parser itself computes (`:354`, `:356`) are applied symmetrically.
- **Non-square kernels**: `conv` derives `kernel_size = SIZE(convWeights, dim=3)` (`layers.f90:209`), a single scalar used for both extents.
- **Conv `auto_pad`**: `auto_pad = True` is set whenever the attribute is *present* (`:342-344`), including `NOTSET` and `VALID`, both of which mean "do not pad like SAME". MaxPool guards with `!= "NOTSET"` (`:407`) — still wrong for `VALID`, but inconsistently so. Conv also dereferences `attributes['pads']` unconditionally at line 360, raising `KeyError` when a Conv node has neither `auto_pad` nor an explicit `pads`.

The right fix at this scope is to **detect and reject** rather than implement four new features. A library whose users trust numerical output should refuse rather than guess.

**Files:**
- Modify: `fLibrary/modelParserONNX.py` (Conv and both pooling branches)
- Modify: `test/test_parser.py`

**Interfaces:**
- Consumes: `check/1`.
- Produces: `onnx_helpers.checkSupported(op: str, attrs: dict) -> None`, raising `NotImplementedError`.

- [ ] **Step 1: Write the failing test**

Add to `test/test_parser.py` and register it:

```python
def test_check_supported_rejects_unimplemented_attrs():
    def raises(fn):
        try:
            fn()
        except NotImplementedError:
            return True
        return False

    check(raises(lambda: H.checkSupported("Conv", {"dilations": [2, 2], "kernel_shape": [3, 3], "pads": [0]*4, "strides": [1, 1]})),
          "rejects dilations != 1")
    check(raises(lambda: H.checkSupported("MaxPool", {"ceil_mode": 1, "kernel_shape": [2, 2], "pads": [0]*4, "strides": [1, 1]})),
          "rejects ceil_mode = 1")
    check(raises(lambda: H.checkSupported("Conv", {"kernel_shape": [3, 3], "pads": [1, 1, 2, 2], "strides": [1, 1]})),
          "rejects asymmetric pads")
    check(raises(lambda: H.checkSupported("Conv", {"kernel_shape": [3, 5], "pads": [0]*4, "strides": [1, 1]})),
          "rejects non-square kernels")
    check(not raises(lambda: H.checkSupported("Conv", {"dilations": [1, 1], "kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [2, 2]})),
          "accepts a supported Conv")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd test && python3 test_parser.py`
Expected: `AttributeError: module 'onnx_helpers' has no attribute 'checkSupported'`

- [ ] **Step 3: Implement `checkSupported`**

Add to `fLibrary/onnx_helpers.py`:

```python
def checkSupported(op, attrs):
    """Raise NotImplementedError for attributes roseNNa parses but ignores."""
    dilations = list(attrs.get("dilations", [1, 1]))
    if any(d != 1 for d in dilations):
        raise NotImplementedError(
            f"{op}: dilations={dilations} is parsed but ignored by roseNNa; "
            f"only dilations of 1 are supported")

    if int(attrs.get("ceil_mode", 0)) != 0:
        raise NotImplementedError(
            f"{op}: ceil_mode=1 is parsed but ignored by roseNNa; "
            f"output extents are always floored")

    kernel = list(attrs.get("kernel_shape", []))
    if len(kernel) == 2 and kernel[0] != kernel[1]:
        raise NotImplementedError(
            f"{op}: non-square kernel {kernel} is not supported by roseNNa")

    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    if len(pads) == 4 and (pads[0] != pads[2] or pads[1] != pads[3]):
        raise NotImplementedError(
            f"{op}: asymmetric pads {pads} are not supported by roseNNa; "
            f"padding is applied symmetrically")
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd test && python3 test_parser.py`
Expected: all `ok`, exit 0

- [ ] **Step 5: Fix Conv's `auto_pad` handling and call the check**

In `fLibrary/modelParserONNX.py`, in the `elif layer == "Conv":` branch, replace lines 342-344:

```python
                elif name == "auto_pad":
                    attributes['auto_pad'] = attr.s.decode('ASCII')
                    if attributes['auto_pad'] not in ("NOTSET", "VALID"):
                        auto_pad = True
```

and add defaults plus the check immediately before the `modelArch.append` on line 360:

```python
            attributes.setdefault('pads', [0, 0, 0, 0])
            attributes.setdefault('strides', [1, 1])
            attributes.setdefault('dilations', [1, 1])
            checkSupported("Conv", attributes)
```

- [ ] **Step 6: Apply the same guard to both pooling branches**

In the `MaxPool` branch, change the `auto_pad` guard (line 407) to `not in ("NOTSET", "VALID")` and add `checkSupported("MaxPool", attributes)` immediately before its `modelArch.append` (line 422).

In the `AveragePool` branch, build a dict and check it before the append on line 436:

```python
            poolAttrs = {
                'ceil_mode': names.get('ceil_mode', 0),
                'pads': names.get('pads', [0,0,0,0]),
                'strides': names.get('strides', [1,1]),
                'kernel_shape': names.get('kernel_shape', [1,1]),
            }
            checkSupported("AveragePool", poolAttrs)
```

Add `checkSupported` to the import line.

- [ ] **Step 7: Run the full suite**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!`

All existing cases use square kernels, symmetric pads, `dilations=1`, `ceil_mode=0`, so none should trip the new guard. If a case now raises, that case was silently wrong — investigate before relaxing the check.

One interaction to expect: the parser's own `SAME_UPPER`/`SAME_LOWER` path emits `[pad, pad, pad+1, pad+1]` when `kernel_shape` is even, which the new symmetric-pads check correctly rejects. That combination was silently producing wrong output before, so the rejection is the intended outcome — do not weaken `checkSupported` to let it through.

- [ ] **Step 8: Document the limits**

Add a "Supported ONNX operators and limits" section to `README.md` after the "Compiling roseNNa" section, listing: supported ops (`Gemm`, `MatMul`, `Conv`, `MaxPool`, `AveragePool`, `LSTM`, `Add`, `Reshape`, `Transpose`, `Squeeze`, `Relu`, `Sigmoid`, `Tanh`), and the limits now enforced (dilations must be 1, `ceil_mode` must be 0, kernels must be square, pads must be symmetric, Gemm `alpha`/`beta` must be 1 and `transA` must be 0).

- [ ] **Step 9: Commit**

```bash
git clean -fdX
git add fLibrary/onnx_helpers.py fLibrary/modelParserONNX.py test/test_parser.py README.md
git commit -m "fix: reject parsed-but-ignored Conv/pool attributes instead of guessing"
```

---

## Phase 3 — API and performance

### Task 14: C API signature mismatch and unchecked file I/O

**Problem:** `initialize()` is `bind(c)` with **no arguments** (`reader.f90:38`), but `examples/cAPI.c:4` declares `void initialize(char*, char*)` and calls it with two paths. The paths are silently discarded; `reader.f90:49-50` hardcodes `onnxModel.txt` and `onnxWeights.txt` on units 10 and 11 with no `status='old'` and no `iostat`, so a missing file is a runtime crash with no message. The C example's own comment implies configurable paths that do not exist. Callers must also supply column-major data, which is documented nowhere (`capiTester.f90:8` does the `RESHAPE(..., order=[2,1])` dance without explanation).

Verified that gfortran accepts `optional` dummies in a `bind(c)` procedure, so one entry point can serve both callers.

**Files:**
- Modify: `fLibrary/reader.f90:38-99`
- Modify: `examples/cAPI.c`
- Modify: `README.md` (memory-layout note)

**Interfaces:**
- Consumes: nothing new.
- Produces: `initialize([model_file, weights_file])` — both optional, C-interoperable; existing `CALL initialize()` sites keep working unchanged.

- [ ] **Step 1: Reproduce**

Run: `grep -n "bind(c,name=\"initialize\")" fLibrary/reader.f90 && grep -n "void initialize" examples/cAPI.c`
Expected: a zero-argument Fortran definition against a two-argument C declaration.

Run: `cd /private/tmp && mkdir -p emptydir && cd emptydir && cp "$(git rev-parse --show-toplevel)"/fLibrary/*.f90 . && gfortran -c activation_funcs.f90 derived_types.f90 layers.f90 reader.f90 2>/dev/null && echo "compiles; missing-file behavior is an unhandled runtime error"`

- [ ] **Step 2: Add a C-string helper and optional path arguments**

In `fLibrary/reader.f90`, add to the `contains` section before `initialize`:

```fortran
    function c_to_f_string(s) result(str)
        character(kind=c_char, len=1), intent(in) :: s(*)
        character(len=:), allocatable :: str
        integer :: i, n
        n = 0
        do
            if (s(n+1) == c_null_char) exit
            n = n + 1
            if (n > 4096) exit
        end do
        allocate(character(len=n) :: str)
        do i = 1, n
            str(i:i) = s(i)
        end do
    end function
```

- [ ] **Step 3: Rewrite `initialize`**

Replace `fLibrary/reader.f90:38-52` (through the `read(10, *) numLayers` line):

```fortran
    subroutine initialize(model_file, weights_file) bind(c,name="initialize")
        character(kind=c_char, len=1), intent(in), optional :: model_file(*)
        character(kind=c_char, len=1), intent(in), optional :: weights_file(*)
        INTEGER :: Reason, ios
        INTEGER :: modelUnit, weightsUnit
        character(len=:), allocatable :: mpath, wpath

        mpath = "onnxModel.txt"
        wpath = "onnxWeights.txt"
        if (present(model_file))   mpath = c_to_f_string(model_file)
        if (present(weights_file)) wpath = c_to_f_string(weights_file)

        ALLOCATE(lstmLayers(0))
        ALLOCATE(linLayers(0))
        ALLOCATE(convLayers(0))
        ALLOCATE(maxpoolLayers(0))
        ALLOCATE(avgpoolLayers(0))
        ALLOCATE(addLayers(0))
        ALLOCATE(reshapeLayers(0))

        open(newunit=modelUnit, file=mpath, status='old', action='read', iostat=ios)
        if (ios /= 0) then
            write(error_unit,'(a)') "roseNNa: cannot open model file '"//mpath//"'"
            error stop 1
        end if
        open(newunit=weightsUnit, file=wpath, status='old', action='read', iostat=ios)
        if (ios /= 0) then
            write(error_unit,'(a)') "roseNNa: cannot open weights file '"//wpath//"'"
            error stop 1
        end if

        read(modelUnit, *, iostat=ios) numLayers
        if (ios /= 0) then
            write(error_unit,'(a)') "roseNNa: '"//mpath//"' is empty or malformed"
            error stop 1
        end if
```

Then replace every remaining literal `10` and `11` in the body of `initialize` with `modelUnit` and `weightsUnit` (lines 55, 60-81 in the original), and close both units before `end subroutine`:

```fortran
        close(modelUnit)
        close(weightsUnit)
    end subroutine
```

Add `use iso_fortran_env, only: error_unit` to the module's use statements at `reader.f90:3-5`.

- [ ] **Step 4: Update the C example**

Replace `examples/cAPI.c`:

```c
#include <stdio.h>

void use_model(double * i0, double * o0);
void initialize(const char * model_file, const char * weights_file);

int main(void) {

    /* roseNNa expects column-major (Fortran) ordering. */
    double a[2] = {1, 1};
    double b[3];

    initialize("onnxModel.txt", "onnxWeights.txt");
    use_model(a, b);

    for (int i = 0; i < 3; i++) {
        printf("%f ", b[i]);
    }
    printf("\n");
    return 0;
}
```

- [ ] **Step 5: Verify both call styles work**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!` (every golden case uses `CALL initialize()` with no arguments, exercising the optional-absent path)

Then the C path:

```bash
cd fLibrary && python3 ../goldenFiles/gemm_small/gemm_small.py >/dev/null 2>&1 && \
  python3 modelParserONNX.py -f ../goldenFiles/gemm_small/gemm_small.onnx >/dev/null && \
  make library >/dev/null 2>&1 && \
  gcc -c ../examples/cAPI.c -o cAPI.o && \
  gfortran -o capi cAPI.o libcorelib.a && ./capi
```
Expected: three floating-point numbers, matching the `print` from `gemm_small.py`

And the missing-file path:

```bash
cd /private/tmp && mkdir -p nofiles && cd nofiles && \
  cp "$(git rev-parse --show-toplevel)"/fLibrary/libcorelib.a . 2>/dev/null; \
  echo "expect a clear 'cannot open model file' message rather than a crash"
```

- [ ] **Step 6: Document the memory layout**

Add to `README.md` in the "Fortran use" section:

> **Memory layout.** `use_model` expects inputs in Fortran (column-major) order. A C caller with a row-major array must transpose it first; a Fortran caller building an array from a row-major literal should use `RESHAPE(..., order=[2,1])`, as `examples/capiTester.f90` does.

- [ ] **Step 7: Commit**

```bash
git clean -fdX
git add fLibrary/reader.f90 examples/cAPI.c README.md
git commit -m "fix: accept optional model/weight paths and report I/O errors

initialize() was bind(c) with no arguments while cAPI.c declared and
called it with two; the paths were silently discarded and a missing
file crashed with no message."
```

---

### Task 15: Eliminate redundant padding copies and the implicit SAVE

**Problem:** Two independent issues in `layers.f90`:

1. `padding` (`:134-181`) declares both its result and a local `formatted` at full padded extent, fills `formatted`, then copies it to `padding` (`:180`). The caller then receives it into another full-extent array. That is three copies of the padded tensor per layer, two of them automatic (stack) arrays. Task 5 already made the pooling callers' `padded` allocatable; `conv` (`:196`) still uses an automatic array.
2. `REAL (c_double) :: sumini = 0` (`:205`) — initialization in a declaration implies `SAVE` in Fortran. It works only because the code resets it to 0 after each use (`:226`). It is not reentrant, so the subroutine cannot be called from an OpenMP region or recursively.

**Files:**
- Modify: `fLibrary/layers.f90:134-233`

**Interfaces:**
- Consumes: nothing new.
- Produces: `padding` and `conv` with unchanged signatures and results.

- [ ] **Step 1: Establish the behavioral baseline**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!` — this task must not change any number.

- [ ] **Step 2: Write the padded result directly**

Replace `fLibrary/layers.f90:134-181`:

```fortran
    function padding(arr, input) result(formatted)
        implicit none
        integer, dimension(:), intent(in) :: arr
        REAL (c_double), dimension(:,:,:,:), intent(in) :: input
        REAL (c_double), dimension(size(input,1),size(input,2), &
            size(input,3)+2*arr(1),size(input,4)+2*arr(2)) :: formatted

        formatted = 0.0d0
        formatted(:, :, arr(1)+1 : arr(1)+size(input,3), &
                        arr(2)+1 : arr(2)+size(input,4)) = input
    end function padding
```

This removes the quadruple-nested manual zero-fill (a whole-array assignment plus one strided section assignment does the same work) and the result-to-result copy.

- [ ] **Step 3: Make `conv`'s padded buffer allocatable and `sumini` local**

In `fLibrary/layers.f90`, in `conv`:

- replace the automatic declaration at `:196`:

```fortran
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:,:) :: padded
```

- replace `REAL (c_double) :: sumini = 0` at `:205` with:

```fortran
        REAL (c_double) :: sumini
```

- the existing `sumini = 0` reset inside the loop (`:226`) stays; add an initialization at the top of the `overImage` loop body so the first iteration is well-defined. Replace the inner accumulation block (`:218-227`) with:

```fortran
                DO overImage = 0, (outRowDim*outColDim)-1 !==iterating kernel through the whole image
                    sumini = 0
                    DO inner = 0, in_channels-1 !==applying kernel to each input image
                        sumini = sumini + SUM(padded(itBatches,inner+1, &
                        (1 + (overImage/outRowDim)*strides(1)):((overImage/outRowDim)*strides(1)+kernel_size) &
                        ,(1 + MODULO(overImage,outRowDim)*strides(2)):(MODULO(overImage,outRowDim)*strides(2)+kernel_size)) &
                            * convWeights(outer+1,inner+1,:,:))
                    END DO
                    out(itBatches,outer+1,overImage/outRowDim + 1,MODULO(overImage,outRowDim)+1) = sumini + bias(outer+1)
                END DO
```

- [ ] **Step 4: Verify nothing changed numerically**

Run: `cd test && make unit`
Expected: `UNIT TESTS: all passed`

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!`

- [ ] **Step 5: Confirm the conv and pooling cases specifically**

Run: `cd test && for c in conv_basic conv_padding conv_strides conv_padding-stride avgpool_basic maxpool_padding mnist; do printf '%-22s ' $c; make testing case=$c 2>&1 | tail -1; done`
Expected: `Outputs match! Pass!` for all seven

- [ ] **Step 6: Commit**

```bash
git clean -fdX
git add fLibrary/layers.f90
git commit -m "perf: write padded buffer in place, drop implicit SAVE on sumini"
```

---

## Phase 4 — Hygiene

### Task 16: De-duplicate `modelCreator.fpp`

**Problem:** `test/modelCreator.fpp` is byte-identical to `fLibrary/modelCreator.fpp` (verified with `diff`). Two copies of the file that generates the entire model will drift.

**Files:**
- Delete: `test/modelCreator.fpp`
- Modify: `test/Makefile`

**Interfaces:**
- Consumes: the `-I` requirement from Global Constraints.

- [ ] **Step 1: Confirm they are identical**

Run: `diff fLibrary/modelCreator.fpp test/modelCreator.fpp && echo IDENTICAL`
Expected: `IDENTICAL`

- [ ] **Step 2: Delete the copy and point the Makefile at the original**

```bash
git rm test/modelCreator.fpp
```

In `test/Makefile`, change `SRC` (line 6) to build only `userTesting.fpp`, and add an explicit rule for the library's copy. Because fypp resolves `#:include 'variables.fpp'` against the include path and `variables.fpp` is generated in `test/`, the rule needs `-I.`:

```make
SRC=userTesting.fpp
OBJ2=${SRC:.fpp=.o} modelCreator.o

modelCreator.f90: $(DIR)/modelCreator.fpp variables.fpp
	fypp -I. $(DIR)/modelCreator.fpp modelCreator.f90
```

Also add `-I.` to the generic `%.f90: %.fpp` rule (lines 15-16) so `userTesting.fpp` keeps resolving its own includes:

```make
%.f90: %.fpp variables.fpp
	fypp -I. $< $*.f90
```

- [ ] **Step 3: Verify fypp resolves the include**

Run: `cd test && rm -f modelCreator.f90 && make ex1 case=gemm_small >/dev/null 2>&1 && make modelCreator.f90 && head -3 modelCreator.f90`
Expected: generated Fortran, no `include file 'variables.fpp' not found`

- [ ] **Step 4: Run the full suite**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!`

- [ ] **Step 5: Commit**

```bash
git clean -fdX
git add -A test/
git commit -m "refactor: build test driver from the library's modelCreator.fpp"
```

---

### Task 17: Makefile defects

**Problem:** In `test/Makefile`: `FFLAGS=-O2` is immediately overwritten by `FFLAGS=-O3` (lines 4-5) — dead line, and the reader cannot tell which was intended; the `capi:` target (lines 33-36) references `capi.c`, which does not exist (the real file is `examples/cAPI.c`); `clean` (line 51) removes `*.o` in `test/` but not the `$(DIR)/%.o` objects the `$(DIR)/%.o` rule writes into `fLibrary/`. In `fLibrary/Makefile`, `clean` (line 22) uses bare `rm`, so it fails the whole target when any of the three patterns is already absent.

**Files:**
- Modify: `test/Makefile`
- Modify: `fLibrary/Makefile`

- [ ] **Step 1: Reproduce the `clean` failure**

Run: `cd fLibrary && make clean; echo "exit=$?"`
Expected: `rm: *.o: No such file or directory` (or similar) and a nonzero exit on a clean tree

- [ ] **Step 2: Fix `test/Makefile`**

- Delete line 4 (`FFLAGS=-O2`), keeping `FFLAGS=-O3`.
- Replace the `capi:` target with one that uses the real example:

```make
capi: $(COMP) modelCreator.o
	gcc -c ../examples/cAPI.c -o cAPI.o
	gfortran -o capi cAPI.o $(COMP) modelCreator.o
	./capi
```

- Replace `clean`:

```make
clean:
	rm -f *.o *.mod output unit_tests capi cAPI.o
	rm -f $(DIR)/*.o
	rm -f modelCreator.f90 userTesting.f90 variables.fpp inputs.fpp
	rm -f test.txt outputCase.txt onnxModel.txt onnxWeights.txt
```

- [ ] **Step 3: Fix `fLibrary/Makefile`**

Replace `clean` (lines 22-25):

```make
clean:
	rm -f *.o objFiles/*.mod
	rm -f modelCreator.f90 variables.fpp
	rm -f libcorelib.a
```

Also delete the dead commented `testing:` block at lines 26-31.

- [ ] **Step 4: Verify both cleans are idempotent**

Run: `cd fLibrary && make clean && make clean; echo "exit=$?"`
Expected: `exit=0` both times

Run: `cd test && make clean && make clean; echo "exit=$?"`
Expected: `exit=0` both times

- [ ] **Step 5: Verify `clean` leaves no stray objects**

Run: `cd test && ./run.sh >/dev/null 2>&1; make clean; git status --short --ignored | grep -c "\.o$"`
Expected: `0`

- [ ] **Step 6: Run the full suite and commit**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!`

```bash
git clean -fdX
git add test/Makefile fLibrary/Makefile
git commit -m "build: fix clean targets, drop dead FFLAGS line, point capi at cAPI.c"
```

---

### Task 18: Module-level scratch state and dead code in `reader.f90`

**Problem:** `reader.f90:17-34` declares `weights`, `midWeights`, `largeWeights`, `biases`, `w_dim1..4`, `layerName`, and `i` at module scope, used as shared scratch by every `read_*` subroutine. They are public module state mutated during parsing — any concurrent or repeated `initialize` call corrupts them. `activation_func` (`:17`) is never used. Dead commented code sits at `:28`, `:268-278` (`reader.f90`) and `:312-350` (`layers.f90`).

**Files:**
- Modify: `fLibrary/reader.f90`
- Modify: `fLibrary/layers.f90:312-350`

- [ ] **Step 1: Confirm the scratch variables are module-scope**

Run: `sed -n 17,34p fLibrary/reader.f90`
Expected: the declarations listed above, outside any subroutine

- [ ] **Step 2: Move scratch state into each subroutine**

For each of `read_reshape2d`, `read_reshape3d`, `read_reshape4d`, `read_add`, `read_avgpool`, `read_maxpool`, `read_conv`, `read_lstm`, `read_linear`, add local declarations for exactly the variables that subroutine uses, e.g. in `read_linear`:

```fortran
    subroutine read_linear(file1, file2)
        INTEGER, INTENT(IN) :: file1
        INTEGER, INTENT(IN) :: file2
        TYPE(linLayer), ALLOCATABLE,DIMENSION(:) :: lin
        REAL (c_double), ALLOCATABLE, DIMENSION(:,:) :: weights
        REAL (c_double), ALLOCATABLE, DIMENSION(:) :: biases
        INTEGER :: w_dim1, w_dim2
```

Then delete the module-level declarations at `:17-34`, keeping only the layer arrays (`:10-16`) and `numLayers`, which `initialize` genuinely shares. Declare `layerName`, `i`, and `readOrNot` locally inside `initialize`.

- [ ] **Step 3: Delete the dead code**

- `reader.f90:28` — the commented `! INTEGER :: activation_func`
- `reader.f90:268-278` — the commented `activation_func` / `fn_ptr` block
- `layers.f90:312-350` — the commented `ad` subroutine and the stray `!hi` comment at `:350`
- `reader.f90` — the `activation_func` declaration itself (verify with `grep -n activation_func fLibrary/reader.f90` that no live code references it)

- [ ] **Step 4: Verify it compiles warning-clean**

Run: `cd fLibrary && make clean >/dev/null 2>&1; mkdir -p objFiles && gfortran -O2 -JobjFiles -Wall -Wextra -c activation_funcs.f90 derived_types.f90 layers.f90 reader.f90 2>&1 | grep -i "unused\|uninitial" | head`
Expected: no output referencing the removed variables

- [ ] **Step 5: Run the full suite and commit**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!`

```bash
git clean -fdX
git add fLibrary/reader.f90 fLibrary/layers.f90
git commit -m "refactor: scope reader scratch state to subroutines, drop dead code"
```

---

### Task 19: Documentation rot

**Problem:** Three separate kinds of rot in `doc/` and `README.md`:

1. **Every doc link 404s.** The repo was renamed `FyeNNa` -> `roseNNa` (confirmed: `gh repo view comp-physics/FyeNNa` follows the redirect and returns `name: roseNNa`), but the rename is the *least* of it — the `develop` branch no longer exists (upstream has only `master`), and the file paths moved into `fLibrary/` and `test/`. Verified 404s: `comp-physics/FyeNNa/tree/develop/goldenFiles`, `comp-physics/roseNNa/blob/develop/derived_types.f90`, `comp-physics/roseNNa/blob/develop/readTester.f90`.
2. **`readTester.f90` no longer exists** — it is `reader.f90`. `doc/opensource.md:60` uses "reader.f90" as the link *text* while pointing at `readTester.f90`; `doc/methodology.md:2` names it in prose as a core source file.
3. **README markup**: a stray `</center>` at `README.md:2` closing a tag that was never opened (the wrapper is `<p align="center">`), and the DOI badge link and image both use `http://` while the adjacent License badge uses `https://` — the DOI badge works only via an insecure redirect hop and is blocked as mixed content by any https renderer outside GitHub's camo proxy.

**Files:**
- Modify: `doc/methodology.md`
- Modify: `doc/opensource.md`
- Modify: `README.md`

- [ ] **Step 1: Confirm the links are dead**

```bash
for u in "https://github.com/comp-physics/FyeNNa/tree/develop/goldenFiles" \
         "https://github.com/comp-physics/roseNNa/blob/develop/derived_types.f90" \
         "https://github.com/comp-physics/roseNNa/blob/develop/readTester.f90"; do
  printf '%-70s ' "$(echo $u | sed 's#https://github.com/##')"
  curl -s -o /dev/null -w '%{http_code}\n' -L "$u"
done
git ls-remote --heads upstream | sed 's#.*refs/heads/##'
```
Expected: three `404` lines, and `master` as the only branch

- [ ] **Step 2: Repoint every doc link**

Apply these replacements across `doc/methodology.md` and `doc/opensource.md` — note that each needs **both** the branch and the path corrected, so a blanket `FyeNNa` -> `roseNNa` substitution is not enough:

| old URL tail | new URL tail |
|---|---|
| `FyeNNa/tree/develop/goldenFiles` | `roseNNa/tree/master/goldenFiles` |
| `FyeNNa/blob/develop/modelParserONNX.py` | `roseNNa/blob/master/fLibrary/modelParserONNX.py` |
| `FyeNNa/blob/develop/modelCreator.fpp` | `roseNNa/blob/master/fLibrary/modelCreator.fpp` |
| `FyeNNa/blob/develop/userTesting.fpp` | `roseNNa/blob/master/test/userTesting.fpp` |
| `FyeNNa/blob/develop/variables.fpp` | *(generated file — replace the link with inline code formatting)* |
| `FyeNNa/blob/develop/onnxModel.txt` | *(generated file — replace the link with inline code formatting)* |
| `FyeNNa/blob/develop/onnxWeights.txt` | *(generated file — replace the link with inline code formatting)* |
| `FyeNNa/blob/develop/run.sh` | `roseNNa/blob/master/test/run.sh` |
| `FyeNNa/blob/develop/goldenFiles/testChecker.py` | `roseNNa/blob/master/test/testChecker.py` |
| `roseNNa/blob/develop/derived_types.f90` | `roseNNa/blob/master/fLibrary/derived_types.f90` |
| `roseNNa/blob/develop/activation_funcs.f90` | `roseNNa/blob/master/fLibrary/activation_funcs.f90` |
| `roseNNa/blob/develop/readTester.f90` | `roseNNa/blob/master/fLibrary/reader.f90` |

- [ ] **Step 3: Fix the prose filename**

In `doc/methodology.md:2`, change `readTester.f90` to `reader.f90` in the list of core compiled files.

- [ ] **Step 4: Fix the README markup**

- `README.md:2`: delete the trailing `</center>`, leaving `  <img src="doc/rosenna.png" alt="roseNNa banner" width="600"/>`
- `README.md:8-9`: change both `http://doi.org/...` and `http://img.shields.io/...` to `https://`

- [ ] **Step 5: Verify every link resolves**

```bash
grep -ohE 'https?://[^)"> ]+' README.md doc/*.md | sort -u | while read u; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -L --max-time 20 "$u")
  [ "$code" = "200" ] || echo "$code  $u"
done
```
Expected: no output (every link returns 200). Investigate anything printed before committing.

- [ ] **Step 6: Verify no stray tags or http badges remain**

Run: `grep -n "</center>\|http://" README.md doc/*.md`
Expected: no output

- [ ] **Step 7: Commit**

```bash
git add README.md doc/
git commit -m "docs: repoint dead links at master, fix readTester/reader, clean markup

Links pointed at the pre-rename FyeNNa name AND a develop branch that no
longer exists AND pre-reorganization paths; all three needed correcting."
```

---

## Phase 5 — Optional

### Task 20: Binary weight files

**Scope warning:** This changes the on-disk format between the parser and the reader. It is a genuine performance win but it is a **feature**, not a defect fix. Do it only after Tasks 0-19 are merged and green, and treat it as its own PR. If time is short, skip it — nothing else in this plan depends on it.

**Problem:** Weights round-trip through ASCII (`modelParserONNX.py` writes `stranspose` output as space-separated decimal; `reader.f90` reads it with list-directed `read`). For a large model this dominates both file size and load time, and decimal formatting loses bits. Combined with the padding copies (Task 15), this is the likely reason `vgg16`, `gemm_huge`, and `turbulentShear` are excluded from `test/run.sh:8`.

**Files:**
- Modify: `fLibrary/modelParserONNX.py`
- Modify: `fLibrary/reader.f90`
- Modify: `test/run.sh` (re-enable a large case)

- [ ] **Step 1: Establish the baseline**

Run: `cd test && make ex1 case=mnist >/dev/null 2>&1 && ls -l onnxWeights.txt && time ./output >/dev/null 2>&1`
Record the file size and wall time.

- [ ] **Step 2: Write weights as a stream**

In `modelParserONNX.py`, open the weights file binary (`open('onnxWeights.bin','wb')`) and replace each `f2.write(stranspose(arr))` with a little-endian float64 dump in the same column-major order:

```python
f2.write(np.asarray(arr, dtype='<f8').flatten(order='F').tobytes())
```

Delete the trailing `f2.write("\n")` at each site — the stream has no record separators.

- [ ] **Step 3: Read it with stream access**

In `reader.f90`, open the weights unit with `access='stream', form='unformatted'` and replace each `read(file2, *) <array>` with `read(file2) <array>`. Fortran stream reads fill in array element order, which is column-major — matching the writer.

- [ ] **Step 4: Keep a compatibility path**

Accept either extension: if the weights path ends in `.txt`, use the old list-directed reads; if `.bin`, use stream. Dispatch on the path in `initialize` and pass a `binary` logical down to each `read_*` subroutine.

- [ ] **Step 5: Verify equivalence**

Run: `cd test && ./run.sh 2>&1 | tail -1`
Expected: `19 out of 19 test cases have passed!` with identical numerical output

- [ ] **Step 6: Measure and re-enable a large case**

Compare file size and load time against Step 1. If the improvement is large enough, remove `gemm_huge` from the skip list in `test/run.sh:8` and confirm it passes.

- [ ] **Step 7: Commit**

```bash
git clean -fdX
git add fLibrary/modelParserONNX.py fLibrary/reader.f90 test/run.sh
git commit -m "perf: stream weights as binary float64 instead of ASCII"
```

---

## Verification Summary

Run at the end of every task:

```bash
cd test && ./run.sh 2>&1 | tail -1
```

| After task | Expected |
|---|---|
| baseline | `1 out of 17 test cases have passed!` |
| 1 | `14 out of 17` (3 maxpool failures are real bugs) |
| 2-3 | `14 out of 17` + unit and parser harnesses pass |
| 4 | `17 out of 17` |
| 5 | `19 out of 19` (adds `maxpool_nonsquare`, `pool_batch`) |
| 6-20 | `19 out of 19`, never lower |

Final acceptance:

```bash
git clean -fdX
cd test && ./run.sh                    # 19 out of 19, exit 0
cd ../fLibrary && make clean && make library   # exit 0
grep -rn "except:" fLibrary/           # no output
grep -n "</center>\|http://" README.md doc/*.md  # no output
git status --short                     # clean
```

---

## Self-Review Notes

**Coverage.** All 23 findings map to tasks: MaxPool attribute index (4), dynamo (1), pooling extents (5), pooling batch (5), uninitialized `time` (7), LSTM gate order and positional streaming (8), `-f`/`-w` swap (8), ignored dilations/ceil_mode/pads/kernel/auto_pad (13), Gemm transB/alpha/beta/transA (10), `fourDTransform` (9), Squeeze and bare excepts (11), `tanhh` overflow (6), C API signature and file I/O (14), padding copies (15), `sumini` (15), identifier sanitization (12), binary weights (20, optional), duplicated `modelCreator.fpp` (16), Makefiles (17), doc links (19), README markup (19), reader globals and dead code (18), CI (3).

**Known risk.** Tasks 10, 11, and 13 convert silent-wrong-answers into hard errors. A user whose model currently "works" (with wrong numbers) will get a `NotImplementedError` after this. That is the correct trade for a numerical library, but it is a breaking change and belongs in release notes — flag it when cutting the next version.

**Ordering constraint.** Task 3 (CI) intentionally leaves CI red until Task 4 lands. Do not merge Task 3 on its own.
