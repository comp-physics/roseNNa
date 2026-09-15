<p align="center">
  <img src="doc/rosenna.png" alt="roseNNa banner" width="600"/>
</p>
<p align="center"> 
<a href="https://github.com/comp-physics/roseNNa/actions/workflows/CI.yml">
  <img src="https://github.com/comp-physics/roseNNa/actions/workflows/CI.yml/badge.svg" />
</a>
<a href="https://lbesson.mit-license.org/">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" />
</a>
<a href="https://doi.org/10.1016/j.cpc.2023.109052">
  <img src="https://img.shields.io/badge/DOI-10.1016/j.cpc.2023.109052-B31B1B.svg" />
</a>
</p>

RoseNNa is a fast, portable, and minimally-intrusive library for neural network inference.
It reads a neural network in [ONNX](https://onnx.ai/) format -- the format PyTorch, TensorFlow and Keras all export -- and **generates** a small, self-contained Fortran module and C library that computes it.
__RoseNNa's intended use case is embedding neural networks in Fortran- and C-based HPC codebases.__
You link the generated code into an existing PDE (e.g. CFD) solver and call it per point, on the CPU or inside your own GPU offload loop.

RoseNNa supports MLPs, CNNs and RNNs.
Because the generated code has literal loop bounds, no runtime shape logic, no allocation and no mutable global state, it inlines into a solver's own compute kernel -- including a device kernel.
RoseNNa is described in <a href="https://arxiv.org/abs/2307.16322">A. Bati, S. H. Bryngelson (2024) Comp. Phys. Comm., 296, 109052.</a>, which describes the earlier runtime-parsing library; the generator replaced it (see [History](#history)).

## Hello RoseNNa

```sh
pip install -e python
rosenna generate model.onnx --lang both --out build/
```

That writes `model_model.F90` and `model.c`/`model.h` (plus build recipes) into `build/`. Then, in Fortran:

``` fortran
program hello_roseNNa
  use model_model
  implicit none
  real(real64) :: input(784), output(10)
  integer :: status

  call model_init("model.rwt", status)   ! only for a file-loaded model
  call model_infer(input, output)        ! run inference
end program
```

or in C:

```c
#include "model.h"

int main(void) {
    double input[784], output[10];
    if (model_init("model.rwt") != 0) return 1;   /* file-loaded models only */
    model_infer(input, output);
}
```

A model under a million parameters embeds its weights into the generated source by default, and then has no `init` to call at all.
`model_infer` is `pure` in Fortran, takes `restrict` pointers in C, does no I/O and allocates nothing, so it is safe to call from inside an OpenMP-target, OpenACC, CUDA or HIP loop.

## Supported ONNX operators and limits

roseNNa generates code for: `Gemm`, `MatMul`, `Conv` (including grouped and depthwise), `MaxPool`, `AveragePool`, `LSTM`, `Add`, `Concat`, `Pad`,
`Reshape`, `Transpose`, `Squeeze`, `Unsqueeze`, `Flatten`, `Identity`, `Relu`, `Sigmoid`, `Tanh`, `Softmax`.
An inference `BatchNormalization` is folded into the `Conv` or `Gemm` that feeds it, so it costs nothing at runtime.

Everything statically knowable is resolved at generation time: shapes, buffer sizes, padding (including `auto_pad`), and every node whose inputs are all constants -- so a `Reshape` of a weight, or an int64 shape tensor, never reaches the emitted code.

A model using something the generator cannot lower is **refused by name at generation time**, never silently mis-computed. `rosenna info model.onnx` reports what it found. The limits:

- 2-D spatial ops only (rank-4 NCHW); `ceil_mode` must be 0
- `Conv` `group` must divide both channel counts, and the weight's channel axis must be `C_in / group`
- `Softmax` normalises the last axis only; `Pad` is constant-mode with constant pads (negative pads, i.e. crops, are fine)
- a `BatchNormalization` that cannot be folded (training mode, non-constant parameters, or an intermediate read elsewhere) is refused
- `Gemm` `alpha` and `beta` must be 1, `transA` must be 0, and weights must be constant
- `LSTM` must be forward-direction with the default activations, no `clip`, `input_forget`, `sequence_lens` or peepholes
- several inputs and several outputs are fine; they arrive concatenated in `x` and leave concatenated in `y` (see below)
- every weight must be a constant initializer, not computed at runtime

## Verify it

```sh
rosenna verify model.onnx --cases 32
```

compiles both backends and compares them against onnxruntime on random inputs. Every model in `goldenFiles/` is checked this way, on both backends, by `python/tests/test_golden_suite.py`.

## Several inputs

A model with more than one graph input -- an LSTM's initial hidden and cell state, say -- takes them **concatenated in declaration order** in the single `x` buffer, and a model with more than one graph output -- that LSTM's `Y`, `Y_h` and `Y_c` -- writes them concatenated the same way in `y`. That keeps one entry point, one input buffer, one output buffer, and so one device contract, for every model; `rosenna info` prints where each tensor sits. A solver that keeps a recurrent model's state per cell feeds `y`'s state slices straight back into `x` next step, on the device.

## GPU use

The generated code is callable from a device loop, and `rosenna gpu-gate` validates that end to end on real hardware. See [python/README.md](python/README.md) for the full story: the batched entry point, the CUDA/HIP kernel, the build recipes, and the measured per-point cost.

## Examples: surrogates inside PDE solvers

[examples/surrogates/](examples/surrogates/) has four self-contained solvers, each in C and Fortran, with a network called inside the time-step loop -- a per-cell closure (coarse-grid Burgers), a batched learned time-stepper (reaction-diffusion), a recurrent per-cell model with resident state (bubbly acoustics), and a whole-field initial guess (Poisson). They are organised by where the network sits and what code structure that forces; `make TOOLCHAIN=amd|nvidia|gnu` in any of them generates, builds and runs.

## Further documentation

- [python/README.md](python/README.md) -- install, generate, build, and call from C or Fortran
- [doc/methodology.md](doc/methodology.md) -- the roseNNa pipeline
- [doc/opensource.md](doc/opensource.md) -- extending roseNNa to new operators

## History

roseNNa began as `fLibrary/`: a Fortran library that parsed a model description at
startup and walked it at runtime. The generator in `python/` replaced it once it
covered every operator the library did and every model in `goldenFiles/`, which it
now verifies against onnxruntime on both backends rather than against recorded
output. The library, its `modelParserONNX.py`, and the shell suite that drove it
were removed at that point; they remain in the git history.

## Citation

You can cite this work as 
```bibtex
@article{bati24,
  author = {Bati, A. and Bryngelson, S. H.},
  title = {{RoseNNa: A} performant, portable library for neural network inference with application to computational fluid dynamics},
  journal = {Computer Physics Communications},
  volume = {296},
  pages = {109052},
  year = {2024},
  doi = {10.1016/j.cpc.2023.109052},
}
```
