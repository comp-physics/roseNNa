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
It can run inference on neural networks in [ONNX](https://onnx.ai/) format, which is universal and can be used with PyTorch, TensorFlow, Keras, and more.
__RoseNNa's intended use case is embedding neural networks in Fortran- and C-based HPC codebases.__
One compiles RoseNNa and links it to an existing PDE (e.g., CFD) solver written in C or Fortran.
You can then evaluate your neural network from the PDE solver at Fortran/C speeds.

RoseNNa currently supports RNNs, CNNs, and MLPs.
The library is optimized Fortran and outperforms PyTorch (by a factor between 2 and 5x) for the relatively small neural networks used in physics applications, like computational fluid dynamics.
RoseNNa is described in detail in <a href="https://arxiv.org/abs/2307.16322">A. Bati, S. H. Bryngelson (2024) Comp. Phys. Comm., 296, 109052.</a>.

## Hello RoseNNa

``` fortran
program hello_roseNNa

  use rosenna
  implicit none

  real, dimension(1,1,28,28) :: input ! model inputs
  real, dimension(1,5) :: output      ! model outputs

  call initialize() ! reads weights
  call use_model(input, output) ! run inference

end program
```

This example program links to the roseNNa library, parses the model inputs, and runs inference on the loaded library. 
Only a few lines are required to use the library: `use rosenna`, `call initialize()`, and `call use_model(args)`.

With no arguments, `initialize` reads `onnxModel.txt` and `onnxWeights.bin` from the working directory.
If `onnxWeights.bin` does not exist, it reads a legacy `onnxWeights.txt` instead and prints a notice to standard error; it never does this when a weights path is passed explicitly.
To read the files from elsewhere, pass the paths.
`initialize` is a `bind(c)` procedure, so a Fortran caller must terminate each path with `c_null_char`:
``` fortran
use iso_c_binding
call initialize("path/onnxModel.txt"//c_null_char, "path/onnxWeights.bin"//c_null_char)
```

## Dependencies

We have minimal dependencies. 
For example, on MacOS you can get away with just
```
brew install wget make cmake coreutils gcc
pip install torch onnx numpy fypp onnxruntime pandas
```
## Basic Example
Here is a quick example of how **roseNNa** works. With just a few steps, you can see how to convert a basic feed-forward neural network originally built with PyTorch into usable, accurate code in Fortran.

First, `cd` into the `fLibrary/` directory.

Then, create PyTorch model and convert to ONNX:
``` bash
python ../goldenFiles/gemm_small/gemm_small.py
```

Read and interpret the corresponding output files from the last step via
``` bash
python modelParserONNX.py -f ../goldenFiles/gemm_small/gemm_small.onnx
```
and compile the library
``` bash
make library
```

Compile the "source files" (`capiTester.f90`) and link to the library file created:
``` bash
gfortran -c ../examples/capiTester.f90 -IobjFiles/
gfortran -o flibrary capiTester.o libcorelib.a
./flibrary
```
and finally check if the output from PyTorch model matches roseNNa's output
``` bash
python ../test/testChecker.py gemm_small
```

## Compiling roseNNa 

1. **Save the neural network model that needs to be converted**

    Make sure to refer to the specific library's documentation about how to save the model.

2. **Convert the saved model to an ONNX format**

    Details on converting a saved model to ONNX format can be found on their [website](https://onnx.ai/supported-tools.html#buildModel). 


    **Converting an LSTM?**

    ONNX's constant folding renames an LSTM's weight initializers and stores the
    four gates in ONNX's `iofc` order, while roseNNa's `lstm_cell` consumes
    PyTorch's `ifgo` order. The parser now remaps the gates internally and looks
    every weight up by name, so a single `do_constant_folding=True` export is all
    that is needed. Earlier versions required a second, unoptimized
    (`do_constant_folding=False`) export passed via `-w`; that flag is now
    accepted but ignored.

```python
torch.onnx.export(model,               # model being run
                  (inp, hidden),                         # model input (or a tuple for multiple inputs)
                  filePath+"lstm_gemm.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input', 'hidden_state','cell_state'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
```

3. **Preprocess the model**

`fLibrary/` holds the library files that recreate and run inference on the model. Run `python modelParserONNX.py -f path/to/model.onnx` to reconstruct the model.

4. **Compiling the library**

Then, in the same `/fLibrary` directory, run `make library`. This compiles the library into `libcorelib.a`, which is required to link other `*.o` files with the library. This library file is now ready to be integrated into any Fortran/C workflow.

## Supported ONNX operators and limits

roseNNa supports the following ONNX operators: `Gemm`, `MatMul`, `Conv`, `MaxPool`, `AveragePool`, `LSTM`, `Add`,
`Reshape`, `Transpose`, `Squeeze`, `Relu`, `Sigmoid`, `Tanh`.

The parser rejects a model with `NotImplementedError` rather than silently producing a wrong answer when it
encounters an attribute it cannot honour. The limits it enforces:

- `kernel_shape` is required for `MaxPool` and `AveragePool` (inferred from the weights for `Conv`)
- `dilations` must be 1
- `ceil_mode` must be 0
- kernels must be square
- pads must be symmetric per axis
- `Conv` `group` must be 1 (no grouped or depthwise convolution)
- `AveragePool` with nonzero pads requires `count_include_pad=1`
- `AveragePool` `auto_pad` must be `NOTSET` or `VALID`
- a `Pad` node must have all-zero pads
- `Gemm` `alpha` and `beta` must be 1, and `transA` must be 0

## Fortran use

One can compile a Fortran example (like the `Hello RoseNNa` example above) by specifying the location of the module files and linking the library to other program files.
In practice, this looks like
``` shell
gfortran -c *.f90 -Ipath/to/objFiles
gfortran -o flibrary *.o path/to/libcorelib.a
./flibrary
```

**Memory layout.** `use_model` expects inputs in Fortran (column-major) order. A C caller with a row-major array must transpose it first; a Fortran caller building an array from a row-major literal should use `RESHAPE(..., order=[2,1])`, as `examples/capiTester.f90` does.

## C use

One can readily call roseNNa from C. 
Compile roseNNa, then use the following C program as an example:
```c
#include <stdio.h>

void use_model(double * i0, double * o0);
void initialize(const char * model_file, const char * weights_file);

int main(void) {

    /* roseNNa expects column-major (Fortran) ordering. */
    double a[2] = {1, 1};
    double b[3];

    initialize("onnxModel.txt", "onnxWeights.bin");
    use_model(a, b);

    for (int i = 0; i < 3; i++) {
        printf("%f ", b[i]);
    }
    printf("\n");
    return 0;
}
```
and compile it as
```shell
gcc -c *.c
gfortran -o capi *.o path/to/libcorelib.a
./capi
```

A weights path ending in `.txt` (in any letter case, trailing blanks ignored) is read as the legacy text format;
any other path is read as little-endian float64 binary, which must match the model exactly, or `initialize`
stops with an error. The `onnxWeights.txt` fallback described under Hello RoseNNa is read as text.

## Further documentation

Please see [this document](https://github.com/comp-physics/roseNNa/blob/master/doc/opensource.md) on how to extend roseNNa to new network models and [this document](https://github.com/comp-physics/roseNNa/blob/master/doc/methodology.md) on the details of the roseNNa pipeline.

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
