<p align="center">
  <img src="doc/rosenna.png" alt="roseNNa banner" width="600"/></center>
</p>
<p align="center">
<a href="https://zenodo.org/badge/latestdoi/466203469">
  <img src="https://zenodo.org/badge/466203469.svg" alt="DOI">
</a>
<a href="https://github.com/comp-physics/roseNNa/actions">
  <img src="https://github.com/comp-physics/roseNNa/actions/workflows/CI.yml/badge.svg" />
</a>
<a href="https://lbesson.mit-license.org/">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" />
</a>
</p>

roseNNA is a fast, portable, and minimally-intrusive library for neural network inference.
It can run inference on neural networks in [ONNX](https://onnx.ai/) format, which is universal and can be used with PyTorch, TensorFlow, Keras, and more.
roseNNa's intended use case is large Fortran- and C-based HPC codebases. 
It currently supports RNNs, CNNs, and MLPs, though more architectures are in the works.
The library is optimized Fortran and outperforms PyTorch (by a factor between 2 and 5x) for the relatively small neural networks used in physics applications, like CFD.

## Hello World

``` fortran
program hello_world

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

## Dependencies

We have minimal dependencies. 
For example, on MacOS you can get away with just
```
brew install wget make cmake coreutils gcc
pip install torch onnx numpy fypp onnxruntime
```

## Compiling roseNNa 

`fLibrary/` holds the library files that recreate the model and run inference on it.
It has a `Makefile` that first pre-processes the model:
```make
    preprocess: modelParserONNX.py
        # arg1 = model structure file (.onnx format)
        # arg2 (optional) = weights file (.onnx format)
        python3 modelParserONNX.py -f $(args)

        #for *.mod and *.o files
        mkdir -p objFiles
```
This encodes the models, writing the weights and architecture to text files called `onnxModel.txt` and `onnxWeights.txt`.
Information about the model is also included in a library helper module `variable.fpp`.

`make library` compiless the library into `libcorelib.a`, which is required to link other `*.o` files with the library.

## Fortran usage 

One can compile a Fortran example (like the `Hello World` exmaple above) by specifying the location of the module files and linking the library to other program files.
In practice, this looks like
``` shell
gfortran -c *.f90 -Ipath/to/objFiles
gfortran -o flibrary path/to/libcorelib.a *.o
./flibrary
```

## C usage

One can call roseNNA from C painlessly. 
Compile the library, then use the the following C program as an example:
``` c
void use_model(double * i0, double * o0);
void initialize(char * model_file, char * weights_file);

int main(void) {

    double input[1][2] = {1,1};
    double out[1][3];
    initialize("onnxModel.txt","onnxWeights.txt");
    use_model(input, out);

    for (int i = 0; i < 3; i++) {
        printf("%f ",b[0][i]);
    }
}
```
and compile it as
``` shell
gcc -c *.c
gfortran -o capi path/to/libcorelib.a *.o
./capi
```

## Further documentation

Please see [this document](https://github.com/comp-physics/roseNNa/blob/master/doc/opensource.md) on how to extend roseNNa to new network models and [this document](https://github.com/comp-physics/roseNNa/blob/master/doc/methodology.md) on the details of the roseNNa pipeline.
