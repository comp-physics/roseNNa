<p align="center">
  <img src="doc/rosenna.png" alt="roseNNa banner" width="600"/></center>
</p>
<p align="center"> <a href="https://zenodo.org/badge/latestdoi/466203469">
  <img src="https://zenodo.org/badge/466203469.svg" alt="DOI">
</a>
<a href="https://github.com/comp-physics/roseNNa/actions">
  <img src="https://github.com/comp-physics/roseNNa/actions/workflows/CI.yml/badge.svg" />
</a>
<a href="https://lbesson.mit-license.org/">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" />
</a>
</p>

RoseNNa is a fast, portable, and minimally-intrusive library for neural network inference.
It can run inference on neural networks in [ONNX](https://onnx.ai/) format, which is universal and can be used with PyTorch, TensorFlow, Keras, and more.
RoseNNa's intended use case is large Fortran- and C-based HPC codebases. 
It currently supports RNNs, CNNs, and MLPs, though more architectures are in the works.
The library is optimized Fortran and outperforms PyTorch (by a factor between 2 and 5x) for the relatively small neural networks used in physics applications, like CFD.
It is described in detail in the following manuscript: <a href="http://arxiv.org/abs/2307.16322">A. Bati, S. H. Bryngelson (2023) arXiv 2307.16322</a>.

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

## Dependencies

We have minimal dependencies. 
For example, on MacOS you can get away with just
```
brew install wget make cmake coreutils gcc
pip install torch onnx numpy fypp onnxruntime pandas
```
## Basic Example
Here is a quick example of how **roseNNa** works. With just a few steps, you can see how to convert a basic feed-forward neural network originally built with PyTorch into usable, accurate code in Fortran.

First `cd` into the `fLibrary/` directory.

``` bash
#file to create pytorch model and convert to ONNX
python3 ../goldenFiles/gemm_small/gemm_small.py
```
``` bash
#read and interpret the correspoding output files from last step
python3 modelParserONNX.py -w ../goldenFiles/gemm_small/gemm_small.onnx -f ../goldenFiles/gemm_small/gemm_small_weights.onnx
```
``` bash
#compile the library
make library
```
``` bash
#compile "source files" (capiTester.f90), link to the library file created, and run
gfortran -c ../examples/capiTester.f90 -IobjFiles/
gfortran -o flibrary libcorelib.a capiTester.o
./flibrary
```
``` bash
#check whether python output from PyTorch model = roseNNa's output
python3 ../test/testChecker.py gemm_small
```

## Compiling roseNNa 

1. **Save the neural network model that needs to be converted**

    Make sure to refer to the specific library's documentation about how to save the model.

2. **Convert the saved model to an ONNX format**

    Details on how to convert a saved model to ONNX format can be found on their [website](https://onnx.ai/supported-tools.html#buildModel). 


    **Converting an LSTM?**

    One important thing to note is sometimes ONNX enables optimizations that will change how the weights are stored internally (this will happen specifically for LSTMs). When converting from any library to ONNX, one should load 2 files: one with optimization and one without. This may or may not apply to all library to ONNX conversions, but here is an example using pytorch (one with `do_constant_folding=True` and another with `do_constant_folding=False`.

```python
#MODEL STRUCTURE FILE
torch.onnx.export(model,               # model being run
                  (inp, hidden),                         # model input (or a tuple for multiple inputs)
                  filePath+"lstm_gemm.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input', 'hidden_state','cell_state'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

#MODEL WEIGHTS FILE
torch.onnx.export(model,               # model being run
                  (inp, hidden),                         # model input (or a tuple for multiple inputs)
                  filePath+"lstm_gemm_weights.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input', 'hidden_state','cell_state'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
```

3. **Preprocess the model**

    `fLibrary/` holds the library files that recreate the model and run inference on it. Run `python3 modelParserONNX.py -f path/to/model/structure -w path/to/weights/file` to reconstruct the model.

4. **Compiling the library**

    Then, in the same `/fLibrary` directory, run `make library`. This compiles the library into `libcorelib.a`, which is required to link other `*.o` files with the library. This library file is now ready to be integrated into any Fortran/C workflow.

## Fortran usage 

One can compile a Fortran example (like the `Hello RoseNNa` example above) by specifying the location of the module files and linking the library to other program files.
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

Please see [this document](https://github.com/comp-physics/roseNNa/blob/master/doc/opensource.md) on how to extend roseNNa to new network models and [this document](https://github.com/comp-physics/roseNNa/blob/master/doc/methodology.md) on the details of the roseNNa pipeline .
