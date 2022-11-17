<p align="center">
  <img src="doc/rosenna.png" alt="roseNNa banner" width="600"/></center>
</p>
<p align="center">
<a href="https://github.com/comp-physics/roseNNa/actions">
  <img src="https://github.com/comp-physics/roseNNa/actions/workflows/CI.yml/badge.svg" />
</a>
<a href="https://lbesson.mit-license.org/">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" />
</a>
</p>

roseNNA is a fast, portable, and minimally-intrusive library for neural network inference.
Its intended use case is large Fortran- and C-based HPC codebases. 
roseNNa currently supports RNNs, CNNs, and MLPs, though more architectures are in the works.
The library is optimized Fortran and outperforms PyTorch (by a factor between 2 and 5x) for the relatively small neural networks used in physics applications, like CFD.

## **Fortran Library**
The fLibrary folder holds all the core files that are needed to recreate the model in Fortran and be linked to a program. It contains a Makefile that compiles all core files and creates a library.

Here are the steps one needs to follow. First preprocess the model down below. This encodes the models: it writes the weights and architecture to text files (onnxModel.txt and onnxWeights.txt) and stores information about the model in an external fpp file (variable.fpp).

```make
    preprocess: modelParserONNX.py
        # arg1 = model structure file (.onnx format)
        # arg2 (optional) = weights file (.onnx format)
        python3 modelParserONNX.py -f $(args)

        #for *.mod and *.o files
        mkdir -p objFiles
```
Then, run "make library" to compile all the core files and create a library called "libcorelib.a". This file must be used to link any other "*.o" files in the program with the library.

Here is an example test file:

``` fortran
program name

    USE rosenna
    implicit none
    REAL (c_double), DIMENSION(1,2) :: inputs
    REAL (c_double), DIMENSION(    1, 3) :: output

    inputs = RESHAPE(    (/1.0, 1.0/),    (/1, 2/), order =     [2 , 1 ])

    CALL initialize()

    CALL use_model(inputs, output)

    print *, output

end program name
```
Compile the files and specify the location to the module files. Lastly, link the library to any other files in the program:

``` shell
gfortran -c *.f90 -Ipath/to/objFiles
gfortran -o flibrary path/to/libcorelib.a *.o
./flibrary
```


## **C Library**
The C library also uses the library created from the previous section (so make sure to read the previous section to create the library file). C and Fortran are interoperable. To call Fortran **from** C, here is the code in C:

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
The two functions that will be used includes **use_model** and **initialize** (same procedure as Fortran). Therefore, the function headers must be defined in C. Then, based on the model encoded, instantiate an input and outupt with the correct dimension. **Call initialize** to allow fortran to read in the weights and **call use_model** which will write the output of the model into **out**.

For compilation follow these steps:
``` shell
gcc -c *.c
gfortran -o capi path/to/libcorelib.a *.o
./capi
```

## **User Example**

``` fortran
    program name

        !must be imported
        USE rosenna
        implicit none

        !user has to provide inputs to the model
        REAL, DIMENSION(1,1,28,28) :: inputs
        REAL, DIMENSION(1,5) :: Plus214_Output_0

        !this must be called somewhere to read all the weights in
        CALL initialize()

        ! this must be called to run inference on the model
        CALL use_model(inputs, Plus214_Output_0)

    end program name
```
This represents a sample program that can be linked with the library created above and run succesfully (given the model's inputs match the inputs provided). Four things are required to use this library: **USE rosenna**, **initializing inputs**, **CALL initialize()**, and **CALL use_model(args)**.

## **Open Source Development**
[Open Source](https://github.com/comp-physics/roseNNa/blob/develop/instructions/opensource.md)

## **roseNNa Pipeline**
[Pipeline Documentation](https://github.com/comp-physics/roseNNa/blob/develop/instructions/methodology.md)
