# FyeNNa
A neural network model to Fortran and C translator (FyeNNa and CyeNNa?)

To run current tests located in [goldenFiles](https://github.com/comp-physics/FyeNNa/tree/develop/goldenFiles), change permissions for [run.sh](https://github.com/comp-physics/FyeNNa/blob/develop/run.sh). Each time the tests are run, new weights are initialized for the given test's model. To look at the model architectures of each test, go to the same **goldenFiles** folder, view each test's folder, and go to the .py file.

## Methodology
First, all the core files are compiled (activation_funcs.f90, derived_types.f90, layers.f90, readTester.f90). activation_funcs.f90 stores activation functions, derived_types.f90 stores derived types for certain layer types, layers.f90 stores the math behind certain layers (**currently we support GEMM, LSTM, Convolutional, and MaxPool layers**), and readTester.f90 loads in the weights that are stored in the system itself.

### Initialization and Preprocessing
Then, in each of the test case files in [goldenFiles](https://github.com/comp-physics/FyeNNa/tree/develop/goldenFiles), the **.py** file is run to create the model, randomly initialized with weights. It creates an intermediary file called inputs.fpp, which stores the exact inputs given to the model, which is later fed to the fortran built model. It also creates a "golden file" which represents the correct shape and output of the model. Lastly, the model that was run is stored in **.onnx** format.

[modelParserONNX.py](https://github.com/comp-physics/FyeNNa/blob/develop/modelParserONNX.py) is run to parse the onnx model and gathers information about the model and creates [onnxModel.txt](https://github.com/comp-physics/FyeNNa/blob/develop/onnxModel.txt) (layer names and weights dimensions) and [onnxWeights.txt](https://github.com/comp-physics/FyeNNa/blob/develop/onnxWeights.txt) (the corresponding weights for each layer). It also creates a [variables.fpp](https://github.com/comp-physics/FyeNNa/blob/develop/variables.fpp) file that stores some key information about the model that fypp will process during model creation.

### Running and Testing
Lastly, we have two **.fpp** files. [modelCreator.fpp](https://github.com/comp-physics/FyeNNa/blob/develop/modelCreator.fpp) is the module that builds the subroutine that stores the correct model architecture. It parses through [variables.fpp](https://github.com/comp-physics/FyeNNa/blob/develop/variables.fpp) and reconstructs the model with the subroutines in **layers.f90**. [userTesting.fpp](https://github.com/comp-physics/FyeNNa/blob/develop/userTesting.fpp) is used to create **userTesting.f90**, a sample file that calls "**initialize**" (which enables fortran to read in the weights and model structure from [onnxModel.txt](https://github.com/comp-physics/FyeNNa/blob/develop/onnxModel.txt) and [onnxWeights.txt](https://github.com/comp-physics/FyeNNa/blob/develop/onnxWeights.txt). Then it passes in the inputs from the intermediary file inputs.fpp, and runs the model. [userTesting.fpp](https://github.com/comp-physics/FyeNNa/blob/develop/userTesting.fpp) then stores the shape and output in a text file.


[testChecker.py](https://github.com/comp-physics/FyeNNa/blob/develop/goldenFiles/testChecker.py) compares the outputted text file to the test's "golden file". If the shapes match and the outputs are within reasonable range, the test case passes. Otherwise, the error is outputted.


## Fortran Library
The fLibrary folder holds all the core files that are needed to recreate the model in Fortran and be linked to a program. It contains a Makefile that compiles all core files and creates a library. 

Here are the steps one needs to follow. First preprocess the model down below. This customizes the modelCreator.f90 file to the model that is currently being used/preprocessed.

```make
    preprocess: modelParserONNX.py
        # arg1 = model structure file (.onnx format)
        # arg2 (optional) = weights file (.onnx format)
        python3 modelParserONNX.py $(arg1) $(arg2)

        #create the model based on modelParserONNX
        fypp modelCreator.fpp modelCreator.f90
```
Then, run "make library" to compile all the core files and create a library called "libcorelib.a". This file must be used to link any other "*.o" files in the program with the library. 

### User Example

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

        print *, Plus214_Output_0
    end program name
```
This represents a sample program that can be linked with the library created above and run succesfully (given the model's inputs match the inputs provided). Four things are required to use this library: **USE rosenna**, **initializing inputs**, **CALL initialize()**, and **CALL use_model(args)**.

## C Library
Need to Update


# Open Source Development
This project is ongoing and does not contain functionality of every layer available in ONNX. In order to embed new layers into roseNNa, certain steps must be followed:

### Parsing in modelParserONNX.py
This file reads in the ONNX interpretation of the model. At a higher level, it iterattes over all the layers in the ONNX model (called nodes in the graph), parses its contents by (1) sending some of its options to be parsed in f90 via fypp and (2) finding the weights that correspond to this layer and writing their dimensions to 'onnxModel.txt' and the weights to 'onnxWeights.txt'. These two files will be read in by Fortran so it can store the weights and layers. Here is a pseudocode example from the "GEMM" layer in ONNX:

```python
#an additional if statement must be added so the parser knows to parse this layer
elif layer == "Gemm":
    #modelArch stores the layer and options for layer (fypp input later on)
    #ioMap is referenced to get the output name from the last layer (which is input to this layer)
    modelArch.append(("Gemm", [ioMap[node.input[0]],node.attribute[2].i], None)) 
    
    #parsing the inputs to the layer
    for inp in node.input[1:3]:
        
        #writing the dimension to 'onnxModel.txt'
        for dim in initializer[inp]:
            f.write(str(dim)+ " ")
        
        #this is mainly for LSTM layers, but when externalWeightsFile is provided, all weights are stored in order
        if externalWeightsFile:
            f2.write(stranspose(numpy_helper.to_array(true_weights[true_index])))
            true_index+=1
        #when it is not provided, we just store weights in a hashmap and call this function to retrieve it for us (since it is not in order)
        #stranspose is to convert the weights into column major order (for F90)
        else:
            f2.write(stranspose(findWeightsInitializer(inp)))
        f2.write("\n")
        f.write("\n")
    #at the end, we have to make sure that the names for the inputs are preseved. The output name (e.g. "out1") will be the input to the next layer, so we will be using "out1" as the input
    #this must be stored in some kind of map
    ioMap[node.output[0]] = ioMap[node.input[0]]
```
### Adding Layer
Most layers come with a set of parameters that are commonly manipulated (number of layers, activation functions, hidden state, etc.). This information can be integrated by creating a derived type of the layer in [derived_types.f90](https://github.com/comp-physics/roseNNa/blob/develop/derived_types.f90). Here is an example:

``` fortran
TYPE lstmLayer
    REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: whh
    REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: wih
    REAL (c_double), ALLOCATABLE, DIMENSION(:) :: bhh
    REAL (c_double), ALLOCATABLE, DIMENSION(:) :: bih
ENDTYPE lstmLayer
```
The LSTM layer requires 4 types of hidden weights that are used while running through the layer. They are stored within the derived type. The layer dimensions cannot be changed later on.

### Adding activation function
For any activation functions that need to be added will go in [activation_funcs.f90](https://github.com/comp-physics/roseNNa/blob/develop/activation_funcs.f90). To do so, just a function needs to be created. Here is an example:

``` fortran
FUNCTION tanhh(x) result(output)
    REAL (c_double), intent(in) :: x(:)
    REAL (c_double) :: output(size(x))
    output = (exp(x)-exp(-1*x))/(exp(x)+exp(-1*x))
END FUNCTION tanhh
```

### Reading layer in reader.f90
In order to read in the weights and layers from the files 'onnxModel.txt' and 'onnxWeights.txt', the file [reader.f90](https://github.com/comp-physics/roseNNa/blob/develop/readTester.f90) has to include the new layer/activation function. First, we will create an array of derived types for the new layer. This will allow us to store multiple of the same layer if the model contains it (we make it allocatable so it can be appended to with no dimension restrictions). Then, we create a new subroutine for the layer, which defines how we will read in the weights/dimensions (this will depend based on how you wrote the dimensions to the files in the first place). Here is an example:

``` fortran
#subroutine definition for GEMM/MLP layer (file1=dimensions, file2=weights)
subroutine read_linear(file1, file2)
    INTEGER, INTENT(IN) :: file1
    INTEGER, INTENT(IN) :: file2
    
    !create temporary derived type for this one layer
    TYPE(linLayer), ALLOCATABLE,DIMENSION(:) :: lin
    
    !read in dimensions from file1 and allocate weights to store the incoming weights
    ALLOCATE(lin(1))
    read(file1, *) w_dim1, w_dim2
    ALLOCATE(weights(w_dim1,w_dim2))
    
    !read in the weights
    read(file2, *) weights
    
    !repeat for biases
    read(file1, *) w_dim1
    ALLOCATE(biases(w_dim1))
    read(file2, *) biases

    !then assign the temporary layer its weights
    lin(1)%weights = weights
    lin(1)%biases = biases

    DEALLOCATE(weights)
    DEALLOCATE(biases)
    
    !append the temporary layer to the list of layers
    linLayers = [linLayers, lin]
    DEALLOCATE(lin)
end subroutine
```

### Fypp to call the layer/activation function
