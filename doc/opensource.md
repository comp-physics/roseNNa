# Open Source Development
This project is ongoing and does not contain functionality of every layer available in ONNX. In order to embed new layers into roseNNa, certain steps must be followed:

## Parsing in modelParserONNX.py
This file reads in the ONNX interpretation of the model. At a higher level, it iterattes over all the layers in the ONNX model (called nodes in the graph), parses its contents by (1) sending some of its options to be parsed in f90 via fypp and (2) finding the weights that correspond to this layer and writing their dimensions to 'onnxModel.txt' and the weights to `onnxWeights.txt`. These two files will be read in by Fortran so it can store the weights and layers. Here is a pseudocode example from the "GEMM" layer in ONNX:

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
## Adding Layer
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

## Adding activation function
For any activation functions that need to be added will go in [activation_funcs.f90](https://github.com/comp-physics/roseNNa/blob/develop/activation_funcs.f90). To do so, just a function needs to be created. Here is an example:

``` fortran
FUNCTION tanhh(x) result(output)
    REAL (c_double), intent(in) :: x(:)
    REAL (c_double) :: output(size(x))
    output = (exp(x)-exp(-1*x))/(exp(x)+exp(-1*x))
END FUNCTION tanhh
```

## Reading layer in reader.f90
In order to read in the weights and layers from the files `onnxModel.txt` and `onnxWeights.txt`, the file [reader.f90](https://github.com/comp-physics/roseNNa/blob/develop/readTester.f90) has to include the new layer/activation function. First, we will create an array of derived types for the new layer. This will allow us to store multiple of the same layer if the model contains it (we make it allocatable so it can be appended to with no dimension restrictions). Then, we create a new subroutine for the layer, which defines how we will read in the weights/dimensions (this will depend based on how you wrote the dimensions to the files in the first place). Here is an example:

``` fortran
!subroutine definition for GEMM/MLP layer (file1=dimensions, file2=weights)
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

## Fypp to call the layer/activation function
After encoding the layer/activation function and reading it, fypp will construct the model. Fypp takes in the model architecture, inputs, outputs, and shapes, all of which have been written to an external fypp file. In [modelCreator.fpp](https://github.com/comp-physics/roseNNa/blob/master/modelCreator.fpp), there is a condition for each of the layers that need to be added. Here is an example for the multilayer perceptron layer (GEMM):

``` fortran
#: if tup[0] == 'Gemm'
    !========Gemm Layer============
    CALL linear_layer(${tup[1][0]}$, linLayers(${layer_dict[tup[0]]}$),${1-tup[1][1]}$)
```
In this example, we call the `linear_layer` implemented in `layers.f90` and pass in arguments that come from the external fypp files. There is a for loop running through each layer in the model architecture (a list of tuples), and `tup` contains certain arguments that enables the tool to call the correct names and arguments. `linLayers` is defined in the reader file and stores information about the **i**th layer. One thing to make sure is to store the correct information in model architecture so it can be referenced during this stage.

## Running Tests
To run current tests located in [goldenFiles](https://github.com/comp-physics/FyeNNa/tree/develop/goldenFiles), change permissions for [run.sh](https://github.com/comp-physics/FyeNNa/blob/develop/run.sh). Each time the tests are run, new weights are initialized for the given test's model. To look at the model architectures of each test, go to the same **goldenFiles** folder, view each test's folder, and go to the .py file.

To add a new test, go to the [goldenFiles](https://github.com/comp-physics/FyeNNa/tree/develop/goldenFiles) directory and create a new folder which will store information about the new test being created: python model (either an imported onnx file, h5 file, etc.) or an actual definition of a model (in PyTorch, Tensorflow, etc.). 

After doing the above, there are a couple of files we need to create/write to:

``` python
with open("inputs.fpp",'w') as f1:
    inputs = inp.flatten().tolist() #store inputs to a file
    inpShapeDict = {'inputs': list(inp.shape)} #store the input shapes
    inpDict = {'inputs':inputs}  #store the inputs themselves

    #write all of this to the inputs.fpp file
    f1.write(f"""#:set inpShape = {inpShapeDict}""")
    f1.write("\n")
    f1.write(f"""#:set arrs = {inpDict}""")
    f1.write("\n")
    f1.write("a")

def stringer(mat):
    s = ""
    for elem in mat:
        s += str(elem) + " "
    return s.strip()
logits = model(inp)

filePath = "goldenFiles/gemm_big/"
#write the outputs of the model to a file so it can be compared to F90's outputs
with open(filePath+"gemm_big.txt", "w") as f2:
    f2.write(stringer(list(logits.shape)))
    f2.write("\n")
    f2.write(stringer(logits.flatten().tolist()))
print(logits.flatten().tolist())

#export the model, inferred shapes, weights, or anything to onnx
torch.onnx.export(model,
                  inp,
                  filePath+"gemm_big.onnx",
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True, 
                  input_names = ['input']
                  output_names = ['output']
                  )
```
To run this test case, we just need to call
``` shell
make testing case=NAME_OF_FILE
```

## Information about variables.fpp File
``` fortran
#:set architecture = [('Gemm', ['input', 1], None), 
                      ('Relu', ['input'], [2]), 
                      ('Gemm', ['input', 1], None), 
                      ('Relu', ['input'], [2])]

#:set inputs = []

#:set trueInputs = [['input', [1, 2]]]

#:set outShape = [['output', [1, 3]]]

#:set outputs = {'output': 'input'}
```

This is an example of the **variables.fpp** file. it contains 
1. architecture
    * list of tuples of each layer and its attributes
2. inputs
    * intermediary inputs that need to be created 
    * for example, lstm outputs 3 different things and they need to be assigned to different variables
3. trueInputs 
    * names of the actual input to the model and the shapes
4. outShape
    * names of the actual outputs of the model and the sshapes
5. outputs
    * the name corresponding to the output and what it maps to at the end of the model
    * in other cases, it may be {"output": "output2"}, which means the last layer's output name is output2, and we assign the actual output named "output" to "output2"


## Important Updates needed:
1. Currently the onnxModel.txt and onnxWeights.txt must be in the same folder as all other files, but it will be changed ASAP
2. LSTM different activation functions