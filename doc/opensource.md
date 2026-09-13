# Open Source Development
This project is ongoing and does not contain functionality of every layer available in ONNX. In order to embed new layers into roseNNa, certain steps must be followed:

## Parsing in modelParserONNX.py
This file reads in the ONNX interpretation of the model. At a higher level, it iterattes over all the layers in the ONNX model (called nodes in the graph), parses its contents by (1) sending some of its options to be parsed in f90 via fypp and (2) finding the weights that correspond to this layer and writing their dimensions to 'onnxModel.txt' and the weights to `onnxWeights.bin`. These two files will be read in by Fortran so it can store the weights and layers. Here is a pseudocode example from the "GEMM" layer in ONNX:

```python
#an additional elif branch must be added so the parser knows to parse this layer
elif layer == "Gemm":
    #the layer name tells reader.f90 which read routine to call
    f.write(layer)
    f.write("\n")
    names = {n.name:n.i if n.type==2 else n.ints for n in node.attribute}
    #(the full branch also rejects attributes roseNNa cannot honour: transA, alpha, beta, and a bias that is not rank 1)

    #modelArch stores the layer and options for layer (fypp input later on)
    #ioMap is referenced to get the output name from the last layer (which is input to this layer)
    modelArch.append(("Gemm", [ioMap[node.input[0]], names.get('transB', 0)], None))

    #parsing the weight and bias inputs to the layer
    #(when the bias is absent, the full branch writes a zero bias instead)
    for inp in node.input[1:3]:

        #writing the dimensions to 'onnxModel.txt'
        for dim in initializer[inp][0]:
            f.write(str(dim)+ " ")
        f.write("\n")

        #writing the weights to 'onnxWeights.bin' as little-endian float64 in column-major (Fortran) order
        #findWeightsInitializer looks the tensor up by name among the initializers and Constant nodes
        f2.write(np.asarray(findWeightsInitializer(inp), dtype='<f8').flatten(order='F').tobytes())

    #at the end, we have to make sure that the names for the inputs are preseved. The output name (e.g. "out1") will be the input to the next layer, so we will be using "out1" as the input
    #this must be stored in some kind of map
    ioMap[node.output[0]] = ioMap[node.input[0]]
```
`onnxWeights.bin` is a raw stream with no header or record markers: every tensor is appended as little-endian float64 values in column-major order, in the same order its dimensions appear in `onnxModel.txt`, so Fortran reads each tensor with a single unformatted `read` into an array of those dimensions.
## Adding Layer
Most layers come with a set of parameters that are commonly manipulated (number of layers, activation functions, hidden state, etc.). This information can be integrated by creating a derived type of the layer in [derived_types.f90](https://github.com/comp-physics/roseNNa/blob/master/fLibrary/derived_types.f90). Here is an example:

``` fortran
TYPE lstmLayer
    REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: whh
    REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: wih
    REAL (c_double), ALLOCATABLE, DIMENSION(:) :: bhh
    REAL (c_double), ALLOCATABLE, DIMENSION(:) :: bih
    REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: hid
    REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: cell
ENDTYPE lstmLayer
```
The LSTM layer requires 4 weight and bias arrays that are used while running through the layer, plus `hid` and `cell`, the zero initial states read when the ONNX node has no `initial_h`/`initial_c` inputs. They are stored within the derived type. The layer dimensions cannot be changed later on.

## Adding activation function
For any activation functions that need to be added will go in [activation_funcs.f90](https://github.com/comp-physics/roseNNa/blob/master/fLibrary/activation_funcs.f90). To do so, just a function needs to be created. Here is an example:

``` fortran
FUNCTION tanhh(x) result(output)
    REAL (c_double), intent(in) :: x(:)
    REAL (c_double) :: output(size(x))
    output = tanh(x)
END FUNCTION tanhh
```

## Reading layer in reader.f90
In order to read in the weights and layers from the files `onnxModel.txt` and `onnxWeights.bin`, the file [reader.f90](https://github.com/comp-physics/roseNNa/blob/master/fLibrary/reader.f90) has to include the new layer/activation function. First, we will create an array of derived types for the new layer. This will allow us to store multiple of the same layer if the model contains it (we make it allocatable so it can be appended to with no dimension restrictions). Then, we create a new subroutine for the layer, which defines how we will read in the weights/dimensions (this will depend based on how you wrote the dimensions to the files in the first place). Here is an example:

``` fortran
!subroutine definition for GEMM/MLP layer (file1=dimensions, file2=weights)
!binary is .true. unless the weights path ends in .txt; the binary weights unit is opened with access='stream', form='unformatted'
subroutine read_linear(file1, file2, binary)
    INTEGER, INTENT(IN) :: file1
    INTEGER, INTENT(IN) :: file2
    LOGICAL, INTENT(IN) :: binary

    !create temporary derived type for this one layer
    TYPE(linLayer), ALLOCATABLE,DIMENSION(:) :: lin
    REAL (c_double), ALLOCATABLE, DIMENSION(:,:) :: weights
    REAL (c_double), ALLOCATABLE, DIMENSION(:) :: biases
    INTEGER :: w_dim1
    INTEGER :: w_dim2

    !read in dimensions from file1 and allocate weights to store the incoming weights
    ALLOCATE(lin(1))
    read(file1, *) w_dim1, w_dim2
    ALLOCATE(weights(w_dim1,w_dim2))

    !read in the weights: one unformatted read from the binary stream, or a list-directed read from a legacy text file
    if (binary) then
        read(file2) weights
    else
        read(file2, *) weights
    end if

    !repeat for biases
    read(file1, *) w_dim1
    ALLOCATE(biases(w_dim1))
    if (binary) then
        read(file2) biases
    else
        read(file2, *) biases
    end if

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
After encoding the layer/activation function and reading it, fypp will construct the model. Fypp takes in the model architecture, inputs, outputs, and shapes, all of which have been written to an external fypp file. In [modelCreator.fpp](https://github.com/comp-physics/roseNNa/blob/master/fLibrary/modelCreator.fpp), there is a condition for each of the layers that need to be added. Here is an example for the multilayer perceptron layer (GEMM):

``` fortran
#: if tup[0] == 'Gemm'
    !========Gemm Layer============
    CALL linear_layer(${tup[1][0]}$, linLayers(${layer_dict[tup[0]]}$),${1-tup[1][1]}$)
```
In this example, we call the `linear_layer` implemented in `layers.f90` and pass in arguments that come from the external fypp files. There is a for loop running through each layer in the model architecture (a list of tuples), and `tup` contains certain arguments that enables the tool to call the correct names and arguments. `linLayers` is defined in the reader file and stores information about the **i**th layer. One thing to make sure is to store the correct information in model architecture so it can be referenced during this stage.

## Running Tests
To run current tests located in [goldenFiles](https://github.com/comp-physics/roseNNa/tree/master/goldenFiles), change permissions for [run.sh](https://github.com/comp-physics/roseNNa/blob/master/test/run.sh). Each time the tests are run, new weights are initialized for the given test's model. To look at the model architectures of each test, go to the same **goldenFiles** folder, view each test's folder, and go to the .py file.

To add a new test, go to the [goldenFiles](https://github.com/comp-physics/roseNNa/tree/master/goldenFiles) directory and create a new folder which will store information about the new test being created: python model (either an imported onnx file, h5 file, etc.) or an actual definition of a model (in PyTorch, Tensorflow, etc.). 

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

filePath = "../goldenFiles/gemm_big/"
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
                  export_params=True, dynamo=False,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output']
                  )
```
To run this test case, we just need to call
``` shell
make testing case=NAME_OF_FILE
```

## Information about variables.fpp File
``` fortran
#:set architecture = [('Gemm', ['v_input', 1], None), 
                      ('Relu', ['v_input'], [2]), 
                      ('Gemm', ['v_input', 1], None), 
                      ('Relu', ['v_input'], [2])]

#:set inputs = []

#:set trueInputs = [['v_input', [1, 2]]]

#:set outShape = [['v_output', [1, 3]]]

#:set outputs = {'v_output': 'v_input'}
```

This is an example of the **variables.fpp** file (for `gemm_small`). Tensor names from the ONNX graph are prefixed with `v_` so they cannot collide with Fortran identifiers. It contains 
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
    * in other cases, it may be {"v_output": "output2"}, which means the last layer's output name is output2, and we assign the actual output named "output" to "output2"


## Important Updates needed:
1. LSTM different activation functions