import torch
import torch.nn as nn
import numpy as np
import torch.onnx
import onnx
import itertools 
from onnx import numpy_helper
import math
import sys


file = sys.argv[1]

onnxModel = onnx.load('goldenFiles/'+file+'/'+file+'.onnx')
onnxModel_weights = onnx.load('goldenFiles/'+file+'/'+file+'_weights.onnx')
nodes = onnxModel.graph.node
value_info = onnx.shape_inference.infer_shapes(onnxModel).graph.value_info
ioMap = {}
initializer = {}
initializerWeights = {}
intermediateShapes = {}
inputs = [] # need to export
for inp in onnxModel.graph.input:
    ioMap[inp.name] = inp.name
    inputs.append([inp.name,len(inp.type.tensor_type.shape.dim)])

for inter in value_info:
    intermediateShapes[inter.name] = [d.dim_value for d in inter.type.tensor_type.shape.dim]

for weights in onnxModel.graph.initializer:
    initializer[weights.name] = weights.dims
    initializerWeights[weights.name] = numpy_helper.to_array(weights)

out = {}
for x in onnxModel.graph.output:
    out[x.name] = [d.dim_value for d in x.type.tensor_type.shape.dim]
outputs = {} #what outputs corresponds to, need to export
outShape = [] #what shape to instantiate the output name to, need to export
modelArch = [] #need to export
extra = "0"

def stranspose(arr):
    shape = arr.shape
    combs = [x for x in range(len(shape))]
    for dim1, dim2 in itertools.combinations(combs,2):
        dim = combs.copy()
        dim[dim1] = dim2
        dim[dim2] = dim1
        arr = np.transpose(arr, dim)
    return stringer(arr.flatten().tolist())

def stringer(mat):
    s = ""
    for elem in mat:
        s += str(elem) + " "
    return s.strip()

def reshapeParser(reshape, trueShape):
    contains = False
    ind = 0
    for index, dim in enumerate(reshape):
        if dim==-1:
            ind = index
            contains = True
            break
    if not contains:
        return reshape
    else:
        res = np.prod(reshape)*-1
        true = np.prod(trueShape)
        missing_dim = int(true/res)
        reshape[ind] = missing_dim
        return reshape
    
#ONNX parser
#onnxModel.txt => holds model structure
#onnxWeights.txt => holds model's weights in corresponding order

#ioMap => dictionary that maps (outputs) -> (inputs)
#initializer => holds weights dims

#modelArch => (layer_name, input_list[], parameters) to call respective subroutines in fypp
with open('onnxWeights.txt', 'w') as f2:
    for w in onnxModel_weights.graph.initializer:
        f2.write(stranspose(numpy_helper.to_array(w)))
        f2.write("\n")


with open('onnxModel.txt','w') as f:
    f.write(str(len(nodes)))
    f.write("\n")
    for node in nodes:
        layer = node.op_type
        f.write(layer)
        f.write("\n")
        for index,x in enumerate(node.output):
            if x in out:
                outShape.append([x,len(out[x])])
        if layer == "Transpose": #for this, make sure order is set to tuple[2] and shape is set accordingly
            modelArch.append(("Transpose",[ioMap[node.input[0]]], [list(map(lambda x: x+1,node.attribute[0].ints))])) #"order"

            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "LSTM": #changes shape
            modelArch.append(("LSTM", [ioMap[node.input[0]], ioMap[node.input[-2]], ioMap[node.input[-1]]], ["output"+extra], None)) #input = ["input", "hidden_state", "cell_state"]
            inputs.append(["output"+extra,len(intermediateShapes[node.output[0]])])
            for inp in node.input[1:3]: #represents ONNX's locations of weights
                for dim in initializer[inp]:
                    f.write(str(dim)+" ")
                f.write("\n")
                # f2.write(stranspose(initializerWeights[inp]))
                # f2.write("\n")
            split = np.split(initializerWeights[node.input[3]],2,1)
            for x in range(2):
                f.write(str(int(initializer[node.input[3]][1]/2)))
                f.write("\n")
                # f2.write(stringer(split[x].flatten().tolist()))
                # f2.write("\n")
            ioMap[node.output[0]] = "output" + extra
            extra = str(int(extra)+1)
            ioMap[node.output[1]] = ioMap[node.input[-2]]
            ioMap[node.output[2]] = ioMap[node.input[-1]]

            #write weights to f2
        elif layer == "Gemm":
            modelArch.append(("Gemm", [ioMap[node.input[0]],node.attribute[2].i], None))

            for inp in node.input[1:3]:
                for dim in initializer[inp]:
                    f.write(str(dim)+ " ") 
                f.write("\n")
                # f2.write(stranspose(initializerWeights[inp]))
                # f2.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]

        #check notion summer start
        elif layer == "Squeeze": #changes shape
            #note for squeeze and for reshape:
            #look at onnx.shape_inference.infer_shapes(onnxModel).graph.value_info. make a map of {"name":dimensions}.
            #then when you arrive at squeeze, look at map[node.input[0]]'s shape and encode that information into the input
            #the input should look like this: {"output" + extra: shape/num_dimensions}
            modelArch.append(("Squeeze", (ioMap[node.input[0]], len(intermediateShapes[node.input[0]])),["output" + extra], [node.attribute[0].ints])) #axes to be squeezed
            inputs.append(["output"+extra, len(intermediateShapes[node.output[0]])])
            ioMap[node.output[0]] = "output" + extra
            extra = str(int(extra)+1)


        #check notion summer start
        elif layer == "Reshape": #changes shape
            modelArch.append(("Reshape", (ioMap[node.input[0]], len(intermediateShapes[node.input[0]])),["output" + extra], [reshapeParser(initializerWeights[node.input[-1]].tolist(), intermediateShapes[node.input[0]])])) #new shape
            inputs.append(["output"+extra, len(intermediateShapes[node.output[0]])])
            ioMap[node.output[0]] = "output" + extra
            extra = str(int(extra)+1)

        elif layer == "Conv":
            modelArch.append(("Conv", [ioMap[node.input[0]]], [node.attribute[0].ints, node.attribute[2].ints, node.attribute[3].ints, node.attribute[4].ints])) #(dilations, kernel_shape, pads, strides)

            for inp in node.input[1:3]:
                for dim in initializer[inp]:
                    f.write(str(dim)+ " ")
                f.write("\n")
                # f2.write(stranspose(initializerWeights[inp]))
                # f2.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "MaxPool":
            modelArch.append(("MaxPool", [ioMap[node.input[0]]], [node.attribute[0].i, node.attribute[2].ints, node.attribute[3].ints])) #(ceil_mode, pads, strides)
            f.write(str(node.attribute[1].ints[0]))
            f.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]
        
        elif layer == "AveragePool":
            modelArch.append(("AveragePool", [ioMap[node.input[0]]], [node.attribute[0].i, node.attribute[2].ints, node.attribute[3].ints])) #(ceil_mode, pads, strides)
            f.write(str(node.attribute[1].ints[0]))
            f.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "Pad":
            ioMap[node.output[0]] = ioMap[node.input[0]]
            
        elif layer == "Relu":
            modelArch.append(("Relu", [ioMap[node.input[0]]], None))

            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "Sigmoid":
            modelArch.append(("Sigmoid", [ioMap[node.input[0]]], None))

            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "Tanh":
            modelArch.append(("Tanh", [ioMap[node.input[0]]], None))

            ioMap[node.output[0]] = ioMap[node.input[0]]
        else:
            continue
    for x in list(ioMap.keys()):
        if x in out:
            outputs[x] = ioMap[x]
    # print(modelArch)
    # print("*"*50)
    # print(ioMap)
    # print("*"*50)
    # print(inputs)
    # print("*"*50)
    # print(outShape)
    # print("*"*50)
    # print(outputs)
    trueInputs = [[x.name, len(x.type.tensor_type.shape.dim)] for x in onnxModel.graph.input]



with open("variables.fpp",'w') as f:
    f.write(f"""#:set architecture = {modelArch}""")
    f.write("\n")
    f.write(f"""#:set inputs = {inputs}""")
    f.write("\n")
    f.write(f"""#:set trueInputs = {trueInputs}""")
    f.write("\n")
    f.write(f"""#:set outShape = {outShape}""")
    f.write("\n")
    f.write(f"""#:set outputs = {outputs}""")
    f.write("\n")
#: set tup = ('Squeeze', ('output0', 4), ['output1'], [[1]])
#${tup[2][0]}$ = RESHAPE(${tup[1][0]}$,(/#{for num in range(tup[1][1])}##{if num not in tup[3][0]}#SIZE(${tup[1][0]}$, dim = ${num+1}$)#{if num < (tup[1][1]-1)}#, #{endif}##{endif}##{endfor}#/))


