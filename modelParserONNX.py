import torch
import torch.nn as nn
import numpy as np
import torch.onnx
import onnx
import itertools
from onnx import numpy_helper
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--onnxfile',"-f", required=True, help="Please provide .onnx file of your pretrained model.")
parser.add_argument('--weights',"-w", help="(Optional) Please provide .onnx file of your pretrained model without any optimizations (do_constant_folding = False).")
parser.add_argument('--inferred',"-i", help="(Optional) Please provide .onnx file that has inferred shapes")


args = parser.parse_args()

file = args.onnxfile
weights = args.weights
inferred = args.inferred


onnxModel = onnx.load(file)

#lstm files require external weights file (unparsed onnx file, look up for details)
externalWeightsFile = True
try:
    onnxModel_weights = onnx.load(weights)
except:
    onnxModel_weights = onnxModel
    externalWeightsFile = False

#sometimes the inferred shapes is too big, so we need to store an external file of the precomputed inferred shapes
try:
    inferred = onnx.load(inferred)
    value_info = inferred.graph.value_info
except:
    value_info = onnx.shape_inference.infer_shapes(onnxModel).graph.value_info

nodes = onnxModel.graph.node #all layers of model that will be parsed

ioMap = {} #mapping input names to output names
initializer = {} #holds (weight dimensions, np.array of weights)
intermediateShapes = {} #holds intermediate shapes of layers
inputs = [] #general inputs to the model (including intermediary stuff needed for fortran to process)
input_shapes = {} #shapes of inputs
constants = {} #constants and initalizer weights are the places where weights of the model could be stored
for inp in onnxModel.graph.input:
    ioMap[inp.name] = inp.name
    input_shapes[inp.name] = [d.dim_value for d in inp.type.tensor_type.shape.dim]

for inter in value_info:
    intermediateShapes[inter.name] = [d.dim_value for d in inter.type.tensor_type.shape.dim]

for weights in onnxModel.graph.initializer:
    initializer[weights.name] = (weights.dims,numpy_helper.to_array(weights))
out = {}
for x in onnxModel.graph.output:
    out[x.name] = [d.dim_value if d.dim_value!=0 else 1 for d in x.type.tensor_type.shape.dim]

for x in onnxModel.graph.node:
    if x.op_type == "Constant":
        constants[x.output[0]] = numpy_helper.to_array(x.attribute[0].t)

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

def findWeightsInitializer(input_name):
    if input_name in initializer:
        return initializer[input_name][1]
    return constants[input_name]

def fourDTransform(trueshape, toBeTransformedShape):
    new = [1,1,1,1]
    i = 0
    for dim in toBeTransformedShape:
        try:
            find = trueshape.index(dim)
            new[find-len(trueshape)] = dim
        except:
            pass
    return new

def fakeFourD(inp):
    add = 4-len(inp)
    return add*[1] + inp

def spreadInfo(trueShape, toBeTransformedShape):
    ret = []
    for index,dim in enumerate(toBeTransformedShape):
        if trueShape[index] != dim:
            ret.append(index+1)
            ret.append(trueShape[index])
    return ret

#ONNX parser
#onnxModel.txt => holds model structure
#onnxWeights.txt => holds model's weights in corresponding order

#ioMap => dictionary that maps (outputs) -> (inputs)
#initializer => holds weights dims

#modelArch => (layer_name, input_list[], parameters) to call respective subroutines in fypp
print("starting to write weights..")
# with open('onnxWeights.txt', 'w') as f2:
#     for w in onnxModel_weights.graph.initializer:
#         f2.write(stranspose(numpy_helper.to_array(w)))
#         f2.write("\n")
print("starting parsing...")
true_index = 0
true_weights = onnxModel_weights.graph.initializer
for node in nodes:
    print(node.op_type)
with open('onnxModel.txt','w') as f, open('onnxWeights.txt', 'w') as f2:
    f.write(str(len(nodes)))
    f.write("\n")
    for node in nodes:
        layer = node.op_type
        
        for index,x in enumerate(node.output):
            if x in out:
                outShape.append([x,out[x]])
        if layer == "Transpose": #for this, make sure order is set to tuple[2] and shape is set accordingly
            f.write(layer)
            f.write("\n")
            modelArch.append(("Transpose",[ioMap[node.input[0]]], [list(map(lambda x: x+1,node.attribute[0].ints))])) #"order"

            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "LSTM": #changes shape
            f.write(layer)
            f.write("\n")
            writeHCs = False
            try:
                modelArch.append(("LSTM", [ioMap[node.input[0]], ioMap[node.input[5]], ioMap[node.input[6]]], ["output"+extra], [0])) #input = ["input", "hidden_state", "cell_state"]
                f.write("0")
                f.write("\n")
            except:
                modelArch.append(("LSTM", [ioMap[node.input[0]], "output"+str(int(extra)+1),"output"+str(int(extra)+2)], ["output"+extra], [1])) #input = ["input", "hidden_state", "cell_state"]
                writeHCs = True
                f.write("1")
                f.write("\n")
            inputs.append(["output"+extra,len(intermediateShapes[node.output[0]])])
            for inp in node.input[1:3]: #represents ONNX's locations of weights
                for dim in initializer[inp][0]:
                    f.write(str(dim)+" ")
                if externalWeightsFile:
                    f2.write(stranspose(numpy_helper.to_array(true_weights[true_index])))
                    true_index+=1
                else:
                    f2.write(stranspose(findWeightsInitializer(inp)))
                f2.write("\n")
                f.write("\n")
            #check if bias exists
            if not externalWeightsFile:
                split = np.split(findWeightsInitializer(node.input[3]),2,axis=1)
            for x in range(2):
                f.write(str(int(initializer[node.input[3]][0][1]/2)))
                f.write("\n")
                if externalWeightsFile:
                    f2.write(stranspose(numpy_helper.to_array(true_weights[true_index])))
                    true_index+=1
                else:
                    f2.write(stranspose(split[x]))
                f2.write("\n")
            if writeHCs:
                inpShape = intermediateShapes[node.input[0]]
                batch_size = inpShape[1]
                if inpShape[1] == 0:
                    batch_size = 1
                hidden = 0
                for x in node.attribute:
                    if x.name == "hidden_size":
                        hidden = x.i
                        break
                shape = (1,batch_size,hidden)
                for x in range(2):
                    for s in shape:
                        f.write(str(s)+" ")
                    f2.write(stranspose(np.zeros(shape)))
                    f.write("\n")
                    f2.write("\n")
            if not writeHCs:
                ioMap[node.output[0]] = "output" + extra
                extra = str(int(extra)+1)
                ioMap[node.output[1]] = ioMap[node.input[-2]]
                
                ioMap[node.output[2]] = ioMap[node.input[-1]]
            else:
                ioMap[node.output[0]] = "output" + extra
                extra = str(int(extra)+1)

                inputs.append(["output"+extra,len(intermediateShapes[node.output[1]])])
                ioMap[node.output[1]] = "output" + extra
                extra = str(int(extra)+1)

                inputs.append(["output"+extra,len(intermediateShapes[node.output[2]])])
                ioMap[node.output[2]] = "output" + extra
                extra = str(int(extra)+1)

        elif layer == "Gemm":
            f.write(layer)
            f.write("\n")
            modelArch.append(("Gemm", [ioMap[node.input[0]],node.attribute[2].i], None))
            numzs = 0
            #check if bias exists
            if len(node.input) < 3:
                for inp in node.input[1:]:
                    numzs = initializer[inp][0][0]
                    for dim in initializer[inp][0]:
                        f.write(str(dim)+ " ")
                    if externalWeightsFile:
                        f2.write(stranspose(numpy_helper.to_array(true_weights[true_index])))
                        true_index+=1
                    else:
                        f2.write(stranspose(findWeightsInitializer(inp)))
                    f2.write("\n")
                    f.write("\n")
                f.write(str(numzs))
                f.write("\n")
                f2.write(stranspose(np.zeros(numzs)))
                f2.write("\n")
            else:
                for inp in node.input[1:3]:
                    for dim in initializer[inp][0]:
                        f.write(str(dim)+ " ")
                    if externalWeightsFile:
                        f2.write(stranspose(numpy_helper.to_array(true_weights[true_index])))
                        true_index+=1
                    else:
                        f2.write(stranspose(findWeightsInitializer(inp)))
                    f2.write("\n")
                    f.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]

        #check notion summer start
        elif layer == "Squeeze": #changes shape
            #note for squeeze and for reshape:
            #look at onnx.shape_inference.infer_shapes(onnxModel).graph.value_info. make a map of {"name":dimensions}.
            #then when you arrive at squeeze, look at map[node.input[0]]'s shape and encode that information into the input
            #the input should look like this: {"output" + extra: shape/num_dimensions}
            f.write(layer)
            f.write("\n")
            try:
                modelArch.append(("Squeeze", (ioMap[node.input[0]], len(intermediateShapes[node.input[0]])),["output" + extra], [node.attribute[0].ints])) #axes to be squeezed
            except:
                modelArch.append(("Squeeze", (ioMap[node.input[0]], len(intermediateShapes[node.input[0]])),["output" + extra], [findWeightsInitializer(node.input[1]).tolist()])) #axes to be squeezed
            inputs.append(["output"+extra, len(intermediateShapes[node.output[0]])])
            ioMap[node.output[0]] = "output" + extra
            extra = str(int(extra)+1)


        #check notion summer start
        elif layer == "Reshape": #changes shape
            f.write(layer)
            f.write("\n")
            try:
                modelArch.append(("Reshape", (ioMap[node.input[0]], len(intermediateShapes[node.input[0]])),["output" + extra], [reshapeParser(findWeightsInitializer(node.input[-1]).tolist(), intermediateShapes[node.input[0]])],[0])) #new shape
                f.write("0")
                f.write("\n")
            except:
                modelArch.append(("Reshape", (ioMap[node.input[0]], len(initializer[node.input[0]][0])),["output" + extra], [reshapeParser(findWeightsInitializer(node.input[-1]).tolist(), initializer[node.input[0]][0])], [1])) #new shape
                f.write(str(len(initializer[node.input[0]][0])))
                f.write("\n")
                for dim in initializer[node.input[0]][0]:
                    f.write(str(dim)+ " ")
                f.write("\n")
                f2.write(stranspose(findWeightsInitializer(node.input[0])))
                f2.write("\n")
            inputs.append(["output"+extra, len(intermediateShapes[node.output[0]])])
            ioMap[node.output[0]] = "output" + extra
            extra = str(int(extra)+1)

        elif layer == "Conv":
            f.write(layer)
            f.write("\n")
            attributes = {}
            auto_pad = False
            for attr in node.attribute:
                name = str(attr.name)
                if name == "group":
                    pass
                elif name == "auto_pad":
                    auto_pad = True
                    attributes['auto_pad'] = attr.s.decode('ASCII')
                else:
                    attributes[attr.name] = attr.ints

            if auto_pad: # DEAL WITH STRIDE > 1?
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
            modelArch.append(("Conv", [ioMap[node.input[0]]], [attributes['dilations'], attributes['kernel_shape'], attributes['pads'], attributes['strides']])) #(dilations, kernel_shape, pads, strides)

            if len(node.input) < 3:
                numzs = 0
                for inp in node.input[1:]:
                    numzs = initializer[inp][0][0]
                    for dim in initializer[inp][0]:
                        f.write(str(dim)+ " ")
                    if externalWeightsFile:
                        f2.write(stranspose(numpy_helper.to_array(true_weights[true_index])))
                        true_index+=1
                    else:
                        f2.write(stranspose(findWeightsInitializer(inp)))
                    f2.write("\n")
                    f.write("\n")
                f.write(str(numzs))
                f.write("\n")
                f2.write(stranspose(np.zeros(numzs)))
                f2.write("\n")
            else:
                for inp in node.input[1:3]:
                    for dim in initializer[inp][0]:
                        f.write(str(dim)+ " ")
                    if externalWeightsFile:
                        f2.write(stranspose(numpy_helper.to_array(true_weights[true_index])))
                        true_index+=1
                    else:
                        f2.write(stranspose(findWeightsInitializer(inp)))
                    f2.write("\n")
                    f.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "MaxPool":
            f.write(layer)
            f.write("\n")
            attributes = {}
            auto_pad = False
            #--SOME CONSTANTS THAT ARE REQUIRED IN ARGUMENTS AND MAY NOT APPEAR IN ONNX--
            attributes['ceil_mode'] = 0
            #-------------
            for attr in node.attribute:
                name = str(attr.name)
                if name == "ceil_mode":
                    attributes[attr.name] = attr.i
                elif name == "auto_pad":
                    attributes['auto_pad'] = attr.s.decode('ASCII')
                    if attributes['auto_pad'] != "NOTSET":
                        auto_pad = True
                else:
                    attributes[attr.name] = attr.ints
            if auto_pad: # DEAL WITH STRIDE > 1?
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
            f.write(str(node.attribute[1].ints[0]))
            f.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "AveragePool":
            f.write(layer)
            f.write("\n")
            modelArch.append(("AveragePool", [ioMap[node.input[0]]], [node.attribute[0].i, node.attribute[2].ints, node.attribute[3].ints])) #(ceil_mode, pads, strides)
            f.write(str(node.attribute[1].ints[0]))
            f.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "Add":
            f.write(layer)
            f.write("\n")
            fourd = fourDTransform(intermediateShapes[node.input[0]],findWeightsInitializer(node.input[-1]).shape)
            true = fakeFourD(intermediateShapes[node.input[0]])
            modelArch.append(("Add",[ioMap[node.input[0]]], [true, spreadInfo(true,fourd),len(intermediateShapes[node.input[0]])])) #[trueshape, need to be broadcasted and added SHAPE]
            for dim in fourd:
                f.write(str(dim) + " ")
            f.write("\n")
            if externalWeightsFile:
                f2.write(stranspose(numpy_helper.to_array(true_weights[true_index])))
                true_index+=1
            else:
                f2.write(stranspose(findWeightsInitializer(node.input[1])))
            f2.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "MatMul":
            try:
                f.write(layer)
                f.write("\n")
                modelArch.append(("MatMul",[ioMap[node.input[0]],ioMap[node.input[1]]], [len(intermediateShapes[node.input[0]])])) #[trueshape, need to be broadcasted and added SHAPE]
            except:
                f.write("Gemm")
                f.write("\n")
                modelArch.append(("Gemm", [ioMap[node.input[0]],1], None))
                numzs = 0
                #check if bias exists
                for inp in node.input[1:]:
                    numzs = initializer[inp][0][0]
                    for dim in initializer[inp][0]:
                        f.write(str(dim)+ " ")
                    if externalWeightsFile:
                        f2.write(stranspose(numpy_helper.to_array(true_weights[true_index])))
                        true_index+=1
                    else:
                        f2.write(stranspose(findWeightsInitializer(inp)))
                    f2.write("\n")
                    f.write("\n")
                f.write(str(numzs))
                f.write("\n")
                f2.write(stranspose(np.zeros(numzs)))
                f2.write("\n")


            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "Pad": #FINISH
            f.write(layer)
            f.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "Relu":
            f.write(layer)
            f.write("\n")
            modelArch.append(("Relu", [ioMap[node.input[0]]], [len(intermediateShapes[node.input[0]])]))

            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "Sigmoid":
            f.write(layer)
            f.write("\n")
            modelArch.append(("Sigmoid", [ioMap[node.input[0]]], [len(intermediateShapes[node.input[0]])]))

            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "Tanh":
            f.write(layer)
            f.write("\n")
            modelArch.append(("Tanh", [ioMap[node.input[0]]], [len(intermediateShapes[node.input[0]])]))

            ioMap[node.output[0]] = ioMap[node.input[0]]
        elif layer == "Constant":
            f.write(layer)
            f.write("\n")
            continue
        else:
            print(modelArch)
            print(f'{layer} NOT SUPPORTED BY RoseNNa CURRENTLY!')
            ioMap[node.output[0]] = ioMap[node.input[0]]
            continue
    for x in list(ioMap.keys()):
        if x in out:
            outputs[x] = ioMap[x]
    trueInputs = [[x.name, [a.dim_value if a.dim_value!=0 else 1 for a in x.type.tensor_type.shape.dim]] for x in onnxModel.graph.input if x.name not in initializer]
    print(modelArch)


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
