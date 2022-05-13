import torch
import torch.nn as nn
import numpy as np
import torch.onnx
import onnx

onnxModel = onnx.load('nn_lstm.onnx')
nodes = onnxModel.graph.node

ioMap = {}
initializer = {}
for inp in onnxModel.graph.input:
    ioMap[inp.name] = inp.name

for weights in onnxModel.graph.initializer:
    initializer[weights.name] = weights.dims
#list of tuples: (layer_name, input_list[], parameters)
modelArch = []
extra = "0"

with open('onnxModel.txt','w') as f:
    f.write(str(len(nodes)))
    f.write("\n")
    for node in nodes:
        layer = node.op_type
        f.write(layer)
        f.write("\n")
        if layer == "Transpose":
            modelArch.append(("Transpose",[ioMap[node.input[0]]], node.attribute[0].ints))
            f.write(str(len(node.attribute[0].ints)))
            f.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]
        elif layer == "LSTM":
            modelArch.append(("LSTM", [ioMap[node.input[0]], node.input[-2], node.input[-1]], None)) #input = ["input", "hidden_state", "cell_state"]
            for inp in node.input[1:3]: #represents ONNX's locations of weights
                for dim in initializer[inp]:
                    f.write(str(dim)+" ")
                f.write("\n")
            for x in range(2):
                f.write(str(int(initializer[node.input[3]][1]/2)))
                f.write("\n")
            ioMap[node.output[0]] = "output" + extra
            extra = str(int(extra)+1)
            ioMap[node.output[1]] = node.input[-2]
            ioMap[node.output[2]] = node.input[-1]
            #PUT OUTPUTS IN IOMAP, but based on my output in f90, for example for lstm, the output hidden state and output cell state are stored in the first two parameters of the function. the cumulative outputs will probably be some external thing
        elif layer == "Gemm":
            modelArch.append(("Gemm", [ioMap[node.input[0]]], None))
            for inp in node.input[1:3]:
                for dim in initializer[inp]:
                    f.write(str(dim)+ " ") 
                f.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]
        elif layer == "Squeeze":
            modelArch.append(("Squeeze", [ioMap[node.input[0]]], node.attribute[0].ints))
            ioMap[node.output[0]] = "output" + extra
            extra = str(int(extra)+1)
        elif layer == "Reshape":
            modelArch.append(("Reshape", [ioMap[node.input[0]]], initializer[node.input[-1]]))
            ioMap[node.output[0]] = "output" + extra
            extra = str(int(extra)+1)
        # elif layer == "Conv":
        #     #do something
        # elif layer == "MaxPool":
        #     #do something
        elif layer == "Relu":
            modelArch.append(("Relu", [ioMap[node.input[0]]], None))
            ioMap[node.output[0]] = ioMap[node.input[0]]
        elif layer == "Sigmoid":
            modelArch.append(("Sigmoid", [ioMap[node.input[0]]], None))
            ioMap[node.output[0]] = ioMap[node.input[0]]
        elif layer == "Tanh":
            modelArch.append(("Tanh", [ioMap[node.input[0]]], None))
            ioMap[node.output[0]] = ioMap[node.input[0]]
    for a in modelArch:
        print(a)


