import torch
import torch.nn as nn
import sys
import os
import timeit
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import random

inpSize, neuron = list(map(int,sys.argv[1:]))

class NN(nn.Module):
    def __init__(self,kernel):
        super(NN, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel)

    def forward(self, inp):
        return self.maxpool(inp)

inps = [30,50,100,500,1000]
neurons = [3,9,15,25]
inpSize = inps[inpSize]
neuron = neurons[neuron]
def put(layer, m):
    return (m-len(layer))*"0"+layer
s = "InputSize"+put(str(inpSize),4)+"KernelDim"+put(str(neuron),4)
model = NN(neuron)
inp = torch.rand(1,1,inpSize,inpSize)


with open("inputs.fpp",'w') as f:
    inputs = inp.flatten().tolist()
    inpShapeDict = {'inputs': list(inp.shape)}
    inpDict = {'inputs':inputs}
    f.write(f"""#:set inpShape = {inpShapeDict}""")
    f.write("\n")
    f.write(f"""#:set arrs = {inpDict}""")
    f.write("\n")
    f.write("a")

print(model(inp))
torch.onnx.export(model,               # model being run
                inp,                         # model input (or a tuple for multiple inputs)
                "graphs/maxpool/"+s+".onnx",   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['output'], # the model's output names
                )