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

run = True
SETUP_CODE = '''
import torch
import torch.nn as nn
import sys
import os
import timeit
torch.set_num_threads(1)
class NN(nn.Module):
    def __init__(self,inp,out,kernel):
        super(NN, self).__init__()
        self.conv = nn.Conv2d(inp,out,kernel)

    def forward(self, inp):
        return self.conv(inp)


models = {}
for inp in [5,50,100,500,1000]:
    for neurons in [10,25,50,100]:
        s = "Layers"+str(layer)+"Neurons"+str(neurons)
        models[s] = (NN(1,1,neurons),torch.rand(1,1,inp,inp))
'''
TEST_CODE = '''
with torch.jit.optimized_execution(False):
    logits = models'''


if run:
    filePath = "graphs/conv/"
    with open(filePath+"times.txt", "w") as f:
        for layer in [1,5]:
            print(f"Doing layer {layer}")
            for neurons in [10,25]:
                
                s = "'Layers"+str(layer)+"Neurons"+str(neurons)+"'"
                inputS = "models["+s+"][1]"
                TEST = TEST_CODE+"["+s+"][0]("+inputS+")"
                t = timeit.repeat(setup = SETUP_CODE,
                                        stmt = TEST,
                                        repeat = 100,
                                        number = 1)
                median = np.median(np.array(t))
                print(f"Time for layer {layer}, {neurons} neurons: {median}")
                f.write(str(median)+ " ")
    # plot(pd.DataFrame(times)).show()

#------------------------------F90 PART-------------------------------------
class NN(nn.Module):
    def __init__(self,inp,out,kernel):
        super(NN, self).__init__()
        self.conv = nn.Conv2d(inp,out,kernel)

    def forward(self, inp):
        return self.conv(inp)


models = {}
for inp in [5,50,100,500,1000]:
    for neurons in [10,25,50,100]:
        s = "Layers"+str(layer)+"Neurons"+str(neurons)
        models[s] = (NN(1,1,neurons),torch.rand(1,1,inp,inp))


with open("inputs.fpp",'w') as f:
    inputs = inp.flatten().tolist()
    inpShapeDict = {'inputs': list(inp.shape)}
    inpDict = {'inputs':inputs}
    f.write(f"""#:set inpShape = {inpShapeDict}""")
    f.write("\n")
    f.write(f"""#:set arrs = {inpDict}""")
    f.write("\n")
    f.write("a")

def stringer(mat):
    s = ""
    for elem in mat:
        s += str(elem) + " "
    return s.strip()

for i,model_name in enumerate(models):
    model = models[model_name]
    print(model(inp))
    torch.onnx.export(model,               # model being run
                    inp,                         # model input (or a tuple for multiple inputs)
                    "graphs/mlp/"+model_name+".onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    )

# torch.onnx.export(model,               # model being run
#                   inp,                         # model input (or a tuple for multiple inputs)
#                   filePath+"gemm_small_weights.onnx",   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=10,          # the ONNX version to export the model to
#                   do_constant_folding=False,  # whether to execute constant folding for optimization
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                   )


