import torch
import torch.nn as nn
import sys
import os
import timeit
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random

torch.set_num_threads(1)
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        ran = [nn.ReLU(), nn.Tanh(), nn.Sigmoid()]
        modules = []
        modules.append(nn.Linear(4, 5))
        modules.append(nn.Linear(5, 3))

        self.linear_relu_stack = nn.Sequential(*modules)

    def forward(self, inp):
        hid = self.linear_relu_stack(inp)
        return hid
inp = torch.ones(1,4)
model = NN()

logits = model(inp)
print(logits.flatten().tolist())
SETUP_CODE = '''
import torch
import torch.nn as nn
import sys
import os
import timeit
import random
torch.set_num_threads(1)
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        ran = [nn.ReLU(), nn.Tanh(), nn.Sigmoid()]
        modules = []
        modules.append(nn.Linear(4, 5))
        modules.append(nn.Linear(5, 3))

        self.linear_relu_stack = nn.Sequential(*modules)

    def forward(self, inp):
        hid = self.linear_relu_stack(inp)
        return hid
inp = torch.ones(1,4)
model = NN()
'''
TEST_CODE = '''
with torch.jit.optimized_execution(False):
    logits = model(inp)'''

t = timeit.repeat(setup = SETUP_CODE,
                    stmt = TEST_CODE,
                    repeat = 100,
                    number = 1)
median = np.median(np.array(t))
print("Python Time:" + str(median))


with open("inputs.fpp",'w') as f1:
    inputs = inp.flatten().tolist()
    inpShapeDict = {'inputs': list(inp.shape)}
    inpDict = {'inputs':inputs}
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


filePath = "goldenFiles/droplet/"
with open(filePath+"droplet.txt", "w") as f2:
    f2.write(stringer(list(logits.shape)))
    f2.write("\n")
    f2.write(stringer(logits.flatten().tolist()))

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"droplet.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"droplet_weights.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
