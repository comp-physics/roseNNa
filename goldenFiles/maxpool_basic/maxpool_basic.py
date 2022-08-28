import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnx import numpy_helper
<<<<<<< HEAD
import timeit
import numpy as np
SETUP_CODE = '''
import torch
import torch.nn as nn
import sys
import os
import timeit
torch.set_num_threads(1)
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.maxpool = nn.MaxPool2d(3)

    def forward(self, inp):
        return self.maxpool(inp)

model = NN()
inp = torch.rand(1,2,6,6)
'''
=======
>>>>>>> 932293133341125e44857a018a79d106ec53632e
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.maxpool = nn.MaxPool2d(3)

    def forward(self, inp):
        return self.maxpool(inp)

model = NN()
inp = torch.rand(1,2,6,6)

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
<<<<<<< HEAD
TEST_CODE = '''
with torch.jit.optimized_execution(False):
    logits = model(inp)'''
times = timeit.repeat(setup = SETUP_CODE,
                          stmt = TEST_CODE,
                          repeat = 10000,
                          number = 1)
print(f"Median is: {np.median(np.array(times))}")
=======
>>>>>>> 932293133341125e44857a018a79d106ec53632e
logits = model(inp)

filePath = "goldenFiles/maxpool_basic/"
with open(filePath+"maxpool_basic.txt", "w") as f2:
    f2.write(stringer(list(logits.shape)))
    f2.write("\n")
    f2.write(stringer(logits.flatten().tolist()))
print(logits.flatten().tolist())

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"maxpool_basic.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"maxpool_basic_weights.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
