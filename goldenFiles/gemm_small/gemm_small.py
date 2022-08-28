import torch
import torch.nn as nn
import sys
import os
import timeit
import numpy as np
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.Linear(100, 2),
            nn.Sigmoid(),
            nn.Linear(2, 3),
            nn.Sigmoid(),
        )

    def forward(self, inp):
        # logits, hid = self.lstm(inp,hidden) #logits will have shape of hidden_dimension
        # print("Logits")
        # print(logits)
        # print("------")
        # print("Hidden")
        # print(hid)
        # print("------")
        # logits = logits.view(-1,logits.size(2)) #.view reshapes it to a valid (1,hid_dim) for lin layer
        # logits = self.linear_relu_stack(logits)
        hid = self.linear_relu_stack(inp)
        return hid

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
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.Linear(100, 2),
            nn.Sigmoid(),
            nn.Linear(2, 3),
            nn.Sigmoid(),
        )

    def forward(self, inp):
        # logits, hid = self.lstm(inp,hidden) #logits will have shape of hidden_dimension
        # print("Logits")
        # print(logits)
        # print("------")
        # print("Hidden")
        # print(hid)
        # print("------")
        # logits = logits.view(-1,logits.size(2)) #.view reshapes it to a valid (1,hid_dim) for lin layer
        # logits = self.linear_relu_stack(logits)
        hid = self.linear_relu_stack(inp)
        return hid
model = NN()
inp = torch.ones(1,2)
'''
model = NN()
inp = torch.ones(1,2)

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
TEST_CODE = '''
with torch.jit.optimized_execution(False):
    logits = model(inp)'''
times = timeit.repeat(setup = SETUP_CODE,
                          stmt = TEST_CODE,
                          repeat = 10000,
                          number = 1)
print(f"Median is: {np.median(np.array(times))}")
logits = model(inp)
filePath = "goldenFiles/gemm_small/"
with open(filePath+"gemm_small.txt", "w") as f:
    f.write(stringer(list(logits.shape)))
    f.write("\n")
    f.write(stringer(logits.flatten().tolist()))
print(logits.flatten().tolist())

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"gemm_small.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"gemm_small_weights.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
