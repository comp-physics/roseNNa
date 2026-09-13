import torch
import torch.nn as nn
import sys
import os
import timeit
import numpy as np
import pathlib
import sys, getopt

opts, args = getopt.getopt(sys.argv[1:],"n")
produce = True
for opt, _ in opts:
    if opt == "-n":
        produce = False
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2,3),
            nn.ReLU()
        )

    def forward(self, inp):
        hid = self.linear_relu_stack(inp)
        return hid


model = NN()
inp = torch.ones(1,2)
if produce:
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

logits = model(inp)
filePath = "../goldenFiles/gemm_small/"
with open(filePath+"gemm_small.txt", "w") as f:
    f.write(stringer(list(logits.shape)))
    f.write("\n")
    f.write(stringer(logits.flatten().tolist()))
print(logits.flatten().tolist())

torch.onnx.export(model,
                  inp,
                  filePath+"gemm_small.onnx",
                  export_params=True, dynamo=False,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  )

torch.onnx.export(model,
                  inp,
                  filePath+"gemm_small_weights.onnx",
                  export_params=True, dynamo=False,
                  opset_version=10,
                  do_constant_folding=False,
                  input_names = ['input'],
                  output_names = ['output'],
                  )
