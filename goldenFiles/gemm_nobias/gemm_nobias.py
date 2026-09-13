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

torch.manual_seed(0)

# nn.Linear(bias=False) exports as MatMul, not Gemm; the parser lowers it to a
# Gemm with a zero bias. Rectangular layers catch transB and bias-length errors,
# and there is no final ReLU so the outputs cannot all be clamped to zero.
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(3, 4, bias=False),
            nn.Linear(4, 2, bias=False),
        )

    def forward(self, inp):
        return self.linear_stack(inp)


model = NN()
inp = torch.randn(1,3)
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
filePath = "../goldenFiles/gemm_nobias/"
with open(filePath+"gemm_nobias.txt", "w") as f:
    f.write(stringer(list(logits.shape)))
    f.write("\n")
    f.write(stringer(logits.flatten().tolist()))
print(logits.flatten().tolist())

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"gemm_nobias.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True, dynamo=False,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"gemm_nobias_weights.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True, dynamo=False,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
