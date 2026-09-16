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
        # A 1-D stack: conv with padding and stride, a pool, then a
        # second conv. On a flat row-major buffer a rank-3 (N,C,W) value and
        # a rank-4 (N,C,1,W) one are the same bytes, so this goes through the
        # 2-D nest with a height of 1 -- and that equivalence is what the
        # golden comparison is here to keep honest.
        self.stack = nn.Sequential(
            nn.Conv1d(3, 6, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(6, 4, 3, padding=1, stride=2),
            nn.ReLU(),
        )

    def forward(self, inp):
        return self.stack(inp)


torch.manual_seed(0)
model = NN()
model.eval()
inp = torch.ones(1, 3, 24)
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
filePath = "../goldenFiles/conv1d_stack/"
with open(filePath+"conv1d_stack.txt", "w") as f:
    f.write(stringer(list(logits.shape)))
    f.write("\n")
    f.write(stringer(logits.flatten().tolist()))
print(logits.flatten().tolist())

torch.onnx.export(model,
                  inp,
                  filePath+"conv1d_stack.onnx",
                  export_params=True, dynamo=False,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  )

torch.onnx.export(model,
                  inp,
                  filePath+"conv1d_stack_weights.onnx",
                  export_params=True, dynamo=False,
                  opset_version=10,
                  do_constant_folding=False,
                  input_names = ['input'],
                  output_names = ['output'],
                  )
