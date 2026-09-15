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
        # groups=2 with 4 in and 6 out: c_in_per_group is 2 and
        # c_out_per_group is 3, so they differ and the group offset is really
        # exercised. A depthwise conv (groups == channels) has both at 1,
        # where an off-by-one in the offset is indistinguishable from correct.
        self.stack = nn.Sequential(
            nn.Conv2d(4, 6, 3, padding=1, groups=2),
            nn.ReLU(),
        )

    def forward(self, inp):
        return self.stack(inp)


torch.manual_seed(0)
model = NN()
model.eval()
inp = torch.ones(1, 4, 6, 6)
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
filePath = "../goldenFiles/conv_grouped/"
with open(filePath+"conv_grouped.txt", "w") as f:
    f.write(stringer(list(logits.shape)))
    f.write("\n")
    f.write(stringer(logits.flatten().tolist()))
print(logits.flatten().tolist())

torch.onnx.export(model,
                  inp,
                  filePath+"conv_grouped.onnx",
                  export_params=True, dynamo=False,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  )

torch.onnx.export(model,
                  inp,
                  filePath+"conv_grouped_weights.onnx",
                  export_params=True, dynamo=False,
                  opset_version=10,
                  do_constant_folding=False,
                  input_names = ['input'],
                  output_names = ['output'],
                  )
