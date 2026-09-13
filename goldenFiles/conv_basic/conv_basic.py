import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnx import numpy_helper
import timeit
import numpy as np

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv = nn.Conv2d(2,3,3)

    def forward(self, inp):
        return self.conv(inp)

model = NN()
inp = torch.rand(2,2,7,7)

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

logits = model(inp)

filePath = "../goldenFiles/conv_basic/"
with open(filePath+"conv_basic.txt", "w") as f2:
    f2.write(stringer(list(logits.shape)))
    f2.write("\n")
    f2.write(stringer(logits.flatten().tolist()))
print(logits)

torch.onnx.export(model,
                  inp,
                  filePath+"conv_basic.onnx",
                  export_params=True, dynamo=False,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  )

torch.onnx.export(model,
                  inp,
                  filePath+"conv_basic_weights.onnx",
                  export_params=True, dynamo=False,
                  opset_version=10,
                  do_constant_folding=False,
                  input_names = ['input'],
                  output_names = ['output'],
                  )
