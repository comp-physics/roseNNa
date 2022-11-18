import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnx import numpy_helper
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv = nn.Conv2d(2,3,3, padding=[2,3], stride=2)

    def forward(self, inp):
        return self.conv(inp)

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
logits = model(inp)

filePath = "../goldenFiles/conv_padding-stride/"
with open(filePath+"conv_padding-stride.txt", "w") as f2:
    f2.write(stringer(list(logits.shape)))
    f2.write("\n")
    f2.write(stringer(logits.flatten().tolist()))
print(logits.flatten().tolist())

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"conv_padding-stride.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"conv_padding-stride_weights.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
