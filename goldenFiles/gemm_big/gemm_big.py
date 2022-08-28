import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnx import numpy_helper
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.Sigmoid(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30,40),
            nn.Tanh(),
            nn.Linear(40,1),
            nn.Sigmoid()
        )
<<<<<<< HEAD
#nn.Linear(2, 20, bias=False),
            # nn.Linear(20, 30),
            # nn.Sigmoid(),
            # nn.Linear(30, 30),
            # nn.ReLU(),
            # nn.Linear(30,40),
            # nn.Tanh(),
            # nn.Linear(40,1),
            # nn.Sigmoid()
#TRY LATER
=======

>>>>>>> 932293133341125e44857a018a79d106ec53632e
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NN()
inp = torch.ones(1,2)

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

filePath = "goldenFiles/gemm_big/"
with open(filePath+"gemm_big.txt", "w") as f2:
    f2.write(stringer(list(logits.shape)))
    f2.write("\n")
    f2.write(stringer(logits.flatten().tolist()))
print(logits.flatten().tolist())

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"gemm_big.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"gemm_big_weights.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
