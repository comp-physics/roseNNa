# A classifier head: the shape Softmax actually turns up in.
#
# opset 13 on purpose, where the other golden models use 10. Before opset 13
# Softmax coerced its input to 2-D and normalised every trailing axis together;
# from 13 it normalises along one axis. Both readings agree here (rank 2, last
# axis), but pinning 13 means this file exercises the semantics the generator
# validates against rather than relying on that coincidence.
import torch
import torch.nn as nn
import sys, getopt

opts, args = getopt.getopt(sys.argv[1:], "n")
produce = True
for opt, _ in opts:
    if opt == "-n":
        produce = False


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 5),
            nn.Softmax(dim=-1),
        )

    def forward(self, inp):
        return self.stack(inp)


torch.manual_seed(0)
model = NN()
inp = torch.ones(1, 4)
if produce:
    with open("inputs.fpp", 'w') as f:
        inputs = inp.flatten().tolist()
        inpShapeDict = {'inputs': list(inp.shape)}
        inpDict = {'inputs': inputs}
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
filePath = "../goldenFiles/softmax_head/"
with open(filePath + "softmax_head.txt", "w") as f:
    f.write(stringer(list(logits.shape)))
    f.write("\n")
    f.write(stringer(logits.flatten().tolist()))
print(logits.flatten().tolist())

for fname, fold in (("softmax_head.onnx", True), ("softmax_head_weights.onnx", False)):
    torch.onnx.export(model,
                      inp,
                      filePath + fname,
                      export_params=True, dynamo=False,
                      opset_version=13,
                      do_constant_folding=fold,
                      input_names=['input'],
                      output_names=['output'],
                      )
