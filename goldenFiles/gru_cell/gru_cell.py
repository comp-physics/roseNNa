# A GRU with its initial hidden state passed in, like the lstm_* goldens.
#
# The state is an explicit forward() argument on purpose. Left to itself
# PyTorch builds h0 from the input's batch dimension with an Expand, which
# exports a SYMBOLIC dimension -- and roseNNa fixes every shape at generation
# time, so it refuses that model by name. Passing the state makes it an
# ordinary graph input with a literal shape.
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
        self.gru = nn.GRU(5, 3, 1)

    def forward(self, inp, h0):
        out, h = self.gru(inp, h0)
        return out


torch.manual_seed(0)
model = NN()
model.eval()
seq_len, batch_size, input_dim, hidden_dim, n_layers = 4, 1, 5, 3, 1
inp = torch.ones(seq_len, batch_size, input_dim)
h0 = torch.ones(n_layers, batch_size, hidden_dim) * 0.1

if produce:
    with open("inputs.fpp", 'w') as f:
        inpShapeDict = {'inputs': list(inp.shape), 'hidden_state': list(h0.shape)}
        inpDict = {'inputs': inp.flatten().tolist(), 'hidden_state': h0.flatten().tolist()}
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


logits = model(inp, h0)
filePath = "../goldenFiles/gru_cell/"
with open(filePath + "gru_cell.txt", "w") as f:
    f.write(stringer(list(logits.shape)))
    f.write("\n")
    f.write(stringer(logits.flatten().tolist()))
print(logits.flatten().tolist())

for fname, fold in (("gru_cell.onnx", True), ("gru_cell_weights.onnx", False)):
    torch.onnx.export(model,
                      (inp, h0),
                      filePath + fname,
                      export_params=True, dynamo=False,
                      opset_version=13,
                      do_constant_folding=fold,
                      input_names=['input', 'hidden_state'],
                      output_names=['output'],
                      )
