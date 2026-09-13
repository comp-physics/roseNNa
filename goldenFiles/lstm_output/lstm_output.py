import torch
import torch.nn as nn
import sys
import os
import numpy as np
sys.path.insert(1, "../test/")
from nnLSTM import LSTM
import timeit
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.lstm = LSTM(10,20,1)

    def forward(self, inp, hidden):
        logits, hid = self.lstm(inp,hidden)
        return logits
SETUP_CODE = '''
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(1, "goldenFiles/")
from nnLSTM import LSTM
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.lstm = LSTM(10,20,1)

    def forward(self, inp, hidden):
        logits, hid = self.lstm(inp,hidden)
        return logits
model = NN()
batch_size = 1
seq_len = 25
hidden_dim = 20
input_dim = 10
n_layers = 1
inp = torch.ones(batch_size, seq_len, input_dim)
hidden_state = torch.ones(n_layers, batch_size, hidden_dim)
cell_state = torch.ones(n_layers, batch_size, hidden_dim)
hidden = (hidden_state, cell_state)
'''
TEST_CODE = '''
with torch.jit.optimized_execution(False):
    logits = model(inp, hidden)'''
t = timeit.repeat(setup = SETUP_CODE,
                    stmt = TEST_CODE,
                    repeat = 100,
                    number = 1)
median = np.median(np.array(t))
print("Python Time:" + str(median))
model = NN()
batch_size = 1
seq_len = 25
hidden_dim = 20
input_dim = 10
n_layers = 1
inp = torch.ones(batch_size, seq_len, input_dim)
hidden_state = torch.ones(n_layers, batch_size, hidden_dim)
cell_state = torch.ones(n_layers, batch_size, hidden_dim)
hidden = (hidden_state, cell_state)

with open("inputs.fpp",'w') as f:
    inputs = inp.flatten().tolist()
    h = hidden_state.flatten().tolist()
    c = cell_state.flatten().tolist()
    inpShapeDict = {'inputs': list(inp.shape), 'hidden_state': list(hidden_state.shape), 'cell_state': list(cell_state.shape)}
    inpDict = {'inputs': inputs, 'hidden_state': h, 'cell_state': c}
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
logits = model(inp, hidden)
filePath = "../goldenFiles/lstm_output/"

with open(filePath+"lstm_output.txt", "w") as f:
    f.write(stringer(list(logits.shape)))
    f.write("\n")
    f.write(stringer(logits.flatten().tolist()))
print(logits.flatten().tolist())

torch.onnx.export(model,
                  (inp, hidden),                         # model input (or a tuple for multiple inputs)
                  filePath+"lstm_output.onnx",
                  export_params=True, dynamo=False,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names = ['input', 'hidden_state','cell_state'],
                  output_names = ['output'],
                  )

torch.onnx.export(model,
                  (inp, hidden),                         # model input (or a tuple for multiple inputs)
                  filePath+"lstm_output_weights.onnx",
                  export_params=True, dynamo=False,
                  opset_version=10,
                  do_constant_folding=False,
                  input_names = ['input', 'hidden_state','cell_state'],
                  output_names = ['output'],
                  )
