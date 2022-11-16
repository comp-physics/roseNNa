import torch
import torch.nn as nn
import sys
import os
import timeit
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import random

seq_len, hid_dim = list(map(int,sys.argv[1:]))

class NN(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers):
        super(NN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hid_dim, num_layers, batch_first=True)

    def forward(self, inp, hidden):
        logits, hid = self.lstm(inp,hidden)
        return logits

seq_lens = [2,5,10,50,100]
hid_dims = [3,25,50,75]
seq_len = seq_lens[seq_len]
hid_dim = hid_dims[hid_dim]
batch_size = 1
input_dim = 5
n_layers = 1
def put(layer, m):
    return (m-len(layer))*"0"+layer
s = "SeqLen"+put(str(seq_len),4)+"Hid_dim"+put(str(hid_dim),4)
model = NN(input_dim,hid_dim,n_layers)
inp = torch.rand(batch_size, seq_len, input_dim)
hidden_state = torch.rand(n_layers, batch_size, hid_dim)
cell_state = torch.rand(n_layers, batch_size, hid_dim)
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

logits = model(inp,hidden)
print(logits.flatten().tolist())
torch.onnx.export(model,               # model being run
                (inp,hidden),                         # model input (or a tuple for multiple inputs)
                "graphs/lstm/"+s+".onnx",   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input', 'hidden_state','cell_state'],  # the model's input names
                output_names = ['output'], # the model's output names
                )

torch.onnx.export(model,               # model being run
                  (inp, hidden),                         # model input (or a tuple for multiple inputs)
                  "graphs/lstm/"+s+"_weights.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input', 'hidden_state','cell_state'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
