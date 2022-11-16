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

run = True
SETUP_CODE = '''
import torch
import torch.nn as nn
import sys
import os
class NN(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers):
        super(NN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hid_dim, num_layers, batch_first=True)

    def forward(self, inp, hidden):
        logits, hid = self.lstm(inp,hidden)
        return logits



batch_size = 1
input_dim = 5
n_layers = 1


models = {}
for seq_len in [2,5,10,50,100]:
    for hidden_dim in [3,25,50,75]:
        s = "Seq"+str(seq_len)+"Neurons"+str(hidden_dim)
        inp = torch.rand(batch_size, seq_len, input_dim)
        hidden_state = torch.rand(n_layers, batch_size, hidden_dim)
        cell_state = torch.rand(n_layers, batch_size, hidden_dim)
        hidden = (hidden_state, cell_state)
        models[s] = (NN(input_dim,hidden_dim,n_layers),(inp,hidden))
'''
TEST_CODE = '''
with torch.jit.optimized_execution(False):
    logits = models'''


if run:
    filePath = "graphs/lstm/"
    with open(filePath+"times.txt", "w") as f:
        for seq_len in [2,5,10,50,100]:
            print(f"Doing sequence length {seq_len}")
            for hidden_dim in [3,25,50,75]:
                s = "'Seq"+str(seq_len)+"Neurons"+str(hidden_dim)+"'"
                inputS = "models["+s+"][1][0], "+"models["+s+"][1][1]"
                TEST = TEST_CODE+"["+s+"][0]("+inputS+")"
                t = timeit.repeat(setup = SETUP_CODE,
                                        stmt = TEST,
                                        repeat = 100,
                                        number = 1)
                median = np.median(np.array(t))
                print(f"Time for seq len {seq_len}, {hidden_dim} hid dim: {median}")
                f.write(str(median)+ " ")



