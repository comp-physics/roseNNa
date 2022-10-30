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
import timeit
torch.set_num_threads(1)
class NN(nn.Module):
    def __init__(self,kernel):
        super(NN, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel)

    def forward(self, inp):
        return self.maxpool(inp)

models = {}
for inp in [30,50,100,500,1000]:
    for neurons in [3,9,15,25]:
        s = "Layers"+str(inp)+"Neurons"+str(neurons)
        models[s] = (NN(neurons),torch.rand(1,1,inp,inp))
'''

TEST_CODE = '''
with torch.jit.optimized_execution(False):
    logits = models'''


if run:
    filePath = "graphs/maxpool/"
    with open(filePath+"times.txt", "w") as f:
        for inp in [30,50,100,500,1000]:
            print(f"Doing inp size {inp}")
            for neurons in [3,9,15,25]:
                s = "'Layers"+str(inp)+"Neurons"+str(neurons)+"'"
                inputS = "models["+s+"][1]"
                TEST = TEST_CODE+"["+s+"][0]("+inputS+")"
                t = timeit.repeat(setup = SETUP_CODE,
                                        stmt = TEST,
                                        repeat = 100,
                                        number = 1)
                median = np.median(np.array(t))
                print(f"Time for layer {inp}, {neurons} neurons: {median}")
                f.write(str(median)+ " ")

