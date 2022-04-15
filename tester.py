import torch
import torch.nn as nn
from nnLSTM import LSTM
# from lstm_linear import NN
# from nntester import NeuralNetwork
# import time
# import torch
# import torch.nn as nn
# import torch.onnx
# import onnx
# from onnx import numpy_helper

# # # model = NeuralNetwork()


# batch_size = 1
# seq_len = 2
# hidden_dim = 2
# input_dim = 5
# n_layers = 1
# model = LSTM(input_dim, hidden_dim, n_layers)
# inp = torch.ones(batch_size, seq_len, input_dim)
# hidden_state = torch.ones(n_layers, batch_size, hidden_dim)
# cell_state = torch.ones(n_layers, batch_size, hidden_dim)
# hidden = (hidden_state, cell_state)


# logits = model(inp, hidden)
# print("*"*10)
# print(logits[0])
# print(logits[1])

m = nn.Conv2d(2,3,3,stride=1)
a = nn.MaxPool2d(2)
print((m.state_dict()["weight"]))
input = torch.ones(1,2,6,6)
print("*"*20)
print(m(input))
print("*"*20)

print(a(input))

# # print("Logits: ")
# # print(logits.shape)

# # print()Z
# # for l in model.state_dict():
# #     print(l, model.state_dict()[l].shape)

# model2 = onnx.load('nn_lstm.onnx')
# weights = model2.graph.initializer
# names = model2.graph.node
# for x,y in zip(weights,names):
#     print(y.name)
#     print(numpy_helper.to_array(x))

# print("-"*20)
# for x in weights:
#     print(x.name)