import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnx import numpy_helper
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 3),
            nn.Sigmoid(),
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3,1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

# model = NeuralNetwork()
# inp = torch.ones(1,2)
# print(model(inp))
# print(dict(model.state_dict()))
# print("*"*20)
# Export the model
# torch.onnx.export(model,               # model being run
#                   inp,                         # model input (or a tuple for multiple inputs)
#                   "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=10,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                   )

# model2 = onnx.load('super_resolution.onnx')
# weights = model2.graph.initializer
# names = model2.graph.node
# for x,y in zip(weights,names):
#     print(y.name)
#     print(numpy_helper.to_array(x))