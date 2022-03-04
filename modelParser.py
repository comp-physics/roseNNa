import torch
import torch.nn as nn
from nntester import NeuralNetwork

model = NeuralNetwork()
model.load_state_dict(torch.load("nntester2.pt"))

activation_functions = {'ReLU()':0,'Sigmoid()':1}

def nested_children(f,m):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        if isinstance(m,nn.Linear):
            f.write('{1} {0}'.format(m.in_features,m.out_features))
        elif isinstance(m,nn.Flatten):
            f.write('Flatten')
        elif isinstance(m,nn.ReLU):
            f.write(str(activation_functions[str(m)]))
        elif isinstance(m,nn.Sigmoid):
            f.write(str(activation_functions[str(m)]))
        f.write('\n')
        #elif isinstance(m,nn.)
        #ADD more as model flexibility increases

        return m
    else:
        for name, child in children.items():
            nested_children(f,child)

with open('model.txt','w') as f:
    f.write(str(int(len(model.state_dict())/2)))
    f.write('\n')
    nested_children(f,model)
# print(nested_children(model))
# print("***")
# print(dict(model.named_children()))
