import torch
import torch.nn as nn
import numpy as np
from lstm_linear import NN
from nntester import NeuralNetwork

#model def
# model = NeuralNetwork()
model = NN()
#model input creation
# model.load_state_dict(torch.load("nntester2.pt"))
model.load_state_dict(torch.load("nntester4.pt"))

activation_functions = {'ReLU()':0,'Sigmoid()':1}
layer_order = []
def write_lstm(f, lstm):
    for index, layer in enumerate(lstm.state_dict()):
        if index%4 == 0: #current fix if lstm is stacked, another solution is to pass another parameter to the function to indicate how many lstms are stacked
            f.write('lstm')
            f.write('\n')
        try:
            f.write('{0} {1}'.format(lstm.state_dict()[layer].shape[0],lstm.state_dict()[layer].shape[1]))
        except:
            f.write('{0}'.format(lstm.state_dict()[layer].shape[0]))
        if index != len(lstm.state_dict()) - 1:
            f.write('\n')
def write_layer(f, layer, name):
    f.write(name)
    f.write('\n')
    for index, typeLayer in enumerate(layer.state_dict()):
        line = ''
        for num in list(layer.state_dict()[typeLayer].shape):
            line += str(num) + " "
        f.write(line.strip())
        if index != len(layer.state_dict()) - 1:
            f.write('\n')
def nested_children(f,m):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        if isinstance(m,nn.Linear):
            layer_order.append('linear')
            write_layer(f, m, 'linear')
        elif isinstance(m,nn.LSTM):
            layer_order.append('lstm')
            write_lstm(f,m)
        elif isinstance(m,nn.Conv2d):
            layer_order.append('conv')
            write_layer(f, m, 'conv')
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

with open('model3.txt','w') as f:
    sum = 0
    for key in model.named_children():
        if isinstance(key[1], nn.Sequential):
            sum += int(len(dict(key[1].named_children()))/2)
        else:
            sum+=1
    f.write(str(sum))
    f.write('\n')
    nested_children(f,model)
    print("Layer order: ", layer_order)
    # with open('test.npy', 'wb') as f2:
    #     np.save(f2, np.array(layer_order))