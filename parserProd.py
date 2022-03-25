import torch
import torch.nn as nn
from lstm_linear import NN
from nntester import NeuralNetwork
import time

# model = NeuralNetwork()
model = NN()
torch.save(model.state_dict(),"nntester3.pt")
# torch.save(model.state_dict(),"nntester2.pt")
batch_size = 1
seq_len = 1
hidden_dim = 2
input_dim = 5
n_layers = 1
inp = torch.ones(batch_size, seq_len, input_dim)
hidden_state = torch.ones(n_layers, batch_size, hidden_dim)
cell_state = torch.ones(n_layers, batch_size, hidden_dim)
hidden = (hidden_state, cell_state)
# X = torch.ones(1,2)
a = time.time()
# logits = model(X)
logits = model(inp, hidden)
print(f"Time taken: {time.time()-a}")
print(logits)
listToParse = []
for l in model.state_dict():
    print(model.state_dict()[l])
    try:
        listToParse.append((list(model.state_dict()[l].shape),torch.transpose(model.state_dict()[l],0,1).tolist()))
    except:
        listToParse.append((list(model.state_dict()[l].shape),model.state_dict()[l].tolist()))

def stringer(mat, dim):
    s = ""
    if dim > 1:
        for row in mat:
            for col in row:
                s += str(col) + " "
            s += "\n"
    else:
        for elem in mat:
            s += str(elem) + " "
    return s.strip()


with open('weights_biases2.txt', 'w') as f:
    for shape, mat in listToParse:
        f.write(stringer(mat,len(shape)))
        f.write('\n')
