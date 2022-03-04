import torch
import torch.nn as nn
from nntester import NeuralNetwork
import time

model = NeuralNetwork()
torch.save(model.state_dict(),"nntester2.pt")
X = torch.ones(1,2)
a = time.time()
logits = model(X)
print("Time taken: {0}", time.time()-a)
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


with open('weights_biases.txt', 'w') as f:
    for shape, mat in listToParse:
        f.write(stringer(mat,len(shape)))
        f.write('\n')

# tensor([[0.4274]], grad_fn=<ReluBackward0>)
# tensor([[ 0.1412, -0.0940],
#         [-0.1593, -0.5962]])
# tensor([-0.0208,  0.2000])
# tensor([[-0.1282,  0.6175],
#         [-0.4850,  0.2070],
#         [-0.2155,  0.2263]])
# tensor([-0.0142,  0.4174,  0.3953])
# tensor([[0.2594, 0.1329, 0.5510]])
# tensor([-0.1094])
