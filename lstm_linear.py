import torch
import torch.nn as nn
from nnLSTM import LSTM
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.lstm = LSTM(5,2,1)
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

    def forward(self, inp, hidden):
        logits, hid = self.lstm(inp,hidden) #logits will have shape of hidden_dimension
        print("Logits")
        print(logits)
        print("------")
        print("Hidden")
        print(hid)
        print("------")
        logits = logits.view(-1,logits.size(2)) #.view reshapes it to a valid (1,hid_dim) for lin layer
        logits = self.linear_relu_stack(logits)
        return logits


# class NN(nn.Module):
#     def __init__(self):
#         super(NN, self).__init__()
#         # self.conv = nn.MaxPool2d(3)
#         self.lstm = LSTM(5,2,1)
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(2, 2),
#             nn.ReLU(),
#             nn.Linear(2, 3),
#             nn.Sigmoid(),
#             nn.Linear(3, 3),
#             nn.ReLU(),
#             nn.Linear(3,1),
#             nn.Sigmoid(),
#         )


#     def forward(self, inp): #hidden
#         logits, hid = self.lstm(inp,hidden) #logits will have shape of hidden_dimension
#         # print(logits)
#         logits = logits.view(-1,logits.size(2)) #.view reshapes it to a valid (1,hid_dim) for lin layer
#         logits = self.linear_relu_stack(logits)
#         return self.conv(inp)

#forward
# for x in model.state_dict():
#     print(x,'  ', model.state_dict()[x].shape)
# print(model(inp, hidden))
