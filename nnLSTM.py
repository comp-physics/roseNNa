import torch
import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, inp, hidden):
        return self.lstm(inp, hidden)

model = LSTM(5,10,1)
for x in model.state_dict():
    print(x, ' ', model.state_dict()[x].shape)
print(model.forward(torch.randn(1,1,5)))