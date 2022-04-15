import torch
import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, inp, hidden):
        # for d in self.lstm.state_dict():
        #     print(self.lstm.state_dict()[d], self.lstm.state_dict()[d].shape)
        # print(self.lstm(inp, hidden))
        return self.lstm(inp, hidden)

# batch_size = 1
# seq_len = 1
# hidden_dim = 10
# input_dim = 5
# n_layers = 1
# model = LSTM(input_dim,hidden_dim,n_layers)
# for x in model.state_dict():
#     print(x, ' ', model.state_dict()[x].shape)
# inp = torch.ones(batch_size, seq_len, input_dim)
# hidden_state = torch.ones(n_layers, batch_size, hidden_dim)
# cell_state = torch.ones(n_layers, batch_size, hidden_dim)
# hidden = (hidden_state, cell_state)
# a,hidden = model(inp,hidden)
# print(a)
# print(a.view(-1,a.size(2)).shape) 