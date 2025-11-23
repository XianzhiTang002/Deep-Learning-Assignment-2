from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        # Define layers here ...
        self.Whx = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.Whh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.bh = nn.Parameter(torch.zeros(self.hidden_dim))
        self.Wph = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        self.bo = nn.Parameter(torch.zeros(self.output_dim))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Implementation here ...
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_dim)
        for t in range(self.seq_length):
            x_t = x[:, t, :]
            h_t = self.relu(self.Whx(x_t) + self.Whh(h_t) + self.bh)
        o_t = self.Wph(h_t) + self.bo
        out = self.softmax(o_t)
        return out

        
    # add more methods here if needed
