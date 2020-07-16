#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.ResNetBlocks import *

class rnn_LSTM(nn.Module):
    def __init__(self,input_size,out_size):
        print('LSTM prep')
        super(rnn_LSTM, self).__init__()
        # self.fc = nn.Linear(input_size, out_size)    
        self.lstm_6 = torch.nn.LSTM(input_size, out_size, 3, bidirectional=True, batch_first=True)
        self.sap_linear = nn.Linear(input_size*2, input_size*2)
        self.attention = self.new_parameter(input_size*2, 1)
        self.fc = nn.Linear(input_size*2, out_size)
        # self.lstm_6.flatten_parameters()

    def forward(self, x):
        # print(x.size())
        x,(y,z) = self.lstm_6(x)
        # x = x.permute(1, 0, 2)
        # print('lstm')
        # print(x.size())
        # print(y.size())
        # print(z.size())
        # print('lstm end')
        h = torch.tanh(self.sap_linear(x))
        w = torch.matmul(h, self.attention).squeeze(dim=2)
        w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
        x = torch.sum(x * w, dim=1)
        # print('attention')
        
        # x = self.fc(x)
        # print(x.size())
        return x

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

def lstm_6l():
    model = rnn_LSTM()
    return model
