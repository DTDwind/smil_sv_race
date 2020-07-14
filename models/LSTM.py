#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.ResNetBlocks import *

class th_Fc(nn.Module):
    def __init__(self,input_size,out_size):
        print('extract fbank prep')
        super(th_Fc, self).__init__()
        self.fc = nn.Linear(input_size, out_size)    
    # def new_parameter(self, *size):
    #     out = nn.Parameter(torch.FloatTensor(*size))
    #     nn.init.xavier_normal_(out)
    #     return out

    def forward(self, x):
        x = self.fc(x)
        return x


def The_Fc():
    model = th_Fc()
    return model
