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
        print('fc prep')
        super(th_Fc, self).__init__()
        #self.fc = nn.Linear(input_size, out_size)   
        self.layer1 = nn.Linear(input_size, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, out_size) 
    
    #定義swish激活函數
    def swish(self,x):
        return x * torch.sigmoid(x)

    # https://blog.csdn.net/out_of_memory_error/article/details/81414986
    #在這裡設定三層，每層加上swish激活函數
    def forward(self, x):
       # x = self.fc(x)
        x = self.layer1(x)
        x = self.swish(x)
        x = self.layer2(x)
        x = self.swish(x)
        x = self.layer3(x)
        return x


def The_Fc():
    model = th_Fc()
    return model