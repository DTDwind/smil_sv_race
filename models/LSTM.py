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
    # def new_parameter(self, *size):
    #     out = nn.Parameter(torch.FloatTensor(*size))
    #     nn.init.xavier_normal_(out)
    #     return out
    #定義swish激活函數
    def swish(self,x):
        return x * torch.sigmoid(x)
#https://blog.csdn.net/out_of_memory_error/article/details/81414986
    #在這裡設定三層，每層加上swish激活函數
    def forward(self, x):
       # x = self.fc(x)
        x = self.layer1(x)
        x = self.swish(x)
        x = self.layer2(x)
        x = self.swish(x)
        x = self.layer3(x)
        return x




# class Activation_Net(nn.Module):
#     #在上面fc的基礎上，每層的輸出部分添加激活函數
#     def __init__(self, in_dim, out_dim):
#         super(Activation_Net, self).__init__()
#         # self.layer1 = nn.Sequential(nn.Linear(in_dim, 1024), nn.ReLU(True))
#         # self.layer2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(True))
#         self.layer1 = nn.Sequential(nn.Linear(in_dim, 1024), swish(3))
#         self.layer2 = nn.Sequential(nn.Linear(1024, 1024), swish(3))
#         self.layer3 = nn.Sequential(nn.Linear(1024, out_dim))
#         '''
#         Sequential()函數的功能是將網絡的層組合再一起
#         '''
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         return x


def The_Fc():
    model = th_Fc()
    return model
