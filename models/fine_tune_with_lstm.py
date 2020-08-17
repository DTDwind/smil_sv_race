#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.ResNetBlocks import *
import time, pdb, numpy
from accuracy import accuracy
from tuneThreshold import tuneThresholdfromScore
import random

class The_fine_tune_network(nn.Module):
    def __init__(self, input_size, out_size, hard_rank=0, hard_prob=0, margin=0):
        print('The_fine_tune_network prep')
        super(The_fine_tune_network, self).__init__()

        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=40)

        self.torch_sigmoid = nn.Sigmoid()
        elayers = 6
        self.nbrnn = torch.nn.LSTM(input_size, out_size, elayers, batch_first=True, dropout=0.05, bidirectional=True)
        print('Initialised The_fine_tune_network Loss')
    
    #定義swish激活函數
    def swish(self,x):
        return x * torch.sigmoid(x)

    def lstm_encoder_model(self, x):
        print('TAT')
        x = self.torchfb(x)
        print(x.size())
        exit()
        ilens
        xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
        x,(_, _) = self.nbrnn(x)
        # https://zhuanlan.zhihu.com/p/59772104
        return x

    def forward(self, target_feat):
        x = self.lstm_encoder_model(target_feat)
     
        return x