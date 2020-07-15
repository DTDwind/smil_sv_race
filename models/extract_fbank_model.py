#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.ResNetBlocks import *

class ex_fbnak(nn.Module):
    def __init__(self):
        print('extract fbank prep')
        super(ex_fbnak, self).__init__()
        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=40)
    # def new_parameter(self, *size):
    #     out = nn.Parameter(torch.FloatTensor(*size))
    #     nn.init.xavier_normal_(out)
    #     return out

    def forward(self, x):
        x = self.torchfb(x)+1e-6
        return x


def Ex_Fbank():
    model = ex_fbnak()
    return model
