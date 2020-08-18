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
        #self.fc = nn.Linear(input_size, out_size)   
        self.layer1 = nn.Linear(1024, 256)
        self.layer2 = nn.Linear(256, 1)
        self.layer3 = nn.Linear(2, 1)

        self.hard_rank  = hard_rank
        self.hard_prob  = hard_prob
        self.margin     = margin
        self.relu = nn.ReLU(inplace=True)
        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=40)

        self.torch_sigmoid = nn.Sigmoid()

        # self.criterion  = torch.nn.CrossEntropyLoss()
        self.criterion  = torch.nn.BCELoss()
        print('Initialised The_fine_tune_network Loss')
    
    #定義swish激活函數
    def swish(self,x):
        return x * torch.sigmoid(x)

    def classifier_model(self, x, ori_score):
        x = self.layer1(x)
        x = self.swish(x)
        x = self.layer2(x)
        x = self.swish(x)
        x = torch.cat(( x, ori_score), 1)
        x = self.layer3(x)
        x = self.torch_sigmoid(x)
        return x

    # https://blog.csdn.net/out_of_memory_error/article/details/81414986
    #在這裡設定三層，每層加上swish激活函數
    def forward(self, x, label=None, eval_mode=False):
        if eval_mode :
            x = x.unsqueeze(0)
            out_anchor      = x[:,0,:]
            out_positive    = x[:,1,:]
            pos_dist = F.pairwise_distance(out_anchor, out_positive)
            pos = torch.cat((out_anchor, out_positive), 1)
            x = self.classifier_model(pos, pos_dist.unsqueeze(-1)).squeeze(1)
            return x

        out_anchor      = x[:,0,:]
        out_positive    = x[:,1,:]

        stepsize        = out_anchor.size()[0]

        output  = -1 * (F.pairwise_distance(out_anchor.unsqueeze(-1), out_positive.unsqueeze(-1).transpose(0,2))**2)

        negidx      = self.mineHardNegative(output.detach())

        out_negative = out_positive[negidx,:]
        labelnp     = torch.tensor(numpy.array([1.0]*len(out_positive)+[0.0]*len(out_negative))).cuda()

        # ## calculate distances
        pos_dist    = F.pairwise_distance(out_anchor, out_positive) # 正
        neg_dist    = F.pairwise_distance(out_anchor, out_negative) # 負

        # pos_score = torch.pow(pos_dist, 2) # 轉成平方分數
        # neg_score = torch.pow(neg_dist, 2) # 轉成平方分數

        pos = torch.cat(( out_anchor, out_positive), 1)
        neg = torch.cat(( out_anchor, out_negative), 1)

        new_pos_score = self.classifier_model(pos, pos_dist.unsqueeze(-1)).squeeze(1)
        new_neg_score = self.classifier_model(neg, neg_dist.unsqueeze(-1)).squeeze(1)
        total_socre = torch.cat((new_pos_score, new_neg_score), dim=0)

        nloss = self.criterion(total_socre, labelnp.float())
        errors  = tuneThresholdfromScore(total_socre.detach().cpu(), labelnp.detach().cpu(), []);
        return nloss, errors[1]

    def mineHardNegative(self, output):

        negidx = []

        for idx, similarity in enumerate(output):

            simval, simidx = torch.sort(similarity, descending=True)

            if self.hard_rank < 0:

                ## Semi hard negative mining

                semihardidx = simidx[(similarity[idx] - self.margin < simval) &  (simval < similarity[idx])]

                if len(semihardidx) == 0:
                    negidx.append(random.choice(simidx))
                else:
                    negidx.append(random.choice(semihardidx))

            else:

                ## Rank based negative mining
                
                simidx = simidx[simidx!=idx]

                if random.random() < self.hard_prob:
                    negidx.append(simidx[random.randint(0, self.hard_rank)])
                else:
                    negidx.append(random.choice(simidx))

        return negidx

# def the_fine_tune_network():
#     model = The_fine_tune_network()
#     return model