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
        # self.layer3 = nn.Linear(256, 128) 
        # self.layer4 = nn.Linear(128, 64)
        # self.layer5 = nn.Linear(64, 1)
        # self.layer6 = nn.Linear(32, 1)

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

    def classifier_model(self, x):
        x = self.layer1(x)
        x = self.swish(x)
        x = self.layer2(x)
        # x = self.swish(x)
        # x = self.layer3(x)
        # x = self.swish(x)
        # x = self.layer4(x)
        # x = self.swish(x)
        # x = self.layer5(x)
        # x = self.swish(x)
        # x = self.layer6(x)
        x = self.torch_sigmoid(x)
        return x

    # https://blog.csdn.net/out_of_memory_error/article/details/81414986
    #在這裡設定三層，每層加上swish激活函數
    def forward(self, x, label=None, eval_mode=False):
        if eval_mode :
            x = x.unsqueeze(0)
            out_anchor      = x[:,0,:]
            out_positive    = x[:,1,:]

            pos = torch.cat((out_anchor, out_positive), 1)
            x = self.classifier_model(pos).squeeze(1)
            return x

        out_anchor      = x[:,0,:]
        out_positive    = x[:,1,:]

        stepsize        = out_anchor.size()[0]

        output  = -1 * (F.pairwise_distance(out_anchor.unsqueeze(-1), out_positive.unsqueeze(-1).transpose(0,2))**2)

        negidx      = self.mineHardNegative(output.detach())

        out_negative = out_positive[negidx,:]
        labelnp     = torch.tensor(numpy.array([1.0]*len(out_positive)+[0.0]*len(out_negative))).cuda()

        pos = torch.cat(( out_anchor, out_positive), 1)
        neg = torch.cat(( out_anchor, out_negative), 1)
        # print(out_positive)
        # print(out_negative)
        new_pos_score = self.classifier_model(pos).squeeze(1)
        new_neg_score = self.classifier_model(neg).squeeze(1)
        # print(new_pos_score)
        # print(new_neg_score)
        # print(labelnp)
        # exit()
        total_socre = torch.cat((new_pos_score, new_neg_score), dim=0)
        # print(total_socre.size())
        # print(labelnp.size())
        # total_socre = torch.cat((total_socre, torch.tensor([2])), dim=0)
        # print(total_socre.unsqueeze(1))
        # nloss   = self.criterion(total_socre, labelnp)

        nloss = self.criterion(total_socre, labelnp.float())
        # exit()
        # print(total_socre.size())
        # x = total_socre[:,0]
        # y= total_socre[:,1]
        # x = y-x
        # x = x.unsqueeze(1)
        # print(x.size())
        # prec1, _ = accuracy(x.detach().cpu(), labelnp.detach().cpu(), topk=(1, 5))
        errors  = tuneThresholdfromScore(total_socre.detach().cpu(), labelnp.detach().cpu(), []);
        # exit()
        # nloss   = torch.mean(F.relu(torch.pow(new_pos_score, 2) - torch.pow(new_neg_score, 2) + self.margin))
        # scores  = torch.cat([new_pos_score,new_neg_score],dim=0).detach().cpu().numpy()
        # errors  = tuneThresholdfromScore(scores, labelnp, []);
        # print(labelnp)
        # print(scores)
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