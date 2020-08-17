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
        # self.layer1 = nn.Linear(64481, 1024)
        # self.layer2 = nn.Linear(1024, 1024)
        # self.layer3 = nn.Linear(1024, 512) 
        # self.layer4 = nn.Linear(512, 128)
        # self.layer5 = nn.Linear(128, 16)
        # self.layer6 = nn.Linear(16, 1)

        self.hard_rank  = hard_rank
        self.hard_prob  = hard_prob
        self.margin     = margin

        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=40)

        self.torch_sigmoid = nn.Sigmoid()

        
        self.nbrnn = torch.nn.LSTM(input_size, out_size/2, elayers=6, batch_first=True, dropout=0.05, bidirectional=True)
        print('Initialised The_fine_tune_network Loss')
    
    #定義swish激活函數
    def swish(self,x):
        return x * torch.sigmoid(x)

    def score_model(self, x):
        x = self.layer1(x)
        x = self.swish(x)
        x = self.layer2(x)
        x = self.swish(x)
        x = self.layer3(x)
        x = self.swish(x)
        x = self.layer4(x)
        x = self.swish(x)
        x = self.layer5(x)
        x = self.swish(x)
        x = self.layer6(x)
        x = self.torch_sigmoid(x)
        return x
    
    def lstm_encoder_model(self, x):
        self.nbrnn            
        # https://zhuanlan.zhihu.com/p/59772104
        return x

    def forward(self, x,  ori_feat, label=None, eval_mode=False):
        if eval_mode :
            x = x.unsqueeze(0)
            ori_feat = ori_feat.unsqueeze(0)
            out_anchor      = x[:,0,:]
            out_positive    = x[:,1,:]

            ori_anchor = ori_feat[:,0,:]
            ori_positive = ori_feat[:,1,:]
            pos_dist = F.pairwise_distance(out_anchor, out_positive)
            pos_score = torch.pow(pos_dist, 2) # 轉成平方分數
            pos = torch.cat((pos_score.unsqueeze(-1), ori_anchor, ori_positive), 1)
            x = self.score_model(pos).squeeze(1)*-1
            return x
        out_anchor      = x[:,0,:]
        out_positive    = x[:,1,:]

        ori_anchor = ori_feat[:,0,:]
        ori_positive = ori_feat[:,1,:]
        
        stepsize        = out_anchor.size()[0]

        output  = -1 * (F.pairwise_distance(out_anchor.unsqueeze(-1), out_positive.unsqueeze(-1).transpose(0,2))**2)

        negidx      = self.mineHardNegative(output.detach())

        out_negative = out_positive[negidx,:]
        labelnp     = numpy.array([1]*len(out_positive)+[0]*len(out_negative))

        ## calculate distances
        pos_dist    = F.pairwise_distance(out_anchor, out_positive) # 正
        neg_dist    = F.pairwise_distance(out_anchor, out_negative) # 負

        pos_score = torch.pow(pos_dist, 2) # 轉成平方分數
        neg_score = torch.pow(neg_dist, 2) # 轉成平方分數
        

        # 先不做fbank 因為維度不好處理
        # ori_anchor = self.torchfb(ori_anchor)
        # ori_positive = self.torchfb(ori_positive)

        ori_negative = ori_positive[negidx,:]

        pos = torch.cat((pos_score.unsqueeze(-1), ori_anchor, ori_positive), 1)
        neg = torch.cat((neg_score.unsqueeze(-1), ori_anchor, ori_negative), 1)
        new_pos_score = self.score_model(pos).squeeze(1)
        new_neg_score = self.score_model(neg).squeeze(1)
            
        nloss   = torch.mean(F.relu(torch.pow(new_pos_score, 2) - torch.pow(new_neg_score, 2) + self.margin))
        scores = -1 * torch.cat([new_pos_score,new_neg_score],dim=0).detach().cpu().numpy()
        errors = tuneThresholdfromScore(scores, labelnp, []);
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