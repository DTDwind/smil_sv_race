#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from accuracy import accuracy
from tuneThreshold import tuneThresholdfromScore
import random

class PairwiseLoss(nn.Module):

    def __init__(self, loss_func=None, hard_rank=0, hard_prob=0, margin=0):
        super(PairwiseLoss, self).__init__()
        self.loss_func  = loss_func
        self.hard_rank  = hard_rank
        self.hard_prob  = hard_prob
        self.margin     = margin

        print('Initialised Pairwise Loss')

    def forward(self, x, label=None):
        
        out_anchor      = x[:,0,:]
        out_positive    = x[:,1,:]
        stepsize        = out_anchor.size()[0]

        output  = -1 * (F.pairwise_distance(out_anchor.unsqueeze(-1).expand(-1,-1,stepsize),out_positive.unsqueeze(-1).expand(-1,-1,stepsize).transpose(0,2))**2)

        negidx      = self.mineHardNegative(output.detach())

        out_negative = out_positive[negidx,:]

        labelnp     = numpy.array([1]*len(out_positive)+[0]*len(out_negative))

        ## calculate distances
        pos_dist    = F.pairwise_distance(out_anchor,out_positive)
        neg_dist    = F.pairwise_distance(out_anchor,out_negative)

        ## loss functions
        if self.loss_func == 'contrastive':
            nloss       = torch.mean(torch.cat([torch.pow(pos_dist, 2),torch.pow(F.relu(self.margin - neg_dist), 2)],dim=0))
        elif self.loss_func == 'triplet':
            nloss   = torch.mean(F.relu(torch.pow(pos_dist, 2) - torch.pow(neg_dist, 2) + self.margin))

        scores = -1 * torch.cat([pos_dist,neg_dist],dim=0).detach().cpu().numpy()

        errors = tuneThresholdfromScore(scores, labelnp, []);

        return nloss, errors[1]

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Hard negative mining
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def mineHardNegative(self, output):

        negidx = []

        for idx, similarity in enumerate(output):

            simval, simidx = torch.sort(similarity,descending=True)

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

    def chnm(self):
        self.margin += 0.1
        return self.margin