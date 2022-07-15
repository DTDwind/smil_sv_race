#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader_new import loadWAV
from DatasetLoader_new import loadFeat
from loss.ge2e import GE2ELoss
from loss.angleproto import AngleProtoLoss
from loss.cosface import AMSoftmax
from loss.arcface import AAMSoftmax
from loss.softmax import SoftmaxLoss
from loss.protoloss import ProtoLoss
from loss.pairwise import PairwiseLoss
from models.lstm import *
from models.fine_tune_dnn import *
from tqdm import tqdm 
import glob
import multiprocessing.dummy as mp 
import istarmap  # import to apply patch


class SpeakerNet(nn.Module):

    def __init__(self, max_frames, lr = 0.0001, margin = 1, scale = 1, hard_rank = 0, hard_prob = 0, model="alexnet50", nOut = 512, nSpeakers = 1000, optimizer = 'adam', encoder_type = 'SAP', normalize = True, trainfunc='contrastive', **kwargs):
        super(SpeakerNet, self).__init__();

        argsdict = {'nOut': nOut, 'encoder_type':encoder_type}
        self.sigmoid = nn.Sigmoid()

        self.__test_normalize__     = True
        self.setfiles_global = ''
        self.feats_global = ''
        
        self.__max_frames__ = max_frames;
        self.feat_keep = False


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Read data from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def readDataFromList(self, listfilename):

        data_list = {};

        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if not line:
                    break;

                data = line.split();
                filename = data[1];
                speaker_name = data[0]

                if not (speaker_name in data_list):
                    data_list[speaker_name] = [];
                data_list[speaker_name].append(filename);

        return data_list


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromListSave(self, listfilename, print_interval=5000, feat_dir='', test_path='', num_eval=10):
        self.eval();
        
        lines       = []
        files       = []
        filedict    = {}
        feats       = {}
        tstart      = time.time()

        if feat_dir != '':
            if not self.feat_keep:
                feat_dir = ''
                print('Saving temporary files to %s'%feat_dir)
                # if not(os.path.exists(feat_dir)):
                #     os.makedirs(feat_dir)

        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line): #  or (len(all_scores)==1000) 
                    break;

                data = line.split();

                files.append(data[1])
                files.append(data[2]) 
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()

        ## Save all features to file
        print("Reading File");
        print("setfiles len: "+str(len(setfiles)))
        for idx, file in tqdm(enumerate(setfiles), ascii=True):
            # if not self.feat_keep:
            #     inp1 = loadWAV(os.path.join(test_path,file), self.__max_frames__, evalmode=True, num_eval=num_eval).cuda()
            #     ref_feat = self.__S__.forward(inp1).detach().cpu()
            # filename = '%06d.wav'%idx
            filename = '%06d.feat.pt'%idx
            feat_dir = 'data/ResNetSE34L_feat_vox1_test/'
            if feat_dir == '':
                feats[file]     = ref_feat
            else:
                filedict[file]  = filename
                feats[file]     = torch.load(os.path.join(feat_dir, filename))

        print('')
        all_scores = [];
        all_plda_scores = [];
        all_labels = [];
        tstart = time.time()

        ## Read files and compute all scores
        print("Computing!");
        # for idx, line in tqdm(enumerate(lines), ascii=True):

        #     data = line.split();
        #     feat_dir = ''
        #     if feat_dir == '':
        #         ref_feat = feats[data[1]]#.cuda()
        #         com_feat = feats[data[2]]#.cuda()
        #     else:
        #         print('ERROR')
        #     if self.__test_normalize__:
        #         ref_feat = F.normalize(ref_feat, p=2, dim=1)
        #         com_feat = F.normalize(com_feat, p=2, dim=1)
            

        #     # dist = F.pairwise_distance(ref_feat.unsqueeze(-1).expand(-1,-1,num_eval), com_feat.unsqueeze(-1).expand(-1,-1,num_eval).transpose(0,2)).detach().cpu().numpy();
        #     dist = F.pairwise_distance(ref_feat.unsqueeze(-1).expand(-1,-1,num_eval), com_feat.unsqueeze(-1).expand(-1,-1,num_eval).transpose(0,2)).numpy();
        #     score = -1 * numpy.mean(dist);

        #     all_scores.append(score);  
        #     all_labels.append(int(data[0]));

        ## ---------------------------------------------------------------

        # with open('dev_score_QAQQQ.txt', 'w') as out:
        #     for ref_idx, ref_file in tqdm(enumerate(setfiles), ascii=True):
        #         for idx, com_file in enumerate(setfiles):
        #             if idx <= ref_idx: continue
        #             feat_dir = ''
        #             if feat_dir == '':
        #                 ref_feat = feats[ref_file]
        #                 com_feat = feats[com_file]
        #             else:
        #                 print('ERROR')
        #             if self.__test_normalize__:
        #                 ref_feat = F.normalize(ref_feat, p=2, dim=1)
        #                 com_feat = F.normalize(com_feat, p=2, dim=1)
        #             dist = F.pairwise_distance(ref_feat.unsqueeze(-1).expand(-1,-1,num_eval), com_feat.unsqueeze(-1).expand(-1,-1,num_eval).transpose(0,2)).numpy();
        #             score = -1 * numpy.mean(dist);
        #             out.write("%s %s %s\n"%(score, ref_file, com_file))
        #             out.write("%s %s %s\n"%(score, com_file, ref_file))
        # # EER 2.3648

        ## ---------------------------------------------------------------
        self.feats_global = feats
        self.setfiles_global = setfiles
        cpus = os.cpu_count() 
        print("# of cpu: "+str(cpus))
        p=mp.Pool(cpus)
        # p.starmap(self.thread_score, enumerate(setfiles))
        for _ in tqdm(p.istarmap(self.thread_score, enumerate(setfiles)), total=len(setfiles)):pass
        p.close()
        p.join()
        print('\n')

        print(' Computing Done! ')
        quit()
        return (all_scores, all_labels);

    def thread_score(self, idx, file_name):
        ref_file = file_name
        ref_idx = idx
        with open('test_score/test_score_'+str(idx)+'.txt', 'w') as out:
            for idx, com_file in enumerate(self.setfiles_global):
                if idx <= ref_idx: continue
                feat_dir = ''
                if feat_dir == '':
                    ref_feat = self.feats_global[ref_file]
                    com_feat = self.feats_global[com_file]
                else:
                    print('ERROR')
                if self.__test_normalize__:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)
                dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).numpy();
                score = -1 * numpy.mean(dist);
                out.write("%s %s %s\n"%(score, ref_file, com_file))
                out.write("%s %s %s\n"%(score, com_file, ref_file))


    def max_min_normalization(self, x):
        return numpy.array([(float(i)-min(x))/float(max(x)-min(x)) for i in x])

    def save_score(self, sc, ref, com):
        with open('dev_score.txt', 'w') as out:
            for query in order:
                self.recomment[query].sort(key=lambda k: k[1])
                out.write("Query %s    %s %s\n"%(counter, query, len(self.recomment[query])))
                for doc in self.recomment[query][::-1]:
                    out.write("%s %s\n"%(doc[0],doc[1]))
                out.write("\n")
                counter += 1
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Update learning rate
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def updateLearningRate(self, alpha):

        learning_rate = []
        for param_group in self.__optimizer__.param_groups:
            param_group['lr'] = param_group['lr']*alpha
            learning_rate.append(param_group['lr'])
        return learning_rate;


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        torch.save(self.state_dict(), path);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):
        print("No Model")

