#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import loadWAV
# from DatasetFeatLoader import loadFeat
from loss.ge2e import GE2ELoss
from loss.angleproto import AngleProtoLoss
from loss.cosface import AMSoftmax
from loss.arcface import AAMSoftmax
from loss.softmax import SoftmaxLoss
from loss.protoloss import ProtoLoss
from loss.pairwise import PairwiseLoss
from models.lstm import *
from models.triplet_fine_tune_network import *
from tqdm import tqdm 
import glob

class SpeakerNet(nn.Module):

    def __init__(self, max_frames, lr = 0.0001, margin = 1, scale = 1, hard_rank = 0, hard_prob = 0, model="alexnet50", nOut = 512, nSpeakers = 1000, optimizer = 'adam', encoder_type = 'SAP', normalize = True, trainfunc='contrastive', **kwargs):
        super(SpeakerNet, self).__init__();

        argsdict = {'nOut': nOut, 'encoder_type':encoder_type}

        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__(model)
        self.__S__ = SpeakerNetModel(**argsdict).cuda();
        self.fine_tune_network = The_fine_tune_network(nOut, nOut, hard_rank=hard_rank, hard_prob=hard_prob, margin=margin).cuda();


        if trainfunc == 'angleproto':
            self.__L__ = AngleProtoLoss().cuda()
            self.__train_normalize__    = True
            self.__test_normalize__     = True
        elif trainfunc == 'ge2e':
            self.__L__ = GE2ELoss().cuda()
            self.__train_normalize__    = True
            self.__test_normalize__     = True
        elif trainfunc == 'amsoftmax':
            self.__L__ = AMSoftmax(in_feats=nOut, n_classes=nSpeakers, m=margin, s=scale).cuda()
            self.__train_normalize__    = False
            self.__test_normalize__     = True
        elif trainfunc == 'aamsoftmax':
            self.__L__ = AAMSoftmax(in_feats=nOut, n_classes=nSpeakers, m=margin, s=scale).cuda()
            self.__train_normalize__    = False
            self.__test_normalize__     = True
        elif trainfunc == 'softmax':
            self.__L__ = SoftmaxLoss(in_feats=nOut, n_classes=nSpeakers).cuda()
            self.__train_normalize__    = False
            self.__test_normalize__     = True
        elif trainfunc == 'proto':
            self.__L__ = ProtoLoss().cuda()
            self.__train_normalize__    = False
            self.__test_normalize__     = False
        elif trainfunc == 'triplet':
            self.__L__ = PairwiseLoss(loss_func='triplet', hard_rank=hard_rank, hard_prob=hard_prob, margin=margin).cuda()
            self.__train_normalize__    = True
            self.__test_normalize__     = True
        elif trainfunc == 'contrastive':
            self.__L__ = PairwiseLoss(loss_func='contrastive', hard_rank=hard_rank, hard_prob=hard_prob, margin=margin).cuda()
            self.__train_normalize__    = True
            self.__test_normalize__     = True
        else:
            raise ValueError('Undefined loss.')

        if optimizer == 'adam':
            # self.__optimizer__ = torch.optim.Adam(self.parameters(), lr = lr);
            # 設定學習率SpeakerNet設定'lr':1e-6、全連結層'lr':0.001
            self.__optimizer__ = torch.optim.Adam([{'params':self.__S__.parameters(),'lr':0},
                                                   {'params':self.fine_tune_network.parameters(),'lr':0.001},
                                                   {'params':self.__L__.parameters(),'lr':0}
                                                  ]);
            # https://blog.csdn.net/qq_34914551/article/details/87699317
        elif optimizer == 'sgd':
            self.__optimizer__ = torch.optim.SGD(self.parameters(), lr = lr, momentum = 0.9, weight_decay=5e-5);
        else:
            raise ValueError('Undefined optimizer.')
        
        self.__max_frames__ = max_frames;
        self.feat_keep = False

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Train network
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader):

        self.train();

        stepsize = loader.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0     # EER or accuracy

        criterion = torch.nn.CrossEntropyLoss()
        for data, data_label in loader:
            tstart = time.time()
            self.zero_grad();
            feat = []
            ori_feat = []

            for inp in data:
                outp      = self.__S__.forward(inp.cuda())

                if self.__train_normalize__:
                    outp   = F.normalize(outp, p=2, dim=1)
                feat.append(outp)
                ori_feat.append(inp.cuda())

            feat = torch.stack(feat,dim=1).squeeze()
            ori_feat = torch.stack(ori_feat,dim=1).squeeze()
            
            label   = torch.LongTensor(data_label).cuda()

            # nloss, prec1 = self.__L__.forward(feat,label)
            nloss, prec1 = self.fine_tune_network.forward(feat, ori_feat, label) # resnet score + ori_wav*2 (希望可以用 lstm 處理...) 輸出 暫定sigmoid

            loss    += nloss.detach().cpu();
            top1    += prec1
            counter += 1;
            index   += stepsize;

            nloss.backward(); # ------------ backward 更新參數
            self.__optimizer__.step();

            telapsed = time.time() - tstart

            sys.stdout.write("\rProcessing (%d/%d) "%(index, loader.nFiles));
            sys.stdout.write("Loss %f EER/T1 %2.3f%% - %.2f Hz "%(loss/counter, top1/counter, stepsize/telapsed));
            sys.stdout.write("Q:(%d/%d)"%(loader.qsize(), loader.maxQueueSize));
            sys.stdout.flush();

        sys.stdout.write("\n");
        
        return (loss/counter, top1/counter);

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
        for idx, file in tqdm(enumerate(setfiles), ascii=True):
            if not self.feat_keep:
                inp1 = loadWAV(os.path.join(test_path,file), self.__max_frames__, evalmode=True, num_eval=1).cuda()
                ref_feat = self.__S__.forward(inp1).detach().cpu()
            filename = '%06d.wav'%idx
            if feat_dir == '':
                feats[file]     = [ref_feat,inp1]
            else:
                filedict[file]  = filename
                if not self.feat_keep:
                    torch.save(ref_feat,os.path.join(feat_dir,filename))
                telapsed = time.time() - tstart
        # if idx % print_interval == 0:
        #     sys.stdout.write("\rReading %d: %.2f Hz, embed size %d"%(idx,idx/telapsed,ref_feat.size()[1]));

        print('')
        all_scores = [];
        all_labels = [];
        tstart = time.time()

        ## Read files and compute all scores
        print("Computing!");
        for idx, line in tqdm(enumerate(lines), ascii=True):

            data = line.split();

            if feat_dir == '':
                ref_feat = feats[data[1]][0].cuda()
                com_feat = feats[data[2]][0].cuda()
                ori_ref = feats[data[1]][1].cuda()
                ori_com = feats[data[2]][1].cuda()
            else:
                ref_feat = torch.load(os.path.join(feat_dir,filedict[data[1]])).cuda()
                com_feat = torch.load(os.path.join(feat_dir,filedict[data[2]])).cuda()

            if self.__test_normalize__:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)
            feat = []
            ori_feat = []
            feat.append(ref_feat)
            feat.append(com_feat)
            ori_feat.append(ori_ref)
            ori_feat.append(ori_com)
            feat = torch.stack(feat,dim=1).squeeze()
            ori_feat = torch.stack(ori_feat,dim=1).squeeze()
            score = self.fine_tune_network.forward(feat, ori_feat, eval_mode=True).detach().cpu().numpy();
            # dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            # print(dist.size())
            # score = -1 * dist
            # score = -1 * numpy.mean(dist);

            all_scores.append(score);  
            all_labels.append(int(data[0]));

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                # sys.stdout.write("\rComputing %d: %.2f Hz"%(idx,idx/telapsed));
                # sys.stdout.flush();

        if feat_dir != '':
            if not self.feat_keep:
                print(' Deleting temporary files.')
                shutil.rmtree(feat_dir)

        print('\n')

        return (all_scores, all_labels);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Update learning rate
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def updateLearningRate(self, alpha):

        learning_rate = []
        for param_group in self.__optimizer__.param_groups:
            param_group['lr'] = param_group['lr']*alpha
            learning_rate.append(param_group['lr'])
        # print('learning_rate')
        # print(learning_rate)
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

        self_state = self.state_dict();
        loaded_state = torch.load(path);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);

    
