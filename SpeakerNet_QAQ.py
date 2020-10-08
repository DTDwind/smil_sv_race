#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader_new import loadWAV
# from DatasetFeatLoader import loadFeat
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

class SpeakerNet(nn.Module):

    def __init__(self, max_frames, lr = 0.0001, margin = 1, scale = 1, hard_rank = 0, hard_prob = 0, model="alexnet50", nOut = 512, nSpeakers = 1000, optimizer = 'adam', encoder_type = 'SAP', normalize = True, trainfunc='contrastive', **kwargs):
        super(SpeakerNet, self).__init__();

        argsdict = {'nOut': nOut, 'encoder_type':encoder_type}

        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__(model)
        self.__S__ = SpeakerNetModel(**argsdict).cuda();
        # self.fc = th_Fc(nOut ,nOut).cuda()
        # self.lstm = rnn_LSTM(nOut, nOut).cuda();
        # self.fine_tune_DNN = th_Fc(nOut, nOut).cuda();


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
                                                #    {'params':self.fine_tune_DNN.parameters(),'lr':0.001},
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
            # print("data:"+str(data))
            
            # print("data len:"+str(len(data)))
            # print("data[0] size:"+str(data[0].size()))
            # print("data_label:"+str(data_label))
            # print("data_label len:"+str(len(data_label)))
            # exit()
            tstart = time.time()

            self.zero_grad();
            # print(data_label)
            # exit()
            feat = []
            
            for inp in data:
                # print('inp')
                # print(inp)
                # # print(inp.size())
                # exit()
                outp      = self.__S__.forward(inp.cuda())
                # outp = self.lstm.forward(outp.cuda())
                # outp = self.fine_tune_DNN.forward(outp.cuda())
                
                # print('outp')
                # print(outp)
                # print('outp size')
                # print(outp.size())
                
                # exit()

                if self.__train_normalize__:
                    outp   = F.normalize(outp, p=2, dim=1)
                feat.append(outp)

            feat = torch.stack(feat,dim=1).squeeze()
            
            label   = torch.LongTensor(data_label).cuda()
            # print('__L__ feat')
            # print(feat.size()) # --- [400, 2, 512]
            # exit()
            nloss, prec1 = self.__L__.forward(feat,label)

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
                inp1 = loadWAV(os.path.join(test_path,file), self.__max_frames__, evalmode=True, num_eval=num_eval).cuda()
            
                # inp1 = loadFeat(os.path.join(test_path,file), self.__max_frames__, evalmode=True, num_eval=num_eval).cuda()
                
                # ref_feat = self.__S__.forward(inp1).detach().cpu()
                # ref_feat = self.lstm.forward(ref_feat).detach().cpu() 
                ref_feat = self.__S__.forward(inp1).detach().cpu()
                # ref_feat = self.fine_tune_DNN.forward(ref_feat).detach().cpu()
                # print(inp1.size())
                # print('inp1_feat')
                # exit()
            filename = '%06d.wav'%idx
            # filename = '%06d.feat.pt'%idx
            # print('QAQ')
            # exit()
            if feat_dir == '':
                feats[file]     = ref_feat
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
        counter = 0 
        for idx, line in tqdm(enumerate(lines), ascii=True):
            counter += 1
            data = line.split();

            if feat_dir == '':
                ref_feat = feats[data[1]].cuda()
                com_feat = feats[data[2]].cuda()
            else:
                ref_feat = torch.load(os.path.join(feat_dir,filedict[data[1]])).cuda()
                com_feat = torch.load(os.path.join(feat_dir,filedict[data[2]])).cuda()

            if self.__test_normalize__:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)
            print(ref_feat.unsqueeze(-1).size()) # torch.Size([10, 512, 1])
            print(com_feat.unsqueeze(-1).transpose(0,2).size()) # torch.Size([1, 512, 10])
            quit()

            # if counter == 1:
            #     a = ref_feat
            #     b = com_feat
            #     continue
            # c = ref_feat
            # d = com_feat

            
            
            # dist = F.pairwise_distance(ref_feat.unsqueeze(-1).expand(-1,-1,num_eval), com_feat.unsqueeze(-1).expand(-1,-1,num_eval).transpose(0,2)).detach().cpu().numpy();
            # EER 2.3436



            dist = F.pdist(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            

            # dist1 = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            # dist2 = F.pairwise_distance(a.unsqueeze(-1), b.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            # feat = torch.stack(feat,dim=1).squeeze()
            # e = torch.stack([a,c],dim=1).squeeze()
            # f = torch.stack([b,d],dim=1).squeeze()
            # print(e.size())
            
            # score1 = -1 * numpy.mean(dist1);
            # score2 = -1 * numpy.mean(dist2);
            # print(score1)
            # print(score2)
            # dist3 = F.pairwise_distance(e.unsqueeze(-1), f.unsqueeze(-1).transpose(0,2));
            # print(dist3.size())
            # quit()

            score = -1 * numpy.mean(dist);

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

