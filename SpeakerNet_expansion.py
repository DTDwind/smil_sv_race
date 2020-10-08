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

        # self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=40)


        argsdict = {'nOut': nOut, 'encoder_type':encoder_type}

        self.sigmoid = nn.Sigmoid()

        self.__test_normalize__     = True
        self.setfiles_global = ''
        self.feats_global = ''
        
        self.__max_frames__ = max_frames;
        self.feat_keep = False


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
        self.all_score_list = {}
        tstart      = time.time()

        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line): #  or (len(all_scores)==1000) 
                    break;
                data = line.split();

                files.append(data[1]) # for dev
                files.append(data[2]) 
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()

        ## Save all features to file
        print("Reading File");
        print("setfiles len: "+str(len(setfiles)))
        for idx, file in tqdm(enumerate(setfiles), ascii=True):
            filename = '%06d.wav'%idx

            # feat_dir = '/share/nas165/chengsam/voxceleb2020_prog/fully_supervised_speaker_verification/voxceleb_trainer/data/Dvector_test'
            feat_dir = '/share/nas165/chengsam/voxceleb2020_prog/fully_supervised_speaker_verification/voxceleb_trainer/data/Dvector_dev'

            if feat_dir == '':
                feats[file]     = ref_feat
            else:
                filedict[file]  = filename
                feats[file]     = torch.load(os.path.join(feat_dir, filename))

        print('Read all score')

        all_score_dir = '/share/nas165/chengsam/voxceleb2020_prog/fully_supervised_speaker_verification/voxceleb_trainer/resnet_all_pair_scroe/dev_score_base.txt' 
        with open(all_score_dir) as listfile:
            while True:
                line = listfile.readline();
                if (not line): #  or (len(all_scores)==1000) 
                    break;
                data = line.split();
                if data[1] in self.all_score_list:
                    self.all_score_list[data[1]].update({data[2]: data[0]})
                else:
                    self.all_score_list.update({data[1]:{data[2]: data[0]}})
        print('Read all score Done')
        all_scores = [];
        all_plda_scores = [];
        all_labels = [];
        tstart = time.time()

        ## Read files and compute all scores
        print("Computing!");
        self.feats_global = feats
        self.setfiles_global = setfiles

        # cpus = os.cpu_count() 
        # print("# of cpu: "+str(cpus))
        # p=mp.Pool(cpus)

        for idx, line in tqdm(enumerate(lines), ascii=True):
            data = line.split();
            score = float(self.all_score_list[data[1]][data[2]])
            score = self.score_reliable(score, data[1], data[2])
            all_scores.append(float(score));  
            all_labels.append(int(data[0]));

        # for _ in tqdm(p.istarmap(self.computing_thread, enumerate(lines)), total=len(lines)):pass
        # p.close()
        # p.join()

        # cpus = os.cpu_count() 
        # print("# of cpu: "+str(cpus))
        # p=mp.Pool(cpus)
        # for _ in tqdm(p.istarmap(self.thread_score, enumerate(setfiles)), total=len(setfiles)):pass
        # p.close()
        # p.join()

        print('\n')

        print(' Computing Done! ')
        return (all_scores, all_labels);
        
    def computing_thread(self, idx, line):
        data = line.split();
        score = float(self.all_score_list[data[1]][data[2]])
        score = self.score_reliable(score, data[1], data[2])
        self.all_scores.append(float(score));  
        self.all_labels.append(int(data[0]));

    def score_reliable(self, score, ref, com):

# 貼著第一個錯誤做
#         right               left
# 100%    -0.85387694835663   -1.37349641323089 
# 99.50%  -0.97980618476868   -1.06000936031341 
# 99%     -1.01144731044769   -1.02438473701477 
# 98.50%  -1.03028070926666   -1.00153410434722 
# 98%     -1.04567658901214   -0.98936873674393 
# 97%     -1.06901240348815   -0.97086042165756 
# 96%     -1.08683192729949   -0.95614105463028 
# 95%     -1.10199260711669   -0.94331932067871 


        th = -0.85387694835663 # 100%
        # th = -0.97980618476868 # 99.50%
        # th = -1.01144731044769 # 99%
        # th = -1.03028070926666 # 98.50%
        # th = -1.04567658901214 # 98%
        low_th = -1.37349641323089 # 100%
        # low_th = -0.94331932067871 # 95%
        ori_eer_th = -1.0168 # 原始EER判斷正確與否的域值
        new_score = score
        # print(score)
        if 1 > score > -10000000 :
        # if th > score > ori_eer_th:

            # k={'Apple':400,'Joey':600,'Bella':50,'2th King':10000}
            # print(sorted(k.items(), key=lambda x:x[1]))
            # output:[('Bella', 50), ('Apple', 400), ('Joey', 600), ('2th King', 10000)]

            sort_list = sorted(self.all_score_list[ref].items(), key=lambda x:x[1], reverse=True)
            ref_feat = self.feats_global[ref]
            com_feat = self.feats_global[com]
            # ref_feat = F.normalize(ref_feat, p=2, dim=1)
            # com_feat = F.normalize(com_feat, p=2, dim=1)
            counter = 0
            top_n = 20
            dv = []
            dn = []
            for doc in sort_list:
                counter += 1
                if counter < top_n+1:
                    v = self.feats_global[doc[0]]
                    # v = F.normalize(v, p=2, dim=1)
                    dv.append(v)
                else:
                    v = self.feats_global[doc[0]]
                    # v = F.normalize(v, p=2, dim=1)
                    dn.append(v)
            dv = torch.stack(dv, dim=0)
            dn = torch.stack(dn, dim=0)
            # new_q = qv + 0.75*np.sum(dv,axis=0)/self.D - 0.15*np.sum(dv,axis=0)/len(Dn_doc)

            new_ref_feat = 1*ref_feat + 0.005*(dv.sum(dim = 0))/top_n - 0.0015*(dn.sum(dim = 0))/(len(sort_list)-top_n)
            # new_ref_feat = F.normalize(new_ref_feat, p=2, dim=1)


            # max_like = max(self.all_score_list[ref], key = self.all_score_list[ref].get)
            # min_like = min(self.all_score_list[ref], key = self.all_score_list[ref].get)
            # max_like_com = max(self.all_score_list[com], key = self.all_score_list[com].get)
            # min_like_com = min(self.all_score_list[com], key = self.all_score_list[com].get)
            # ref_feat = self.feats_global[ref]
            # com_feat = self.feats_global[com]
            # TF_feat = self.feats_global[max_like]
            # IDF_feat = self.feats_global[min_like]
            # TF_com_feat = self.feats_global[max_like_com]
            # IDF_com_feat = self.feats_global[min_like_com]
            # if self.__test_normalize__:
            #     ref_feat = F.normalize(ref_feat, p=2, dim=1)
            #     com_feat = F.normalize(com_feat, p=2, dim=1)
            #     TF_feat = F.normalize(TF_feat, p=2, dim=1)
            #     IDF_feat = F.normalize(IDF_feat, p=2, dim=1)
            #     TF_com_feat = F.normalize(TF_com_feat, p=2, dim=1)
            #     IDF_com_feat = F.normalize(IDF_com_feat, p=2, dim=1)




            # print(ori_feat.size())
            # a = 1
            # new_ref_feat = (ref_feat + 0.75*TF_feat - 0.15*IDF_feat)
            # new_com_feat = (com_feat + 0.75*TF_com_feat - 0.15*IDF_com_feat)
            # new_feat = ref_feat + 0.075*TF_feat 

            # dist = F.pairwise_distance(new_ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).numpy();
            # dist = F.pairwise_distance(new_ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            dist = F.cosine_similarity(new_ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).numpy();
            
            new_score = 1 * numpy.mean(dist);




        #     # print(new_score)
        #     # quit()
        #     # self.feats_global
        #     # a = 0.8
        #     # new_score = a*score + (1-a)*new_score



        # ref_feat = self.feats_global[ref]
        # com_feat = self.feats_global[com]
        # # com_feat = F.normalize(com_feat, p=2, dim=1)
        # # ref_feat = F.normalize(ref_feat, p=2, dim=1)
        
        # # dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).numpy();
        # dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).numpy();
        # new_score = 1 * numpy.mean(dist);

        return new_score

    def thread_score(self, idx, file_name):
        ref_file = file_name
        ref_idx = idx
        
        if idx < 50000: return # sota
        with open('/home/chengsam/sota_test_score/test_score_'+str(idx)+'.txt', 'w') as out:
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

