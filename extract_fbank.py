#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy
import random
import pdb
import os
import threading
import time
import math
from scipy.io import wavfile
from queue import Queue
from models.extract_fbank_model import *
from tqdm import tqdm

file_list_path = 'data_list/vox2020Baseline/train_list.txt'
train_path = '/share/nas165/chengsam/vox2/voxceleb2_dev/aac'
max_frames = 200

def round_down(num, divisor):
    return num - (num%divisor)

def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    sample_rate, audio  = wavfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = math.floor( ( max_audio - audiosize + 1 ) / 2 )
        audio       = numpy.pad(audio, (shortage, shortage), 'constant', constant_values=0)
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = numpy.stack(feats,axis=0)

    feat = torch.FloatTensor(feat)

    return feat;

class DatasetLoader(object):
    def __init__(self, dataset_file_name, max_frames, train_path):
        self.dataset_file_name = dataset_file_name;
        self.max_frames = max_frames;
        self.data_dict = {};
        self.data_list = [];
        self.dataLoaders = [];
        
        # QwQ
        self.__S__ = Ex_Fbank().cuda();

        ### Read Training Files...
        with open(dataset_file_name) as dataset_file:
            while True:
                line = dataset_file.readline();
                if not line:
                    break;
                
                data = line.split();
                speaker_name = data[0];
                filename = os.path.join(train_path,data[1]);

                if not (speaker_name in self.data_dict):
                    self.data_dict[speaker_name] = [];

                self.data_dict[speaker_name].append(filename);

        ### Initialize Workers...

    def extract_feat(self):
        print(self.data_dict.keys())
        for i in tqdm(self.data_dict, ascii=True):
            # print(i)
            for j in self.data_dict[i]:
                # print(j)
                wav_feat = loadWAV(j, self.max_frames, evalmode=False)
                new_path = j.replace('/aac','/fbank_feat')
                new_path = new_path.replace('.wav','.feat.pt')
                outp     = self.__S__.forward(wav_feat.cuda())
                # print(new_path)
                torch.save(outp, new_path)

if __name__ == '__main__': 
    extract_worker = DatasetLoader(file_list_path, max_frames, train_path)
    extract_worker.extract_feat()
