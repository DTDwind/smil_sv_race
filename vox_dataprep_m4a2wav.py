#!/usr/bin/python
#-*- coding: utf-8 -*-
# Requirement: ffmpeg running on a Linux system.

import argparse
import os
import subprocess
import pdb
import hashlib
import time
import glob
from zipfile import ZipFile
from tqdm import tqdm

## ========== Parse input arguments ===========
parser = argparse.ArgumentParser(description = "VoxCeleb downloader");
parser.add_argument('--convert',  dest='convert',  action='store_true', help='Enable convert')
args = parser.parse_args();

## ========== Convert ===========
def convert(args):
    print('Converting files from AAC to WAV')
    with open('data_rounter.txt', encoding='utf8') as f: # 讀取檔案
        # local = '/share/nas165/chengsam/vox2/voxceleb2/'
        for fname in tqdm(f.readlines(), ascii=True): # 逐行讀入避免OOM
            fname = fname.replace('\n','')
            outfile = fname.replace('.m4a','.wav')
            out = subprocess.call('ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' %(fname,outfile), shell=True)
            # out = subprocess.call('pwd', shell=True)
            if out != 0:
                raise ValueError('Conversion failed %s.'%fname)

## ========== Main script ===========
if __name__ == "__main__":
    if args.convert:
        convert(args)
        
