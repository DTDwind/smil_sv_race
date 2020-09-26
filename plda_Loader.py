import numpy as np
import argparse

class XvectorLoader():
    def __init__(self):
        self.score_path = ''

        self.scp_dict = {}
        self.xvector_dict = {}
    
    def get_parser(self):
        parser = argparse.ArgumentParser()

        ## Required parameters
        parser.add_argument("--score",
                            default="spk_xvector.scp",
                            type=str)

        args = parser.parse_args()

        self.score_path = args.score

    def read_scp(self, path=None):
        if path == None: path = self.scp_path
        with open(path) as lines:
            for line in lines:
                utt = line.split(' ')
                idx = utt[0]
                ark = utt[1].split('/')[-1] # ex: spk_xvector.ark:8
                self.scp_dict[idx] = ark.replace('\n','')

    def get_xvector(self, idx):
        return self.xvector_dict[idx]

    def process(self):
        self.read_scp()


if __name__ == '__main__':
    xvl = XvectorLoader()
    xvl.get_parser()
    xvl.process()