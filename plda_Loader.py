import numpy as np
import argparse

class PLDALoader():
    def __init__(self):
        self.score_path = ''
        self.plad_dict = {}
    
    def get_parser(self):
        parser = argparse.ArgumentParser()

        ## Required parameters
        parser.add_argument("--score",
                            default="plda_dev_score",
                            type=str)

        args = parser.parse_args()
        self.score_path = args.score

    def read_score(self, path=None):
        if path == None: path = self.score_path
        with open(path) as lines:
            for line in lines:
                utt = line.split(' ')
                idx = utt[0]+'-'+utt[1]
                self.plad_dict[idx] = utt[2].replace('\n','')

    def get_plda_score(self, idx):
        return float(self.plad_dict[idx])

    def process(self):
        self.read_score()


if __name__ == '__main__':
    xvl = PLDALoader()
    xvl.get_parser()
    xvl.process()