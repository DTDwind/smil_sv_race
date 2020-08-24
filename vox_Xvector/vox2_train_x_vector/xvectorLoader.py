import kaldiio
import numpy as np
import argparse

class XvectorLoader():

    def __init__(self):
        self.scp_path = ''
        self.ark_path = ''
        self.save_path = ''

        self.scp_dict = {}
        self.xvector_dict = {}
    
    def get_parser(self):
        parser = argparse.ArgumentParser()

        ## Required parameters
        parser.add_argument("--scp",
                            default="spk_xvector.scp",
                            type=str)

        parser.add_argument("--ark",
                            default=".",
                            type=str)

        parser.add_argument("--save_path",
                            default='.',
                            type=str)

        args = parser.parse_args()

        self.scp_path = args.scp
        self.ark_path = args.ark
        self.save_path = args.save_path

    def read_scp(self, path=None):
        if path == None: path = self.scp_path
        with open(path) as lines:
            for line in lines:
                utt = line.split(' ')
                idx = utt[0]
                ark = utt[1].split('/')[-1] # ex: spk_xvector.ark:8
                self.scp_dict[idx] = ark.replace('\n','')

    def read_ark(self, path=None):
        if path == None: path = self.ark_path
        for idx in self.scp_dict:
            # with kaldiio.ReadHelper('ark:'+self.scp_dict[idx]) as reader:
            #     for key, numpy_array in reader:
            #         print("key: %s , numpy_array: %s \n"%(key, numpy_array))
            #         print(numpy_array.shape)
            #         exit()
            numpy_array = kaldiio.load_mat(path+'/'+self.scp_dict[idx])
            self.xvector_dict[idx] = numpy_array

    def get_xvector(self, idx):
        return self.xvector_dict[idx]

    def process(self):
        self.read_scp()
        self.read_ark()


if __name__ == '__main__':
    xvl = XvectorLoader()
    xvl.get_parser()
    xvl.process()
