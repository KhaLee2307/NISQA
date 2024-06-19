import yaml
import random
import argparse

import numpy as np

import torch
import torch.backends.cudnn as cudnn

from nisqa.NISQA_model import nisqaModel


parser = argparse.ArgumentParser()
parser.add_argument('--yaml', required=True, type=str, help='YAML file with config')

args = parser.parse_args()
args = vars(args)

if __name__ == "__main__":
    
    with open(args['yaml'], "r") as ymlfile:
        args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = {**args_yaml, **args}

    """ Seed and GPU setting """   
    random.seed(111)
    np.random.seed(111)
    torch.manual_seed(111)
    torch.cuda.manual_seed(111)

    cudnn.benchmark = True  # it fasten training.
    cudnn.deterministic = True
    
    nisqa = nisqaModel(args)
    nisqa.semi()