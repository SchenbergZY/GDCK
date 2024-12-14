import sys
from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from tester_other_arcs import Evaluator
from utils_graphsaint import DataGraphSAINT


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--keep_ratio', type=float, default=1)
parser.add_argument('--reduction_rate', type=float, default=1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--epsilon', type=float, default=-1)
parser.add_argument('--nruns', type=int, default=20)
parser.add_argument('--mt', type=str, default='GCN')
parser.add_argument('--test_lr', type=float, default=0.001)
parser.add_argument('--test_wd', type=float, default=5e-4)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(300008)#args.seed)
np.random.seed(300008)#args.seed)
torch.manual_seed(300008)#args.seed)
torch.cuda.manual_seed(300008)#args.seed)

if args.dataset in ['cora', 'citeseer']:
    args.epsilon = args.epsilon#-1# 0.05
else:
    args.epsilon = args.epsilon#-1# 0.01

print(args)

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)

agent = Evaluator(data, args, device='cuda')
agent.train()
