import os
default_n_threads=1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
from time import gmtime, strftime
import json
import argparse

import numpy as onp
import scipy as sp

import gc
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Sized, Tuple, Type, TypeVar, Union

import torch
from utils import get_dataset, Transd2Ind

from utils_graphsaint import DataGraphSAINT
from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor,to_tensor,is_sparse_tensor,normalize_adj_tensor
from utils_gdck import *

parser = argparse.ArgumentParser(description="Hyperparameters")
parser.add_argument('--architecture', type=str, default='FC', help="param ['FC', 'Conv', 'Myrtle']; choice of neural network architecture yielding the corresponding NTK")
parser.add_argument('--parameterization', type=str, default='ntk', help="param ['ntk', 'standard']; whether to use standard or NTK parameterization, see https://arxiv.org/abs/2001.07301")
parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help="param ['ogbn-products', 'ogbn-proteins', 'ogbn-arxiv', 'ogbn-papers100M','ogbn-mag']")
parser.add_argument('--lr', type=float, default=4e-2, help="param {'type': float}, learning rate")
parser.add_argument('--lr-train-feat', type=float, default=4e-2, help="param {'type': float}, learning rate feat")
parser.add_argument('--lr-train-adj', type=float, default=4e-2, help="param {'type': float}, learning rate adjacent matrix")
parser.add_argument('--lr-train-y', type=float, default=4e-2, help="param {'type': float}, learning rate y")
parser.add_argument('--support-size', type=int, default=120, help="param {'type': int}; number of nodes to learn")
parser.add_argument('--target-batch-size', type=int, default=6000, help="param {'type': int}; number of target nodes to use in KRR for each step")
parser.add_argument('--a-strategy', type=str, default='Continuous', help="param ['Keep_init', 'Continuous']")
parser.add_argument('--ntk-layers', type=int, default=2, help="ntk layers")
parser.add_argument('--optimizer', type=str, default='adam', help="use adam or sgd")
parser.add_argument("--is-em-strategy", default=False, action="store_true", help="use changed sigmoid for adjacent matrix")
parser.add_argument('--em-strategy-epochs', type=int, default=1, help="ntk layers")
parser.add_argument("--is-kip-valid", default=False, action="store_true", help="use kip loss to valid")
parser.add_argument("--is-regular-adj-syn", default=False, action="store_true", help="use regular loss or not")
parser.add_argument('--beta', type=float, default=0.1, help="param {'type': float}, regular term for adj")
parser.add_argument('--hop', type=int, default=1, help="param {'type': int}; depth of multi-hop")
parser.add_argument('--hop-coefficient', type=float, default=1.0, help="param {'type': float}; how much if a deamplification of different hop parameter")
parser.add_argument('--hop-type', type=str, default='reverse', help="param ['exp', 'reverse']")
parser.add_argument("--is-test-hop", default=False, action="store_true", help="use hop in testing")
parser.add_argument('--loss-coeff', type=float, default=1.0, help="param {'type': float}; loss coefficient of KIP loss")
parser.add_argument('--init-way', type=str, default='none', help="param ['none', 'Center', 'K-Center', 'Random_real', 'K-means']; as y rearranged which way to init syn data")
parser.add_argument("--is-smooth-adj-syn", default=False, action="store_true", help="use smooth loss or not")
parser.add_argument('--gamma', type=float, default=0.1, help="param {'type': float}, smooth term for adj")
parser.add_argument("--is-use-layer-norm", default=False, action="store_true", help="use layernorm in ntk or not")
parser.add_argument("--is-with-bn", default=False, action="store_true", help="use bn in val_with_val or not")
parser.add_argument("--is-use-clscore-sort", default=False, action="store_true", help="use clscore to sort nodes not")
parser.add_argument('--clscore-type', type=str, default='exp', help="use exp or addone to manage clscore raw")
parser.add_argument("--is-regular-kss", default=False, action="store_true", help="use regular kss or not")
parser.add_argument('--coreset-fix-seed', type=int, default=-1, help="-1 means seed like args.seed, else fix a seed for coreseting")
parser.add_argument("--is-kss-class-fix", default=False, action="store_true", help="use fix class kss loss or not")


parser.add_argument("--is-doscond-sigmoid", default=False, action="store_true", help="use changed sigmoid for adjacent matrix")
parser.add_argument("--is-sparse-adj", default=False, action="store_true", help="is adj sparse or not")
parser.add_argument('--a-init-multiple-number', type=float, default=1.e8, help="param {'type': float} zooming omega to large number")
parser.add_argument('--y-init-multiple-number', type=float, default=1.e8, help="param {'type': float} zooming omega to large number")
parser.add_argument("--is-relu-adj", default=False, action="store_true", help="use relu for adjacent matrix")
parser.add_argument("--is-none-adj", default=False, action="store_true", help="use none sigmoid for adjacent matrix")
parser.add_argument("--is-relu-y", default=False, action="store_true", help="use relu for y support")
parser.add_argument("--is-kcenter-sfgc", default=False, action="store_true", help="use kcenter in sfgc or not")
parser.add_argument("--is-not-clip-y", default=False, action="store_true", help="use clip in y or not")
parser.add_argument('--adding-y', type=float, default=0., help="param {'type': float} add y during training")
parser.add_argument('--temp-init', type=float, default=1., help="param {'type': float} initial temperature")
parser.add_argument('--temp-end', type=float, default=1., help="param {'type': float} end temperature")

parser.add_argument("--keep-batch-edge", default=False, action="store_true", help="param {'type': bool}; whether to take parents outside the batch")
parser.add_argument("--is-testing", default=False, action="store_true", help="param {'type': bool}; testing or not")
parser.add_argument("--is-accumulate", default=False, action="store_true", help="param {'type': bool}; jump knowledge or not")
parser.add_argument('--reg', type=float, default=128.e-4, help="param {'type': float} regularization paremeter for matrix inverse")
parser.add_argument('--epochs', type=int, default=150, help="param {'type': int}; epochs")
parser.add_argument('--seed', type=int, default=8, help="param {'type': int}; seed")
parser.add_argument('--add-seed', type=int, default=0, help="param {'type': int}; add seed")
parser.add_argument("--is-x-fix", default=False, action="store_true", help="if x is fix or not")
parser.add_argument("--is-y-fix", default=False, action="store_true", help="if y is fix or not, which is the same as learn label")
parser.add_argument("--is-a-fix", default=False, action="store_true", help="if a is fix or not")
parser.add_argument('--weight-decay', type=float, default=0., help="param {'type': float} weight decay parameter")
parser.add_argument("--is-a-target-sum", default=False, action="store_true", help="if a sum is used")
parser.add_argument("--is-rearranged-y", default=False, action="store_true", help="if y rearranged like GCond")

parser.add_argument("--is-delete-min", default=False, action="store_true", help="if min of init data is deleted")
parser.add_argument("--is-return-last", default=False, action="store_true", help="if return last of argmin or argmax")

parser.add_argument("--is-largest-a-target-sum-init", default=False, action="store_true", help="if a target sum sorting is used when init")
parser.add_argument("--is-gcond-valid", default=False, action="store_true", help="use GCond validation and testing")
parser.add_argument("--is-y-pred-softmax", default=False, action="store_true", help="use softmax for y pred")
parser.add_argument("--is-torch-one-hot", default=False, action="store_true", help="use torch_one_hot or paper origin one hot")
parser.add_argument("--is-lambda-trainable", default=False, action="store_true", help="set ridge lambda tranable or fix-tuned")
parser.add_argument("--is-class-coefficient", default=False, action="store_true", help="use support vector class-balanced coefficient or not")
parser.add_argument("--is-warm-up-cosine", default=False, action="store_true", help="use warm up cosine or not")
parser.add_argument("--is-warm-up-linear", default=False, action="store_true", help="use warm up linear or not")
parser.add_argument("--is-warm-up-constant", default=False, action="store_true", help="use warm up constant or not")
parser.add_argument("--is-kl-div", default=False, action="store_true", help="use dkl distance instead of mse")
parser.add_argument("--is-mean-loss", default=False, action="store_true", help="use mean in loss or not")


# gcond parser
parser.add_argument('--gpu_id', type=int, default=3, help='gpu id')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
#parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=1)
#parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--one_step', type=int, default=0)
parser.add_argument('--test-lr', type=float, default=0.01, help='testing learning rate')
parser.add_argument('--test-wd', type=float, default=0.0, help='testing weight decay')
args, unknown = parser.parse_known_args()

if int(args.is_doscond_sigmoid) + int(args.is_relu_adj) + int(args.is_none_adj)>1:
    raise ValueError('Too many activations for adj.')

if os.path.isfile('is_gpu_clusters.txt'): 
    print('On cluster')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("RUNNING ON: ",device)
args.temp = args.temp_init


# training params 
if args.is_x_fix and rgs.is_y_fix and args.is_a_fix:
    raise ValueError('You train nothing!')

if args.dataset == 'citeseer' and (args.support_size ==120):
    args.nlayers = 2
    args.sgc = 1
    args.lr_feat = args.test_lr #0.01
    args.lr_adj = args.test_lr #0.01
    args.reduction_rate = 1.
    args.inner = 3
    args.fit_with_val_epochs = 600

if args.dataset == 'citeseer' and (args.support_size ==60):
    args.nlayers = 2
    args.sgc = 1
    args.lr_feat = args.test_lr #0.01
    args.lr_adj = args.test_lr #0.01
    args.reduction_rate = 1.
    args.inner = 3
    args.fit_with_val_epochs = 600

if args.dataset == 'citeseer' and (args.support_size ==30):
    args.nlayers = 2
    args.sgc = 1
    args.lr_feat = args.test_lr #0.01
    args.lr_adj = args.test_lr #0.01
    args.reduction_rate = 1.
    args.inner = 3
    args.fit_with_val_epochs = 600

if args.dataset == 'cora' and (args.support_size ==140):
    args.nlayers = 2
    args.sgc = 1
    args.lr_feat = args.test_lr #0.01
    args.lr_adj = args.test_lr #0.01
    args.reduction_rate = 1.
    args.inner = 3
    args.fit_with_val_epochs = 600

if args.dataset == 'cora' and (args.support_size ==70):
    args.nlayers = 2
    args.sgc = 1
    args.lr_feat = args.test_lr #0.01
    args.lr_adj = args.test_lr #0.01
    args.reduction_rate = 1.
    args.inner = 3
    args.fit_with_val_epochs = 600

if args.dataset == 'cora' and (args.support_size ==35):
    args.nlayers = 2
    args.sgc = 1
    args.lr_feat = args.test_lr #0.01
    args.lr_adj = args.test_lr #0.01
    args.reduction_rate = 1.
    args.inner = 3
    args.fit_with_val_epochs = 600

if args.dataset == 'ogbn-arxiv' and (args.support_size ==909):
    args.nlayers = 2
    args.sgc = 1
    args.lr_feat = args.test_lr #0.01
    args.lr_adj = args.test_lr #0.01
    args.reduction_rate = 0.01
    args.inner = 3
    args.fit_with_val_epochs = 600
    if args.is_kcenter_sfgc:
        if args.support_size == 909:
            kcenter_ind_init = onp.load('coreset_sfgc/idx_ogbn-arxiv_0.01_kcenter_15.npy') # idx_ogbn-arxiv_0.01_kcenter_15_from_paper.npy
        elif args.support_size == 454:
            kcenter_ind_init = onp.load('coreset_sfgc/idx_ogbn-arxiv_0.005_kcenter_15.npy')
        elif args.support_size == 90:
            kcenter_ind_init = onp.load('coreset_sfgc/idx_ogbn-arxiv_0.001_kcenter_15.npy')
        else:
            raise NotImplementedError

if args.dataset == 'ogbn-arxiv' and (args.support_size ==454):                         
    args.nlayers = 2                                        
    args.sgc = 1                                            
    args.lr_feat = args.test_lr #0.001                                                                              
    args.lr_adj = args.test_lr  #0.001                                 
    args.reduction_rate = 0.005
    args.inner = 3
    args.fit_with_val_epochs = 300
    if args.is_kcenter_sfgc:                                    
        if args.support_size == 909:                                
            kcenter_ind_init = onp.load('coreset_sfgc/idx_ogbn-arxiv_0.01_kcenter_15.npy') # idx_ogbn-arxiv_0.01_kcenter_15_from_paper.npy
        elif args.support_size == 454:                              
            kcenter_ind_init = onp.load('coreset_sfgc/idx_ogbn-arxiv_0.005_kcenter_15.npy')                             
        elif args.support_size == 90:                               
            kcenter_ind_init = onp.load('coreset_sfgc/idx_ogbn-arxiv_0.001_kcenter_15.npy')
        else:                                                       
            raise NotImplementedError

if args.dataset == 'ogbn-arxiv' and (args.support_size ==90):                            
    args.nlayers = 2
    args.sgc = 1
    args.lr_feat = args.test_lr #0.001
    args.lr_adj = args.test_lr # 0.001
    args.reduction_rate = 0.001
    args.inner = 3
    args.fit_with_val_epochs = 1000 
    if args.is_kcenter_sfgc:
        if args.support_size == 909:
            kcenter_ind_init = onp.load('coreset_sfgc/idx_ogbn-arxiv_0.01_kcenter_15.npy') # idx_ogbn-arxiv_0.01_kcenter_15_from_paper.npy
        elif args.support_size == 454:
            kcenter_ind_init = onp.load('coreset_sfgc/idx_ogbn-arxiv_0.005_kcenter_15.npy')
        elif args.support_size == 90:
            kcenter_ind_init = onp.load('coreset_sfgc/idx_ogbn-arxiv_0.001_kcenter_15.npy')
        else:
            raise NotImplementedError

if args.dataset == 'flickr':
    args.nlayers = 2
    args.sgc = 2
    args.lr_feat = args.test_lr #0.005
    args.lr_adj = args.test_lr #0.005
    args.reduction_rate = 0.01
    args.inner = 1
    args.fit_with_val_epochs = 600
    if args.is_kcenter_sfgc:
        if args.support_size == 446:
            kcenter_ind_init = onp.load('coreset_sfgc/idx_flickr_0.01_kcenter_15.npy') # idx_ogbn-arxiv_0.01_kcenter_15.npy
        elif args.support_size == 223:
            kcenter_ind_init = onp.load('coreset_sfgc/idx_flickr_0.005_kcenter_15.npy')
        elif args.support_size == 44:
            kcenter_ind_init = onp.load('coreset_sfgc/idx_flickr_0.001_kcenter_15.npy')
        else:
            raise NotImplementedError

if args.dataset == 'reddit':
    args.nlayers = 2
    args.sgc = 1
    args.lr_feat = args.test_lr #0.1
    args.lr_adj = args.test_lr #0.1
    args.reduction_rate = 0.01
    args.inner = 1
    args.fit_with_val_epochs = 400
    if args.is_kcenter_sfgc:
        if args.support_size == 769:
            kcenter_ind_init = onp.load('coreset_sfgc/idx_reddit_0.005_kcenter_15.npy') # idx_ogbn-arxiv_0.01_kcenter_15.npy
        elif args.support_size == 307:
            kcenter_ind_init = onp.load('coreset_sfgc/idx_reddit_0.002_kcenter_15.npy')
        elif args.support_size == 153:
            kcenter_ind_init = onp.load('coreset_sfgc/idx_reddit_0.001_kcenter_15.npy')
        elif args.support_size == 76:
            kcenter_ind_init = onp.load('coreset_sfgc/idx_reddit_0.0005_kcenter_15.npy')
        else:
            raise NotImplementedError

print(args)

# time save module
current_time_str = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
save_score_name = 'GPU'+str(args.gpu_id)+'_'+current_time_str+'.txt'
FILE = open(save_score_name,"w")
json.dump(args.__dict__, FILE)
FILE.close()
with open(save_score_name, 'r') as original: tx = original.read()
with open(save_score_name, 'w') as modified: modified.write(tx+"\n")

# set device after json dump
args.device = device

# data loading
data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)

# get sum of all parent edges
A_ALL = sparse_mx_to_torch_sparse_tensor(data.adj_full.astype(onp.int8))
A_SUM_All = A_ALL.sum(0).to_dense()
A_TRAIN = data.adj_train.astype(onp.int8).toarray() # torch.index_select(torch.index_select(A_ALL, 0, train_idx), 1, train_idx).numpy()
A_VALID = data.adj_val.astype(onp.int8).toarray() # torch.index_select(torch.index_select(A_ALL, 0, valid_idx), 1, valid_idx).numpy()
A_TEST = data.adj_test.astype(onp.int8).toarray() # torch.index_select(torch.index_select(A_ALL, 0, test_idx), 1, test_idx).numpy()
A_SUM_TRAIN = torch.index_select(A_SUM_All, 0, torch.tensor(data.idx_train)) # A_SUM_TRAIN = torch.index_select(A_SUM_All, 0, train_idx)
A_SUM_VALID = torch.index_select(A_SUM_All, 0, torch.tensor(data.idx_val)) # A_SUM_VALID = torch.index_select(A_SUM_All, 0, valid_idx)
A_SUM_TEST = torch.index_select(A_SUM_All, 0, torch.tensor(data.idx_test)) # A_SUM_TEST = torch.index_select(A_SUM_All, 0, test_idx)
print(A_ALL.shape,A_TRAIN.shape,A_VALID.shape,A_TEST.shape)

# weights
if args.is_use_clscore_sort:
    difficult_scores_raw = do_geom_sort_nodes(data,args).cpu().numpy()
    if args.clscore_type == "exp":
        difficult_scores_raw =  onp.exp(difficult_scores_raw)
    elif args.clscore_type == 'addone':
        difficult_scores_raw = difficult_scores_raw+1
    elif args.clscore_type == 'sqrt':
        difficult_scores_raw = onp.sqrt(difficult_scores_raw+1)
    else:
        raise NotImplementedError
    args.difficult_scores_total = float(difficult_scores_raw.sum())
    if args.dataset in ['flickr','reddit']:
        difficult_scores_final = difficult_scores_raw[:,None]
    else:
        difficult_scores_final = onp.zeros(data.labels_full.shape)
        difficult_scores_final[data.idx_train] = difficult_scores_raw
        difficult_scores_final = difficult_scores_final[:,None]

X_TRAIN, X_VALID, X_TEST = data.feat_train, data.feat_val, data.feat_test # graph.x[train_idx].numpy(),graph.x[valid_idx].numpy(),graph.x[test_idx].numpy()
LABELS_TRAIN,LABELS_VALID, LABELS_TEST = data.labels_train, data.labels_val, data.labels_test # graph.y[train_idx].numpy().flatten(),graph.y[valid_idx].numpy().flatten(),graph.y[test_idx].numpy().flatten()

if args.is_torch_one_hot:
    Y_ALL = torch.nn.functional.one_hot(torch.tensor(data.labels_full).to(torch.int64),len(onp.unique(data.labels_full))).to(torch.int32).numpy()
    Y_TRAIN, Y_VALID, Y_TEST = torch.nn.functional.one_hot(torch.tensor(LABELS_TRAIN).to(torch.int64),len(onp.unique(LABELS_TRAIN))).to(torch.int32).numpy(), torch.nn.functional.one_hot(torch.tensor(LABELS_VALID).to(torch.int64),len(onp.unique(LABELS_TRAIN))).to(torch.int32).numpy(), torch.nn.functional.one_hot(torch.tensor(LABELS_TEST).to(torch.int64),len(onp.unique(LABELS_TRAIN))).to(torch.int32).numpy()
    if args.adding_y>0:
        Y_TRAIN[Y_TRAIN==1.] == 1.+args.adding_y
else:
    Y_TRAIN, Y_VALID, Y_TEST = one_hot(LABELS_TRAIN,len(onp.unique(LABELS_TRAIN))), one_hot(LABELS_VALID,len(onp.unique(LABELS_TRAIN))), one_hot(LABELS_TEST,len(onp.unique(LABELS_TRAIN)))

if args.dataset not in ['flickr','reddit']:
    if args.target_batch_size>data.feat_full.shape[0]:
        args.target_batch_size = data.feat_full.shape[0]

# hop not used
if args.hop>1:# hold
    if args.dataset in ['flickr', 'reddit']:
        A_ALL = None
        A_TRAIN = make_hop_a(A_TRAIN,args.hop,args).to_dense().numpy()
        A_VALID = make_hop_a(A_VALID,args.hop,args).to_dense().numpy()
        A_TEST = make_hop_a(A_TEST,args.hop,args).to_dense().numpy()
    else:
        A_ALL = make_hop_a(A_ALL,args.hop)
        A_TRAIN = torch.index_select(torch.index_select(A_ALL, 0, torch.tensor(data.idx_train)), 1, torch.tensor(data.idx_train))#make_hop_a(A_TRAIN,args.hop,args)
        A_VALID = torch.index_select(torch.index_select(A_ALL, 0, torch.tensor(data.idx_val)), 1, torch.tensor(data.idx_val))#make_hop_a(A_VALID,args.hop,args)
        A_ALL = sp.sparse.csr_matrix(A_ALL.to_dense().numpy())
        gc.collect()
        A_TRAIN = A_TRAIN.to_dense().numpy()
        A_VALID = A_VALID.to_dense().numpy()
else:
    pass #del A_ALL
gc.collect()

# testing pipeline
if args.is_testing:
    args.target_batch_size = 480
    
    classes = onp.unique(LABELS_TRAIN)
    n_per_class = 480
    _, class_counts = torch.tensor(LABELS_TRAIN).unique(return_counts=True)
    class_counts = class_counts.numpy()
    inds = onp.concatenate([
      onp.random.choice(onp.where(LABELS_TRAIN == c)[0], n_per_class if class_counts[c]>=n_per_class else class_counts[c], replace=False)
      for c in classes
    ])
    X_TRAIN = X_TRAIN[inds]
    LABELS_TRAIN = LABELS_TRAIN[inds]
    Y_TRAIN = Y_TRAIN[inds]
    A_TRAIN = A_TRAIN[inds]
    A_TRAIN = (A_TRAIN.T)[inds].T
    
    _, class_counts = torch.tensor(LABELS_VALID).unique(return_counts=True)
    class_counts = class_counts.numpy()
    inds = onp.concatenate([
      onp.random.choice(onp.where(LABELS_VALID == c)[0], n_per_class if class_counts[c]>=n_per_class else class_counts[c], replace=False)
      for c in classes
    ])
    X_VALID = X_VALID[inds]
    LABELS_VALID = LABELS_VALID[inds]
    Y_VALID = Y_VALID[inds]
    A_VALID = A_VALID[inds]
    A_VALID = (A_VALID.T)[inds].T
    
    _, class_counts = torch.tensor(LABELS_TEST).unique(return_counts=True)
    A_VALID = (A_VALID.T)[inds].T
    
    _, class_counts = torch.tensor(LABELS_TEST).unique(return_counts=True)
    class_counts = class_counts.numpy()
    inds = onp.concatenate([
      onp.random.choice(onp.where(LABELS_TEST == c)[0], n_per_class if class_counts[c]>=n_per_class else class_counts[c], replace=False)
      for c in classes
    ])
    X_TEST = X_TEST[inds]
    LABELS_TEST = LABELS_TEST[inds]
    Y_TEST = Y_TEST[inds]
    A_TEST = A_TEST[inds]
    A_TEST = (A_TEST.T)[inds].T

# train
params_final, params_init, params_init_raw = train(args, log_freq=int(len(Y_TRAIN)/args.target_batch_size), IS_A_FIX=args.is_a_fix,A_TRAIN=A_TRAIN,A_VALID=A_VALID,A_TEST=A_TEST, X_TRAIN=X_TRAIN, X_VALID=X_VALID, X_TEST=X_TEST, Y_TRAIN=Y_TRAIN, Y_VALID=Y_VALID, Y_TEST=Y_TEST, A_SUM_TRAIN=A_SUM_TRAIN, LABELS_TRAIN=LABELS_TRAIN, save_score_name=save_score_name, data=data, Y_ALL=Y_ALL, difficult_scores_final=None if not args.is_use_clscore_sort else difficult_scores_final)


# save log
FILE = open('seed '+str(args.seed)+' finish at '+current_time_str+'.txt',"w")
FILE.close()

with open(save_score_name, 'r') as file: 
    
    for line in (file.readlines() [-3:]):
        try:
            with open('temp_all_seed_scores.txt', 'a') as modified: modified.write(line+" seed "+str(args.seed)+"\n")
        except:
            with open('temp_all_seed_scores.txt', 'w') as modified: modified.write(line+" seed "+str(args.seed)+"\n")

with open('temp_all_seed_scores.txt', 'r') as file, open('temp_all_seed_scores_totaltest.txt','w') as out:
    abc = (file.read().split('TEST'))[1:]
    #pdb.set_trace()
    total_score = 0.0
    for ijj,iii in enumerate(abc):
        if ijj!=0 and ijj%5==0:
            out.write(str(total_score/5)+'\n')
            out.write('------------\n')
            total_score = 0.0
        total_score+=(float(iii[9:15]))
    out.write('Total TEST: '+str(total_score/5)+'\n')
    print('Total TEST: '+str(total_score/5))
    out.write('------------\n')
