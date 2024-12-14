import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from utils import match_loss, regularization, row_normalize_tensor
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from models.gcn_custom import GCN
from models.sgc import SGC
from models.sgc_multi_custom import SGC as SGC1
from models.myappnp import APPNP
from models.myappnp1_custom import APPNP1
from models.mycheby_custom import Cheby
from models.mygraphsage_custom import GraphSage
from models.gat_custom import GAT
import scipy.sparse as sp
import pdb


class Evaluator:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        n = int(data.feat_train.shape[0] * args.reduction_rate)
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.adj_param= nn.Parameter(torch.FloatTensor(n, n).to(device))
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)
        self.reset_parameters()
        print('adj_param:', self.adj_param.shape, 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        self.adj_param.data.copy_(torch.randn(self.adj_param.size()))
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def generate_labels_syn(self, data):
        from collections import Counter
        counter = Counter(data.labels_train)
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn


    def test_gat(self, nlayers, model_type, verbose=False):
        res = []
        args = self.args

        if args.dataset in ['cora', 'citeseer']:
            pass#args.epsilon = -1 #0.5 # Make the graph sparser as GAT does not work well on dense graph
        else:
            pass#args.epsilon = -1 #0.01

        print('======= testing %s' % model_type)
        data, device = self.data, self.device


        feat_syn, adj_syn, labels_syn = self.get_syn_data(model_type)
        # with_bn = True if self.args.dataset in ['ogbn-arxiv'] else False
        with_bn = False
        if 1==1: #args.dataset == 'citeseer':
            test_lr = args.test_lr#0.003
        #elif args.dataset == 'cora':
        #    test_lr = args.test_lr#0.003
        else:
            raise NotImplementedError
        print(args.test_wd)
        #pdb.set_trace()
        if model_type == 'GAT':
            model = GAT(nfeat=feat_syn.shape[1], nhid=16, heads=16, dropout=0.0,
                        weight_decay=0e-4, nlayers=self.args.nlayers, lr=test_lr, # default wd 0e-4
                        nclass=data.nclass, device=device, dataset=self.args.dataset).to(device)
            if args.dataset == 'cora':
                model = GAT(nfeat=feat_syn.shape[1], nhid=16, heads=4, dropout=0.5,
                            weight_decay=args.test_wd, nlayers=self.args.nlayers, lr=test_lr, # default wd 0e-4
                            nclass=data.nclass, device=device, dataset=self.args.dataset).to(device)
            else: #if args.dataset == 'ogbn-arxiv':
                model = GAT(nfeat=feat_syn.shape[1], nhid=16, heads=16, dropout=0.2,
                            weight_decay=args.test_wd, nlayers=self.args.nlayers, lr=test_lr,
                            nclass=data.nclass, device=device, dataset=self.args.dataset).to(device)


        noval = True if args.dataset in ['reddit', 'flickr'] else False
        if args.dataset == 'cora':
            model.fit(feat_syn, adj_syn, labels_syn, np.arange(len(feat_syn)), noval=noval, data=data,
                     train_iters=10000 if noval else 6000, normalize=True, verbose=verbose)
        elif args.dataset == 'ogbn-arxiv':
            pass
            model.fit(feat_syn, adj_syn, labels_syn, np.arange(len(feat_syn)), noval=noval, data=data,
                                         train_iters=10000 if noval else 600, normalize=True, verbose=verbose)
        else:
            model.fit(feat_syn, adj_syn, labels_syn, np.arange(len(feat_syn)), noval=noval, data=data,
                     train_iters=10000 if noval else 3000, normalize=True, verbose=verbose)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        if args.dataset in ['reddit', 'flickr']:
            output = model.predict(data.feat_test, data.adj_test)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = utils.accuracy(output, labels_test)
            res.append(acc_test.item())
            if verbose:
                print("Test set results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

        else:
            # Full graph
            output = model.predict(data.feat_full, data.adj_full)
            loss_test = F.nll_loss(output[data.idx_test], labels_test)
            acc_test = utils.accuracy(output[data.idx_test], labels_test)
            res.append(acc_test.item())
            if verbose:
                print("Test set results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

        labels_train = torch.LongTensor(data.labels_train).cuda()
        output = model.predict(data.feat_train, data.adj_train)
        loss_train = F.nll_loss(output, labels_train)
        acc_train = utils.accuracy(output, labels_train)
        if verbose:
            print("Train set results:",
                  "loss= {:.4f}".format(loss_train.item()),
                  "accuracy= {:.4f}".format(acc_train.item()))
        res.append(acc_train.item())
        return res

    def get_syn_data(self, model_type=None):
        data, device = self.data, self.device
        feat_syn, adj_param, labels_syn = self.feat_syn.detach(), \
                                self.adj_param.detach(), self.labels_syn

        args = self.args
        #adj_syn = torch.load(f'saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cuda')
        #feat_syn = torch.load(f'saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location='cuda')
        #labels_syni
        if args.dataset == 'citeseer':
            adj_syn = torch.tensor(np.load(f'citeseer_final_a.npy'))
            adj_syn[adj_syn < 0.] = 0.
            feat_syn = torch.tensor(np.load(f'citeseer_final_x.npy'))
            labels_syn = torch.tensor(np.load(f'citeseer_final_y.npy'))
        elif args.dataset == 'cora':
            adj_syn = torch.tensor(np.load(f'cora_final_a.npy'))
            adj_syn[adj_syn < 0.] = 0.
            feat_syn = torch.tensor(np.load(f'cora_final_x.npy'))
            labels_syn = torch.tensor(np.load(f'cora_final_y.npy'))
        elif args.dataset == 'ogbn-arxiv':
            adj_syn = torch.tensor(np.load(f'ogbn_arxiv_final_a.npy'))
            adj_syn[adj_syn < 0.] = 0.
            feat_syn = torch.tensor(np.load(f'ogbn_arxiv_final_x.npy'))
            labels_syn = torch.tensor(np.load(f'ogbn_arxiv_final_y.npy'))
        elif args.dataset == 'flickr':
            adj_syn = torch.tensor(np.load(f'flickr_final_a.npy'))
            adj_syn[adj_syn < 0.] = 0.
            feat_syn = torch.tensor(np.load(f'flickr_final_x.npy'))
            labels_syn = torch.tensor(np.load(f'flickr_final_y.npy'))
        elif args.dataset == 'reddit':
            adj_syn = torch.tensor(np.load(f'reddit_final_a.npy'))
            adj_syn[adj_syn < 0.] = 0.
            feat_syn = torch.tensor(np.load(f'reddit_final_x.npy'))
            labels_syn = torch.tensor(np.load(f'reddit_final_y.npy'))
        else:
            adj_syn = torch.tensor(np.load(f'final_a_{args.seed}.npy'))
            adj_syn[adj_syn < 0.] = 0.
            feat_syn = torch.tensor(np.load(f'final_x_{args.seed}.npy'))
            labels_syn = torch.tensor(np.load(f'final_y_{args.seed}.npy'))


        if model_type == 'MLP':
            adj_syn = adj_syn.to(self.device)
            adj_syn = adj_syn - adj_syn + torch.eye(adj_syn.size(0)).to(self.device)
        else:
            adj_syn = adj_syn.to(self.device)

        print('Sum:', adj_syn.sum(), adj_syn.sum()/(adj_syn.shape[0]**2))
        print('Sparsity:', adj_syn.nonzero().shape[0]/(adj_syn.shape[0]**2))

        if self.args.epsilon > 0:
            adj_syn[adj_syn < self.args.epsilon] = 0
            print('Sparsity after truncating:', adj_syn.nonzero().shape[0]/(adj_syn.shape[0]**2))
        feat_syn = feat_syn.to(self.device)

        # edge_index = adj_syn.nonzero().T
        # adj_syn = torch.sparse.FloatTensor(edge_index,  adj_syn[edge_index[0], edge_index[1]], adj_syn.size())

        return feat_syn, adj_syn, labels_syn


    def test(self, nlayers, model_type, verbose=True):
        res = []

        args = self.args
        data, device = self.data, self.device

        feat_syn, adj_syn, labels_syn = self.get_syn_data(model_type)
        #if model_type == 'Cheby':
        #    adj_syn = adj_syn+torch.eye(adj_syn.size(0)).to(self.device)

        print('======= testing %s' % model_type)
        if model_type == 'MLP':
            model_class = GCN
        else:
            model_class = eval(model_type)
        weight_decay = args.test_wd# 5e-4 if model_type != 'Cheby' else args.test_wd #if model_type not in ['MLP','GCN'] else 0
        print('wd: ',weight_decay)

        dropout = 0.5 if args.dataset in ['reddit'] else 0

        if 1==1: #args.dataset == 'citeseer':
            #if model_type == 'GCN' or model_type == 'MLP':
            #    test_lr = 0.01
            if model_type == 'MLP':                                                                                                                                                                                              test_lr = args.test_lr
            elif model_type == 'SGC1':
                test_lr = args.test_lr#0.009
            elif model_type == 'GraphSage':
                test_lr = args.test_lr#0.001
            elif model_type == 'APPNP1':
                test_lr = args.test_lr#0.025
            elif model_type == 'Cheby':
                test_lr = args.test_lr#0.023
            else:
                raise NotImplementedError
        #elif args.dataset == 'cora':
        #    if model_type == 'MLP':
        #        test_lr = args.test_lr
        #    elif model_type == 'SGC1':
        #        test_lr = args.test_lr#0.051
        #    elif model_type == 'GraphSage':
        #        test_lr = args.test_lr
        #    elif model_type == 'APPNP1':
        #        test_lr = args.test_lr
        #    elif model_type == 'Cheby':
        #        test_lr = args.test_lr
        #    else:
        #        raise NotImplementedError
        else:
            raise NotImplementedError
        
        if model_type in ['Cheby','GraphSage']:
            model = model_class(nfeat=feat_syn.shape[1], nhid=args.hidden, dropout=dropout,
                                weight_decay=weight_decay, nlayers=nlayers, lr = test_lr,
                                nclass=data.nclass, device=device,dataset = self.args.dataset).to(device)
        else:
            model = model_class(nfeat=feat_syn.shape[1], nhid=args.hidden, dropout=dropout,
                                weight_decay=weight_decay, nlayers=nlayers, lr = test_lr,
                                nclass=data.nclass, device=device).to(device)
        # with_bn = True if self.args.dataset in ['ogbn-arxiv'] else False
        if args.dataset in ['ogbn-arxiv', 'arxiv']:
            if model_type in ['Cheby','GraphSage']:
                model = model_class(nfeat=feat_syn.shape[1], nhid=args.hidden, dropout=0.5 if model_type=='Cheby' else 0.5,
                                    weight_decay=weight_decay, nlayers=nlayers, lr = test_lr,
                                    nclass=data.nclass, device=device,dataset = self.args.dataset).to(device)
            else:
                model = model_class(nfeat=feat_syn.shape[1], nhid=args.hidden, dropout=0.5 if model_type=='Cheby' else 0.5,
                                    weight_decay=weight_decay, nlayers=nlayers, lr = test_lr, with_bn=False,
                                    nclass=data.nclass, device=device).to(device)

        noval = True if args.dataset in ['reddit', 'flickr'] else False
        if 1==0: #args.dataset == 'ogbn-arxiv':
            model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                     train_iters=600, normalize=True if model_type != 'MLP' else False, verbose=True, noval=noval)
        else:
            model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                     train_iters=3000, normalize=True if model_type != 'MLP' else False, verbose=True, noval=noval)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        # if model_type == 'MLP':
        #     output = model.predict_unnorm(data.feat_test, sp.eye(len(data.feat_test)))
        # else:
        #     output = model.predict(data.feat_test, data.adj_test)

        if args.dataset in ['reddit', 'flickr']:
            if model_type == 'MLP':
                output = model.predict_unnorm(data.feat_test, sp.eye(len(data.feat_test)))
            else:
                output = model.predict(data.feat_test, data.adj_test)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = utils.accuracy(output, labels_test)
            res.append(acc_test.item())
            if verbose:
                print("Test set results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

        # if not args.dataset in ['reddit', 'flickr']:
        else:
            # Full graph
            output = model.predict(data.feat_full, data.adj_full)
            loss_test = F.nll_loss(output[data.idx_test], labels_test)
            acc_test = utils.accuracy(output[data.idx_test], labels_test)
            res.append(acc_test.item())
            if verbose:
                print("Test full set results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

            labels_train = torch.LongTensor(data.labels_train).cuda()
            output = model.predict(data.feat_train, data.adj_train)
            loss_train = F.nll_loss(output, labels_train)
            acc_train = utils.accuracy(output, labels_train)
            if verbose:
                print("Train set results:",
                      "loss= {:.4f}".format(loss_train.item()),
                      "accuracy= {:.4f}".format(acc_train.item()))
            res.append(acc_train.item())
        return res

    def train(self, verbose=True):
        args = self.args
        data = self.data

        final_res = {}
        runs = self.args.nruns
        #total_score=[]
        
        #for model_type in []: ['GraphSage', 'SGC1', 'APPNP1', 'Cheby']:
        #    res = []
        #    nlayer = 2
        #    for i in range(runs):
        #        res.append(self.test(nlayer, verbose=False, model_type=model_type))
        #    res = np.array(res)
        #    print('Test/Train Mean Accuracy:',
        #            repr([res.mean(0), res.std(0)]))
        #    final_res[model_type] = [res.mean(0), res.std(0)]
        #PDb.set_trace() 
        if args.dataset in ['ogbn-arxiv','flickr','reddit']:
            from gcond_agent_transduct_custom_large import GCond
        else:
            from gcond_agent_transduct_custom import GCond
        #valid_loss, valid_acc, test_loss, test_acc = agent.val_with_val(params_init[0].detach().to(torch.float), params_init[2].detach().to(torch.float), params_init[1].detach(),None,A_TEST)    
        if args.dataset == 'citeseer':
            adj_syn = torch.tensor(np.load(f'citeseer_final_a.npy'))
            adj_syn[adj_syn < 0.] = 0.
            feat_syn = torch.tensor(np.load(f'citeseer_final_x.npy'))
            labels_syn = torch.tensor(np.load(f'citeseer_final_y.npy'))
        elif args.dataset == 'cora':
            adj_syn = torch.tensor(np.load(f'cora_final_a.npy'))
            adj_syn[adj_syn < 0.] = 0.
            feat_syn = torch.tensor(np.load(f'cora_final_x.npy'))
            labels_syn = torch.tensor(np.load(f'cora_final_y.npy'))
        elif args.dataset == 'ogbn-arxiv':
            adj_syn = torch.tensor(np.load(f'ogbn_arxiv_final_a.npy'))
            adj_syn[adj_syn < 0.] = 0.
            feat_syn = torch.tensor(np.load(f'ogbn_arxiv_final_x.npy'))
            labels_syn = torch.tensor(np.load(f'ogbn_arxiv_final_y.npy'))
        elif args.dataset == 'flickr':
            adj_syn = torch.tensor(np.load(f'flickr_final_a.npy'))
            adj_syn[adj_syn < 0.] = 0.
            feat_syn = torch.tensor(np.load(f'flickr_final_x.npy'))
            labels_syn = torch.tensor(np.load(f'flickr_final_y.npy'))
        elif args.dataset == 'reddit':
            adj_syn = torch.tensor(np.load(f'reddit_final_a.npy'))
            adj_syn[adj_syn < 0.] = 0.
            feat_syn = torch.tensor(np.load(f'reddit_final_x.npy'))
            labels_syn = torch.tensor(np.load(f'reddit_final_y.npy'))
        else:
            adj_syn = torch.tensor(np.load(f'final_a_{args.seed}.npy'))
            adj_syn[adj_syn < 0.] = 0.
            feat_syn = torch.tensor(np.load(f'final_x_{args.seed}.npy'))
            labels_syn = torch.tensor(np.load(f'final_y_{args.seed}.npy'))
        # general args
        args.lr_feat = 0 #0.01
        #args.gpu_id=3
        args.lr_adj = 0  #0.01
        args.reduction_rate = 0.5
        #args.seed = 1
        #args.test_wd = 5e-4
        args.is_with_bn = False
        args.a_strategy = 'Keep_init'
        args.inner = 3
        args.is_doscond_sigmoid = False
        args.is_relu_adj = True
        args.device = 'cuda'
        args.save = False
        args.hop = 1
        args.fit_with_val_epochs = 600 if args.dataset=='ogbn-arxiv' else 3000
                                                                                                                                                                                    
        if args.mt == 'GCN':
            print('=== testing GCN')
            model_type = 'GCN'
            args.nlayers = 2
            args.sgc = 1
            if 1==1: #args.dataset == 'citeseer':
                args.test_lr = args.test_lr#0.005
            #elif args.dataset == 'cora':
            #    args.test_lr = args.test_lr#0.005
            else:
                raise NotImplementedError
            args.lr_feat = args.test_lr#0.01
            #args.gpu_id=3
            args.lr_adj = args.test_lr #0.01
            #args.reduction_rate = 0.5
            #args.seed = 1
            #args.test_wd = 5e-4
            #args.is_with_bn = False
            #args.a_strategy = 'Keep_init'
            #args.inner = 3
            #args.is_doscond_sigmoid = False
            #args.is_relu_adj = True
            #args.device = 'cuda'
            #args.save = False
            #args.hop = 1
            #args.fit_with_val_epochs = 3000
            #agent = GCond(data, args, device='cuda')
            gcn_res = []
            for _ in range(runs):
                #pdb.set_trace()
                agent = GCond(data, args, device='cuda')
                #pdb.set_trace()
                valid_loss, valid_acc, test_loss, test_acc = agent.val_with_val(feat_syn, adj_syn, labels_syn, None,data.adj_test)
                print('GCN Results: ',valid_acc,' ',test_acc)
                gcn_res.append(test_acc.cpu().numpy())
            print('test_lr: ',args.test_lr)
            print(gcn_res,np.array(gcn_res).mean())
            final_res[model_type] = [np.array(gcn_res).mean(), np.array(gcn_res).std()]
        '''
        if args.mt == 'MLP':
            print('=== testing MLP')
            model_type = 'MLP'
            adj_syn = adj_syn - adj_syn
            if args.dataset == 'citeseer':
                args.test_lr = 0.005
            elif args.dataset == 'cora':
                args.test_lr = args.test_lr#0.001
            else:
                raise NotImplementedError
            args.lr_feat = args.test_lr#0.01
            #args.gpu_id=3
            args.lr_adj = args.test_lr #0.01
            args.model_type = model_type
            gcn_res = []
            for _ in range(runs):
                agent = GCond(data, args, device='cuda')
                valid_loss, valid_acc, test_loss, test_acc = agent.val_with_val(feat_syn, adj_syn, labels_syn, None,data.adj_test)
                print('MLP Results: ',valid_acc,' ',test_acc)
                gcn_res.append(test_acc.cpu().numpy(i))
            print('test_lr: ',args.test_lr)
            print(gcn_res,np.array(gcn_res).mean())
            final_res[model_type] = [np.array(gcn_res).mean(), np.array(gcn_res).std()]
        if args.mt == 'GCN' or args.mt ==  'MLP':
            return 0
        '''
        if args.mt == 'GCN':
            return 0
        if args.mt!='GAT':
            model_list = [args.mt]
        else:
            model_list = []
        for model_type in model_list: #['GraphSage', 'SGC1', 'APPNP1', 'Cheby']:
            res = []
            nlayer = 2
            for i in range(runs):
                res.append(self.test(nlayer, verbose=False, model_type=model_type))
                print('this try: ',res[-1])
            res = np.array(res)
            print('Test/Train Mean Accuracy:',   repr([res.mean(0), res.std(0)]))
            final_res[model_type] = [res.mean(0), res.std(0)]
        if args.mt!='GAT':
            print('test_lr',args.test_lr)
            print(res)
            print('Final result:', final_res)
        if args.mt != 'GAT':
            return 0
        print('=== testing GAT')
        #adj_syn = torch.tensor(np.load(f'final_a_{args.seed}.npy'))
        #adj_syn[adj_syn<0.] = 0.
        res = []
        nlayer = 2
        for i in range(runs):
            res.append(self.test_gat(verbose=True, nlayers=nlayer, model_type='GAT'))
        res = np.array(res)
        print('Layer:', nlayer)
        print('Test/Full Test/Train Mean Accuracy:',
                repr([res.mean(0), res.std(0)]))
        final_res['GAT'] = [res.mean(0), res.std(0)]
        print('test_lr: ',args.test_lr)
        print(res)
        print('Final result:', final_res)
        
