import numpy as np
import random
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
from models.sgc_multi import SGC as SGC1
from models.parametrized_adj import PGE
import scipy.sparse as sp
from torch_sparse import SparseTensor
import pdb


def get_discrete_graphs(adj, args, inference=True):
    if not hasattr(args, 'cnt'):
        args.cnt = 0

    #if args.dataset not in ['CIFAR10']:
    #    adj = (adj.transpose(1,2) + adj) / 2

    if not inference:
        N = adj.size()[1]
        #vals = torch.rand(adj.size(0) * N * (N+1) // 2)
        vals = torch.rand(N * (N+1) // 2)
        #vals = vals.view(adj.size(0), -1).to(args.device)
        vals = vals.view(-1).to().to(torch.float).to(args.device)
        i, j = torch.triu_indices(N, N)
        epsilon = torch.zeros_like(adj).to(torch.float).to(args.device)
        #epsilon[:, i, j] = vals
        epsilon[i, j] = vals
        #epsilon.transpose(1,2)[:, i, j] = vals

        tmp = (torch.log(epsilon) - torch.log(1-epsilon)).to(args.device)
        #self.tmp = tmp
        adj = tmp + adj.to(args.device)
        t0 = 1
        tt = 0.01
        end_iter = 200
        t = t0*(tt/t0)**(args.cnt/end_iter)
        if args.cnt == end_iter:
            print('===reached the end of anealing...')
        args.cnt += 1

        t = max(t, tt)
        adj = torch.sigmoid(adj/t).to(args.device)
        #adj = adj * (1-torch.eye(adj.size(1)).to(args.device))
    else:
        adj = torch.sigmoid(adj).to(args.device)
        #adj = adj * (1-torch.eye(adj.size(1)).to(args.device))
        adj = adj + (torch.eye(adj.size(1)).to(args.device))
        #adj[adj> 0.5] = 1
        #adj[adj<= 0.5] = 0
    return adj

def swish_activation(a_init,args, inference=True):
    return torch.nn.functional.silu(a_init)

class GCond:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device

        # n = data.nclass * args.nsamples
        n = int(data.feat_train.shape[0] * args.reduction_rate)
        # from collections import Counter; print(Counter(data.labels_train))

        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
        self.pge = PGE(nfeat=d, nnodes=n, device=device,args=args).to(device)

        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)

        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        #print('adj_syn:', (n,n), 'feat_syn:', self.feat_syn.shape)

    def make_hop_a(self,adj_matrix,args):
        adj_matrix = torch.tensor(adj_matrix).to_sparse()
        adj_matrix += torch.eye(*adj_matrix.shape).to_sparse().to(torch.int8)
        #print(adj_matrix.dtype)
        for single_hop in range(1,args.hop):
            if single_hop ==1:
                new_adj_matrix_before = adj_matrix
                #print((new_adj_matrix_before @ new_adj_matrix_before).to_dense()>0.)
                #print(new_adj_matrix_before.dtype,new_adj_matrix_before.dtype)
                new_adj_matrix_after = ((new_adj_matrix_before.to(torch.float) @ new_adj_matrix_before.to(torch.float)).to(torch.int8).to_dense()>0).to(torch.int8)
                final_adj_matrix = adj_matrix+((single_hop+1)*(new_adj_matrix_after - new_adj_matrix_before)).to_sparse()
            else:
                new_adj_matrix_before = new_adj_matrix_after
                new_adj_matrix_after = ((new_adj_matrix_before.to(torch.float) @ new_adj_matrix_before.to(torch.float)).to(torch.int8).to_dense()>0).to(torch.int8)
                final_adj_matrix += ((single_hop+1)*(new_adj_matrix_after - new_adj_matrix_before)).to_sparse()
        #print(final_adj_matrix.to_dense())
        if args.hop_type == 'exp':
            final_adj_matrix = ((args.hop_coefficient.to_dense())**(final_adj_matrix.to_dense()))#.to_sparse()
            #final_adj_matrix[final_adj_matrix> 1] = 0.
        elif args.hop_type == 'reverse':
            final_adj_matrix = (1./final_adj_matrix.to_dense())#.to_sparse()
            final_adj_matrix[final_adj_matrix>1.] = 0.
        else:
            raise NotImplementedError
        print('hopping finish')
        return (final_adj_matrix - torch.eye(*adj_matrix.shape).to_sparse()).to_dense().numpy() #.to(torch.int8)

    def seed_everything(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def a_init_to_param(self,a_init,args): #sigmoid((log(a)-log(1-a)+omegaij)/temp) #
        a = torch.rand_like(a_init)
        if args.is_doscond_sigmoid:
            return torch.sigmoid((torch.log(a)-torch.log(1.-a)+a_init)/torch.tensor(args.temp,requires_grad=False))#+torch.eye(*a_init.size(),requires_grad=False)
        else:
            return torch.sigmoid(a_init/torch.tensor(args.temp,requires_grad=False)) #

    def get_discrete_graphs(self, adj, args, inference=True):
        if args.is_doscond_sigmoid:
            if not hasattr(args, 'cnt'):
                args.cnt = 0

            #if args.dataset not in ['CIFAR10']:
            #    adj = (adj.transpose(1,2) + adj) / 2

            if not inference:
                N = adj.size()[1]
                #vals = torch.rand(adj.size(0) * N * (N+1) // 2)
                vals = torch.rand(N * (N+1) // 2)
                #vals = vals.view(adj.size(0), -1).to(args.device)
                vals = vals.view(-1).to(torch.float).to(args.device)
                i, j = torch.triu_indices(N, N)
                epsilon = torch.zeros_like(adj).to(torch.float).to(args.device)
                #epsilon[:, i, j] = vals
                epsilon[i, j] = vals
                #epsilon.transpose(1,2)[:, i, j] = vals

                tmp = (torch.log(epsilon) - torch.log(1-epsilon)).to(args.device)
                #self.tmp = tmp
                adj = tmp + adj.to(args.device)
                t0 = 1
                tt = 0.01
                end_iter = 200
                t = t0*(tt/t0)**(args.cnt/end_iter)
                if args.cnt == end_iter:
                    print('===reached the end of anealing...')
                args.cnt += 1

                t = max(t, tt)
                adj = torch.sigmoid(adj/t).to(args.device)
                #adj = adj * (1-torch.eye(adj.size(1)).to(args.device))
            else:
                adj = torch.sigmoid(adj).to(args.device)
                #adj = adj * (1-torch.eye(adj.size(1)).to(args.device))
                #adj = adj + (torch.eye(adj.size(1)).to(args.device))
                #adj[adj> 0.5] = 1
                #adj[adj<= 0.5] = 0
            return adj
        elif args.is_relu_adj:
            return torch.nn.functional.relu(adj).to(args.device)
        elif args.is_none_adj:
            return adj.to(args.device)
        else:
            raise NotImplementedError

    def swish_activation(self, a_init,args, inference=True):
        return torch.nn.functional.silu(a_init)

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

    def val_with_val(self, feat_syn, adj_syn, labels_syn, hop_a_full, hop_a_test, verbose=True):
        #self.seed_everything(self.args.seed)
        res = []

        data, device = self.data, self.device # build data outside, data = {a:b,c:d}

        #feat_syn, pge, labels_syn = self.feat_syn.detach(), \
        #                       self.pge, self.labels_syn
        #feat_syn = 
        #pge = None
        #labels_syn = 

        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        print('val featsyn shape: ',feat_syn.shape,' adj_syn.shape: ',adj_syn.shape)
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                    weight_decay=self.args.test_wd, nlayers=self.args.nlayers,lr=self.args.test_lr, with_bn=self.args.is_with_bn,
                    nclass=data.nclass, device=device).to(device)

        if self.args.dataset in ['ogbn-arxiv']:
            model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,lr=self.args.test_lr,weight_decay=self.args.test_wd, nlayers=self.args.nlayers, with_bn=False,
                        nclass=data.nclass, device=device).to(device)
            
        #adj_syn = pge.inference(feat_syn)
        #adj_syn = a_init_to_param(adj_syn, self.args)#torch.sigmoid(adj_syn)
        #if self.args.is_y_pred_softmax:
        if self.args.a_strategy == 'Swish':
            self.get_discrete_graphs = self.swish_activation
        adj_syn = self.get_discrete_graphs(adj_syn, self.args, inference=True)
        #print('adj syn val: ',adj_syn[0])
        args = self.args

        if self.args.save:
            torch.save(adj_syn, f'saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
            torch.save(feat_syn, f'saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')

        if self.args.lr_adj == 0:
            n = len(labels_syn) 
            adj_syn = torch.zeros((n, n))
        #print(']===')
        #print(labels_syn.max(),labels_syn.min())
        model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                     train_iters=self.args.fit_with_val_epochs, normalize=True, verbose=False,noval=True if args.dataset in ['flickr','reddit'] else False)

        model.eval()
        labels_valid = torch.LongTensor(data.labels_val).cuda()

        labels_train = torch.LongTensor(data.labels_train).cuda()
        
        labels_test = torch.LongTensor(data.labels_test).cuda()
        #print('===')
        #print(labels_train.max(),labels_train.min())
        #print('===')
        '''output = model.predict(data.feat_train, data.adj_train)
        #def custom_loss(logits, targets):
        #    targets = torch.softmax(targets,-1)
        #    return -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=-1))
        loss_train = F.nll_loss(F.log_softmax(output,dim=-1), labels_train)
        #loss_train = custom_loss(output, labels_train)
        acc_train = utils.accuracy(output, labels_train)
        if verbose:
            print("Train set results:",
                  "loss= {:.4f}".format(loss_train.item()),
                  "accuracy= {:.4f}".format(acc_train.item()))
        res.append(acc_train.item())
        '''

        # Full graph
        if args.dataset in ['flickr', 'reddit']:
            print('examine in induct like sfgc')
            output_val = model.predict(data.feat_val, data.adj_val)
            loss_valid = F.nll_loss(F.log_softmax(output_val,dim=-1), labels_valid)
            acc_valid = utils.accuracy(output_val, labels_valid)
            #
            if self.args.hop>1 and self.args.is_test_hop:
                output_test = model.predict(data.feat_test, hop_a_test.to(torch.float16))
            else:
                output_test = model.predict(data.feat_test, data.adj_test)
            loss_test = F.nll_loss(F.log_softmax(output_test,dim=-1), labels_test)
            acc_test = utils.accuracy(output_test, labels_test)
            if verbose:
                print("Valid set results:",
                      "loss= {:.4f}".format(loss_valid.item()),
                      "accuracy= {:.4f}".format(acc_valid.item()))
                print("Test set results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

        else:
            if self.args.hop>1 and self.args.is_test_hop:
                output = model.predict(data.feat_full, hop_a_full)
            else:
                output = model.predict(data.feat_full, data.adj_full)
            #loss_train
            loss_train = F.nll_loss(F.log_softmax(output[data.idx_train],dim=-1), labels_train)
            acc_train = utils.accuracy(output[data.idx_train], labels_train)
            if verbose:
                print("Train set results:",
                      "loss= {:.4f}".format(loss_train.item()),
                      "accuracy= {:.4f}".format(acc_train.item()))
            res.append(acc_train.item())
            
            loss_valid = F.nll_loss(F.log_softmax(output[data.idx_val],dim=-1), labels_valid)
            #loss_valid = custom_loss(output[data.idx_val], labels_valid)
            acc_valid = utils.accuracy(output[data.idx_val], labels_valid)
            loss_test = F.nll_loss(F.log_softmax(output[data.idx_test],dim=-1), labels_test)
            #loss_test = custom_loss(output[data.idx_test], labels_test)
            acc_test = utils.accuracy(output[data.idx_test], labels_test)
            res.append(acc_valid.item())
            if verbose:
                print("Valid set results:",
                      "loss= {:.4f}".format(loss_valid.item()),
                      "accuracy= {:.4f}".format(acc_valid.item()))
                print("Test set results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))
        #return res
        '''
        if args.dataset in ['flickr', 'reddit']:
            print('examine in induct like sfgc')
            output_val_again = model.predict(data.feat_val, data.adj_val)
            loss_valid_again = F.nll_loss(F.log_softmax(output_val_again,dim=-1), labels_valid)
            acc_valid_again = utils.accuracy(output_val_again, labels_valid)
            #
            output_test_again = model.predict(data.feat_test, data.adj_test)
            loss_test_again = F.nll_loss(F.log_softmax(output_test_again,dim=-1), labels_test)
            acc_test_again = utils.accuracy(output_test_again, labels_test)
            if verbose:
                print("##########")
                print("Valid set AGAIN results:",
                      "loss= {:.4f}".format(loss_valid_again.item()),
                      "accuracy= {:.4f}".format(acc_valid_again.item()))
                print("Test set AGAIN results:",
                      "loss= {:.4f}".format(loss_test_again.item()),
                      "accuracy= {:.4f}".format(acc_test_again.item()))
                print("##########")
            if acc_valid != acc_valid_again:
                loss_valid = loss_valid_again
                acc_valid = acc_valid_again
                loss_test = loss_test_again
                acc_test = acc_test_again
        '''
        return loss_valid, acc_valid,loss_test, acc_test
    

    def test_with_val(self, feat_syn, adj_syn, labels_syn, verbose=True):
        res = []

        data, device = self.data, self.device # build data outside, data = {a:b,c:d}

        #feat_syn, pge, labels_syn = self.feat_syn.detach(), \
        #                       self.pge, self.labels_syn
        #feat_syn = 
        #pge = None
        #labels_syn = 

        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device).to(device)

        if self.args.dataset in ['ogbn-arxiv']:
            model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                        weight_decay=0e-4, nlayers=2, with_bn=False,
                        nclass=data.nclass, device=device).to(device)

        #adj_syn = pge.inference(feat_syn)
        #adj_syn = a_init_to_param(adj_syn, self.args) #torch.sigmoid(adj_syn)
        #if self.args.is_y_pred_softmax:
        if self.args.a_strategy == 'Swish':
            self.get_discrete_graphs = self.swish_activation
        adj_syn = self.get_discrete_graphs(adj_syn, self.args, inference=True)
        args = self.args

        if self.args.save:
            torch.save(adj_syn, f'saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
            torch.save(feat_syn, f'saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')

        if self.args.lr_adj == 0:
            n = len(labels_syn)
            adj_syn = torch.zeros((n, n))
        #print(']===')
        #print(labels_syn.max(),labels_syn.min())
        model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                     train_iters=600, normalize=True, verbose=False)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        labels_train = torch.LongTensor(data.labels_train).cuda()
        #print('===')
        #print(labels_train.max(),labels_train.min())
        #print('===')
        output = model.predict(data.feat_train, data.adj_train)
        #def custom_loss(logits, targets):
        #    targets = torch.softmax(targets,-1)
        #    return -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=-1))
        loss_train = F.nll_loss(F.log_softmax(output,dim=-1), labels_train)
        #loss_train = custom_loss(output, labels_train)
        acc_train = utils.accuracy(output, labels_train)
        if verbose:
            print("Train set results:",
                  "loss= {:.4f}".format(loss_train.item()),
                  "accuracy= {:.4f}".format(acc_train.item()))
        res.append(acc_train.item())

        # Full graph
        output = model.predict(data.feat_full, data.adj_full)
        loss_test = F.nll_loss(F.log_softmax(output[data.idx_test],dim=-1), labels_test)
        #loss_test = custom_loss(output[data.idx_test], labels_test)
        acc_test = utils.accuracy(output[data.idx_test], labels_test)
        res.append(acc_test.item())
        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
        #return res
        return loss_test, acc_test

    def train(self, verbose=True):
        args = self.args
        data = self.data
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        idx_train = data.idx_train

        syn_class_indices = self.syn_class_indices

        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)

        feat_sub, adj_sub = self.get_sub_adj_feat(features)
        self.feat_syn.data.copy_(feat_sub)

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()


        outer_loop, inner_loop = get_loops(args)
        loss_avg = 0

        for it in range(args.epochs+1):
            if args.dataset in ['ogbn-arxiv']:
                model = SGC1(nfeat=feat_syn.shape[1], nhid=self.args.hidden,
                            dropout=0.0, with_bn=False,
                            weight_decay=0e-4, nlayers=2,
                            nclass=data.nclass,
                            device=self.device).to(self.device)
            else:
                if args.sgc == 1:
                    model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                                nclass=data.nclass, dropout=args.dropout,
                                nlayers=args.nlayers, with_bn=False,
                                device=self.device).to(self.device)
                else:
                    model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                                nclass=data.nclass, dropout=args.dropout, nlayers=args.nlayers,
                                device=self.device).to(self.device)


            model.initialize()

            model_parameters = list(model.parameters())

            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()

            for ol in range(outer_loop):
                adj_syn = pge(self.feat_syn)
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
                feat_syn_norm = feat_syn

                BN_flag = False
                for module in model.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    model.train() # for updating the mu, sigma of BatchNorm
                    output_real = model.forward(features, adj_norm)
                    for module in model.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer

                loss = torch.tensor(0.0).to(self.device)
                for c in range(data.nclass):
                    batch_size, n_id, adjs = data.retrieve_class_sampler(
                            c, adj, transductive=True, args=args)
                    if args.nlayers == 1:
                        adjs = [adjs]
                    #pdb.set_trace()
                    adjs = [adj.to(self.device) for adj in adjs]
                    output = model.forward_sampler(features[n_id], adjs)
                    loss_real = F.nll_loss(output, labels[n_id[:batch_size]])
                    #pdb.set_trace()
                    gw_real = torch.autograd.grad(loss_real, model_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    output_syn = model.forward(feat_syn, adj_syn_norm)

                    ind = syn_class_indices[c]
                    loss_syn = F.nll_loss(
                            output_syn[ind[0]: ind[1]],
                            labels_syn[ind[0]: ind[1]])
                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss += coeff  * match_loss(gw_syn, gw_real, args, device=self.device)

                loss_avg += loss.item()
                # TODO: regularize
                if args.alpha > 0:
                    loss_reg = args.alpha * regularization(adj_syn, utils.tensor2onehot(labels_syn))
                else:
                    loss_reg = torch.tensor(0)

                loss = loss + loss_reg

                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()
                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                if args.debug and ol % 5 ==0:
                    print('Gradient matching loss:', loss.item())

                if ol == outer_loop - 1:
                    # print('loss_reg:', loss_reg.item())
                    # print('Gradient matching loss:', loss.item())
                    break

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    # print(loss_syn_inner.item())
                    optimizer_model.step() # update gnn param


            loss_avg /= (data.nclass*outer_loop)
            if it % 50 == 0:
                print('Epoch {}, loss_avg: {}'.format(it, loss_avg))

            eval_epochs = [21,41,61]#[400, 600, 800, 1000, 1200, 1600, 2000, 3000, 4000, 5000]

            if verbose and it in eval_epochs:
            # if verbose and (it+1) % 50 == 0:
                res = []
                runs = 1 if args.dataset in ['ogbn-arxiv'] else 3
                for i in range(runs):
                    if args.dataset in ['ogbn-arxiv']:
                        res.append(self.test_with_val())
                    else:
                        res.append(self.test_with_val())

                res = np.array(res)
                print('Train/Test Mean Accuracy:',
                        repr([res.mean(0), res.std(0)]))

    def get_sub_adj_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter;
        counter = Counter(self.labels_syn.cpu().numpy())

        for c in range(data.nclass):
            tmp = data.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[self.data.idx_train][idx_selected]

        # adj_knn = torch.zeros((data.nclass*args.nsamples, data.nclass*args.nsamples)).to(self.device)
        # for i in range(data.nclass):
        #     idx = np.arange(i*args.nsamples, i*args.nsamples+args.nsamples)
        #     adj_knn[np.ix_(idx, idx)] = 1

        from sklearn.metrics.pairwise import cosine_similarity
        # features[features!=0] = 1
        k = 2
        sims = cosine_similarity(features.cpu().numpy())
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[: -k]] = 0
        adj_knn = torch.FloatTensor(sims).to(self.device)
        return features, adj_knn


def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.one_step:
        if args.dataset =='ogbn-arxiv':
            return 5, 0
        return 1, 0
    if args.dataset in ['ogbn-arxiv']:
        return args.outer, args.inner
    if args.dataset in ['cora']:
        return 20, 15 # sgc
    if args.dataset in ['citeseer']:
        return 20, 15
    if args.dataset in ['physics']:
        return 20, 10
    else:
        return 20, 10

