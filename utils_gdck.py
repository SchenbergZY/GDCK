import os
from time import gmtime, strftime

import numpy as onp
import scipy as sp
import random

import gc
import functools
import operator as op
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Sized, Tuple, Type, TypeVar, Union

import datetime
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor,to_tensor,is_sparse_tensor,normalize_adj_tensor

import transformers
from sklearn.cluster import KMeans
import pickle


# seed everything
def seed_everything(seed: int):
    random.seed(seed)
    onp.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot(x,
            num_classes,
            center=True,
            dtype=onp.float32):
    assert len(x.shape) == 1
    one_hot_vectors = onp.array(x[:, None] == onp.arange(num_classes), dtype)
    if center:
        one_hot_vectors = one_hot_vectors - 1. / num_classes
    return one_hot_vectors


# neighborhood-based difficulty measurer
def neighborhood_difficulty_measurer(data, adj, label, args):
    edge_index = adj.coalesce().indices()
    edge_value = adj.coalesce().values()

    neighbor_label, _ = add_self_loops(edge_index)  #[[1, 1, 1, 1],[2, 3, 4, 5]]
    neighbor_label[1] = label[neighbor_label[1]]   #[[1, 1, 1, 1],[40, 20, 19, 21]]
    neighbor_label = torch.transpose(neighbor_label, 0, 1)  # [[1, 40], [1, 20], [1, 19], [1, 21]]
    index, count = torch.unique(neighbor_label, sorted=True, return_counts=True, dim=0)

    neighbor_class = torch.sparse_coo_tensor(index.T, count) 
    neighbor_class = neighbor_class.to_dense().float()

    neighbor_class = neighbor_class[data.idx_train]
    neighbor_class = F.normalize(neighbor_class, 1.0, 1)
    neighbor_entropy = -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
    local_difficulty = neighbor_entropy.sum(1)

    print('done')
    return local_difficulty.to(args.device)


def neighborhood_difficulty_measurer_in(data, adj, label, args):
    edge_index = adj.coalesce().indices()
    edge_value = adj.coalesce().values()
    
    neighbor_label, _ = add_self_loops(edge_index)  #[[1, 1, 1, 1],[2, 3, 4, 5]]
    neighbor_label[1] = label[neighbor_label[1]]   #[[1, 1, 1, 1],[40, 20, 19, 21]]
    neighbor_label = torch.transpose(neighbor_label, 0, 1)  # [[1, 40], [1, 20], [1, 19], [1, 21]]
    index, count = torch.unique(neighbor_label, sorted=True, return_counts=True, dim=0)
    
    neighbor_class = torch.sparse_coo_tensor(index.T, count) 
    neighbor_class = neighbor_class.to_dense().float()
 
    neighbor_class = F.normalize(neighbor_class, 1.0, 1)
    neighbor_entropy = -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
    local_difficulty = neighbor_entropy.sum(1)

    print('done')
    return local_difficulty.to(args.device)

def difficulty_measurer(data, adj, label, args):
    local_difficulty = neighborhood_difficulty_measurer(data, adj, label, args)
    node_difficulty = local_difficulty 
    return node_difficulty


def sort_training_nodes(data, adj, label, args):
    node_difficulty = difficulty_measurer(data, adj, label, args)
    return node_difficulty


def difficulty_measurer_in(data, adj, label, args):
    local_difficulty = neighborhood_difficulty_measurer_in(data, adj, label, args)
    node_difficulty = local_difficulty 
    return node_difficulty


def sort_training_nodes_in(data, adj, label, args):
    node_difficulty = difficulty_measurer_in(data, adj, label, args)
    return node_difficulty 

def do_geom_sort_nodes(data,args):
    if args.dataset in ['flickr','reddit']:
        features, adj, labels = data.feat_train, data.adj_train, data.labels_train
        features_sort, adj_sort, labels_sort = data.feat_full, data.adj_full, data.labels_full
        
        adj, features, labels = to_tensor(adj, features, labels, device=args.device)
        adj_sort, features_sort, labels_sort = to_tensor(adj_sort, features_sort, labels_sort, device=args.device)

        if is_sparse_tensor(adj):
            adj_norm = normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = normalize_adj_tensor(adj)
        
        adj = adj_norm
        sorted_scores = sort_training_nodes_in(data, adj, labels, args) #sorted_trainset

    else:
        features, adj, labels = data.feat_train, data.adj_train, data.labels_train
        adj, features, labels = to_tensor(adj, features, labels, device=args.device)

        if is_sparse_tensor(adj):
            adj_norm = normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = normalize_adj_tensor(adj)

        adj = adj_norm
        sorted_scores = sort_training_nodes_in(data, adj, labels, args) # sorted_scores = sort_training_nodes(data, adj, labels, args) #sorted_trainset

    return sorted_scores

# hop
def make_hop_a(adj_matrix,args_hop,args):
    if not os.path.exists('hop'+str(args.hop)+' '+args.dataset+' '+args.hop_type+'.pt'):
        print('0')
        adj_matrix = torch.tensor(adj_matrix).to_sparse()
        print('1')
        adj_matrix += torch.eye(*adj_matrix.shape).to_sparse().to(torch.int8)
        print('2')
        #print(adj_matrix.dtype)
        for single_hop in range(1,args_hop):
            if single_hop ==1:
                new_adj_matrix_before = adj_matrix
                print('3')
                new_adj_matrix_after = (torch.mm(new_adj_matrix_before, new_adj_matrix_before.to_dense()).to(torch.int8).to_dense()>0).to_sparse().to(torch.int8) #.to(torch.float)
                print('4')
                final_adj_matrix = adj_matrix+((single_hop+1)*(new_adj_matrix_after - new_adj_matrix_before)).to_sparse()
                print('5')
            else:
                new_adj_matrix_before = new_adj_matrix_after
                print('6')
                new_adj_matrix_after = torch.tensor((sp.sparse.csr_matrix(new_adj_matrix_before.to_dense().numpy()) * sp.sparse.csr_matrix(new_adj_matrix_before.dense().to_dense().numpy())).todense()>0).to_sparse().to(torch.int8)
                print('7')
                final_adj_matrix += ((single_hop+1)*(new_adj_matrix_after - new_adj_matrix_before)).to_sparse()
                print('8')
        if args.hop_type == 'exp':
            final_adj_matrix = ((args.hop_coefficient.to_dense())**(final_adj_matrix.to_dense()))#.to_sparse()
            torch.save(final_adj_matrix, 'hop'+str(args.hop)+' '+args.dataset+' '+args.hop_type+'.pt')
        elif args.hop_type == 'reverse':
            final_adj_matrix = (1./final_adj_matrix.to_dense())#.to_sparse()
            print('9')
            final_adj_matrix[final_adj_matrix>1.] = 0.
            final_adj_matrix = final_adj_matrix.to_sparse()
        else:
            raise NotImplementedError
        torch.save(final_adj_matrix.to_sparse(), 'hop'+str(args.hop)+' '+args.dataset+' '+args.hop_type+'.pt')
    else:
        final_adj_matrix = torch.load('hop'+str(args.hop)+' '+args.dataset+' '+args.hop_type+'.pt')#.to_dense()
    print('hopping finish')
    return (final_adj_matrix - torch.eye(*adj_matrix.shape).to_sparse())#.to_dense().numpy() #.to(torch.int8)


## Define Kernel ##
def linear_kernel(cov1, nngp, cov2, ntk, args, 
                  W_std=torch.sqrt(torch.tensor(2)).to(torch.float), #onp.sqrt(2),
                  b_std=torch.tensor(0.1).to(torch.float),
                  parameterization='ntk'):
  cov1, nngp, cov2, ntk = cov1, nngp, cov2, ntk
  W_std = W_std.to(args.device)
  b_std = b_std.to(args.device)
  def _affine(
      mat,
      W_std,
      b_std):

    if mat is not None:
      mat = mat* W_std**2

      if b_std is not None:
        mat = mat + b_std**2

    return mat

  def fc(x):
    return _affine(x, W_std, b_std)

  if parameterization == 'ntk':
    cov1, nngp, cov2 = map(fc, (cov1, nngp, cov2))
    if ntk is not None:
      ntk = nngp + W_std**2 * ntk
  elif parameterization == 'standard':
    raise NotImplementedError

  return cov1, nngp, cov2, ntk


def _sqrt(y, tol=0.):
    near_zeros = y < 1e-10
    y = y * (near_zeros.logical_not())
    y = y + (near_zeros * torch.tensor(1e-10).to(torch.float))
    return torch.sqrt(torch.maximum(y, torch.tensor(tol).to(torch.float))) # onp;onp
def _arctan2(x, y, fill_zero: Optional[float] = None):
    if fill_zero is not None:
        near_zeros = y < 1e-10
        y = y * (near_zeros.logical_not())
        y = y + (near_zeros * torch.tensor(1e-10).to(torch.float))
        return torch.where(torch.bitwise_and(x == 0., y == 0.), # onp, onp
                    fill_zero,
                    torch.arctan2(x, y)) # onp
    near_zeros = y < 1e-10
    y = y * (near_zeros.logical_not())
    y = y + (near_zeros * torch.tensor(1e-10).to(torch.float))
    return torch.arctan2(x, y) # onp


_ArrayOrShape = TypeVar('_ArrayOrShape',
                        onp.ndarray,
                        torch.Tensor, 
                        List[int],
                        Tuple[int, ...])

def get_diagonal_outer_prods(
    cov1,#: jnp.ndarray,
    cov2,#: Optional[jnp.ndarray],
    diagonal_batch,#: bool, true
    diagonal_spatial,#: bool, false
    operation,#: Callable[[float, float], float],
    axis=[],#: Sequence[int] = (),
    mask1=None,#: Optional[jnp.ndarray] = None,
    mask2=None,#: Optional[jnp.ndarray] = None
):#-> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

  axis = [] # None # [] #utils.canonicalize_axis(axis, cov1) # will be changed if use conv net
  def get_diagonal(
      cov, #: Optional[jnp.ndarray],
      diagonal_batch, #: bool,
      diagonal_spatial, #: bool
  ): # -> Optional[jnp.ndarray]:

    if cov is None:
      return cov
    def _zip_axes(x: torch.Tensor, # onp.ndarray,
                  start_axis: int = 0,
                  end_axis: Optional[int] = None,
                  unzip: bool = False) -> torch.Tensor: #onp.ndarray:

      if end_axis is None:
        end_axis = x.ndim

      half_ndim, ragged = divmod(end_axis - start_axis, 2)
      if ragged:
        raise ValueError(
            f'Need even number of axes to zip, got {end_axis - start_axis}.')

      odd_axes = range(start_axis + 1, end_axis, 2)
      last_axes = range(end_axis - half_ndim, end_axis)

      if unzip:
        x = torch.moveaxis(x, list(odd_axes), list(last_axes)) # onp
      else:
        x = torch.moveaxis(x, list(last_axes), list(odd_axes)) # onp
      return x

    batch_ndim = 1 # if diagonal_batch else 2
    start_axis = 2 - batch_ndim
    end_axis = batch_ndim # if diagonal_spatial else cov.ndim
    cov = _zip_axes(cov, start_axis, end_axis, unzip=True)#utils.unzip_axes(cov, start_axis, end_axis)

    def diagonal_between(x,#: jnp.ndarray,
                        start_axis,#: int = 0,
                        end_axis,#: Optional[int] = None
                         ):# -> jnp.ndarray:
      """Returns the diagonal along all dimensions between start and end axes."""
      if end_axis is None:
        end_axis = x.ndim

      half_ndim, ragged = divmod(end_axis - start_axis, 2)
      if ragged:
        raise ValueError(
            f'Need even number of axes to flatten, got {end_axis - start_axis}.')
      if half_ndim == 0:
        return x

      side_shape = x.shape[start_axis:start_axis + half_ndim]
      def size_at(
          x: Optional[_ArrayOrShape,], # jax.ShapedArray],
          axes: Optional[Iterable[int]] = None
      ) -> int:
        if hasattr(x, 'shape'):
          x = x.shape

        if axes is None:
          axes = range(len(x))

        return functools.reduce(op.mul, [x[a] for a in axes], 1)
      side_size = size_at(side_shape)

      shape_2d = x.shape[:start_axis] + (side_size, side_size) + x.shape[end_axis:]
      shape_result = x.shape[:start_axis] + side_shape + x.shape[end_axis:]

      x = torch.diagonal(x.reshape(shape_2d), axis1=start_axis, axis2=start_axis+1) # onp
      x = torch.moveaxis(x, -1, start_axis) # onp
      return x.reshape(shape_result)

    return diagonal_between(cov, start_axis, end_axis)

  cov1 = get_diagonal(cov1, diagonal_batch, diagonal_spatial)
  cov2 = get_diagonal(cov2, diagonal_batch, diagonal_spatial)

  def mean_and_var(
      x: Optional[torch.Tensor],#[onp.ndarray],
      axis: Optional[int] = None, #Axis
      dtype: Optional[torch.dtype] = None, # onp
      out: Optional[None] = None,
      ddof: int = 0,
      keepdims: bool = False,
      mask: Optional[torch.Tensor] = None, # onp.ndarray
      get_var: bool = False
  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: #[Optional[onp.ndarray], Optional[onp.ndarray]]
    var = None
    if x is None:
      return x, var

    if mask is None:
      if axis==[]:
        mean = x
      else:
        print('gogogo')
        mean = torch.mean(x, dim=axis, keepdim=keepdims, dtype=dtype, out=out)# torch.mean(x, axis, dtype, out, keepdims) #onp
      if get_var:
        var =  torch.var(x, dim=axis, keepdim=keepdims, dtype=dtype, out=out)# torch.var(x, axis, dtype, out, ddof, keepdims) # onp

    else:
      raise NotImplementedError

    return mean, var

  cov1, _ = mean_and_var(cov1, axis=axis, keepdims=True, mask=mask1)
  cov2, _ = mean_and_var(cov2, axis=axis, keepdims=True, mask=mask2)

  end_axis = 1 # if diagonal_spatial else cov1.ndim # but here is cov1.ndim, just =1 coincidently

  def interleave_ones(x, start_axis, end_axis, x_first):
    x_axes = x.shape[start_axis:end_axis]
    ones = (1,) * (end_axis - start_axis)
    shape = x.shape[:start_axis]
    def zip_flat(x, y):
      return tuple(c for xy in zip(x, y) for c in xy)
    if x_first:
      shape += zip_flat(x_axes, ones)
    else:
      shape += zip_flat(ones, x_axes)
    shape += x.shape[end_axis:]
    return x.reshape(shape)


  def outer_prod(x, y, start_axis, end_axis, prod_op):
    if y is None:
      y = x
    x = interleave_ones(x, start_axis, end_axis, True)
    y = interleave_ones(y, start_axis, end_axis, False)
    #print('x y: ',x.shape,y.shape)
    return (torch.matmul(x,y))#[...,None]#prod_op(x, y) # onp.dot
    
  prod12 = outer_prod(cov1, cov2, 0, end_axis, operation) #utils.

  start_axis = 1 # if diagonal_batch else 0
  prod11 = outer_prod(cov1, cov1, start_axis, end_axis, operation) #utils.
  prod22 = (outer_prod(cov2, cov2, start_axis, end_axis, operation) #utils.
            if cov2 is not None else prod11)
  return prod11, prod12, prod22
  

def ReLU_kernel(cov1, nngp, cov2, ntk, args, 
                  W_std=torch.sqrt(torch.tensor(2)).to(torch.float), #onp.sqrt(2),
                  b_std=torch.tensor(0.1).to(torch.float),
                  a=0,b=1,
                  parameterization='ntk', do_stabilize=False,
                  diagonal_batch=True, diagonal_spatial=False):
    W_std = W_std.to(args.device)
    if do_stabilize:
      raise NotImplementedError 

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1,
                                                      cov2,
                                                      diagonal_batch,
                                                      diagonal_spatial,
                                                      op.mul)
    def nngp_ntk_fn(nngp, prod, ntk=None):
      square_root = _sqrt(prod - nngp**2)
      angles = _arctan2(square_root, nngp, fill_zero=torch.tensor(onp.pi / 2).to(torch.float))

      factor = (a - b)**2 / (2 * torch.tensor(onp.pi).to(torch.float))
      dot_sigma = (a**2 + b**2) / 2 - factor * angles
      nngp = factor * square_root + dot_sigma * nngp

      if ntk is not None:
        ntk *= dot_sigma

      return nngp, ntk

    def nngp_fn_diag(nngp):
      return (a**2 + b**2) / 2 * nngp

    nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk=ntk)

    if diagonal_batch and diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22)

    if do_stabilize:
      raise NotImplementedError

    return cov1,nngp,cov2, ntk


def a_init_to_param(a_init,args,inference=True): #sigmoid((log(a)-log(1-a)+omegaij)/temp) #
    a = torch.rand_like(a_init)
    if args.is_doscond_sigmoid:
        return torch.sigmoid((torch.log(a)-torch.log(1.-a)+a_init)/torch.tensor(args.temp,requires_grad=False))#+torch.eye(*a_init.size(),requires_grad=False)
    else:
        return torch.sigmoid(a_init/torch.tensor(args.temp,requires_grad=False)) #

# aggre layer
def aggregate_kernel(cov1, nngp, cov2, ntk, A1, A2, args,
                  is_kts=False,ind=None,
                  parameterization='ntk', do_stabilize=False,
                  diagonal_batch=True, diagonal_spatial=False,
                  is_first_layer = False, a_target_sum=None):
    # 2 strategy to represent A (Bernoulli and continuous)
    if args.is_a_target_sum and is_kts: 
        def add_all_parent(A1,mat,A2):
            if args.is_sparse_adj and A1.shape!=A2.shape:
                new_mat = torch.matmul(torch.sparse.mm(A1.T,mat),A2)
            else:
                new_mat = torch.matmul(torch.matmul(A1.T,mat),A2)
            return new_mat #torch.matmul(A1.T,new_mat) - ((A1.sum(0,keepdim=True)-1).T)*torch.matmul(A1.T,mat)
            
        nngp = add_all_parent(A1,nngp,A2)/a_target_sum.reshape(-1,1)/A2.sum(0,keepdim=True)
        if not is_first_layer:
            ntk = add_all_parent(A1,ntk,A2)/a_target_sum.reshape(-1,1)/A2.sum(0,keepdim=True)
        if args.is_sparse_adj and A1.shape!=A2.shape:
            cov1 = torch.sparse.mm(A1.T,cov1)/a_target_sum
        else:
            cov1 = torch.matmul(A1.T,cov1)/a_target_sum
        cov2 = torch.matmul(A2.T,cov2)/A2.sum(0,keepdim=False) # make sure cov is (-1,1) shape
    else: #elif not args.is_a_target_sum: #args.a_strategy == 'Continuous' or args.a_strategy == 'Keep_init' or args.a_strategy == 'Identity_a_fix':
        def add_all_parent(A1,mat,A2):      
            if args.is_sparse_adj and A1.shape!=A2.shape:
                new_mat = torch.matmul(torch.sparse.mm(A1.T,mat),A2)
            else:
                new_mat = torch.matmul(torch.matmul(A1.T,mat),A2)
            return new_mat #torch.matmul(A1.T,new_mat) - ((A1.sum(0,keepdim=True)-1).T)*torch.matmul(A1.T,mat)
        nngp = add_all_parent(A1,nngp,A2)/A1.to_dense().sum(0,keepdim=True).T/A2.sum(0,keepdim=True)
        if not is_first_layer:
            ntk = add_all_parent(A1,ntk,A2)/A1.to_dense().sum(0,keepdim=True).T/A2.sum(0,keepdim=True)
        if args.is_sparse_adj and A1.shape!=A2.shape:
            cov1 = torch.sparse.mm(A1.T,cov1.view(-1,1)).view(-1)/A1.to_dense().sum(0,keepdim=False)
        else:
            cov1 = torch.matmul(A1.T,cov1)/A1.sum(0,keepdim=False)
        cov2 = torch.matmul(A2.T,cov2)/A2.sum(0,keepdim=False) # make sure cov is (-1,1) shape
    return cov1,nngp,cov2,ntk   


def layernorm_kernel(cov1, nngp, cov2, ntk, args, channel_axis=1,batch_axis=0,is_reversed=False,diagonal_batch=True, diagonal_spatial=False):
    eps = torch.tensor(1e-12).to(args.device)
    ndim = 2 #len(k.shape1) # 2
    _channel_axis = channel_axis % ndim # 1
    _batch_axis = batch_axis % ndim # 0
    _axis = [1] #utils.canonicalize_axis(axis, k.shape1) #[1]

    if _channel_axis not in _axis:
        raise ValueError(f'Normalisation over channels (axis {_channel_axis})'
                       f'necessary for convergence to an asymptotic kernel; '
                       f'got axis={_axis}.')

    _axis.remove(_channel_axis)
    spatial_axes = tuple(i for i in range(ndim) # len(k.shape1)
                         if i not in (_channel_axis, batch_axis))

    # Batch axis
    if _batch_axis in _axis:
        kernel_axis = (0,)
        _axis.remove(_batch_axis)
    else:
        kernel_axis = ()

    # Spatial axes
    kernel_axis += tuple(
        1 + spatial_axes[::(-1 if is_reversed else 1)].index(i)
        for i in _axis)
    # Prepare masks for normalization
    def prepare_mask(m):
        if m is None:
            return m

        if m.shape[channel_axis] != 1:
            raise NotImplementedError('`LayerNorm` with different per-channel masks'
                                  'not implemented in the infinite limit.')

        m = torch.squeeze(m, channel_axis) #jnp
        if is_reversed:
            m = torch.moveaxis(m, range(1, m.ndim), range(m.ndim - 1, 0, -1)) #jnp

        return m

    prod11, prod12, prod22 = get_diagonal_outer_prods(
        eps + cov1,
        cov2 if cov2 is None else eps + cov2,
        diagonal_batch,#k.diagonal_batch,
        diagonal_spatial,#k.diagonal_spatial,
        op.mul
        )

    nngp /= torch.sqrt(prod12) #jnp

    if ntk is not None:
        ntk /= torch.sqrt(prod12) #jnp

    cov1 /= torch.sqrt(prod11) #jnp
    if cov2 is not None:
        cov2 /= torch.sqrt(prod22) #jnp

    return cov1, nngp, cov2, ntk #k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)


#data preprocessed func

def class_balanced_sample(sample_size: int, args: str,
                          labels: onp.ndarray, 
                          *arrays: onp.ndarray, **kwargs: int):

  if labels.ndim != 1:
    raise ValueError(f'Labels should be one-dimensional, got shape {labels.shape}')
  n = len(labels)
  if not all([n == len(arr) for arr in arrays[1:]]):
    raise ValueError(f'All arrays to be subsampled should have the same length. Got lengths {[len(arr) for arr in arrays]}')
  classes = onp.unique(labels)
  n_classes = len(classes)
  n_per_class, remainder = divmod(sample_size, n_classes)
  if remainder != 0:
    print(f'Number of classes {n_classes} in labels must divide sample size {sample_size}.')
    print('So do n_per_class +1.')
    n_per_class+=1
  if kwargs.get('seed') is not None:
    onp.random.seed(kwargs['seed'])
  _, class_counts = torch.tensor(LABELS_TRAIN).unique(return_counts=True)
  class_counts = class_counts.numpy()
 
  if args.is_largest_a_target_sum_init:
      all_ind = []
      for i,c in enumerate(classes):
          ind_data_onp = onp.where(labels==c)[0]
          c_a_target_sum_m = A_SUM_TRAIN[ind_data_onp] #  get a target sum belonging to class c
          order_temp_a_target_sum_m = onp.argsort(c_a_target_sum_m.numpy())[::-1] # get sorted index from a target sum belonging to class c
          new_ind_data_onp = ind_data_onp[order_temp_a_target_sum_m]
          all_ind+=list(new_ind_data_onp[:n_per_class if class_counts[c]>=n_per_class else class_counts[c]])
      inds = onp.array(all_ind)
      print('1: ',onp.unique(labels[inds], return_counts=True)[1])
  else:
      for i,c in enumerate(classes):
          if i==0:
              real_n_per_class = n_per_class if class_counts[c]>=n_per_class else class_counts[c]
              inds = onp.random.choice(onp.where(labels == c)[0], real_n_per_class, replace=False)
          else:
              real_n_per_class = n_per_class if class_counts[c]>=n_per_class else class_counts[c]
              inds = onp.concatenate([inds,
                  onp.random.choice(onp.where(labels == c)[0], real_n_per_class, replace=False)
              ])
      print('2: ',onp.unique(labels[inds], return_counts=True)[1])
  current_n = len(inds)
  print('current_n: ',current_n)
  def argmax_last(a,func):
    b = a[::-1]
    return len(b) - func(b) - 1
  while current_n>sample_size:
      temp_labels = LABELS_TRAIN[inds]
      temp_cc,temp_nn = onp.unique(temp_labels, return_counts=True)
      if args.is_delete_min:
        max_class =  argmax_last(temp_nn,onp.argmin) if args.is_return_last else onp.argmin(temp_nn)
      else:
        max_class = argmax_last(temp_nn,onp.argmax) if args.is_return_last else onp.argmax(temp_nn)
      temp_max_labels = temp_labels[temp_labels==max_class]
      temp_max_inds = inds[temp_labels==max_class]
      temp_not_max_inds = inds[temp_labels!=max_class]
      if kwargs['seed']!=8:
         onp.random.shuffle(temp_max_inds)
      inds = onp.concatenate((temp_max_inds[:-1],temp_not_max_inds))
      current_n-=1
  # having deletion to origin sample size if remainder >0
  print('current_n after deletion: ',current_n)
  print('3: ',onp.unique(labels[inds], return_counts=True)[1])
 
  return (inds, labels[inds].copy()) + tuple(
      [arr[inds].copy() for arr in arrays])


def generate_labels_syn(labels_m,args):
    from collections import Counter
    counter = Counter(labels_m)
    num_class_dict = {}
    n = len(labels_m)
    reduction_rate = args.support_size/float(n)

    sorted_counter = sorted(counter.items(), key=lambda x:x[1])
    sum_ = 0
    labels_syn = []
    syn_class_indices = {}
    for ix, (c, num) in enumerate(sorted_counter):
        if ix == len(sorted_counter) - 1:
            num_class_dict[c] = int(n * reduction_rate) - sum_
            syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]
        else:
            num_class_dict[c] = max(int(num * reduction_rate), 1)
            sum_ += num_class_dict[c]
            syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]
    num_class_dict = num_class_dict
    return labels_syn,num_class_dict


def generate_all_syn(data_m, a_m, a_target_sum_m, labels_m, y_m, args, seed=8, data=None):
    labels_syn, syn_class_indices = generate_labels_syn(labels_m,args)
    all_ind = []
    print('labels_syn.shape: ',len(labels_syn))

    if args.init_way == 'Center':
        for c in range(data.nclass):
            features_c = data.feat_train[data.labels_train == c]
            kmeans_init = KMeans(n_clusters=1, random_state=args.seed, n_init='auto', verbose=1)
            labels_init = kmeans_init.fit_predict(features_c)
            feature_init[c] = kmeans_init.cluster_centers_
            ind = syn_class_indices[c]
            feat_syn[ind[0]: ind[1]] = torch.tensor(feature_init[c])
    
    elif args.init_way == 'K-Center':
        for c in range(data.nclass):
            features_c = data.feat_train[data.labels_train == c]
            ind = syn_class_indices[c]
            n_clu = ind[1] - ind[0]
            kmeans_init = KMeans(n_clusters=n_clu, random_state=args.seed, n_init='auto', verbose=1)
            labels_init = kmeans_init.fit_predict(features_c)
            feature_init[c] = kmeans_init.cluster_centers_
            feat_syn[ind[0]: ind[1]] = torch.tensor(feature_init[c])
        
    elif args.init_way == 'Random_real':
        for c in range(data.nclass):
            features_c = data.feat_full[data.labels_full == c]
            ind = syn_class_indices[c]
            num = ind[1] - ind[0]
            feat_syn[ind[0]: ind[1]] = torch.tensor(np.random.permutation(features_c)[:num])
    elif args.init_way == 'K-means':
        cc,nn = onp.unique(labels_syn, return_counts=True)
        for c in range(data.nclass):
            features_c = data.feat_train[data.labels_train == c]
            _, where_feature_c = onp.where([data.labels_train == c])
            n_clu = syn_class_indices[c]#ind[1] - ind[0]
            kmeans_init = KMeans(n_clusters=n_clu, random_state=args.seed, n_init='auto', verbose=0)
            labels_init = kmeans_init.fit_predict(features_c)
            selected_indices = []
            for cluster_label in range(n_clu):
                cluster_indices = onp.where(labels_init == cluster_label)[0]
                selected_index = onp.random.choice(cluster_indices)
                selected_indices.append(selected_index)
            selected_features_where = where_feature_c[selected_indices]
            all_ind.append(list(selected_features_where))

    else:
        cc,nn = onp.unique(labels_syn, return_counts=True)
        for c,n in zip(cc,nn):
            ind_data_onp = onp.where(labels_m==c)[0]
            if not args.is_largest_a_target_sum_init:
                onp.random.seed(seed)
                onp.random.shuffle(ind_data_onp)
                all_ind.append(list(ind_data_onp[:n]))
            else: # get max a target sum points as initial
                c_a_target_sum_m = a_target_sum_m[ind_data_onp] #  得到 a target sum中类别为c的
                order_temp_a_target_sum_m = onp.argsort(c_a_target_sum_m.numpy())[::-1] # 得到 a target sum中类别为c的如果sort了的index
                new_ind_data_onp = ind_data_onp[order_temp_a_target_sum_m]
                all_ind.append(list(new_ind_data_onp[:n]))
    # get real data by inde
    all_ind = onp.array(all_ind)
    print('a_syn shape at the start: ',len(onp.hstack(all_ind)))
    while len(onp.hstack(all_ind))<args.support_size:
        minimum_class = onp.argmin(nn)
        ind_data_raw = onp.where(labels_m==minimum_class)[0]
        ind_data_raw_minus = onp.array([x for x in ind_data_raw.tolist() if x not in all_ind]) 
        number_this_class = len(all_ind[minimum_class])
        onp.random.shuffle(ind_data_raw_minus)
        all_ind[minimum_class] = onp.array(all_ind[minimum_class]+ind_data_raw_minus[:1].tolist())#ind_data_raw[:number_this_class+1]
        nn[minimum_class]+=1

    all_ind = onp.hstack(all_ind)
    data_syn = data_m[all_ind]
    a_syn = a_m[all_ind]
    a_syn = a_syn[:,all_ind]
    print('a_syn shape after the start: ',a_syn.shape)
    labels_syn = labels_m[all_ind]
    y_syn = y_m[all_ind]
    return labels_syn, data_syn, y_syn, a_syn


def pytorch_kernel_fn(x1,x2,A1,A2, args, is_kts=False,ind=None,a_target_sum=None): # 1 and 2 opposite as custom loss
    
    if args.architecture == 'FC':
        # first clear not related part in Aij
        if type(ind)==torch.Tensor:
            if args.keep_batch_edge == False:  
                A1 = torch.tensor(A1)
            else:
                pass
        if ind is not None: 
            ind = ind.to(x1.device)
        if args.is_sparse_adj and A1.shape!=A2.shape:
            A1 = ((torch.eye(*A1.size(),dtype=torch.int8).to_sparse()+A1.to_sparse()).to(x1.device).to(torch.float))
        else:
            A1 = (torch.eye(*A1.size(),dtype=torch.int8).to(x1.device)+A1.to(x1.device)).to(torch.float)
        A2 = (torch.eye(*A2.size(),dtype=torch.int8).to(x1.device)+A2.to(x1.device))
        nngp = (torch.matmul(x1.reshape(len(x1), -1), x2.reshape(len(x2), -1).T)/x1.reshape(len(x1), -1).shape[1]) # onp.dot
        cov1 = (torch.diag((torch.matmul(x1.reshape(len(x1) ,-1), x1.reshape(len(x1) ,-1).T)/x1.reshape(len(x1) ,-1).shape[1]))) # onp.dot
        cov2 = (torch.diag((torch.matmul(x2.reshape(len(x2) ,-1), x2.reshape(len(x2) ,-1).T)/x2.reshape(len(x2) ,-1).shape[1]))) # onp.dot
        ntk = torch.zeros_like(nngp).to(args.device)
        #cov1, nngp, cov2, ntk = layernorm_kernel(cov1, nngp, cov2, ntk, args)
        if args.is_accumulate:
            total_ntk = ntk
        for layer in range(args.ntk_layers):
            if type(ind)!=torch.Tensor:
                cov1, nngp, cov2, ntk = aggregate_kernel(cov1, nngp, cov2, ntk, A1, A2, args, is_kts=is_kts, is_first_layer=True, a_target_sum=a_target_sum)
            else:
                cov1, nngp, cov2, ntk = aggregate_kernel(cov1, nngp, cov2, ntk, A1, A2, args, is_kts=is_kts,ind=ind, is_first_layer=True, a_target_sum=a_target_sum)
            cov1, nngp, cov2, ntk = linear_kernel(cov1, nngp, cov2, ntk, args)
            
            if args.is_use_layer_norm:
                cov1, nngp, cov2, ntk = layernorm_kernel(cov1, nngp, cov2, ntk, args)
            cov1, nngp, cov2, ntk = ReLU_kernel(cov1, nngp, cov2, ntk, args)
            if args.is_accumulate:
                total_ntk += ntk
        return ntk if not args.is_accumulate else total_ntk

    else:
        raise NotImplementedError

def get_discrete_graphs(adj, args, inference=True):
    if args.is_doscond_sigmoid:
        if not hasattr(args, 'cnt'):
            args.cnt = 0

        if not inference:
            N = adj.size()[1]
            vals = torch.rand(N * (N+1) // 2)
            vals = vals.view(-1).to(torch.float).to(args.device)
            i, j = torch.triu_indices(N, N)
            epsilon = torch.zeros_like(adj).to(torch.float).to(args.device)
            epsilon[i, j] = vals

            tmp = (torch.log(epsilon) - torch.log(1-epsilon)).to(args.device)
            adj = tmp + adj.to(args.device)
            t0 = 1
            tt = 0.01
            end_iter = 200
            t = t0*(tt/t0)**(args.cnt/end_iter)
            if args.cnt == end_iter:
                print('===reached the end of anealing...')
            args.cnt += 1

            t = max(t, tt)
            adj = torch.sigmoid(adj/t).to(torch.float).to(args.device)
        else:
            adj = torch.sigmoid(adj).to(torch.float).to(args.device)
        return adj
    elif args.is_relu_adj:
        return torch.nn.functional.relu(adj).to(torch.float).to(args.device)
    elif args.is_none_adj:
        return adj.to(args.device)
    else:
        raise NotImplementedError


def custom_loss(x_support, y_support, x_target, y_target, A_support, A_target, ind=None, reg=None, labels_counts=None, a_target_sum=None, args=None, difficult_scores_batch=None, LABELS_TRAIN=None, data=None, Y_ALL=None, Y_TRAIN=None):

    k_ss = pytorch_kernel_fn(x_support, x_support, A_support, A_support, args)
    x_target, y_target = x_target.to(torch.float).to(args.device), y_target.to(torch.float).to(args.device)

    if type(ind)!=onp.ndarray:
        k_ts = pytorch_kernel_fn(x_target, x_support, A_target, A_support, args, is_kts=True, a_target_sum=a_target_sum)
    else:
        # k_ts = pytorch_kernel_fn(x_target, x_support, A_target.to('cpu'), A_support, args, is_kts=True,ind=torch.tensor(ind).to('cpu'), a_target_sum=a_target_sum) 
        k_ts = pytorch_kernel_fn(x_target, x_support, A_target, A_support, args, is_kts=True,ind=torch.tensor(ind), a_target_sum=a_target_sum)
 
    k_ss_reg = (k_ss + torch.abs(reg) * torch.trace(k_ss) * (torch.eye(k_ss.shape[0]) / k_ss.shape[0]).to(args.device))
    if args.is_y_pred_softmax:
        #pred = torch.softmax(torch.matmul(k_ts, torch.linalg.solve(k_ss_reg.to(torch.float), y_support.to(torch.float)).to(torch.float).to(args.device)),-1) mistake to all softmax
        pred = torch.matmul(k_ts, torch.linalg.solve(k_ss_reg.to(torch.float), torch.softmax(y_support/args.temp,-1).to(torch.float)).to(torch.float).to(args.device)) 
    elif args.is_relu_y:
        pred = torch.matmul(k_ts, torch.linalg.solve(k_ss_reg.to(torch.float), torch.nn.functional.relu(y_support).to(torch.float)).to(torch.float).to(args.device)) 
    else:
        pred = torch.matmul(k_ts, torch.linalg.solve(k_ss_reg.to(torch.float), y_support.to(torch.float)).to(torch.float).to(args.device)) # sp.linalg.solve(k_ss_reg, y_support, sym_pos=True)) # jnp
    if args.is_class_coefficient:
        origin_loss = torch.nn.functional.mse_loss(pred, y_target,reduction='none')
        mse_loss = args.loss_coeff*((((labels_counts.repeat(y_target.shape[0],1)*y_target).max(-1))[0].reshape(-1,1))*origin_loss).sum() #*0.5
    elif args.is_kl_div:
        mse_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)(pred, y_target)
    elif args.is_mean_loss:
        if args.is_use_clscore_sort:
            mse_loss = args.loss_coeff*torch.sum(((pred - y_target) ** 2)*difficult_scores_batch.to(args.device))/args.difficult_scores_total # 0.5 jnp
            if args.dataset in ['reddit']:
                for cls_iter in range(len(onp.unique(LABELS_TRAIN))):
                    count_idx_cls = torch.tensor(onp.where(data.labels_train==cls_iter)[0])#.to(args.device)# torch.tensor(onp.intersect1d(data.idx_train,onp.arange(len(data.labels_full))[onp.where(data.labels_full==cls_iter)])).to(args.device)#data.idx_train[onp.where(data.labels_train==cls_iter)]
                    if type(ind)!=onp.ndarray:
                        k_ts = pytorch_kernel_fn(torch.tensor(data.feat_train[onp.where(data.train_full==cls_iter)]).to(args.device), x_support, torch.tensor(data.adj_train.astype(onp.int8)[onp.ix_(data.labels_train==cls_iter,data.labels_train==cls_iter)].toarray()).to(args.device), A_support, args, is_kts=True, a_target_sum=a_target_sum)
                    else: 
                        k_ts = pytorch_kernel_fn(torch.tensor(data.feat_train[onp.where(data.labels_train==cls_iter)]).to(torch.float).to(args.device), x_support, torch.tensor(data.adj_train.astype(onp.int8)[onp.ix_(data.labels_train==cls_iter,data.labels_train==cls_iter)].toarray()), A_support, args, is_kts=True,ind=torch.tensor(ind), a_target_sum=a_target_sum)
                    if args.is_kss_class_fix:
                        pass
                    else:
                        pred = torch.matmul(k_ts, torch.linalg.solve(k_ss_reg.to(torch.float), y_support.to(torch.float)).to(torch.float).to(args.device)) # sp.linalg.solve(k_ss_reg, y_support, sym_pos=True)) # jnp
                    mse_loss += args.loss_coeff*torch.sum((pred - (torch.tensor(Y_TRAIN)[count_idx_cls].to(args.device))) ** 2)/(pred.shape[0]) 
                    count_idx_cls = count_idx_cls.detach().numpy()
                    del count_idx_cls
        else:
            mse_loss = args.loss_coeff*torch.sum((pred - y_target) ** 2)/(pred.shape[0]) # 0.5 jnp        
            if args.dataset in ['reddit']:
                for cls_iter in range(len(onp.unique(LABELS_TRAIN))):
                    count_idx_cls = torch.tensor(onp.where(data.labels_train==cls_iter)[0])#.to(args.device)# torch.tensor(onp.intersect1d(data.idx_train,onp.arange(len(data.labels_full))[onp.where(data.labels_full==cls_iter)])).to(args.device)#data.idx_train[onp.where(data.labels_train==cls_iter)]
                    if type(ind)!=onp.ndarray:
                        k_ts = pytorch_kernel_fn(torch.tensor(data.feat_train[onp.where(data.train_full==cls_iter)]).to(args.device), x_support, torch.tensor(data.adj_train.astype(onp.int8)[onp.ix_(data.labels_train==cls_iter,data.labels_train==cls_iter)].toarray()).to(args.device), A_support, args, is_kts=True, a_target_sum=a_target_sum)
                    else: 
                        k_ts = pytorch_kernel_fn(torch.tensor(data.feat_train[onp.where(data.labels_train==cls_iter)]).to(torch.float).to(args.device), x_support, torch.tensor(data.adj_train.astype(onp.int8)[onp.ix_(data.labels_train==cls_iter,data.labels_train==cls_iter)].toarray()), A_support, args, is_kts=True,ind=torch.tensor(ind), a_target_sum=a_target_sum)
                    if args.is_kss_class_fix:
                        pass
                    else:                    
                        pred = torch.matmul(k_ts, torch.linalg.solve(k_ss_reg.to(torch.float), y_support.to(torch.float)).to(torch.float).to(args.device)) # sp.linalg.solve(k_ss_reg, y_support, sym_pos=True)) # jnp
                    mse_loss += args.loss_coeff*torch.sum((pred - (torch.tensor(Y_TRAIN)[count_idx_cls].to(args.device))) ** 2)/(pred.shape[0]) 
                    count_idx_cls = count_idx_cls.detach().numpy()
                    del count_idx_cls

    else:
        if args.is_use_clscore_sort:
            mse_loss = args.loss_coeff*torch.sum(((pred - y_target) ** 2)*difficult_scores_batch.to(args.device)) # 0.5 jnp
        else:
            mse_loss = args.loss_coeff*torch.sum((pred - y_target) ** 2) # 0.5 jnp
    with torch.no_grad():
        acc = 0.0 #torch.mean((torch.argmax(pred, axis=1) == torch.argmax(y_target, axis=1)).to(torch.float)) # jnp jnp jnp
    return mse_loss, acc

def custom_loss_transduct(x_support, y_support, x_target, y_target, A_support, A_target, ind=None, reg=None, labels_counts=None, a_target_sum=None, args=None, difficult_scores_batch=None, counted_idx=None, LABELS_TRAIN=None, data=None, Y_ALL=None):

    k_ss = pytorch_kernel_fn(x_support, x_support, A_support, A_support, args)
    x_target, y_target = x_target.to(torch.float).to(args.device), y_target.to(torch.float).to(args.device)
    if type(ind)!=onp.ndarray:
        k_ts = pytorch_kernel_fn(x_target, x_support, A_target, A_support, args, is_kts=True, a_target_sum=a_target_sum)
    else: 
        k_ts = pytorch_kernel_fn(x_target, x_support, A_target, A_support, args, is_kts=True,ind=torch.tensor(ind), a_target_sum=a_target_sum)

    k_ss_reg = (k_ss + torch.abs(reg) * torch.trace(k_ss) * (torch.eye(k_ss.shape[0]) / k_ss.shape[0]).to(args.device))

    if args.is_y_pred_softmax:
        #pred = torch.softmax(torch.matmul(k_ts, torch.linalg.solve(k_ss_reg.to(torch.float), y_support.to(torch.float)).to(torch.float).to(args.device)),-1) mistake to all softmax
        pred = torch.matmul(k_ts, torch.linalg.solve(k_ss_reg.to(torch.float), torch.softmax(y_support/args.temp,-1).to(torch.float)).to(torch.float).to(args.device)) 
    elif args.is_relu_y:
        pred = torch.matmul(k_ts, torch.linalg.solve(k_ss_reg.to(torch.float), torch.nn.functional.relu(y_support).to(torch.float)).to(torch.float).to(args.device)) 
    else:
        pred = torch.matmul(k_ts, torch.linalg.solve(k_ss_reg.to(torch.float), y_support.to(torch.float)).to(torch.float).to(args.device)) # sp.linalg.solve(k_ss_reg, y_support, sym_pos=True)) # jnp
    if args.is_class_coefficient:
        origin_loss = torch.nn.functional.mse_loss(pred, y_target,reduction='none')
        mse_loss = args.loss_coeff*((((labels_counts.repeat(y_target.shape[0],1)*y_target).max(-1))[0].reshape(-1,1))[counted_idx]*origin_loss[counted_idx]).sum() #*0.5
    elif args.is_kl_div:
        mse_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)(pred[counted_idx], y_target[counted_idx])
    elif args.is_mean_loss:
        if args.is_use_clscore_sort:
            mse_loss = args.loss_coeff*torch.sum(((pred - y_target) ** 2)*difficult_scores_batch.to(args.device))/args.difficult_scores_total # 0.5 jnp
            if args.dataset not in ['ogbn-arxiv']:
                for cls_iter in range(len(onp.unique(LABELS_TRAIN))):
                    count_idx_cls_full = onp.intersect1d(data.idx_train,onp.arange(len(data.labels_full))[onp.where(data.labels_full==cls_iter)]) #data.idx_train[onp.where(data.labels_train==cls_iter)]
                    count_idx_cls = torch.tensor(onp.where(onp.in1d(onp.where(data.labels_full==cls_iter),count_idx_cls_full))[0]).to(args.device)
                    count_idx_cls_full = torch.tensor(count_idx_cls_full).to(args.device)
                    if args.is_kss_class_fix:
                        kss_reg_class = torch.index_select(torch.index_select(k_ss_reg, 0, torch.where(y_support.argmax(-1)==cls_iter)[0]), 1, torch.where(y_support.argmax(-1)==cls_iter)[0])
                        x_support_class = x_support[y_support.argmax(-1)==cls_iter]
                        A_support_class = torch.index_select(torch.index_select(A_support, 0, torch.where(y_support.argmax(-1)==cls_iter)[0]), 1, torch.where(y_support.argmax(-1)==cls_iter)[0])
                    else:
                        x_support_class, A_support_class = x_support, A_support
                    if type(ind)!=onp.ndarray:
                        k_ts = pytorch_kernel_fn(torch.tensor(data.feat_full[onp.where(data.labels_full==cls_iter)]).to(args.device), x_support_class, torch.tensor(data.adj_full.astype(onp.int8)[onp.ix_(data.labels_full==cls_iter,data.labels_full==cls_iter)].toarray()).to(args.device), A_support_class, args, is_kts=True, a_target_sum=a_target_sum)
                    else: 
                        k_ts = pytorch_kernel_fn(torch.tensor(data.feat_full[onp.where(data.labels_full==cls_iter)]).to(args.device), x_support_class, torch.tensor(data.adj_full.astype(onp.int8)[onp.ix_(data.labels_full==cls_iter,data.labels_full==cls_iter)].toarray()).to(args.device), A_support_class, args, is_kts=True,ind=torch.tensor(ind), a_target_sum=a_target_sum)
                    if args.is_kss_class_fix:
                        pred = torch.matmul(k_ts, torch.linalg.solve(kss_reg_class.to(torch.float), y_support[y_support.argmax(-1)==cls_iter].to(torch.float)).to(torch.float).to(args.device))
                    else:
                        pred = torch.matmul(k_ts, torch.linalg.solve(k_ss_reg.to(torch.float), y_support.to(torch.float)).to(torch.float).to(args.device)) # sp.linalg.solve(k_ss_reg, y_support, sym_pos=True)) # jnp
                    mse_loss += args.loss_coeff*torch.sum((pred[count_idx_cls] - (torch.tensor(Y_ALL).to(args.device))[count_idx_cls_full]) ** 2)/(pred[count_idx_cls].shape[0]) 
        else:
            mse_loss = args.loss_coeff*torch.sum((pred[counted_idx] - y_target[counted_idx]) ** 2)/(pred.shape[0]) # 0.5 jnp  
            if args.dataset not in ['ogbn-arxiv']:
                for cls_iter in range(len(onp.unique(LABELS_TRAIN))):
                    count_idx_cls_full = onp.intersect1d(data.idx_train,onp.arange(len(data.labels_full))[onp.where(data.labels_full==cls_iter)]) #data.idx_train[onp.where(data.labels_train==cls_iter)]
                    count_idx_cls = torch.tensor(onp.where(onp.in1d(onp.where(data.labels_full==cls_iter),count_idx_cls_full))[0]).to(args.device)
                    count_idx_cls_full = torch.tensor(count_idx_cls_full).to(args.device)
                    if type(ind)!=onp.ndarray:
                        k_ts = pytorch_kernel_fn(torch.tensor(data.feat_full[onp.where(data.labels_full==cls_iter)]).to(args.device), x_support, torch.tensor(data.adj_full.astype(onp.int8)[onp.ix_(data.labels_full==cls_iter,data.labels_full==cls_iter)].toarray()).to(args.device), A_support, args, is_kts=True, a_target_sum=a_target_sum)
                    else: 
                        k_ts = pytorch_kernel_fn(torch.tensor(data.feat_full[onp.where(data.labels_full==cls_iter)]).to(args.device), x_support, torch.tensor(data.adj_full.astype(onp.int8)[onp.ix_(data.labels_full==cls_iter,data.labels_full==cls_iter)].toarray()).to(args.device), A_support, args, is_kts=True,ind=torch.tensor(ind), a_target_sum=a_target_sum)
                    if args.is_kss_class_fix:
                        pass
                    else:
                        pred = torch.matmul(k_ts, torch.linalg.solve(k_ss_reg.to(torch.float), y_support.to(torch.float)).to(torch.float).to(args.device)) # sp.linalg.solve(k_ss_reg, y_support, sym_pos=True)) # jnp
                    mse_loss += args.loss_coeff*torch.sum((pred[count_idx_cls] - (torch.tensor(Y_ALL).to(args.device))[count_idx_cls_full]) ** 2)/(pred[count_idx_cls].shape[0]) 
    else:
        if args.is_use_clscore_sort:
            mse_loss = args.loss_coeff*torch.sum(((pred - y_target) ** 2)*difficult_scores_batch.to(args.device)) # 0.5 jnp
        else:
            mse_loss = args.loss_coeff*torch.sum((pred[counted_idx] - y_target[counted_idx]) ** 2) # 0.5 jnp
    if args.is_regular_kss:
        mse_loss += 0

    with torch.no_grad():
        acc = 0.0 #torch.mean((torch.argmax(pred[counted_idx], axis=1) == torch.argmax(y_target[counted_idx], axis=1)).to(torch.float)) # jnp jnp jnp
    return mse_loss, acc


def train(args, log_freq=None, IS_A_FIX=False, A_TRAIN=None, A_VALID=None,A_TEST=None, X_TRAIN=None, X_VALID=None, X_TEST=None, Y_TRAIN=None, Y_VALID=None, Y_TEST=None, A_SUM_TRAIN=None, LABELS_TRAIN=None, save_score_name=None, data=None, Y_ALL=None, difficult_scores_final=None):
    if args.is_y_pred_softmax:
        from gcond_agent_transduct_custom import GCond
    else:
        from gcond_agent_transduct_custom import GCond
    seed_everything(args.seed)
    num_train_epochs = args.epochs
    REG = torch.tensor(args.reg).to(args.device)
    best_init_val_acc = 0.
    best_init_seed = None
    best_x_init,best_a_init,best_y_init = None,None,None
    best_kip_loss = 1e10

    if args.is_kcenter_sfgc:
        labels_init, x_init_raw, y_init, a_init = data.labels_full[kcenter_ind_init], data.feat_full[kcenter_ind_init], (torch.nn.functional.one_hot(torch.tensor(data.labels_full).to(torch.int64)).numpy())[kcenter_ind_init], data.adj_full[onp.ix_(kcenter_ind_init, kcenter_ind_init)].toarray()
    elif args.is_rearranged_y:
        labels_init, x_init_raw, y_init, a_init = generate_all_syn(X_TRAIN, A_TRAIN, A_SUM_TRAIN, LABELS_TRAIN, Y_TRAIN, args, seed=args.coreset_fix_seed if args.coreset_fix_seed != -1 else args.seed, data=data) 
        print('labels_init.shape, x_init_raw.shape, y_init.shape, a_init.shape: ',labels_init.shape, x_init_raw.shape, y_init.shape, a_init.shape)
    else:
        ind_raw, labels_init, x_init_raw, y_init, a_init = class_balanced_sample(args.support_size, args, LABELS_TRAIN, X_TRAIN, Y_TRAIN, A_TRAIN, seed=args.seed)
        a_init =  a_init[:,ind_raw]
    seed_everything(args.seed)
    x_init = x_init_raw.astype(onp.float32)
    a_init = a_init.astype(onp.float32)

    if args.is_not_clip_y:
        y_init = y_init.astype(onp.float32)
    else:
        y_init = y_init.astype(onp.float32)+1e-6

    _, labels_counts = onp.unique(labels_init, return_counts=True)
    labels_counts = torch.tensor(labels_counts[0]).reshape(-1,1).to(torch.float).to(args.device)
    labels_counts /= labels_counts.max()

    # end using coefficient
    if args.is_y_pred_softmax:
        y_init[y_init==0.] = -1
        y_init = y_init*args.y_init_multiple_number 
    else:
        pass

    if args.a_strategy == 'Keep_init':
        if args.is_doscond_sigmoid:
            a_init[a_init==0.] = -1
        else:
            a_init[a_init==0.] = 1e-6
        a_init = a_init*args.a_init_multiple_number 
    elif args.a_strategy == 'Identity_a_fix':
        a_init = onp.zeros_like(a_init)+1e-6
        if args.is_doscond_sigmoid:
            a_init[a_init==0.] = -1
        a_init = a_init*args.a_init_multiple_number
    elif args.a_strategy == 'Continuous':
        a_init = torch.tensor(a_init)
        torch.nn.init.xavier_uniform_(a_init)
        for i in range(len(a_init)): 
            a_init[i,i] = 1.
        a_init = a_init*args.a_init_multiple_number
    best_x_init,best_a_init,best_y_init = x_init,a_init,y_init      

    with open(save_score_name, 'r') as original: tx = original.read()
    with open(save_score_name, 'w') as modified: modified.write(tx+strftime("%Y_%m_%d_%H_%M_%S", gmtime())+' STEP'+str(1-1)+' kip  '+str(best_init_val_acc)+" \n")
    with open(save_score_name, 'r') as original: tx = original.read()
    with open(save_score_name, 'w') as modified: modified.write(tx+strftime("%Y_%m_%d_%H_%M_%S", gmtime())+' STEP'+str(1-1)+'  SEED: '+str(best_init_seed)+" \n")
    print('===========')
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(args.seed)

    if  args.is_lambda_trainable:
        params_init = [torch.nn.Parameter(torch.tensor(best_x_init).to(args.device), requires_grad=not args.is_x_fix), torch.nn.Parameter(torch.tensor(best_y_init).to(args.device), requires_grad=not args.is_y_fix),
                          torch.nn.Parameter(torch.tensor(best_a_init).to(args.device), requires_grad=not args.is_a_fix),torch.nn.Parameter(torch.tensor(REG).to(args.device), requires_grad=True)]
        params_init_raw = [torch.tensor(x_init_raw), torch.tensor(labels_init), torch.tensor(a_init)]
        if args.optimizer == 'adam':
            optimizer = torch.optim.AdamW([
              {'params': params_init[0], 'lr': args.lr_train_feat, 'weight_decay': args.weight_decay},
              {'params': params_init[1], 'lr': args.lr_train_y, 'weight_decay': args.weight_decay},
              {'params': params_init[2], 'lr': args.lr_train_adj, 'weight_decay': args.weight_decay},
              {'params': params_init[3], 'lr': args.lr_train_feat, 'weight_decay': args.weight_decay},
                ], lr = args.lr, weight_decay = args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD([
              {'params': params_init[0], 'lr': args.lr_train_feat, 'weight_decay': args.weight_decay},
              {'params': params_init[1], 'lr': args.lr_train_y, 'weight_decay': args.weight_decay},
              {'params': params_init[2], 'lr': args.lr_train_adj, 'weight_decay': args.weight_decay},
              {'params': params_init[3], 'lr': args.lr_train_feat, 'weight_decay': args.weight_decay},
                ], lr = args.lr, weight_decay = args.weight_decay)
        else:
            raise NotImplementedError
    else:
        params_init = [torch.nn.Parameter(torch.tensor(best_x_init).to(args.device), requires_grad=not args.is_x_fix), torch.nn.Parameter(torch.tensor(best_y_init).to(args.device), requires_grad=not args.is_y_fix),
                          torch.nn.Parameter(torch.tensor(best_a_init).to(args.device), requires_grad=not args.is_a_fix)]
        params_init_raw = [torch.tensor(x_init_raw), torch.tensor(labels_init), torch.tensor(a_init)]

        if args.optimizer == 'adam':
            optimizer = torch.optim.AdamW([
              {'params': params_init[0], 'lr': args.lr_train_feat, 'weight_decay': args.weight_decay},
              {'params': params_init[1], 'lr': args.lr_train_y, 'weight_decay': args.weight_decay},
              {'params': params_init[2], 'lr': args.lr_train_adj, 'weight_decay': args.weight_decay},
                ], lr = args.lr, weight_decay = args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD([
              {'params': params_init[0], 'lr': args.lr_train_feat, 'weight_decay': args.weight_decay},
              {'params': params_init[1], 'lr': args.lr_train_y, 'weight_decay': args.weight_decay},
              {'params': params_init[2], 'lr': args.lr_train_adj, 'weight_decay': args.weight_decay},
                ], lr = args.lr, weight_decay = args.weight_decay)
        else:
            raise NotImplementedError

    if int(args.is_warm_up_cosine) + int(args.is_warm_up_linear) + int(args.is_warm_up_constant)>1:
        raise ValueError('Too many schedulers.')
    elif args.is_warm_up_cosine:
        num_training_steps = num_train_epochs*int(len(Y_TRAIN)/args.target_batch_size)
        num_warmup_steps = int(0.05*num_training_steps)
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles= 0.5)
    elif args.is_warm_up_linear:
        num_training_steps = num_train_epochs*int(len(Y_TRAIN)/args.target_batch_size)
        num_warmup_steps = int(0.05*num_training_steps)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.is_warm_up_constant:
        num_training_steps = num_train_epochs*int(len(Y_TRAIN)/args.target_batch_size)
        num_warmup_steps = int(0.05*num_training_steps)
        scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    else:
        scheduler = None

    i_accu=1
    best_acc = 0.
    best_test_acc = 0.
    best_epoch = 0

    time_a = datetime.datetime.now()
    accumulate_time = time_a-time_a
    best_accumulate_time = time_a-time_a
    for jj in range(1,num_train_epochs+1):
        time_a = datetime.datetime.now()
        if args.is_em_strategy and jj==1:
            params_init[0].requires_grad = True
            params_init[1].requires_grad = False
        elif args.is_em_strategy and (jj+1)%(args.em_strategy_epochs) == 0:
            params_init[0].requires_grad = not params_init[0].requires_grad
            params_init[1]. requires_grad= not params_init[1].requires_grad            
        args.temp = args.temp_init*(args.temp_end/args.temp_init)**((jj-1.)/num_train_epochs)
        if args.dataset in ['flickr','reddit']:
            all_temp_ind = onp.arange(len(Y_TRAIN))
            onp.random.seed(args.seed+jj+args.add_seed)
            onp.random.shuffle(all_temp_ind)
        else:
            all_temp_ind = onp.arange(data.labels_full.shape[0])
            onp.random.seed(args.seed+jj+args.add_seed)
            onp.random.shuffle(all_temp_ind)
        for ii in range(int(len(Y_TRAIN if args.dataset in ['flickr','reddit'] else Y_ALL)/args.target_batch_size)):
            ind = onp.arange(args.target_batch_size) + ii*args.target_batch_size
            if args.dataset in ['flickr','reddit']:
                x_target_batch = X_TRAIN[all_temp_ind[ind]]
                y_target_batch = Y_TRAIN[all_temp_ind[ind]]
                A_TRAIN_batch = A_TRAIN[onp.ix_(all_temp_ind[ind],all_temp_ind[ind])]
                if args.is_use_clscore_sort:
                    difficult_scores_batch = torch.tensor(difficult_scores_final[all_temp_ind[ind]])
            else:
                x_target_batch = (data.feat_full)[all_temp_ind[ind]]
                y_target_batch = (Y_ALL)[all_temp_ind[ind]]
                A_TRAIN_batch = ((data.adj_full)[onp.ix_(all_temp_ind[ind],all_temp_ind[ind])]).toarray()
                if args.is_use_clscore_sort:
                    difficult_scores_batch = torch.tensor(difficult_scores_final[all_temp_ind[ind]])
                counted_idx = onp.in1d(all_temp_ind[ind], data.idx_train)
            
            
            if args.dataset in ['flickr','reddit']:
                train_loss, train_acc = custom_loss(params_init[0], params_init[1], torch.tensor(x_target_batch).to(args.device), torch.tensor(y_target_batch).to(args.device),
                                                get_discrete_graphs(params_init[2],args,inference=False), torch.tensor(A_TRAIN_batch), ind,  reg=params_init[3] if args.is_lambda_trainable else REG, labels_counts=labels_counts, a_target_sum=A_SUM_TRAIN[all_temp_ind], args=args, difficult_scores_batch=difficult_scores_batch if args.is_use_clscore_sort else None, LABELS_TRAIN=LABELS_TRAIN, data=data, Y_ALL=Y_ALL, Y_TRAIN=Y_TRAIN)
            else:
                train_loss, train_acc = custom_loss_transduct(params_init[0], params_init[1], torch.tensor(x_target_batch).to(args.device), torch.tensor(y_target_batch).to(args.device),
                                                get_discrete_graphs(params_init[2],args,inference=False), torch.tensor(A_TRAIN_batch), ind,  reg=params_init[3] if args.is_lambda_trainable else REG, labels_counts=labels_counts, a_target_sum=None, args=args, difficult_scores_batch=difficult_scores_batch if args.is_use_clscore_sort else None, counted_idx=counted_idx, LABELS_TRAIN=LABELS_TRAIN, data=data, Y_ALL=Y_ALL)

            if args.is_regular_adj_syn:
                train_loss += args.beta*torch.sqrt(torch.sum(torch.square(get_discrete_graphs(params_init[2],args,inference=False))))
            if args.is_smooth_adj_syn:
                adj_syn = get_discrete_graphs(params_init[2],args,inference=False)
                edge_index_syn = torch.nonzero(adj_syn).T
                edge_weight_syn = adj_syn[edge_index_syn[0], edge_index_syn[1]]
                # smoothness loss
                # feat_difference=torch.pow(feat_syn[edge_index_syn[0]]-feat_syn[edge_index_syn[1]],2)
                feat_difference = torch.exp(-0.5 * torch.pow(params_init[0][edge_index_syn[0]] - params_init[0][edge_index_syn[1]], 2))
                smoothness_loss = torch.dot(edge_weight_syn,torch.mean(feat_difference,1).flatten())/torch.sum(edge_weight_syn) 
                train_loss += smoothness_loss               

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()
            i_accu+=1

        if 1==1: #i_accu % log_freq == 0: every epoch end
            print(f'----step {i_accu-1}:')
            print(f'----EPOCH {jj}')
            print('my train loss:', train_loss)
            print('my train acc:', train_acc)
            if args.is_gcond_valid:
                agent = GCond(data, args, device=args.device)
                if args.is_y_pred_softmax:
                    valid_loss, valid_acc, test_loss, test_acc = agent.val_with_val(params_init[0].detach().to(torch.float), params_init[2].detach().to(torch.float), params_init[1].detach(),A_ALL if args.hop>1 else None,A_TEST)
                else: 
                    valid_loss, valid_acc, test_loss, test_acc = agent.val_with_val(params_init[0].detach().to(torch.float), params_init[2].detach().to(torch.float), params_init[1].detach(),A_ALL if args.hop>1 else None,A_TEST)
                print('valid GCond loss:', valid_loss)
                print('valid GCond acc:', valid_acc) 
                print('test GCond loss:', test_loss)
                print('test GCond acc:', test_acc)
                if args.is_kip_valid: 
                    with torch.no_grad():
                        valid_kip_loss, valid_acc = custom_loss(params_init[0], params_init[1], torch.tensor(X_VALID).to(args.device), torch.tensor(Y_VALID).to(args.device),
                                                     get_discrete_graphs(params_init[2], args).to(args.device), torch.tensor(A_VALID).to(args.device),ind=onp.arange(len(Y_VALID)),  reg=params_init[3] if args.is_lambda_trainable else REG, labels_counts=labels_counts, a_target_sum=A_SUM_VALID, args=args, LABELS_TRAIN=LABELS_TRAIN, data=data, Y_ALL=Y_ALL, Y_TRAIN=Y_TRAIN)

            else:
                with torch.no_grad():
                    valid_kip_loss, valid_kip_acc = custom_loss(params_init[0], params_init[1], torch.tensor(X_VALID).to(args.device), torch.tensor(Y_VALID).to(args.device),
                                                 get_discrete_graphs(params_init[2], args).to(args.device), torch.tensor(A_VALID).to(args.device),ind=onp.arange(len(Y_VALID)),  reg=params_init[3] if args.is_lambda_trainable else REG, labels_counts=labels_counts, a_target_sum=A_SUM_VALID, args=args, LABELS_TRAIN=LABELS_TRAIN, data=data, Y_ALL=Y_ALL, Y_TRAIN=Y_TRAIN)  # compute in batches for expensive kernels
                print('valid loss:', valid_kip_loss)
                print('valid acc:', valid_kip_acc) 
                print() 

            time_b = datetime.datetime.now()
            accumulate_time+=(time_b-time_a)

            if valid_acc>best_acc:
                print("Is temperarily best.")
                print('a_init process: ')
                print(params_init[1][0].detach().cpu().numpy()) # y
                print(params_init[2][0][:20].detach().cpu().numpy()) # a
                print('a_init process end.')
                best_acc = valid_acc
                best_test_acc = test_acc
                best_epoch = jj
                best_accumulate_time = accumulate_time
                with open(save_score_name, 'r') as original: tx = original.read()
                with open(save_score_name, 'w') as modified: modified.write(tx+strftime("%Y_%m_%d_%H_%M_%S", gmtime())+' STEP'+str(i_accu-1)+' accumulate time:  '+str(accumulate_time)+" \n")
                with open(save_score_name, 'r') as original: tx = original.read()
                with open(save_score_name, 'w') as modified: modified.write(tx+strftime("%Y_%m_%d_%H_%M_%S", gmtime())+' STEP'+str(i_accu-1)+'  TRAIN: '+str(valid_acc)+" \n")

                print('----')
                print('Testing original')
                if args.is_gcond_valid:
                    
                    print('test GCond loss again:', test_loss)
                    print('test GCond again acc:', test_acc)  
                    print('Validation time: ',accumulate_time)
                else:            
                    with torch.no_grad():
                        test_loss, test_acc = custom_loss(params_init[0], params_init[1], torch.tensor(X_TEST).to(args.device), torch.tensor(Y_TEST).to(args.device),
                                                     get_discrete_graphs(params_init[2],args).to(args.device), torch.tensor(A_TEST).to(args.device),ind=onp.arange(len(Y_TEST)), reg=params_init[3] if args.is_lambda_trainable else REG, a_target_sum=A_SUM_TEST, args=args, LABELS_TRAIN=LABELS_TRAIN, data=data, Y_ALL=Y_ALL, Y_TRAIN=Y_TRAIN)  # compute in batches for expensive kernels
                    print('test loss:', test_loss)
                    print('test acc:', test_acc)  
                with open(save_score_name, 'r') as original: tx = original.read()
                with open(save_score_name, 'w') as modified: modified.write(tx+strftime("%Y_%m_%d_%H_%M_%S", gmtime())+' STEP'+str(i_accu-1)+'  TEST: '+str(test_acc)+" \n")
                print('----')
                saved_x, saved_y, saved_a = params_init[0].detach().cpu().numpy(), params_init[1].detach().cpu().numpy(), params_init[2].detach().cpu().numpy()
            print()
                
    params_final =  {'x':saved_x,'y':saved_y,'a':saved_a}
    params_init = {'x':best_x_init,'y':best_y_init,'a':best_a_init}
    params_init_raw = {'x':x_init_raw,'y':labels_init,'a':a_init}
    print(" The currently best EPOCH is: ",best_epoch, " The currently best VALID is: ",best_acc, " The currently best TEST is: ",best_test_acc)
    onp.save('final_x_'+str(args.seed),saved_x)
    onp.save('final_y_'+str(args.seed),saved_y)
    onp.save('final_a_'+str(args.seed),saved_a)
    onp.save('init_x',best_x_init)
    onp.save('init_y',best_y_init)
    onp.save('init_a',best_a_init)
    return params_final, params_init, params_init_raw
