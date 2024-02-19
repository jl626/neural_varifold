import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
import jax 
from jax import jit
from absl import app
from absl import flags
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import util
from neural_tangents._src.utils.kernel import Kernel
import numpy as onp
from jax import random

#from generate_graph import generate_graph
#from utils import *

# double precision (computationally much slower)
#from jax.config import config ; config.update('jax_enable_x64', True)

# device check
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)


def main(seed=0, mode='modelnet40', use_graph=False):
  # Build data pipelines.
  print('Loading data.')
  onp.random.seed(seed)
  sample_size = 5  
  if mode == 'modelnet40':
    data = onp.load('../../varifold/modelnet/modelnet40_core3.npz')
    #graph = onp.load('./modelnet/modelnet40_graph2.npz')
    x_train = data['x_train'][:9843]
    y_train = data['y_train'][:9843]
    y_label = np.argmax(y_train,axis=-1)

    new_train = []
    new_label = []
    new_graph = []
    for i in range(40):
      idx = onp.argwhere(y_label==i)[:,0]
      sample = onp.random.choice(len(idx),sample_size,replace=False)
      new_train.append(x_train[idx[sample]])
      new_label.append(y_train[idx[sample]])
      #new_graph.append( graph1[idx[sample]])
    
    # train
    x_train = onp.concatenate(new_train,0)[:,:1024,:]
    y_train = onp.concatenate(new_label,0)
    #x_train = encoder(embed_params,x_train)
    
    # test
    x_test = data['x_test'][:2480]
    y_test = data['y_test'][:2480] 
    #x_test  = encoder(embed_params,x_test)
  elif mode == 'modelnet10': # modelnet10
    data = onp.load('../data/modelnet10_core3.npz')
    #graph = onp.load('./modelnet/modelnet10_graph2.npz')
    x_train = data['x_train'][:3991]
    y_train = data['y_train'][:3991]    
    y_label = np.argmax(y_train,axis=-1)
    #'''
    new_train = []
    new_label = []
    new_graph = []
    #'''
    for i in range(10):
      idx = onp.argwhere(y_label==i)[:,0]
      sample = onp.random.choice(len(idx),sample_size,replace=False)
      new_train.append(x_train[idx[sample]])
      new_label.append(y_train[idx[sample]])
      #new_graph.append( graph1[idx[sample]])
    # sampled train
    x_train = onp.concatenate(new_train,0)[:,:1024,:]
    y_train = onp.concatenate(new_label,0)
    #'''

    #x_train = x_train
    #y_train = y_train    # test
    
    x_test = data['x_test'][:920,:1024,:]
    y_test = data['y_test'][:920]  
    #x_test  = encoder(embed_params,x_test)
 
  print(x_train.shape)
  print(y_train.shape)
  print(x_test.shape)
  print(y_test.shape)

  n_width = 1024
  # Build the empirical NTK
  layers = []
  for k in range(9):
    layers += [stax.Dense(n_width, 1., 0.05),stax.LayerNorm(), stax.Relu()]
  print(len(layers)//3, '-layer network')
  #init_fn, apply_fn, _ = stax.serial(*(layers + [stax.Dense(1, 1., 0.05),stax.GlobalAvgPool()])) # finite case accelerate!
  init_fn, apply_fn, _ = stax.serial(*(layers + [stax.GlobalAvgPool()])) # finite case accelerate!
  

  kwargs = dict(
      f=apply_fn,
      trace_axes=(),
      vmap_axes=0
  )


  # Empirical NTK 
  kernel_fn = nt.batch(nt.empirical_kernel_fn(apply_fn, vmap_axes=0, implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES), batch_size=1, device_count=0)
  key = random.PRNGKey(1)
  _, params = init_fn(key, (-1, 1024, 6))
  print('Compute train kernel')
  g_dd = kernel_fn(x_train, x_train,'ntk', params)
  print('Compute test  kernel')
  g_td = kernel_fn(x_test,  x_train,'ntk', params)
  
  print('compute predict function')
  loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
  
  print(g_dd.shape)
  print(g_td.shape)
  #predict_fn = nt.predict.gradient_descent(loss, g_dd, y_train)
  predict_fn = nt.predict.gradient_descent_mse(g_dd, y_train)
  predict_fn = jit(predict_fn)
  fx_train_t, fx_test_ntk = predict_fn(t=None,k_test_train=g_td, fx_test_0=0.0) # get closed form solution
  if mode == 'modelnet10':
    util.print_summary('NTK  test', y_test[:908], fx_test_ntk[:908] , None, loss)
    return np.mean(np.argmax(fx_test_ntk[:908], axis=1) == np.argmax(y_test[:908], axis=1)), np.mean(np.argmax(fx_test_ntk[:908], axis=1) == np.argmax(y_test[:908], axis=1))
  elif mode =='modelnet40':
    util.print_summary('NTK  test', y_test[:2468],fx_test_ntk[:2468] ,None, loss)
  
if __name__ == '__main__':
  ng,nt = main(0,'modelnet10',False)
  