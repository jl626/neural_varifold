import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
import jax 
from absl import app
from absl import flags
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util
from neural_tangents._src.utils.kernel import Kernel
import numpy as onp
from jax import random
from generate_graph import generate_graph
#from utils import *
# double precision (computationally much slower)
#from jax.config import config ; config.update('jax_enable_x64', True)
# device check
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

#FLAGS = flags.FLAGS

# analoguous to traditional Gaussian - Cauchy-binet varifold kernel [K_gauss * K_caucy_binet]
@stax.layer
def Prod():
  def init_fn(rng, input_shape):
    return input_shape, ()

  def apply_fn(params, inputs, **kwargs):
    raise NotImplementedError()

  def kernel_fn(k: Kernel, **kwargs):
    def prod(mat, batch_ndim):
      if mat is None or mat.ndim == 0:
        return mat

      if k.diagonal_spatial:
        # Assuming `mat.shape == (N1[, N2], 2, n)`.
        return np.take(mat, 0, batch_ndim) * np.take(mat, 1, batch_ndim)

      # Assuming `mat.shape == (N1[, N2], 2, 2, n, n)`.
      concat_dim = batch_ndim if not k.is_reversed else -1
      return (np.take(np.take(mat, 0, concat_dim), 0, concat_dim) *
              np.take(np.take(mat, 1, concat_dim), 1, concat_dim))
    
    # Output matrices are `(N1[, N2], n[, n])`.
    return k.replace(nngp=prod(k.nngp, 2),
                     cov1=prod(k.cov1, 1 if k.diagonal_batch else 2),
                     cov2=prod(k.cov2, 1 if k.diagonal_batch else 2),
                     ntk=prod(k.ntk, 2))

  return init_fn, apply_fn, kernel_fn


def main(seed=0, mode='modelnet40', use_graph=False):
  # Build data pipelines.
  print('Loading data.')
  onp.random.seed(seed)
  sample_size = 1  
  fsl = np.asarray([4,5,15,18,23,24,25,28,29,37])
  k = onp.random.choice(10,5,replace=False)
  fsl = fsl[k].tolist()
  print(fsl)
  if mode == 'modelnet40':
    data = onp.load('../../varifold/modelnet/modelnet40_core3.npz')
    x_train = data['x_train'][:9843]
    y_train = data['y_train'][:9843]
    y_label = np.argmax(y_train,axis=-1)
    
    new_train = []
    new_label = []
    new_graph = []
    for i in range(40):
      if i not in fsl: continue
      idx = onp.argwhere(y_label==i)[:,0]
      sample = onp.random.choice(len(idx),sample_size,replace=False)
      new_train.append(x_train[idx[sample]])
      new_label.append(y_train[idx[sample]])
    
    # train
    x_train = onp.concatenate(new_train,0)[:,:1024,:]
    y_train = onp.concatenate(new_label,0)
    
    # test
    x_test = data['x_test'][:2480]
    y_test = data['y_test'][:2480] 

    #'''
    new_test = []
    new_testlabel = [] 
    y_testlabel = np.argmax(y_test,axis=-1)
    for i in range(40):
      if i not in fsl: continue
      idx = onp.argwhere(y_testlabel==i)[:,0]
      sample = onp.random.choice(len(idx),15,replace=False)
      new_test.append(x_test[idx[sample]])
      new_testlabel.append(y_test[idx[sample]])
    # sampled train
    x_test = onp.concatenate(new_test,0)[:,:1024,:]
    y_test = onp.concatenate(new_testlabel,0)
    #'''
 
  print(x_train.shape)
  print(y_train.shape)
  print(x_test.shape)
  print(y_test.shape)

  # graph 
  if use_graph:
    graph1 = generate_graph(x_train, 10)
    graph2 = generate_graph(x_test,  10)


  # Build the infinite network.
  layers = []
  for k in range(9):
    if use_graph:
      layers += [stax.Aggregate(aggregate_axis=1, batch_axis=0, channel_axis=2)] 
    layers += [stax.Dense(1, 1., 0.05),stax.LayerNorm(), stax.Relu()]
    #layers += [stax.Dense(1, 1., 0.05), stax.Relu()]
    #layers += [stax.Conv(1, (1, ), (1, ), 'SAME'),stax.LayerNorm(), stax.Relu()]
  print(len(layers)//3, '-layer network')
  init_fn, apply_fn, kernel_fn = stax.serial(*(layers + [stax.GlobalAvgPool()]))

  # Optionally, compute the kernel in batches, in parallel.
  kernel_fn = nt.batch(kernel_fn,device_count=-1,batch_size=5)
  start = time.time()
  # Bayesian and infinite-time gradient descent inference with infinite network.
  if use_graph:
    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train,y_train, diag_reg=1e-2,pattern=(graph1,graph1))
    fx_test_nngp, fx_test_ntk = predict_fn(x_test=x_test, pattern=(graph2,graph2))
  else:
    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train,y_train, diag_reg=1e-2)
    fx_test_nngp, fx_test_ntk = predict_fn(x_test=x_test)
  fx_test_nngp.block_until_ready()
  fx_test_ntk.block_until_ready()

  duration = time.time() - start
  print('Kernel construction and inference done in %s seconds.' % duration)
  # Print out accuracy and loss for infinite network predictions.
  loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
  if mode == 'modelnet10':
    util.print_summary('NNGP test', y_test[:908], fx_test_nngp[:908], None, loss)
    util.print_summary('NTK  test', y_test[:908], fx_test_ntk[:908] , None, loss)
    return np.mean(np.argmax(fx_test_nngp[:908], axis=1) == np.argmax(y_test[:908], axis=1)), np.mean(np.argmax(fx_test_ntk[:908], axis=1) == np.argmax(y_test[:908], axis=1))
  elif mode =='modelnet40':
    util.print_summary('NNGP test', y_test[:2468],fx_test_nngp[:2468],None, loss)
    util.print_summary('NTK  test', y_test[:2468],fx_test_ntk[:2468] ,None, loss)
    return np.mean(np.argmax(fx_test_nngp[:2468], axis=1) == np.argmax(y_test[:2468], axis=1)), np.mean(np.argmax(fx_test_ntk[:2468], axis=1) == np.argmax(y_test[:2468], axis=1))
  
  
if __name__ == '__main__':
  ng,nt = main(0,'modelnet10',False)
  