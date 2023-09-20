import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
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
#from generate_graph import generate_graph
#from utils import *
# double precision (computationally much slower)
from jax.config import config ; config.update('jax_enable_x64', True)
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


# positional encoding // fourier encoding (NIPS 2020)
def encoder(embed_params, inputs):
  input_encoder = lambda x, a, b: (np.concatenate([a * np.sin((2.*np.pi*x) @ b.T), 
                                                     a * np.cos((2.*np.pi*x) @ b.T)], axis=-1) / np.linalg.norm(a)) if a is not None else x#(x * 2. - 1.)

  embedding_method, embedding_size, embedding_scale = embed_params

  if embedding_method == 'gauss':
    print('gauss bvals')
    bvals = onp.random.normal(size=[embedding_size,6]) * embedding_scale

  if embedding_method == 'posenc':
    print('posenc bvals')
    v = 6
    bvals = 2.**np.linspace(0,embedding_scale,embedding_size//v) - 1
    bvals = np.reshape(np.eye(v)*bvals[:,None,None], [len(bvals)*v, v])

  if embedding_method == 'basic':
    print('basic bvals')
    bvals = onp.eye(3)


  if embedding_method == 'none':
    print('NO abvals')
    avals = None
    bvals = None
  else:
    avals = np.ones_like(bvals[:,0])
  ab = (avals, bvals)
  x = input_encoder(inputs, *ab)
  return x 

embed_tasks = [
               ['gauss', 256, 0.001],
               ['posenc',256, 1.0],
               ['basic', None, None],
               ['none', None, None],
]


def main(seed=0, mode='modelnet40', use_graph=False):
  # Build data pipelines.
  print('Loading data.')
  onp.random.seed(seed)
  sample_size = 5  
  if mode == 'modelnet40':
    data = onp.load('./modelnet/modelnet40_core3.npz')
    #graph = onp.load('./modelnet/modelnet40_graph2.npz')
    x_train = data['x_train'][:9843]
    y_train = data['y_train'][:9843]
    y_label = np.argmax(y_train,axis=-1)

    embed_params = embed_tasks[1]
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
    x_train = onp.concatenate(new_train,0)[:,:1024,:].reshape(sample_size*40,1024,2,3).transpose((0,2,1,3)) # varifold setting
    y_train = onp.concatenate(new_label,0)
    #x_train = encoder(embed_params,x_train)
    
    # test
    x_test = data['x_test'][:2480].reshape(2480,1024,2,3).transpose((0,2,1,3))
    y_test = data['y_test'][:2480] 
    #x_test  = encoder(embed_params,x_test)
  elif mode == 'modelnet10': # modelnet10
    data = onp.load('./modelnet/modelnet10_core3.npz')
    n_pts = 1024
    #graph = onp.load('./modelnet/modelnet10_graph2.npz')
    x_train = data['x_train'][:3991]
    y_train = data['y_train'][:3991]    
    y_label = np.argmax(y_train,axis=-1)
    #'''
    embed_params = embed_tasks[1]
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
    x_train = onp.concatenate(new_train,0)[:,:n_pts,:].reshape(sample_size*10,n_pts,2,3).transpose((0,2,1,3))
    #x_train = onp.concatenate([x_train, onp.ones((100,1024,1),dtype=np.float64)],2)#.reshape(100,1024,2,3).transpose((0,2,1,3))
    y_train = onp.concatenate(new_label,0)
    #'''
    #x_train = encoder(embed_params,x_train)

    #x_train = x_train.reshape(4000,1024,2,3).transpose((0,2,1,3))
    #y_train = y_train    # test
    
    x_test = data['x_test'][:920,:n_pts,:].reshape(920,n_pts,2,3).transpose((0,2,1,3))
    #x_test = onp.concatenate([x_test,onp.ones((920,1024,1),dtype=np.float64)],2)#.reshape(920,1024,2,3).transpose((0,2,1,3))
    y_test = data['y_test'][:920]  
    #x_test  = encoder(embed_params,x_test)
 
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
  for k in range(5):
    #if use_graph:
    #  layers += [stax.Aggregate(aggregate_axis=1, batch_axis=0, channel_axis=2)] 
    layers += [stax.Dense(1, 1., 0.05),stax.LayerNorm(), stax.Relu()]
    #layers += [stax.Dense(1, 1., 0.05), stax.Relu()]
    #layers += [stax.Conv(1, (1, ), (1, ), 'SAME'),stax.LayerNorm(), stax.Relu()]
  print(len(layers)//3, '-layer network')
  init_fn, apply_fn, kernel_fn = stax.serial(*(layers + [Prod(),stax.GlobalAvgPool()]))

  # Optionally, compute the kernel in batches, in parallel.
  kernel_fn = nt.batch(kernel_fn,device_count=-1,batch_size=1)
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
  