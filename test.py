# Python Library dependencies (include jax, neural_tangents)
# note that there are two numpy lib (onp - standard numpy library // np - Jax competible autodiff implmentation of numpy)
import sys
sys.path.append( '..' )
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
from absl import app
from absl import flags

import jax 
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util
from neural_tangents.utils.kernel import Kernel
import numpy as onp
from jax import random
from jax.config import config ; config.update('jax_enable_x64', True)
import util
from util import Prod

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# GPU device check
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# define data parameter setting 
n_pts = 1024 # number of points 
onp.random.seed(0) # random seed for numpy
sample_size = 1

# Load the dataset
data = onp.load('./data/modelnet10_1024_core3.npz')

# first - train dataset
x_train = data['x_train'][:3991]
y_train = data['y_train'][:3991]    
y_label = np.argmax(y_train,axis=-1)

# subsample training data  
new_train = []
new_label = []
new_graph = []
for i in range(10):
    idx = onp.argwhere(y_label==i)[:,0]
    sample = onp.random.choice(len(idx),sample_size,replace=False)
    new_train.append(x_train[idx[sample]])
    new_label.append(y_train[idx[sample]])
    #new_graph.append( graph1[idx[sample]])

# train data
x_train = onp.concatenate(new_train,0)
y_train = onp.concatenate(new_label,0)

# test (note that it is actually 908 samples + duplication to fit hardward requirement)
x_test = data['x_test']
y_test = data['y_test'] 

print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape(sample_size*10,n_pts,2,3).transpose((0,2,1,3))
x_test  = x_test.reshape(920,n_pts,2,3).transpose((0,2,1,3))

print(x_train.shape)
print(x_test.shape)

# plot a sample from each class in the training data 
for i in range(10):
    sample = x_train[i*sample_size+0,0].squeeze()
    ax = plt.axes(projection='3d')
    ax.scatter(sample[:,0],sample[:,1],sample[:,2])
    plt.show()

# Build the 5-layer infinite network.
layers = []
for k in range(5):
    layers += [stax.Dense(1, 1., 0.05),stax.LayerNorm(), stax.Relu()] # PointNet like architecture with fully connected layer
    #layers += [stax.Conv(1, (1,1), (1,1), 'SAME', 1., 0.05),stax.LayerNorm(), stax.Relu()] # PointNet like architecture with pointwise convolution
print(len(layers)//3, '-layer network')
init_fn, apply_fn, kernel_fn = stax.serial(*(layers + [Prod(),stax.GlobalAvgPool()])) # element-wise product to compute varifold kernel, then compute average to get a norm 

# compute the kernel in batches, in parallel.
kernel_fn = nt.batch(kernel_fn,device_count=-1,batch_size=5)

# conduct kernel regression with diagonal regularisation (1e-2)
predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train,y_train, diag_reg=1e-2) # train a function
fx_test_nngp, fx_test_ntk = predict_fn(x_test=x_test) # regression on the test dataset
fx_test_nngp.block_until_ready()
fx_test_ntk.block_until_ready()

# Print out accuracy and loss for infinite network predictions.
loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
util.print_summary('NNGP test', y_test[:908], fx_test_nngp[:908], None, loss)
util.print_summary('NTK  test', y_test[:908], fx_test_ntk[:908] , None, loss)