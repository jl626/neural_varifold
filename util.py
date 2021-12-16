# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A set of utility operations for running examples.
"""


import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from neural_tangents.utils.kernel import Kernel
import numpy as onp
from jax import random
from jax.config import config ; config.update('jax_enable_x64', True)
#FLAGS = flags.FLAGS


def _accuracy(y, y_hat):
  """Compute the accuracy of the predictions with respect to one-hot labels."""
  return np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))


def print_summary(name, labels, net_p, lin_p, loss):
  """Print summary information comparing a network with its linearization."""
  print('\nEvaluating Network on {} data.'.format(name))
  print('---------------------------------------')
  print('Network Accuracy = {}'.format(_accuracy(net_p, labels)))
  print('Network Loss = {}'.format(loss(net_p, labels)))
  if lin_p is not None:
    print('Linearization Accuracy = {}'.format(_accuracy(lin_p, labels)))
    print('Linearization Loss = {}'.format(loss(lin_p, labels)))
    print('RMSE of predictions: {}'.format(
        np.sqrt(np.mean((net_p - lin_p) ** 2))))
  print('---------------------------------------')

def accuracy(y, y_hat):
  return _accuracy(net_p, labels)

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