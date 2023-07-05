import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

import jax 
import neural_tangents
from neural_tangents import stax
import jax.numpy as jnp

class tangent_kernel(nn.Module):
    def __init__(self, num_layers=1,weight=1.0, bias=0.05, dim_size = 3, mode='NTK1'):
        super(tangent_kernel, self).__init__()
        self.num_layers = num_layers
        self.weight = weight
        self.bias = bias
        self.dim_size = dim_size
        self.mode = mode
        
    def nngp_ntk_fn(self,K,prod, ntk):
        square_root = torch.sqrt(F.threshold(prod - (K ** 2),0.0, 0.0))
        angles = torch.atan2(square_root, K)
        
        factor = 1/ (2 * np.pi)
        dot_sigma = 1/2 - factor * angles
        new_k = factor * square_root + dot_sigma * K
        ntk_new = ntk*dot_sigma
        return new_k, ntk_new
    
    def nngp_fn_diag(self,nngp):
        return 0.5 * nngp

    def fc(self, mat):
        return (self.weight**2) * mat + self.bias**2
    
    def gaussian(self,x1,x2):
        n = x1.size(0) 
        m = x2.size(0)
        d = x1.size(1)
        x = x1.unsqueeze(1).expand(n, m, d) 
        y = x2.unsqueeze(0).expand(n, m, d) 
        scale = 1/0.1**2
        gauss = torch.exp(-(torch.norm(x-y,2,2)**2*scale))
        return gauss

    def ntk_fn(self,x1,x2):
        # initial covariance & NTK kernel
        K = torch.matmul(x1,x2.T)/self.dim_size
        ntk = torch.matmul(x1,x2.T)/self.dim_size
        # diagonal initial
        diag_x = torch.sum(x1**2, 1)/self.dim_size
        diag_y = torch.sum(x2**2, 1)/self.dim_size
        # N-hidden layer infinite-width neural network 
        for i in range(3):#self.num_layers):
            # Dense operation
            diag_x = self.fc(diag_x)
            diag_y = self.fc(diag_y)
            K = self.fc(K)
            if i == 0:
                ntk = self.fc(ntk)
            else: 
                ntk = ntk * self.weight**2 + K
            # activation
            prod = (diag_x[:,None] * diag_y[None,:])#(norms1[:,None] * norms2[None,:])
            K, ntk = self.nngp_ntk_fn(K, prod, ntk)
            diag_x = self.nngp_fn_diag(diag_x)
            diag_y = self.nngp_fn_diag(diag_y)
        return ntk + K + self.bias**2
    
    def binet(self,x1,x2):
        return torch.sqrt(torch.matmul(x1,x2.T)**2)

    def forward(self,x1,x2):
        #'''
        if self.mode == 'NTK1':
            g1 = self.ntk_fn(x1[:,:3], x2[:,:3])
            g2 = self.ntk_fn(x1[:,3:],x2[:,3:])
            g = g1*g2
        elif self.mode == 'NTK2':
            g = self.ntk_fn(x1, x2)
        elif self.mode == 'binet':
            g1 = self.gaussian(x1[:,:3], x2[:,:3])
            g2 = self.binet(x1[:,3:],x2[:,3:])
            g = g1*g2
        return g
    
# build a batch version
class tangent_kernel_batch(nn.Module):
    def __init__(self, num_layers=1,weight=1.0, bias=0.05, dim_size = 3):
        super(tangent_kernel_batch, self).__init__()
        self.num_layers = num_layers
        self.weight = weight
        self.bias = bias
        self.dim_size = dim_size
        
    def nngp_ntk_fn(self,K,prod, ntk):
        square_root = torch.sqrt(F.threshold(prod - (K ** 2),1e-16, 1e-10))
        angles = torch.atan2(square_root, K)
        
        factor = 1/ (2 * np.pi)
        dot_sigma = 1/2 - factor * angles
        new_k = factor * square_root + dot_sigma * K
        ntk_new = ntk*dot_sigma
        return new_k, ntk_new
    
    def nngp_fn_diag(self,nngp):
        return 0.5 * nngp

    def fc(self, mat):
        return (self.weight**2) * mat + self.bias**2

    def ntk_fn(self,x1,x2):
        """
        batch version 
        x1: B X N X 3 
        x2: B X M X 3
        # initial covariance & NTK kernel
        
        """
        K = torch.matmul(x1,x2.transpose(2,1))/self.dim_size
        ntk = torch.matmul(x1,x2.transpose(2,1))/self.dim_size
        # diagonal initial
        diag_x = torch.sum(x1**2, 2)/self.dim_size
        diag_y = torch.sum(x2**2, 2)/self.dim_size
        # N-hidden layer infinite-width neural network 
        for i in range(3):
            # Dense operation
            diag_x = self.fc(diag_x)
            diag_y = self.fc(diag_y)
            K = self.fc(K)
            if i == 0:
                ntk = self.fc(ntk)
            else: 
                ntk = ntk * self.weight**2 + K
            # activation
            prod = (diag_x[:,:,None] * diag_y[:,None,:])#(norms1[:,None] * norms2[None,:])
            K, ntk = self.nngp_ntk_fn(K, prod, ntk)
            diag_x = self.nngp_fn_diag(diag_x)
            diag_y = self.nngp_fn_diag(diag_y)
        return ntk + K + 0.05**2

    def forward(self,x1,x2):
        #'''
        g2 = self.ntk_fn(x1,x2)
        return g2


def Comp_normal(F, V):

    V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
    N =  .5 * torch.cross(V1 - V0, V2 - V0)

    return N


if __name__=="__main__":
    # sanity check
    x = torch.rand(10,100,3)
    y = torch.rand(10,100,3)
    kernel_torch = tangent_kernel()
    kernel_torch_batch = tangent_kernel_batch()
    
    xx = x[0].numpy()
    yy = y[0].numpy()
    init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(1,1.,0.05), stax.Relu(),
    stax.Dense(1,1.,0.05), stax.Relu(),
    stax.Dense(1,1.,0.05), stax.Relu(),
    stax.Dense(1,1.,0.05))
    kernel_nt = kernel_fn(xx, yy, 'ntk')
    #print(np.asarray(kernel_nt) - kernel_torch(x[0],y[0]).numpy())
    print(np.sum(np.abs(np.asarray(kernel_nt) -kernel_torch_batch(x,y).numpy()[0])))