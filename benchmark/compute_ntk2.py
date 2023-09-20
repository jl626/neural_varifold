import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from infinite_analytic2 import main as compute_engine

import numpy as np 
ntk = []
nngp= []

for i in range(20):
    print(i+1)
    ng,nt = compute_engine(i,'modelnet10',False)
    ntk.append(np.array(nt).item())
    nngp.append(np.array(ng).item())
print(np.asarray(ntk))
print(np.asarray(nngp))
print(np.mean(ntk))
print(np.mean(nngp))
print(np.std(ntk))
print(np.std(nngp))