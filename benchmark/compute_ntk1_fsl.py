import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
#from infinite_fcn4 import main as compute_engine
from infinite_fsl1 import main as compute_engine

import numpy as np 
import time 
ntk = []
nngp= []

for i in range(20):
    print(i+1, ' ModelNet10 ablation ')
    start = time.time()
    ng,nt = compute_engine(i,'modelnet40',False)
    duration = time.time() - start
    print('Kernel construction and inference done in %s seconds.' % duration)
    ntk.append(np.array(nt).item())
    nngp.append(np.array(ng).item())
print(np.asarray(ntk))
print(np.asarray(nngp))
print(np.mean(ntk))
print(np.mean(nngp))
print(np.std(ntk))
print(np.std(nngp))