import os 
os.environ['CUDA_VISIBLE_DEVICES']='3'
import sys 
sys.path.append( '..' )

import input_output 

import torch
import time 

# EMD metric
from evaluation_recon_emd import emdModule
from energy import tangent_kernel

# Chamfer metric
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

torch.manual_seed(0)
torch.backends.cudnn.deterministic =True

experiments = "hippo"
#experiments = "cup"
#experiments = "dolphin"
#experiments = 'bunny'
#experiments = 'plane'
#experiments = 'ablation'

#names = 'chamfer'
#names = 'binet'
#names = 'emd'
#names = 'pointnet_ntk1'
names = 'pointnet_ntk2'

types = '_red'
#types = ''


if names == 'binet':
    verts1,faces1,_ = load_obj('../results/%s/%s_%s%s.obj'%(experiments,names,experiments,types))
elif names == "chamfer":
    verts1,faces1,_ = load_obj('../results/%s/%s_%s%s.obj'%(experiments,names,experiments,types))
elif names == "emd":
    verts1,faces1,_ = load_obj('../results/%s/%s_%s%s.obj'%(experiments,names,experiments,types))
elif names == "pointnet_ntk1":
    verts1,faces1,_ = load_obj('../results/%s/%s_%s%s.obj'%(experiments,names,experiments,types))
    #verts1,faces1,_ = load_obj('../results/%s/pointnet_ntk1_hippo_red_9layer.obj'%(experiments)) # ablation case
elif names == "pointnet_ntk2":
    verts1,faces1,_ = load_obj('../results/%s/%s_%s%s.obj'%(experiments,names,experiments,types))
    #verts1,faces1,_ = load_obj('../results/%s/pointnet_ntk2_hippo_red_1layer.obj'%(experiments)) # ablation case


if experiments == "hippo":
    verts2,faces2,_ = load_obj('../results/%s/hippo_target.obj'%experiments)
elif experiments == "dolphin":
    verts2,faces2,_ = load_obj("../data/matching/dolphin.obj")
elif experiments == "cup":
    [VT,FT,FunT] = input_output.loadData("../data/matching/cup3_broken.ply")
    param_decimation = {'factor':31/32,'Vol_preser':1, 'Fun_Error_Metric': 1, 'Fun_weigth':0.00} #decimate by a factor of 16
    [verts2,faces2]= input_output.decimate_mesh(VT,FT,param_decimation)
    verts2 = torch.FloatTensor(verts2)
    faces2 = torch.LongTensor(faces2)
elif experiments == 'bunny':
    verts2,faces2,_ = load_obj('../results/%s/bunny_red.obj'%experiments)
elif experiments =='plane':
    verts2,faces2,_ = load_obj('../results/%s/plane_target_red.obj'%experiments)
elif experiments == "ablation":
    verts2,faces2,_ = load_obj('../results/hippo/hippo_target.obj')

res = Meshes(verts=[verts1], faces=[faces1.verts_idx])
if experiments == 'dolphin' or experiments =='hippo' or experiments == 'bunny' or experiments=='plane' or experiments=='ablation':
    gt_mesh = Meshes(verts=[verts2], faces=[faces2.verts_idx])
else:
    gt_mesh = Meshes(verts=[verts2], faces=[faces2])

# EMD loss function
emd = emdModule()
# NTK1
ntk1 = tangent_kernel(5,1.,0.05,3,mode='NTK1') # 5 Layer 
# NTK2
ntk2 = tangent_kernel(9,1.,0.05,3,mode='NTK2') # 9 Layer 
# Binet
binet = tangent_kernel(1,1.,0.05,3,mode='binet') # Gaussian RBF - Postion & Cauchy-Binet - normal (Grassmannian)



# sample 4096 points to compute EMD (EMD implementation only accept ones divide by 1024)
sample_gt,sample_gt_nor = sample_points_from_meshes(gt_mesh, 4096, return_normals=True)
print(sample_gt)
sample_res,sample_res_nor = sample_points_from_meshes(res, 4096, return_normals=True)
print('names: %s'%names)
print('Results for Experiment: %s'%experiments)

emd_val, _ = emd(sample_res*10,sample_gt*10, 1e-3*2, 10000)
loss_emd = emd_val.sum()

if experiments =='dolphin' or experiments=='plane':
    sample_gt  = sample_gt*10
    sample_res = sample_res*10


def CompCLNn(F, V):
    if F.shape[1] == 2:
        V0, V1 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1])
        C, N  =  (V0 + V1)/2, V1 - V0
    else:
        V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
        C, N =  (V0 + V1 + V2)/3, .5 * torch.cross(V1 - V0, V2 - V0)

    L = (N ** 2).sum(dim=1)[:, None].sqrt()
    return C, L, N / L, 1#Fun_F

def compute_engine(V1,V2,L1,L2,K):
    cst_tmp = []
    n_batch = 10000#4096
    for i in range(len(V1)//n_batch + 1):
        tmp = V1[i*n_batch:(i+1)*n_batch,:]
        l_tmp = L1[i*n_batch:(i+1)*n_batch,:]
        v = torch.matmul(K(tmp,V2),L2)*l_tmp
        cst_tmp.append(v)
    cst = torch.sum(torch.cat(cst_tmp,0))
    return cst

def compute_varifold_loss(c1,c2,l1,l2,k):
    v11 = compute_engine(c1,c1,l1,l1,k)
    v22 = compute_engine(c2,c2,l2,l2,k) 
    v12 = compute_engine(c2,c1,l2,l1,k)

    loss_varifold = v11 + v22 -2*v12
    return loss_varifold

f1 = res.faces_packed()
v1 = res.verts_packed()
f2 = gt_mesh.faces_packed()
v2 = gt_mesh.verts_packed()

if experiments =='dolphin':
    v1 = v1*10
    v2 = v2*10


# results
c1,l1,n1,_ = CompCLNn(f1,v1)
# gt
c2,l2,n2,_ = CompCLNn(f2,v2)
print(c1.shape)
print(c2.shape)

loss_cd, _ = chamfer_distance(sample_res,sample_gt)

c1 = torch.cat([c1,n1],1)
c2 = torch.cat([c2,n2],1)
start = time.time()
loss_ntk1  = compute_varifold_loss(c1,c2,l1,l2,ntk1)
duration = time.time() - start
print('Kernel construction and inference done in %s seconds.' % duration)

start = time.time()
loss_ntk2  = compute_varifold_loss(c1,c2,l1,l2,ntk2)
duration = time.time() - start
print('Kernel construction and inference done in %s seconds.' % duration)


loss_binet = compute_varifold_loss(c1,c2,l1,l2,binet)



print('Chamfer distance: %.6f'%loss_cd)
print('Earth Movers distance: %.6f'%loss_emd)
print('Charon Trouve Varifold Loss: %.6f'%loss_binet)
print('NTK1 Varifold Loss: %.8f'%loss_ntk1)
print('NTK2 Varifold Loss: %.8f'%loss_ntk2)



