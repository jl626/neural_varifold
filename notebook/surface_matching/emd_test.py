import sys
sys.path.append( '../..' )
import os 
os.environ['CUDA_VISIBLE_DEVICES']='2'
import torch
import torch.nn as nn 
from torch.autograd import Function

from pytorch3d.io import load_obj, save_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import chamfer_distance

import numpy as np

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

import input_output 
from benchmark.evaluation_recon_emd import emdModule


torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)


#experiment_name = 'hippo'
#experiment_name = 'cup'
#experiment_name = 'dolphin'
#experiment_name ='bunny'
experiment_name = 'plane'

# We read the target 3D model using load_obj

#low = 31 mid = 26 high = 20

if experiment_name=='cup':
    # two different Cups 
    [VS,FS,FunS] = input_output.loadData("../../data/matching/cup2.ply")
    [VT,FT,FunT] = input_output.loadData("../../data/matching/cup3_broken.ply")
    source = [VS,FS]
    target= [VT,FT]
    # original size
    print(VS.shape)
    print(VT.shape)
    # Option 1 Sampling
    # Decimate source mesh to compute initialization for the multires algorithm 
    param_decimation = {'factor':31/32,'Vol_preser':1, 'Fun_Error_Metric': 1, 'Fun_weigth':0.00} #decimate by a factor of 16
    [verts1,faces1]= input_output.decimate_mesh(VS,FS,param_decimation)
    param_decimation = {'factor':31/32,'Vol_preser':1, 'Fun_Error_Metric': 1, 'Fun_weigth':0.00} #decimate by a factor of 16
    [verts2,faces2]= input_output.decimate_mesh(VT,FT,param_decimation)
    print(verts1.shape)
    print(verts2.shape)
    verts1 = torch.FloatTensor(verts1)
    verts2 = torch.FloatTensor(verts2)
    faces1 = torch.LongTensor(faces1)
    faces2 = torch.LongTensor(faces2)
# Case 2 Dolphin
elif experiment_name == 'dolphin':
    # sphere to dolphine 
    src_mesh = ico_sphere(4)
    VT, FT, FunS = load_obj("../../data/matching/dolphin.obj")
    VS, FS = src_mesh.verts_packed(), src_mesh.faces_packed()
    verts1, faces1 = VS, FS
    verts2, faces2 = VT, FT.verts_idx
# Case 3 Hippocampus 
elif experiment_name == 'hippo':
    verts1, faces1, verts2, faces2 = torch.load('../../data/matching/hippos_red.pt')

elif experiment_name == 'bunny':
    # sphere to dolphine 
    src_mesh = ico_sphere(4)
    VT, FT, FunS = load_obj("../../data/matching/bunny.obj")
    VS, FS = src_mesh.verts_packed(), src_mesh.faces_packed()
    verts1, faces1 = VS, FS
    # Decimate source mesh to compute initialization for the multires algorithm 
    #param_decimation = {'factor':11/32,'Vol_preser':1, 'Fun_Error_Metric': 1, 'Fun_weigth':0.00} #decimate by a factor of 16
    #[verts1,faces1]= input_output.decimate_mesh(VS.numpy(),FS.numpy(),param_decimation)
    param_decimation = {'factor':29/32,'Vol_preser':1, 'Fun_Error_Metric': 1, 'Fun_weigth':0.00} #decimate by a factor of 16
    [verts2,faces2]= input_output.decimate_mesh(VT.numpy(),FT.verts_idx.numpy(),param_decimation)
    print(verts1.shape)
    print(verts2.shape)
    verts1 = torch.FloatTensor(verts1)
    verts2 = torch.FloatTensor(verts2)
    faces1 = torch.LongTensor(faces1)
    faces2 = torch.LongTensor(faces2)
    save_obj('../../results/bunny/sphere.obj', verts1, faces1)
    save_obj('../../results/bunny/bunny_red.obj', verts2, faces2)
elif experiment_name == 'plane':
    # sphere to dolphin
    VS, FS, _ = load_obj("../../data/matching/plane_source.obj")
    VT, FT, _ = load_obj("../../data/matching/plane_target.obj")
    # Decimate source mesh to compute initialization for the multires algorithm 
    param_decimation = {'factor':15/16,'Vol_preser':1, 'Fun_Error_Metric': 1, 'Fun_weigth':0.00} #decimate by a factor of 16
    [verts1,faces1]= input_output.decimate_mesh(VS.numpy(),FS.verts_idx.numpy(),param_decimation)
    param_decimation = {'factor':15/16,'Vol_preser':1, 'Fun_Error_Metric': 1, 'Fun_weigth':0.00} #decimate by a factor of 16
    [verts2,faces2]= input_output.decimate_mesh(VT.numpy(),FT.verts_idx.numpy(),param_decimation)
    print(verts1.shape)
    print(verts2.shape)
    verts1 = torch.FloatTensor(verts1)
    verts2 = torch.FloatTensor(verts2)
    faces1 = torch.LongTensor(faces1)
    faces2 = torch.LongTensor(faces2)

# verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
# faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
# For this tutorial, normals and textures are ignored.
faces_idx1 = faces1.to(device)
faces_idx2 = faces2.to(device)

# Mark: obj files
#faces_idx1 = faces1.to(device)#.verts_idx.to(device)
#faces_idx2 = faces2.verts_idx.to(device)

verts1 = verts1.to(device)
verts2 = verts2.to(device)
# We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
# (scale, center) will be used to bring the predicted mesh to its original center and scale
# Note that normalizing the target mesh, speeds up the optimization but is not necessary!
#'''
center1 = verts1.mean(0)
center2 = verts2.mean(0)
verts1 = verts1 - center1
verts2 = verts2 - center2
scale1 = max(verts1.abs().max(0)[0])
scale2 = max(verts2.abs().max(0)[0])
verts1 = verts1 / scale1
verts2 = verts2 / scale2
#'''
# We construct a Meshes structure for the target mesh
src_mesh = Meshes(verts=[verts1], faces=[faces_idx1])
trg_mesh = Meshes(verts=[verts2], faces=[faces_idx2])
print(verts1.shape)
print(verts2.shape)

class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
        self.net = nn.Sequential(nn.Linear(6,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),nn.Linear(128,3))
    def forward(self,x):
        return self.net(x)

# optimizer setting
models = testnet().cuda()
optimizer = torch.optim.Adam(models.parameters(), lr=.001)

# Number of optimization steps
Niter = 200001

# loss parameters
# weight for emd loss
w_emd = 1.0
# Plot period for the losses
plot_period = 1000

# EMD loss function
emd_fc = emdModule()

best = None
best_loss = 0
best_iter = 0
for i in range(Niter):

    # Initialize optimizer
    optimizer.zero_grad()
    # model train
    sv, sf = src_mesh.get_mesh_verts_faces(0)
    sn = src_mesh.verts_normals_packed()
    inputs = torch.cat([sv.cuda(),sn.cuda()],1)
    deform_verts = models(inputs) 
    
    # Deform the mesh
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    f1 = new_src_mesh.faces_packed()
    v1 = new_src_mesh.verts_packed()
    # We sample 5k points from the surface of each mesh 
    sample_trg,sample_trg_nor = sample_points_from_meshes(trg_mesh, 4096, return_normals=True)
    sample_src,sample_src_nor = sample_points_from_meshes(new_src_mesh, 4096, return_normals=True)

    # loss EMD 
    #print(sample_src.shape)
    emd_val, _ = emd_fc(sample_src,sample_trg, 1e-3*5, 50)
    loss_emd = emd_val.sum()
    
    loss_chamfer, _ = chamfer_distance(sample_src, sample_trg)

    # Weighted sum of the losses
    loss = loss_emd 

    if best_loss == 0:
        best = deform_verts
        best_loss = loss.detach()
        best_iter = 0
    elif best_loss > loss.detach():
        best = deform_verts
        best_loss = loss.detach()
        best_iter = i
        #torch.save(models.state_dict(),'../../results/emd_%s_n.pth'%experiment_name)
    
    # Print the losses
    if i % plot_period==0:
        print('%d Iter: total_loss %.6f EMD_loss %.6f Chamfer_loss %.6f'% (i,loss,loss_emd, loss_chamfer))
        print('current best loss is %d: %.6f'%(best_iter,best_loss))
        
    # Optimization step
    loss.backward()
    optimizer.step()

# Fetch the verts and faces of the final predicted mesh
new_src_mesh = src_mesh.offset_verts(best)
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

# Scale normalize back to the original target size
final_verts = (final_verts) * scale2 + center2

# Store the predicted mesh using save_obj
save_obj('../../results/%s/emd_%s_red.obj'%(experiment_name,experiment_name), final_verts, final_faces)
print('Done!')

final_chamfer,_ = chamfer_distance((final_verts.unsqueeze(0).double() - center2)/scale2, verts2.unsqueeze(0).double())
print('final Chamfer distance: %.6f'%(final_chamfer.detach().cpu().numpy()))
