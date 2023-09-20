import sys
sys.path.append( '../..' )
import os
import torch
import torch.nn as nn 
import torch.nn.functional as F
from pytorch3d.io import load_obj, save_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import numpy as np
from energy import tangent_kernel

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")
import input_output 


#experiment_name = 'hippo'
experiment_name = 'cup'
#experiment_name = 'dolphin'

# We read the target 3D model using load_obj
# case 1.  "Cup"
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
    param_decimation = {'factor':13/16,'Vol_preser':1, 'Fun_Error_Metric': 1, 'Fun_weigth':0.00} #decimate by a factor of 16
    [verts1,faces1]= input_output.decimate_mesh(VS,FS,param_decimation)
    print(verts1.shape)
    param_decimation = {'factor':13/16,'Vol_preser':1, 'Fun_Error_Metric': 1, 'Fun_weigth':0.00} #decimate by a factor of 16
    [verts2,faces2]= input_output.decimate_mesh(VT,FT,param_decimation)
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


class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
        self.net = nn.Sequential(nn.Linear(3,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),nn.Linear(128,3))
    def forward(self,x):
        return self.net(x)
    

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

# optimizer setting
#deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
#optimizer = torch.optim.SGD([deform_verts], lr=.01, momentum=0.9)
#optimizer = torch.optim.Adam([deform_verts], lr=.1)
models = testnet().cuda()
optimizer = torch.optim.Adam(models.parameters(), lr=.001)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[501,1001], gamma=0.1)

# Number of optimization steps
Niter = 20001

# loss parameters

# weight for varifold loss
w_varifold = 1.0 
# Weight for the chamfer loss
w_chamfer = 1.0 
# Weight for mesh edge loss
w_edge = 1.0 
# Weight for mesh normal consistency
w_normal = 0.1 
# Weight for mesh laplacian smoothing
w_laplacian = 0.1 
# Plot period for the losses
plot_period = 250


chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []

varifold = tangent_kernel(1,1.,0.05,3,mode='NTK2')
def compute_engine(V1,V2,L1,L2,K):
    cst_tmp = []
    n_batch = 10000#4096
    for i in range(len(V1)//n_batch + 1):
        tmp = V1[i*n_batch:(i+1)*n_batch,:]
        l_tmp = L1[i*n_batch:(i+1)*n_batch,:]
        v = torch.matmul(K(tmp,V2)*1000000,L2)*l_tmp
        cst_tmp.append(v)
    cst = torch.sum(torch.cat(cst_tmp,0))
    return cst

def CompCLNn(F, V):
    if F.shape[1] == 2:
        V0, V1 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1])
        C, N  =  (V0 + V1)/2, V1 - V0
    else:
        V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
        C, N =  (V0 + V1 + V2)/3, .5 * torch.cross(V1 - V0, V2 - V0)

    L = (N ** 2).sum(dim=1)[:, None].sqrt()
    return C, L, N / L, 1#Fun_F

c,l,n,_ = CompCLNn(faces_idx1,verts1)


best = None
best_loss = 0
best_iter = 0

for i in range(Niter):
    # Initialize optimizer
    optimizer.zero_grad()
    # Deform the mesh
    sv, sf = src_mesh.get_mesh_verts_faces(0)
    deform_verts = models(sv.cuda()) 
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    f1 = new_src_mesh.faces_packed()
    v1 = new_src_mesh.verts_packed()
    # We sample 5k points from the surface of each mesh 
    c1,l1,n1,_ = CompCLNn(f1,v1)
    c2,l2,n2,_ = CompCLNn(faces_idx2,verts2)
    c1 = torch.cat([c1,n1],1)
    c2 = torch.cat([c2,n2],1)
    v11 = compute_engine(c1,c1,l1,l1,varifold)
    v22 = compute_engine(c2,c2,l2,l2,varifold) 
    v12 = compute_engine(c2,c1,l2,l1,varifold)
    loss_varifold = v11 + v22 -2*v12
    
    # loss chamfer as reference
    loss_chamfer, _ = chamfer_distance(c1.unsqueeze(0), c2.unsqueeze(0))
    
    # and (b) the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(new_src_mesh)
    
    # mesh normal consistency
    loss_normal = mesh_normal_consistency(new_src_mesh)
    #print(loss_normal)
    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    # Weighted sum of the losses
    # note PointNet-NTK1 param = 20 & learning rate 0.05 PointNet-NTK2 = 10 learning rate 0.1
    loss = w_varifold *loss_varifold 

    if best_loss == 0:
        best = deform_verts
        best_loss = loss.detach()
        best_iter = 0
    elif best_loss > loss.detach():
        best = deform_verts
        best_loss = loss.detach()
        best_iter = i
        torch.save(models.state_dict(),'../../results/pointnet_ntk2_%s.pth'%experiment_name)

    # Print the losses
    if i % plot_period==0:
        print('%d Iter: total_loss %.6f Chamfer_loss %.6f Varifold loss %.8f'% (i,loss,loss_chamfer, loss_varifold))
        print('current best loss is %d: %.6f'%(best_iter,best_loss))

    # Save the losses for plotting
    chamfer_losses.append(float(loss_chamfer.detach().cpu()))
    edge_losses.append(float(loss_edge.detach().cpu()))
    normal_losses.append(float(loss_normal.detach().cpu()))
    laplacian_losses.append(float(loss_laplacian.detach().cpu()))
        
    # Optimization step
    loss.backward()
    optimizer.step()

# Fetch the verts and faces of the final predicted mesh
new_src_mesh = src_mesh.offset_verts(best)
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

# Scale normalize back to the original target size
final_verts = (final_verts) * scale2 + center2

# Store the predicted mesh using save_obj
#save_obj('./chamfer.obj', final_verts, final_faces)
#save_obj('./charon_trouve.obj', final_verts, final_faces)
save_obj('../../results/pointnet_ntk2_%s2m.obj'%experiment_name, final_verts, final_faces)
#save_obj('./pointnet_ntk2.obj', final_verts, final_faces)
print('Done!')

final_chamfer,_ = chamfer_distance((final_verts.unsqueeze(0).double() - center2)/scale2, verts2.unsqueeze(0).double())
print('final Chamfer distance: %.6f'%(final_chamfer.detach().cpu().numpy()))