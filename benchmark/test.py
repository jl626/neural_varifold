import os 
import sys 
sys.path.append( '..' )

import input_output 

import torch

# EMD metric
from evaluation_recon_emd import emdModule
from energy import tangent_kernel

# Chamfer metric
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

torch.manual_seed(0)
torch.backends.cudnn.deterministic =True

experiments = "hippo"
#experiments = "cup"
#experiments = "dolphin"

#names = 'chamfer'
#names = 'binet'
#names = 'emd'
names = 'pointnet_ntk1'
#names = 'pointnet_ntk2'

#types = '_low'
types = ''


if names == 'binet':
    verts1,faces1,_ = load_obj('../results/%s/%s_%s%s.obj'%(experiments,names,experiments,types))
elif names == "chamfer":
    verts1,faces1,_ = load_obj('../results/%s/%s_%s_pts%s.obj'%(experiments,names,experiments,types))
elif names == "emd":
    verts1,faces1,_ = load_obj('../results/%s/%s_%s%s.obj'%(experiments,names,experiments,types))
elif names == "pointnet_ntk1":
    verts1,faces1,_ = load_obj('../results/%s/%s_%s%s.obj'%(experiments,names,experiments,types))
elif names == "pointnet_ntk2":
    verts1,faces1,_ = load_obj('../results/%s/%s_%s%s.obj'%(experiments,names,experiments,types))



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

# no need scale - GT/matching must have the same scale!
center1 = verts1.mean(0)
center2 = verts2.mean(0)
verts1 = verts1 - center1
verts2 = verts2 - center2
scale1 = max(verts1.abs().max(0)[0])
scale2 = max(verts2.abs().max(0)[0])
verts1 = verts1 / scale1
verts2 = verts2 / scale2

new_verts = verts1 * scale2 + center2
#save_obj('../results/test_chamfer.obj', verts1, faces1.verts_idx)
save_obj('../results/test_ntk.obj', new_verts, faces1.verts_idx)
#save_obj('../results/test_gt.obj', verts2, faces2.verts_idx)