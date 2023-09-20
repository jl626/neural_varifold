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

torch.manual_seed(0)
torch.backends.cudnn.deterministic =True

experiment_name = 'hippo'
#experiment_name = 'cup'
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
    param_decimation = {'factor':31/32,'Vol_preser':1, 'Fun_Error_Metric': 1, 'Fun_weigth':0.00} #decimate by a factor of 16
    [verts1,faces1]= input_output.decimate_mesh(VS,FS,param_decimation)
    print(verts1.shape)
    param_decimation = {'factor':31/32,'Vol_preser':1, 'Fun_Error_Metric': 1, 'Fun_weigth':0.00} #decimate by a factor of 16
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