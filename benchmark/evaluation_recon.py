import torch
import pytorch3d
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

import numpy as np

def read_meshes(path):
    if path[-3:] =='ply':
        verts, faces_idx = load_ply(path)
        mesh = Meshes(verts=[verts], faces=[faces_idx])
    elif path[-3:] =='obj':
        verts, faces, aux = load_obj(path)
        faces_idx = faces.verts_idx
        mesh= Meshes(verts=[verts], faces=[faces_idx])
    else:
        raise ValueError('not supported')
    return mesh

def evaluate_results(orignial_path, recon_path):
    torch.manual_seed(0)
    original_mesh = read_meshes(orignial_path)
    recon_mesh    = read_meshes(recon_path)

    sample_src = sample_points_from_meshes(recon_mesh, 2048)
    sample_trg = sample_points_from_meshes(original_mesh, 2048)

    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
    return loss_chamfer

if __name__=='__main__':
    # 13 clases shapenet objects
    classes = ['airplane','bench','cabinet','car','chair',
    'display','lamp','loudspeaker','rifle','sofa',
    'table','telephone','watercraft']
    # data path
    data_path = '../data/processed/'
    
    # PointNet-NTK
    #exp = 'ntk'
    #recon_path = '../results/ntkz/'
    # Neural-Spline
    #exp = 'spline'
    #recon_path = '../../neural_spline_temp/results/shapenet/'
    # SIREN
    exp = 'siren'
    recon_path = '../../siren/results/'

    print('initiate evaluating method for %s'%exp)
    res = []
    for j,cl in enumerate(classes):
        tmp = torch.zeros(20)
        print('processing class: %s'%cl)
        for i in range(20):
            val = evaluate_results(data_path+'%s/%s%02d.obj'%(cl,cl,i),recon_path+'%s%02d_%s.ply'%(cl,i,exp))
            tmp[i] = val
        res.append(torch.median(tmp))

    print('accuracy airplane %.6f'%res[0])
    print('accuracy bench %.6f'%res[1])    
    print('accuracy cabinet %.6f'%res[2])
    print('accuracy car %.6f'%res[3])
    print('accuracy chair %.6f'%res[4])
    print('accuracy display %.6f'%res[5])
    print('accuracy lamp %.6f'%res[6])
    print('accuracy loudspeaker %.6f'%res[7])
    print('accuracy rifle %.6f'%res[8])
    print('accuracy sofa %.6f'%res[9])
    print('accuracy table %.6f'%res[10])
    print('accuracy telephone %.6f'%res[11])
    print('accuracy watercraft %.6f'%res[12])