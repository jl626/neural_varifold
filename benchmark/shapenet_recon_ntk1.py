import sys
sys.path.append( '..' )
import os 
os.environ['CUDA_VISIBLE_DEVICES']='0'
import glob
import time
from absl import app
from absl import flags

import jax 
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util
from neural_tangents._src.utils.kernel import Kernel
from jax import random

# double precision otherwise it may return NaN
from jax.config import config ; config.update('jax_enable_x64', True)

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
from utils import *
from util import Prod

import numpy as onp
from skimage import measure
import trimesh 
import torch_geometric
import open3d as o3d

def run_shapenet_test(paths):

    print('Loading data.')

    # hyper parameters
    onp.random.seed(0)
    n_train = 2048 # subsample n out of 10000 points 
    idx = np.random.choice(10000,n_train,replace=False)

    # Load data 
    #data = onp.loadtxt('/home/juheonlee/learning/pointnet2/data/modelnet40_normal_resampled/airplane/airplane_0002.txt',delimiter=',')
    #data = onp.loadtxt('/home/juheonlee/learning/neural_varifold/data/processed/airplane/airplane00.txt',delimiter=' ')
    data = np.loadtxt(paths,delimiter=' ')
    pos = torch.DoubleTensor(data[idx,:3]) # convert it Double Precision 
    nor = torch.DoubleTensor(data[idx,3:]) # convert it Double Precision

    # 3D regular grid structure parameters 
    nb = 128  # voxel resolution (128 X 128 X 128) 
    eps = 0.5 # 
    scaled_bbox = point_cloud_bounding_box(pos, 1.1) # compute bound eq: (max_val_per_axis - min_val_per_axis)/scale_param
    out_grid_size = torch.round(scaled_bbox[1] / scaled_bbox[1].max() * nb).to(torch.int32)   
    voxel_size = scaled_bbox[1] / out_grid_size  # size of one voxel
    eps_world_coords = eps * torch.norm(voxel_size).item()
    print(eps_world_coords)

    # Pre-processing of the point clouds (borrowed from Williams et a. 2021)

    # step 1 - generate extended dataset by moving orignial point clouds towards positive/negative directions toward the normals * eps i.e. data size = N_points * 3 X 3
    print(pos.shape)
    x_train, y_train = triple_points_along_normals(pos, nor, eps_world_coords, homogeneous=False)
    print(x_train.shape)

    # step 2 - normalize point clouds: as we are going to estimate SDF for all regular grid points 
    tx = normalize_pointcloud_transform(x_train)

    # step 3 conduct affine transform to fit the shape into the regular grid box we chose
    x_train = affine_transform_pointcloud(x_train, tx)
        
    # generating normal triplet (i.e. simply duplicating normals on the original point clouds) 
    nors = torch.cat([nor,-nor,nor], dim=0)  # normal triplelet  

    # final training data set
    x_train = torch.cat([x_train, nors], dim=-1).numpy().reshape(n_train*3,2,3)
    y_train = y_train.unsqueeze(-1).numpy()

    print(x_train.shape)
    
    # inference grid structure
    bbox_origin, bbox_size = scaled_bbox
    voxel_size = bbox_size / out_grid_size  # size of a single voxel cell

    cell_vox_min = torch.tensor([0, 0, 0], dtype=torch.int32)
    cell_vox_max = out_grid_size


    print(f"Evaluating model on grid of size {[_.item() for _ in (cell_vox_max - cell_vox_min)]}.")
    eval_start_time = time.time()

    xmin = bbox_origin + (cell_vox_min + 0.5) * voxel_size
    xmax = bbox_origin + (cell_vox_max - 0.5) * voxel_size

    xmin = affine_transform_pointcloud(xmin.unsqueeze(0), tx).squeeze()
    xmax = affine_transform_pointcloud(xmax.unsqueeze(0), tx).squeeze()

    xmin, xmax = xmin.numpy(), xmax.numpy()
    cell_vox_size = (cell_vox_max - cell_vox_min).numpy()

    # generate regular voxel
    xgrid = np.stack([_.ravel() for _ in np.mgrid[xmin[0]:xmax[0]:cell_vox_size[0] * 1j,
                                                    xmin[1]:xmax[1]:cell_vox_size[1] * 1j,
                                                    xmin[2]:xmax[2]:cell_vox_size[2] * 1j]], axis=-1)
    xgrid = torch.from_numpy(xgrid).to(torch.float64)


    # applying pseudo-normals on inference grids
    # method 1. k-nearnest neighbour interpolation given (high error)
    print(pos.dtype)
    #idx1,idx2 = torch_geometric.nn.knn(pos.cuda(),xgrid.cuda(),1,cosine=True)
    #pseudo = nor[idx2.cpu()]
    #pseudo = torch.FloatTensor([0,0,1]).repeat(len(xgrid),1)
    pseudo = torch.FloatTensor([np.sqrt(1/3),np.sqrt(1/3),np.sqrt(1/3)]).repeat(len(xgrid),1)
    print(xgrid.shape)
    print(pseudo.shape)


    # test data (regular 3D voxel grids)
    x_test = torch.cat([xgrid, pseudo.to(xgrid)], dim=-1).to(torch.float64).numpy().reshape(len(xgrid),2,3)


    # main
    n_layers = 1
    print('build %d-hidden_layer infinite network'%n_layers)
    # Build the infinite network.
    layers = []
    for _ in range(n_layers):
        layers += [stax.Dense(1, 1., 100.), stax.Relu()]
    init_fn, apply_fn, kernel_fn = stax.serial(*(layers + [Prod()]))

    # Optionally, compute the kernel in batches, in parallel.
    kernel_fn = nt.batch(kernel_fn,
                        device_count=0,
                        batch_size=n_train)
    
    start = time.time()

    print('train kernel')
    # Bayesian and infinite-time gradient descent inference with infinite network.
    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train,y_train, diag_reg=1e-6)

    n_batch=n_train//2
    nngp = []
    ntk  = []
    print('inference kenrel')
    # batch process (otherwise cost too much memory)
    for i in range(len(x_test)//n_batch):
        if i == (len(x_test)//n_batch) -1:
            x2 = x_test[int(i*n_batch):]
        else:
            x2 = x_test[int(i*n_batch):int((i+1)*n_batch)]
        fx_test_nngp, fx_test_ntk = predict_fn(x_test=x2)
        fx_test_nngp.block_until_ready()
        fx_test_ntk.block_until_ready()
        ntk.append(onp.array(fx_test_ntk))
        nngp.append(onp.array(fx_test_nngp))

    # merge
    ntk  = onp.concatenate(ntk,0)
    nngp = onp.concatenate(nngp,0)
    ntk  = ntk.reshape(tuple(cell_vox_size.astype(np.int32)))
    nngp = nngp.reshape(tuple(cell_vox_size.astype(np.int32)))
    duration = time.time() - start
    print('Kernel construction and inference done in %s seconds.' % duration)

    # marching cube algorithm to reconstruct
    verts, faces, normals, values = measure.marching_cubes(ntk, level=0.0, spacing=voxel_size)

    verts += scaled_bbox[0].numpy() + 0.5 * voxel_size.numpy()

    # save file 
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    names = paths.split('/')[-1][:-4]
    o3d.io.write_triangle_mesh('../results/ntku_2048/%s_ntk.ply'%names, mesh)

if __name__=="__main__":
    # 13 clases shapenet objects
    classes = ['airplane','bench','cabinet','car','chair',
    'display','lamp','loudspeaker','rifle','sofa',
    'table','telephone','watercraft']
    for name in classes:
        base_path = '/home/juheonlee/learning/neural_varifold/data/processed/'
        data_path = glob.glob(base_path + name + '/*.txt')
        for i in range(len(data_path)):
            run_shapenet_test(data_path[i])