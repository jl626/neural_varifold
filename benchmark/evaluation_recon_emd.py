import torch
import pytorch3d
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
#from sinkhorn import sinkhorn
import numpy as np

import time
import numpy as np
from torch import nn
from torch.autograd import Function
import emd




class emdFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters):

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        assert(n == m)
        assert(xyz1.size()[0] == xyz2.size()[0])
        #assert(n % 1024 == 0)
        assert(batchsize <= 512)

        xyz1 = xyz1.contiguous().float().cuda()
        xyz2 = xyz2.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batchsize, m, device='cuda', dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batchsize, m, device='cuda').contiguous()
        bid = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device='cuda').contiguous()
        max_increments = torch.zeros(batchsize, m, device='cuda').contiguous()
        unass_idx = torch.zeros(batchsize * n, device='cuda', dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batchsize * m, device='cuda', dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()

        emd.forward(xyz1, xyz2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx, unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters)

        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx):
        xyz1, xyz2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()

        gradxyz1 = torch.zeros(xyz1.size(), device='cuda').contiguous()
        gradxyz2 = torch.zeros(xyz2.size(), device='cuda').contiguous()

        emd.backward(xyz1, xyz2, gradxyz1, graddist, assignment)
        return gradxyz1, gradxyz2, None, None

class emdModule(nn.Module):
    def __init__(self):
        super(emdModule, self).__init__()

    def forward(self, input1, input2, eps, iters):
        return emdFunction.apply(input1, input2, eps, iters)

def test_emd():
    x1 = torch.rand(20, 8192, 3).cuda()
    x2 = torch.rand(20, 8192, 3).cuda()
    emd = emdModule()
    start_time = time.perf_counter()
    dis, assigment = emd(x1, x2, 1e-3, 3000)
    print("Input_size: ", x1.shape)
    print("Runtime: %lfs" % (time.perf_counter() - start_time))
    print("EMD: %lf" % np.sqrt(dis.cpu()).mean())
    print("|set(assignment)|: %d" % assigment.unique().numel())
    assigment = assigment.cpu().numpy()
    assigment = np.expand_dims(assigment, -1)
    x2 = np.take_along_axis(x2, assigment, axis = 1)
    d = (x1 - x2) * (x1 - x2)
    print("Verified EMD: %lf" % np.sqrt(d.cpu().sum(-1)).mean())


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

    sample_src = sample_points_from_meshes(recon_mesh, 2048)#.squeeze()
    sample_trg = sample_points_from_meshes(original_mesh, 2048)#.squeeze()
    #print(sample_src.shape)
    niters = 500
    device = 'cuda'
    dtype = torch.float32
    eps = 1e-3
    stop_error = 1e-3
    #loss, corrs_1_to_2, corrs_2_to_1 = \
    #    sinkhorn(sample_trg, sample_src, p=2, eps=eps, max_iters=niters, stop_thresh=stop_error, verbose=True)
    emd = emdModule()
    loss_emd, assigment = emd(sample_src,sample_trg, 1e-3, 3000)#chamfer_distance(sample_trg, sample_src)
    #print(loss_emd)
    return loss_emd.mean()

if __name__=='__main__':
    # 13 clases shapenet objects
    classes = ['airplane','bench','cabinet','car','chair',
    'display','lamp','loudspeaker','rifle','sofa',
    'table','telephone','watercraft']
    #classes = ['telephone']
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
        #break
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