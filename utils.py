import numpy as np
import torch 


def point_cloud_bounding_box(x, scale=1.0):
    """
    Get the axis-aligned bounding box for a point cloud (possibly scaled by some factor)
    :param x: A point cloud represented as an [N, 3]-shaped tensor
    :param scale: A scale factor by which to scale the bounding box diagonal
    :return: The (possibly scaled) axis-aligned bounding box for a point cloud represented as a pair (origin, size)
    """
    bb_min = x.min(0)[0]
    bb_size = x.max(0)[0] - bb_min
    return scale_bounding_box_diameter((bb_min, bb_size), scale)
    
def affine_transform_pointcloud(x, tx):
    """
    Apply the affine transform tx to the point cloud x
    :param x: A pytorch tensor of shape [N, 3]
    :param tx: An affine transformation represented as a tuple (t, s) where t is a translation and s is scale
    :return: The transformed point cloud
    """
    translate, scale = tx
    return scale * (x + translate)


def scale_bounding_box_diameter(bbox, scale):
    """
    Scale the diagonal of the bounding box bbox while maintaining its center position
    :param bbox: A bounding box represented as a pair (origin, size)
    :param scale: A scale factor by which to scale the input bounding box's diagonal
    :return: The (possibly scaled) axis-aligned bounding box for a point cloud represented as a pair (origin, size)
    """
    bb_min, bb_size = bbox
    bb_diameter = torch.norm(bb_size)
    bb_unit_dir = bb_size / bb_diameter
    scaled_bb_size = bb_size * scale
    scaled_bb_diameter = torch.norm(scaled_bb_size)
    scaled_bb_min = bb_min - 0.5 * (scaled_bb_diameter - bb_diameter) * bb_unit_dir
    return scaled_bb_min, scaled_bb_size

def normalize_pointcloud_transform(x):
    """
    Compute an affine transformation that normalizes the point cloud x to lie in [-0.5, 0.5]^2
    :param x: A point cloud represented as a tensor of shape [N, 3]
    :return: An affine transformation represented as a tuple (t, s) where t is a translation and s is scale
    """
    min_x, max_x = x.min(0)[0], x.max(0)[0]
    bbox_size = max_x - min_x

    translate = -(min_x + 0.5 * bbox_size)
    scale = 1.0 / torch.max(bbox_size)

    return translate, scale


def affine_transform_pointcloud(x, tx):
    """
    Apply the affine transform tx to the point cloud x
    :param x: A pytorch tensor of shape [N, 3]
    :param tx: An affine transformation represented as a tuple (t, s) where t is a translation and s is scale
    :return: The transformed point cloud
    """
    translate, scale = tx
    return scale * (x + translate)


def affine_transform_bounding_box(bbox, tx):
    """
    Apply the affine transform tx to the bounding box bbox
    :param bbox: A bounding box reprented as 2 3D vectors (origin, size)
    :param tx: An affine transformation represented as a tuple (t, s) where t is a translation and s is scale
    :return: The transformed point bounding box
    """
    translate, scale = tx
    return scale * (bbox[0] + translate), scale * bbox[1]


def triple_points_along_normals(x, n, eps, homogeneous=False):
    """
    Convert a point cloud equipped with normals into a point cloud with points pertubed along those normals.
    Each point X with normal N, in the input gets converted to 3 points:
        (X, X+eps*N, X-eps*N) which have occupancy values (0, eps, -eps)
    :param x: The input points of shape [N, 3]
    :param n: The input normals of shape [N, 3]
    :param eps: The amount to perturb points about each normal
    :param homogeneous: If true, return the points in homogeneous coordinates
    :return: A pair, (X, O) consisting of the new point cloud X and point occupancies O
    """
    x_in = x - n * eps
    x_out = x + n * eps

    x_triples = torch.cat([x, x_in, x_out], dim=0)
    occ_triples = torch.cat([torch.zeros(x.shape[0]),
                             -torch.ones(x.shape[0]),
                             torch.ones(x.shape[0])]).to(x) * eps
    if homogeneous:
        x_triples = torch.cat([x_triples, torch.ones(x_triples.shape[0], 1, dtype=x_triples.dtype)], dim=-1)

    return x_triples, occ_triples
