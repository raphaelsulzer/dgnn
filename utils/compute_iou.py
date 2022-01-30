import open3d as o3d
import numpy as np
import os
import trimesh

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


b="/home/adminlocal/PhD/data/ShapeNet/meshes/02691156/d18592d9615b01bbbc0909d98a1ff2b4"
#
mesh = trimesh.load(b+".off")

file = os.path.join(b, 'eval', 'points.npz')
data = np.load(file)

points = data["points"]
occ = data["occupancies"]
occ = np.unpackbits(occ)
occ_recon = mesh.contains(points)

iou = compute_iou(occ, occ_recon)

print(iou)

