import numpy as np
from scipy.spatial import KDTree

# def compute_iou(gt_mesh,recon_mesh):
#     test_points = 10000
#     succesfully_tested_points = 0
#     intersection = 0
#
#     while(succesfully_tested_points < test_points):
#
#         point = (np.random.rand(1,3)-0.5)*1.05
#
#         gt = check_mesh_contains(gt_mesh,point)
#         recon = check_mesh_contains(recon_mesh,point)
#
#         if(gt and recon):
#             succesfully_tested_points+=1
#             intersection+=1
#         elif(gt or recon):
#             succesfully_tested_points+=1
#
#
#     iou = intersection / test_points



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

def compute_chamfer(points1,points2):

    k1 = KDTree(points1)
    k2 = KDTree(points2)

    dist1 = k1.query(points2)[0].mean()
    dist2 = k2.query(points1)[0].mean()


    return (dist1+dist2)/2