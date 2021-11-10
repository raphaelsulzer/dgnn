import argparse
import os
import numpy as np
import pandas as pd
import os, sys
import trimesh
sys.path.append(os.path.join(os.path.dirname(__file__), '..','utils'))
from libmesh import check_mesh_contains
import gco
import torch.nn.functional as F


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



def graph_cut(labels,prediction,edges):

    gc = gco.GCO()
    gc.create_general_graph(edges.max()+1, 2)
    # data_cost = F.softmax(prediction, dim=-1)
    prediction[:, [0, 1]] = prediction[:, [1, 0]]
    data_cost = (prediction*100).round()

    data_cost = np.array(data_cost,dtype=int)
    ### append high cost for inside for infinite cell
    # data_cost = np.append(data_cost, np.array([[-10000, 10000]]), axis=0)
    gc.set_data_cost(data_cost)
    smooth = (1 - np.eye(2)).astype(int)
    gc.set_smooth_cost(smooth)
    gc.set_all_neighbors(edges[:,0],edges[:,1],np.ones(edges.shape[0],dtype=int)*100)

    for i,l in enumerate(labels):
        gc.init_label_at_site(i,l)

    # print("before smooth: ",gc.compute_smooth_energy())
    # print("before data: ", gc.compute_data_energy())
    # print("before: ", gc.compute_smooth_energy()+gc.compute_data_energy())

    gc.expansion()

    # print("after: ", gc.compute_smooth_energy()+gc.compute_data_energy())

    # print("after smooth: ",gc.compute_smooth_energy())
    # print("after data: ", gc.compute_data_energy())

    labels = gc.get_labels()

    return labels


def generate(data, prediction, clf):


    labels = F.log_softmax(prediction[data.y[:, 4] == 0], dim=-1).argmax(1).numpy()



    ### reconstruction
    mfile = os.path.join(clf.paths.data, data.category, "gt", str(clf.data.scan_confs[0]), \
                         data.id, data.category + "_" + data.id + "_3dt.npz")
    mdata = np.load(mfile)
    assert(len(labels)==len(mdata["tetrahedra"]))

    # take out all the infinite cells for the graph cut
    edges = mdata['nfacets']


    # w/o gc 0.715
    # with alpha=1 iou: 0.716
    # with alpha=100 iou: 0.752
    if(clf.inference.graph_cut):
        mask = (edges >= 0).all(axis=1)
        gc_edges = edges[mask]
        labels=graph_cut(labels,prediction[data.y[:, 4] == 0],gc_edges)

    # add a last cell as the infinite cell
    for i, e in enumerate(edges):
        for j, c in enumerate(e):
            if (c == -1):
                edges[i, j] = labels.shape[0]
    # make it an outside cell
    labels=np.append(labels, 1)

    # extract the interface and save it as a surface mesh
    interfaces = []
    for fi,f in enumerate(edges):
        if(labels[f[0]]!=labels[f[1]]):
            interfaces.append(fi)

    recon_mesh = trimesh.Trimesh(mdata["vertices"], mdata["facets"][interfaces],process=True)
    if(clf.inference.fix_orientation):
        trimesh.repair.fix_normals(recon_mesh)


    ### gt_mesh
    # gt_file = os.path.join(clf.paths.data, file.category, "2_watertight", file.category + "_" + file.id + ".off")
    # gt_mesh = trimesh.load(gt_file, process=False)
    # n_points_uniform = 10000
    #
    # boxsize = 1 + 0.1
    # points = np.random.rand(n_points_uniform, 3)
    # points = boxsize * (points - 0.5)
    # gt_occ = check_mesh_contains(gt_mesh,points)
    # recon_occ = check_mesh_contains(recon_mesh,points)

    gt_file = os.path.join(clf.paths.data, data.category, "convonet", str(clf.data.scan_confs[0]), data.id, "points.npz")
    gt_data = np.load(gt_file)

    points = gt_data["points"]

    gt_occ = gt_data["occupancies"]
    gt_occ = np.unpackbits(gt_occ)[:points.shape[0]]
    gt_occ = gt_occ.astype(np.float32)

    recon_occ = check_mesh_contains(recon_mesh,points)

    iou = compute_iou(gt_occ,recon_occ)

    return recon_mesh,iou





