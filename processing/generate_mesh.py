import argparse
import os
import numpy as np
import pandas as pd
import os, sys
import trimesh
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from libmesh import check_mesh_contains
import gco
import torch.nn.functional as F
from evaluate_mesh import compute_iou, compute_chamfer



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
    mfile = os.path.join(data.path, "gt", data.scan_conf, data.id, data.filename + "_3dt.npz")



    mdata = np.load(mfile)
    assert(len(labels)==len(mdata["tetrahedra"]))

    # take out all the infinite cells for the graph cut
    edges = mdata['nfacets']

    if(clf.temp.graph_cut):
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
    if(clf.temp.fix_orientation):
        trimesh.repair.fix_normals(recon_mesh)

    eval_dict = dict()

    ### watertight ###
    if("watertight" in clf.temp.eval):
        eval_dict["watertight"] = int(recon_mesh.is_watertight)
        # print("WARNING: Mesh is not watertight: ", data['filename'])

    subfolder = data['id'] if data['id'] else data['category']

    ### IOU ###
    if('iou' in clf.temp.eval):
        occ_file = os.path.join(data.path, "eval", subfolder, "points.npz")
        occ = np.load(occ_file)
        occ_points = occ["points"]
        gt_occ = occ["occupancies"]
        gt_occ = np.unpackbits(gt_occ)[:occ_points.shape[0]]
        gt_occ = gt_occ.astype(np.float32)

        try:
            recon_occ = check_mesh_contains(recon_mesh,occ_points)
            eval_dict["iou"] = compute_iou(gt_occ, recon_occ)
        except:
            print("WARNING: Could not calculate IoU for mesh ",data['filename'])
            eval_dict["iou"] = 0.0

    ### Chamfer ###
    if('chamfer' in clf.temp.eval):
        points_file = os.path.join(data.path, "eval", subfolder, "pointcloud.npz")
        points = np.load(points_file)
        gt_points = points["points"]
        gt_points = gt_points.astype(np.float32)

        ## TODO: set return index = True and get face normals to compute a normal consistency
        recon_points = recon_mesh.sample(gt_points.shape[0],return_index=False)

        try:
            eval_dict["chamfer"] = compute_chamfer(gt_points, recon_points) # this is already two-sided
        except:
            print("WARNING: Could not calculate Chamfer distance for mesh ",data['filename'])
            eval_dict["iou"] = 0.0



    return recon_mesh, eval_dict





