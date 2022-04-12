import argparse
import os
import numpy as np
import pandas as pd
import os, sys
import trimesh
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from libmesh import check_mesh_contains
import gco # pip install gco-wrapper
import torch.nn.functional as F
from evaluate_mesh import compute_iou, compute_chamfer



def graph_cut(labels,prediction,edges,clf):

    dtype = np.int64
    # Internally, in the pyGCO code datatype is always converted to np.int32
    # I would need to use my own version of GCO (or modify the one used by pyGCO) to change that
    # Should probably be done at some point to avoid int32 overflow for larger scenes.

    gc = gco.GCO()
    gc.create_general_graph(edges.max()+1, 2, energy_is_float=False)
    # data_cost = F.softmax(prediction, dim=-1)
    prediction[:, [0, 1]] = prediction[:, [1, 0]]
    data_cost = (prediction*clf.graph_cut.unary_weight).round()
    # data_cost = prediction
    data_cost = np.array(data_cost,dtype=dtype)
    ### append high cost for inside for infinite cell
    # data_cost = np.append(data_cost, np.array([[-10000, 10000]]), axis=0)
    gc.set_data_cost(data_cost)
    smooth = (1 - np.eye(2)).astype(dtype)
    gc.set_smooth_cost(smooth)
    if(not clf.graph_cut.binary_type):
        edge_weight = np.ones(edges.shape[0],dtype=dtype)
    else:
        # TODO: retrieve the beta-skeleton and area value from the features to weight the binaries
        edge_weight = np.ones(edges.shape[0], dtype=dtype)

    gc.set_all_neighbors(edges[:,0],edges[:,1],edge_weight*clf.graph_cut.binary_weight)

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

    """This function generates a mesh from cell predictions and evaluates the result using presampled points in (for IoU) and on the ground truth
    mesh (for chamfer).
    It returns the mesh (as trimesh object) and a dictionary with the evaluation metrics."""

    # get an initial label to init the graph cut, as max of prediction
    labels = F.log_softmax(prediction, dim=-1).argmax(1).numpy()

    mfile = os.path.join(data.path, data.gtfile + "_3dt.npz")
    mdata = np.load(mfile)
    assert (len(labels) == len(mdata["tetrahedra"]))
    edges = mdata['nfacets']

    # graph cut
    if(clf.temp.graph_cut):
        try:
            labels = graph_cut(labels, prediction, edges, clf)
        except Exception as e:
            print('\n')
            print(e)
            print("WARNING: Graph cut for {} didn't work. Using raw predictions for mesh generation.".format(data.filename))

    # open or closed object?
    # if open object, set closed to False, and the graph-cut / network label will be used for infinite cells
    closed = True
    if closed:
        # forcing infinite cells to be outside
        labels[mdata["inf_tetrahedra"].astype(bool)] = 1
    # # add a last cell as the infinite cell
    # edges[(edges[:,0] == -1).astype(bool),0] = labels.shape[0]
    # edges[(edges[:,1] == -1).astype(bool),1] = labels.shape[0]
    #
    # # make it an outside cell
    # labels=np.append(labels, 1)

    # get index of interface facets (i.e. where labels are not equal)
    interfaces = np.argwhere(labels[edges[:,0]]!=labels[edges[:,1]])
    interfaces = np.squeeze(interfaces)

    # make the mesh
    recon_mesh = trimesh.Trimesh(mdata["vertices"], mdata["facets"][interfaces], process=True)
    if(clf.temp.fix_orientation):
        trimesh.repair.fix_normals(recon_mesh)
        # trimesh.repair.fix_inversion(recon_mesh)
        # TODO: check to see if this shouldn't be rather trimesh.repair.fix_inversion (maybe it is faster)

    eval_dict = dict()

    ### watertight ###
    if("watertight" in clf.temp.metrics):
        # TODO: this does not really give the correct result, as it seems to simply count boundary edges,
        # which are also all non-manifold edges. Thus, if the mesh has any non-manifold edge, it will also be counted as non-watertight
        # maybe use Open3D for this task instead, which has support for both, non-manifold and watertight.
        # UPDATE: actually recon_mesh.as_open3d works
        eval_dict["watertight"] = int(recon_mesh.as_open3d.is_watertight())
        # eval_dict["watertight"] = int(recon_mesh.is_watertight)
        # print("WARNING: Mesh is not watertight: ", data['filename'])

    subfolder = data['id'] if data['id'] else data['category']

    ### IOU ###
    if('iou' in clf.temp.metrics):
        if("ioufile" in data):
            occ_file = os.path.join(data.path,data["ioufile"])
        else:
            if(os.path.exists(os.path.join(data.path, "eval", "points.npz"))):
                occ_file = os.path.join(data.path, "eval", "points.npz")
            else:
                occ_file = os.path.join(data.path, "eval", subfolder, "points.npz")
        occ = np.load(occ_file)
        occ_points = occ["points"]
        gt_occ = occ["occupancies"]
        gt_occ = np.unpackbits(gt_occ)
        try:
            recon_occ = check_mesh_contains(recon_mesh,occ_points)
            eval_dict["iou"] = compute_iou(recon_occ, gt_occ)
        except Exception as e:
            print('\n')
            print(e)
            print("\nWARNING: Could not calculate IoU for mesh ",data['filename'])
            eval_dict["iou"] = 0.0

    ### Chamfer ###
    if('chamfer' in clf.temp.metrics):
        points_file = os.path.join(data.path, "eval", subfolder, "pointcloud.npz")
        points = np.load(points_file)
        gt_points = points["points"]
        gt_points = gt_points.astype(np.float32)

        ## TODO: set return index = True and get face normals to compute a normal consistency
        recon_points = recon_mesh.sample(gt_points.shape[0],return_index=False)

        try:
            eval_dict["chamfer"] = compute_chamfer(gt_points, recon_points) # this is already two-sided
        except Exception as e:
            print('\n')
            print(e)
            print("WARNING: Could not calculate Chamfer distance for mesh ",data['filename'])
            eval_dict["iou"] = 0.0



    return recon_mesh, eval_dict





