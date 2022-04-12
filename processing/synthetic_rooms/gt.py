# import the 3DT
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..', 'utils'))
from pointInTetrahedron import pit
from libmesh import check_mesh_contains
import numpy as np
import trimesh
import datetime


n_points = 10
export_points = 0

def gt(path,model):

    start = datetime.datetime.now()

    mesh = trimesh.load_mesh(os.path.join(path,model, "mesh.off"),process=False,use_embree=False)
    bbox = mesh.bounds
    dt = np.load(os.path.join(path,model,"dgnn",model+"_3dt.npz"))

    points = dt["vertices"]

    test_points = []
    occs = []

    for i,tet in enumerate(dt["tetrahedra"]):
        for j in range(n_points):
            point = pit(points[tet[0]], points[tet[1]], points[tet[2]], points[tet[3]])
            test_points.append(point)

            if(point[1] > bbox[1,1]): # in the sky
                occs.append(0)
            elif((point < bbox[0,:]).any() or (point > bbox[1,:]).any()): # behind wall or floor
                occs.append(1)
            else:
                occs.append(-1) # check with check_mesh_contains()

    print("Loop Time (s): ", datetime.datetime.now() - start)

    occs = np.array(occs)
    test_points = np.array(test_points)

    print("To Array (s): ", datetime.datetime.now() - start)

    occs[occs == -1] = check_mesh_contains(mesh,test_points[occs == -1])

    print("Check Mesh Contains (s): ", datetime.datetime.now() - start)


    if export_points:
        colors = np.zeros(shape=(occs.shape[0],4))
        colors[occs==0] = np.array([1,0,0,1])
        colors[occs==1] = np.array([0,0,1,1])

        pc = trimesh.PointCloud(test_points,colors=colors)
        pc.export(os.path.join(path,model,"points.ply"))


    # test_points = np.reshape(test_points,(dt["tetrahedra"].shape[0],n_points,3))
    occs = np.reshape(occs,(dt["tetrahedra"].shape[0],n_points))
    occs = np.mean(occs,axis=1)


    np.savez(os.path.join(path,model,"dgnn",model+"_labels.npz"),
             infinite=dt["inf_tetrahedra"],
             inside_perc=occs,
             outside_perc=1-occs)


    print("Labelling Time (s): ",datetime.datetime.now() - start)




if __name__ == "__main__":

    path = "/home/adminlocal/PhD/data/synthetic_room"
    # files = os.listdir(path)
    files = ["00000003"]
    for f in files:
        try:
            gt(path,f)
        except Exception as e:
            print(e)