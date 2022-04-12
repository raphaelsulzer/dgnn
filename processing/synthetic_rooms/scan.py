import trimesh
import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..', 'utils'))
from libmesh import check_mesh_contains
import datetime
import copy
# import open3d as o3d

scan_conf = 9

export_sensors = 0
export_ply = 1
export_npz = 1

def getSettings(scan_conf):

    if scan_conf == 9:
        n_points = 1000+int(np.abs(np.random.normal(15000,3000)))  # we want 3*sigma to be 30000 (so factor should be 10000 but made it a bit lower)
        n_cameras = 2 + int(np.abs(np.random.randn()) * 6)  # we want 3*sigma to be 20 (so factor should be 6.66 but made it a bit lower), and at least 2 cameras
        n_noise = np.abs(np.random.randn()) * 0.003  # we want 3*sigma to be 0.03 (so factor should be 0.01)
        n_outliers = np.abs(np.random.randn()) * 0.01  # we want 3*sigma to be 0.15 (so factor should be 0.1)
    else:
        n_points = 15000
        n_cameras = 5
        n_noise = 0.0
        n_outliers = 0.01

    print("Config:")
    print("points = "+str(n_points))
    print("cameras = "+str(n_cameras))
    print("noise = "+str(n_noise))
    print("outliers = "+str(n_outliers))

    return {"n_points":n_points,"n_cameras":n_cameras,"n_noise":n_noise,"n_outliers":n_outliers}


def addOutliers(mesh, points, settings):

    bbox = copy.deepcopy(mesh.bounds)

    bbox[0,1]*=2
    bbox[1,1]*=-4

    outliers = np.random.uniform(low=bbox[0,:], high=bbox[1,:],size=(int(settings["n_points"]*settings["n_outliers"]),3))
    points[-int(settings["n_points"] * settings["n_outliers"]):, :] = outliers

    return points


def addNoise(points,settings):

    return points+np.random.normal(0.0,settings["n_noise"],(points.shape[0],3))



def scan(path,model):

    start = datetime.datetime.now()

    # load the mesh
    mesh = trimesh.load_mesh(os.path.join(path,model,"mesh.off"),process=False)
    settings = getSettings(scan_conf)

    # get the bounds of the mesh
    bbox = mesh.bounds


    cams = np.random.uniform(low=bbox[0,:]*0.7, high=bbox[1,:]*0.7,size=(100,3))
    contains = check_mesh_contains(mesh,cams)

    cams = cams[np.invert(contains),:]
    cams = cams[:settings["n_cameras"],:]

    if(export_sensors):
        colors = np.zeros(shape=(cams.shape[0],4))+[255,255,0,1]
        pc = trimesh.PointCloud(cams,colors=colors)
        pc.export(os.path.join(path,model,"sensors.ply"))


    # scanning
    aims = np.random.rand(settings['n_points']*settings["n_cameras"],3)-0.5
    cams = np.tile(cams,(settings['n_points'],1))

    intersecter = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    points, index_ray, index_tri  = intersecter.intersects_location(cams,aims,multiple_hits=False)
    points = points[:settings["n_points"],:]

    cams = cams[index_ray][:settings["n_points"],:]

    gt_normals = mesh.face_normals[index_tri][:settings["n_points"],:]

    # add noise
    points = addNoise(points,settings)

    # add outliers
    points = addOutliers(mesh, points,settings)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.estimate_normals()
    # normals = np.array(pcd.normals)
    #
    # # export PLY
    # if(export_ply):
    #     # pcd.normals = o3d.utility.Vector3dVector(cams-points)
    #     o3d.io.write_point_cloud(os.path.join(path,model,"scan.ply"), pcd)


    # export NPZ
    if(export_npz):
        np.savez(os.path.join(path,model,"scan","9.npz"),
                 points=points,
                 normals=gt_normals,
                 gt_normals=gt_normals,
                 sensor_position=cams,
                 n_cameras=np.array(settings["n_cameras"],dtype=np.float64),
                 std_noise=np.array(settings["n_noise"],dtype=np.float64),
                 p_outliers=np.array(settings["n_outliers"],dtype=np.float64))



    print("Scanning Time (s): ",datetime.datetime.now() - start)


    a = 5


if __name__ == "__main__":

    path = "/home/adminlocal/PhD/data/synthetic_room"
    files = os.listdir(path)
    # files = ["00000007"]
    for f in files:
        print('\n',f)
        try:
            scan(path,f)
        except Exception as e:
            print(e)



