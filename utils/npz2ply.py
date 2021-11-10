import open3d as o3d
import numpy as np


# TODO: check if this file really has the same noise as when i scan with 0.05 std of gaussian noise
data = np.load("/home/adminlocal/PhD/data/ModelNet/monitor_0130.npz")
# points = np.load("/home/adminlocal/PhD/data/ShapeNet/import/pointcloud_00.npz")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data["points"][:3000,:3])
pcd.normals = o3d.utility.Vector3dVector(data["normals"][:3000,:])
# s = data["sensors"][:3000,:]
# p = data["points"][:3000,:3]
# s = s - p
# s = s / np.linalg.norm(s, axis=1)[:, np.newaxis]
# pcd.normals = o3d.utility.Vector3dVector(s)

o3d.io.write_point_cloud("/home/adminlocal/PhD/data/ModelNet/42.ply", pcd)
# o3d.io.write_point_cloud("/home/adminlocal/PhD/data/ShapeNet/import/pointcloud_00.ply", pcd)