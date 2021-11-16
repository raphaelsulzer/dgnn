import open3d as o3d
import numpy as np


# data = np.load("/home/adminlocal/PhD/data/synthetic_room_dataset_sample/pointclouds/rooms_08/00000044/pointcloud/pointcloud_01.npz")
# # points = np.load("/home/adminlocal/PhD/data/ShapeNet/import/pointcloud_00.npz")
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(data["points"])
# pcd.normals = o3d.utility.Vector3dVector(data["normals"])
# # s = data["sensors"][:3000,:]
# # p = data["points"][:3000,:3]
# # s = s - p
# # s = s / np.linalg.norm(s, axis=1)[:, np.newaxis]
# # pcd.normals = o3d.utility.Vector3dVector(s)
#
# o3d.io.write_point_cloud("/home/adminlocal/PhD/data/synthetic_room_dataset_sample/pointclouds/rooms_08/00000044/pointcloud/pointcloud_01.ply", pcd)
# # o3d.io.write_point_cloud("/home/adminlocal/PhD/data/ShapeNet/import/pointcloud_00.ply", pcd)


data = np.load("/home/adminlocal/PhD/data/synthetic_room_dataset_sample/pointclouds/rooms_08/00000044/pointcloud/pointcloud_01.npz")
# points = np.load("/home/adminlocal/PhD/data/ShapeNet/import/pointcloud_00.npz")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data["points"])
pcd.normals = o3d.utility.Vector3dVector(data["normals"])
# s = data["sensors"][:3000,:]
# p = data["points"][:3000,:3]
# s = s - p
# s = s / np.linalg.norm(s, axis=1)[:, np.newaxis]
# pcd.normals = o3d.utility.Vector3dVector(s)

o3d.io.write_point_cloud("/home/adminlocal/PhD/data/synthetic_room_dataset_sample/pointclouds/rooms_08/00000044/pointcloud/pointcloud_01.ply", pcd)
# o3d.io.write_point_cloud("/home/adminlocal/PhD/data/ShapeNet/import/pointcloud_00.ply", pcd)