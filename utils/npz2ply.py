import open3d as o3d
import numpy as np
import os
# data = np.load("/home/adminlocal/PhD/data/ShapeNet/meshes/02691156/d18f2aeae4146464bd46d022fd7d80aa/eval/points.npz")
# # points = np.load("/home/adminlocal/PhD/data/ShapeNet/import/pointcloud_00.npz")
# pcd = o3d.geometry.PointCloud()
# # pcd.points = o3d.utility.Vector3dVector(data["points"].astype(np.float16)[:,[0,2,1]])
# pcd.points = o3d.utility.Vector3dVector(data["points"].astype(np.float16))
#
# occ = np.unpackbits(data['occupancies'])
#
# for i, o in enumerate(occ):
#     if(not o):
#         pcd.points[i] = [0,0,0]


# cols = []
# for o in occ:
#     if(o):
#         cols.append([255, 0, 0])
#     else:
#         cols.append([0, 0, 255])
#
# cols = np.array(cols)
#
# pcd.colors = o3d.utility.Vector3dVector(cols)

# pcd.normals = o3d.utility.Vector3dVector(data["normals"])
# s = data["sensors"][:3000,:]
# p = data["points"][:3000,:3]
# s = s - p
# s = s / np.linalg.norm(s, axis=1)[:, np.newaxis]
# pcd.normals = o3d.utility.Vector3dVector(s)

# o3d.io.write_point_cloud("/home/adminlocal/PhD/data/ShapeNet/meshes/02691156/d18f2aeae4146464bd46d022fd7d80aa/eval/points.ply", pcd)
# o3d.io.write_point_cloud("/home/adminlocal/PhD/data/ShapeNet/import/pointcloud_00.ply", pcd)


#### MESH

b="/home/adminlocal/PhD/data/ShapeNet/meshes/02691156/422700fbef58a0ee1fd12d3807585791/"
#
file = os.path.join(b, "mesh", "mesh.off")
mesh = o3d.io.read_triangle_mesh(file)
R = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, np.pi))
meshr = mesh.rotate(R)
o3d.io.write_triangle_mesh(file[:-4]+'1.off', mesh)

# pointcloud
file = os.path.join(b, "scan", "4.npz")
data = np.load(file)

points = np.matmul(data["points"], R)
normals = np.matmul(data["normals"], R)
gt_normals = np.matmul(data["gt_normals"], R)
sensor_position = np.matmul(data["sensor_position"], R)

np.savez(file, points=points, normals=normals, gt_normals=gt_normals, sensor_position=sensor_position,
         cameras=data['cameras'], noise=data['noise'], outliers=data['outliers'])


omesh = o3d.io.read_triangle_mesh("/home/adminlocal/PhD/data/ModelNet/conv_onet/scan/mesh.off")


R = omesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, np.pi))

omesh = omesh.rotate(R)

o3d.io.write_triangle_mesh("/home/adminlocal/PhD/data/ModelNet/conv_onet/scan/mesh_r.off",omesh)


# iou points
file = os.path.join(b, "eval", "pointcloud.npz")
data = np.load(file)
points = np.matmul(data["points"], R)
normals = np.matmul(data["normals"], R)

np.savez(file, points=points, normals=normals, loc=data['loc'], scale=data['scale'])

file = os.path.join(b, 'eval', 'points.npz')
data = np.load(file)

points = np.matmul(data["points"], R)
np.savez(file, points=points, occupancies=data['occupancies'], loc=data['loc'], scale=data['scale'])