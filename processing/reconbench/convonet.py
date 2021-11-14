import os, argparse, shutil
import numpy as np
import open3d as o3d
import trimesh

def scale(args):

    """For now this function is simply used to rescale the ConvOnet output from unit cube to
    75mm cube as used in Berger et al. benchmark"""

    args.scene_conf = args.scene + "_" + str(args.conf)

    gt_file = os.path.join(args.user_dir,args.data_dir,"ground_truth_surface",args.scene+".off")
    print("ground truth file ", gt_file)
    gt_mesh = trimesh.load(gt_file,'off',process=False)

    recon_file = os.path.join(args.user_dir,args.data_dir,"reconstructions","occ",args.scene_conf+".off")
    print("reconstruction file ", recon_file)
    recon_mesh = trimesh.load(recon_file,'off',process=False)

    # get the centroid
    pc_file = os.path.join(args.user_dir,args.data_dir,"scans","with_normals",args.scene_conf+".ply")
    pc = o3d.io.read_point_cloud(pc_file)
    centroid=pc.get_center()

    recon_mesh.vertices*=75
    recon_mesh.vertices+=centroid

    recon_mesh.export(os.path.join(args.user_dir,args.data_dir,"reconstructions","occ","75",args.scene_conf+".off"))



def npz(args):

    """This function is simply used to convert from ply files to ConvOnet or IGR input"""

    args.scene_conf = args.scene + "_" + str(args.conf)

    in_file = os.path.join(args.user_dir,args.data_dir,"scans","with_normals",args.scene_conf+".ply")
    print("convert file ", in_file)

    pc = o3d.io.read_point_cloud(in_file)
    pc.translate(-pc.get_center())
    pc.scale(scale=1 / 75, center=False)

    out_path = os.path.join(args.user_dir,args.data_dir,"scans","occ",args.scene_conf)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path,"pointcloud.npz")

    np.savez_compressed(out_file, \
                            points=np.array(pc.points), \
                            normals=np.array(pc.normals))

    o3d.io.write_point_cloud(os.path.join(args.user_dir,args.data_dir,"scans","with_normals_scaled",args.scene_conf+".ply"), pc)

    ## input for IGR
    # np.savez_compressed(file.split('.')[0]+".npz", \
    #                         np.concatenate((np.array(points.points),np.array(points.normals)),1))

    ## test if it worked and load the file again to inspect that the correct fields are there
    # data = np.load(in_file.split('.')[0]+".npz")



def copy(args):

    """This is for copying the scene.npz files to /scene/pointcloud.npz"""

    input_dir = args.user_dir + args.data_dir + "reconbench/occ/pointclouds/"
    output_dir = args.user_dir + args.data_dir + "occ/reconbench/"

    # onlyfiles = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(working_dir, f))]

    for i in os.listdir(input_dir):
        pointcloud_file=os.path.join(input_dir, i)
        # print(pointcloud_file)
        # if path doesn't exist
        if not os.path.exists(os.path.join(output_dir, i[:-4])):
            os.makedirs(os.path.join(output_dir, i[:-4]))
        # if file isn't already copied
        copy_to = os.path.join(output_dir,i[:-4],'pointcloud.npz')
        if not os.path.isfile(copy_to) or args.overwrite:
            print("copy from {} to {}".format(pointcloud_file,copy_to))
            shutil.copyfile(pointcloud_file,copy_to)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make onet input data')


    parser.add_argument('--user_dir', type=str, default="/home/raphael/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="data/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scenes', nargs='+', type=str, default=["Ship"],
                        help='on which scene to execute pipeline.')
    parser.add_argument('-c', '--confs', nargs='+', type=int, default=[0],
                        help='which config file to load')

    parser.add_argument('-i', '--input_file_extension', type=str, default="",
                        help='the mesh file to be evaluated')

    parser.add_argument('--reconbench_dir', type=str, default="cpp/reconbench-CMake/build",
                        help='Indicate the openMVS binary directory, pointing to .../bin folder starting from user_dir')

    parser.add_argument('--sure_dir', type=str, default="cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure build directory, pointing to .../build/release folder starting from user_dir')
    parser.add_argument('--gco', type=str, default="area-1.0",
                        help='graph cut optimization type,weight. default: area,1.0')

    # additional Mesh reconstruction options:
    parser.add_argument('-p', '--steps', type=str, default='e',
                        help='pipeline steps. default: idmr. extra options: sampling = s, evaluation = e')

    parser.add_argument('--overwrite', action='store_true', default=True,
                        help='Whether to overwrite output.')

    args = parser.parse_args()

    scale(args)

    npz(args)

    copy(args)

