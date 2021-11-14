import os, argparse, shutil
import numpy as np
import open3d as o3d
import trimesh
import subprocess

def rescale(args):

    """For now this function is simply used to rescale the ConvOnet output from unit cube to
    75mm cube as used in Berger et al. benchmark"""

    args.scene_conf = args.scene + "_" + str(args.conf)

    # gt_file = os.path.join(args.user_dir,args.data_dir,"ground_truth_surface",args.scene+".off")
    # print("ground truth file ", gt_file)
    # gt_mesh = trimesh.load(gt_file,'off',process=False)

    recon_file = os.path.join(args.user_dir,args.data_dir,"reconstructions","igr",args.scene_conf+".ply")
    print("reconstruction file ", recon_file)
    recon_mesh = trimesh.load(recon_file,'ply',process=False)

    # get the centroid
    pc_file = os.path.join(args.user_dir,args.data_dir,"scans","with_normals",args.scene_conf+".ply")
    pc = o3d.io.read_point_cloud(pc_file)
    centroid=pc.get_center()

    recon_mesh.vertices*=75
    recon_mesh.vertices+=centroid

    recon_mesh.export(os.path.join(args.user_dir,args.data_dir,"reconstructions","igr",args.scene_conf+".off"))

def scale(args):
    args.scene_conf = args.scene + "_" + str(args.conf)

    in_file = os.path.join(args.user_dir,args.data_dir,"scans","with_normals",args.scene_conf+".ply")
    print("convert file ", in_file)

    pc = o3d.io.read_point_cloud(in_file)
    pc.translate(-pc.get_center())
    pc.scale(scale=1 / 75, center=False)

    out_path = os.path.join(args.user_dir,args.data_dir,"scans","with_normals_scaled",args.scene_conf)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # TODO:
    # save as ply


def npz(args):

    """This function is simply used to convert from ply files to ConvOnet or IGR input"""

    args.scene_conf = args.scene + "_" + str(args.conf)

    in_file = os.path.join(args.user_dir,args.data_dir,"scans","with_normals_scaled",args.scene_conf+".ply")
    print("convert file ", in_file)

    pc = o3d.io.read_point_cloud(in_file)

    out_path = os.path.join(args.user_dir,args.data_dir,"scans","igr")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path,args.scene_conf+".npz")


    # o3d.io.write_point_cloud(os.path.join(args.user_dir,args.data_dir,"scans","with_normals_scaled",args.scene_conf+".ply"), pc)

    # input for IGR
    np.savez_compressed(out_file, \
                            np.concatenate((np.array(pc.points),np.array(pc.normals)),1))

    # test if it worked and load the file again to inspect that the correct fields are there
    data = np.load(out_file)
    a=5



def upload(args):

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


def download(args):

    infile = "igr_"+args.checkpoint+"_"+args.scene+".ply"
    infile = os.path.join("enpc:/home/raphael/remote_python/IGR/exps",args.scene,str(args.conf),"plots",infile)
    outfile = args.scene+"_"+str(args.conf)+".ply"
    outfile = os.path.join("/home/adminlocal/PhD/data/reconbench/reconstructions/igr",outfile)

    command = ["scp", infile, outfile]

    print("Run command: ", command)
    p = subprocess.Popen(command)
    p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make onet input data')

    parser.add_argument('--mode', type=str, default="npz",
                        help="'+'-seperated mode choice: scale, rescale, npz, upload")

    parser.add_argument('--user_dir', type=str, default="/home/adminlocal/PhD/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="data/reconbench/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scenes', nargs='+', type=str, default=["anchor"],
                        help='on which scene to execute pipeline.')
    parser.add_argument('-c', '--confs', nargs='+', type=int, default=[0],
                        help='which config file to load')

    parser.add_argument('--checkpoint', type=str, default="30000",
                        help='Whether to overwrite output.')
    parser.add_argument('--overwrite', action='store_true', default=True,
                        help='Whether to overwrite output.')

    args = parser.parse_args()

    args.mode = args.mode.split('+')

    if(args.confs[0] == -1):
        args.confs = [0,1,2,3,4]

    if(args.scenes[0] == "all"):
        args.scenes = ["anchor", "gargoyle", "dc", "daratech", "lordquas"]

    for s in args.scenes:
        args.scene = s
        for c in args.confs:
            args.conf = c
            if('scale' in args.mode):
                scale(args)
            if('npz' in args.mode):
                npz(args)
            if('upload' in args.mode):
                upload(args)
            if('rescale' in args.mode):
                rescale(args)
            if('download' in args.mode):
                download(args)

