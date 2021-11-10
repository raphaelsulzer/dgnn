import open3d as o3d
import argparse
import os, sys
import subprocess

def normal(args):

    print("\n##################################################")
    print("############ Downsample {} ############".format(args.scene))
    print("##################################################")

    pc_file = os.path.join(args.data_dir,args.scene,args.scene+".ply")
    pc = o3d.io.read_point_cloud(pc_file)
    n = len(pc.points)
    pc=pc.voxel_down_sample(0.04)

    print("Downsampled point cloud from {} to {} points".format(n,len(pc.points)))

    write_file = os.path.join(args.data_dir,args.scene,args.scene+"_downsampled.ply")
    o3d.io.write_point_cloud(write_file,pc)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='crop and save poisson as off')

    parser.add_argument('-d', '--data_dir', type=str, default="/mnt/raphael/TanksAndTemples/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scenes', nargs = '+', type=str, default=["Truck"],
                        help='on which scene to execute pipeline.')

    args = parser.parse_args()

    if(args.scenes[0] == 'all'):
        args.scenes = []
        # for scene in os.listdir(args.user_dir+args.data_dir):
        for scene in os.listdir(args.data_dir):
            if (not scene[0] == "x"):
                args.scenes+=[scene]


    for i,scene in enumerate(args.scenes):
        args.scene=scene
        normal(args)
