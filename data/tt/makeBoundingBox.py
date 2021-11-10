import open3d as o3d
import argparse
import os, sys
import subprocess

def box(args):

    print("\n##################################################")
    print("############ Make bounding box {} ############".format(args.scene))
    print("##################################################")


    command = [args.sure_dir+"boundingBox",
        "-w", os.path.join(args.data_dir,args.scene),
        "-i", args.scene
        ]

    print("Execute: ",command)

    p = subprocess.Popen( command )
    p.wait()

    if(p.returncode):
        print("\nEXITING WITH SURE ERROR\n")
        sys.exit(1)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='crop and save poisson as off')

    parser.add_argument('-d', '--data_dir', type=str, default="/mnt/raphael/TanksAndTemples/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scenes', nargs = '+', type=str, default=["Truck"],
                        help='on which scene to execute pipeline.')
    parser.add_argument('--sure_dir', type=str, default="/home/raphael/cpp/surfaceReconstruction/build/no_omvs/")

    args = parser.parse_args()

    if(args.scenes[0] == 'all'):
        args.scenes = []
        # for scene in os.listdir(args.user_dir+args.data_dir):
        for scene in os.listdir(args.data_dir):
            if (not scene[0] == "x"):
                args.scenes+=[scene]


    for i,scene in enumerate(args.scenes):
        args.scene=scene
        box(args)
