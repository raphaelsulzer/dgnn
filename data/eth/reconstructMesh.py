import argparse
import subprocess
import sys


def reconstruct(args):
    title=args.scene
    if(args.free_space_support):
        title+="_Jancosek"
    else:
        title+="_Labatu"
    if(args.clean_mesh):
        title+="_cleaned"
    else:
        title+="_initial"
    print("\n##########################################################")
    print("############ 3. omvs {} reconstruction  ############".format(title))
    print("##########################################################")

    working_dir = args.data_dir + args.scene + "/openMVS/"
    input_file = args.input_file
    output_file = "mesh"
    if(args.free_space_support):
        output_file+="_Jancosek"
    else:
        output_file+="_Labatu"
    if(args.clean_mesh):
        output_file+="_cleaned"
    else:
        output_file+= "_initial"
    output_file+=".mvs"

    print ("Using input dir  : ", input_file)
    print ("      output_dir : ", output_file)

    command = [args.user_dir+args.openMVS_dir+"/ReconstructMesh",
    "-w", working_dir,
    "-i", input_file,
    "-o", output_file,
    "--export-type", "ply",     # only other option is obj, not off
    "--free-space-support", str(args.free_space_support),
    "--min-point-distance", str(args.min_point_distance)]
    if(not args.clean_mesh):    # else the standard settings are used for cleaning
        command+=[
        "--decimate", "1",
        "--remove-spurious", "0",
        "--remove-spikes", "0",
        "--close-holes", "0",
        "--smooth", "0"
        ]

    print("you called: ", command)
    p = subprocess.Popen( command)
    p.wait()


    if(p.returncode):
        sys.exit(1)

    return output_file[:-4]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='openMVG_openMVS_reconstruction')

    parser.add_argument('-m','--machine', type=str, default="ign-laptop",
                        help='choose the machine, ign-laptop, cnes or enpc.')

    parser.add_argument('--user_dir', type=str, default="/home/adminlocal/PhD/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="data/ETH3D/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scene', type=str, default="pipes",
                        help='on which scene to execute pipeline.')

    parser.add_argument('--openMVS_dir', type=str, default="cpp/openMVS_release/bin",
                        help='Indicate the openMVS binary directory, pointing to .../bin folder starting from user_dir')

    # additional Mesh reconstruction options:
    parser.add_argument('--min_point_distance', type=float, default=0.0,
                        help='minimum distance in pixels between the projection'
                             ' of two 3D points to consider them different while triangulating (0 -disabled)')
    parser.add_argument('--free_space_support', type=int, default=1,
                        help='free space suppport, 0 = off (Labatu), 1 = on (Jancosek)')
    parser.add_argument('--clean_mesh', type=int, default=0,
                        help='enable/disable all mesh clean options. default: disabled.')


    args = parser.parse_args()
    reconstruct(args)