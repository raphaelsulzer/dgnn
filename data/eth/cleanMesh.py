import argparse
import subprocess
import sys, os

def clean(args):
    print("\n##########################################################")
    print("############ 3b. Mesh cleaning {} ############".format(args.scene))
    print("##########################################################")

    working_dir = os.path.join(args.data_dir,args.scene)

    input_file = os.path.join("openMVS","densify_file.mvs")
    mesh_file = os.path.join("poisson",args.scene+"_" + str(args.bType)+".ply")

    output_file = os.path.join("poisson",args.scene+"_" + str(args.bType)+"_cleaned.ply")

    print ("Using input dir  : ", input_file)
    print ("      output_dir : ", output_file)

    command = [args.user_dir+args.openMVS_dir+"/ReconstructMesh",
    "-w", working_dir,
    "-i", input_file,
    "-o", output_file,
    "--mesh-file", mesh_file,
    "--decimate", "1",
    "--remove-spurious", "20",
    "--remove-spikes", "1",
    "--close-holes", "30",
    "--smooth", "2"]


    p = subprocess.Popen( command)
    p.wait()


    if(p.returncode):
        sys.exit(1)


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
    parser.add_argument('-mf', '--mesh_file', type=str, default="",
                        help='the mesh to be cleaned')

    parser.add_argument('--openMVS_dir', type=str, default="cpp/openMVS_release/bin",
                        help='Indicate the openMVS binary directory, pointing to .../bin folder starting from user_dir')

    # additional Mesh reconstruction options:
    parser.add_argument('--min_point_distance', type=float, default=0.0,
                        help='minimum distance in pixels between the projection'
                             ' of two 3D points to consider them different while triangulating (0 -disabled)')
    parser.add_argument('--clean_mesh', type=int, default=1,
                        help='enable/disable all mesh clean options. default: disabled.')

    args = parser.parse_args()
    clean(args)