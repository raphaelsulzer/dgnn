import argparse
import subprocess
import sys


def refine(args):
    print("\n######################################################")
    print("############ 3. Mesh refinement {} ############".format(args.scene))
    print("######################################################")

    working_dir = args.user_dir + args.data_dir + args.scene + "/openMVS/"
    # input_file = "mesh_Labatu_initial.mvs"
    # output_file = "mesh_file_Labatu_refined.ply"



    if(args.method == 'omvs' or args.method == 'omvs_clean'):
        mesh_file="mesh"
        if(args.free_space_support):
            mesh_file+="_Jancosek"
        else:
            mesh_file+="_Labatu"
        output_file = mesh_file
        if(args.clean_mesh):
            mesh_file+="_cleaned"
        else:
            mesh_file+= "_initial"
    else:
        print("{} is not a valid method. choose either omvs or clf.".format(args.method))
        sys.exit(1)

    input_file = mesh_file + ".mvs"
    if (args.clean_mesh):
        output_file+= "_cleaned_refined"
    else:
        output_file+= "_refined"

    print ("Using input dir  : ", input_file)
    print ("      output_dir : ", output_file)

    p = subprocess.Popen( [args.user_dir+args.openMVS_dir+"/RefineMesh",
    "-w", working_dir,
    "-i", input_file,
    "-o", output_file,
   "--resolution-level", str(args.resolution_level),
   "--max-views", "2"
    ] )
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
                        help='Indicate the openMVS binary directory, pointing to .../bin folder starting from user dir')

    args = parser.parse_args()
    refine(args)