import argparse
import subprocess
import sys, os

def texture(args):
    print("\n##########################################################")
    print("############ Texture mesh {} ############".format(args.scene))
    print("##########################################################")

    working_dir = args.data_dir + args.scene + "/openMVS/"
    input_file = "densify_file.mvs"

    if(args.method == 'omvs' or args.method == 'omvs_clean'):
        mesh_file="mesh"
        if(args.free_space_support):
            mesh_file+="_Jancosek"
        else:
            mesh_file+="_Labatu"
        output_file = mesh_file + "_textured"
        if(args.clean_mesh):
            mesh_file+="_cleaned"
        else:
            mesh_file+= "_initial"
    elif(args.method == 'clf' or args.method == 'clf_cleaned'):
        sm = "_"+args.sure_method.split(',')[0]+"_"
        mesh_file="../"+args.scene+sm+args.reg_weight
        output_file = args.scene+ sm + "_textured"
        if(args.clean_mesh):
            mesh_file+="_cleaned"
        else:
            mesh_file+= "_initial"
    elif(args.method == 'poisson' or args.method == 'poisson_cleaned'):
        mesh_file="../poisson/"+args.scene
        output_file = args.scene + "_textured"
        if(args.clean_mesh):
            mesh_file+="_cleaned"
    else:
        print("{} is not a valid method. choose either omvs or clf.".format(args.method))
        sys.exit(1)

    mesh_file+=".ply"


    print ("Using input dir  : ", input_file)
    print ("      output_dir : ", output_file)

    command = [args.user_dir+args.openMVS_dir+"/TextureMesh",
    "-w", working_dir,
    "-i", input_file,
    "-o", output_file,
    "--mesh-file", mesh_file,
               "--resolution-level", str(args.resolution_level)
               ]


    p = subprocess.Popen( command)
    p.wait()


    if(p.returncode):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='openMVG_openMVS_reconstruction')

    parser.add_argument('--user_dir', type=str, default="/home/docker/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="data/ETH3D/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scenes', type=str, nargs='+', default=["pipes"],
                        help='on which scene to execute pipeline.')

    # additional SURE reconstruction options:
    parser.add_argument('-m','--method', type=str, default="clf",
                        help='omvs or clf')

    parser.add_argument('--openMVS_dir', type=str, default="cpp/openMVS_release/bin",
                        help='Indicate the openMVS binary directory, pointing to .../bin folder starting from user_dir')

    # additional input options:
    parser.add_argument('--free_space_support', type=int, default=1,
                        help='free space suppport, 0 = off (Labatu), 1 = on (Jancosek)')
    parser.add_argument('--clean_mesh', type=int, default=1,
                        help='enable/disable all mesh clean options. default: disabled.')
    parser.add_argument('--resolution_level', type=int, default=4,
                        help='how many times to scale down the images before point cloud computation')

    args = parser.parse_args()

    if(args.scenes[0] == 'all'):
        args.scenes = []
        for scene in os.listdir(args.user_dir+args.data_dir):
            if (not scene[0] == "x"):
                args.scenes+=[scene]

    for i,scene in enumerate(args.scenes):
        args.scene = scene
        print("\n#####################################################################")
        print("############ Texture scene: {} ({}/{})############".format(args.scene, i+1, len(args.scenes)))
        print("#####################################################################")
        texture(args)
