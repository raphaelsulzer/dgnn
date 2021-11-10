import argparse
import subprocess
import sys, os

def texture(args):
    print("\n##########################################################")
    print("############ Download mesh {} ############".format(args.scene))
    print("##########################################################")

    working_dir = args.user_dir + args.data_dir + args.scene + "/openMVS/"
    output_user_dir = args.output_user_dir + args.data_dir + args.scene + "/import/"

    if not os.path.exists(output_user_dir):
        os.makedirs(output_user_dir)

    if(args.method == 'omvs' or args.method == 'omvs_clean'):
        mesh_file="mesh"
        if(args.free_space_support):
            mesh_file+="_Jancosek"
        else:
            mesh_file+="_Labatu"
        if(args.textured):
            mesh_file = mesh_file + "_textured"
        else:
            if(args.clean_mesh):
                mesh_file = mesh_file + "_cleaned"
            else:
                mesh_file = mesh_file + "_initial"
    elif(args.method == 'clf' or args.method == 'clf_cleaned'):
        if(args.textured):
            mesh_file = args.scene+"_cl_05_textured"
        else:
            if(args.clean_mesh):
                mesh_file = "../"+args.scene+"_cl_0.5_cleaned"
            else:
                mesh_file = "../"+args.scene+"_cl_0.5_mesh"
    elif(args.method == "point_cloud"):
        mesh_file = "densify_file"
    elif(args.method == 'poisson'):
        mesh_file = "../poisson/"+args.scene
        if (args.clean_mesh):
            mesh_file = mesh_file + "_cleaned"
    else:
        print("{} is not a valid method. choose either omvs or clf.".format(args.method))
        sys.exit(1)

    if(args.ecc):
        files=mesh_file+"_sampled_*"
    else:
        files=mesh_file+".p*"
    # mesh_file+=".ply"

    # print ("Using input dir  : ", png_file)
    # print ("      output_dir : ", output_file)

    if(args.overwrite):
        command = ["scp",
                   "enpc:" + working_dir + files,
                   output_user_dir]
    else:
        command = ["rsync",
        "enpc:"+working_dir+files,
        output_user_dir]

    print("you called: ", command)

    p = subprocess.Popen( command)
    p.wait()


    if(p.returncode):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='openMVG_openMVS_reconstruction')

    # "/Users/Raphael/Library/Mobile\ Documents/com\~apple\~CloudDocs/Studium/PhD/Paris/"

    parser.add_argument('--user_dir', type=str, default="/mnt/raphael/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="ETH3D/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scenes', type=str, nargs='+', default=["kicker"],
                        help='on which scene to execute pipeline.')

    # additional SURE reconstruction options:
    parser.add_argument('-m','--method', type=str, choices=["clf","omvs","poisson"], default="clf",
                        help='point_cloud, omvs or clf')

    parser.add_argument('--output_user_dir', type=str, default="/home/adminlocal/PhD/data/",
                        help='Indicate the output_user_dir. default: /home/adminlocal/PhD/')

    # additional input options:
    parser.add_argument('--free_space_support', type=int, default=1,
                        help='free space suppport, 0 = off (Labatu), 1 = on (Jancosek)')
    parser.add_argument('--clean_mesh', type=int, default=1,
                        help='enable/disable all mesh clean options. default: disabled.')
    parser.add_argument('--textured', type=int, default=1,
                        help='enable/disable all mesh clean options. default: disabled.')
    parser.add_argument('--ecc', type=int, default=0,
                        help='get the colored completeness and accuracy clouds')
    parser.add_argument('--resolution_level', type=int, default=4,
                        help='how many times to scale down the images before point cloud computation')
    parser.add_argument('--overwrite', type=int, default=0,
                        help='rsync or scp')


    args = parser.parse_args()

    if(args.scenes[0] == 'all'):
        args.scenes = []
        for scene in os.listdir(args.output_user_dir+args.data_dir):
            if (not scene[0] == "x" and not scene[0] == "."):
                args.scenes+=[scene]

    for i,scene in enumerate(args.scenes):
        args.scene = scene
        print("\n#####################################################################")
        print("############ Texture scene: {} ({}/{})############".format(args.scene, i+1, len(args.scenes)))
        print("#####################################################################")
        texture(args)
