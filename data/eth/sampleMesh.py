import argparse
import subprocess
import sys, os

def sample(args):
    print("\n##################################################")
    print("############ Sample {} {} ############".format(args.scene, args.method))
    print("##################################################")

    if(args.method == 'omvs' or args.method == 'omvs_cleaned'):
        working_dir = os.path.join(args.data_dir,args.scene,'openMVS',"")
        input_file = "mesh"
        if (args.free_space_support):
            input_file += "_Jancosek"
        else:
            input_file += "_Labatu"
    elif(args.method == 'clf' or args.method == 'clf_cleaned'):
        working_dir = os.path.join(args.data_dir,args.scene,"")
        sm = "_"+args.sure_method.split(',')[0]+"_"
        input_file = args.scene+ sm + args.reg_weight
    elif(args.method == 'poisson' or args.method == 'poisson_cleaned'):
        working_dir = os.path.join(args.data_dir,args.scene, 'poisson',"")
        input_file = args.scene + "_" + str(args.bType)
    else:
        print("{} is not a valid method. choose either omvs or clf.".format(args.method))
        sys.exit(1)


    if(args.clean_mesh):
        input_file+="_cleaned"
        if (args.refine):
            input_file += "_refined"
    else:
        if(args.method == 'omvs'):
            if(args.refine):
                input_file += "_refined"
            else:
                input_file+= "_initial" # uncleaned clf is always directly sampled in sure, so this will not be used with clf




    print ("Using input file  : ", input_file)
    print ("      output file : ", input_file+"_sampled")



    command = [args.user_dir+args.sure_dir+"/sample",
    "-w", working_dir,
    "-i", input_file+".ply",
    "-o", input_file,
    "--output_sampling", args.output_sampling,
    "--omanifold", "1"
    ]

    print("run command: ", command)
    p = subprocess.Popen(command, stdout=subprocess.PIPE)

    for line in iter(p.stdout.readline, b''):
        print(line.decode("utf-8")[:-1])

    if(p.returncode):
        sys.exit(1)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample reconstructed mesh')


    parser.add_argument('--user_dir', type=str, default="/home/raphael/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="/mnt/raphael/ETH3D/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scene', type=str, default="pipes",
                        help='on which scene to execute pipeline.')

    parser.add_argument('--sure_dir', type=str, default="cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure release dir starting from the user_dir')

    parser.add_argument('--output_sampling', type=str, default="as-100",
                        help='number of points to sample from the mesh. default: 1000000')


    parser.add_argument('--free_space_support', type=int, default=1,
                        help='free space suppport, 0 = off (Labatu), 1 = on (Jancosek)')
    parser.add_argument('--clean_mesh', type=int, default=1,
                        help='enable/disable all mesh clean options. default: disabled.')
    parser.add_argument('--refine', type=int, default=0,
                        help='refine mesh. default: disabled.')
    parser.add_argument('-m','--sure_method', type=str, default="cl,sm",
                        help='the m, method parameter for sure. default: cl,sm')

    args = parser.parse_args()

    args.method = "clf_cleaned"
    args.reg_weight = "0.5"

    sample(args)