import argparse
import subprocess
import sys, os
import pandas as pd

def intrinsics(args):



    print("\n#####################################################")
    print("############ Intrinsics {} {} ############".format(args.scene, args.method))
    print("#####################################################")

    if(args.method == 'omvs' or args.method == 'omvs_cleaned'):
        working_dir = os.path.join(args.data_dir,args.scene,"openMVS","")
        input_file = "mesh"
        if (args.free_space_support):
            input_file += "_Jancosek"
        else:
            input_file += "_Labatu"
        if (args.clean_mesh):
            input_file += "_cleaned"
            if(args.refine):
                input_file += "_refined"
        else:
            if(args.refine):
                input_file += "_refined"
            else:
                input_file += "_initial"
    elif(args.method == 'clf' or args.method == 'clf_cleaned'):
        working_dir = os.path.join(args.data_dir,args.scene,"")
        sm = "_"+args.sure_method.split(',')[0]+"_"
        input_file = args.scene+ sm + args.reg_weight
        if (args.clean_mesh):
            input_file += "_cleaned"
        else:
            input_file += "_mesh"
    elif(args.method == 'poisson' or args.method == 'poisson_cleaned'):
        working_dir = os.path.join(args.data_dir,args.scene,"poisson","")
        input_file = args.scene + "_" + str(args.bType)
        if(args.clean_mesh):
            input_file+="_cleaned"
    else:
        print("{} is not a valid method. choose either omvs or clf.".format(args.method))
        sys.exit(1)

    print ("Using input file: ", input_file)

    command = [args.user_dir+args.sure_dir+"/eval",
    "-w", working_dir,
    "-i", input_file+".ply",
     "--omanifold", "1"
    ]
    
    print("run command: ",command)
    p = subprocess.Popen( command , stdout=subprocess.PIPE)
    # exit the whole programm if this step didn't work
    if(p.returncode):
        sys.exit(1)
    # get the stdout output and save it in an array
    # from here: https://stackoverflow.com/questions/18421757/live-output-from-subprocess-command
    output = []
    for line in iter(p.stdout.readline, b''):
        # print(line.decode("utf-8")[:-1])
        output.append(line.decode("utf-8")[:-1])
    data = pd.DataFrame(columns=['Intrinsics:','Values:'])
    output = output[-10:-5]
    for i in range(len(output)):
        d = output[i].split(":")
        data.loc[i] = [d[0][2:], float(d[1])]
    data = data.astype({'Values:': int})
    print(data)
    p.wait()

    return data





if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='evaluates_reconstruction')

    parser.add_argument('-m','--machine', type=str, default="ign-laptop",
                        help='choose the machine, ign-laptop, cnes or enpc.')

    parser.add_argument('--user_dir', type=str, default="/home/adminlocal/PhD/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="data/ETH3D/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scene', type=str, default="meadow",
                        help='on which scene to execute pipeline.')
    parser.add_argument('-i', '--input_file', type=str, default="",
                        help='input file name for evaluation method mine, e.g. lrt_0.')

    # additional EVAL options:
    parser.add_argument('--sure_dir', type=str, default="cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure release dir starting from the user_dir')
    parser.add_argument('--method', type=str, default="clf",
                        help='evaluate point_cloud or mesh.')

    # additional SURE reconstruction options:
    parser.add_argument('-sm','--sure_method', type=str, default="cl,lrt,0,re",
                        help='the m, method parameter for sure. default: cl,lrt,0,re')
    parser.add_argument('--reg_weight', type=str, default="1",
                        help='regularization weight for sure mesh reconstruction as string. default: 1')

    args = parser.parse_args()
    intrinsics(args)