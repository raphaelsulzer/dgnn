import argparse
import subprocess
import sys, os
import pandas as pd

def extrinsics(args):
    print("\n#####################################################")
    print("############ Extrinsics {} {} ############".format(args.scene, args.method))
    print("#####################################################")

    if(args.method == 'point_cloud'):
        working_dir = os.path.join(args.data_dir,args.scene,"openMVS","")
        input_file = working_dir
        input_file+="densify_file"
    elif(args.method == 'omvs' or args.method == 'omvs_cleaned'):
        working_dir = os.path.join(args.data_dir,args.scene,"openMVS","")
        input_file = working_dir
        input_file+= "mesh"
        if (args.free_space_support):
            input_file += "_Jancosek"
        else:
            input_file += "_Labatu"
        if (args.clean_mesh):
            if(args.refine):
                input_file += "_cleaned_refined_sampled"
            else:
                input_file += "_cleaned_sampled"
        else:
            if(args.refine):
                input_file += "_refined_sampled"
            else:
                input_file += "_initial_sampled"
    elif(args.method == 'clf' or args.method == 'clf_cleaned'):
        working_dir = os.path.join(args.data_dir,args.scene,"openMVS","")
        input_file=working_dir
        sm = "_"+args.sure_method.split(',')[0]+"_"
        input_file+= "../"+args.scene+ sm + args.reg_weight
        if (args.clean_mesh):
            input_file += "_cleaned_sampled"
        else:
            input_file += "_sampled"
    elif (args.method == 'poisson' or args.method == 'poisson_cleaned'):
        working_dir = os.path.join(args.data_dir,args.scene,"poisson","")
        input_file=working_dir+args.scene+ "_" + str(args.bType)
        if (args.clean_mesh):
            input_file += "_cleaned_sampled"
        else:
            input_file += "_sampled"
    else:
        print("{} is not a valid method. choose either omvs or clf.".format(args.method))
        sys.exit(1)

    input_file+=".ply"

    gt_file = working_dir+"../dslr_scan_eval/scan_alignment.mlp"
    print ("Using input file  : ", input_file)
    print ("      output file : ", gt_file)

    command = [args.user_dir+args.eval_tool_dir+"/ETH3DMultiViewEvaluation",
    "--ground_truth_mlp_path", gt_file,
    "--reconstruction_ply_path", input_file,
    "--tolerances", args.tolerances,
    ]
    if(args.export_colored_clouds):
        command.append("--completeness_cloud_output_path")
        command.append(input_file[:-4]+"_completeness")
        command.append("--accuracy_cloud_output_path")
        command.append(input_file[:-4]+"_accuracy")

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
    data = pd.DataFrame()
    output = output[-4:]
    for i in range(len(output)):
        d = output[i].split(" ")
        data[d[0]] = d[1:]
        data[d[0]] = data[d[0]].astype(float)
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
    parser.add_argument('--eval_tool_dir', type=str, default="cpp/multi-view-evaluation/build",
                        help='Indicate the eval tool dir starting from the user dir')
    parser.add_argument('-t', '--tolerances', type=str, default="0.01,0.02,0.05,0.1,0.2,0.5",
                        help='tolerances to evaluate. default: "0.01,0.02,0.05,0.1,0.2,0.5"')
    parser.add_argument('-ecc','--export_colored_clouds', type=int, default=0,
                        help='should colored completeness and accuracy clouds be exported')
    parser.add_argument('--method', type=str, default="point_cloud",
                        help='evaluate point_cloud or mesh.')

    # additional SURE reconstruction options:
    parser.add_argument('-sm','--sure_method', type=str, default="cl,lrt,0,re",
                        help='the m, method parameter for sure. default: cl,lrt,0,re')
    parser.add_argument('--reg_weight', type=str, default="1",
                        help='regularization weight for sure mesh reconstruction as string. default: 1')

    args = parser.parse_args()
    extrinsics(args)