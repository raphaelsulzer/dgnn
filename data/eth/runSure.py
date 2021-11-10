import argparse
import subprocess
import sys
import os

def sure(args):
    print("\n##################################################")
    print("############ Sure reconstruction {}_{} ############".format(args.scene, args.method))
    print("##################################################")

    working_dir = os.path.join(args.user_dir,args.data_dir,args.scene,'')
    print ("Using input file  : ", "openMVS/densify_file.mvs")
    print ("      output file : ", args.scene)

    command = [args.user_dir+args.sure_dir+"/sure",
        "clf",
        "-s", "omvs",
        "-w", working_dir,
        "-i", args.scene,
        "-o", args.scene,
        "-m", args.sure_method,
        "--omanifold", "1"
        ]
    if(not args.sure_method[:3] == "lrt"):
        command += ["-p", "prediction/"+args.scene+"_lrt_0_"+args.pi+".npz"]
        command += ["--output_sampling", args.output_sampling]
        if(args.gco):
            command += ["--gco", args.gco]
        if(args.lff > 0):
            command += ["--lff", str(args.lff)]
        if(args.nsc > 0):
            command += ["--nsc", str(args.nsc)]
        if(args.clean_mesh):
            command += ["-e", "ivr"]
            command += ["--clean", "1"]
        else:
            command += ["-e", "imsvr"]
    else:
        command += ["-e", ""]
        command += ["--scale", "1000"]

    if (args.adt):
        command += ["--adt", str(args.adt)]

    print("Execute: ",command)

    p = subprocess.Popen( command )
    p.wait()

    if(p.returncode):
        print("\nEXITING WITH SURE ERROR\n")
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

    parser.add_argument('--sure_dir', type=str, default="cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure release dir starting from the user_dir')

    parser.add_argument('--method', type=str, default="point_cloud",
                        help='evaluate point_cloud or mesh.')
    parser.add_argument('-i', '--input_file', type=str, default="",
                        help='input file name for sample method mine, e.g. lrt_0.')


    # additional SURE reconstruction options:
    parser.add_argument('-sm','--sure_method', type=str, default="cl,lrt,0,re",
                        help='the m, method parameter for sure. default: cl,lrt,0,re')
    parser.add_argument('--adt', type=float, default=0.0,
                        help='epsilon for adaptive delaunay triangulation. default: 0.0')
    parser.add_argument('--gco', type=str, default="area-1.0",
                        help='graph cut optimization type,weight. default: area,3.0')

    parser.add_argument('--lff', type=int, default=0,
                        help="Factor Y to determine surface area"
                                          "threshold X for removing large facets,"
                                          "with X = Y * mean_surface_area. default: 0")
    parser.add_argument('--nsc', type=int, default=0,
                        help="Number of surface mesh components to keep")
    parser.add_argument('--clean_mesh', type=int, default=1,
                        help='enable/disable all mesh clean options. default: disabled.')

    parser.add_argument('--output_sampling', type=int, default="as-300",
                        help='number of points to sample from the mesh. default: 1000000')

    args = parser.parse_args()
    sure(args)