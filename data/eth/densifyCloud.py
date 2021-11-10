import argparse
import subprocess
import sys

def densify(args):
    print("\n##############################################")
    print("############ 2. Densify {} ############".format(args.scene))
    print("##############################################")

    working_dir = args.data_dir + args.scene + "/openMVS/"
    input_file = "scene_file.mvs"
    output_file = "densify_file.mvs"
    print ("Using input dir  : ", input_file)
    print ("      output_dir : ", output_file)

    p = subprocess.Popen( [args.user_dir+args.openMVS_dir+"/DensifyPointCloud",
    "-w", working_dir,
    "-i", input_file,
    "-o", output_file,
    "--number-views-fuse", "2",
    "--optimize", "0",
    "--resolution-level", str(args.resolution_level),
    "--estimate-normals", "1",
    "--filter-point-cloud", str(args.filter_point_cloud)
    ] )
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

    parser.add_argument('--openMVS_dir', type=str, default="cpp/openMVS_release/bin",
                        help='Indicate the openMVS binary directory, pointing to .../bin folder starting from user_dir')

    # additional Densify options:
    parser.add_argument('--resolution_level', type=int, default=1,
                        help='how many times to scale down the images before point cloud computation')
    parser.add_argument('--filter_point_cloud', type=int, default=0,
                        help='filter dense point-cloud based on visibility')

    args = parser.parse_args()

    densify(args)