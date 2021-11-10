import sys
import subprocess
import argparse

def colmap2mvs(args):
    print("\n#################################################")
    print("############ 1. Initialize {} ############".format(args.scene))
    print("#################################################")


    working_dir = args.data_dir + args.scene + "/openMVS/"
    input_folder = "../dslr_calibration_undistorted/"
    # input_folder = "../images/"
    output_file = "scene_file.mvs"
    undist_image_folder = "../images/"
    print ("Using input dir  : ", input_folder)
    print ("      output_dir : ", output_file)

    p = subprocess.Popen([args.user_dir+args.openMVS_dir+"/InterfaceCOLMAP",
    "-w", working_dir,
    "-i", input_folder,
    "-o", output_file,
    "--image-folder", undist_image_folder,
    "--normalize", "0"
    ] )
    p.wait()

    if(p.returncode):
        sys.exit(1)

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Colmap SfM to openMVS project')

    parser.add_argument('-m','--machine', type=str, default="ign-laptop",
                        help='choose the machine, ign-laptop, cnes or enpc.')

    parser.add_argument('--user_dir', type=str, default="/home/adminlocal/PhD/",
                        help='the user folder, or PhD folder. default: /home/adminlocal/PhD/')
    parser.add_argument('-d', '--data_dir', type=str, default="data/ETH3D/",
                        help='working directory which should include the different scene folders. default: data/ETH3D/')
    parser.add_argument('-s', '--scene', type=str, default="pipes",
                        help='on which scene to execute pipeline.')

    parser.add_argument('--openMVS_dir', type=str, default="cpp/openMVS_release/bin",
                        help='Indicate the openMVS binary directory, pointing to .../bin folder starting from user_dir')

    args = parser.parse_args()

    colmap2mvs(args)