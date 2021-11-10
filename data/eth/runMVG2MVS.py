import os
import subprocess
import argparse


def mvg2mvs(args):
    # home_dir = "/home/adminlocal/PhD/data/TanksAndTemples/" + scene +"/"
    input_file = args.base_dir + args.scene + "/openMV/reconstruction_global/sfm_data.bin"
    output_dir = args.base_dir + args.scene + "/openMV/scene_file.mvs"
    undist_images_out = args.base_dir + args.scene + "/openMV/undist_images/"

    print ("Using input dir  : ", input_file)
    print ("      output_dir : ", output_dir)

    # Create the ouput/matches folder if not present
    if not os.path.exists(undist_images_out):
      os.mkdir(undist_images_out)

    print("\n#######################################")
    print ("######## 7. openMVG 2 openMVS ########")
    print("#######################################")
    pIntrisics = subprocess.Popen(
        [os.path.join(args.openMVG_dir, "openMVG_main_openMVG2openMVS"),
      "-i", input_file, "-o", output_dir, "-d", undist_images_out, "-n", "3"] )
    pIntrisics.wait()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='openMVG_openMVS_reconstruction')

    parser.add_argument('-d', '--base_dir', type=str, default="/home/adminlocal/PhD/data/ETH3D/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scene', type=str, default="meadow",
                        help='on which scene to execute pipeline.')

    parser.add_argument('--openMVG_dir', type=str, default="/home/adminlocal/PhD/cpp/openMVG_release/Linux-x86_64-RELEASE",
                        help='Indicate the openMVG binary directory, pointing to the Linux-x.... folder')

    args = parser.parse_args()

    mvg2mvs(args)