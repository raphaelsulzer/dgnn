import os
import subprocess
import argparse
import sys

def sfm_eth(args):

    # image_dir = args.base_dir+args.scene+"/images/dslr_images_undistorted"
    image_dir = args.base_dir+args.scene
    gt_dir = args.base_dir+args.scene+"/dslr_calibration_undistorted"

    output_dir = args.base_dir+args.scene+"/openMV"

    matches_dir = os.path.join(output_dir, "matches")
    reconstruction_dir = os.path.join(output_dir, "reconstruction_" + args.reconstruction_method)
    camera_file_params = os.path.join(args.CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")



    # Create the ouput/matches folder if not present
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)
    if not os.path.exists(matches_dir):
      os.mkdir(matches_dir)

    if("i" in args.pipeline_steps):
        print("\n####################################################")
        print("############## 1. Intrinsics analysis ##############")
        print("####################################################")
        print ("Using image dir  : ", image_dir)
        print ("      output_dir : ", output_dir)
        pIntrisics = subprocess.Popen( [os.path.join(args.openMVG_dir, "openMVG_main_SfMInit_ImageListingFromKnownPoses"),
          "-i", image_dir, "-o", matches_dir, "-g", gt_dir, "-t", "4"] )
        pIntrisics.wait()

    if("l" in args.pipeline_steps):
        print("\n####################################################")
        print("############## 1. Intrinsics analysis ##############")
        print("####################################################")
        print ("Using image dir  : ", image_dir)
        print ("      output_dir : ", output_dir)
        pIntrisics = subprocess.Popen([os.path.join(args.openMVG_dir, "openMVG_main_SfMInit_ImageListing"),
                                       "-i", image_dir, "-o", matches_dir, "-d", camera_file_params, "-f", "2304"])
        pIntrisics.wait()



    if("f" in args.pipeline_steps):
        print("\n#################################################")
        print("############## 2. Compute features ##############")
        print("#################################################")
        print ("Using input file  : ", matches_dir+"/sfm_data.json")
        print ("      output_dir : ", matches_dir)
        pFeatures = subprocess.Popen( [os.path.join(args.openMVG_dir, "openMVG_main_ComputeFeatures"),
                "-i", matches_dir+"/sfm_data.json", "-o", matches_dir,
                "-m", "SIFT", "-p", "HIGH", "-f", str(args.force_recompute)] )
        pFeatures.wait()

    if ("m" in args.pipeline_steps):
        print("\n################################################")
        print("############## 3. Compute matches ##############")
        print("################################################")
        print ("Using input file  : ", matches_dir+"/sfm_data.json")
        print ("      output_dir : ", matches_dir)
        if(args.reconstruction_method == "global"):
            print("computing matches for a global SfM pipeline")
            g_p = "e"
        if(args.reconstruction_method == "sequential"):
            print("computing matches for a sequential SfM pipeline")
            g_p = "f"
        pMatches = subprocess.Popen( [os.path.join(args.openMVG_dir, "openMVG_main_ComputeMatches"),
                "-i", matches_dir+"/sfm_data.json", "-o", matches_dir,
                "-n", "ANNL2", "-g", g_p, "-f", str(args.force_recompute)] )
        pMatches.wait()




    if ("r" in args.pipeline_steps):
        # Create the reconstruction if not present
        if not os.path.exists(reconstruction_dir):
            os.mkdir(reconstruction_dir)
        if(args.reconstruction_method == "global"):
            print("\n#############################################")
            print("######## 4. Do Global reconstruction ########")
            print("#############################################")
            print("Using input file  : ", matches_dir + "/sfm_data.json")
            print("      output_dir : ", reconstruction_dir)
            pRecons = subprocess.Popen( [os.path.join(args.openMVG_dir, "openMVG_main_GlobalSfM"),
                    "-i", matches_dir+"/sfm_data.json", "-m", matches_dir, "-o", reconstruction_dir] )
            pRecons.wait()
        elif(args.reconstruction_method == "sequential"):
            print("\n#############################################################")
            print("######## 4. Do Sequential/Incremental reconstruction ########")
            print("#############################################################")
            print("Using input file  : ", matches_dir + "/sfm_data.json")
            print("      output_dir : ", reconstruction_dir)
            pRecons = subprocess.Popen(
                [os.path.join(args.openMVG_dir, "openMVG_main_IncrementalSfM"),
                 "-i", matches_dir + "/sfm_data.json", "-m", matches_dir, "-o", reconstruction_dir])
            pRecons.wait()
        else:
            print("not a valid method. choose either global or sequential.")
            sys.exit(1)

    if("c" in args.pipeline_steps):
        print("\n############################################")
        print("############ 5. Colorize Structure ############")
        print("############################################")
        print("Using input file  : ", reconstruction_dir+"/sfm_data.bin")
        print("      output file : ", os.path.join(reconstruction_dir,"colorized.ply"))
        pRecons = subprocess.Popen( [os.path.join(args.openMVG_dir, "openMVG_main_ComputeSfM_DataColor"),
                "-i", reconstruction_dir+"/sfm_data.bin", "-o", os.path.join(reconstruction_dir,"colorized.ply")] )
        pRecons.wait()

    # optional, compute final valid structure from the known camera poses
    if ("s" in args.pipeline_steps):
        print("\n####################################################################")
        print("####### 6. Structure from Known Poses (robust triangulation) #######")
        print("####################################################################")
        if(args.reconstruction_method == "global"):
            print("computing structure from a global SfM pipeline")
            g_p = "e"
        if(args.reconstruction_method == "sequential"):
            print("computing structure from a sequential SfM pipeline")
            g_p = "f"
        pRecons = subprocess.Popen( [os.path.join(args.openMVG_dir, "openMVG_main_ComputeStructureFromKnownPoses"),
                    "-i", reconstruction_dir+"/sfm_data.bin", "-m", matches_dir, "-f", os.path.join(matches_dir, "matches."+g_p+".bin"), "-o", os.path.join(reconstruction_dir,"robust.bin")] )
        pRecons.wait()
        pRecons = subprocess.Popen( [os.path.join(args.openMVG_dir, "openMVG_main_ComputeSfM_DataColor"),
                    "-i", reconstruction_dir+"/robust.bin", "-o", os.path.join(reconstruction_dir,"robust_colorized.ply")] )
        pRecons.wait()

    if("2" in args.pipeline_steps):
        input_file = reconstruction_dir+"/sfm_data.bin"
        output_dir = args.base_dir + args.scene + "/openMV/scene_file.mvs"
        undist_images_out = args.base_dir + args.scene + "/openMV/undist_images/"

        print("Using input file  : ", input_file)
        print("      output file : ", output_dir)

        # Create the ouput/matches folder if not present
        if not os.path.exists(undist_images_out):
            os.mkdir(undist_images_out)

        print("\n########################################")
        print("######### 7. openMVG 2 openMVS #########")
        print("########################################")
        pIntrisics = subprocess.Popen(
            [os.path.join(args.openMVG_dir, "openMVG_main_openMVG2openMVS"),
             "-i", input_file, "-o", output_dir, "-d", undist_images_out, "-n", "3"])
        pIntrisics.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='openMVG_openMVS_reconstruction')

    parser.add_argument('-d', '--base_dir', type=str, default="/home/adminlocal/PhD/data/ETH3D/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scene', type=str, default="meadow",
                        help='on which scene to execute pipeline.')
    parser.add_argument('--openMVG_dir', type=str, default="/home/adminlocal/PhD/cpp/openMVG_release/Linux-x86_64-RELEASE",
                        help='Indicate the openMVG binary directory, pointing to the Linux-x.... folder')

    parser.add_argument('-p', '--pipeline_steps', type=str, default='ifmrcs2',
                        help='pipeline steps, default: ifmrcs2')
    parser.add_argument('-f', '--force_recompute', type=int, default=0,
                        help='recompute features and matches')
    parser.add_argument('-r', '--reconstruction_method', type=str, default="global",
                        help='choose SfM method, either global or sequential.')

    args = parser.parse_args()

    # Indicate the openMVG camera sensor width directory
    args.CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/adminlocal/PhD/cpp/openMVG_release/openMVG/exif/sensor_width_database"

    sfm_eth(args)
