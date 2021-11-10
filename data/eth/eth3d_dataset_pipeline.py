# Tanks and Temples input
import argparse
import subprocess
import sys, os

def renameScans(args):

    print("Move lidar scans")


    input_dir = os.path.join(args.user_dir,args.data_dir,args.scene,"is")
    output_dir = os.path.join(args.user_dir,args.data_dir,args.scene,"scan_clean")

    if not os.path.exists(input_dir):
        print("ERROR: input dir {} does not exist".format(input_dir))
        sys.exit(1)

    # Create the depth maps folder if not present
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # get list of all the scans
    all_scans = os.listdir(input_dir)
    if(len(all_scans) < 1):
        print("ERROR: input dir is empty")
        sys.exit(1)

    for i, scan in enumerate(all_scans):
        print ("move scan ", scan)
        pMove = subprocess.call("mv "+
                                input_dir+"/"+scan+" "+
                                output_dir+"/scan"+str(i+1)+".ply"
                                , shell=True)



# copy all the scans from is/XXX01.ply to scan_clean/scan01.ply
def makeCubeMaps(args):

    print("Make cube maps")

    input_dir = os.path.join(args.user_dir,args.data_dir,args.scene,"scan_clean")
    output_dir = os.path.join(args.user_dir,args.data_dir,args.scene,"cube_maps")

    if not os.path.exists(input_dir):
        print("ERROR: input dir {} does not exist".format(input_dir))
        sys.exit(1)

    # Create the depth maps folder if not present
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # get list of all the scans
    all_scans = os.listdir(input_dir)
    if(len(all_scans) < 1):
        print("ERROR: input dir {} is empty".format(input_dir))
        sys.exit(1)

    for i, scan in enumerate(all_scans):
        print ("make cube map of ", scan)
        command = [args.user_dir+args.dataset_pipeline_dir + "CubeMapRenderer",
                   "-c", input_dir+"/"+scan,
                   "-o", output_dir+"/"+scan,
                   "--size", "2048"]

        p = subprocess.Popen(command)
        p.wait()


def run(args):


    if ("i" in args.pipeline_steps):
        renameScans(args)

    if ("c" in args.pipeline_steps):
        makeCubeMaps(args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='openMVG_openMVS_reconstruction')

    parser.add_argument('-m','--machine', type=str, default="ign-laptop",
                        help='choose the machine, ign-laptop, cnes or enpc.')

    parser.add_argument('--user_dir', type=str, default="/home/docker/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="data/TanksAndTemples/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scene', type=str, default="Barn",
                        help='on which scene to execute pipeline.')

    parser.add_argument('--dataset_pipeline_dir', type=str, default="cpp/dataset-pipeline/build/release/",
                        help='Indicate the dataset-pipeline binary directory, pointing to .../build/ folder starting from user_dir')

    # additional Mesh reconstruction options:
    parser.add_argument('-p', '--pipeline_steps', type=str, default='c',
                        help='pipeline steps. default: idmr. extra options: sampling = s, evaluation = e')



    args = parser.parse_args()
    run(args)