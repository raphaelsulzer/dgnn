import argparse, subprocess, sys, os

def run(args):
    # print("\n##########################################################")
    # print("############ Run reconbench of scene {} ############".format(args.scene))
    # print("##########################################################")

    for c in os.listdir(args.data_dir):

        input = os.path.join(args.data_dir,c,"import",c+"_cl_05_textured.ply")
        output = os.path.join(args.data_dir,c,"import",c+"_cl_05_textured.obj")
        command = ["meshlabserver",
                   '-i', input,
                   '-o', output,
                   '-om', "vn", "wt"]

        print("run command: ",command)
        p = subprocess.Popen(command)
        p.wait()

        if (p.returncode):
            sys.exit(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ply to obj with meshlabserver')

    parser.add_argument('-d', '--data_dir', type=str, default="/home/adminlocal/PhD/data/ETH3D",
                        help='working directory which should include the different scene folders.')

    args = parser.parse_args()

    run(args)