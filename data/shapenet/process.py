import argparse, subprocess, sys, os

def run(args):
    # print("\n##########################################################")
    # print("############ Run reconbench of scene {} ############".format(args.scene))
    # print("##########################################################")

    for c in os.listdir(args.data_dir):

        class_dir = os.path.join(args.data_dir,c,'2_watertight','')
        for i in os.listdir(class_dir):
            command = [args.sure_dir + "/sure", "clf",
                       '-w', class_dir,
                       # '-i', os.path.join(class_dir,'2_watertight',i),
                       '-i', i,
                       '-g', i,
                       '-s', "scan,3000,10,0.05,0.0",
                       '-m', "lrtcs,100",
                       '-e', '']

            print(command)
            p = subprocess.Popen(command)
            p.wait()

            if (p.returncode):
                sys.exit(1)












if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')

    parser.add_argument('-d', '--data_dir', type=str, default="/mnt/raphael/ProcessedShapeNet/",
                        help='working directory which should include the different scene folders.')

    parser.add_argument('--sure_dir', type=str, default="/home/raphael/cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure build directory, pointing to .../build/release folder')
    parser.add_argument('--gco', type=str, default="angle-1.0",
                        help='graph cut optimization type,weight. default: area,1.0')

    # additional Mesh reconstruction options:
    parser.add_argument('-p', '--steps', type=str, default='e',
                        help='pipeline steps. default: idmr. extra options: sampling = s, evaluation = e')

    args = parser.parse_args()

    run(args)