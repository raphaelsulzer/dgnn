import argparse, subprocess, os, glob



def main(args):

    # outfile = os.path.join(args.wdir,args.o+".npz")
    # if(os.path.isfile(outfile) and not args.overwrite):
    #     print("exists!")
    #     return

    # extract features from mpu
    command = [args.sure_dir + "/feat",
               "-w", str(args.wdir),
               "-i", str(args.i[:-4]),
               "-o", str(args.o),
               "-g", str(args.g),
               "-s", "npz",
               "e", ""]
    print("run command: ", command)
    p = subprocess.Popen(command)
    p.wait()

    a=5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench feature extraction')


    parser.add_argument('-d', '--dataset_dir', type=str, default="/mnt/raphael/reconbench/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('--overwrite', type=int, default=0,
                        help='overwrite existing files')
    parser.add_argument('--sure_dir', type=str, default="/home/raphael/cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure build directory, pointing to .../build/release folder starting from user_dir')
    parser.add_argument('--conf', type=int, default=4,
                        help='The scan conf')

    args = parser.parse_args()



    ### train
    args.input = os.listdir(os.path.join(args.dataset_dir, "3_scan"))

    conf_dir = os.path.join(args.dataset_dir,"gt",str(args.conf))
    if not os.path.exists(conf_dir):
        os.makedirs(conf_dir)

    for i in args.input:
        args.wdir = os.path.join(args.dataset_dir)
        args.i = os.path.join("3_scan",i)
        name = i.split('.')[0]
        conf = name.split('_')[1]
        args.o = name
        args.g = "2_watertight/"+i.split('_')[0]+".off"
        main(args)

        # move
        in_dir = os.path.join(args.dataset_dir, "gt", name + "*")
        out_dir = os.path.join(args.dataset_dir, "gt", conf)
        for f in glob.glob(in_dir):
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            p = subprocess.Popen(["mv", f, out_dir])
            p.wait()



