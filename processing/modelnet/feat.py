import argparse, subprocess, os, glob



def main(args):

    outfile = os.path.join(args.wdir,'dgnn',str(args.conf),args.i+"_3dt.npz")
    if(os.path.isfile(outfile) and not args.overwrite):
        print("exists!")
        return

    # extract features from mpu
    command = [args.meshtools_dir + "/feat",
               "-w", str(args.wdir),
               "-i", str(args.i),
               "-o", str(args.o),
               "-g", str(args.g),
               "--occ", str(args.occ),
               "-s", "npz",
               "-e",""]
    print("run command: ", command)
    p = subprocess.Popen(command)
    p.wait()






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')


    parser.add_argument('-d', '--dataset_dir', type=str, default="/mnt/raphael/ModelNet10/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('--overwrite', type=int, default=0,
                        help='overwrite existing files')
    parser.add_argument('--meshtools_dir', type=str, default="/home/raphael/cpp/mesh-tools/build/release",
                        help='Indicate the sure build directory, pointing to .../build/release folder starting from user_dir')
    parser.add_argument('--conf', type=int, default=43,
                        help='The scan conf')


    parser.add_argument('--category', type=str, default=None,
                        help='Indicate the category class')

    args = parser.parse_args()

    if args.category is not None:
        categories = [args.category]
    else:
        categories = os.listdir(args.dataset_dir)

    for idx,c in enumerate(categories):
        if c.startswith('.'):
            continue
        print("\n############## Processing {}/{} ############\n".format(idx+1,len(categories)))

        ### train
        args.input = os.listdir(os.path.join(args.dataset_dir, c, "scan", str(args.conf)))

        conf_dir = os.path.join(args.dataset_dir,c,"dgnn",str(args.conf))
        if not os.path.exists(conf_dir):
            os.makedirs(conf_dir)

        for i in args.input:
            args.wdir = os.path.join(args.dataset_dir, c)
            args.i = os.path.join("scan", str(args.conf),i,"scan")
            name = c+"_"+i
            args.o = name
            # args.g = "mesh/"+i+".off"
            args.g = os.path.join('..','..','ModelNet10_watertight',name+'.off')
            args.occ = os.path.join("eval",i,"points.npz")

            try:
                main(args)

                # move
                in_dir =  os.path.join(args.dataset_dir,c,"dgnn",name+"*")
                out_dir = os.path.join(conf_dir,i)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                for f in glob.glob(in_dir):
                    p = subprocess.Popen(["mv",f,out_dir])
                    p.wait()
            except Exception as e:
                print('\n')
                print(e)
                print("Problem with shape {}-{}".format(c,i))

