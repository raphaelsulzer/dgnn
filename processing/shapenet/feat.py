import argparse, subprocess, os, glob



def main(args):

    outfile = os.path.join(args.wdir,'gt',args.o+"_3dt.npz")
    if(os.path.isfile(outfile) and not args.overwrite):
        print("exists!")
        return

    # extract features from mpu
    command = [args.sure_dir + "/feat",
               "-w", str(args.wdir),
               "-i", str(args.i),
               "-o", str(args.o),
               "-g", str(args.g),
               "-s", "npz",
               "-e",""]
    print(*command)
    p = subprocess.Popen(command)
    p.wait()






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')


    parser.add_argument('-d', '--dataset_dir', type=str, default="/mnt/raphael/ShapeNet/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('--overwrite', type=int, default=0,
                        help='overwrite existing files')
    parser.add_argument('--sure_dir', type=str, default="/home/raphael/cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure build directory, pointing to .../build/release folder starting from user_dir')
    parser.add_argument('--conf', type=int, default=4,
                        help='The scan conf')


    parser.add_argument('--category', type=str, default=None,
                        help='Indicate the category class')

    args = parser.parse_args()

    if args.category is not None:
        categories = [args.category]
    else:
        categories = os.listdir(args.dataset_dir)
    if 'x' in categories:
        categories.remove('x')
    if 'real' in categories:
        categories.remove('real')
    if 'metadata.yaml' in categories:
        categories.remove('metadata.yaml')

    for idx,c in enumerate(categories):
        if c.startswith('.'):
            continue
        print("\n############## Processing {}/{} ############\n".format(idx+1,len(categories)))

        split_file = os.path.join(args.dataset_dir,c,"test100.lst")
        with open(split_file, 'r') as f:
            models = f.read().split('\n')
        models = list(filter(None, models))

        ### train
        # args.input = os.listdir(os.path.join(args.dataset_dir, c, "scan", str(args.conf)))

        # conf_dir = os.path.join(args.dataset_dir,c,"gt",str(args.conf))
        # if not os.path.exists(conf_dir):
        #     os.makedirs(conf_dir)

        for m in models:
            args.wdir = os.path.join(args.dataset_dir, c,m)
            args.i = os.path.join("scan", str(args.conf))
            args.o = str(args.conf)
            args.g = "mesh/mesh.off"

            try:
                main(args)

                # # move
                # in_dir =  os.path.join(args.dataset_dir,c,"gt",i+"*")
                # out_dir = os.path.join(conf_dir,name)
                # if not os.path.exists(out_dir):
                #     os.makedirs(out_dir)
                # for f in glob.glob(in_dir):
                #     p = subprocess.Popen(["mv",f,out_dir])
                #     p.wait()
            except:
                print("Problem with shape {}-{}".format(c,m))

        # # night_stand to nightstand
        # in_dir =  os.path.join(args.dataset_dir,c,"train")
        # fl = os.listdir(in_dir)
        # for f in fl:
        #     inf = os.path.join(in_dir,f)
        #     fn = f.split('/')[-1][11:]
        #     out = os.path.join(in_dir,"nightstand"+fn)
        #     p = subprocess.Popen(["mv",inf,out])
        #     p.wait()

