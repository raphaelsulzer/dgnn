import argparse, subprocess, os, glob



def main(args):

    # outfile = os.path.join(args.wdir,args.o+".npz")
    # if(os.path.isfile(outfile) and not args.overwrite):
    #     print("exists!")
    #     return

    # extract features from mpu
    command = [args.sure_dir + "/feat",
               "-w", str(args.wdir),
               "-i", str(args.i),
               "-o", str(args.o),
               "-g", str(args.g),
               "-s", "npz",
               '-e', ""]
    print("run command: ", *command)
    p = subprocess.Popen(command)
    p.wait()






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')


    parser.add_argument('-d', '--dataset_dir', type=str, default="/mnt/raphael/ShapeNetWatertight/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('--overwrite', type=int, default=0,
                        help='overwrite existing files')
    parser.add_argument('--scan_conf', type=int, default=99,
                        help='the scan conf')
    parser.add_argument('--sure_dir', type=str, default="/home/raphael/cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure build directory, pointing to .../build/release folder starting from user_dir')
    # parser.add_argument('--conf', type=int, default=4,
    #                     help='The scan conf')


    parser.add_argument('--category', type=str, default='x',
                        help='Indicate the category class')

    args = parser.parse_args()

    if args.category is not None:
        categories = [args.category]
    else:
        categories = os.listdir(args.dataset_dir)


    # scan all training data with random configuration from 0,1,2
    # and test data with 0,1,2

    ### scanner confs
    # 0 (easy) --cameras 15 --points 12000 --noise 0.000 --outliers 0.0
    # 1 (medium) --cameras 15 --points 3000 --noise 0.0025 --outliers 0.0
    # 2 (hard) --cameras 15 --points 12000 --noise 0.005 --outliers 0.33
    # 3 (convonet) --cameras 50 --points 3000 --noise 0.005 --outliers 0.0

    for i,c in enumerate(categories):
        if c.startswith('.'):
            continue
        print("\n############## Processing {}/{} ############\n".format(i+1,len(categories)))

        ### train
        args.input = os.listdir(os.path.join(args.dataset_dir, c))
        # conf_dir = os.path.join(args.dataset_dir,c,"gt",str(args.conf))
        # if not os.path.exists(conf_dir):
        #     os.makedirs(conf_dir)

        for i in args.input:
            args.wdir = os.path.join(args.dataset_dir, c,i)
            args.i = os.path.join("scan",str(args.scan_conf))
            args.o = str(args.scan_conf)
            args.g = os.path.join("mesh",'mesh.off')
            main(args)

            # # move
            # in_dir =  os.path.join(args.dataset_dir,c,"gt",name+"*")
            # out_dir = os.path.join(args.dataset_dir,c,"gt",conf,name)
            # for f in glob.glob(in_dir):
            #     if not os.path.exists(out_dir):
            #         os.makedirs(out_dir)
            #     p = subprocess.Popen(["mv",f,out_dir])
            #     p.wait()




        # # night_stand to nightstand
        # in_dir =  os.path.join(args.dataset_dir,c,"train")
        # fl = os.listdir(in_dir)
        # for f in fl:
        #     inf = os.path.join(in_dir,f)
        #     fn = f.split('/')[-1][11:]
        #     out = os.path.join(in_dir,"nightstand"+fn)
        #     p = subprocess.Popen(["mv",inf,out])
        #     p.wait()

