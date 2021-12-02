import argparse, subprocess, os, glob
from tqdm import tqdm


def main(args):

    # outfile = os.path.join(args.wdir,args.o+".npz")
    # if(os.path.isfile(outfile) and not args.overwrite):
    #     print("exists!")
    #     return

    # extract features from mpu
    command = ["obj2off", str(args.i),
               "-o", str(args.o)]
    # print("run command: ", command)
    p = subprocess.Popen(command)
    p.wait()






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')


    parser.add_argument('-d', '--dataset_dir', type=str, default="/mnt/raphael/ShapeNetWatertight/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('--category', type=str, default=None,
                        help='Indicate the category class')

    args = parser.parse_args()

    if args.category is not None:
        categories = [args.category]
    else:
        categories = os.listdir(args.dataset_dir)
    if 'x' in categories:
        categories.remove('x')

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


        for i in tqdm(args.input,ncols=50):
            os.makedirs(os.path.join(args.dataset_dir, c, i, "mesh"),exist_ok=True)


            args.i = os.path.join(args.dataset_dir,c,i,"isosurf.obj")
            if(not os.path.isfile(args.i)):
                continue

            args.o = os.path.join(args.dataset_dir,c,i,"mesh","mesh.off")

            main(args)

            os.remove(args.i)





