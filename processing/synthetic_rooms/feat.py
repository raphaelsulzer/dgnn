import argparse, subprocess, os, glob
from tqdm import tqdm


def main(args):

    outfile = os.path.join(args.wdir,"gt",args.i.split('/')[1]+"_labels.txt")
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
               '-e', ""]
    print("run command: ", *command)
    p = subprocess.Popen(command)
    p.wait()






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')


    parser.add_argument('-d', '--dataset_dir', type=str, default="/home/rsulzer/data/synthetic_room_dataset",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('--overwrite', type=int, default=0,
                        help='overwrite existing files')
    parser.add_argument('--scan_conf', type=int, default=99,
                        help='the scan conf')
    parser.add_argument('--sure_dir', type=str, default="/home/rsulzer/cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure build directory, pointing to .../build/release folder starting from user_dir')
    parser.add_argument('--category', type=str, default='x',
                        help='Indicate the category class')

    args = parser.parse_args()

    if args.category is not None:
        categories = [args.category]
    else:
        categories = os.listdir(args.dataset_dir)
    if 'x' in categories and not args.category == 'x':
        categories.remove('x')

    for j,c in enumerate(categories):
        if c.startswith('.'):
            continue
        ### train
        args.input = os.listdir(os.path.join(args.dataset_dir, c))
        # conf_dir = os.path.join(args.dataset_dir,c,"gt",str(args.conf))
        # if not os.path.exists(conf_dir):
        #     os.makedirs(conf_dir)
        for n,i in enumerate(args.input):
            print("[{}/{}][{}/{}]".format(j+1,len(categories),n+1,len(args.input)))
            args.wdir = os.path.join(args.dataset_dir, c,i)
            args.i = os.path.join("scan",str(args.scan_conf))
            args.o = str(args.scan_conf)
            args.g = os.path.join("..","..","..","synthetic_room_watertight",c,i+'.off')
            try:
                main(args)
            except:
                pass
