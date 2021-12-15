import argparse, subprocess, os, glob
from tqdm import tqdm
import shutil


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

    for j,c in enumerate(categories):
        if c.startswith('.'):
            continue
        ### train
        args.input = os.listdir(os.path.join(args.dataset_dir, c))
        # conf_dir = os.path.join(args.dataset_dir,c,"gt",str(args.conf))
        # if not os.path.exists(conf_dir):
        #     os.makedirs(conf_dir)
        for n,i in enumerate(args.input):
            dir = os.path.join(args.dataset_dir,c,i,'scan')
            try:
                shutil.rmtree(dir)
            except:
                pass