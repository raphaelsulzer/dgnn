import argparse, subprocess, os, glob
from tqdm import tqdm
import multiprocessing


dataset_dir = '/mnt/raphael/synthetic_room_watertight'

def convert_one(file):

    # outfile = os.path.join(args.wdir,args.o+".npz")
    # if(os.path.isfile(outfile) and not args.overwrite):
    #     print("exists!")
    #     return

    # os.makedirs(os.path.join(args.dataset_dir, c, i, "mesh"), exist_ok=True)

    inp = os.path.join(dataset_dir,file)
    out = os.path.join(dataset_dir,file[:-4]+'.off')

    # try:
    # extract features from mpu
    command = ["obj2off", inp,
               "-o", out,
               "-d", str(8)]
    # print("run command: ", *command)
    p = subprocess.Popen(command)
    p.wait()

    a=5
    os.remove(inp)
    # except:
    #     pass


def main(obj_list,njobs=0):

    if njobs>0:
        # multiprocessing.set_start_method('spawn', force=True)
        pool = multiprocessing.Pool(njobs)
        try:
            for _ in tqdm(pool.imap_unordered(convert_one, obj_list), total=len(obj_list)):
                pass
            # pool.map_async(process_one, obj_list).get()
        except KeyboardInterrupt:
            # Allow ^C to interrupt from any thread.
            exit()
        pool.close()
    else:
        for obj in tqdm(obj_list):
            convert_one(obj)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')


    # parser.add_argument('-d', '--dataset_dir', type=str, default="/home/rsulzer/data/synthetic_room_watertight",
    #                     help='working directory which should include the different scene folders.')
    parser.add_argument('--category', type=str, default=None,
                        help='Indicate the category class')
    parser.add_argument('--njobs', type=int, default=2,
                        help='number of workers. > 0 uses multiprocessing.')

    args = parser.parse_args()

    if args.category is not None:
        categories = [args.category]
    else:
        categories = os.listdir(dataset_dir)
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
        input = os.listdir(os.path.join(dataset_dir, c))
        # conf_dir = os.path.join(args.dataset_dir,c,"gt",str(args.conf))
        # if not os.path.exists(conf_dir):
        #     os.makedirs(conf_dir)

        obj_list = []
        for i in input:

            file = c+'/'+i

            ipath = os.path.join(dataset_dir,c,i)
            # args.o = os.path.join(args.dataset_dir,c,i[:-4]+".off")

            if(not os.path.isfile(ipath[:-4]+".obj")):
                continue

            obj_list.append(file)

        main(obj_list,args.njobs)






