import argparse, subprocess, os, random, sys
import numpy as np
from tqdm import tqdm
import multiprocessing

dataset_dir = "/mnt/raphael/"
sure_dir = "/home/raphael/cpp/surfaceReconstruction/build/release"

# dataset_dir = "/home/rsulzer/data/"
# sure_dir = "/home/rsulzer/cpp/surfaceReconstruction/build/release"
scan_conf = 99
overwrite = False

np.random.seed(seed=42)


def scan_one(obj):

    # choose random scan parameters

    if(scan_conf == 0):
        points = 12000
        cameras = 15
        noise = 0.0
        outliers = 0.0
    elif(scan_conf == 1):
        points = 3000
        cameras = 15
        noise = 0.0025
        outliers = 0.0
    elif(scan_conf == 2):
        points = 12000
        cameras = 15
        noise = 0.005
        outliers = 0.33
    elif(scan_conf == 3): # convonet configuration, 50 cameras
        points = 3000
        cameras = 50
        noise = 0.005
        outliers = 0.0
    elif(scan_conf == 4): # convonet configuration, 10 cameras
        points = 3000
        cameras = 10
        noise = 0.005
        outliers = 0.0
    elif(scan_conf == 99):
        points = int(500)+int(np.abs(np.random.randn())*10000) # we want 3*sigma to be 30000 (so factor should be 6666.66 but made it a bit lower)
        cameras = 2+int(np.abs(np.random.randn())*6) # we want 3*sigma to be 20 (so factor should be 6.66 but made it a bit lower), and at least 2 cameras
        noise = np.abs(np.random.randn())*0.01 # we want 3*sigma to be 0.03 (so factor should be 0.01)
        outliers = np.abs(np.random.randn())*0.1 # we want 3*sigma to be 0.3 (so factor should be 0.1)
    else:
        print("\nERROR: not a valid config. choose [0,1,2]")
        sys.exit(1)


    outfile = os.path.join(dataset_dir,"synthetic_room_dataset",obj,"scan",str(scan_conf)+".npz")
    if(os.path.isfile(outfile) and not overwrite):
        print("exists!")
        return

    outdir = os.path.join(dataset_dir,"synthetic_room_dataset")

    os.makedirs(os.path.join(outdir, obj,'scan'), exist_ok=True)


    # extract features from mpu
    command = [sure_dir + "/scan",
               "-w", str(dataset_dir),
               "-i", str(os.path.join("synthetic_room_watertight",obj+".off")),
               "-o", str(os.path.join("synthetic_room_dataset",obj,"scan",str(scan_conf))),
               '--normal_method', 'jet',
               "--noise", str(noise),
               "--outliers", str(outliers),
               "--points", str(points),
               "--cameras", str(cameras),
               "--export", "npz",
               "--gclosed", "0"]
    print('\n',*command)
    p = subprocess.Popen(command)
    p.wait()


    npzfile = np.load(outfile)
    np.savez(outfile,
             points=npzfile["points"],
             normals=npzfile["normals"],
             gt_normals=npzfile["gt_normals"],
             sensor_position=npzfile["sensor_position"],
             cameras=np.array(cameras,dtype=np.float64),
             noise=np.array(noise,dtype=np.float64),
             outliers=np.array(outliers,dtype=np.float64))

def main(obj_list,njobs=0):

    if njobs>0:
        # multiprocessing.set_start_method('spawn', force=True)
        pool = multiprocessing.Pool(njobs)
        try:
            for _ in tqdm(pool.imap_unordered(scan_one, obj_list), total=len(obj_list)):
                pass
            # pool.map_async(process_one, obj_list).get()
        except KeyboardInterrupt:
            # Allow ^C to interrupt from any thread.
            exit()
        pool.close()
    else:
        for obj in tqdm(obj_list):
            scan_one(obj)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')


    parser.add_argument('--njobs', type=int, default=0,
                        help='number of workers. > 0 uses multiprocessing.')
    parser.add_argument('--category', type=str, default=None,
                        help='Indicate the category class')

    args = parser.parse_args()

    if args.category is not None:
        categories = [args.category]
    else:
        categories = os.listdir(os.path.join(dataset_dir,"synthetic_room_watertight"))
    if 'x' in categories and  not args.category == 'x':
        categories.remove('x')


    # scan all training data with random configuration from 0,1,2
    # and test data with 0,1,2

    ### scanner confs
    # 0 (easy) --cameras 15 --points 12000 --noise 0.000 --outliers 0.0
    # 1 (medium) --cameras 15 --points 3000 --noise 0.0025 --outliers 0.0
    # 2 (hard) --cameras 15 --points 12000 --noise 0.005 --outliers 0.33
    # 3 (convonet) --cameras 50 --points 3000 --noise 0.005 --outliers 0.0

    for i,c in enumerate(categories):
        # if c.startswith('.'):
        #     continue
        print("\n\n############## Processing {} - {}/{} ############\n\n".format(c,i+1,len(categories)))

        ### train
        cdir = os.path.join(dataset_dir, "synthetic_room_watertight", c)
        inputs = os.listdir(cdir)

        obj_list = []
        for i in inputs:
            file = os.path.join(c,i[:-4])
            obj_list.append(file)
        main(obj_list,args.njobs)










