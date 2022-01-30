import argparse, subprocess, os, random, sys
import numpy as np
from tqdm import tqdm
import multiprocessing

def scan_one(args):

    # choose random scan parameters

    if(args.scan_conf == 0):
        points = 12000
        cameras = 15
        noise = 0.0
        outliers = 0.0
    elif(args.scan_conf == 1):
        points = 3000
        cameras = 15
        noise = 0.0025
        outliers = 0.0
    elif(args.scan_conf == 2):
        points = 12000
        cameras = 15
        noise = 0.005
        outliers = 0.33
    elif(args.scan_conf == 3): # convonet configuration, 50 cameras
        points = 3000
        cameras = 50
        noise = 0.005
        outliers = 0.0
    elif(args.scan_conf == 4): # convonet configuration, 10 cameras
        points = 3000
        cameras = 10
        noise = 0.005
        outliers = 0.0
    elif(args.scan_conf == 99):
        points = int(np.abs(np.random.randn())*6000) # we want 3*sigma to be 20000 (so factor should be 6666.66 but made it a bit lower)
        cameras = 2+int(np.abs(np.random.randn())*6) # we want 3*sigma to be 20 (so factor should be 6.66 but made it a bit lower), and at least 2 cameras
        noise = np.abs(np.random.randn())*0.01 # we want 3*sigma to be 0.03 (so factor should be 0.01)
        outliers = np.abs(np.random.randn())*0.1 # we want 3*sigma to be 0.3 (so factor should be 0.1)
    else:
        print("\nERROR: not a valid config. choose [0,1,2]")
        sys.exit(1)

    outfile = os.path.join(args.wdir,args.o+".npz")
    if(os.path.isfile(outfile) and not args.overwrite):
        print("exists!")
        return

    # extract features from mpu
    command = [args.sure_dir + "/scan",
               "-w", str(args.wdir),
               "-i", str(args.i),
               "-o", str(args.o),
               '--normal_method', 'jet',
               "--noise", str(noise),
               "--outliers", str(outliers),
               "--points", str(points),
               "--cameras", str(cameras),
               "--export", "npz",
               "--gclosed", "1"]
    # print("run command: ", command)
    p = subprocess.Popen(command)
    p.wait()


    npzfile = np.load(os.path.join(args.wdir,'scan',str(args.scan_conf)+".npz"))
    np.savez(os.path.join(args.wdir,'scan',str(args.scan_conf)+".npz"),
             points=npzfile["points"],
             normals=npzfile["normals"],
             gt_normals=npzfile["gt_normals"],
             sensor_position=npzfile["sensor_position"],
             cameras=np.array(cameras,dtype=np.float64),
             noise=np.array(noise,dtype=np.float64),
             outliers=np.array(outliers,dtype=np.float64))

    a=5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')

    parser.add_argument('-d', '--dataset_dir', type=str, default="/home/rsulzer/data2/ShapeNetWatertight/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('--scan_conf', type=int, default=4,
                        help='the scan conf')
    parser.add_argument('--overwrite', type=int, default=1,
                        help='overwrite existing files')
    parser.add_argument('--njobs', type=int, default=0,
                        help='number of workers. > 0 uses multiprocessing.')
    parser.add_argument('--sure_dir', type=str, default="/home/rsulzer/cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure build directory, pointing to .../build/release folder starting from user_dir')


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
        print("\n\n############## Processing {} - {}/{} ############\n\n".format(c,i+1,len(categories)))

        ### train
        args.cdir = os.path.join(args.dataset_dir, c)
        files = os.listdir(os.path.join(args.cdir,"4_watertight_scaled"))

        for i in tqdm(files,ncols=50):
            try:
                i=i[:-4]
                os.makedirs(os.path.join(args.cdir,i,"mesh"),exist_ok=True)
                os.rename(os.path.join(args.cdir,"4_watertight_scaled",i+".off"),os.path.join(args.cdir,i,"mesh","mesh.off"))
                args.wdir = os.path.join(args.cdir,i)
                args.i = os.path.join("mesh","mesh.off")
                args.o = os.path.join('scan',str(args.scan_conf))
                os.makedirs(os.path.join(args.wdir,'scan'),exist_ok=True)
                scan_one(args)
            except:
                pass









