import argparse, subprocess, os, random, sys
from tqdm import tqdm


def main(args):

    # choose random scan parameters

    if(args.conf == 0):
        points = 12000
        cameras = 15
        noise = 0.0
        outliers = 0.0
    elif(args.conf == 1):
        points = 3000
        cameras = 15
        noise = 0.0025
        outliers = 0.0
    elif(args.conf == 2):
        points = 12000
        cameras = 15
        noise = 0.005
        outliers = 0.33
    elif(args.conf == 3): # convonet configuration, 50 cameras
        points = 3000
        cameras = 50
        noise = 0.005
        outliers = 0.0
    elif(args.conf == 4): # convonet configuration, 10 cameras, same as 42 but oriented normals with sensor
        points = 3000
        cameras = 10
        noise = 0.005
        outliers = 0.0
    elif(args.conf == 42): # same as 4 but adding ground truth normals to the npz file
        points = 3000
        cameras = 10
        noise = 0.005
        outliers = 0.0
    elif(args.conf == 43): # same as 42 but orienting the normals with mst instead of sensor
        points = 3000
        cameras = 10
        noise = 0.005
        outliers = 0.0
    else:
        print("\nERROR: not a valid config. choose [0,1,2,3,4]")
        sys.exit(1)

    outfile = os.path.join(args.wdir,args.o+".npz")
    if(os.path.isfile(outfile) and not args.overwrite):
        print("exists!")
        return

    # infofile = open(os.path.join(args.class_dir,"scans","mine",args.scene+"_info.txt"),"w+")
    # infofile.write("n_points: "+n_points+"\n")
    # infofile.write("n_cameras: "+n_cameras+"\n")
    # infofile.write("std_noise: "+std_noise+"\n")
    # infofile.write("perc_outliers: "+perc_outliers+"\n")
    # infofile.close()

    # extract features from mpu
    command = [args.sure_dir + "/scan",
               "-w", str(args.wdir),
               "-i", str(args.i),
               "-o", str(args.o),
               "--noise", str(noise),
               "--outliers", str(outliers),
               "--points", str(points),
               "--cameras", str(cameras),
               "--normal_method", "jet",
               "--normal_neighborhood", str(8),
               "--normal_orient", str(2),
               "--export", "npz"]
    print("run command: ", command)
    p = subprocess.Popen(command)
    p.wait()






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')


    parser.add_argument('-d', '--dataset_dir', type=str, default="/mnt/raphael/ModelNet10/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('--overwrite', type=int, default=0,
                        help='overwrite existing files')
    parser.add_argument('--sure_dir', type=str, default="/home/raphael/cpp/surfaceReconstruction/build/release",
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
    # if 'x' in categories:
    #     categories.remove('x')



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
        args.wdir = os.path.join(args.dataset_dir, c, "2_watertight")
        args.input = os.listdir(args.wdir)
        args.outdir = os.path.join(args.dataset_dir, c, '3_scan', str(args.conf))
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        for i in args.input:
            args.i = i
            args.o = os.path.join('..','3_scan',str(args.conf),i[:-4])
            main(args)











