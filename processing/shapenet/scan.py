import argparse, subprocess, os, random, sys



def scan(args):

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
    elif(args.conf == 4): # convonet configuration, 10 cameras
        points = 3000
        cameras = 10
        noise = 0.005
        outliers = 0.0
    else:
        print("\nERROR: not a valid config. choose [0,1,2]")
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



    parser.add_argument('--category', type=str, default=None,
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
        # if c.startswith('.'):
        #     continue
        print("\n############## Processing {}/{} ############\n".format(i+1,len(categories)))

        ### train
        args.wdir = os.path.join(args.dataset_dir, c, "2_watertight", "train")
        args.input = os.listdir(args.wdir)
        args.outdir0 = os.path.join(args.dataset_dir, c, '3_scan', 'train','0')
        if not os.path.exists(args.outdir0):
            os.makedirs(args.outdir0)
        args.outdir1 = os.path.join(args.dataset_dir, c, '3_scan', 'train','1')
        if not os.path.exists(args.outdir1):
            os.makedirs(args.outdir1)
        args.outdir2 = os.path.join(args.dataset_dir, c, '3_scan', 'train','2')
        if not os.path.exists(args.outdir2):
            os.makedirs(args.outdir2)
        args.outdir3 = os.path.join(args.dataset_dir, c, '3_scan', 'train','3')
        if not os.path.exists(args.outdir3):
            os.makedirs(args.outdir3)

        for i in args.input:
            args.i = i
            # args.conf = random.randint(0,2)
            args.conf = 3
            args.o = os.path.join('..','..','3_scan','train',str(args.conf),i[:-4])
            scan(args)

        ### test
        args.wdir = os.path.join(args.dataset_dir, c, "2_watertight", "test")
        args.input = os.listdir(args.wdir)
        args.outdir0 = os.path.join(args.dataset_dir, c, '3_scan', 'test','0')
        if not os.path.exists(args.outdir0):
            os.makedirs(args.outdir0)
        args.outdir1 = os.path.join(args.dataset_dir, c, '3_scan', 'test','1')
        if not os.path.exists(args.outdir1):
            os.makedirs(args.outdir1)
        args.outdir2 = os.path.join(args.dataset_dir, c, '3_scan', 'test','2')
        if not os.path.exists(args.outdir2):
            os.makedirs(args.outdir2)
        args.outdir3 = os.path.join(args.dataset_dir, c, '3_scan', 'test','3')
        if not os.path.exists(args.outdir3):
            os.makedirs(args.outdir3)

        for i in args.input:
            for conf in range(3,4):
                args.i = i
                args.conf = conf
                args.o = os.path.join('..', '..', '3_scan','test',str(args.conf), i[:-4])
                scan(args)










