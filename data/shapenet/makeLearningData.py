import argparse, subprocess, sys, os
import configparser
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'reconbench'))
# from RunSampler import runUniform
from random import randrange

import json
import glob

def makeImplicit(args):
    # make a ground truth mesh to an implicit, which is needed for the scanning with reconbench scanner
    print("############## Make Implicit:")

    outdir = os.path.join(args.class_dir,'mpu')
    if(not os.path.exists(outdir)):
        os.makedirs(outdir)

    outfile = os.path.join(outdir, args.scene + ".mpu")
    if(os.path.isfile(outfile) and not args.overwrite):
        print("exists!")
        return


    command = [args.user_dir + args.reconbench_dir + "/mesh_to_implicit",
               os.path.join(args.class_dir, '2_watertight', args.scene + ".off"),
               outfile,
               "6", "0.009", "1.1"
               ]
    print("Run command: ", command)
    p = subprocess.Popen(command)
    p.wait()

def isosurface(args):
    # isosurface an mpu; needed to have a ground truth for features extraction, because it is faster than loading mpu
    print("############## Isosurface:")

    indir = os.path.join(args.class_dir,'mpu')
    outdir = os.path.join(args.class_dir,'isosurface')
    if(not os.path.exists(outdir)):
        os.makedirs(outdir)

    outfile = os.path.join(outdir, args.scene + ".off")
    if(os.path.isfile(outfile) and not args.overwrite):
        print("exists!")
        return
        # TODO: try to open the .off file (e.g. with open3d) and if it cannot be opened, redo it


    command = [args.user_dir + args.reconbench_dir + "/isosurface",
               os.path.join(indir, args.scene + ".mpu"),
               "512",
               outfile
               ]
    print("Run command: ", command)
    p = subprocess.Popen(command)
    p.wait()


def scanShape(args):
    # make the scans
    print("############## Scan:")

    if (args.normal_type == 1):
        outdir = os.path.join(args.class_dir,"scans","with_normal")
    elif (args.normal_type == 4):
        outdir = os.path.join(args.class_dir,"scans","with_sensor")
    if(not os.path.exists(outdir)):
        os.makedirs(outdir)

    outfile = os.path.join(outdir,args.scene_conf+'.ply')
    if(os.path.isfile(outfile) and not args.overwrite):
        print("exists!")
        return

    config = configparser.ConfigParser()
    conf_path = os.path.join(args.data_dir,"..","confs", "bumps_" + str(args.conf) + ".cnf")
    print("Read scanning configuration from: " + conf_path)
    config.read(conf_path)
    infile = os.path.join(args.class_dir,"mpu",args.scene + ".mpu")

    sensor_file = "asd"
    pathdir = os.path.join(args.user_dir, args.reconbench_dir)

    command = []

    command.append(pathdir + "/" + config.get("uniform", "exec_name"))
    command.append(pathdir)
    command.append(infile)
    command.append(outfile)
    command.append(sensor_file)

    # required
    command.append(config.get("uniform", "camera_res_x"))
    command.append(config.get("uniform", "camera_res_y"))
    command.append(config.get("uniform", "scan_res"))

    # optional
    if config.has_option("uniform", "min_range"):
        command.append("min_range")
        command.append(config.get("uniform", "min_range"))

    if config.has_option("uniform", "max_range"):
        command.append("max_range")
        command.append(config.get("uniform", "max_range"))

    if config.has_option("uniform", "num_stripes"):
        command.append("num_stripes")
        command.append(config.get("uniform", "num_stripes"))

    if config.has_option("uniform", "laser_fov"):
        command.append("laser_fov")
        command.append(config.get("uniform", "laser_fov"))

    if config.has_option("uniform", "peak_threshold"):
        command.append("peak_threshold")
        command.append(config.get("uniform", "peak_threshold"))

    if config.has_option("uniform", "std_threshold"):
        command.append("std_threshold")
        command.append(config.get("uniform", "std_threshold"))

    if config.has_option("uniform", "additive_noise"):
        command.append("additive_noise")
        command.append(config.get("uniform", "additive_noise"))

    if config.has_option("uniform", "outlier_percentage"):
        command.append("outlier_percentage")
        command.append(config.get("uniform", "outlier_percentage"))

    if config.has_option("uniform", "laser_smoother"):
        command.append("laser_smoother")
        command.append(config.get("uniform", "laser_smoother"))

    if config.has_option("uniform", "registration_error"):
        command.append("registration_error")
        command.append(config.get("uniform", "registration_error"))

    # if config.has_option("uniform", "normal_type"):
    #         command.append("normal_type")
    #         command.append(config.get("uniform", "normal_type"))
    command.append("normal_type")
    command.append(str(args.normal_type))

    if config.has_option("uniform", "pca_knn"):
        command.append("pca_knn")
        command.append(config.get("uniform", "pca_knn"))

    if config.has_option("uniform", "random_sample_rotation"):
        command.append("random_sample_rotation")
        command.append(config.get("uniform", "random_sample_rotation"))

    print(command)
    subprocess.check_call(command)

def extractFeatures(args):
    # extract features from mpu
    print("############## Extract Features:")

    outfile = os.path.join(args.class_dir,"gt",args.scene_conf+'_lrtcs_0_labels.txt')
    if(os.path.isfile(outfile) and not args.overwrite):
        print("exists!")
        return

    command = [args.user_dir + args.sure_dir + "/sure",
               "-w", args.class_dir,
               "-i", "scans/with_sensor/"+args.scene_conf+".ply",
               "-o", args.scene_conf,
               "-m", "lrtcs,100",
               "-s", "lidar",
               "--gco", "angle-0.5",
               "-g", "isosurface/" + args.scene + ".off",
               "-e", ""]
    p = subprocess.Popen(command)
    p.wait()

# def files(path):
#     for file in os.listdir(path):
#         if os.path.isfile(os.path.join(path, file)):
#             yield file
#
# def move(args):
#
#     for i in range(0,5):
#         gt_path = os.path.join(args.class_dir, "gt", str(i))
#         command = ["rm", "-rf", gt_path]
#         print(command)
#         p = subprocess.Popen(command)
#         p.wait()
#
#
#     # for i in range(0,5):
#     #     inpath = os.path.join(args.class_dir, "gt",str(i))
#     #     for f in files(inpath):
#     #         # conf=f.split('_')[1][0]
#     #         outpath = os.path.join(args.class_dir, "gt")
#     #         command = ['mv', os.path.join(inpath, f), outpath]
#     #         print(command)
#     #         p = subprocess.Popen(command)
#     #         p.wait()

def scanMine(args):

    # choose random scan parameters
    n_points = str(randrange(750,75000))
    n_cameras = str(randrange(3,8))
    std_noise = str(randrange(0,50)/100)
    perc_outliers = str(randrange(0,15)/100)



    outfile = os.path.join(args.class_dir,"gt",args.scene+'_lrtcs_0_labels.txt')
    if(os.path.isfile(outfile) and not args.overwrite):
        print("exists!")
        return

    if(not os.path.exists(os.path.join(args.class_dir,"scans","mine"))):
        os.makedirs(os.path.join(args.class_dir,"scans","mine"))

    infofile = open(os.path.join(args.class_dir,"scans","mine",args.scene+"_info.txt"),"w+")
    infofile.write("n_points: "+n_points+"\n")
    infofile.write("n_cameras: "+n_cameras+"\n")
    infofile.write("std_noise: "+std_noise+"\n")
    infofile.write("perc_outliers: "+perc_outliers+"\n")
    infofile.close()

    # extract features from mpu
    command = [args.user_dir + args.sure_dir + "/sure",
               "-w", args.class_dir,
               "-i", args.scene,
               "-o", "scans/mine/"+args.scene,
               "-m", "lrtcs,100",
               "-g", "isosurface/" + args.scene + ".off",
               "--gco", "angle-0.5",
               "-s", "scan,"+n_points+","+n_cameras+","+std_noise+","+perc_outliers,
               "-e", "xp"]
    print("run command: ", command)
    p = subprocess.Popen(command)
    p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')

    parser.add_argument('mode', type=str, default="scan",
                        help='what to do. '+'-separated list of choices=["implicit", "isosurface", "myscan", "scan", "features"]')
    parser.add_argument('--user_dir', type=str, default="/home/raphael/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="/mnt/raphael/ProcessedShapeNet/ShapeNet.build/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-c', '--confs', nargs='+', type=int, default=[0],
                        help='which config file to load')
    parser.add_argument('-n', '--number_of_shapes', type=int, default=1,
                        help='how many shapes')
    parser.add_argument('--overwrite', type=int, default=0,
                        help='overwrite existing files')

    parser.add_argument('--reconbench_dir', type=str, default="cpp/reconbench-CMake/build/release",
                        help='Indicate the openMVS binary directory, pointing to .../bin folder starting from user_dir')
    parser.add_argument('--normal_type', type=int, choices=[1, 4], default=4,
                        help='1 = pca normals; 4 = sensor position')

    parser.add_argument('--sure_dir', type=str, default="cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure build directory, pointing to .../build/release folder starting from user_dir')

    parser.add_argument('--classes', type=str, default="1-13",
                        help='Indicate the classes')

    args = parser.parse_args()

    args.mode = args.mode.split('+')


    if (args.confs[0] == -1):
        args.confs = [0, 1, 2, 3, 4]

    cl = os.listdir(args.data_dir)
    if(len(args.classes.split('-'))>1):
        classes = cl[int(args.classes.split('-')[0])-1:int(args.classes.split('-')[1])-1]
    else:
        classes = [cl[int(args.classes.split('-')[0])-1]]
    # classes = [:]

    with open(os.path.join(args.data_dir,"..","ShapeNetCore.v1","taxonomy.json")) as json_file:
        taxonomy = json.load(json_file)


    for ic in range(len(classes)):

        c = classes[ic]
        for si in taxonomy:
            if(si['synsetId']==c):
                args.class_name = si['name']
                args.class_name = args.class_name.split(',')[0]

        args.class_dir=os.path.join(args.data_dir,c)
        args.scenes = os.listdir(os.path.join(args.class_dir,"2_watertight"))
        args.scenes = args.scenes[:args.number_of_shapes]

        # if("move" in args.mode):
        #     move(args)

        for i,scene in enumerate(args.scenes):

            print("############## Processing class {} - {} {}/{}, Shape Nr {}/{} ##############".format(c,args.class_name,ic+1,len(classes),i+1,len(args.scenes)))
            args.scene = scene[:-4]

            if ("implicit" in args.mode):
                makeImplicit(args)

            if ("isosurface" in args.mode):
                isosurface(args)


            if("myscan" in args.mode):
                scanMine(args)

            if("scan" in args.mode or "features" in args.mode):

                for conf in args.confs:
                    print("############## Processing class {} - {} {}/{}, Shape Nr {}/{}, Congig {}/{} ##############".format(c,
                                                                                                                args.class_name,
                                                                                                                ic + 1, len(
                            classes), i + 1, len(args.scenes), conf + 1, len(args.confs)))

                    args.conf = conf
                    # input_file = args.working_dir+args.scene+args.input_file_extension
                    args.scene_conf = args.scene + "_" + str(args.conf)

                    if ("scan" in args.mode):
                        scanShape(args)

                    if ("features" in args.mode):
                        extractFeatures(args)