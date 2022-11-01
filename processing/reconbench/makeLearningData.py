import argparse, subprocess, os
import configparser

# from pytorch3d.utils import ico_sphere
# from pytorch3d.io import load_obj, load_ply
# from pytorch3d.structures import Meshes, Pointclouds
# from pytorch3d.ops import sample_points_from_meshes
# from pytorch3d.loss import chamfer_distance

from random import randrange

def makeImplicit(args):
    # make a ground truth mesh to an implicit, which is needed for the scanning with reconbench scanner
    command = [args.user_dir + args.reconbench_dir + "/mesh_to_implicit",
               os.path.join(args.working_dir, 'surface_input', args.scene + ".off"),
               os.path.join(args.working_dir, 'mpu', args.scene + ".mpu"),
               "6", "0.009", "1.1"
               ]
    print("Run command: ", command)
    p = subprocess.Popen(command)
    p.wait()



def scanShape(args):
    # make the scans
    config = configparser.ConfigParser()
    conf_path = args.working_dir + "confs/bumps_" + str(args.conf) + ".cnf"
    print("Read scanning configuration from: " + conf_path)
    config.read(conf_path)
    infile = args.working_dir + "mpu/" + args.scene + ".mpu"
    if (args.normal_type == 1):
        outfile = args.working_dir + "scans/with_normals/" + args.scene_conf + ".ply"
    elif (args.normal_type == 4):
        outfile = args.working_dir + "scans/with_sensor/" + args.scene_conf + ".ply"

    sensor_file = args.working_dir + "scans/" + args.scene_conf + "_sensor"
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

    print(*command)
    subprocess.check_call(command)


def extractFeatures(args):
    # extract features from mpu
    command = [args.user_dir + args.sure_dir + "/sure", "clf",
               "-w", args.working_dir,
               "-i", "/scans/with_sensor/"+args.scene_conf,
               "-o", args.scene_conf,
               "-m", "lrtcs,100", "-s", "lidar",
               "-g", "ground_truth_surface/" + args.scene + ".off",
               "--gco", "angle-0.5",
               "-e", ""]
    p = subprocess.Popen(command)
    p.wait()


def isosurface(args):
    # isosurface an mpu; needed to have a ground truth for features extraction, because it is faster than loading mpu
    command = [args.user_dir + args.reconbench_dir + "/isosurface",
               args.working_dir + "mpu/" + args.scene + ".mpu",
               str(args.resolution),
               args.working_dir + "ground_truth_surface/" + args.scene + ".off"

               ]
    print("Run command: ", command)
    p = subprocess.Popen(command)
    p.wait()


def uploadData(args):
    command = ['scp', args.working_dir+"gt/*", "enpc:/home/raphael/data/reconbench/gt/"]
    print("Run command: ", command)
    p = subprocess.Popen(command)
    p.wait()



def scanMine(args):

    # choose random scan parameters
    n_points = str(randrange(750,75000))
    n_cameras = str(randrange(3,8))
    std_noise = str(randrange(0,40)/100)
    perc_outliers = str(randrange(0,15)/100)


    # extract features from mpu
    command = [args.user_dir + args.sure_dir + "/sure", "clf",
               "-w", args.working_dir,
               "-i", "/scans/mine/"+args.scene_conf,
               "-o", args.scene_conf,
               "-m", "lrtcs,100", "-s", "lidar",
               "-g", "ground_truth_surface/" + args.scene + ".off",
               "--gco", "angle-0.5",
               "s", "scan,"+n_points+","+n_cameras+","+std_noise+","+perc_outliers,
               "-e", ""]
    p = subprocess.Popen(command)
    p.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')

    parser.add_argument('mode', type=str, default="scan",
                        help='what to do. '+'-separated list of choices=["implicit", "isosurface", "scan", "features"]')
    parser.add_argument('--user_dir', type=str, default="/home/adminlocal/PhD/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="data/reconbench/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scenes', nargs='+', type=str, default=["Ship"],
                        help='on which scene to execute pipeline.')
    parser.add_argument('-c', '--confs', nargs='+', type=int, default=[0],
                        help='which config file to load')
    parser.add_argument('--resolution', type=int,default=512,
                        help="Isosurface resolution")

    parser.add_argument('--reconbench_dir', type=str, default="cpp/reconbench-CMake/build/release",
                        help='Indicate the reconbench directory, pointing to release folder starting from user_dir')
    parser.add_argument('--normal_type', type=int, choices=[1, 4], default=4,
                        help='1 = pca normals; 4 = sensor position')

    parser.add_argument('--sure_dir', type=str, default="cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure build directory, pointing to .../build/release folder starting from user_dir')

    args = parser.parse_args()

    args.working_dir = args.user_dir + args.data_dir

    args.mode = args.mode.split('+')


    if (args.confs[0] == -1):
        args.confs = [0, 1, 2, 3, 4]

    if (args.scenes[0] == 'all'):
        args.scenes = ['anchor', 'gargoyle', 'dc', 'daratech', 'lordquas']


    for scene in args.scenes:

        args.scene = scene

        if("upload" in args.mode):
            uploadData(args)

        if ("implicit" in args.mode):
            makeImplicit(args)

        if ("isosurface" in args.mode):
            isosurface(args)

        for conf in args.confs:
            args.conf = conf
            # input_file = args.working_dir+args.scene+args.input_file_extension
            args.scene_conf = args.scene + "_" + str(args.conf)

            if ("scan" in args.mode):
                scanShape(args)

            if ("features" in args.mode):
                extractFeatures(args)