import argparse, subprocess, sys, os

import open3d as o3d
import numpy as np
import torch
import pandas as pd


def copy_prediction(args):

    infile = os.path.join("/home/raphael/data/eth/prediction/",args.scene+"_lrtcs_0_"+args.p+".npz")
    outdir = os.path.join(args.user_dir, args.data_dir, args.scene[:-1], "prediction")
    if(not os.path.exists(outdir)):
        os.makedirs(outdir)

    command = ["cp", infile, outdir]

    print("run command: "+str(command))
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    # exit the whole programm if this step didn't work
    if (p.returncode):
        sys.exit(1)
    p.wait()


def eval(working_dir,outfile):

    print("Mesh evaluation...")
    print("\t-of file ", outfile)


    recon_mesh = o3d.io.read_triangle_mesh(outfile)
    recon_points = recon_mesh.sample_points_uniformly(args.n_sample_points)

    gt_points = o3d.io.read_point_cloud(
        os.path.join(working_dir, "ground_truth_sampling", args.scene + "_" + str(args.n_sample_points) + ".xyz"))

    dists1 = recon_points.compute_point_cloud_distance(gt_points)
    dists2 = gt_points.compute_point_cloud_distance(recon_points)


    data.at[args.conf, 'components'] = int(len(recon_mesh.cluster_connected_triangles()[2]))
    # data.at[args.conf, 'closed'] = int(recon_mesh.is_watertight())
    data.at[args.conf, 'orientable'] = int(recon_mesh.is_orientable())
    # data.at[args.conf, 'sintersecting'] = int(recon_mesh.is_self_intersecting())
    data.at[args.conf, 'nm_edges'] = len(recon_mesh.get_non_manifold_edges())
    data.at[args.conf, 'nm_vertices'] = len(recon_mesh.get_non_manifold_vertices())
    data.at[args.conf, 'chamfer'] = (sum(dists1) + sum(dists2)) * 100 / (args.n_sample_points * 75)


    print("\t-components: ", data.at[args.conf, 'components'])
    print("\t-nm-edges: ", data.at[args.conf, 'nm_edges'])
    # print("\t-closed: ", data.at[args.conf, 'components'])
    print("\t-chamfer: {0:.2f}".format(data.at[args.conf, 'chamfer']))

    return data


def poisson(args,data):

    working_dir = os.path.join(args.user_dir,args.data_dir)
    args.scene_conf=args.scene+"_"+str(args.conf)


    outdir = os.path.join(working_dir,"reconstructions","poisson")
    if(not os.path.exists(outdir)):
        os.makedirs(outdir)
    outfile = os.path.join(outdir,args.scene_conf+".ply")

    # run poisson
    command = [args.user_dir + args.poisson_dir + "/PoissonRecon",
               "--in", working_dir+"scans/with_normals/"+args.scene_conf+".ply",
               "--out", outfile,
               "--depth", str(args.depth),
               "--density"]
    print("run command: "+str(command))
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    for line in iter(p.stdout.readline, b''):
        print(line.decode("utf-8")[:-1])
    # exit the whole program if this step didn't work
    if (p.returncode):
        sys.exit(1)

    # run trimmer
    if(args.trim):
        infile=outfile
        outfile=os.path.join(outdir,"trimmed",args.scene_conf+".ply")
        command = [args.user_dir + args.poisson_dir + "/SurfaceTrimmer",
                   "--in", infile,
                   "--trim", str(args.trim),
                   "--out", outfile]
        print("run command: "+str(command))
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        for line in iter(p.stdout.readline, b''):
            print(line.decode("utf-8")[:-1])
        # exit the whole program if this step didn't work
        if (p.returncode):
            sys.exit(1)

    # run evaluation
    if(args.trim):
        infile=os.path.join("trimmed",args.scene_conf+".ply")
    else:
        infile=args.scene_conf+".ply"
    command = [args.user_dir + args.sure_dir + "/eval",
               "-w", outdir,
               "-i", infile,
               "-g", "../../ground_truth_surface/" + args.scene + ".off"]
    print("run command: "+str(command))
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    # exit the whole programm if this step didn't work
    if (p.returncode):
        sys.exit(1)

    # get the stdout output and save it in an array
    # from here: https://stackoverflow.com/questions/18421757/live-output-from-subprocess-command
    output = []
    for line in iter(p.stdout.readline, b''):
        print(line.decode("utf-8")[:-1])
        output.append(line.decode("utf-8")[:-1])
    output = output[-10:-3]
    for i in range(len(output)):
        d = output[i].split(":")
        data.at[args.conf,d[0][2:]]=float(d[1])
        # data.loc[i] = [d[0][2:], float(d[1])]
        # data.loc[i] = float(d[1])
    # data = data.astype({'Values:': int})
    # print(data)
    p.wait()

    return eval(working_dir,outfile)


def onet(args,data):

    working_dir = os.path.join(args.user_dir,args.data_dir)
    args.scene_conf=args.scene+"_"+str(args.conf)

    # run evaluation
    command = [args.user_dir + args.sure_dir + "/sure", "eval",
               "-w", working_dir + "reconstructions/occ/",
               "-i", args.scene_conf+".off",
               "-g", "../../ground_truth_surface/" + args.scene + ".off"]
    # print("run command: "+str(command))
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    # exit the whole programm if this step didn't work
    if (p.returncode):
        sys.exit(1)

    # get the stdout output and save it in an array
    # from here: https://stackoverflow.com/questions/18421757/live-output-from-subprocess-command
    output = []
    for line in iter(p.stdout.readline, b''):
        print(line.decode("utf-8")[:-1])
        output.append(line.decode("utf-8")[:-1])
    output = output[-10:-3]
    for i in range(len(output)):
        d = output[i].split(":")
        data.at[args.conf,d[0][2:]]=float(d[1])
        # data.loc[i] = [d[0][2:], float(d[1])]
        # data.loc[i] = float(d[1])
    # data = data.astype({'Values:': int})
    # print(data)
    p.wait()

    outfile = os.path.join(working_dir,"reconstructions","occ",args.scene_conf+".off")

    return eval(working_dir,outfile)


def sure(args,data):

    working_dir = os.path.join(args.data_dir,args.scene[:-1])

    if(args.sure_method.split(',')[0]=="rt"):
        folder='labatu/'
    else:
        folder='clf/'+args.sure_method.split(',')[0]+'/'
        if('sv' in args.gco):
            folder+='sv/'
        elif('angle' in args.gco):
            folder+='angle/'
        elif('cc' in args.gco):
            folder+='cc/'
        else:
            folder+='no/'

    outfolder = os.path.join(working_dir,'reconstructions',folder)
    if(not os.path.exists(outfolder)):
        os.makedirs(outfolder)

    command = [args.user_dir + args.sure_dir + "/sure", "clf",
               "-w", working_dir,
               "-i", args.scene,
               "-o", "reconstructions/"+folder+args.scene,
               "-g", "gt/"+args.scene[:-1]+"_mesh_cropped"+args.scene[-1]+".off",
               "--gclosed", "0",
               "--p", "prediction/"+args.scene+"_lrtcs_0_"+args.p+".npz",
               "-m", args.sure_method,
               "-s", "omvs",
               "-e", args.export_options,
               "--omanifold", "0",
               "--eval", "0"]
    if(args.gco.split('-')[0] != "no"):
        command += ["--gco", args.gco]

    print("run command: "+str(command))
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    # exit the whole programm if this step didn't work
    if (p.returncode):
        sys.exit(1)
    # get the stdout output and save it in an array
    # from here: https://stackoverflow.com/questions/18421757/live-output-from-subprocess-command
    output = []
    for line in iter(p.stdout.readline, b''):
        print(line.decode("utf-8")[:-1])
        output.append(line.decode("utf-8")[:-1])
    output = output[-10:-3]
    for i in range(len(output)):
        d = output[i].split(":")
        data.at[args.conf,d[0][2:]]=float(d[1])
        # data.loc[i] = [d[0][2:], float(d[1])]
        # data.loc[i] = float(d[1])
    # data = data.astype({'Values:': int})
    # print(data)
    p.wait()

    # add chamfer
    method=args.sure_method.split(',')[0]
    if(args.gco.split('-')[0] != "no"):
        outfile=args.scene+"_"+method+"_"+args.gco.split('-')[1]+"_optimized.ply"
    else:
        outfile = args.scene + "_" + method + "_0_initial.ply"

    # outfile = args.scene_conf + "_" + method + "_0_isomesh.ply"
    outfile = os.path.join(working_dir, "reconstructions", folder, outfile)

    return eval(working_dir,outfile)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')

    # choose method:
    parser.add_argument('mode', type=str, default="clf",
                        help='choose a mode. choices=["download","clf","poisson","onet"]')

    parser.add_argument('--user_dir', type=str, default="/home/raphael/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('--data_dir', type=str, default="/mnt/raphael/ETH3D/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scenes', nargs='+', type=str, default=["anchor"],
                        help='on which scene to execute pipeline.')
    parser.add_argument('-c', '--confs', nargs='+', type=int, default=[0],
                        help='which config file to load')

    # Sure options
    parser.add_argument('--sure_dir', type=str, default="cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure build directory, pointing to .../build/release folder starting from user_dir')
    parser.add_argument('-m','--sure_method', type=str, default="rt,1,labatu,-1",
                        help='the reconstruction method, default: rt,1,labatu')
    parser.add_argument('-p', type=str, default="",
                        help='which prediction. e.g. 9575')
    parser.add_argument('--gco', type=str, default="angle-5.0",
                        help='graph cut optimization type,weight. default: area,1.0')
    parser.add_argument('-e', '--export_options', type=str, default="i",
                        help='graph cut optimization type,weight. default: area,1.0')

    # Poisson options
    parser.add_argument('--poisson_dir', type=str, default="cpp/PoissonReconOri/Bin/Linux",
                        help='Indicate the poisson build directory, starting from user_dir')
    parser.add_argument('--depth', type=int, default=6,
                        help='Poisson depth')
    parser.add_argument('--trim', type=int, default=5,
                        help='Poisson trimming value')

    # whether to download the predictino files
    parser.add_argument('-d', '--copy_prediction', type=int, default=0,
                        help='copy the predictino')
    # wether to upload results
    parser.add_argument('-u', '--upload', type=int, default=1,
                        help='upload the results to google spreadsheet')

    # eval options
    parser.add_argument('--n_sample_points', type=int, default=100000, help='how many points to sample for IoU and Chamfer')

    args = parser.parse_args()

    args.mode = args.mode.split('+')



    if(args.scenes[0] == 'all'):
        args.scenes = ['meadow1','meadow2']


    for scene in args.scenes:
        args.scene = scene
        data = pd.DataFrame(columns=['Scene','Method','gco',
                                     'vertices','nm_vertices','edges','nm_edges','faces','area',
                                     'components','closed','orientable','sintersecting',
                                     'iou','chamfer'])

        if(args.copy_prediction):
            copy_prediction(args)


        if('onet' in args.mode):
            data.at[scene, 'Method'] = 'ConvONet'
            onet(args,data)
        if('poisson' in args.mode):
            if(args.trim):
                data.at[scene, 'Method'] = 'Poisson_trimmed_'+str(args.trim)
            else:
                data.at[scene, 'Method'] = 'Poisson'
            data.at[scene, 'gco'] = args.depth
            poisson(args,data)
        if('clf' in args.mode):
            data.at[scene, 'Method'] = args.sure_method+"_"+args.p
            data.at[scene, 'gco'] = args.gco
            sure(args,data)

        if(args.upload):
            upload(data, args)