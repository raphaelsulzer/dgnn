import argparse, datetime
import pandas as pd
import shutil
import os, sys, math
import subprocess
import numpy as np

from colmap2MVS import colmap2mvs
from densifyCloud import densify
from reconstructMesh import reconstruct
from cleanMesh import clean
from refineMesh import refine
from sampleMesh import sample
from runSure import sure
from evaluateExtrinsics import extrinsics
from evaluateIntrinsics import intrinsics
from uploadEthResults import gcUpload
from textureMesh import texture
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', ''))


def removeDir(path):

    if(os.path.exists(path)):
        shutil.rmtree(path)


def pipeline(args):

    data_list = []

    if ("i" in args.pipeline_steps):
        colmap2mvs(args)

    if ("d" in args.pipeline_steps):
        densify(args)
        args.method = "point_cloud"
        data_pc = extrinsics(args)
        data_pc.name = args.method
        data_pc.insert(0, args.scene, data_pc.name, True)
        data_list.append(data_pc)

    ## feature extraction
    if("f" in args.pipeline_steps):
        args.sure_method = "lrt,1"
        args.method = args.sure_method
        sure(args)
        return pd.DataFrame()


    if ("m" in args.pipeline_steps):
        args.clean_mesh = 0
        args.method = "omvs"
        args.input_file = "densify_file.mvs"
        args.mesh_file_name = reconstruct(args)
        mesh_intrinsics = intrinsics(args)
        # surface_area = mesh_intrinsics["Values:"][3]
        # args.n_sample_points*=surface_area
        sample(args)
        mesh_extrinsics = extrinsics(args)

        # mesh_extrinsics.name = args.mesh_file_name
        mesh_extrinsics.insert(0, args.scene, args.mesh_file_name, True)
        mesh_data = pd.concat((mesh_extrinsics, mesh_intrinsics), axis=1)
        mesh_data=mesh_data.fillna(0)
        data_list.append(mesh_data)

    if ("n" in args.pipeline_steps):
        args.clean_mesh = 1
        args.method = "omvs_cleaned"
        args.input_file = "densify_file.mvs"

        args.mesh_file_name = reconstruct(args)

        mesh_intrinsics = intrinsics(args)
        # surface_area = mesh_intrinsics["Values:"][3]
        # args.n_sample_points*=surface_area
        sample(args)
        mesh_extrinsics = extrinsics(args)

        mesh_extrinsics.insert(0, args.scene, args.mesh_file_name, True)
        mesh_data = pd.concat((mesh_extrinsics, mesh_intrinsics), axis=1)
        mesh_data=mesh_data.fillna(0)
        data_list.append(mesh_data)

        if ("t" in args.pipeline_steps):
            texture(args)

    if ("r" in args.pipeline_steps):
        args.refine=1
        args.method = "omvs"

        if(args.clean_mesh):
            args.method+="_clean"

        args.mesh_file_name = refine(args)
        mesh_intrinsics = intrinsics(args)
        # surface_area = mesh_intrinsics["Values:"][3]
        # args.n_sample_points*=surface_area
        sample(args)
        mesh_extrinsics = extrinsics(args)

        mesh_extrinsics.insert(0, args.scene, args.mesh_file_name, True)
        mesh_data = pd.concat((mesh_extrinsics, mesh_intrinsics), axis=1)
        mesh_data = mesh_data.fillna(0)
        data_list.append(mesh_data)

    ## run sure, sample and evaluate result
    if ("x" in args.pipeline_steps):
        args.clean_mesh = 0
        args.method = "clf"
        sure(args) # input file is automatically densify file; method is args.sure_method, e.g. cl,lrt_2i9e
        if(args.gco):
            args.regs = args.gco.split(',')
            args.reg_weight=0.0
            for i,reg in enumerate(args.regs):
                args.reg_weight+=float(args.regs[i].split('-')[1])
            args.reg_weight = str(round(args.reg_weight, 5 - int(math.floor(math.log10(abs(args.reg_weight)))) - 1))
        else:
            args.reg_weight = "0"

        if ("e" in args.pipeline_steps):
            mesh_intrinsics = intrinsics(args)
            mesh_extrinsics = extrinsics(args)

            mesh_extrinsics.name = args.sure_method+"_"+args.pi+"_"+args.gco
            mesh_extrinsics.insert(0, args.scene, mesh_extrinsics.name, True)
            mesh_data = pd.concat((mesh_extrinsics, mesh_intrinsics), axis=1)
            mesh_data=mesh_data.fillna(0)
            data_list.append(mesh_data)

    ## run sure, clean result with OMVS, sample and evaluate result
    if ("y" in args.pipeline_steps):
        args.clean_mesh = 1
        args.method = "clf_cleaned"
        sure(args)
        if(args.gco):
            args.regs = args.gco.split(',')
            args.reg_weight=0
            for i,reg in enumerate(args.regs):
                args.reg_weight+=float(args.regs[i].split('-')[1])
            args.reg_weight = str(round(args.reg_weight, 5 - int(math.floor(math.log10(abs(args.reg_weight)))) - 1))
        else:
            args.reg_weight = "0"

        if ("e" in args.pipeline_steps):
            mesh_intrinsics = intrinsics(args)
            # surface_area = mesh_intrinsics["Values:"][3]
            # args.n_sample_points*=surface_area
            sample(args)
            mesh_extrinsics = extrinsics(args)
            mesh_extrinsics.name = args.sure_method+"_"+args.pi+"_"+args.gco+"_cleaned"
            mesh_extrinsics.insert(0, args.scene, mesh_extrinsics.name, True)
            mesh_data = pd.concat((mesh_extrinsics, mesh_intrinsics), axis=1)
            mesh_data = mesh_data.fillna(0)
            data_list.append(mesh_data)

        if ("t" in args.pipeline_steps):
            texture(args)


    if("p" in args.pipeline_steps):
        args.clean_mesh=0
        args.method = 'poisson'
        # run poisson
        working_dir = os.path.join(args.data_dir, args.scene)
        if(not os.path.exists(os.path.join(working_dir,"poisson"))):
            os.makedirs(os.path.join(working_dir,"poisson"))

        outfile = os.path.join(working_dir, "poisson", args.scene + "_" + str(args.bType) + ".ply")

        # run poisson
        command = [args.user_dir + args.poisson_dir + "/PoissonRecon",
                   "--in", os.path.join(working_dir,"openMVS", "densify_file.ply"),
                   "--out", outfile,
                   "--bType", str(args.bType),
                   "--depth", str(args.depth)]
        print("run command: " + str(command))
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        for line in iter(p.stdout.readline, b''):
            print(line.decode("utf-8")[:-1])
        # exit the whole program if this step didn't work
        if (p.returncode):
            sys.exit(1)


        args.reg_weight=str(args.depth)

        mesh_intrinsics = intrinsics(args)
        # surface_area = mesh_intrinsics["Values:"][3]
        # args.n_sample_points*=surface_area
        sample(args)
        mesh_extrinsics = extrinsics(args)
        mesh_extrinsics.name = args.method+"_"+str(args.depth)
        mesh_extrinsics.insert(0, args.scene, mesh_extrinsics.name, True)
        mesh_data = pd.concat((mesh_extrinsics, mesh_intrinsics), axis=1)
        mesh_data = mesh_data.fillna(0)
        data_list.append(mesh_data)


    if("q" in args.pipeline_steps):
        args.clean_mesh=1
        args.method = 'poisson_cleaned'

        clean(args)

        mesh_intrinsics = intrinsics(args)
        # surface_area = mesh_intrinsics["Values:"][3]
        # args.n_sample_points*=surface_area
        sample(args)
        mesh_extrinsics = extrinsics(args)
        mesh_extrinsics.name = args.method+"_"+str(args.depth)
        mesh_extrinsics.insert(0, args.scene, mesh_extrinsics.name, True)
        mesh_data = pd.concat((mesh_extrinsics, mesh_intrinsics), axis=1)
        mesh_data = mesh_data.fillna(0)
        data_list.append(mesh_data)

        if ("t" in args.pipeline_steps):
            texture(args)

    # ## extrinsics
    # if("e" in args.pipeline_steps):
    #
    #     if (args.gco):
    #         args.regs = args.gco.split(',')
    #         args.reg_weight = 0
    #         for i, reg in enumerate(args.regs):
    #             args.reg_weight += float(args.regs[i].split('-')[1])
    #         args.reg_weight = str(round(args.reg_weight, 5 - int(math.floor(math.log10(abs(args.reg_weight)))) - 1))
    #     else:
    #         args.reg_weight = "0"
    #     extrinsics(args)
    #     return pd.DataFrame()

    df = pd.concat(data_list)
    df = df.fillna(0)

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='colmap_openMVS_reconstruction')

    parser.add_argument('--user_dir', type=str, default="/home/raphael/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="/mnt/raphael/ETH3D/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scenes', nargs = '+', type=str, default=["all"],
                        help='on which scene to execute pipeline.')

    parser.add_argument('-c', '--clear', type=int, default=0,
                        help='delete the openMVS folder, and thus recompute everything.')
    parser.add_argument('-u', '--upload', type=int, default=1,
                        help='upload the results to google spreadsheet')

    parser.add_argument('-p', '--pipeline_steps', type=str, default='xy',
                        help='pipeline steps. default: idmr. extra options: sampling = s, evaluation = e')

    # additional Densify options:
    parser.add_argument('--openMVS_dir', type=str, default="cpp/openMVS_release/bin",
                        help='Indicate the openMVS binary directory, pointing to .../bin folder starting from user_dir')
    parser.add_argument('--resolution_level', type=int, default=4,
                        help='how many times to scale down the images before point cloud computation')
    parser.add_argument('--filter_point_cloud', type=int, default=0,
                        help='filter dense point-cloud based on visibility')

    # additional Mesh reconstruction options:
    parser.add_argument('--min_point_distance', type=float, default=0.0,
                        help='minimum distance in pixels between the projection'
                             ' of two 3D points to consider them different while triangulating (0 -disabled)')
    parser.add_argument('--free_space_support', type=int, default=1,
                        help='free space suppport, 0 = off (Labatu), 1 = on (Jancosek)')
    parser.add_argument('--clean_mesh', type=int, default=1,
                        help='enable/disable all mesh clean options. default: disabled.')
    parser.add_argument('--refine', type=int, default=0,
                        help='refine mesh. default: disabled.')

    # additional SURE reconstruction options:
    parser.add_argument('-m','--sure_method', type=str, default="cl,sm",
                        help='the m, method parameter for sure. default: cl,sm')
    parser.add_argument('--pi', type=str, default="9575",
                        help='which prediction')

    parser.add_argument('-em', '--method', type=str, default="clf",
                        help='use only for evaluation procedure')
    parser.add_argument('--adt', type=float, default=0.0,
                        help='epsilon for adaptive delaunay triangulation. default: 0.0')
    parser.add_argument('--gco', type=str, default="",
                        help='graph cut optimization type,weight. default: cc-0.0,area-0.0,angle-0.0')
    parser.add_argument('--lff', type=int, default=0,
                        help="Factor Y to determine surface area"
                                          "threshold X for removing large facets,"
                                          "with X = Y * mean_surface_area. default: 0")
    parser.add_argument('--nsc', type=int, default=0,
                        help="Number of surface mesh components to keep")

    # additional sampling options:
    parser.add_argument('--sure_dir', type=str, default="cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure release dir starting from user_dir')
    parser.add_argument('--output_sampling', type=str, default="as,900",
                        help='sure sample output parameter. default: as,900')

    # Poisson options
    parser.add_argument('--poisson_dir', type=str, default="cpp/PoissonReconOri/Bin/Linux",
                        help='Indicate the poisson build directory, starting from user_dir')
    parser.add_argument('--depth', type=int, default=6,
                        help='Poisson depth')
    parser.add_argument('--bType', type=int, default=3,
                        help='Poisson bType value:\n1] free\n2] Dirichlet\n3] Neumann')



    # additional evaluation options:
    parser.add_argument('--eval_tool_dir', type=str, default="cpp/multi-view-evaluation/build",
                        help='Indicate the eval tool dir')
    parser.add_argument('-t', '--tolerances', type=str, default="0.01,0.02,0.05,0.1,0.2,0.5",
                        help='tolerances to evaluate. default: "0.01,0.02,0.05,0.1,0.2,0.5"')
    parser.add_argument('-ecc','--export_colored_clouds', type=int, default=0,
                        help='should colored completeness and accuracy clouds be exported')

    args = parser.parse_args()

    if(args.scenes[0] == 'all'):
        args.scenes = []
        # for scene in os.listdir(args.user_dir+args.data_dir):
        for scene in os.listdir(args.data_dir):
            if (not scene[0] == "x"):
                args.scenes+=[scene]


    # The ID and range of a sample spreadsheet.
    # gc_results
    args.spreadsheet_id = '1K-HStDyVj199-LLJ04PwX26ma7XzEDd7N_OU5vi4xQQ'
    # classification_results

    # scenes_data_list = []
    all =pd.DataFrame(np.zeros((12,5)),columns=["Tolerances:",  "Completenesses:",  "Accuracies:",  "F1-scores:", "Values:"])
    for i,scene in enumerate(args.scenes):
        if(args.clear):
            # removeDir(args.user_dir+args.data_dir+scene+"/openMVS/")
            removeDir(args.data_dir+scene+"/openMVS/")

        args.scene = scene
        # if(scene=="facade"):
        #     continue
        print("\n#####################################################################")
        print("############ 0. Process scene: {} ({}/{})############".format(args.scene, i+1, len(args.scenes)))
        print("#####################################################################")
        print(datetime.datetime.now())
        scene_data = pipeline(args)
        scene_data.name = args.scene
        # scenes_data_list.append(scene_data)
        scene_data.reset_index(drop=True, inplace=True)
        all+=scene_data
        if(args.upload):
            gcUpload(scene_data, args)


    print(all.div(len(args.scenes)))

