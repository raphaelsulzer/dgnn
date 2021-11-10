import argparse, subprocess, sys, os
import open3d as o3d
import numpy as np

import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = "Times New Roman"
plt.rcParams['mathtext.it'] = "Times New Roman:italic"
plt.rcParams['mathtext.bf'] = "Times New Roman:bold"


def arrange_2(args):

    glob_image_dir = os.path.join(args.user_dir, args.data_dir, args.scene, 'images')

    args.methods = ['ground_truth', 'image', 'input', 'clf_textured',
                    'poisson_2', 'labatu_2', 'jancosek_2', 'clf_2',
                    'poisson', 'labatu', 'jancosek', 'clf']

    # fig,axes = plt.subplots(nrows=3, ncols=4, figsize=(6.875, 4.5), dpi=200)
    fig,axes = plt.subplots(nrows=3, ncols=4, figsize=(13.75, 9), dpi=300)

    images=[]
    for method in args.methods:
        im_file = os.path.join(glob_image_dir, method, args.scene+'.png')
        images.append(plt.imread(im_file))

    # plt.rcParams['text.usetex'] = True
    # # plt.rcParams['font.family'] = "serif"
    # # plt.rcParams['font.serif'] = "Times"
    # plt.rcParams['font.family'] = 'Liberation Serif'

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = "Times New Roman"
    plt.rcParams['mathtext.it'] = "Times New Roman:italic"
    plt.rcParams['mathtext.bf'] = "Times New Roman:bold"


    name="(b) Image of $\it{delivery}$."
    # name="(a) Image of $\it{"+args.scene+"}$."
    titles=["(a) Ground truth.", name, "(c) Dense MVS input.", "(d) Ours textured.",
            "(e) Poisson.", "(f) Vu $\it{et\ al}$.", "(g) Jancosek $\it{et\ al}$.", "(h) Ours.",
            "(i) Poisson.", "(j) Vu $\it{et\ al}$.", "(k) Jancosek $\it{et\ al}$.", "(l) Ours."]
    # name="(a) Image of "+args.scene
    # titles=[name, "(b) Ground truth", "(c) Dense MVS input", "(d) Ours textured", "(e) Poisson", "(f) Labatut et al.", "(g) Vu et al.", "(h) Ours"]
    m=0
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].imshow(images[m])
            axes[i, j].set_xlabel(titles[m],fontsize=18, fontname = "Times New Roman")
            # axes[i, j].set_xlabel(titles[m],fontsize=14,fontname="Liberation Serif")
            # axes[i, j].xaxis.set_visible(False)
            axes[i, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plt.setp(axes[i, j].spines.values(), visible=False)

            m+=1

    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.show(block=False)
    plt.savefig(os.path.join(glob_image_dir,args.scene+".pdf"))



def arrange(args):

    glob_image_dir = os.path.join(args.user_dir, args.data_dir, args.scene, 'images')

    args.methods = ['image','ground_truth', 'input', 'clf_textured', 'poisson', 'labatu', 'jancosek', 'clf']

    fig,axes = plt.subplots(nrows=2, ncols=4, figsize=(6.875, 3), dpi=200)
    # fig,axes = plt.subplots(nrows=2, ncols=4, figsize=(13.75, 6))

    images=[]
    for method in args.methods:
        im_file = os.path.join(glob_image_dir, method, args.scene+'.png')
        images.append(plt.imread(im_file))

    # plt.rcParams['text.usetex'] = True
    # # plt.rcParams['font.family'] = "serif"
    # # plt.rcParams['font.serif'] = "Times"
    # plt.rcParams['font.family'] = 'Liberation Serif'

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = "Times New Roman"
    plt.rcParams['mathtext.it'] = "Times New Roman:italic"
    plt.rcParams['mathtext.bf'] = "Times New Roman:bold"


    name="(a) Image of $\it{delivery}$."
    # name="(a) Image of $\it{"+args.scene+"}$."
    titles=[name, "(b) Ground truth.", "(c) Dense MVS input.", "(d) Ours textured.", "(e) Poisson.", "(f) Vu $\it{et\ al}$.", "(g) Jancosek $\it{et\ al}$.", "(h) Ours."]

    # name="(a) Image of "+args.scene
    # titles=[name, "(b) Ground truth", "(c) Dense MVS input", "(d) Ours textured", "(e) Poisson", "(f) Labatut et al.", "(g) Vu et al.", "(h) Ours"]
    m=0
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].imshow(images[m])
            axes[i, j].set_xlabel(titles[m],fontsize=9, fontname = "Times New Roman")
            # axes[i, j].set_xlabel(titles[m],fontsize=14,fontname="Liberation Serif")
            # axes[i, j].xaxis.set_visible(False)
            axes[i, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plt.setp(axes[i, j].spines.values(), visible=False)

            m+=1

    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.show(block=False)
    plt.savefig(os.path.join(glob_image_dir,args.scene+".pdf"))


def arrange_with_uncleaned(args):

    glob_image_dir = os.path.join(args.user_dir, args.data_dir, args.scene, 'images')

    args.methods = ['ground_truth', 'image', 'input', 'clf_textured', \
                    'poisson', 'labatu', 'jancosek', 'clf', \
                    'poisson_uncleaned', 'labatu_uncleaned', 'jancosek_uncleaned', 'clf_uncleaned']

    fig,axes = plt.subplots(nrows=3, ncols=4, figsize=(13.75, 9), dpi=300)

    images=[]
    for method in args.methods:
        im_file = os.path.join(glob_image_dir, method, args.scene+'.png')
        images.append(plt.imread(im_file))

    name="(b) Image of $\it{"+args.scene+"}$."
    titles=["(a) Ground truth.", name, "(c) Dense MVS input.", "(d) Ours textured.", \
            "(e) Poisson.", r'(f) Vu $\it{et\ al}$.', "(g) Jancosek $\it{et\ al}$.", "(h) Ours.", \
            "(i) Poisson uncleaned.", "(j) Vu $\it{et\ al}$ uncleaned.", "(k) Jancosek $\it{et\ al}$ uncleaned.", "(l) Ours uncleaned."]
    m=0
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].imshow(images[m])
            axes[i, j].set_xlabel(titles[m],fontsize=18,fontname="Times New Roman")
            # axes[i, j].set_xlabel(titles[m],fontsize=14,fontname="Liberation Serif")
            # axes[i, j].xaxis.set_visible(False)
            axes[i, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plt.setp(axes[i, j].spines.values(), visible=False)

            m+=1

    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.show(block=False)
    plt.savefig(os.path.join(glob_image_dir,args.scene+".pdf"))


def dump(args):

    # args.set_view = True

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800,height=600, visible=True)

    working_dir = os.path.join(args.user_dir, args.data_dir, args.scene, 'import')
    if(args.method=='input'):
        mesh_file = os.path.join(working_dir, "densify_file.ply")
        print("read input file: ", mesh_file)
        recon_mesh = o3d.io.read_point_cloud(mesh_file)
        vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.ZCoordinate
    elif (args.method == 'ground_truth'):
        mesh_file = os.path.join(working_dir, "ground_truth.ply")
        print("read input file: ", mesh_file)
        recon_mesh = o3d.io.read_point_cloud(mesh_file)
        vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.ZCoordinate
    elif(args.method=='clf' or args.method=='clf_uncleaned'):
        if(args.method=='clf'):
            mesh_file = os.path.join(working_dir, args.scene + "_cl_0.5_cleaned.ply")
        else:
            mesh_file = os.path.join(working_dir, args.scene + "_cl_0.5_mesh.ply")
        print("read reconstruction file: ", mesh_file)
        recon_mesh = o3d.io.read_triangle_mesh(mesh_file)
        recon_mesh.compute_vertex_normals()
        recon_mesh.vertex_colors = recon_mesh.vertex_normals
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Default
        vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    elif(args.method=='clf_textured'):
        mesh_file = os.path.join(working_dir, args.scene + "_cl_05_textured.obj")
        print("read reconstruction file: ", mesh_file)
        recon_mesh = o3d.io.read_triangle_mesh(mesh_file)
        recon_mesh.compute_vertex_normals()
        recon_mesh.vertex_colors = recon_mesh.vertex_normals
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Color
        vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    elif(args.method=='labatu' or args.method=='labatu_uncleaned'):
        if(args.method=='labatu'):
            mesh_file = os.path.join(working_dir, "mesh_Labatu_cleaned.ply")
        else:
            mesh_file = os.path.join(working_dir, "mesh_Labatu_initial.ply")
        print("read reconstruction file: ", mesh_file)
        recon_mesh = o3d.io.read_triangle_mesh(mesh_file)
        recon_mesh.compute_vertex_normals()
        recon_mesh.vertex_colors = recon_mesh.vertex_normals
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Default
        vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    elif(args.method=='jancosek' or args.method=='jancosek_uncleaned'):
        if(args.method=='jancosek'):
            mesh_file = os.path.join(working_dir, "mesh_Jancosek_cleaned.ply")
        else:
            mesh_file = os.path.join(working_dir, "mesh_Jancosek_initial.ply")
        print("read reconstruction file: ", mesh_file)
        recon_mesh = o3d.io.read_triangle_mesh(mesh_file)
        recon_mesh.compute_vertex_normals()
        recon_mesh.vertex_colors = recon_mesh.vertex_normals
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Default
        vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    elif(args.method=='poisson' or args.method =='poisson_uncleaned'):
        if(args.method=='poisson'):
            mesh_file = os.path.join(working_dir, args.scene+"_cleaned.ply")
        else:
            mesh_file = os.path.join(working_dir, args.scene + ".ply")
        print("read reconstruction file: ", mesh_file)
        recon_mesh = o3d.io.read_triangle_mesh(mesh_file)
        recon_mesh.compute_vertex_normals()
        recon_mesh.vertex_colors = recon_mesh.vertex_normals
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Default
        vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    else:
        print("NOT A VALID METHOD")
        sys.exit(1)



    vis.add_geometry(recon_mesh)


    if(not os.path.exists(os.path.join(working_dir, "../..", "views"))):
        os.makedirs(os.path.join(working_dir, "../..", "views"))

    view_file = os.path.join(working_dir, "../..", "views", args.scene + "_viewpoint.json")
    if(os.path.isfile(view_file)):
        # load viewfile
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(view_file)
        ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()

    # set viewfile
    if(args.set_view):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        print("write view file: ", view_file)
        o3d.io.write_pinhole_camera_parameters(view_file, param)


    # make image
    image_dir=os.path.join(working_dir, "../..", "images", args.method)
    if(not os.path.exists(image_dir)):
        os.makedirs(image_dir)
    image_file = os.path.join(image_dir,args.scene+".png")
    vis.capture_screen_image(image_file, do_render=False)
    vis.destroy_window()


    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')

    parser.add_argument('--mode', type=str, nargs='+', default=['arrange'], choices=['dump', 'arrange', 'arrange_2', 'arrange_uncleaned'],  help='set the mode')

    parser.add_argument('--user_dir', type=str, default="/home/adminlocal/PhD/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="data/ETH3D/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scenes', nargs='+', type=str, default=["delivery_area"],
                        help='on which scene to execute pipeline.')
    parser.add_argument('-c', '--confs', nargs='+', type=int, default=[0],
                        help='which config file to load')

    # Sure options
    parser.add_argument('--sure_dir', type=str, default="cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure build directory, pointing to .../build/release folder starting from user_dir')
    parser.add_argument('--gco', type=str, default="",
                        help='graph cut optimization type,weight. default: area,1.0')
    parser.add_argument('-m','--methods', nargs='+', type=str, default=["ground_truth"],
                        help='the reconstruction method, default: rt,1,labatu')

    # Poisson options
    parser.add_argument('--poisson_dir', type=str, default="cpp/PoissonReconOri/Bin/Linux",
                        help='Indicate the poisson build directory, starting from user_dir')
    parser.add_argument('--depth', type=int, default=6,
                        help='Poisson depth')


    # choose method:
    parser.add_argument('-p', '--steps', type=str, default='r',
                        help='pipeline steps. default: idmr. extra options: sampling = s, evaluation = e')

    parser.add_argument('-u', '--upload', type=int, default=1,
                        help='upload the results to google spreadsheet')

    parser.add_argument('-v','--set_view', action='store_true',
                        help='set and save the view')


    args = parser.parse_args()

    if(args.confs[0] == -1):
        args.confs = [0,1,2,3,4]

    if(args.methods[0] == 'all'):
        args.methods = ['input', 'ground_truth', 'poisson', 'clf_textured', 'clf', 'labatu', 'jancosek']
    elif(args.methods[0] == 'uncleaned'):
        args.methods = ['input', 'ground_truth', 'clf_textured', \
                        'poisson', 'labatu', 'jancosek', 'clf', \
                        'poisson_uncleaned', 'labatu_uncleaned', 'jancosek_uncleaned', 'clf_uncleaned']


    if('dump' in args.method):
        for method in args.methods:
            args.method = method
            for scene in args.scenes:
                args.scene = scene
                for i,conf in enumerate(args.confs):
                    args.conf = conf
                    dump(args)
    if('arrange' in args.method):
        for scene in args.scenes:
            args.scene = scene
            arrange(args)
    if('arrange_2' in args.method):
        for scene in args.scenes:
            args.scene = scene
            arrange_2(args)
    if('arrange_uncleaned' in args.method):
        for scene in args.scenes:
            args.scene = scene
            arrange_with_uncleaned(args)
