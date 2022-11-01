import argparse, os, sys
import open3d as o3d
import numpy as np



import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = "Times New Roman"
plt.rcParams['mathtext.it'] = "Times New Roman:italic"
plt.rcParams['mathtext.bf'] = "Times New Roman:bold"




### feature plots
def bins_labels(bins, **kwclf):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwclf)
    plt.xlim(bins[0], bins[-1])

# def graphPlotting(clf):
#
#     print("\nMake feature plots:")
#
#     clf.n_epoch_test = int(9)
#     clf.n_class = 2
#     clf.temp.device = "cuda:" + str(clf.gpu)
#     clf.lr = 5e-3
#     clf.lr_sd = 8
#     clf.wd = 0
#     clf.with_rays = False
#     clf.with_label = True
#
#
#     ## prepare graph data
#     graph_loader = io.dataLoader(clf)
#
#     graph_list = clf.train_files.split("#")
#     print("using input files:")
#     for graph in graph_list:
#         print("-", graph)
#         graph_loader.load(graph + "_lrtcs_0", totensor=False)
#
#     data = np.concatenate((graph_loader.features,graph_loader.gt.values),axis=1)
#     np.random.shuffle(data)
#     data=data[:10000,:-2]
#     # data=np.array(data.data)
#     names=graph_loader.feature_names+["label"]
#     df = pd.DataFrame(data=data,columns=names)
#
#     df_outside = df[df['label'] == 1.0]
#     df_inside = df[df['label'] == 0.0]
#
#     for column in df:
#         fig, axs = plt.subplots(2, sharex=True, sharey=True)
#         fig.suptitle(column)
#         if(column == 'label' ):
#             bins=[0,1,2]
#             scale = "linear"
#         elif(column == 'n_inside_rays_first' or column == 'n_outside_rays_first'):
#             bins=[0,1,2,3,4]
#             scale = "linear"
#         elif(column == 'n_inside_rays' or column == 'n_outside_rays' or column == 'n_outside_rays_last'):
#             bins = [0,1,10,100,1000,10000,100000]
#             scale = "log"
#         else: # shape features and ray dists
#             # bins = [0,0.00001,0.5, 1, 2, 3, 5, 50, 100,1000,5000]
#             bins = [0,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]
#             # bins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
#             scale = "log"
#         axs[0].hist(np.array(df_outside[column]), bins=bins, color='b', label='outside')
#         axs[0].legend(loc="upper right")
#         axs[1].hist(np.array(df_inside[column]), bins=bins, color='r', label='inside')
#         axs[1].legend(loc="upper right")
#         plt.xscale(scale)
#         # plt.xticks(bins)
#         fig.tight_layout()
#
#         plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/results/feature_plots/"+column+".pdf", dpi=200, facecolor='w', edgecolor='w',
#                 orientation='portrait', papertype=None, format='pdf',
#                 transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
#         plt.show(block=False)

def arrange_gt(args):

    glob_image_dir = os.path.join(args.user_dir, args.data_dir, "images")
    fig,axes = plt.subplots(nrows=1, ncols=5, figsize=(6.875, 1.75), dpi=300)

    args.scenes = ['anchor', 'gargoyle', 'dc', 'daratech', 'lordquas']
    for i,scene in enumerate(args.scenes):
        im_file = os.path.join(glob_image_dir, args.method, scene + '.png')
        # print('read im file ', im_file)
        image = plt.imread(im_file)
        axes[i].imshow(image)
        # hide the axis: https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots
        axes[i].xaxis.set_visible(False)
        axes[i].tick_params(left=False, labelleft=False)
        plt.setp(axes[i].spines.values(), visible=False)

    # for ax, col in zip(axes, ['Anchor', 'Gargoyle', 'DC', 'Daratech', 'Lord Quas']):
    #     ax.set_title(str(col))
    #
    # for ax, row in zip(axes, ["Ground Truth"]):
    #     ax.set_ylabel(row, rotation=90, size='large')

    # fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    plt.show(block=False)
    plt.savefig(os.path.join(glob_image_dir,"ground_truth.png"))

def arrange_input(args):

    glob_image_dir = os.path.join(args.user_dir, args.data_dir, "images")
    fig,axes = plt.subplots(nrows=1, ncols=5, figsize=(6.875, 1.75), dpi=300)

    # args.scenes = ['anchor', 'gargoyle', 'dc', 'daratech', 'lordquas']
    # args.confs = ['anchor', 'gargoyle', 'dc', 'daratech', 'lordquas']
    for i,conf in enumerate(args.confs):
        im_file = os.path.join(glob_image_dir, args.method, args.scenes[0] + "_" + str(conf) + '.png')
        # print('read im file ', im_file)
        image = plt.imread(im_file)
        axes[i].imshow(image)
        # hide the axis: https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots
        axes[i].xaxis.set_visible(False)
        axes[i].tick_params(left=False, labelleft=False)
        plt.setp(axes[i].spines.values(), visible=False)

    # for ax, col in zip(axes, ['Anchor', 'Gargoyle', 'DC', 'Daratech', 'Lord Quas']):
    #     ax.set_title(str(col))
    #
    # for ax, row in zip(axes, ["Ground Truth"]):
    #     ax.set_ylabel(row, rotation=90, size='large')

    # fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    plt.show(block=False)
    plt.savefig(os.path.join(glob_image_dir,"input_arranged.png"))



def arrange_trans(args):

    glob_image_dir = os.path.join(args.user_dir, args.data_dir, "images")

    # args.methods = ['input', 'occ', 'igr', 'poisson', 'labatut/'+args.gco.split('-')[0], 'clf/cl/'+args.gco.split('-')[0]]
    args.methods = ['input', 'occ', 'poisson', 'labatut/'+args.gco.split('-')[0], 'clf/cl/'+args.gco.split('-')[0]]


    # args.methods = ['csrt/angle', 'cs/angle', 'cs/cc', 'cs/sv', 'cs/no']

    # row=args.confs
    # col=args.methods

    # fig,axes = plt.subplots(nrows=len(args.methods), ncols=len(args.confs), figsize=(8., 13.), dpi=400)
    fig,axes = plt.subplots(nrows=len(args.confs), ncols=len(args.methods), figsize=(16., 12.))
    # fig,axes = plt.subplots(nrows=len(args.confs), ncols=len(args.methods), figsize=(27., 24.))

    # k = 5-len(args.confs)
    # for i,method in enumerate(args.methods):
    #     for j in args.confs:
    #         im_file=os.path.join(glob_image_dir,method,args.scene+'_'+str(j)+'.png')
    #         # print('read im file ', im_file)
    #         image=plt.imread(im_file)
    #         axes[j-k,i].imshow(image)
    #         # hide the axis: https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots
    #         axes[j-k,i].xaxis.set_visible(False)
    #         axes[j-k,i].tick_params(left=False, labelleft=False)
    #         plt.setp(axes[j-k,i].spines.values(), visible=False)
    k = 5-len(args.confs)
    for i,method in enumerate(args.methods):
        for j, conf in enumerate(args.confs):
            im_file=os.path.join(glob_image_dir, method, args.scene+'_'+str(conf)+'.png')
            # print('read im file ', im_file)
            image=plt.imread(im_file)
            axes[j,i].imshow(image)
            # hide the axis: https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots
            axes[j,i].xaxis.set_visible(False)
            axes[j,i].tick_params(left=False, labelleft=False)
            plt.setp(axes[j,i].spines.values(), visible=False)

    conf_names = ["Low res.", "High res.", "High res. w/\nnoise", "High res. w/\noutliers", "High res. w/\nnoise a. outliers"]
    # conf_names = ["High res. w/\nnoise", "High res. w/\noutliers", "High res. w/\nnoise a. outliers"]
    temp = []
    for i,conf in enumerate(args.confs):
        temp.append(conf_names[conf])
    conf_names = temp

    # for ax, col in zip(axes[:,0], conf_names):
    #     ax.set_ylabel(str(col),  fontsize=42)
    #
    # # for ax, row in zip(axes[:,0], args.methods):
    # #     ax.set_ylabel(row, rotation=90, size='large',fontsize=18)
    # for ax, row in zip(axes[0], ["Input Point Cloud", "ConvONet", "IGR", "Poisson", "Labatut $\it{et\ al}$.", "Ours"]):
    #     ax.set_title(row, size='large',fontsize=42)

    fig.align_ylabels(axes[0])

    # fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    plt.show(block=False)
    plt.savefig(os.path.join(glob_image_dir,args.scene+"_trans.png"))


def arrange(args):

    glob_image_dir = os.path.join(args.user_dir, args.data_dir, "images")

    args.methods = ['input', 'occ', 'igr', 'poisson', 'labatut/'+args.gco.split('-')[0], 'clf/cl/'+args.gco.split('-')[0]]

    # args.methods = ['csrt/angle', 'cs/angle', 'cs/cc', 'cs/sv', 'cs/no']

    # row=args.confs
    # col=args.methods

    # fig,axes = plt.subplots(nrows=len(args.methods), ncols=len(args.confs), figsize=(8., 13.), dpi=400)
    # fig,axes = plt.subplots(nrows=len(args.methods), ncols=len(args.confs), figsize=(8., 9.))
    fig,axes = plt.subplots(nrows=len(args.methods), ncols=len(args.confs), figsize=(22., 27.))

    # k = 5-len(args.confs)
    # for i,method in enumerate(args.methods):
    #     for j in args.confs:
    #         im_file=os.path.join(glob_image_dir,method,args.scene+'_'+str(j)+'.png')
    #         # print('read im file ', im_file)
    #         image=plt.imread(im_file)
    #         axes[j-k,i].imshow(image)
    #         # hide the axis: https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots
    #         axes[j-k,i].xaxis.set_visible(False)
    #         axes[j-k,i].tick_params(left=False, labelleft=False)
    #         plt.setp(axes[j-k,i].spines.values(), visible=False)
    k = 5-len(args.confs)
    for i,method in enumerate(args.methods):
        for j, conf in enumerate(args.confs):
            im_file=os.path.join(glob_image_dir, method, args.scene+'_'+str(j)+'.png')
            # print('read im file ', im_file)
            image=plt.imread(im_file)
            axes[i,j].imshow(image)
            # hide the axis: https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots
            axes[i,j].xaxis.set_visible(False)
            axes[i,j].tick_params(left=False, labelleft=False)
            plt.setp(axes[i,j].spines.values(), visible=False)

    # conf_names = ["Low res. (LR)", "High res. (HR)", "HR + noise", "HR + outliers", "HR + noise + outliers"]
    conf_names = ["Low res.", "High res.", "High res. w/\nnoise", "High res. w/\noutliers", "High res. w/\nnoise a. outliers"]
    temp = []
    for i in args.confs:
        temp.append(conf_names[i])
    conf_names = temp

    for ax, col in zip(axes[0], conf_names):
        ax.set_title(str(col),fontsize=24)

    # for ax, row in zip(axes[:,0], args.methods):
    #     ax.set_ylabel(row, rotation=90, size='large',fontsize=18)
    for ax, row in zip(axes[:,0], ["Input Point Cloud", "ConvONet", "IGR", "Poisson", "Labatut $\it{et\ al}$.", "Ours"]):
        ax.set_ylabel(row, rotation=90, size='large',fontsize=24)

    fig.align_ylabels(axes[:,0])

    # fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    plt.show(block=False)
    plt.savefig(os.path.join(glob_image_dir,args.scene+".pdf"))





def dump(args):

    # args.set_view = True

    args.scene_conf= args.scene + "_" + str(args.conf)

    method = args.method.split(',')
    gco = args.gco.split('-')
    if(method[0] == "clf" or method[0] == "labatut"):
        if(len(gco) < 2):
            print("specify --gco angle-0.5")
        sys.exit(1)

    if(method[0]=='input'):
        working_dir = os.path.join(args.user_dir,args.data_dir,"scans","with_normals")
        mesh_file = os.path.join(working_dir, args.scene_conf + ".ply")
        print("read input file: ", mesh_file)
        recon_mesh = o3d.io.read_point_cloud(mesh_file)
    elif(method[0]=='ground_truth'):
        working_dir = os.path.join(args.user_dir,args.data_dir,"ground_truth_surface")
        args.scene_conf = args.scene
        mesh_file = os.path.join(working_dir, args.scene_conf + ".off")
        print("read ground truth file: ", mesh_file)
        recon_mesh = o3d.io.read_triangle_mesh(mesh_file)
        recon_mesh.compute_vertex_normals()
        recon_mesh.vertex_colors = recon_mesh.vertex_normals
    elif(method[0]=='clf'):
        if(len(method)<2):
            print("\n specify which clf, e.g. with clf,rt or clf,cs")
            sys.exit(1)
        working_dir = os.path.join(args.user_dir, args.data_dir, "reconstructions", method[0], method[1], gco[0])
        mesh_file = os.path.join(working_dir, args.scene_conf + "_" + method[1] + "_" + gco[1] + "_optimized.ply")
        if(gco[0] == 'no'):
            mesh_file = os.path.join(working_dir, args.scene_conf + "_cs_0_initial.ply")
        print("read reconstruction file: ", mesh_file)
        recon_mesh = o3d.io.read_triangle_mesh(mesh_file)
        recon_mesh.compute_vertex_normals()
        recon_mesh.vertex_colors = recon_mesh.vertex_normals
    elif(method[0]=='labatut'):
        working_dir = os.path.join(args.user_dir,args.data_dir,"reconstructions",args.method)
        mesh_file = os.path.join(working_dir, args.scene_conf + "_rt_5.0_optimized.ply")
        print("read reconstruction file: ", mesh_file)
        recon_mesh = o3d.io.read_triangle_mesh(mesh_file)
        recon_mesh.compute_vertex_normals()
        recon_mesh.vertex_colors = recon_mesh.vertex_normals
    else:
        working_dir = os.path.join(args.user_dir,args.data_dir,"reconstructions",args.method)
        mesh_file = os.path.join(working_dir, args.scene_conf + ".ply")
        print("read reconstruction file: ", mesh_file)
        recon_mesh = o3d.io.read_triangle_mesh(mesh_file)
        recon_mesh.compute_vertex_normals()
        recon_mesh.vertex_colors = recon_mesh.vertex_normals


    vis = o3d.visualization.Visualizer()
    vis.create_window(width=600,height=600, visible=True)
    # vis.create_window()
    vis.add_geometry(recon_mesh)

    # if(not args.set_view):
    vis.get_render_option().mesh_show_wireframe=False
    vis.get_render_option().light_on=True
    vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Default
    # vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Default
    vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    view_file = os.path.join(args.user_dir,args.data_dir,"views",args.scene+"_viewpoint.json")



    # set viewfile
    if(args.set_view):
        vis.run()
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        print("write view file: ", view_file)
        o3d.io.write_pinhole_camera_parameters(view_file, param)
    else:
        # load viewfile
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(view_file)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.run()

    # make image
    image_dir = os.path.join(args.user_dir, args.data_dir, "images", args.method, gco[0])
    if(len(method)>1):
        image_dir=os.path.join(args.user_dir,args.data_dir,"images",method[0], method[1], gco[0])
    os.makedirs(image_dir,exist_ok=True)
    image_file = os.path.join(image_dir,args.scene_conf+".png")
    print("save to: ", image_file)
    vis.capture_screen_image(image_file, do_render=True)
    vis.close()


    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')

    parser.add_argument('mode', type=str, nargs='+', default=['dump','arrange'],
                        choices=['dump', 'arrange', 'arrange_gt', 'arrange_trans', 'arrange_input'],  help='set the mode')

    parser.add_argument('--user_dir', type=str, default="/home/adminlocal/PhD/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="data/reconbench/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scenes', nargs='+', type=str, default=["anchor"],
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


    parser.add_argument('-i', '--input_file_extension', type=str, default="",
                        help='the mesh file to be evaluated')

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

    parser.add_argument('--set_view', default=False,
                        help='set and save the view')


    args = parser.parse_args()

    if(args.confs[0] == -1):
        args.confs = [0,1,2,3,4]

    if(args.methods[0] == 'all'):
        args.methods = ['input', 'poisson', 'occ', 'labatu', 'clf']

    if(args.scenes[0] == 'all'):
        args.scenes = ['anchor', 'gargoyle', 'dc', 'daratech', 'lordquas']

    if('dump' in args.mode):
        for method in args.methods:
            args.method = method
            for scene in args.scenes:
                args.scene = scene
                for i,conf in enumerate(args.confs):
                    args.conf = conf
                    dump(args)

    if('arrange' in args.mode):
        for scene in args.scenes:
            args.scene = scene
            arrange(args)

    if('arrange_trans' in args.mode):
        for scene in args.scenes:
            args.scene = scene
            arrange_trans(args)

    if('arrange_gt' in args.mode):
        args.method = 'ground_truth'
        arrange_gt(args)

    if('arrange_input' in args.mode):
        args.method = 'input'
        arrange_input(args)
