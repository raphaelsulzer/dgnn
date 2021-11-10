import argparse
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', ''))
import matplotlib.pyplot as plt
import gspread
import gspread_dataframe as gd
import seaborn as sns

def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(int(height)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=45)

def plot_the_bars(df, args, n_of_bars=0):

    # plt.figure()
    plt.figure(figsize=(10, 5))

    # label_and_linestyle=[("point cloud","tab:blue","."),
    #            ("Jancosek et al.","tab:orange",None),
    #            ("Jancosek et al. cleaned","tab:orange","//"),
    #            ("GCN2i9e_no_reg", "tab:green", None),
    #            ("GCN2i9e no reg. cleaned", "tab:green", "//"),
    #            ("GCN2i9e area reg", "tab:red", None),
    #            ("GCN2i9e area reg cleaned", "tab:red", "//"),
    #            ("GCN2i9e cc reg", "tab:purple", None),
    #            ("GCN2i9e cc reg cleaned", "tab:purple", "//"),
    #              ("GCN2i9e cc,area,angle reg", "tab:cyan", None),
    #              ("GCN2i9e cc,area,angle reg cleaned", "tab:cyan", "//")
    #            ]

    cleaned= [None,"//"]
    if(not n_of_bars):
        n_of_bars = int(df.shape[0] / 6)

    color_palette = sns.color_palette(n_colors=n_of_bars,palette="muted")

    print("Plotting {} * 4 bars".format(n_of_bars))

    # ax = plt.subplot(221)
    # ax.set_title("vertices",loc="right")
    # for i in range(1,n_of_bars):
    #     # get the name of the method
    #
    #     label=df[args.scene][6*i]
    #     k=0
    #     if(label[-7:]=="cleaned"):
    #         k=1
    #     rects=plt.bar(np.array([0])+i/n_of_bars,
    #         df["Value:"][6*i:6*i+1],
    #         width=1/n_of_bars, bottom=None, align='edge',
    #         # label=label_and_linestyle[i][0],
    #         # edgecolor="black", color=label_and_linestyle[i][1], hatch=label_and_linestyle[i][2],
    #         label=label,
    #         edgecolor="black", color=color_palette[int((i - k)/2)], hatch=cleaned[k],
    #         tick_label=df["Intrinsics:"][6])
    #     autolabel(rects,ax)
    # plt.axis("off")
    #
    # ax = plt.subplot(222)
    # ax.set_title("faces",loc="right")
    # for i in range(1,n_of_bars):
    #     label=df[args.scene][6*i]
    #     k=0
    #     if(label[-7:]=="cleaned"):
    #         k=1
    #     rects=plt.bar(np.array([0])+i/n_of_bars,
    #             df["Value:"][6*i+1:6*i+2],
    #                   width=1 / n_of_bars, bottom=None, align='edge',
    #                   # label=label_and_linestyle[i][0],
    #                   # edgecolor="black", color=label_and_linestyle[i][1], hatch=label_and_linestyle[i][2],
    #                   label=label,
    #                   edgecolor="black", color=color_palette[int((i - k)/2)], hatch=cleaned[k],
    #                   tick_label=df["Intrinsics:"][6])
    #     autolabel(rects,ax)
    # plt.axis("off")

    ax = plt.subplot(223)
    ax.set_title("components",loc="right")
    for i in range(1,n_of_bars):
        label=df[args.scene][6*i]
        k=0
        if(label[-7:]=="cleaned"):
            k=1
        rects=plt.bar(np.array([0])+i/n_of_bars,
                df["Value:"][6*i+2:6*i+3],
                      width=1 / n_of_bars, bottom=None, align='edge',
                      # label=label_and_linestyle[i][0],
                      # edgecolor="black", color=label_and_linestyle[i][1], hatch=label_and_linestyle[i][2],
                      label=label,
                      edgecolor="black", color=color_palette[int((i - k)/2)], hatch=cleaned[k],
                      tick_label=df["Intrinsics:"][6])
        autolabel(rects,ax)
    plt.axis("off")

    ax = plt.subplot(224)
    ax.set_title("area",loc="right")
    for i in range(1,n_of_bars):
        label=df[args.scene][6*i]
        k=0
        if(label[-7:]=="cleaned"):
            k=1
        rects=plt.bar(np.array([0])+i/n_of_bars,
                df["Value:"][6*i+3:6*i+4],
                      width=1 / n_of_bars, bottom=None, align='edge',
                      # label=label_and_linestyle[i][0],
                      # edgecolor="black", color=label_and_linestyle[i][1], hatch=label_and_linestyle[i][2],
                      label=label,
                      edgecolor="black", color=color_palette[int((i - k)/2)], hatch=cleaned[k],
                      tick_label=df["Intrinsics:"][6])
        autolabel(rects,ax)
    plt.axis("off")



    # ### Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 1.5, box.height])
    # ### Put a legend to the right of the current axis
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.legend(loc='best')
    plt.suptitle("Intrinsics")
    plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/results/gc/intrinsics.pdf", dpi=400,
                facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format='pdf',
                transparent=False, bbox_inches=None, pad_inches=0.01, metadata=None)
    plt.show(block=False)

    a=5


def plot_the_curves(df, args, which_curve, n_of_curves=0, thresholds=[0,6]):

    min_threshold=thresholds[0]
    max_threshold=thresholds[1]

    plt.figure(figsize=(10, 5))
    ax = plt.subplot(111)

    # label_and_linestyle=[("point cloud","tab:blue",":"),
    #            ("Jancosek et al.","tab:orange","-"),
    #            ("Jancosek et al. cleaned","tab:orange","--"),
    #            ("GCN2i9e no reg.", "tab:green", "-"),
    #            ("GCN2i9e no reg. cleaned", "tab:green", "--"),
    #            ("GCN2i9e area reg", "tab:red", "-"),
    #            ("GCN2i9e area reg cleaned", "tab:red", "--"),
    #            ("GCN2i9e cc reg", "tab:purple", "-"),
    #            ("GCN2i9e cc reg cleaned", "tab:purple", "--"),
    #                      ("GCN2i9e cc,area,angle reg", "tab:cyan", "-"),
    #                      ("GCN2i9e cc,area,angle reg cleaned", "tab:cyan", "--")
    #            ]

    if(not n_of_curves):
        n_of_curves = int(df.shape[0] / 6)
    color_palette = sns.color_palette(n_colors=n_of_curves,palette="muted")

    cleaned = ["-", "--"]
    print("Plotting {} curves".format(n_of_curves))
    # ax.plot(df['Tolerances:'][min_threshold:max_threshold], df[which_curve][min_threshold:max_threshold],
    #         label="point cloud", color=color_palette[0], linestyle=':')
    for i in range(1,n_of_curves):
        label=df[args.scene][6*i]
        k=0
        if(label[-7:]=="cleaned"):
            k=1
        # ax.plot(df['Tolerances:'][6*i:6*i+6],df[which_curve][6*i:6*i+6], label=df[args.scene][6*i])
        ax.plot(df['Tolerances:'][6*i+min_threshold:6*i+max_threshold],df[which_curve][6*i+min_threshold:6*i+max_threshold],
                label=label, color=color_palette[int((i - k)/2)], linestyle=cleaned[k])
        # marker = 's'
        # label=label_and_linestyle[i][0], color=label_and_linestyle[i][1],
        #         linestyle=label_and_linestyle[i][2])

    # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.3, box.height])
    #
    # # Put a legend to the right of the current axis
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(loc='lower right')
    ax.set_title(which_curve)
    ax.set_xticks(df['Tolerances:'][min_threshold:max_threshold])

    plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/results/gc/" + which_curve[:-1] + ".pdf", dpi=200,
                facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format='pdf',
                transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
    plt.show(block=False)





def visu(args):

    # TODO: needs to be redone with gspread instead of old toSpreadsheet.py functions

    args.service = ts.getService()

    gc = gspread.service_account('data-upload-project-277415-cc222b36b36b.json')
    sh = gc.open("gc_results")

    # args.scene="all_plot"
    ws = sh.worksheet(args.scene)

    df = gd.get_as_dataframe(ws, header=0, evaluate_formulas=True)
    df.dropna(inplace=True, how='all', axis=0)
    df.dropna(inplace=True, how='all', axis=1)

    how_many_methods=None
    min_threshold=2
    max_threshold=6
    thresholds=[min_threshold,max_threshold]
    plot_the_bars(df, args, how_many_methods)
    plot_the_curves(df, args, 'F1-scores:',how_many_methods,thresholds)
    plot_the_curves(df, args, 'Accuracies:',how_many_methods,thresholds)
    plot_the_curves(df, args, 'Completenesses:',how_many_methods,thresholds)








if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='reconstruction visualization')
    parser.add_argument('-s', '--scenes', nargs = '+', type=str, default=["paper"],
                        help='on which scene to execute pipeline.')

    args = parser.parse_args()

    # if(args.scenes[0] == 'all'):
    #     args.scenes = os.listdir(args.user_dir+args.data_dir)

    # The ID and range of a sample spreadsheet.
    # gc_results
    args.spreadsheet_id = '1AmK3VzAgAD_kiruQNFAMagC0C7JbnlahNAXfvUFjvy0'
    # classification_results
    args.service = ts.getService()

    scenes_data_list = []
    for i,scene in enumerate(args.scenes):

        args.scene = scene
        scene_data = visu(args)


