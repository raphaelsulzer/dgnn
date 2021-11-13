import os, sys, re
import numpy as np
import torch

def reduceDataset(num_nodes, percentage):

    n_samples = int(num_nodes * percentage)

    ## make a train and test mask
    # 0 = unused; 1 = train;
    tt_mask = np.full(num_nodes, 0);
    tt_mask[:n_samples] = 1
    np.random.shuffle(tt_mask)
    train_mask = tt_mask == 1;
    test_mask = tt_mask == 0;

    return torch.from_numpy(train_mask), torch.from_numpy(test_mask)


def getDataset(clf):

    if (clf.data.scan_confs[0] == -1):
        clf.data.scan_confs = [0, 1, 2, 3, 4]

    if (clf.data.dataset == "reconbench"):

        clf.paths.data = "/home/raphael/data/reconbench/"

        if(clf.temp.args.training):
            for i,f in enumerate(clf.training.files):
                clf.training.files[i] = os.path.join(clf.paths.data,"gt",f+"_lrtcs_0")


        if (clf.inference.files[0] == "all"):
            clf.inference.files = ["anchor", "gargoyle", "dc", "daratech", "lordquas"]
        temp = []
        for s in clf.inference.files:
            for n in clf.data.scan_confs:
                file=os.path.join(clf.paths.data,"gt",s + "_" + str(n) + "_lrtcs_0")
                temp.append(file)
        clf.temp.inference_files = temp

    elif(clf.data.dataset == "modelnet"):

        if not clf.data.classes:
            classes = os.listdir(clf.paths.data)
        else:
            classes = clf.data.classes
        if 'x' in classes: classes.remove('x')
        clf.training.files = []
        clf.validation.files = []

        temp = []
        if(clf.inference.files is not None):
            for f in clf.inference.files:
                t = f.split('_')
                temp.append({"category":t[0],"id":t[1]})
            clf.inference.files = temp
            return
        else:
            clf.inference.files = []

        for c in classes:

            if(clf.temp.mode == "training"):
                models = os.listdir(os.path.join(clf.paths.data,c,"train"))
                models = models[:clf.training.shapes_per_conf_per_class]
                for m in models:
                    if os.path.isfile(os.path.join(clf.paths.data,c,"2_watertight",m)):
                        n = re.split(r'[_.]+',m)
                        d = {"category":n[0],"id":n[1]}
                        clf.training.files.append(d)

                models = os.listdir(os.path.join(clf.paths.data,c,"test"))
                models = models[:clf.validation.shapes_per_conf_per_class]
                for m in models:
                    if os.path.isfile(os.path.join(clf.paths.data,c,"2_watertight",m)):
                        n = re.split(r'[_.]+',m)
                        d = {"category":n[0],"id":n[1]}
                        clf.validation.files.append(d)
            elif(clf.temp.mode == "inference"):
                models = os.listdir(os.path.join(clf.paths.data,c,"train"))
                models = models[:clf.inference.shapes_per_conf_per_class]
                for m in models:
                    if os.path.isfile(os.path.join(clf.paths.data,c,"2_watertight",m)):
                        n = re.split(r'[_.]+',m)
                        d = {"category":n[0],"id":n[1]}
                        clf.inference.files.append(d)
            else:
                print("ERROR: not a valid method for getDataset.py")
                sys.exit(1)


        a=5



    elif(clf.data.dataset == "ethtrain"):
        clf.paths.data = "/home/raphael/data/eth"
        clf.training.files = []
        clf.temp.inference_files = []
        windows = os.listdir(os.path.join(clf.paths.data,'gt'))
        for f in windows:
            f = f.split('_')[0]
            clf.training.files.append(os.path.join(clf.paths.data,'gt',f+'_lrtcs_0'))
            if(f in clf.inference.files or clf.inference.files == "all"):
                clf.temp.inference_files.append(os.path.join(clf.paths.data,'gt',f+'_lrtcs_0'))

        clf.training.files = list(dict.fromkeys(clf.training.files))
        clf.temp.inference_files = list(dict.fromkeys(clf.temp.inference_files))
        clf.training.files = clf.training.files[:clf.temp.shapes_per_conf_per_class]


    elif (clf.data.dataset == "eth3d"):
        clf.paths.data = "/mnt/raphael/ETH3D/"
        clf.temp.data_path = "/mnt/raphael/ETH3D/"

        clf.temp.inference_files = []
        if (clf.inference.files[0] == "all"):
            clf.inference.files = ["meadow", "terrace", "delivery_area", "kicker", "pipes", "office", \
                                        "playground", "terrains",  "relief", "relief_2", "electro", \
                                        "courtyard", "facade"]
        for i,scene in enumerate(clf.inference.files):
            clf.temp.inference_files.append(os.path.join("/mnt/raphael/ETH3D", scene, "gt", scene+"_lrt_0"))


    elif(clf.data.dataset == "myshapenet"):
        clf.paths.data = "/mnt/raphael/ShapeNetManifoldPlus/"
        classes = os.listdir(clf.paths.data)
        clf.training.files = []

        clf.temp.total_shapes_per_class = np.zeros(len(classes))

        for cc,cl in enumerate(classes):

            class_files = []

            path = os.path.join(clf.paths.data, cl, "scans", "mine","")
            shapes = os.listdir(path)

            s = []
            for shape in shapes:
                s.append(shape.split('_')[0])
            # remove duplicates, meaning same shape with differnt scan_conf:
            shapes = list(dict.fromkeys(s))

            a=5
            # take some shapes per scan config
            while(len(class_files) <  clf.temp.shapes_per_conf_per_class):
                if(len(shapes) == 0):
                    break
                shape_conf = shapes[0]
                file = os.path.join(clf.paths.data, cl, "gt", shape_conf + "_lrtcs_0")
                if(not os.path.isfile(os.path.join(file + "_cbvf.txt"))):
                    shapes.pop(0)
                    continue
                class_files.append(file)
                clf.temp.total_shapes_per_class[cc]+=1
                shapes.pop(0)

            clf.training.files+=class_files

        a=5

    elif(clf.data.dataset == "shapenet"):
        clf.paths.data = "/mnt/raphael/ProcessedShapeNet/ShapeNet.build/"
        classes = os.listdir(clf.paths.data)
        clf.training.files = []

        # count how many shapes per conf I have
        clf.temp.total_shapes_per_conf = np.zeros(5)

        for cc,cl in enumerate(classes):

            path = os.path.join(clf.paths.data, cl, "scans", "with_sensor","")
            shapes = os.listdir(path)

            s = []
            for shape in shapes:
                s.append(shape.split('_')[0])
            # remove duplicates, meaning same shape with differnt scan_conf:
            shapes = list(dict.fromkeys(s))
            a=5
            # take some shapes per scan config
            for i in clf.data.scan_confs*clf.temp.shapes_per_conf_per_class:
                shape_conf = shapes[0] + '_' + str(i)
                file = os.path.join(clf.paths.data, cl, "gtb", shape_conf + "_lrtcs_0")
                if(not os.path.isfile(os.path.join(file + "_cbvf.txt"))):
                    shapes.pop(0)
                    continue
                # print("class {}   {} shape {}".format(cc,cl,file))
                clf.temp.total_shapes_per_conf[i]+=1
                clf.training.files.append(file)
                shapes.pop(0)

    elif (clf.data.dataset == "tat"):
        for s in clf.data.inference_files:
            clf.paths.data=os.path.join("/mnt/raphael/TanksAndTemples",s)
            clf.temp.inference_file = s+"_lrt_0"

    else:
        print("{} is not a valid dataset".format(clf.data.dataset))
        sys.exit(1)