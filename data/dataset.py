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

        models = ["anchor", "daratech", "dc", "lordquas", "gargoyle"]
        clf.inference.files = []
        if(clf.temp.mode == "inference"):
            for s in clf.data.scan_confs:
                for m in models:
                    clf.inference.files.append({"category":"","id":m,"scan_conf":str(s)})
        else:
            print("NOT IMPLEMENTED ERROR: can't train on reconbench dataset!")
            sys.exit(1)
        a=5


    elif(clf.data.dataset == "ModelNet10"):

        if not clf.data.classes:
            classes = os.listdir(clf.training.path)
        else:
            classes = clf.data.classes
        if 'x' in classes: classes.remove('x')
        clf.training.files = []
        clf.validation.files = []

        temp = []
        if(clf.inference.files is not None):
            for f in clf.inference.files:
                for s in clf.data.scan_confs:
                    t = f.split('_')
                    temp.append({"category":t[0],"id":t[1],"scan_conf":str(s)})
                clf.inference.files = temp
            return
        else:
            clf.inference.files = []

        for c in classes:
            for s in clf.data.scan_confs:
                if(clf.temp.mode == "training"):
                    models = os.listdir(os.path.join(clf.training.path,c,"train"))
                    models = models[:clf.training.shapes_per_conf_per_class]
                    for m in models:
                        if os.path.isfile(os.path.join(clf.training.path,c,"2_watertight",m)):
                            n = re.split(r'[_.]+',m)
                            d = {"path":clf.training.path,"category":n[0],"id":n[1],"scan_conf":str(s)}
                            clf.training.files.append(d)

                    models = os.listdir(os.path.join(clf.validation.path,c,"test"))
                    models = models[:clf.validation.shapes_per_conf_per_class]
                    for m in models:
                        if os.path.isfile(os.path.join(clf.validation.path,c,"2_watertight",m)):
                            n = re.split(r'[_.]+',m)
                            d = {"path":clf.validation.path,"category":n[0],"id":n[1],"scan_conf":str(s)}
                            clf.validation.files.append(d)
                elif(clf.temp.mode == "inference"):
                    models = os.listdir(os.path.join(clf.inference.path,c,"test"))
                    models = models[:clf.inference.shapes_per_conf_per_class]
                    for m in models:
                        if os.path.isfile(os.path.join(clf.inference.path,c,"2_watertight",m)):
                            n = re.split(r'[_.]+',m)
                            d = {"category":n[0],"id":n[1],"scan_conf":str(s)}
                            clf.inference.files.append(d)
                else:
                    print("ERROR: not a valid method for getDataset.py")
                    sys.exit(1)


        a=5


    elif(clf.data.dataset == "ShapeNetManifoldPlus"):

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



    else:
        print("{} is not a valid dataset".format(clf.data.dataset))
        sys.exit(1)