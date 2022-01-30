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


def getConfig(clf,mode):

    path = os.path.join(clf.paths.data,mode.dataset)

    if not mode.classes:
        if os.path.isfile(os.path.join(path,"classes.lst")):
            classfile = os.path.join(path,"classes.lst")
            with open(classfile, 'r') as f:
                classes = f.read().split('\n')
            classes = list(filter(None, classes))
        else: classes = mode.classes
    else:
        classes = mode.classes

    if (not isinstance(mode.scan_confs, list)):
        mode.scan_confs = [mode.scan_confs]
    if(mode.scan_confs[0] == -1):
        mode.scan_confs = [0, 1, 2, 3, 4]




    return path,classes,mode.scan_confs


def getDataset(clf,dataset,mode):



    if(dataset == "ModelNet10"):

        # TODO: this code should be simplified, because it is basically repeated for training, validation and inference mode
        if(mode == "training"):

            clf.training.path, clf.training.classes, clf.training.scan_confs = getConfig(clf,clf.training)
            clf.training.files = []
            for c in clf.training.classes:
                splitfile = os.path.join(clf.training.path,c,clf.paths.train_split+".lst")
                for s in clf.training.scan_confs:
                    with open(splitfile, 'r') as f: models = f.read().split('\n')
                    models = models[:clf.training.shapes_per_conf_per_class]
                    for m in models:
                        if os.path.exists(os.path.join(clf.training.path, c, "gt", str(s), m)):
                            d = {"path": os.path.join(clf.training.path,c), "filename": c+"_"+m,
                                 "category": c, "id": m, "scan_conf": str(s), "gtfile": os.path.join("gt",str(s),m,c+"_"+m)}
                            clf.training.files.append(d)

        elif(mode == "validation"):

            clf.validation.path, clf.validation.classes, clf.validation.scan_confs = getConfig(clf,clf.validation)
            clf.validation.files = []
            for c in clf.validation.classes:
                splitfile = os.path.join(clf.validation.path,c,clf.paths.val_split+".lst")
                for s in clf.validation.scan_confs:
                    with open(splitfile, 'r') as f: models = f.read().split('\n')
                    models = models[:clf.validation.shapes_per_conf_per_class]
                    for m in models:
                        if os.path.exists(os.path.join(clf.validation.path, c, "gt", str(s), m)):
                            d = {"path": os.path.join(clf.validation.path,c), "filename": c+"_"+m, "category": c,
                                 "id": m, "scan_conf": str(s), "gtfile": os.path.join("gt",str(s),m,c+"_"+m)}
                            clf.validation.files.append(d)

        elif(mode == "inference"):

            clf.inference.path, clf.inference.classes, clf.inference.scan_confs = getConfig(clf,clf.inference)
            temp = []
            if(clf.inference.files is not None):
                for f in clf.inference.files:
                    for s in clf.data.scan_confs:
                        n = f.split('_')
                        temp.append({"path": os.path.join(clf.inference.path,n[0]), "filename": n[0]+"_"+n[1], "category":n[0],"id":n[1],"scan_conf":str(s)})
                    clf.inference.files = temp
                return
            else:
                clf.inference.files = []
                for c in clf.inference.classes:
                    splitfile = os.path.join(clf.inference.path, c, clf.paths.test_split + ".lst")
                    for s in clf.inference.scan_confs:
                        with open(splitfile, 'r') as f:
                            models = f.read().split('\n')
                        models = list(filter(None, models))
                        models = models[:clf.inference.shapes_per_conf_per_class]
                        for m in models:
                            if os.path.exists(os.path.join(clf.inference.path, c, "gt", str(s), m)):
                                d = {"path": os.path.join(clf.inference.path, c), "filename": c + "_" + m,
                                     "category": c, "id": m, "scan_conf": str(s), "gtfile": os.path.join("gt",str(s),m,c+"_"+m)}
                                clf.inference.files.append(d)
        else:
            print("ERROR: not a valid method for getDataset.py")
            sys.exit(1)

        a=4


    elif(dataset == "ShapeNet"):

        # TODO: this code should be simplified, because it is basically repeated for training, validation and inference mode
        if (mode == "training"):

            clf.training.path, clf.training.classes, clf.training.scan_confs = getConfig(clf, clf.training)
            clf.training.files = []
            for c in clf.training.classes:
                splitfile = os.path.join(clf.training.path, c, clf.paths.train_split + ".lst")
                for s in clf.training.scan_confs:
                    with open(splitfile, 'r') as f:
                        models = f.read().split('\n')
                    models = models[:clf.training.shapes_per_conf_per_class]
                    for m in models:
                        if os.path.exists(os.path.join(clf.training.path, c, m, "gt", str(s), m)):
                            d = {"path": os.path.join(clf.training.path, c), "filename": str(s),
                                 "category": c,
                                 "id": m, "scan_conf": str(s), "gtfile": os.path.join("gt",str(s))}
                            clf.training.files.append(d)

        elif (mode == "validation"):

            clf.validation.path, clf.validation.classes, clf.validation.scan_confs = getConfig(clf, clf.validation)
            clf.validation.files = []
            for c in clf.validation.classes:
                splitfile = os.path.join(clf.validation.path, c, clf.paths.val_split + ".lst")
                for s in clf.validation.scan_confs:
                    with open(splitfile, 'r') as f:
                        models = f.read().split('\n')
                    models = models[:clf.validation.shapes_per_conf_per_class]
                    for m in models:
                        if os.path.exists(os.path.join(clf.validation.path, c, m, "gt", str(s), m)):
                            d = {"path": os.path.join(clf.validation.path, c), "filename": str(s),
                                 "category": c,
                                 "id": m, "scan_conf": str(s), "gtfile": os.path.join("gt",str(s))}
                            clf.validation.files.append(d)

        elif (mode == "inference"):

            clf.inference.path, clf.inference.classes, clf.inference.scan_confs = getConfig(clf, clf.inference)
            temp = []
            if (clf.inference.files is not None):
                for f in clf.inference.files:
                    for s in clf.data.scan_confs:
                        n = f.split('_')
                        temp.append({"path": os.path.join(clf.inference.path, n[0]), "filename": n[0] + "_" + n[1],
                                     "category": n[0], "id": n[1], "scan_conf": str(s)})
                    clf.inference.files = temp
                return
            else:
                clf.inference.files = []
                for c in clf.inference.classes:
                    splitfile = os.path.join(clf.inference.path, c, clf.paths.test_split + ".lst")
                    for s in clf.inference.scan_confs:
                        with open(splitfile, 'r') as f:
                            models = f.read().split('\n')
                        models = list(filter(None, models))
                        models = models[:clf.inference.shapes_per_conf_per_class]
                        for m in models:
                            if os.path.exists(os.path.join(clf.inference.path, c, m, "gt", str(s)+"_3dt.npz")):
                                d = {"path": os.path.join(clf.inference.path, c, m), "filename": c + "_" + m,
                                     "category": c, "id": m, "scan_conf": str(s), "gtfile": os.path.join("gt",str(s))}
                                clf.inference.files.append(d)
        else:
            print("ERROR: not a valid method for getDataset.py")
            sys.exit(1)

        a = 4

    elif(dataset == "synthetic_room_dataset"):

        # TODO: this code should be simplified, because it is basically repeated for training, validation and inference mode
        if(mode == "training"):

            clf.training.path, clf.training.classes, clf.training.scan_confs = getConfig(clf,clf.training)
            clf.training.files = []
            for c in clf.training.classes:
                for s in clf.training.scan_confs:
                    models = np.loadtxt(os.path.join(clf.training.path,c, "train.lst"), dtype=str)[:clf.training.shapes_per_conf_per_class]
                    for m in models:
                        if os.path.exists(os.path.join(clf.training.path, c, m, "gt")):
                            d = {"path": os.path.join(clf.training.path,c,m), "filename": m, "category": c, "id": m, "scan_conf": str(s)}
                            clf.training.files.append(d)

        elif(mode == "validation"):

            clf.validation.path, clf.validation.classes, clf.validation.scan_confs = getConfig(clf,clf.validation)
            clf.validation.files = []
            for c in clf.validation.classes:
                for s in clf.validation.scan_confs:
                    models = np.loadtxt(os.path.join(clf.validation.path,c, "val.lst"), dtype=str)[:clf.validation.shapes_per_conf_per_class]
                    for m in models:
                        if os.path.exists(os.path.join(clf.validation.path, c, m, "gt")):
                            d = {"path": os.path.join(clf.validation.path, c, m), "filename": m, "category": c, "id": m,"scan_conf": str(s)}
                            clf.validation.files.append(d)

        elif(mode == "inference"):
            clf.inference.path, clf.inference.classes, clf.inference.scan_confs = getConfig(clf,clf.inference)
            temp = []
            if(clf.inference.files is not None):
                for f in clf.inference.files:
                    for s in clf.data.scan_confs:
                        n = f.split('_')
                        temp.append({"path": os.path.join(clf.inference.path,n[0]), "filename": n[0]+"_"+n[1], "category":n[0],"id":n[1],"scan_conf":str(s)})
                    clf.inference.files = temp
                return
            else:
                clf.inference.files = []
                for c in clf.inference.classes:
                    for s in clf.data.scan_confs:
                        models = np.loadtxt(os.path.join(clf.inference.path, c, m, "test.lst"), dtype=str)[:clf.inference.shapes_per_conf_per_class]
                        for m in models:
                            if os.path.exists(os.path.join(clf.inference.path,c,"gt")):
                                d = {"path": os.path.join(clf.inference.path, c, m), "filename": m, "category": c,"id": m, "scan_conf": str(s)}
                                clf.inference.files.append(d)
        else:
            print("ERROR: not a valid method for getDataset.py")
            sys.exit(1)

    elif (dataset == "reconbench"):

        if(mode == "validation"):
            if(clf.validation.classes is None):
                clf.validation.classes = ["anchor", "gargoyle", "lordquas", "daratech", "dc"]
            clf.validation.path, _, clf.validation.scan_confs = getConfig(clf, clf.validation)
            clf.validation.files = []
            for s in clf.validation.scan_confs:
                for m in clf.validation.classes:
                    clf.validation.files.append({"path": clf.validation.path,
                                                 "filename": m+"_"+str(s), "category":m,"id":"",
                                                 "scan_conf":str(s), "gtfile": os.path.join("gt",str(s),m+"_"+str(s))})
        elif(mode == "inference"):
            if (clf.inference.classes is None):
                clf.inference.classes = ["anchor", "gargoyle", "lordquas", "daratech", "dc"]
            clf.inference.path, _, clf.inference.scan_confs = getConfig(clf, clf.inference)
            clf.inference.files = []
            for s in clf.inference.scan_confs:
                for m in clf.inference.classes:
                    clf.inference.files.append({"path": clf.inference.path,
                                                "filename": m+"_"+str(s), "category":m,"id":"",
                                                "scan_conf":str(s), "gtfile": os.path.join("gt",str(s),m+"_"+str(s))})
        else:
            print("NOT IMPLEMENTED ERROR: can't train on reconbench dataset!")
            sys.exit(1)
        a=5

    elif(dataset == "ETH3D"):

        # TODO: update this to new layout

        clf.paths.data = "/mnt/raphael/ETH3D/"
        clf.temp.data_path = "/mnt/raphael/ETH3D/"

        clf.temp.inference_files = []
        if (clf.inference.files[0] == "all"):
            clf.inference.files = ["meadow", "terrace", "delivery_area", "kicker", "pipes", "office", \
                                        "playground", "terrains",  "relief", "relief_2", "electro", \
                                        "courtyard", "facade"]
        for i,scene in enumerate(clf.inference.files):
            clf.temp.inference_files.append(os.path.join("/mnt/raphael/ETH3D", scene, "gt", scene+"_lrt_0"))



    elif(dataset == "aerial"):

        if(mode == "training"):

            clf.training.path, _, _ = getConfig(clf,clf.training)
            clf.training.files = []
            models = np.loadtxt(os.path.join(clf.training.path, "train.lst"),dtype=str)[:clf.training.shapes_per_conf_per_class]
            for m in models:
                if os.path.isfile(os.path.join(clf.training.path, "mesh", m+".off")):
                    d = {"path": os.path.join(clf.training.path), "filename": m, "category": m, "id": "", "scan_conf": ""}
                    clf.training.files.append(d)

        elif(mode == "validation"):

            clf.validation.path, _, _ = getConfig(clf,clf.validation)
            clf.validation.files = []
            models = np.loadtxt(os.path.join(clf.validation.path, "test_crop.lst"),dtype=str)[:clf.validation.shapes_per_conf_per_class]
            for m in models:
                if os.path.isfile(os.path.join(clf.validation.path, "mesh", m+".off")):
                    d = {"path": os.path.join(clf.validation.path), "filename": m, "category": m, "id": "", "scan_conf": ""}
                    clf.validation.files.append(d)

        elif(mode == "inference"):

            clf.inference.path, _, _ = getConfig(clf,clf.inference)

            temp = []
            if(clf.inference.files is not None):
                for f in clf.inference.files:
                    for s in clf.data.scan_confs:
                        n = f.split('_')
                        temp.append({"path": os.path.join(clf.inference.path,n[0]), "filename": n[0]+"_"+n[1], "category":n[0],"id":n[1],"scan_conf":str(s)})
                    clf.inference.files = temp
                return
            else:
                clf.inference.files = []
                models = np.loadtxt(os.path.join(clf.inference.path, "test_crop.lst"), dtype=str)[
                         :clf.inference.shapes_per_conf_per_class]
                for m in models:
                    if os.path.isfile(os.path.join(clf.inference.path, "mesh", m + ".off")):
                        d = {"path": os.path.join(clf.inference.path), "filename": m, "category": m, "id": "", "scan_conf": ""}
                        clf.inference.files.append(d)
        else:
            print("ERROR: not a valid method for getDataset.py")
            sys.exit(1)


        a=5


    else:

        if (mode == "validation"):

            clf.validation.path, _, clf.validation.scan_confs = getConfig(clf, clf.validation)
            clf.validation.files = []
            for s in clf.validation.scan_confs:
                clf.validation.files.append({"path": clf.validation.path,
                                             "filename": str(s), "category": "", "id": "",
                                             "scan_conf": str(s),
                                             "gtfile": os.path.join("gt", str(s))})
        elif (mode == "inference"):
            clf.inference.path, _, clf.inference.scan_confs = getConfig(clf, clf.inference)
            clf.inference.files = []
            for s in clf.inference.scan_confs:
                clf.inference.files.append({"path": clf.inference.path,
                                            "filename": str(s), "category": "", "id": "",
                                            "scan_conf": str(s),
                                            "gtfile": os.path.join("gt", str(s))})
        else:
            print("NOT IMPLEMENTED ERROR: can't train on reconbench dataset!")
            sys.exit(1)
        a = 5