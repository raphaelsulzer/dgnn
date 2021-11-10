import sys, os, argparse, datetime, copy
from shutil import copyfile
import pandas as pd

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '', 'learning'))
import surfaceNetEdgePrediction as epsage
import surfaceNetStaticEdgeFilters as efsage

import runModel as rm
sys.path.append(os.path.join(os.path.dirname(__file__), '', 'generation'))
import generate_mesh as gm
sys.path.append(os.path.join(os.path.dirname(__file__), '', 'data'))
from dataset import reduceDataset, getDataset
sys.path.append(os.path.join(os.path.dirname(__file__), '', 'utils'))
from log import Logger
import data as io

from tqdm import tqdm
from munch import *
from torch import load
from torch_geometric.data import Data, DataLoader, NeighborSampler
from torch_geometric.utils import add_self_loops
import warnings
warnings.filterwarnings('ignore') # to supress the CUDA titan black X warning


def training(clf):

    print("\n######## TRAINING MODEL ########")

    clf.temp.lr = 5e-3  # adjust learning rate to lr
    clf.temp.lr_sd = 8  # adjust learning rate for lr every lr_sd epochs
    clf.temp.start_epoch = 1
    clf.temp.reg_epoch = clf.regularization.reg_epoch

    #############################
    ###### load train data ######
    #############################
    all_graphs = []; num_nodes = 0
    my_loader = io.dataLoader(clf)

    print("Load {} graph(s) for training:".format(len(clf.training.files)))
    for graph in tqdm(clf.training.files, ncols=50):
        # print("\t-",graph.split("/")[-1])
        my_loader.run(graph)
        all_graphs.append(Data(x=my_loader.features, y=my_loader.gt,
                 edge_index=my_loader.edge_lists, edge_attr=my_loader.edge_features, pos=None))


    print("\nLoaded graphs:")
    if(clf.data.dataset == 'shapenet'):
        print("\t-confs {}: {}".format(clf.data.scan_confs, clf.temp.total_shapes_per_conf))
    elif(clf.data.dataset == 'myshapenet'):
        print("\t-confs {}: {}".format(clf.data.scan_confs, clf.temp.total_shapes_per_class))
    num_nodes = my_loader.getInfo()
    num_train_nodes = int(num_nodes * clf.training.data_percentage)
    train_mask, test_mask = reduceDataset(num_nodes, clf.training.data_percentage)

    print("\t-reduced data to {}% from {} to {} cells".format(clf.training.data_percentage*100, num_nodes, num_train_nodes))

    ## get a first batch with batch_size = n_train_files from all the dataset together (the batch includes all the nodes of all graphs)
    temp_loader = DataLoader(all_graphs, batch_size=len(clf.training.files), shuffle=True)
    data = Munch()
    data.sampler = Munch()
    data.train = next(iter(temp_loader))

    print("\nSample neighborhoods with:\n\t-clique size {}\n\t-{}+{} hops\n\t-batch size {}\n\t-self loops {}"\
          .format(clf.graph.clique_sizes, clf.graph.num_hops, clf.graph.additional_num_hops, clf.training.batch_size, clf.graph.self_loops))
    clf.temp.clique_size=clf.graph.clique_sizes*(clf.graph.num_hops+clf.graph.additional_num_hops)

    if(not clf.model.edge_convs and clf.graph.self_loops):
        data.train.edge_index = add_self_loops(data.train.edge_index)[0]
    data.sampler.train = NeighborSampler(edge_index=data.train.edge_index, node_idx=train_mask,
                                   sizes=clf.temp.clique_size, batch_size=clf.training.batch_size, sampler=None,
                                   shuffle=True, drop_last=True, return_e_id=clf.model.edge_convs)

    ############################
    ###### load validation data ######
    ############################
    data.sampler.validation = []
    data.validation = []
    if(clf.validation.files is not None):
        data.validation_names = clf.validation.files
        clf.temp.batch_size = clf.inference.batch_size
        print("\nLoad {} graph(s) for testing:".format(len(data.validation_names)))
        for file in tqdm(clf.validation.files, ncols=50):
            # print("\t-", file)
            _,d,subgraph_sampler = prepareSample(clf, file)
            data.validation.append(d)
            data.sampler.validation.append(subgraph_sampler)

    ##############################
    ### create and train model ###
    ##############################
    model = createModel(clf)
    model.to("cuda:" + str(clf.temp.args.gpu))
    print("\nModel:\n",model)
    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    # start training
    if(clf.regularization.reg_epoch):
        print("\nApply regularization starting from epoch {}, with cc weight {}".format(clf.regularization.reg_epoch, clf.regularization.cc))
    print("\nTrain for {} epochs with {} on gpu {}:\n".format(clf.training.epochs, clf.training.loss, clf.temp.args.gpu))
    rm.train_test(model, data, clf)



def retraining(clf):
    print("\n######## TRAINING MODEL ########")

    clf.temp.lr = 5e-3  # adjust learning rate to lr
    clf.temp.lr_sd = 8  # adjust learning rate for lr every lr_sd epochs

    #############################
    ###### load train data ######
    #############################
    all_graphs = []; num_nodes = 0
    my_loader = io.dataLoader(clf)

    print("Load {} graph(s) for training:".format(len(clf.training.files)))
    for graph in tqdm(clf.training.files, ncols=50):
        # print("\t-",graph.split("/")[-1])
        my_loader.run(graph)
        all_graphs.append(Data(x=my_loader.features, y=my_loader.gt,
                 edge_index=my_loader.edge_lists, edge_attr=my_loader.edge_features, pos=None))


    print("\nLoaded graphs:")
    if(clf.data.dataset == 'shapenet'):
        print("\t-confs {}: {}".format(clf.data.scan_confs, clf.temp.total_shapes_per_conf))
    elif(clf.data.dataset == 'myshapenet'):
        print("\t-confs {}: {}".format(clf.data.scan_confs, clf.temp.total_shapes_per_class))
    num_nodes = my_loader.getInfo()
    num_train_nodes = int(num_nodes * clf.training.data_percentage)
    train_mask, test_mask = reduceDataset(num_nodes, clf.training.data_percentage)

    print("\t-reduced data to {}% from {} to {} cells".format(clf.training.data_percentage*100, num_nodes, num_train_nodes))

    ## get a first batch with batch_size = n_train_files from all the dataset together (the batch includes all the nodes of all graphs)
    temp_loader = DataLoader(all_graphs, batch_size=len(clf.training.files), shuffle=True)
    data = Munch()
    data.sampler = Munch()
    data.train = next(iter(temp_loader))

    print("\nSample neighborhoods with:\n\t-clique size {}\n\t-{}+{} hops\n\t-batch size {}\n\t-self loops {}"\
          .format(clf.graph.clique_sizes, clf.graph.num_hops, clf.graph.additional_num_hops, clf.training.batch_size, clf.graph.self_loops))
    clf.temp.clique_size=clf.graph.clique_sizes*(clf.graph.num_hops+clf.graph.additional_num_hops)

    if(not clf.model.edge_convs and clf.graph.self_loops):
        data.train.edge_index = add_self_loops(data.train.edge_index)[0]
    data.sampler.train = NeighborSampler(edge_index=data.train.edge_index, node_idx=train_mask,
                                   sizes=clf.temp.clique_size, batch_size=clf.training.batch_size, sampler=None,
                                   shuffle=True, drop_last=True, return_e_id=clf.model.edge_convs)


    ############################
    ###### load test data ######
    ############################
    data.sampler.validation = []
    data.validation = []
    if(clf.validation.files is not None):
        data.validation_names = clf.validation.files
        clf.temp.batch_size = clf.validation.batch_size
        print("\nLoad {} graph(s) for validation:".format(len(data.validation_names)))
        for file in clf.validation.files:
            print("\t-", file)
            _,data,subgraph_sampler = prepareSample(clf, file)
            data.validation.append(data)
            data.sampler.validation.append(subgraph_sampler)

    data.sampler.validation.append(NeighborSampler(edge_index=data.train.edge_index, node_idx=test_mask,
                                   sizes=clf.temp.clique_size, batch_size=clf.validation.batch_size, sampler=None,
                                   shuffle=True, drop_last=True, return_e_id=clf.model.edge_convs))


    ##############################
    ######### load model #########
    ##############################
    model = createModel(clf)

    model.to("cuda:" + str(clf.temp.args.gpu))

    print("\nModel:\n",model)
    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print("\nReTrain from epoch {} another {} epochs on gpu {}:\n".format(clf.retraining.load_epoch,clf.retraining.additional_epochs,clf.temp.args.gpu))

    model_file = os.path.join(clf.paths.out_dir,"{}_{}.ptm".format(clf.temp.args.conf,str(clf.inference.epoch)))
    if(not os.path.isfile(model_file)):
        print("\nERROR: The model {} does not exist. Check your path and the UUID of the model!".format(model_file))
        sys.exit(1)
    model.load_state_dict(load(model_file))

    if(clf.regularization.reg_epoch):
        print("\nApply regularization starting from epoch {}, with cc weight {}".format(clf.regularization.reg_epoch, clf.regularization.cc))
    clf.temp.reg_epoch = clf.regularization.reg_epoch
    clf.temp.start_epoch = clf.retraining.load_epoch
    clf.training.epochs = clf.retraining.load_epoch + clf.retraining.additional_epochs
    rm.train_test(model, data, clf)



def inference(clf):
    print("\n######## START INFERENCE OF {} FILES ########".format(len(clf.inference.files)))

    clf.temp.device = "cuda:" + str(clf.temp.args.gpu)
    clf.temp.print_cm = True
    # clf.temp.current_epoch = clf.inference.epoch # needed for the loss calculation
    clf.temp.batch_size = clf.inference.batch_size
    clf.temp.reg_epoch = None
    if(clf.inference.per_layer):
        clf.temp.clique_size = clf.graph.clique_sizes
    else:
        clf.temp.clique_size = clf.graph.clique_sizes*clf.graph.num_hops

    # load one file to know the feature dimensions
    my_loader, _,_ = prepareSample(clf, clf.inference.files[0])
    my_loader.getInfo()

    ##############################
    ######### load model #########
    ##############################
    model_file = os.path.join(clf.paths.out_dir,"model_"+clf.inference.model+".ptm")
    print("\nLOAD MODEL {}\n".format(model_file))
    model = createModel(clf)
    model.to(clf.temp.device)
    if(not os.path.isfile(model_file)):
        print("\nERROR: The model {} does not exist!".format(model_file))
        sys.exit(1)
    model.load_state_dict(load(model_file))
    print("\nTurn of regularization for inference")

    iou_all = 0
    count = 0
    # for clf.temp.inference_file in tqdm(clf.inference.files, ncols=50):
    for clf.temp.inference_file in clf.inference.files:

        ###############################
        ########## load data ##########
        ###############################
        # try:
        loader, data, subgraph_sampler = prepareSample(clf, clf.temp.inference_file)
        loader.getInfo()
        if (clf.temp.batch_size):
            print("\nSample neighborhoods with:\n\t-clique size {}\n\t-{} hops\n\t-batch size {}\n\t-self loops {}" \
                  .format(clf.graph.clique_sizes, clf.graph.num_hops, clf.temp.batch_size, clf.graph.self_loops))


        prediction = rm.inference(model, data, subgraph_sampler, clf)
        my_loader.exportScore(prediction)

        mesh, iou = gm.generate(data, prediction, clf)
        print("Mesh {} : IoU {}".format(data["category"]+"_"+data["id"],iou))
        iou_all += iou
        count += 1
        # export one shape per class
        mesh.export(os.path.join(clf.paths.out_dir, "generation", data["category"] + "_" + data['id'] + ".ply"))

        # cuda.empty_cache()


    print("Mean IoU: ", iou_all/count)
    clf.time.end = str(datetime.datetime.now())




def prepareSample(clf, file):
    """this function only supports loading one sampler per graph (as used for testing and inference)
    but not yet loading one sampler for multiple graphs (as used in training)"""

    # filename = file.split('_')
    # if(filename[-2] == 'lrtcs'):
    #     clf.temp.with_label = True
    # elif(filename[-2] == 'lrt'):
    #     clf.temp.with_label = False
    # else:
    #     print('not sure if this file has labels or not. set the clf.with_label parameter!')
    #     sys.exit(1)

    verbosity=0
    if(verbosity):
        print("Load data:")
    my_loader = io.dataLoader(clf,verbosity=verbosity)
    my_loader.run(file)


    torch_dataset = Data(x=my_loader.features, y=my_loader.gt, edge_index=my_loader.edge_lists,
                   edge_attr=my_loader.edge_features, category=my_loader.category, id=my_loader.id)

    if(not clf.model.edge_convs and clf.graph.self_loops):
        torch_dataset.edge_index = add_self_loops(torch_dataset.edge_index)[0]


    start = datetime.datetime.now()
    if(clf.temp.batch_size):
        batch_loader = NeighborSampler(edge_index=torch_dataset.edge_index, sizes=clf.temp.clique_size,
                                       batch_size=clf.temp.batch_size, shuffle=False, drop_last=False, return_e_id=clf.model.edge_convs)
    else:
        batch_loader = []
    stop = datetime.datetime.now()
    # clf.temp.subgraph_time += ((stop - start).seconds)

    return my_loader, torch_dataset, batch_loader





def createModel(clf):

    if (clf.model.type == "sage" and clf.model.edge_prediction):
        model = epsage.SurfaceNet(clf=clf)
    elif(clf.model.type == "sage" and not clf.model.edge_prediction):
        model = efsage.SurfaceNet(clf=clf)
    # elif (clf.model.type == "sage" and clf.model.edge_convs):
    #     if (clf.model.concatenate):
    #         model = cefsage.SurfaceNet(n_node_features=fs, clf=clf)
    #     else:
    #         model = efsage.SurfaceNet(n_node_features=fs, clf=clf)
    # elif(clf.model.type == "sage" and not clf.model.edge_prediction and not clf.model.edge_features):
    #     model = sage.SurfaceNet(n_node_features=fs, clf=clf)
    else:
        print("\nERROR: {} is not a valid model_name".format(clf.model.type))
        sys.exit(1)

    return model





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train SurfaceNet')

    parser.add_argument('-t', '--training', action='store_true',
                        help='do training')
    parser.add_argument('-r', '--retraining', action='store_true',
                        help='do retraining')
    parser.add_argument('-i', '--inference', action='store_true',
                        help='do inference after training')
    parser.add_argument('-c', '--conf', type=str, default="../configs/debug.yaml",
                        help='which config to load')
    parser.add_argument('--gpu', type=int, default=0,
                        help='on which gpu device [0,1] to train. default: 0')
    args = parser.parse_args()

    # args.conf = 'reconbench'


    ################# load conf #################
    clf = Munch.fromYAML(open(args.conf, 'r'))


    clf.temp = Munch()
    clf.data = Munch()
    clf.temp.args = args


    ################# create the model dir #################
    if(not os.path.exists(os.path.join(clf.paths.out_dir,"prediction"))):
        os.makedirs(os.path.join(clf.paths.out_dir,"prediction"))
    if(not os.path.exists(os.path.join(clf.paths.out_dir,"generation"))):
        os.makedirs(os.path.join(clf.paths.out_dir,"generation"))
    # save conf file to out_dir
    if(not os.path.isfile(os.path.join(clf.paths.out_dir,"config.yaml"))):
        copyfile(args.conf,os.path.join(clf.paths.out_dir,"config.yaml"))


    ################# print time before training/classification #################
    clf.temp.start = datetime.datetime.now()
    print(clf.temp.start)
    clf.time.start = str(clf.temp.start)
    print("READ CONFIG FROM ", args.conf)

    ############ TRAINING #############
    clf.temp.mode = "training"
    clf.data.dataset = clf.training.dataset
    clf.data.classes = clf.training.classes
    clf.data.scan_confs = clf.training.scan_confs
    clf.temp.shapes_per_conf_per_class = clf.training.shapes_per_conf_per_class
    if (args.training):
        # save all training print outputs to the .log file below
        sys.stdout = Logger(os.path.join(clf.paths.out_dir, "log.txt"))
        clf.temp.logger = sys.stdout
        getDataset(clf)  # dataset for learning, coming from .yaml
        training(clf)
    elif(args.retraining):
        # save all training print outputs to the .log file below
        sys.stdout = Logger(os.path.join(clf.paths.out_dir, args.conf + ".log"))
        clf.temp.logger = sys.stdout
        getDataset(clf)  # dataset for learning, coming from .yaml
        retraining(clf)

    ############ INFERENCE #############
    clf.temp.mode = "inference"
    clf.data.dataset = clf.inference.dataset
    clf.data.classes = clf.inference.classes
    clf.data.scan_confs = clf.inference.scan_confs
    getDataset(clf)
    clf.temp.memory = []
    clf.temp.inference_time = 0
    clf.temp.subgraph_time = 0
    if(args.inference):
        ############ INFERENCE #############
        clf.results.OA_test = 0.0
        if (clf.inference.dataset == "eth3d"):
            clf.paths.data = os.path.join(clf.temp.data_path, clf.temp.inference_file.split("/")[-3])
        inference(clf)


    # print("peak memory: ",max(clf.temp.memory))
    print("inference time in sec: ", clf.temp.inference_time)
    print("subgraph time in sec: ", clf.temp.subgraph_time)

    # print time after training/classification
    clf.temp.end = datetime.datetime.now()
    print(clf.temp.end)
    print("FINISHED AFTER {} seconds".format((clf.temp.end - clf.temp.start).seconds))
    print("THE CONFIG WAS ", args.conf)

    del clf.temp