import sys, os, argparse, datetime, copy
from shutil import copyfile
import pandas as pd
import gc
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '', 'learning'))
import surfaceNetEdgePrediction as epsage
import surfaceNetStaticEdgeFilters as efsage

import runModel as rm
sys.path.append(os.path.join(os.path.dirname(__file__), '', 'processing'))
from dataset import reduceDataset, getDataset
import generate_mesh as gm
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

    ####################################################################################################################
    ########################################## load train data #########################################################
    ####################################################################################################################
    all_graphs = [];
    my_loader = io.dataLoader(clf)

    print("Load {} graph(s) for training:".format(len(clf.training.files)))
    for mesh in tqdm(clf.training.files, ncols=50):
        # print("\t-",graph.split("/")[-1])
        try:
            my_loader.run(mesh)
            all_graphs.append(Data(x=my_loader.features, y=my_loader.gt, infinite=my_loader.infinite,
                     edge_index=my_loader.edge_lists, edge_attr=my_loader.edge_features, pos=None))
        except:
            print("WARNING: Couldn't load object ",mesh)


    print("\nLoaded graphs:")
    num_nodes = my_loader.getInfo()
    num_train_nodes = int(num_nodes * clf.training.data_percentage)
    train_mask, test_mask = reduceDataset(num_nodes, clf.training.data_percentage)

    print("\t-reduced data to {}% from {} to {} cells".format(clf.training.data_percentage*100, num_nodes, num_train_nodes))

    # make a Munch for all the data
    data = Munch()

    ## get a first batch with batch_size = n_train_files from all the dataset together (the batch includes all the nodes of all graphs)
    temp_loader = DataLoader(all_graphs, batch_size=len(clf.training.files), shuffle=True)
    data.train = Munch()
    data.train.all = next(iter(temp_loader))
    del temp_loader
    gc.collect()

    print("\nSample neighborhoods with:\n\t-clique size {}\n\t-{}+{} hop(s)\n\t-batch size {}\n\t-self loops {}"\
          .format(clf.graph.clique_sizes, clf.graph.num_hops, clf.graph.additional_num_hops, clf.training.batch_size, clf.graph.self_loops))

    clf.graph.num_hops+=clf.graph.additional_num_hops

    if(not clf.model.edge_convs and clf.graph.self_loops):
        data.train.all.edge_index = add_self_loops(data.train.all.edge_index)[0]
    data.train.batches = NeighborSampler(edge_index=data.train.all.edge_index, node_idx=train_mask,
                                   sizes=clf.graph.clique_sizes*clf.graph.num_hops, batch_size=clf.training.batch_size, sampler=None,
                                   shuffle=True, drop_last=True, return_e_id=clf.model.edge_convs)

    ##################################
    ###### load validation data ######
    ##################################
    #### determine subgraph size, see the different inference functions in the model implementation for details
    clf.temp.clique_size = clf.graph.clique_sizes if clf.validation.per_layer else clf.graph.clique_sizes*clf.graph.num_hops
    data.validation = Munch()
    data.validation.all = []
    data.validation.batches = []
    if(clf.validation.files is not None):
        data.validation_names = clf.validation.files
        clf.temp.batch_size = clf.validation.batch_size
        print("\nLoad {} graph(s) for testing:".format(len(data.validation_names)))
        for file in tqdm(clf.validation.files, ncols=50):
            # print("\t-", file)
            _,d,subgraph_sampler = prepareSample(clf, file)
            data.validation.all.append(d)
            data.validation.batches.append(subgraph_sampler)

    ##############################
    ### create (or load) and (re)train model ###
    ##############################
    model = createModel(clf)
    model.to("cuda:" + str(clf.temp.args.gpu))
    print("\nModel:\n",model)
    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))

    if(clf.training.load_epoch):
        # load model
        model_file = os.path.join(clf.paths.out,"models", "model_" + clf.training.load_epoch + ".ptm")
        print("\nLoad existing model at epoch ",clf.training.load_epoch)
        # # load old results df TODO: need to load it from the args.conf directory, not from the clf.files.config file!
        # clf.results_df = pd.read_csv(clf.files.results)
        # res = clf.results_df.loc[clf.results_df['epoch'] == clf.training.load_epoch]
        # print(res)
        if (not os.path.isfile(model_file)):
            print("\nERROR: The model {} does not exist. Check that you have set the correct path in data:out in the config file!".format(model_file))
            sys.exit(1)
        model.load_state_dict(load(model_file))

    ### start training
    if(clf.regularization.edge_epoch):
        print("\nApply edge regularization starting from epoch {}, with weight {}".format(clf.regularization.reg_epoch,clf.regularization.reg_weight))
    print("\nTrain for {} epochs with {} on gpu {}:\n".format(clf.training.epochs, clf.training.loss, clf.temp.args.gpu))
    trainer = rm.Trainer(model)
    trainer.train_test(data, clf)



def inference(clf):
    print("\n######## START INFERENCE OF {} FILE(S) ########".format(len(clf.inference.files)))

    clf.temp.device = "cuda:" + str(clf.temp.args.gpu)
    clf.temp.batch_size = clf.inference.batch_size
    clf.temp.current_epoch = 1000 # needed in case regularization is turned on, to compute reg_loss
    # clf.regularization.reg_epoch = None
    # print("Turn of regularization for inference")

    #### determine subgraph size, see the different inference functions in the model implementation for details
    clf.temp.clique_size = clf.graph.clique_sizes if clf.inference.per_layer else clf.graph.clique_sizes*clf.graph.num_hops

    # load one file to know the feature dimensions
    # TODO: should however be done differently because it is quite cryptic that this
    # is modifying clf which is in turn used to create the model
    my_loader, _,_ = prepareSample(clf, clf.inference.files[0])
    my_loader.getInfo()
    if (clf.temp.batch_size):
        print("\nSample neighborhoods with:\n\t-clique size {}\n\t-{} hop(s)\n\t-batch size {}\n\t-self loops {}" \
              .format(clf.graph.clique_sizes, len(clf.graph.clique_sizes), clf.temp.batch_size, clf.graph.self_loops))

    ##############################
    ######### load model #########
    ##############################
    model_file = os.path.join(clf.paths.out,"models","model_"+str(clf.inference.model)+".ptm")
    print("\nLOAD MODEL {}\n".format(model_file))
    model = createModel(clf)
    if(not os.path.isfile(model_file)):
        print("\nERROR: The model {} does not exist!".format(model_file))
        sys.exit(1)
    ### load model to cpu first, can bug otherwise, see here: https://discuss.pytorch.org/t/cuda-error-out-of-memory-when-load-models/38011/4
    ### to check which device model is on do: https://discuss.pytorch.org/t/cuda-error-out-of-memory-when-load-models/38011/4
    model.load_state_dict(load(model_file,map_location='cpu'))
    model.to(clf.temp.device)
    trainer = rm.Trainer(model)

    ############################################
    ######### Reconstruct and evaluate #########
    ############################################
    results_dict = {}
    df = pd.DataFrame(index=clf.inference.scan_confs, columns=clf.inference.classes)
    df.index.name = "scan_conf"
    results_dict['loss'] = df.copy()
    # results_dict['count'] = df.copy()
    for key in clf.temp.metrics:
        results_dict[key] = df.copy()
    for clf.temp.inference_file in tqdm(clf.inference.files, ncols=50):

        ###############################
        ########## load data ##########
        ###############################
        # try:
        loader, data, subgraph_sampler = prepareSample(clf, clf.temp.inference_file)
        loader.getInfo()
        row = data['scan_conf'] if data['scan_conf'] else str(0)

        prediction = trainer.inference(data, subgraph_sampler, clf)
        if clf.inference.has_label:
            results_dict["loss"].loc[int(row), data["category"]] = clf.inference.metrics.cell_sum/clf.inference.metrics.weight_sum
        if("prediction" in clf.inference.export):
            loader.exportScore(prediction)
        if("mesh" in clf.inference.export):
            mesh, eval_dict = gm.generate(data, prediction, clf)
            for key,value in eval_dict.items():
                results_dict[key].loc[int(row), data["category"]] = value

            # export one shape per class
            mesh.export(os.path.join(clf.paths.out, "generation", data["filename"]+".ply"))

    for key, value in results_dict.items():
        value["mean"] = value.mean(numeric_only=False, axis=1)
        print("{}\n{}\n".format(key,value))




def prepareSample(clf, file):
    """this function only supports loading one sampler per graph (as used for testing and inference)
    but not yet loading one sampler for multiple graphs (as used in training)"""

    verbosity=0
    if(verbosity):
        print("Load data:")
    my_loader = io.dataLoader(clf,verbosity=verbosity)
    my_loader.run(file)


    torch_dataset = Data(x=my_loader.features, y=my_loader.gt, infinite=my_loader.infinite, edge_index=my_loader.edge_lists,
                   edge_attr=my_loader.edge_features, path=my_loader.path, gtfile=my_loader.gtfile, ioufile=my_loader.ioufile,
                    filename= my_loader.filename, category=my_loader.category, id=my_loader.id, scan_conf=my_loader.scan_conf)

    if(not clf.model.edge_convs and clf.graph.self_loops):
        torch_dataset.edge_index = add_self_loops(torch_dataset.edge_index)[0]


    start = datetime.datetime.now()
    ### shuffle has to be False, otherwise inference_layer_batch does not work correctly
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
    # elif(clf.model.type == "sage" and not clf.model.edge_prediction and clf.model.edge_convs):
    elif(clf.model.type == "sage" and not clf.model.edge_prediction):
        model = efsage.SurfaceNet(clf=clf)  # this works with and without edge features
    # elif (clf.model.type == "sage" and clf.model.edge_convs):
    #     if (clf.model.concatenate):
    #         model = cefsage.SurfaceNet(n_node_features=fs, clf=clf)
    #     else:
    #         model = efsage.SurfaceNet(n_node_features=fs, clf=clf)
    # elif(clf.model.type == "sage" and not clf.model.edge_prediction and not clf.model.edge_convs):
    #     model = sage.SurfaceNet(clf=clf)
    else:
        print("\nERROR: {} is not a valid model_name".format(clf.model.type))
        sys.exit(1)

    return model





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train SurfaceNet')

    parser.add_argument('-t', '--training', action='store_true',
                        help='do training')
    parser.add_argument('-i', '--inference', action='store_true',
                        help='do inference after training')
    parser.add_argument('-c', '--conf', type=str, default="configs/pretrained/reconbench.yaml",
                        help='which config to load')
    parser.add_argument('--gpu', type=int, default=0,
                        help='on which gpu device [0,1] to train. default: 0')
    args = parser.parse_args()

    # args.conf = 'reconbench'

    if(not os.path.isabs(args.conf)):
        args.conf = os.path.join(os.path.dirname(__file__),args.conf)

    ################# load conf #################
    clf = Munch.fromYAML(open(args.conf, 'r'))


    clf.temp = Munch()
    clf.data = Munch()
    clf.files = Munch()
    clf.temp.args = args

    ### adjust paths to current working_dir if they are relative
    if(not os.path.isabs(clf.paths.out)):
        clf.paths.out = os.path.join(os.path.dirname(__file__),clf.paths.out)
    if(not os.path.isabs(clf.paths.data)):
        clf.paths.data = os.path.join(os.path.dirname(__file__),clf.paths.data)


    ################# create the model dir #################
    os.makedirs(os.path.join(clf.paths.out,"generation"), exist_ok=True)
    os.makedirs(os.path.join(clf.paths.out,"prediction"), exist_ok=True)
    os.makedirs(os.path.join(clf.paths.out,"metrics"), exist_ok=True)
    os.makedirs(os.path.join(clf.paths.out,"models"), exist_ok=True)
    # save conf file to out
    clf.files.config = os.path.join(clf.paths.out,"config.yaml")
    clf.files.results = os.path.join(clf.paths.out,"metrics","results.csv")

    # create the results df
    cols = ['iteration', 'epoch',
               'train_loss_cell', 'train_loss_reg', 'train_loss_total', 'train_OA',
               'test_loss_cell', 'test_loss_reg', 'test_loss_total', 'test_OA',
                'test_current_'+clf.validation.metrics[0], 'test_best_'+clf.validation.metrics[0]]
    clf.results_df = pd.DataFrame(columns=cols)
    if(clf.validation.metrics[0] == "loss" or clf.validation.metrics[0] == "chamfer"):
        clf.best_metric = 100000
    elif(clf.validation.metrics[0] == "iou" or clf.validation.metrics[0] == "oa"):
        clf.best_metric = 0
    else:
        print("ERROR: {} is not a valid validation metric. Choose either loss, chamfer or iou.".format(clf.validation.metric))
        sys.exit(1)


    ################# print time before training/classification #################
    clf.temp.start = datetime.datetime.now()
    print(clf.temp.start)
    # clf.time.start = str(clf.temp.start)
    print("READ CONFIG FROM ", args.conf)
    print("SAVE CONFIG TO ", clf.files.config)

    ############ TRAINING #############
    if(args.training):
        clf.temp.graph_cut = clf.validation.graph_cut; clf.temp.fix_orientation = clf.validation.fix_orientation; clf.temp.metrics = clf.validation.metrics
        # log output
        # copy conf file to out dir
        copyfile(args.conf, clf.files.config)
        # log output
        sys.stdout = Logger(os.path.join(clf.paths.out, "log.txt"))
        clf.temp.logger = sys.stdout
        # save all training print outputs to the .log file below
        getDataset(clf,clf.training.dataset,"training")
        getDataset(clf,clf.validation.dataset,"validation")
        training(clf)

    ############ INFERENCE #############
    clf.temp.memory = []
    clf.temp.inference_time = 0
    clf.temp.subgraph_time = 0
    if(args.inference):
        clf.temp.graph_cut = clf.inference.graph_cut; clf.temp.fix_orientation = clf.inference.fix_orientation; clf.temp.metrics = clf.inference.metrics
        ############ INFERENCE #############
        getDataset(clf, clf.inference.dataset, "inference")
        inference(clf)


    # print("peak memory: ",max(clf.temp.memory))
    # print("inference time in sec: ", clf.temp.inference_time)
    # print("subgraph time in sec: ", clf.temp.subgraph_time)

    # print time after training/classification
    clf.temp.end = datetime.datetime.now()
    print(clf.temp.end)
    print("FINISHED AFTER {} seconds".format((clf.temp.end - clf.temp.start).seconds))
    print("THE CONFIG WAS ", args.conf)

    del clf.temp