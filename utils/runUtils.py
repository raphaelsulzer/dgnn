import os, sys
import pandas as pd
from torch_geometric.data import Data, DataLoader, NeighborSampler
from torch_geometric.utils import add_self_loops


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'learning'))
import surfaceNetEdgePrediction as epsage
import surfaceNetStaticEdgeFilters as efsage

sys.path.append(os.path.join(os.path.dirname(__file__), '', 'processing'))
import data as io

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
                    category=my_loader.category, id=my_loader.id, scan_conf=my_loader.scan_conf)

    if(not clf.model.edge_convs and clf.graph.self_loops):
        torch_dataset.edge_index = add_self_loops(torch_dataset.edge_index)[0]

    ### shuffle has to be False, otherwise inference_layer_batch does not work correctly
    if(clf.temp.batch_size):
        batch_loader = NeighborSampler(edge_index=torch_dataset.edge_index, sizes=clf.temp.clique_size,
                                       batch_size=clf.temp.batch_size, shuffle=False, drop_last=False, return_e_id=clf.model.edge_convs)
    else:
        batch_loader = []

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

def createResultsDF(clf):
    # create the results df
    cols = ['iteration', 'epoch',
               'train_loss_cell', 'train_loss_reg', 'train_loss_total', 'train_OA',
               'test_loss_cell', 'test_loss_reg', 'test_loss_total', 'test_OA',
                'test_current_'+clf.validation.metrics[0], 'test_best_'+clf.validation.metrics[0]]
    clf.results_df = pd.DataFrame(columns=cols)
    clf.temp.load_iteration = 0 # for when load_epoch != 0
    if(clf.validation.metrics[0] == "loss" or clf.validation.metrics[0] == "chamfer"):
        clf.best_metric = 100000
    elif(clf.validation.metrics[0] == "iou" or clf.validation.metrics[0] == "oa"):
        clf.best_metric = 0
    else:
        print("ERROR: {} is not a valid validation metric. Choose either loss, chamfer or iou.".format(clf.validation.metric))
        sys.exit(1)
