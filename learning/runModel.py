import sys, os
import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
from munch import Munch
import torchnet as tnt
import subprocess as sp
import datetime

import trimesh

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'generation'))
import generate_mesh as gm

class ConfusionMatrix:
    def __init__(self, n_class, class_names):
        self.CM = np.zeros((n_class, n_class))
        self.n_class = n_class
        self.class_names = class_names

    def clear(self):
        self.CM = np.zeros((self.n_class, self.n_class))

    def add_batch(self, gt, pred):
        self.CM += confusion_matrix(gt, pred, labels=list(range(self.n_class)))

    def overall_accuracy(self):  # percentage of correct classification
        # for normalized CM
        # acc=0
        # for i in range(self.n_class):
        #  acc+=self.CM[i,i]
        return 100*np.trace(self.CM) / np.sum(self.CM)

    def class_IoU(self, show=0):
        ious = np.full(self.n_class, 0.)
        for i_class in range(self.n_class):
            ious[i_class] = self.CM[i_class, i_class] / (
                        np.sum(self.CM[:, i_class]) + np.sum(self.CM[i_class, :]) - self.CM[i_class, i_class])
        if show:
            print('  |  '.join('{} : {:3.2f}%'.format(name, 100 * iou) for name, iou in zip(self.class_names, ious)))
        # we do not count classes that are not present in the dataset in the mean IoU
        return 100 * np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()

class Metrics():

    def __init__(self):

        self.samples_sum = 0
        self.OA_sum = 0

        self.cell_sum = 0
        self.weight_sum = 0

        self.reg_sum = 0
        self.edges_sum = 0

    def addOAItem(self,OA,n):
        self.OA_sum+=OA
        self.samples_sum+=float(n)
    def getOA(self):
        return self.OA_sum*100/self.samples_sum

    def addCellLossItem(self,cell,weight):
        self.cell_sum+=cell
        self.weight_sum+=weight
    def getCellLoss(self):
        return (self.cell_sum/self.weight_sum).item() # it's a tensor, so just return the value

    def addRegLossItem(self,reg,n):
        self.reg_sum+=reg
        self.edges_sum+=n
    def getRegLoss(self):
        if(self.reg_sum > 0.0 and self.edges_sum > 0.0):
            return (self.reg_sum/self.edges_sum).item() # it's a tensor, so just return the value
        else:
            return 0

def get_gpu_memory(device):
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    device=int(device.split(":")[1])

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

    capacity = 12212

    return capacity-memory_free_values[device]

def adjust_learning_rate(optimizer, clf):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = clf.temp.lr * (0.1 ** (clf.temp.current_epoch // clf.temp.lr_sd))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calcRegularization(model, logits_cell, data_all, clf):

    # this works because layer K-n embeddings are actually calculate for all layers K-m
    # with m=[n,..,K]; not just for m = n
    # it is done this way to apply skip connections

    # this current implementation is done for additional_hops = 2
    # it should already work with additional_hops = 1, with
    inner_prediction = F.log_softmax(logits_cell[:data_all.adjs[model.num_layers].size[0]], dim=-1)
    surface_triangle_prob_kl = torch.abs(inner_prediction[data_all.adjs[model.num_layers].edge_index[0, :]][:, 0] \
                                         - inner_prediction[data_all.adjs[model.num_layers].edge_index[1, :]][:, 0])


    ## for their source and target nodes
    # li=inner_prediction[adjs[model.num_layers+1].edge_index[0,:]][:,0]
    # ri=inner_prediction[adjs[model.num_layers+1].edge_index[1,:]][:,0]
    # lo=inner_prediction[adjs[model.num_layers+1].edge_index[0,:]][:,1]
    # ro=inner_prediction[adjs[model.num_layers+1].edge_index[1,:]][:,1]
    # inner_subgraph_indicies = adjs[model.num_layers+1].size[0]
    # inner_prediction = F.log_softmax(logits_cell[:data_all.adjs[model.num_layers + 1].size[0]], dim=-1)
    # ## now take only the edge of the inner subgraph, because only for these edges, I have predictions
    # surface_triangle_prob_kl = torch.abs(inner_prediction[data_all.adjs[model.num_layers + 1].edge_index[0, :]][:, 0] \
    #                                      - inner_prediction[data_all.adjs[model.num_layers + 1].edge_index[1, :]][:, 0])

    # surface_area = data_all.edge_attr[data_all.adjs[model.num_layers + 1].e_id].squeeze()[:, 0].to(clf.temp.device)
    weight = data_all.edge_attr[data_all.adjs[model.num_layers].e_id].squeeze()[:, 0].to(clf.temp.device)
    # area_loss = torch.mean(surface_triangle_prob_kl * surface_area) * clf.regularization.area
    # angle_loss = torch.mean(surface_triangle_prob_kl * angle) * clf.regularization.angle

    reg_loss = surface_triangle_prob_kl * weight * clf.regularization.reg_weight

    reg_loss = reg_loss.sum() / weight.sum()

    clf.temp.metrics.addRegLossItem(reg_loss, surface_triangle_prob_kl.size()[0])

    return reg_loss


def calcLossAndOA(model, logits_cell, logits_edge, data_all, clf):

    ###############################################################
    ########################## cell loss ##########################
    ###############################################################
    if (clf.regularization.cell_reg_type):

        if (clf.training.loss == "kl"):
            cell_loss = F.kl_div(F.log_softmax(logits_cell, dim=-1), data_all.gt_batch[:, :2].to(clf.temp.device), reduction='none')
            cell_loss = torch.sum(cell_loss, dim=1)  # cf. formula for kl_div, summing over X (the dimensions)
            clf.temp.metrics.addOAItem(
                torch.sum(data_all.gt_batch[:, 2] == F.log_softmax(logits_cell, dim=-1).argmax(1).cpu()).item(),
                data_all.x_batch.shape[0])
        elif(clf.training.loss == "bce"):
            # supervise with graph cut label
            cell_loss = F.binary_cross_entropy_with_logits(logits_cell.squeeze(dim=-1), data_all.gt_batch[:, 3].to(clf.temp.device), reduction='none')
            clf.temp.metrics.addOAItem(
                torch.sum(data_all.gt_batch[:, 3] == torch.round(F.sigmoid(logits_cell.squeeze(dim=-1))).cpu()).item(),
                data_all.x_batch.shape[0])
        elif (clf.training.loss == "mse"):
            cell_loss = F.mse_loss(F.sigmoid(logits_cell).squeeze(), data_all.gt_batch[:, 0].to(clf.temp.device))
        else:
            print("{} is not a valid loss. choose either kl or mse".format(clf.training.loss))
            sys.exit(1)

        # multiply loss per cell with volume of the cell
        shape_weight = data_all.x_batch[:, 0].to(clf.temp.device)
        # shape_weight =  torch.sqrt(x_batch[:,0].to(clf.temp.device))
        # shape_weight =  torch.log(1+x_batch[:,0].to(clf.temp.device))

        cell_loss = cell_loss * shape_weight

        # add loss to metrics for statistics
        clf.temp.metrics.addCellLossItem(cell_loss.sum(),shape_weight.sum())
        # only works if additional_hops > 0


        cell_loss = cell_loss.sum() / shape_weight.sum()


        # # final loss is mean over all batch entries
        # if (clf.regularization.shape_weight_batch_normalization):
        #     cell_loss = cell_loss.sum() / shape_weight.sum()
        # else:
        #     if (clf.regularization.inside_outside_weight[0] != clf.regularization.inside_outside_weight[1]):
        #         io_weight = data_all.gt_batch[:, 1].to(clf.temp.device) * clf.regularization.inside_outside_weight[0] + \
        #                     data_all.gt_batch[:, 2].to(clf.temp.device) * clf.regularization.inside_outside_weight[1]
        #         cell_loss = cell_loss * io_weight
        #         cell_loss = cell_loss.sum() / io_weight.sum()
        #     else:
        #         cell_loss = torch.mean(cell_loss)*10**6
    # else: # without any normalization of the loss:
    #     cell_loss = F.kl_div(F.log_softmax(logits_cell, dim=-1), data_all.gt_batch[:, 1:].to(clf.temp.device), reduction='batchmean')
    loss = cell_loss

    if ((loss != loss).any()):
        print("nan in loss")



    ###############################################################
    ########################## edge loss ##########################
    ###############################################################
    # only works if additional_hops > 0
    edge_loss=0
    if(logits_edge is not None):
        gt_left = data_all.y[data_all.adjs[model.num_layers].edge_index[0]][:,1]
        gt_right = data_all.y[data_all.adjs[model.num_layers].edge_index[1]][:,1]
        edge_loss=F.kl_div(F.log_softmax(logits_edge).squeeze(),(gt_left-gt_right).to(clf.temp.device))
        if((edge_loss != edge_loss).any()):
            print("here")
        # TODO: get inner ground truth and inner edges and calc an edge_loss
        loss+=edge_loss

    ###############################################################################################################
    ############################ area+angle+cc loss / total variation / regularization ############################
    ###############################################################################################################
    if(clf.temp.reg_epoch is not None): # check if reg_epoch is not null
        if(clf.graph.additional_num_hops != 1):
            print("ERROR: clf.graph.additional_num_hops == 1 to use regularization")
            sys.exit(1)
        if (clf.temp.current_epoch >= clf.temp.reg_epoch):
            reg_loss = calcRegularization(model,logits_cell,data_all,clf)
            loss+=reg_loss

    # return loss for mini batch gradient decent
    return loss

####################################################
############## TRAINING AND TESTING ################
####################################################
def train(model, data_all, batch_loader, optimizer, clf):

    model.train() #switch the model in training mode
    clf.temp.metrics = Metrics()
    for batch_size, n_id, adjs in tqdm(batch_loader, ncols=50):
        data_all.n_id = n_id
        data_all.adjs = adjs
        data_all.x_batch = data_all.x[n_id[:adjs[model.num_layers-1].size[1]]]
        data_all.gt_batch = data_all.y[n_id[:adjs[model.num_layers-1].size[1]]]
        # outer_subgraph_indicies = adjs[model.num_layers].size[0] = adjs[model.num_layers-1].size[1]

        logits_edge = None
        if(clf.model.edge_prediction):
            logits_cell, logits_edge = model(data_all)
        else:
            logits_cell = model(data_all)
        loss = calcLossAndOA(model, logits_cell, logits_edge, data_all, clf)

        optimizer.zero_grad()  # put gradient to zero
        loss.backward()
        # for p in model.parameters():  # we clip the gradient at norm 1
        #     p.grad.data.clamp_(-1, 1)
        optimizer.step()  # one SGD (=stochastic gradient descent) step


# Out[1]: torch.Size([3238964, 11])
# x[:size[1]].size()
# Out[2]: torch.Size([2668722, 11])
def test(model, data_all, batch_loader, clf):

    model.eval()  # batchnorms in eval mode
    clf.temp.metrics = Metrics()  # init new one for test
    for batch_size, n_id, adjs in batch_loader:
        data_all.n_id = n_id
        data_all.adjs = adjs
        data_all.x_batch = data_all.x[n_id[:adjs[model.num_layers-1].size[1]]]
        data_all.gt_batch = data_all.y[n_id[:adjs[model.num_layers-1].size[1]]]
        with torch.no_grad():
            logits_edge = None
            if(clf.model.edge_prediction):
                logits_cell, logits_edge = model(data_all)
            else:
                logits_cell = model(data_all)
            calcLossAndOA(model, logits_cell, logits_edge, data_all, clf)


def train_test(model, data, clf):

    clf.temp.device = "cuda:" + str(clf.temp.args.gpu)

    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=clf.temp.lr)

    # define some color to help distinguish between train and test outputs
    TESTCOLOR = '\033[1;0;46m'
    FULLTESTCOLOR = '\033[1;0;44m'
    EXPORTCOLOR = '\033[1;0;42m'
    NORMALCOLOR = '\033[0m'

    epoch_iou = []

    for current_epoch in range(clf.temp.start_epoch,clf.training.epochs+1):
        # adjust learning rate to current epoch
        clf.temp.current_epoch = current_epoch
        adjust_learning_rate(optimizer, clf)
        # train one epoch
        train(model, data.train, data.sampler.train, optimizer, clf)
        print('Epoch %3d -> Train OA cell: %3.2f%%,  Train Loss (cell): %1.4f,  Train Loss (reg): %1.4f\n'
            % (current_epoch,
               clf.temp.metrics.getOA(),
            clf.temp.metrics.getCellLoss(),
               clf.temp.metrics.getRegLoss()))
        # this is for saving the train loss of the last epoch
        clf.results.loss_train = clf.temp.metrics.getCellLoss()

        # do testing (ie inference in my case), if this is a test epoch
        if current_epoch % clf.validation.val_every == 0:

            OA = 0;loss = 0;samples = 0;weight = 0;reg = 0;edges = 0; iou = 0;
            for i,d in tqdm(enumerate(data.validation), ncols=50):
                prediction = inference(model,d,[],clf)
                temp=gm.generate(d, prediction, clf)
                iou+=temp[1]
                # export one shape per class
                if(i%clf.validation.shapes_per_conf_per_class == 0):
                    temp[0].export(os.path.join(clf.paths.out_dir,"generation",d["category"]+"_"+d['id']+".ply"))
                # keep track of metrics over all scenes
                OA+= clf.temp.metrics.OA_sum; samples+=clf.temp.metrics.samples_sum
                loss+= clf.temp.metrics.cell_sum; weight+=clf.temp.metrics.weight_sum
                reg+= clf.temp.metrics.reg_sum; edges+=clf.temp.metrics.edges_sum

            if(reg>0.0 and edges>0.0):
                re = reg/edges
            else:
                re = 0.0

            iou = iou*100/len(data.validation)
            if(iou > clf.results.iou):
                clf.results.iou = iou
                model_path = os.path.join(clf.paths.out_dir,"model_best.ptm")
                torch.save(model.state_dict(), model_path)


            epoch_iou.append((current_epoch*len(data.sampler.train),iou))

            print('Epoch %3d -> Mean OA cell: %3.2f%%, Mean IoU: %3.2f%%, Best Mean IoU: %3.2f%%, Test Loss (cell): %1.4f,   Test Loss (reg): %1.4f' % (
                    current_epoch,
                    OA*100/samples,
                    iou,
                    clf.results.iou,
                    loss/weight,
                    re
                    ))


        # save model every 10 epochs
        if current_epoch % clf.training.export_every == 0 or current_epoch == clf.training.epochs:
            if current_epoch == clf.temp.start_epoch:
                continue
            model_path = os.path.join(clf.paths.out_dir, "model_"+str(int(current_epoch))+".ptm")
            print(EXPORTCOLOR)
            print('Epoch {} -> Export model to {}'.format(current_epoch, model_path))
            print(NORMALCOLOR)
            torch.save(model.state_dict(), model_path)

        # if current_epoch % clf.training.val_every == 0:
        #
        #
        #     OA = 0; loss = 0; samples = 0; weight = 0; reg = 0; edges = 0;
        #     for i,t in enumerate(data.sampler.test[:-1]):
        #         test(model, data.test[i], data.sampler.test[i], clf)
        #         # keep track of metrics over all scenes
        #         OA+= clf.temp.metrics.OA_sum; samples+=clf.temp.metrics.samples_sum
        #         loss+= clf.temp.metrics.cell_sum; weight+=clf.temp.metrics.weight_sum
        #         reg+= clf.temp.metrics.reg_sum; edges+=clf.temp.metrics.edges_sum
        #         print(TESTCOLOR)
        #         print('Epoch %3d -> File %s Test OA cell: %3.2f%%,   Test Loss (cell): %1.4f,   Test Loss (reg): %1.4f' % (
        #                 current_epoch,
        #                 data.test_names[i].split('/')[-1],
        #                 clf.temp.metrics.getOA(),
        #                 clf.temp.metrics.getCellLoss(),
        #                 clf.temp.metrics.getRegLoss()))
        #         print(NORMALCOLOR)
        #     if(clf.testing.files is not None):
        #         print(FULLTESTCOLOR)
        #         if(reg>0.0 and edges>0.0):
        #             re = reg/edges
        #         else:
        #             re = 0.0
        #         print('Epoch %3d -> Mean OA cell: %3.2f%%,   Test Loss (cell): %1.4f,   Test Loss (reg): %1.4f' % (
        #                 current_epoch,
        #                 OA*100/samples,
        #                 loss/weight,
        #                 re
        #                 ))
        #         print(NORMALCOLOR)
        #
        #     ## test on shapenet
        #     test(model, data.train, data.sampler.test[-1], clf)
        #     print(TESTCOLOR)
        #     print('Epoch %3d -> Test OA cell: %3.2f%%,   Test Loss (cell): %1.4f,   Test Loss (reg): %1.4f' % (
        #         current_epoch,
        #         clf.temp.metrics.getOA(),
        #         clf.temp.metrics.getCellLoss(),
        #         clf.temp.metrics.getRegLoss()))
        #     print(NORMALCOLOR)






######################################################
###################### INFERENCE #####################
######################################################
def inference(model, data_all, subgraph_loader, clf):

    logits_edge = None
    model.eval()  # batchnorms in eval mode
    with torch.no_grad():
        if(clf.inference.per_layer and clf.temp.batch_size):
            # print("\nInference per layer per batch")
            logits_cell = model.inference_layer_batch(data_all, subgraph_loader)
        elif(clf.inference.per_layer and not clf.temp.batch_size):
            # print("\nInference per layer")
            logits_cell = model.inference_layer(data_all)
        elif(not clf.inference.per_layer and clf.temp.batch_size):
            # print("\nInference per batch")
            logits_cell = model.inference_batch_layer(data_all, subgraph_loader)
        else:
            print("not a valid inference method set either per_layer to true or specify batch_size")
            sys.exit(1)

    # inference_end=datetime.datetime.now()
    # clf.temp.inference_time+=((inference_end - inference_start).seconds)

    # calc loss and OA
    if(clf.inference.has_label):
        clf.temp.metrics = Metrics()
        data_all.x_batch = data_all.x
        data_all.gt_batch = data_all.y

        calcLossAndOA(model, logits_cell, logits_edge, data_all, clf)
        clf.results.loss_test = clf.temp.metrics.getCellLoss()
        clf.results.OA_test = clf.temp.metrics.getOA()

        # clf.results.loss_test = clf.temp.metrics.loss.cell.value()[0]
        # clf.results.OA_test = clf.temp.metrics.OA * 100 / clf.temp.metrics.n_samples

        # clf.results.loss_test = calcLossAndOA(model, logits_cell, logits_edge, data_all, clf)
        # clf.results.loss_test = clf.results.loss_test.item()
        # print("Confusion matrix of {} cells:".format(len(data_all.y)))
        # print("pre_in, pre_out")
        # print(clf.temp.metrics.confusion_matrix.CM)
        # print("OA cells: ", clf.temp.metrics.confusion_matrix.overall_accuracy())
        # clf.results.OA_test = float(clf.temp.metrics.confusion_matrix.overall_accuracy())


        # print("OA cells: ", clf.results.OA_test)
        # print("loss cells:", clf.results.loss_test)
    else:
        clf.results.OA_test = 0.0

    if(clf.training.loss == "mse"):
        logits_cell = torch.cat((1-logits_cell, logits_cell),dim=1)

    return logits_cell.to('cpu')

