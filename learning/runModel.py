import sys, os
import torch
import torch.optim as optim
import torch.nn.functional as F
from shutil import copyfile

import numpy as np
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import subprocess as sp

import pprint


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
        self.cell_sum+=cell.item()
        self.weight_sum+=weight.item()
    def getCellLoss(self):
        return self.cell_sum/self.weight_sum # it's a tensor, so just return the value

    def addRegLossItem(self,reg,n):
        self.reg_sum+=reg.item()
        self.edges_sum+=n
    def getRegLoss(self):
        if(self.reg_sum > 0.0 and self.edges_sum > 0.0):
            return (self.reg_sum/self.edges_sum) # it's a tensor, so just return the value
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
    lr = clf.training.learning_rate * (0.1 ** (clf.temp.current_epoch // clf.training.adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Trainer():

    def __init__(self,model):

        self.model = model


    def calcRegularization(self, logits_cell, data, clf, metrics):

        # this works because layer K-n embeddings are actually calculate for all layers K-m
        # with m=[n,..,K]; not just for m = n
        # it is done this way to have the possibility to apply skip connections

        # it should already work with additional_hops = 1, with
        if(data.batch_adjs):
            # this should be done with probabilities, so take softmax of logits
            inner_prediction = F.softmax(logits_cell[:data.batch_adjs[self.model.num_layers].size[0]], dim=-1)
            surface_triangle_probability_kl = torch.abs(inner_prediction[data.batch_adjs[self.model.num_layers].edge_index[0, :]][:, 0] \
                                                 - inner_prediction[data.batch_adjs[self.model.num_layers].edge_index[1, :]][:, 0])
        else:
            # this should be done with probabilities, so take softmax of logits
            inner_prediction = F.softmax(logits_cell, dim=-1)
            surface_triangle_probability_kl = torch.abs(inner_prediction[data.edge_index[0, :]][:, 0] \
                                                 - inner_prediction[data.edge_index[1, :]][:, 0])


        ## for their source and target nodes
        # li=inner_prediction[batch_adjs[self.model.num_layers+1].edge_index[0,:]][:,0]
        # ri=inner_prediction[batch_adjs[self.model.num_layers+1].edge_index[1,:]][:,0]
        # lo=inner_prediction[batch_adjs[self.model.num_layers+1].edge_index[0,:]][:,1]
        # ro=inner_prediction[batch_adjs[self.model.num_layers+1].edge_index[1,:]][:,1]
        # inner_subgraph_indicies = batch_adjs[self.model.num_layers+1].size[0]
        # inner_prediction = F.log_softmax(logits_cell[:data_all.batch_adjs[self.model.num_layers + 1].size[0]], dim=-1)
        # ## now take only the edge of the inner subgraph, because only for these edges, I have predictions
        # surface_triangle_prob_kl = torch.abs(inner_prediction[data_all.batch_adjs[self.model.num_layers + 1].edge_index[0, :]][:, 0] \
        #                                      - inner_prediction[data_all.batch_adjs[self.model.num_layers + 1].edge_index[1, :]][:, 0])

        # surface_area = data_all.edge_attr[data_all.batch_adjs[self.model.num_layers + 1].e_id].squeeze()[:, 0].to(clf.temp.device)

        # if(clf.regularization.reg_type):
        #     weight = data_all.edge_attr[data_all.batch_adjs[self.model.num_layers].e_id].squeeze()[:, 0].to(clf.temp.device)
        # else:
        #     weight =

        # area_loss = torch.mean(surface_triangle_prob_kl * surface_area) * clf.regularization.area
        # angle_loss = torch.mean(surface_triangle_prob_kl * angle) * clf.regularization.angle

        # reg_loss = surface_triangle_prob_kl * weight * clf.regularization.reg_weight

        reg_loss = surface_triangle_probability_kl * clf.regularization.reg_weight
        # reg_loss = reg_loss.sum() / weight.sum()
        # reg_loss = reg_loss.sum()

        # size of this should be batch_size * 4, because every cell has 4 neighbors
        # and the weight is 1, thus simply pass the size, otherwise I would need to pass the sum of the weight
        metrics.addRegLossItem(reg_loss.sum(), surface_triangle_probability_kl.size()[0])

        # currently weight per triangle is one, thus simply return the mean
        return reg_loss.mean()


    def calcLossAndOA(self, logits_cell, logits_edge, data, clf, metrics):


        ###############################################################
        ########################## cell loss ##########################
        ###############################################################
        if (clf.regularization.cell_reg_type):



            if (clf.training.loss == "kl"):
                # input is always expected in log-probabilities (hence log_softmax) while target is expected in probabilities
                cell_loss = F.kl_div(F.log_softmax(logits_cell, dim=-1), data.batch_gt[:, :2].to(clf.temp.device), reduction='none')
                cell_loss = torch.sum(cell_loss, dim=1)  # cf. formula for kl_div, summing over X (the dimensions)
                metrics.addOAItem(
                    torch.sum(data.batch_gt[:, 2] == F.log_softmax(logits_cell, dim=-1).argmax(1).cpu()).item(),
                    data.batch_x.shape[0])
            elif(clf.training.loss == "bce"):
                # supervise with graph cut label
                cell_loss = F.binary_cross_entropy_with_logits(logits_cell.squeeze(dim=-1), data.batch_gt[:, 3].to(clf.temp.device), reduction='none')
                metrics.addOAItem(
                    torch.sum(data.batch_gt[:, 3] == torch.round(F.sigmoid(logits_cell.squeeze(dim=-1))).cpu()).item(),
                    data.batch_x.shape[0])
            elif (clf.training.loss == "mse"):
                cell_loss = F.mse_loss(F.sigmoid(logits_cell).squeeze(), data.batch_gt[:, 0].to(clf.temp.device))
            else:
                print("{} is not a valid loss. choose either kl or mse".format(clf.training.loss))
                sys.exit(1)

            # multiply loss per cell with volume of the cell
            shape_weight = data.batch_x[:, 0].to(clf.temp.device)
            # shape_weight =  torch.sqrt(batch_x[:,0].to(clf.temp.device))
            # shape_weight =  torch.log(1+batch_x[:,0].to(clf.temp.device))

            cell_loss = cell_loss * shape_weight

            # add loss to metrics for statistics
            metrics.addCellLossItem(cell_loss.sum(),shape_weight.sum())
            # only works if additional_hops > 0


            cell_loss = cell_loss.sum() / shape_weight.sum()


            # # final loss is mean over all batch entries
            # if (clf.regularization.shape_weight_batch_normalization):
            #     cell_loss = cell_loss.sum() / shape_weight.sum()
            # else:
            #     if (clf.regularization.inside_outside_weight[0] != clf.regularization.inside_outside_weight[1]):
            #         io_weight = data_all.batch_gt[:, 1].to(clf.temp.device) * clf.regularization.inside_outside_weight[0] + \
            #                     data_all.batch_gt[:, 2].to(clf.temp.device) * clf.regularization.inside_outside_weight[1]
            #         cell_loss = cell_loss * io_weight
            #         cell_loss = cell_loss.sum() / io_weight.sum()
            #     else:
            #         cell_loss = torch.mean(cell_loss)*10**6
        # else: # without any normalization of the loss:
        #     cell_loss = F.kl_div(F.log_softmax(logits_cell, dim=-1), data_all.batch_gt[:, 1:].to(clf.temp.device), reduction='batchmean')
        loss = cell_loss

        if ((loss != loss).any()):
            print("nan in loss")



        ###############################################################
        ########################## edge loss ##########################
        ###############################################################
        # only works if additional_hops > 0
        edge_loss=0
        if(logits_edge is not None):
            gt_left = data.y[data.batch_adjs[self.model.num_layers].edge_index[0]][:,1]
            gt_right = data.y[data.batch_adjs[self.model.num_layers].edge_index[1]][:,1]
            edge_loss=F.kl_div(F.log_softmax(logits_edge).squeeze(),(gt_left-gt_right).to(clf.temp.device))
            if((edge_loss != edge_loss).any()):
                print("here")
            # TODO: get inner ground truth and inner edges and calc an edge_loss
            loss+=edge_loss

        ###############################################################################################################
        ############################ area+angle+cc loss / total variation / regularization ############################
        ###############################################################################################################
        if(clf.regularization.reg_epoch is not None): # check if reg_epoch is not null
            if(clf.graph.additional_num_hops != 1):
                print("ERROR: clf.graph.additional_num_hops == 1 to use regularization")
                sys.exit(1)
            if (clf.temp.current_epoch >= clf.regularization.reg_epoch):
                reg_loss = self.calcRegularization(logits_cell,data,clf, metrics)
                loss+=reg_loss

        # return loss for mini batch gradient decent
        return loss

    ####################################################
    ############## TRAINING AND TESTING ################
    ####################################################
    def train(self, data_train, optimizer, clf):

        self.model.train() #switch the model in training mode

        logits_edge = None
        if(clf.model.edge_prediction):
            logits_cell, logits_edge = self.model(data_train)
        else:
            logits_cell = self.model(data_train)

        data_train.batch_x = data_train.all.x[data_train.batch_n_id[:data_train.batch_adjs[self.model.num_layers-1].size[1]]]
        data_train.batch_gt = data_train.all.y[data_train.batch_n_id[:data_train.batch_adjs[self.model.num_layers-1].size[1]]]
        loss = self.calcLossAndOA(logits_cell, logits_edge, data_train, clf, clf.training.metrics)

        optimizer.zero_grad()  # put gradient to zero
        loss.backward()
        # for p in self.model.parameters():  # we clip the gradient at norm 1
        #     p.grad.data.clamp_(-1, 1)
        optimizer.step()  # one SGD (=stochastic gradient descent) step


    def train_test(self, data, clf):

        clf.temp.device = "cuda:" + str(clf.temp.args.gpu)

        # define the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=clf.training.learning_rate)

        # define some color to help distinguish between train and test outputs
        TESTCOLOR = '\033[1;0;46m'
        FULLTESTCOLOR = '\033[1;0;44m'
        EXPORTCOLOR = '\033[1;0;42m'
        NORMALCOLOR = '\033[0m'

        row = dict.fromkeys(list(clf.results_df.columns))

        # init metrics
        clf.training.metrics = Metrics()
        iterations = 0

        for current_epoch in range(1,clf.training.epochs+1):
            # adjust learning rate to current epoch
            clf.temp.current_epoch = current_epoch
            adjust_learning_rate(optimizer, clf)
            # train one epoch
            for data.train.batch_size, data.train.batch_n_id, data.train.batch_adjs in data.train.batches:

                iterations += 1

                self.train(data.train, optimizer, clf)

                row['iteration'] = iterations
                row['epoch'] = current_epoch
                row['train_loss'] = clf.training.metrics.getCellLoss()
                row['train_loss_reg'] = clf.training.metrics.getRegLoss()
                row['train_OA'] = clf.training.metrics.getOA()

                if(iterations % clf.training.print_every) == 0 or iterations == 1:

                    print('[%3d] Epoch %3d -> Train Loss (cell): %1.4f,  Train Loss (reg): %1.4f, Train OA: %3.2f%%'
                        % (iterations, current_epoch,
                           clf.training.metrics.getCellLoss(),
                           clf.training.metrics.getRegLoss(),
                           clf.training.metrics.getOA()))

                    # reinit new metrics:
                    clf.training.metrics = Metrics()

                # do testing (ie inference in my case), if this is a test epoch
                if(iterations % clf.training.val_every) == 0:

                    OA = 0;loss = 0;samples = 0;weight = 0;reg = 0;edges = 0; iou = 0;
                    for i,d in enumerate(tqdm(data.validation.all, ncols=50)):
                        prediction = self.inference(d,[],clf)
                        temp=gm.generate(d, prediction, clf)
                        iou+=temp[1]
                        # export one shape per class
                        if(i%clf.validation.shapes_per_conf_per_class == 0):
                            temp[0].export(os.path.join(clf.paths.out_dir,"generation",d["category"]+"_"+d['id']+".ply"))
                        # keep track of metrics over all scenes
                        OA+= clf.inference.metrics.OA_sum; samples+=clf.inference.metrics.samples_sum
                        loss+= clf.inference.metrics.cell_sum; weight+=clf.inference.metrics.weight_sum
                        reg+= clf.inference.metrics.reg_sum; edges+=clf.inference.metrics.edges_sum

                    re = reg/edges if (reg>0.0 and edges>0.0) else float("nan")

                    iou = iou*100/len(data.validation)
                    if(iou > clf.best_iou):
                        clf.best_iou = iou
                        model_path = os.path.join(clf.paths.out_dir,"model_best.ptm")
                        torch.save(self.model.state_dict(), model_path)

                    ## save everything in the dataframe
                    row['test_loss'] = loss/weight
                    row['test_loss_reg'] = re
                    row['test_OA'] = OA*100/samples
                    row['test_iou'] = iou
                    row['test_best_iou'] = clf.best_iou

                    pprint.pprint(row)
                    clf.results_df = clf.results_df.append(row,ignore_index=True)
                    clf.results_df.to_csv(clf.files.results,index=False)


                # save model every 10 epochs
                if (iterations % clf.training.export_every) == 0:
                    model_path = os.path.join(clf.paths.out_dir, "model_"+str(int(current_epoch))+".ptm")
                    print(EXPORTCOLOR)
                    print('[{}] Epoch {} -> Export model to {}'.format(iterations, current_epoch, model_path))
                    print(NORMALCOLOR)
                    torch.save(self.model.state_dict(), model_path)

                    # backup results file
                    copyfile(clf.files.results, os.path.splitext(clf.files.results)[0]+"_"+str(current_epoch)+".csv")



    ######################################################
    ###################### INFERENCE #####################
    ######################################################
    def inference(self, data_inference, subgraph_loader, clf):

        logits_edge = None
        self.model.eval()  # batchnorms in eval mode
        with torch.no_grad():
            if(clf.inference.per_layer and clf.temp.batch_size):
                # print("\nInference per layer per batch")
                assert(subgraph_loader)
                logits_cell = self.model.inference_layer_batch(data_inference, subgraph_loader)
            elif(clf.inference.per_layer and not clf.temp.batch_size):
                # print("\nInference per layer")
                assert(not subgraph_loader)
                data_inference.batch_adjs = []  # necessary to know in the calcRegularization function if data is batched or not
                logits_cell = self.model.inference_layer(data_inference)
            elif(not clf.inference.per_layer and clf.temp.batch_size):
                # print("\nInference per batch")
                assert(subgraph_loader)
                logits_cell = self.model.inference_batch_layer(data_inference, subgraph_loader)
            else:
                print("not a valid inference method set either per_layer to true or specify batch_size")
                sys.exit(1)

        # inference_end=datetime.datetime.now()
        # clf.temp.inference_time+=((inference_end - inference_start).seconds)

        # calc loss and OA
        if(clf.inference.has_label):
            clf.inference.metrics = Metrics()
            data_inference.batch_x = data_inference.x
            data_inference.batch_gt = data_inference.y

            self.calcLossAndOA(logits_cell, logits_edge, data_inference, clf, clf.inference.metrics)
            # clf.results.loss_test = clf.inference.metrics.getCellLoss()
            # clf.results.OA_test = clf.inference.metrics.getOA()

            # clf.results.loss_test = clf.inference.metrics.loss.cell.value()[0]
            # clf.results.OA_test = clf.inference.metrics.OA * 100 / clf.inference.metrics.n_samples

            # clf.results.loss_test = calcLossAndOA(model, logits_cell, logits_edge, data_inference_all, clf)
            # clf.results.loss_test = clf.results.loss_test.item()
            # print("Confusion matrix of {} cells:".format(len(data_inference_all.y)))
            # print("pre_in, pre_out")
            # print(clf.inference.metrics.confusion_matrix.CM)
            # print("OA cells: ", clf.inference.metrics.confusion_matrix.overall_accuracy())
            # clf.results.OA_test = float(clf.inference.metrics.confusion_matrix.overall_accuracy())


            # print("OA cells: ", clf.results.OA_test)
            # print("loss cells:", clf.results.loss_test)
        else:
            clf.results.OA_test = 0.0

        if(clf.training.loss == "mse"):
            logits_cell = torch.cat((1-logits_cell, logits_cell),dim=1)

        return logits_cell.to('cpu')


