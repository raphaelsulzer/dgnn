import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch_geometric.nn.conv import SAGEConv

class SurfaceNet(nn.Module):

    def __init__(self, n_node_features, clf):

        super(SurfaceNet, self).__init__()  # necessary for all classes extending the module class

        self.clf = clf
        self.n_classes = 2
        self.n_node_feat = n_node_features

        self.sage_convs = torch.nn.ModuleList()

        self.sage_convs.append(SAGEConv(self.n_node_feat, self.clf.training.model_params[0]))
        for i in range(len(self.clf.training.model_params)-1):
            self.sage_convs.append(SAGEConv(self.clf.training.model_params[i], self.clf.training.model_params[i+1], normalize=False))

        self.num_layers = len(self.sage_convs)

        if(self.clf.training.model_name == "sage+"):
            self.vertex_net = nn.Sequential(nn.ReLU(True), nn.Linear(self.clf.training.model_params[-1], 128), nn.ReLU(True), nn.Linear(128, 2))

        if(self.clf.training.model_name == "sage++"):
            self.vertex_net = nn.Sequential(nn.ReLU(True), nn.Linear(self.clf.training.model_params[-1], 128), nn.ReLU(True), nn.Linear(128, 2))
            self.edge_net = nn.Sequential(nn.Linear(self.clf.training.model_params[-1]*2, 128), nn.ReLU(True), nn.Linear(128, 1))


    #######################################################
    ##################### TRAIN FORWARD ###################
    #######################################################
    def forward(self, data_all):
        # produces final embedding batch per batch


        """
        the forward function producing the embeddings for each cell of 'input'
        input = [n_batch, input_feat, n_channels=1] float array: input features
        output = [n_batch, n_class, n_channels=1] float array: cell class scores
        """

        if(self.clf.features.normalization_feature and not self.clf.features.keep_normalization_feature):
            x=data_all.x[data_all.n_id, 1:].to(self.clf.temp.device) # put this batch on gpu
        else:
            x=data_all.x[data_all.n_id, :].to(self.clf.temp.device) # put this batch on gpu

        # for i, (edge_index, _, size) in enumerate(data_all.adjs):
        for i in range(self.num_layers):
            edge_index, e_id, size = data_all.adjs[i]
            x = self.sage_convs[i]((x, x[:size[1]]), edge_index.to(self.clf.temp.device))
            if i != self.num_layers - 1:
                x = F.relu(x)
                # x = F.dropout(x, p=0.5, training=self.training)

            # torch.cuda.empty_cache()

        x_edge = None

        if(self.clf.training.model_name[-1] == "+"):
            x = F.relu(x)
            x = self.vertex_net(x)

        if(self.clf.training.model_name[:-1] == "epsage"):
            x = F.relu(x)
            # concatenate left and right cell embeddings
            # take the edge_indices from self.num_layers + 1, for those I have the final embeddings calculated
            x_edge = self.edge_net(torch.cat((x[data_all.adjs[i+1][0][0]],x[data_all.adjs[i+1][0][1]]),dim=1))
            x = self.vertex_net(x)





        # self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))

        return x, x_edge

    #######################################################
    ################## INFERENCE FORWARDS #################
    #######################################################
    def inference_batch_layer(self, x_all, batch_loader):
        # produces embeddings layer by layer, batch per batch
        # subgraph sampling necessary
        # needed when full graph does not fit in memory


        if (self.clf.training.loss == "kl"):
            x_out = torch.zeros([x_all.size()[0], 2], dtype=torch.float32, requires_grad=False).to('cpu')
        elif (self.clf.training.loss == "mse"):
            x_out = torch.zeros([x_all.size()[0], 1], dtype=torch.float32, requires_grad=False).to('cpu')
        else:
            print("{} is not a valid loss. choose either kl or mse".format(self.clf.training.loss))
            sys.exit(1)


        if (self.clf.features.normalization_feature and not self.clf.features.keep_normalization_feature):
            x_all = x_all[:, 1:]  # do not put it on gpu yet, because they will be used batch by batch

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for batch_size, n_id, adjs in tqdm(batch_loader, ncols=50):
            x = x_all[n_id, :].to(self.clf.temp.device)
            for i in range(self.num_layers):
                edge_index, _, size = adjs[i]
                x = self.sage_convs[i]((x, x[:size[1]]), edge_index.to(self.clf.temp.device))
                if i != self.num_layers - 1:
                    x = F.relu(x)
                # torch.cuda.empty_cache()
                # self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))
            x_out[n_id[:batch_size]] = x.to('cpu')


        if (self.clf.training.model_name == "sage+"):
            x = F.relu(x_out.to(self.clf.temp.device))
            x = self.vertex_net(x)
        else:
            x = x_out.to(self.clf.temp.device)

        self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))
        return x



    def inference_layer_batch(self, x_all, batch_loader):
        # produces embeddings layer by layer, batch per batch
        # subgraph sampling necessary
        # needed when full graph does not fit in memory

        if (self.clf.features.normalization_feature and not self.clf.features.keep_normalization_feature):
            x_all = x_all[:, 1:]  # do not put it on gpu yet, because they will be used batch by batch

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in tqdm(range(self.num_layers), ncols=50):
            xs = []
            for batch_size, n_id, adj in batch_loader:
                edge_index, _, size = adj  # get adjacencies of current layer / hop
                x = x_all[n_id].to(self.clf.temp.device)
                x_target = x[:size[1]]
                x = self.sage_convs[i]((x, x_target), edge_index.to(self.clf.temp.device))
                if i != self.num_layers - 1:
                    x = F.relu(x)
                # torch.cuda.empty_cache()
                # self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        if (self.clf.training.model_name == "sage+"):
            x = F.relu(x_all.to(self.clf.temp.device))
            x = self.vertex_net(x)
        else:
            x = x_all.to(self.clf.temp.device)

        self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))
        return x


    def inference_layer(self, data_all):
        # produces embeddings layer by layer directly for the whole graph
        # no subgraph sampling necessary
        # only works if full graph fits in memory


        if(self.clf.features.normalization_feature and not self.clf.features.keep_normalization_feature):
            x=data_all.x[:, 1:].to(self.clf.temp.device) # put all on gpu, because they will all be used directly
        else:
            x=data_all.x[:, :].to(self.clf.temp.device)

        edge_index = data_all.edge_index.to(torch.long).to(self.clf.temp.device) # put all on gpu, because they will all be used directly
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            x = self.sage_convs[i]((x, x), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
            torch.cuda.empty_cache()

            # todo run on cnes with 36gb memory to achieve faster gpu time

        if(self.clf.training.model_name == "sage+"):
            x = F.relu(x)
            x = self.vertex_net(x)

        self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))

        return x
