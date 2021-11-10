import sys
import torch
import torch.nn as nn

from tqdm import tqdm

# sageconv
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing, SAGEConv


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, edge_in_channels: int, concatenate: bool = True, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        super(SAGEConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.edge_in_channels = edge_in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_e = Linear(edge_in_channels, int(in_channels[0]/2), bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_e.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_attr: Tensor, edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # lin_l is W2 and should be filtered by an edge network shared for all edges but different per layer
        # propagate_type: (x: OptPairTensor)

        edge_attr = self.lin_e(edge_attr)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        # TODO: what I should probably do is not have lin_l and lin_r anymore
        # but concatenate out (before lin_l) and x_r and then only have one net

        # lin_r is W1 and does not need to be modified
        out = torch.cat((out,x[1]),dim=1)
        out = self.lin_l(out)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # TODO: maybe simply return MLP(edge_features) \hadamard x_j, where
        # and return should also be MLP(edge_features) for outputting edge labels with dim=1 for last edge_MLP out_channel

        return x_j*edge_attr

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}(in:{}, edge_in:{}, edge_out:{}, out:{})'.format(self.__class__.__name__,
                            int(self.in_channels/2), self.edge_in_channels, int(self.in_channels/2), self.out_channels)



class SurfaceNet(nn.Module):

    def __init__(self, n_node_features, clf):

        super(SurfaceNet, self).__init__()  # necessary for all classes extending the module class

        self.clf = clf
        self.n_classes = 2
        self.n_node_feat = n_node_features
        self.n_edge_feat = 2



        if(self.clf.model.encoder):
            self.encoder = nn.Sequential(nn.Linear(self.n_node_feat, self.clf.model.encoder), nn.ReLU(True))
            self.sage_convs = torch.nn.ModuleList()
        else:
            self.sage_convs = torch.nn.ModuleList()
            self.sage_convs.append(SAGEConv(self.n_node_feat, self.clf.model.convs[0], self.n_edge_feat))

        for i in range(len(self.clf.model.convs)):
            self.sage_convs.append(SAGEConv(int(self.clf.model.convs[i]), self.clf.model.convs[i], self.n_edge_feat))

        self.num_layers = len(self.sage_convs)

        if(self.clf.model.decoder):
            self.decoder = nn.Sequential(nn.ReLU(True), nn.Linear(self.clf.model.decoder, 2))


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

        x = self.encoder(x)
        for i in range(self.num_layers):
            edge_index, e_id, size = data_all.adjs[i]
            x = self.sage_convs[i]((x, x[:size[1]]), data_all.edge_attr[e_id].to(self.clf.temp.device), edge_index.to(self.clf.temp.device))
            if i != self.num_layers - 1:
                x = F.relu(x)
                # x = F.dropout(x, p=0.5, training=self.training)

        if(self.clf.model.decoder):
            x = F.relu(x)
            x = self.decoder(x)

        # self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))

        return x

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


        if(self.clf.model.decoder):
            x = F.relu(x_out.to(self.clf.temp.device))
            x = self.decoder(x)
        else:
            x = x_out.to(self.clf.temp.device)

        self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))
        return x



    def inference_layer_batch(self, data_all, batch_loader):
        # produces embeddings layer by layer, batch per batch
        # subgraph sampling necessary
        # needed when full graph does not fit in memory

        if (self.clf.features.normalization_feature and not self.clf.features.keep_normalization_feature):
            x_all = data_all.x[:, 1:]  # do not put it on gpu yet, because they will be used batch by batch
        else:
            x_all = data_all.x
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in tqdm(range(self.num_layers), ncols=50):
            xs = []
            for batch_size, n_id, adj in batch_loader:
                edge_index, e_id, size = adj  # get adjacencies of current layer / hop
                x = x_all[n_id].to(self.clf.temp.device)
                x_target = x[:size[1]]
                x = self.sage_convs[i]((x, x_target), data_all.edge_attr[e_id].to(self.clf.temp.device), edge_index.to(self.clf.temp.device))
                if i != self.num_layers - 1:
                    x = F.relu(x)
                # torch.cuda.empty_cache()
                # self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        if(self.clf.model.decoder):
            x = F.relu(x_all.to(self.clf.temp.device))
            x = self.decoder(x)
        else:
            x = x_all.to(self.clf.temp.device)

        # self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))
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

        if(self.clf.model.decoder):
            x = F.relu(x)
            x = self.decoder(x)

        self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))

        return x

