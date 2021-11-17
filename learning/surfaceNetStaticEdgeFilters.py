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
from torch_geometric.nn.norm import LayerNorm, BatchNorm


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
    # def __init__(self, in_channels: Union[int, Tuple[int, int]],
    #              out_channels: int, edge_in_channels: int, concatenate: bool = True, normalize: bool = False,
    #              bias: bool = True, **kwargs):  # yapf: disable
    #     super(SAGEConv, self).__init__(aggr='mean', **kwargs)
    def __init__(self, lin_i, lin_j, lin_e, **kwargs):  # yapf: disable
        super(SAGEConv, self).__init__(aggr='mean', **kwargs)

        self.lin_i = lin_i
        self.lin_j = lin_j
        self.lin_e = lin_e

    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     self.lin_i.reset_parameters()
    #     self.lin_j.reset_parameters()
    #     if(self.lin_e is not None):
    #         if(len(self.lin_e)>1):
    #             self.lin_e[0].reset_parameters
    #             self.lin_e[2].reset_parameters
    #         else:
    #             self.lin_e.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_attr: Tensor, edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # lin_j is W2 and should be filtered by an edge network shared for all edges but different per layer
        # propagate_type: (x: OptPairTensor)

        if self.lin_e is not None:
            edge_attr = self.lin_e(edge_attr)
        else:
            edge_attr = None

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.lin_j(out)

        # lin_i is W1 and does not need to be modified
        x_r = x[1]
        if x_r is not None:
            out += self.lin_i(x_r)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # TODO: maybe simply return MLP(edge_features) \hadamard x_j, where
        # and return should also be MLP(edge_features) for outputting edge labels with dim=1 for last edge_MLP out_channel

        if edge_attr is not None:
            return x_j*edge_attr
        else:
            return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        # return '{}(in:{}, edge_in:{}, edge_out:{}, out:{})'.format(self.__class__.__name__,
        #                     self.in_channels, self.edge_in_channels, self.in_channels, self.out_channels)
        return '{}:\n' \
               'W1: {}\n' \
               'W2: {}\n' \
               '\u03A6: {}'.format(self.__class__.__name__, self.lin_i, self.lin_j, self.lin_e)




class SurfaceNet(nn.Module):

    def normLayer(self, size):

        if self.norm_type == 'b':
            return BatchNorm(size)
        elif self.norm_type == 'l':
            return LayerNorm(size)
        else:
            return None

    def sageLayer(self, input, output):

        li = Linear(input, output, bias=False)
        lj = Linear(input, output, bias=True)
        if(self.clf.model.edge_convs == 1):
            le = Linear(self.n_edge_feat, input, bias=True)
        elif(self.clf.model.edge_convs == 2):
            le = torch.nn.Sequential()
            le.add_module("0", Linear(self.n_edge_feat,int(self.n_edge_feat*2)))
            le.add_module("1", self.normLayer(int(self.n_edge_feat*2)))
            le.add_module("2", nn.ReLU(True))
            le.add_module("3", Linear(int(self.n_edge_feat*2),input))
        else:
            le = None

        return SAGEConv(li,lj,le)





    def __init__(self, clf):
        super(SurfaceNet, self).__init__()  # necessary for all classes extending the module class

        self.clf = clf
        self.n_classes = 2
        self.n_node_feat = clf.temp.num_node_features
        self.n_edge_feat = clf.temp.num_edge_features
        self.norm_type = clf.model.normalization
        if(clf.training.loss == "kl"):
            self.output_dim = 2
        else:
            self.output_dim = 1

        ## init module lists for convs and norms
        self.convs = torch.nn.ModuleList()

        ## first layer
        first = torch.nn.Sequential()
        first.add_module("conv",self.sageLayer(self.n_node_feat,self.clf.model.convs[0]))
        first.add_module("norm",self.normLayer(self.clf.model.convs[0]))
        first.add_module("relu",nn.ReLU(True))
        self.convs.append(first)

        ## hidden layers
        for i in range(len(self.clf.model.convs)-1):
            hidden = torch.nn.Sequential()
            hidden.add_module("conv", self.sageLayer(self.clf.model.convs[i], self.clf.model.convs[i+1]))
            hidden.add_module("norm", self.normLayer(self.clf.model.convs[i+1]))
            hidden.add_module("relu", nn.ReLU(True))
            self.convs.append(hidden)

        self.num_layers = len(self.convs)

        ## out layers
        self.decoder = nn.Sequential()
        if(self.clf.model.decoder==1):
            self.decoder.add_module("0",nn.Linear(self.clf.model.convs[-1], self.output_dim))
        elif(self.clf.model.decoder==2):
            self.decoder.add_module("0",nn.Linear(self.clf.model.convs[-1], int(self.clf.model.convs[-1]/2)))
            self.decoder.add_module("1",self.normLayer(int(self.clf.model.convs[-1]/2)))
            self.decoder.add_module("2",nn.ReLU(True))
            self.decoder.add_module("3",nn.Linear( int(self.clf.model.convs[-1]/2), self.output_dim))
            # TODO: turn off batch_norm and relu in this layer, and maybe even in last conv layer?




    #######################################################
    ##################### TRAIN FORWARD ###################
    #######################################################
    def forward(self, data):
        # produces final embedding batch per batch

        """
        the forward function producing the embeddings for each cell of 'input'
        input = [n_batch, input_feat, n_channels=1] float array: input features
        output = [n_batch, n_class, n_channels=1] float array: cell class scores
        """

        if(self.clf.regularization.cell_reg_type):
            x=data.all.x[data.batch_n_id, 1:].to(self.clf.temp.device) # put this batch on gpu
        else:
            x=data.all.x[data.batch_n_id, :].to(self.clf.temp.device) # put this batch on gpu
        # if(self.clf.regularization.reg_type):
        #     xe=data_all.edge_attr[:,1:]
        # else:
        #     xe=data_all.edge_attr

        for i in range(self.num_layers):
            edge_index, e_id, size = data.batch_adjs[i]
            # apply conv, norm, relu
            x = self.convs[i][0]((x, x[:size[1]]), data.all.edge_attr[e_id].to(self.clf.temp.device), edge_index.to(self.clf.temp.device))
            x = self.convs[i][1](x)
            x = self.convs[i][2](x)
            # x = F.dropout(x, p=0.5, training=self.training)

        if(self.clf.model.decoder):
            x = self.decoder(x)

        # self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))

        return x

    #######################################################
    ################## INFERENCE FORWARDS #################
    #######################################################
    def inference_batch_layer(self, data_all, batch_loader):
        # produces embeddings layer by layer, batch per batch
        # subgraph sampling necessary
        # needed when full graph does not fit in VRAM


        if (self.clf.training.loss == "kl"):
            x_out = torch.zeros([data_all.x.size()[0], 2], dtype=torch.float32, requires_grad=False, device=self.clf.temp.device)
        else:
            x_out = torch.zeros([data_all.x.size()[0], 1], dtype=torch.float32, requires_grad=False, device=self.clf.temp.device)


        if (self.clf.regularization.cell_reg_type):
            x_all = data_all.x[:, 1:]  # do not put it on gpu yet, because they will be used batch by batch
        else:
            x_all = data_all.x

        if(self.clf.regularization.reg_type):
            xe=data_all.edge_attr[:,1:]
        else:
            xe=data_all.edge_attr
        # edge_index = data_all.edge_index.to(torch.long).to(self.clf.temp.device) # put all on gpu, because they will all be used directly

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for batch_size, n_id, adjs in tqdm(batch_loader, ncols=50):
            x = x_all[n_id, :].to(self.clf.temp.device)
            for i in range(self.num_layers):
                edge_index, e_id, size = adjs[i]
                x = self.convs[i][0]((x, x[:size[1]]), xe[e_id].to(self.clf.temp.device), edge_index.to(self.clf.temp.device))
                x = self.convs[i][1](x)
                x = self.convs[i][2](x)
                # torch.cuda.empty_cache()
                # self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))

            if (self.clf.model.decoder):
                x = self.decoder(x)

            x_out[n_id[:batch_size]] = x


        # self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))
        return x_out



    def inference_layer_batch(self, data_all, batch_loader):
        # produces embeddings layer by layer, batch per batch
        # subgraph sampling necessary
        # needed when full graph does not fit in VRAM

        if (self.clf.regularization.cell_reg_type):
            x_all = data_all.x[:, 1:]  # do not put it on gpu yet, because they will be used batch by batch
        else:
            x_all = data_all.x

        if(self.clf.regularization.reg_type):
            xe=data_all.edge_attr[:,1:]
        else:
            xe=data_all.edge_attr

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        # for i in tqdm(range(self.num_layers), ncols=50):
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in batch_loader:
                edge_index, e_id, size = adj[i]  # get adjacencies of current layer / hop
                x = x_all[n_id].to(self.clf.temp.device)
                x_target = x[:size[1]]
                x = self.convs[i][0]((x, x_target), xe[e_id].to(self.clf.temp.device), edge_index.to(self.clf.temp.device))
                x = self.convs[i][1](x)
                x = self.convs[i][2](x)
                # torch.cuda.empty_cache()
                # self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        if(self.clf.model.decoder):
            x = self.decoder(x_all.to(self.clf.temp.device))
        else:
            x = x_all.to(self.clf.temp.device)

        # self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))
        return x


    def inference_layer(self, data_all):
        # produces embeddings layer by layer directly for the whole graph
        # no subgraph sampling necessary
        # only works if full graph fits in VRAM


        if(self.clf.regularization.cell_reg_type):
            x=data_all.x[:, 1:].to(self.clf.temp.device) # put all on gpu, because they will all be used directly
        else:
            x=data_all.x[:, :].to(self.clf.temp.device)

        if(self.clf.regularization.reg_type):
            xe=data_all.edge_attr[:,1:].to(self.clf.temp.device)
        else:
            xe=data_all.edge_attr.to(self.clf.temp.device)

        edge_index = data_all.edge_index.to(torch.long).to(self.clf.temp.device) # put all on gpu, because they will all be used directly
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            x = self.convs[i][0]((x, x), xe, edge_index)
            x = self.convs[i][1](x)
            x = self.convs[i][2](x)
            # torch.cuda.empty_cache()


        if(self.clf.model.decoder):
            x = self.decoder(x)

        # self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))

        return x

