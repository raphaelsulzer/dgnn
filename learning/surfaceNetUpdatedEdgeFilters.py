import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import learningHelper as lh

import datetime

# sageconv
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


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
                 out_channels: int, edge_in_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        super(SAGEConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.edge_in_channels = edge_in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.lin_e = Linear(edge_in_channels, in_channels[0], bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_e.reset_parameters()

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            adj (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)

        # # Run "fused" message and aggregation (if applicable).
        # if (isinstance(edge_index, SparseTensor) and self.fuse
        #         and not self.__explain__):
        #     coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
        #                                  size, kwargs)
        #
        #     msg_aggr_kwargs = self.inspector.distribute(
        #         'message_and_aggregate', coll_dict)
        #     out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        #
        #     update_kwargs = self.inspector.distribute('update', coll_dict)
        #     return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        if isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)






    def forward(self, x: Union[Tensor, OptPairTensor], edge_attr: Tensor, edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)


        # lin_l is W2 and should be filtered by an edge network shared for all edges but different per layer
        # propagate_type: (x: OptPairTensor)

        edge_attr = self.lin_e(edge_attr)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.lin_l(out)

        # lin_r is W1 and does not need to be modified
        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out, edge_attr

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
                            self.in_channels, self.edge_in_channels, self.in_channels, self.out_channels)



class SurfaceNet(nn.Module):

    def __init__(self, n_node_features, clf):

        super(SurfaceNet, self).__init__()  # necessary for all classes extending the module class

        self.clf = clf
        self.n_classes = 2
        self.n_node_feat = n_node_features

        self.convs = torch.nn.ModuleList()


        self.convs.append(SAGEConv(self.n_node_feat, self.clf.training.model_params[0], 2))
        self.convs.append(SAGEConv(self.clf.training.model_params[0], self.clf.training.model_params[1], self.n_node_feat))
        for i in range(len(self.clf.training.model_params)-2):
            self.convs.append(SAGEConv(self.clf.training.model_params[i+1], self.clf.training.model_params[i+2], self.clf.training.model_params[i], normalize=False))

        self.num_layers = len(self.convs)

        if(self.clf.training.model_name[-1] == "+"):
            self.out_net = nn.Sequential(nn.ReLU(True), nn.Linear(self.clf.training.model_params[-1], 128), nn.ReLU(True), nn.Linear(128, 2))


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

        edge_attr = data_all.edge_attr
        for i in range(self.num_layers):
            edge_index, e_id, size = data_all.adjs[i]

            # TODO: first try the dumb version where edge_attr is not updated and goes always from
            # dim(edge_features) to dim(self.convs[i].in_channels)
            new_edge_attr = torch.zeros([data_all.edge_attr.shape[0], self.convs[i].in_channels],device=self.clf.temp.device)
            x,new_edge_attr[e_id] = self.convs[i]((x, x[:size[1]]), edge_attr[e_id,:self.convs[i].edge_in_channels].to(self.clf.temp.device), edge_index.to(self.clf.temp.device))
            edge_attr = new_edge_attr
            if i != self.num_layers - 1:
                x = F.relu(x)
                edge_attr = F.relu(edge_attr)
                # x = F.dropout(x, p=0.5, training=self.training)
            torch.cuda.empty_cache()

        if(self.clf.training.model_name[-1] == "+"):
            x = F.relu(x)
            x = self.out_net(x)

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
                x = self.convs[i]((x, x[:size[1]]), edge_index.to(self.clf.temp.device))
                if i != self.num_layers - 1:
                    x = F.relu(x)
                # torch.cuda.empty_cache()
                # self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))
            x_out[n_id[:batch_size]] = x.to('cpu')


        if(self.clf.training.model_name[-1] == "+"):
            x = F.relu(x_out.to(self.clf.temp.device))
            x = self.out_net(x)
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

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in tqdm(range(self.num_layers), ncols=50):
            xs = []
            for batch_size, n_id, adj in batch_loader:
                edge_index, _, size = adj  # get adjacencies of current layer / hop
                x = x_all[n_id].to(self.clf.temp.device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index.to(self.clf.temp.device))
                if i != self.num_layers - 1:
                    x = F.relu(x)
                # torch.cuda.empty_cache()
                # self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        if(self.clf.training.model_name[-1] == "+"):
            x = F.relu(x_all.to(self.clf.temp.device))
            x = self.out_net(x)
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
            x = self.convs[i]((x, x), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
            torch.cuda.empty_cache()

            # todo run on cnes with 36gb memory to achieve faster gpu time

        if(self.clf.training.model_name[-1] == "+"):
            x = F.relu(x)
            x = self.out_net(x)

        self.clf.temp.memory.append(lh.get_gpu_memory(self.clf.temp.device))

        return x

