import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_scatter import scatter

"""
Additional SignNet layers and Networks used for the SignNet encoder. 
Code is used from https://github.com/rampasek/GraphGPS/blob/main/graphgps/encoder/signnet_pos_encoder.py with their work
on the GraphGPS algorithm.
"""

class MLP(nn.Module):
    """
        provides an MLP instance used in the SignNet GIN network. Uses standard practices of MLPs and is based on the
        MLP implementation of Pytorch Geometric.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 use_bn=False, use_ln=False, dropout=0.5, activation='relu',
                 residual=False):
        super().__init__()
        self.lins = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()

        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        if activation == 'relu':
            self.activation = nn.ReLU()

        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual

    def forward(self, x):
        """
            Forward pass of the MLP layer
            """
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2, 1)).transpose(2, 1)
            if self.use_ln: x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape: x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x

    def reset_parameters(self):
        """
            Resets the layer and batch normalization parameters
            """
        for layer in self.lins:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        return

class GIN(nn.Module):
    """
        GIN module used for SignNet, accepting MLP layers as an input to the GIN layer.
        """
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers,
                 use_bn=True, dropout=0.5, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        self.use_bn = use_bn
        # input layer
        update_net = MLP(in_channels, hidden_channels, hidden_channels, 2,
                         use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv(update_net))
        # hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(hidden_channels, hidden_channels, hidden_channels,
                             2, use_bn=use_bn, dropout=dropout,
                             activation=activation)
            self.layers.append(GINConv(update_net))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        # output layer
        update_net = MLP(hidden_channels, hidden_channels, out_channels, 2,
                         use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv(update_net))
        if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        """
            GIN forward pass for each layer for 2D and 3D data.
            """
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_bn:
                    if x.ndim == 2:
                        x = self.bns[i - 1](x)
                    elif x.ndim == 3:
                        x = self.bns[i - 1](x.transpose(2, 1)).transpose(2, 1)

            x = layer(x, edge_index)
        return x


    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        return

class GINDeepSigns(nn.Module):
    """ Sign invariant neural network with MLP aggregation.
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 k, dim_pe, rho_num_layers, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)
        rho_dim = out_channels * k
        self.rho = MLP(rho_dim, hidden_channels, dim_pe, rho_num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)

    def forward(self, x, edge_index, batch_index):
        N = x.shape[0]  # Total number of nodes in the batch.
        x = x.transpose(0, 1) # N x K x In -> K x N x In
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)
        x = x.transpose(0, 1).reshape(N, -1)  # K x N x Out -> N x (K * Out)
        x = self.rho(x)  # N x dim_pe (Note: in the original codebase dim_pe is always K)
        return x

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.rho.reset_parameters()
        return
class MaskedGINDeepSigns(nn.Module):
    """ Sign invariant neural network with sum pooling and DeepSet.
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dim_pe, rho_num_layers, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)
        self.rho = MLP(out_channels, hidden_channels, dim_pe, rho_num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)

    def batched_n_nodes(self, batch_index):
        batch_size = batch_index.max().item() + 1
        one = batch_index.new_ones(batch_index.size(0))
        n_nodes = scatter(one, batch_index, dim=0, dim_size=batch_size,
                          reduce='add')  # Number of nodes in each graph.
        n_nodes = n_nodes.unsqueeze(1)
        return torch.cat([size * n_nodes.new_ones(size) for size in n_nodes])

    def forward(self, x, edge_index, batch_index):
        N = x.shape[0]  # Total number of nodes in the batch.
        K = x.shape[1]  # Max. number of eigen vectors / frequencies.
        x = x.transpose(0, 1)  # N x K x In -> K x N x In
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)  # K x N x Out
        x = x.transpose(0, 1)  # K x N x Out -> N x K x Out

        batched_num_nodes = self.batched_n_nodes(batch_index)
        mask = torch.cat([torch.arange(K).unsqueeze(0) for _ in range(N)])
        mask = (mask.to(batch_index.device) < batched_num_nodes.unsqueeze(1)).bool()
        # print(f"     - mask: {mask.shape} {mask}")
        # print(f"     - num_nodes: {num_nodes}")
        # print(f"     - batched_num_nodes: {batched_num_nodes.shape} {batched_num_nodes}")
        x[~mask] = 0
        x = x.sum(dim=1)  # (sum over K) -> N x Out
        x = self.rho(x)  # N x Out -> N x dim_pe (Note: in the original codebase dim_pe is always K)
        return x

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.rho.reset_parameters()
        return