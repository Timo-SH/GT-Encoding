import math
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch_geometric.nn
import torch_geometric.nn as gnn
from torch import Tensor
import torch_scatter
import torch_sparse

class Identity(nn.Module):
    """Identity preserving function"""
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass


class DiscreteEncoder(nn.Module):
    """Discrete encoder for multiple features"""
    def __init__(self, hidden_channels, max_num_features=10, max_num_values=500):  # 10
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(max_num_values, hidden_channels)
                                         for i in range(max_num_features)])

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x):
        if x.dim() == 1: #unsqueeze data if necessary
            x = x.unsqueeze(1)
        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])
        return out

class SAN_MLP_readout(nn.Module):
    """SAN readout layer to generate the final output value"""

    def __init__(self, input_dim, output_dim, layers=2):
        super().__init__()
        self.end_layer = nn.Linear(input_dim // 2 ** layers, output_dim, bias=True)
        self.layers =nn.ModuleList( [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l+1), bias=True) for l in range(layers)])
        self.L = layers
        self.layers.append(self.end_layer)

    def forward(self, x):
        """Forward pass of the linear layers"""
        y = x
        for l in range(self.L):
            y = self.layers[l](y)
            y = F.relu(y, inplace=True)
        y = self.layers[self.L](y)
        return y

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


class SAN_Transformer_Layer(torch_geometric.nn.MessagePassing):
    """SAN Transformer layer for the primary architecture. Calculates the attention scores and uses the SAN algorithm designed
    by Kreuzer et al.

    args:
    in_channels: number of input channels
    out_channels: number of output channels
    heads: number of heads
    concat: unused parameter
    beta: unused parameter
    dropout: the dropout probability
    edge_dim: unused parameter
    root_weigt: unused parameter
    device: the computing device
    gamma: gamma parameter from SAN"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            beta: bool = False,
            dropout: float = 0.,
            edge_dim: int = None,
            bias: bool = True,
            root_weight: bool = True,
            device = "cuda",
            gamma = 1,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None
        self.layer_norm = False
        self.batch_norm = True
        self.residual = True
        self.gamma = gamma
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        #feed forward layers and O_h from the SAN algorithm
        self.O_h = torch.nn.Linear(self.heads * self.out_channels, self.heads * self.out_channels)
        self.FFlayer1 = torch.nn.Linear(self.heads * self.out_channels, self.heads * self.out_channels * 2)
        self.FFlayer2 = torch.nn.Linear(self.heads * self.out_channels * 2, self.heads * self.out_channels)

        self.layer_norm1_h = torch.nn.LayerNorm(self.heads * self.out_channels)
        self.batch_norm1_h = torch.nn.BatchNorm1d(self.heads * self.out_channels)
        self.layer_norm2_h = torch.nn.LayerNorm(self.heads * self.out_channels)
        self.batch_norm2_h = torch.nn.BatchNorm1d(self.heads * self.out_channels)
        #Multihead attention layer with SAN modifications
        self.attention = MultiHeadAttentionSAN(in_dim=in_channels[0], out_dim=out_channels, num_heads=self.heads, dropout=self.dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters of the SAN layer"""
        super().reset_parameters()

        self.O_h.reset_parameters()
        self.FFlayer1.reset_parameters()
        self.FFlayer2.reset_parameters()
        self.layer_norm1_h.reset_parameters()
        self.batch_norm1_h.reset_parameters()
        self.layer_norm2_h.reset_parameters()
        self.batch_norm2_h.reset_parameters()
    def forward(
            self,
            x,
            edge_index,
            edge_attr = None,
            return_attention_weights = None,
            batch_index = None
    ):
        r"""Runs the forward pass of the SAN transformer layer. Uses the attention function to generate the output
        """
        H, C = self.heads, self.out_channels

        #computing the output value and edge attributes
        out, edge_attr = self.attention(x, edge_index, edge_attr )


        self._alpha = None
        #reshaping the output
        out = out.view(-1, self.heads * self.out_channels)

        out = F.dropout(out, self.dropout, training=self.training)
        out = self.O_h(out)

        #computing the linear layers
        if self.layer_norm:
            out = self.layer_norm1_h(out)

        if self.batch_norm:
            out = self.batch_norm1_h(out)

        h_2 = out
        out = self.FFlayer1(out)
        out = F.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.FFlayer2(out)

        if self.residual:
            out = h_2 + out

        if self.layer_norm:
            out = self.layer_norm2_h(out)

        if self.batch_norm:
            out = self.batch_norm2_h(out)

        return out, edge_attr



class MultiHeadAttentionSAN(nn.Module):
    """The multihead attention module of the SAN architecture. Calculates the forward pass for the transformer architecture.
    args:

    in_dim: input dimension
    out_dim: output dimension
    num_heads: number of heads in the transformer
    dropout: dropout probability
    gamma: gamma parameter of SAN
    beta: beta parameter of SAN
    model: whether we use SAN, GraphiT or RWPE-SAN"""
    def __init__(self, in_dim, out_dim, num_heads, dropout, gamma=1, beta = 0.25, model="SAN"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim  = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.gamma = gamma
        self.beta = beta
        self.p = 16
        self.model = model
        if self.model == "SAN":
            #generating the required weight matrices
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.E = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.Qve = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.Kve = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.Eve = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            nn.init.xavier_normal_(self.Q.weight)
            nn.init.xavier_normal_(self.K.weight)
            nn.init.xavier_normal_(self.E.weight)
            nn.init.xavier_normal_(self.V.weight)
            nn.init.xavier_normal_(self.Qve.weight)
            nn.init.xavier_normal_(self.Kve.weight)
            nn.init.xavier_normal_(self.Eve.weight)
        elif self.model == "RWPE_SAN":
            # generating the required weight matrices
            self.Q_h = nn.Linear(in_dim * 2, out_dim * num_heads, bias=True)
            self.K_h = nn.Linear(in_dim * 2, out_dim * num_heads, bias=True)
            self.E_h = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.Qve_h = nn.Linear(in_dim * 2, out_dim * num_heads, bias=True)
            self.Kve_h = nn.Linear(in_dim * 2, out_dim * num_heads, bias=True)
            self.Eve_h = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V_h = nn.Linear(in_dim * 2, out_dim * num_heads, bias=True)
            self.Q_p = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K_p = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.E_p = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.Qve_p = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.Kve_p = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.Eve_p = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V_p = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        elif self.model == "RWPE_GraphiT":
            # generating the required weight matrices
            self.Q_h = nn.Linear(in_dim * 2, out_dim * num_heads, bias=True)
            self.K_h = nn.Linear(in_dim * 2, out_dim * num_heads, bias=True)
            self.E_h = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.Qve_h = nn.Linear(in_dim * 2, out_dim * num_heads, bias=True)
            self.Kve_h = nn.Linear(in_dim * 2, out_dim * num_heads, bias=True)
            self.Eve_h = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V_h = nn.Linear(in_dim * 2, out_dim * num_heads, bias=True)
            self.Q_p = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K_p = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.E_p = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.Qve_p = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.Kve_p = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.Eve_p = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V_p = nn.Linear(in_dim, out_dim * num_heads, bias=True)
    def propagate_attn(self, edge_index, edge_attr, Q_h, Q_hve, K_h, K_hve, E, E_ve, V_h):
        """The attention mechanism calculation including fake edges introduced by SAN"""
        #generating source, destination and key similar to the transformer encoder
        src = K_h[edge_index[0]]
        dest = Q_h[edge_index[1]]
        src_ve = K_hve[edge_index[0]]
        dest_ve = Q_hve[edge_index[1]]

        edge_attr_real = (edge_attr == 0)
        edge_attr_mask = torch.all(edge_attr_real, dim=1)

        src_total = torch.where(edge_attr_mask[:, None, None], src_ve, src) #generating edge attr mask for fake edges
        dest_total = torch.where(edge_attr_mask[:, None, None], dest_ve, dest)

        e_total = torch.where(edge_attr_mask[:, None, None], E_ve, E)

        score = (src_total * dest_total * e_total)/math.sqrt(self.out_dim) #calculate the intermediate output
        score = score.sum(dim=-1, keepdim=True)
        score = self.pyg_softmax(score, edge_index[1]) #applying softmax function
        if self.model == "SAN" or "RWPE_SAN": #use the SAN style of combining real and fake edges
            score = torch.where(edge_attr_mask[:, None, None],(self.gamma/(1+self.gamma)) * score , (1/(1 + self.gamma)) * score)

        if self.model == "RWPE_GraphiT": #using normal attention
            score = self.K[edge_index[0], edge_index[1]] * score
        score = F.dropout(score, self.dropout, training=self.training)

        msg = V_h[edge_index[0]] * score
        wV = torch.zeros_like(V_h)
        torch_scatter.scatter(msg, edge_index[1], dim=0, out=wV, reduce="add") #scatter the message using the edge indices
        return wV

    def forward(self, x, edge_index, edge_attr, **kwargs):
        """

        Forward pass of the multihead attention module.
        """
        if self.model == "SAN":
            #generate intermediate matrices
            Q_h = self.Q(x)
            K_h = self.K(x)
            V_h = self.V(x)
            Q_hve = self.Qve(x)
            K_hve = self.Kve(x)
            E = self.E(edge_attr)
            Eve = self.Eve(edge_attr)

            #reshape matrices
            Q_h1 = Q_h.view(-1, self.num_heads, self.out_dim)
            K_h1 = K_h.view(-1, self.num_heads, self.out_dim)
            V_h1 = V_h.view(-1, self.num_heads, self.out_dim)
            #generate fake edges weight matrices
            Q_hve1 = Q_hve.view(-1, self.num_heads, self.out_dim)
            K_hve1 =K_hve.view(-1, self.num_heads, self.out_dim)
            E_ve1 =Eve.view(-1, self.num_heads, self.out_dim)
            E1 = E.view(-1, self.num_heads, self.out_dim)
            #apply propagate attention with real and fake edges
            wV = self.propagate_attn(edge_index, edge_attr, Q_h1, Q_hve1, K_h1, K_hve1, E1, E_ve1, V_h1)
            h_out = wV
            e_out = edge_attr
            return h_out, e_out
        elif self.model == "RWPE_SAN" or self.model == "RWPE_GraphiT":
            p = kwargs.get("p", None)
            #generate matrices for RWPE SAN and RWPE GraphiT
            Q_h = self.Q_h(x).view(-1, self.num_heads, self.out_dim)
            K_h = self.K_h(x).view(-1, self.num_heads, self.out_dim)
            V_h = self.V_h(x).view(-1, self.num_heads, self.out_dim)
            Q_hve = self.Qve_h(x).view(-1, self.num_heads, self.out_dim)
            K_hve = self.Kve_h(x).view(-1, self.num_heads, self.out_dim)
            E = self.E_h(edge_attr).view(-1, self.num_heads, self.out_dim)
            Eve = self.Eve_h(edge_attr).view(-1, self.num_heads, self.out_dim)
            E_p = self.E_p(edge_attr).view(-1, self.num_heads, self.out_dim)
            Eve_p = self.Eve_p(edge_attr).view(-1, self.num_heads, self.out_dim)
            Q_p = self.Q_h(p).view(-1, self.num_heads, self.out_dim)
            K_p = self.K_h(p).view(-1, self.num_heads, self.out_dim)
            V_p = self.V_h(p).view(-1, self.num_heads, self.out_dim)
            Q_pve = self.Qve_h(p).view(-1, self.num_heads, self.out_dim)
            K_pve = self.Kve_h(p).view(-1, self.num_heads, self.out_dim)

            #calculation of K for GraphiT
            K_laplacian, _ = torch_geometric.utils.get_laplacian(edge_index)
            identity_mat = torch.eye(K_laplacian.size(0))
            self.K_mat = torch.pow(identity_mat - self.beta * K_laplacian, self.p)

            wh = self.propagate_attn(edge_index, edge_attr, Q_h, Q_hve, K_h, K_hve, E, Eve, V_h)
            wp = self.propagate_attn(edge_index, edge_attr, Q_p, Q_pve, K_p, K_pve, E_p, Eve_p, V_p)
            return wh, wp



    def pyg_softmax(self, src, index, num_nodes=None):
        """Computes a sparsely evaluated softmax function using src as input and index to
        sort the results to the respective nodes.
        """

        num_nodes = torch_geometric.utils.num_nodes.maybe_num_nodes(index, num_nodes)

        out = src - torch_scatter.scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
        out = out.exp()
        out = out / (
                torch_scatter.scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

        return out




