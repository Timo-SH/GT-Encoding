import torch
import torch_geometric
import torch.nn as nn
import numpy as np
import torch_scatter
import torch_geometric.graphgym.register as register
#from torch_geometric.graphgym.register import register_act
from layers.GT_layer_utils import GatedGCNLayer

class GT_layer(torch.nn.Module):
    """
    Creates the GT layer used in the GraphGPS algorithm, adapted from
    https://github.com/rampasek/GraphGPS

    args:
    dim_h: hidden dimension of
    gnn_type: type of GNN to be used in the GT layer
    global_model: type of transformer used
    num_heads: number of heads in the transformer
    act: activation function used in linear layers
    pna_degrees: degree information for the PNA algorithm
    equivstable_pe: whether equivstable positional encodings are used: https://openreview.net/pdf?id=e95i1IHcWj
    dropout: dropout probability
    attn_dropout: attention dropout probability
    layer_norm: Using layer norm
    batch_norm: Using batch norm
    """
    def __init__(self, dim_h, gnn_type, global_model, num_heads, act="relu", pna_degrees=None, equivstable_pe=False, dropout=0, attn_dropout=0, layer_norm =False, batch_norm=True, log_attn_weights = False):
        super().__init__()
        #register_act('gelu', nn.GELU)
        self.dim_h = dim_h
        self.global_model = global_model
        self.log_attn_weights = log_attn_weights
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.act = nn.ReLU #Set for our experiments
        self.local_gnn_with_edge_attr = True #whether edge features are used

        #Create the GNN model from the configuration file
        if gnn_type == 'None':
            self.local_model = None
        elif gnn_type == "GCN":
            self.local_gnn_with_edge_attr = False
            self.local_model = torch_geometric.nn.GCNConv(dim_h, dim_h)
        elif gnn_type == 'GIN':
            self.local_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(torch_geometric.nn.Linear(dim_h, dim_h),
                                   self.act(),
                                   torch_geometric.nn.Linear(dim_h, dim_h))
            self.local_model = torch_geometric.nn.GINConv(gin_nn)

        elif gnn_type == 'GENConv':
            self.local_model = torch_geometric.nn.GENConv(dim_h, dim_h)
        elif gnn_type == 'GINE':
            gin_nn = nn.Sequential(torch_geometric.nn.Linear(dim_h, dim_h),
                                   self.act(),
                                   torch_geometric.nn.Linear(dim_h, dim_h))
            self.local_model = torch_geometric.nn.GINEConv(gin_nn)

        elif gnn_type == 'GAT':
            self.local_model = torch_geometric.nn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = torch_geometric.nn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=min(128, dim_h),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)


        elif gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             act=act,
                                             equivstable_pe=equivstable_pe)
        #Using self attention
        if global_model == None:
            self.self_attn = None
        #Building the transformer model
        elif global_model in ['Transformer', 'BiasedTransformer']:
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)


        #Layer norm and batch norm
        if self.layer_norm:
            self.norm1_local = torch_geometric.nn.norm.LayerNorm(dim_h)
            self.norm1_attn = torch_geometric.nn.norm.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
        #Linear layers
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)

        if self.layer_norm:
            self.norm2 = torch_geometric.nn.norm.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)


    def forward(self, batch):
        """
                Computes the forward pass of a GraphGPS layer
         """
        h = batch.x
        h_in1 = h
        h_out = []

        #MPNN and first norm
        if self.local_model is not None:
            self.local_model: torch_geometric.nn.conv.MessagePassing
        if self.local_gnn_with_edge_attr:
            h_local = self.local_model(h,batch.edge_index,batch.edge_attr)
        else:
            h_local = self.local_model(h, batch.edge_index)
        h_local = self.dropout_local(h_local)
        h_local = h_in1 + h_local  # Residual connection

        if self.layer_norm:
            h_local = self.norm1_local(h_local, batch.batch)
        if self.batch_norm:
            h_local = self.norm1_local(h_local)
        h_out.append(h_local)

        #Multihead Attention of the transformer model
        if self.self_attn is not None:
            h_dense, mask = torch_geometric.utils.to_dense_batch(h, batch.batch)
            if self.global_model == 'Transformer': #use self attention block
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model == 'BiasedTransformer':
                # Use Graphormer-like conditioning, requires `batch.attn_bias`.
                h_attn = self._sa_block(h_dense, batch.attn_bias, ~mask)[mask]

            #application of residual connection and norms
            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out.append(h_attn)
        #Compute the sum of the output
        h = sum(h_out)

        # Feed Forward layers
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    # MLP layer and output
    def _ff_block(self, x):
        """
             Feedforward linear layers with dropout connection
        """
        act = nn.GELU()
        x = act(self.ff_linear1(x))
        x = self.ff_dropout1(x)
        return self.ff_dropout2(self.ff_linear2(x))

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x


    def reset_parameters(self):
        """
            Resets the model parameters
        """
        self.local_model.reset_parameters()
        self.self_attn._reset_parameters()
        self.ff_linear1.reset_parameters()
        self.ff_linear2.reset_parameters()
        self.norm1_local.reset_parameters()
        self.norm1_attn.reset_parameters()
        self.norm2.reset_parameters()
        return

class SAN_prediction_head(torch.nn.Module):
    """
        Prediction head for the GraphGPS architecture using linear layers
        args:

        dim_in: input dimension
        dim_out: output dimension, has to be equal to the number of targets
        L: Number of linear layers
    """
    def __init__(self,config, dim_in, dim_out, L):
        super().__init__()

        if config.GT_gnn.agg == "sum":
            self.pooling = torch_geometric.nn.global_add_pool #global add pooling
        else:
            self.pooling = torch_geometric.nn.global_mean_pool #global mean pooling
        self.FC_layers = [nn.Linear(dim_in // 2 **l, dim_in //2 ** (l+1), bias=True) for l in range(L)] #Generation of linear layers
        self.FC_layers.append(nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.layers = nn.ModuleList(self.FC_layers)
        self.L = L
        self.act = nn.ReLU()

    def forward(self, data):
        """
                Computes the forward pass of the SAN prediction head
            """
        graph_emb = self.pooling(data.x, data.batch)
        for l in range(self.L): #Compute the forward pass for each layer
            graph_emb = self.layers[l](graph_emb)
            graph_emb = self.act(graph_emb)
        graph_emb = self.layers[self.L](graph_emb)
        return graph_emb #result of the GraphGPS architecture

    def reset_parameters(self):
        """
                Resets parameters for the prediction head
            """
        for layer in self.layers:
            layer.reset_parameters()
        return
