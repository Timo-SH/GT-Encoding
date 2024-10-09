import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.transformer_layers import SAN_MLP_readout, SAN_Transformer_Layer
import torch_geometric
from layers.SAN_utils import get_new_edge_feat, get_full_edge_idx

class SAN(nn.Module):
    """Class for the SAN architecture as derived by Kreuzer et al. Computes the complete SAN model with the selected parameters

    args:
    edge_lpe (not used): whether we use edge features in SAN"""
    def __init__(self, config, edge_lpe=False):
        super().__init__()

        self.node_lpe = True

        if edge_lpe:
            self.edge_lpe = True
            self.node_lpe = False
        else:
            self.edge_lpe = False
            #SAN configuration parameters
        self.num_atom_type = config.model_SAN.num_atom_type
        self.num_bond_type = config.model_SAN.num_bond_type
        self.gamma = config.model_SAN.gamma
        self.device = config.device

        self.Transformer_layers = config.model_SAN.transformer_layers
        self.Transformer_in_dim_nodes = config.model_SAN.transformer_in_dim_nodes
        self.Transformer_in_dim_edges = config.model_SAN.transformer_in_dim_edges
        self.Transformer_out_dim = config.model_SAN.transformer_out_dim
        self.Transformer_hidden_dim = config.model_SAN.transformer_hidden_dim
        self.Transformer_heads = config.model_SAN.transformer_heads

        self.SAN_readout = config.model_SAN.readout
        self.SAN_residual = config.model_SAN.residual
        self.SAN_dropout = nn.Dropout(config.model_SAN.dropout)
        self.in_feat_dropout = nn.Dropout(config.model_SAN.dropout)
        self.SAN_layer_norm = config.model_SAN.layer_norm
        self.SAN_batch_norm = config.model_SAN.batch_norm

        self.SAN_lpe_dim = config.model_SAN.lpe_dim
        self.SAN_lpe_nheads = config.model_SAN.lpe_heads
        self.SAN_lpe_layers = config.model_SAN.lpe_layers
        #Embeddings for SAN
        self.embedding_h = nn.Embedding(self.num_atom_type, self.Transformer_hidden_dim - self.SAN_lpe_dim)
        self.embedding_e = nn.Embedding(self.num_bond_type, self.Transformer_hidden_dim)
        #Transformer encoder layer for the SAN encoding
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.SAN_lpe_dim, nhead=self.SAN_lpe_nheads)
        self.PE_Transformer = nn.TransformerEncoder(self.encoder_layer, self.SAN_lpe_layers)

        #SAN layers from the primary architecture
        self.layers = nn.ModuleList([SAN_Transformer_Layer(in_channels=self.Transformer_in_dim_nodes if i == 0 else self.Transformer_hidden_dim, out_channels=self.Transformer_hidden_dim // self.Transformer_heads, heads=self.Transformer_heads, concat=True, dropout=0, edge_dim=self.Transformer_in_dim_edges) for i in
                                     range(self.Transformer_layers - 1)])
        #MLP output layer
        self.MLP_layer = SAN_MLP_readout(self.Transformer_out_dim, config.model_SAN.output_dim)
        self.linear_A_node = nn.Linear(2, self.SAN_lpe_dim)
        self.linear_A_edge = nn.Linear(3, self.SAN_lpe_dim)
    def forward(self, data):
        """forward pass of the SAN architecture, computing the encoding and main architecture"""
        #retrieve eigenvalues and eigenvectors
        h = data.x.squeeze(1)
        h = self.embedding_h(h)
        EigVecs = data.san_node_eigvec
        EigVals = data.san_node_eigval
        #sign flipping in SAN
        sign_flip = torch.rand(EigVecs.size(1)).to(self.device)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        EigVecs = EigVecs * sign_flip.float()



        e = self.embedding_e(data.edge_attr)
        #SAN encoding computation
        if self.node_lpe:
            posenc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2).float()
            empty_mask = torch.isnan(posenc)


            posenc[empty_mask] = 0
            posenc = torch.transpose(posenc, 0, 1).float()

            posenc = self.linear_A_node(posenc)

            posenc = self.PE_Transformer(src=posenc, src_key_padding_mask=empty_mask[:,:,0])#Using transformer layer


            posenc[torch.transpose(empty_mask, 0, 1)[:, :, 0]] = float('nan')

            # Sum pooling
            posenc = torch.nansum(posenc, 0, keepdim=False)

            # Concatenate learned encoding to input embedding
            h = torch.cat((h, posenc), 1)


        h = self.in_feat_dropout(h)
        complete_edge_idx, complete_edge_attr = get_new_edge_feat(data, e)
        for conv in self.layers:
            h, e = conv(h, complete_edge_idx, complete_edge_attr)
        #readout method
        if self.SAN_readout == "sum":
            h = torch_geometric.nn.global_add_pool(h, batch=data.batch)
        else:
            h = torch_geometric.nn.global_mean_pool(h, batch=data.batch)

        return self.MLP_layer(h)


    def reset_parameters(self):
        """Reset the SAN parameters"""
        for layer in self.layers:
            layer.reset_parameters()
        #self.PE_Transformer.reset_parameters()
        self.MLP_layer.reset_parameters()
        self.linear_A_edge.reset_parameters()
        self.linear_A_node.reset_parameters()
        return
