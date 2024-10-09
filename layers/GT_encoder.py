import torch
import torch_geometric
import torch.nn as nn
import numpy as np
import torch_scatter
import torch_geometric.graphgym.register as register
from layers.SignNet_utils import GINDeepSigns, MaskedGINDeepSigns
from layers.SPE_layers import GINPhi
from layers.SPE_utils import DeepSets
import torch_sparse

@register.register_node_encoder("SAN_PE")
class LaplacianNodeEncoder(torch.nn.Module):
    """
        Computes the Eigenvalue and Eigenvector based SAN encoding to be used in the graphGPS layer.
        The configuration of the model is given by the configuration file.
    """
    def __init__(self,config, dim_emb):
        super().__init__()
        dim_in = config.GT_encLAP.dim_in
        dim_pe = config.GT_encLAP.dim_pe
        #Number of transformer layers and heads
        n_layers = config.GT_encLAP.layers
        n_heads = config.GT_encLAP.heads
        #Number of post MLP layers
        post_n_layers = config.GT_encLAP.post_layers
        #Maximum amount of eigenvalues and eigenvectors used
        max_freq = config.GT_encLAP.max_freq
        norm_type = config.GT_encLAP.norm_type
        self.embedding = config.GT_encLAP.embedding

        #Computing the initial embedding

        if self.embedding == "linear":
            self.embedding_h = nn.Linear(config.GT_gt.num_linear_emb_node, dim_emb - dim_pe)
        else:
            self.embedding_h = nn.Embedding(config.dataset_node_emb, dim_emb - dim_pe)
        if config.GT_gt.with_edges:
            if self.embedding == "linear":
                self.embedding_e = nn.Linear(config.GT_gt.num_linear_emb_edge, config.GT_gnn.dim_inner)
            else:
                self.embedding_e = nn.Embedding(config.dataset_edge_emb, config.GT_gnn.dim_inner)

        else:
            self.embedding_e = None
        #Initial linear layer
        self.linear_A = nn.Linear(2, dim_pe)

        if norm_type == "bn":
            self.raw_norm = nn.BatchNorm1d(max_freq)
        else:
            self.raw_norm = None

        activation = nn.ReLU

        #Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_pe, nhead=n_heads, batch_first=True)
        self.pe_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        #Create post transformer layer
        self.post_mlp = None
        if post_n_layers > 0:
            layers = []
            if post_n_layers == 1:
                layers.append(nn.Linear(dim_pe, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(dim_pe, 2 * dim_pe))
                layers.append(activation())
                for _ in range(post_n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.post_mlp = nn.Sequential(*layers)


    def forward(self, data):
        """
            Computes the SAN encoding to be used in the graphGPS architecture
        """
        #Gather eigenvalues and eigenvectors
        EigVals = data.san_node_eigval
        EigVecs = data.san_node_eigvec

        #Initial embedding calculation
        if self.embedding == "linear":
            h = self.embedding_h(data.x.float())
        else:
            h = self.embedding_h(data.x.long())

        if self.embedding_e is not None:
            if self.embedding == "linear":
                data.edge_attr = self.embedding_e(data.edge_attr.unsqueeze(1).float())
            else:
                data.edge_attr = self.embedding_e(data.edge_attr.unsqueeze(1).long())


        #Enable sign flipping for training
        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)


        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2).float()
        empty_mask  = torch.isnan(pos_enc)
        pos_enc[empty_mask] = 0
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.linear_A(pos_enc)
        #Transformer computation using the empty mask defined earlier
        pos_enc = self.pe_encoder(src=pos_enc, src_key_padding_mask= empty_mask[:,:,0])
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:,:,0].unsqueeze(2), 0)
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)

        #Compute post mlp layer
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)
        if self.embedding != "linear":
            h = h.squeeze(1)
        #Concatenate the node feature with the computed encoding
        data.x = torch.cat((h, pos_enc), 1)
        data.batch_SAN = pos_enc
        return data

    def reset_parameters(self):
        """
        Reset the embedding parameters
        """
        self.embedding_h.reset_parameters()
        if self.embedding_e is not None:
            self.embedding_e.reset_parameters()

        return
@register.register_node_encoder("SignNet_PE")
class SignNetEncoder(torch.nn.Module):
    """
    Computes the SignNet encoding to be used in the graphGPS layer from eigenvalues and eigenvectors of a graph.
    """
    def __init__(self,config, dim_emb):
        super().__init__()
        dim_in = config.GT_SignNetPE.dim_in
        dim_pe = config.GT_SignNetPE.dim_pe


        self.model_type = config.GT_SignNetPE.model_type
        self.embedding = config.GT_SignNetPE.embedding
        #Number of layers for phi and rho
        phi_layers = config.GT_SignNetPE.phi_layers
        rho_layers = config.GT_SignNetPE.rho_layers

        #Maximum number of eigenvalues/eigenvectors
        max_freq = config.GT_SignNetPE.max_freq

        #Initial node and edge embedding
        if self.embedding == "linear":
            self.embedding_h = nn.Linear(config.GT_gt.num_linear_emb_node, dim_emb - dim_pe)
        else:
            self.embedding_h = nn.Embedding(config.dataset_node_emb, dim_emb - dim_pe)
        if config.GT_gt.with_edges:
            if self.embedding == "linear":
                self.embedding_e = nn.Linear(config.GT_gt.num_linear_emb_edge, config.GT_gnn.dim_inner)
            else:
                self.embedding_e = nn.Embedding(config.dataset_edge_emb, config.GT_gnn.dim_inner)

        else:
            self.embedding_e = None

        #Selection of model for phi and rho
        if self.model_type == "MLP":
            self.net = GINDeepSigns(in_channels=1, hidden_channels=config.GT_SignNetPE.hidden_channel, out_channels=config.GT_SignNetPE.out_channel, num_layers=phi_layers, k=max_freq, dim_pe=dim_pe, rho_num_layers=rho_layers, use_bn=True, dropout=0, activation="relu")

        elif self.model_type == "DeepSet":
            self.net = MaskedGINDeepSigns(in_channels=1, hidden_channels=config.GT_SignNetPE.hidden_channel,
                                    out_channels=config.GT_SignNetPE.out_channel, num_layers=phi_layers,
                                    dim_pe=dim_pe, rho_num_layers=rho_layers, use_bn=True, dropout=0, activation="relu")

    def forward(self, data):
        """
        Computes the SignNet encoding to be used in the graphGPS architecture
        """

        #Get eigenvectors from data object
        EigVec = data.laplacian_pe
        #Compute initial embedding
        if self.embedding == "linear":
            h = self.embedding_h(data.x.float())
        else:
            h = self.embedding_h(data.x.long())

        if self.embedding_e is not None:
            if self.embedding == "linear":
                data.edge_attr = self.embedding_e(data.edge_attr.unsqueeze(1).float())
            else:
                data.edge_attr = self.embedding_e(data.edge_attr.unsqueeze(1).long())
        pos_enc = EigVec.unsqueeze(-1)
        #Compute mask
        empty_mask = torch.isnan(pos_enc)
        pos_enc[empty_mask] = 0
        #Apply phi and rho (given in a single architecture)
        pos_enc = self.net(pos_enc, data.edge_index, data.batch)
        #Concatenate node features and encoding
        data.x = torch.cat((h, pos_enc), 1)
        data.SignNet_pe = pos_enc

        return data

    def reset_parameters(self):
        """
        Reset the embedding parameters
        """
        self.embedding_h.reset_parameters()
        if self.embedding_e is not None:
            self.embedding_e.reset_parameters()
        self.net.reset_parameters()

        return

@register.register_node_encoder("SPE_PE")
class SPEEncoder(torch.nn.Module):
    """
    Computes the SPE encoding following https://github.com/Graph-COM/SPE/tree/master using elementwise MLPs and a GIN.
    """

    def __init__(self, config, dim_emb):
        super().__init__()

        self.embedding = config.GT_SPE.embedding
        #Psi configured as elementwise MLP
        self.psi_list = torch.nn.ModuleList([DeepSets(n_layers=config.GT_SPE.psi_layers, in_dims=1, hidden_dims=config.GT_SPE.psi_hidden_dim, out_dims=1, activation="relu") for _ in range(config.GT_SPE.n_psis)] )
        #GIN layer for phi
        self.phi = GINPhi(n_layers=config.GT_SPE.phi_n_layers, in_dim=config.GT_SPE.n_psis, hidden_dim=config.GT_SPE.phi_hidden_dim, out_dim=config.GT_SPE.phi_out_dim, bn=config.GT_SPE.phi_batch_norm, config=config )
        # Initial node embedding
        if self.embedding == "linear":
            self.embedding_h = nn.Linear(config.GT_gt.num_linear_emb_node, dim_emb - config.GT_SPE.phi_out_dim)
        else:
            self.embedding_h = nn.Embedding(config.dataset_node_emb, dim_emb - config.GT_SPE.phi_out_dim)
        if config.GT_gt.with_edges:
            if self.embedding == "linear":
                self.embedding_e = nn.Linear(config.GT_gt.num_linear_emb_edge, config.GT_gnn.dim_inner)
            else:
                self.embedding_e = nn.Embedding(config.dataset_edge_emb, config.GT_gnn.dim_inner)

        else:
            self.embedding_e = None
    def forward(self, data):
        """
        Computes the SPE encoding and concatenates it with the node features of the data object
        """
        #Compute initial embeddings
        if self.embedding == "linear":
            h = self.embedding_h(data.x.float())
        else:
            h = self.embedding_h(data.x.long())

        if self.embedding_e is not None:
            if self.embedding == "linear":
                data.edge_attr = self.embedding_e(data.edge_attr.unsqueeze(1).float())
            else:
                data.edge_attr = self.embedding_e(data.edge_attr.unsqueeze(1).long())
        #Gather eigenvalues and eigenvectors
        EigVals = data.Lambda.unsqueeze(dim=2)

        EigVecMat = data.V
        #Compute \phi{EigVals} for all Psi
        Z = torch.stack([psi(EigVals).squeeze(dim=2) for psi in self.psi_list], dim=2)
        #Create list of Eigenvector matrices
        V_list = torch_geometric.utils.unbatch(EigVecMat, data.batch, dim=0)
        Z_list = list(Z)
        W_list = []
        for V, Z in zip(V_list, Z_list):

            V = V.unsqueeze(dim=0)
            Z = Z.permute(0, 1)
            Z = Z.diag_embed() #Compute diagonal of \phi{EigVals}
            V_t = V.mT
            W = V.matmul(Z).matmul(V_t) #Compute V diag(\phi{EigVals}) V^T
            W = W.permute(1, 2, 0) #Permute to fix dimensions
            W_list.append(W) #Concatenate results of computation


        pos_enc = self.phi(W_list, data.edge_index) #Apply phi to the
        data.x = torch.cat((h, pos_enc), 1) #Compute concatenated node features and encoding
        data.SPE_pe = pos_enc

        return data

    def reset_parameters(self):
        """
        Resets parameter of the SPE encoding
        """
        self.embedding_h.reset_parameters()
        if self.embedding_e is not None:
            self.embedding_e.reset_parameters()
        for psi in self.psi_list:
            psi.reset_parameters()
        self.phi.reset_parameters()


        return


@register.register_node_encoder("RWPE_PE")
class RWPEEncoder(torch.nn.Module):
    """
        Computes the RWPE encoding using the encoding provided by the pytorch geometric transform. Following
        the description of RWPE by Dwivedi et al. the encoding is provided using a MLP architecture.
    """
    def __init__(self, config, dim_emb, expand_x=True):
        super().__init__()
        dim_pe = config.GT_RWPE.dim_pe
        self.embedding = config.GT_RWPE.embedding
        rw_steps = config.GT_RWPE.steps #Number of random walk steps
        model_type = config.GT_RWPE.model_type
        n_layers = config.GT_RWPE.layers #Number of MLP layers
        norm_type = config.GT_RWPE.norm
        #Initial embeddings
        if self.embedding == "linear":
            self.embedding_h = nn.Linear(config.GT_gt.num_linear_emb_node, dim_emb - dim_pe)
        else:
            self.embedding_h = nn.Embedding(config.dataset_node_emb, dim_emb - dim_pe)
        if config.GT_gt.with_edges:
            if self.embedding == "linear":
                self.embedding_e = nn.Linear(config.GT_gt.num_linear_emb_edge, config.GT_gnn.dim_inner)
            else:
                self.embedding_e = nn.Embedding(config.dataset_edge_emb, config.GT_gnn.dim_inner)

        else:
            self.embedding_e = None


        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(rw_steps)
        else:
            self.raw_norm = None
        activation = nn.ReLU
        #MLP layer for RWPE
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(rw_steps, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(rw_steps, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        #Alternative linear layer for RWPE
        elif model_type == 'linear':
            self.layers = nn.ModuleList([nn.Linear(rw_steps, dim_pe)])




    def forward(self, data):
        """
            Computes the RWPE encoding using the random walk features for each node and edge.
        """
        #Initial encoding computation
        if self.embedding == "linear":
            h = self.embedding_h(data.x.float())
        else:
            h = self.embedding_h(data.x.long())

        if self.embedding_e is not None:
            if self.embedding == "linear":
                data.edge_attr = self.embedding_e(data.edge_attr.unsqueeze(1).float())
            else:
                data.edge_attr = self.embedding_e(data.edge_attr.unsqueeze(1).long())
        #Gather random walk encoding for all nodes
        pos_enc = data.random_walk_pe
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        #Compute the MLP application of the encoding
        pos_enc = self.pe_encoder(pos_enc)
        #Concatenate the node features and encoding values
        data.x = torch.cat((h, pos_enc), 1)
        return data

    def reset_parameters(self):
        """
            resets embedding parameters
        """
        self.embedding_h.reset_parameters()
        if self.embedding_e is not None:
            self.embedding_e.reset_parameters()
        if self.raw_norm:
            self.raw_norm.reset_parameters()

        return

@register.register_node_encoder("RRWP_PE")
class RRWPEncoder(torch.nn.Module):
    """
    Computes the RRWP encoding using the random walk matrices. The encoding is adapted from
    https://github.com/LiamMa/GRIT/tree/main to be used in graphGPS layers
    """
    def __init__(self, config, dim_emb):
        super().__init__()
        dim_pe = config.GT_RWPE.dim_pe
        rw_steps = config.GT_RWPE.steps #Number of random walk steps
        model_type = config.GT_RWPE.model_type
        n_layers = config.GT_RWPE.layers #Number of MLP layers
        norm_type = config.GT_RWPE.norm
        self.embedding = config.GT_RWPE.embedding
        #Initial embeddings
        if self.embedding == "linear":
            self.embedding_h = nn.Linear(config.GT_gt.num_linear_emb_node, dim_emb - dim_pe)
        else:
            self.embedding_h = nn.Embedding(config.dataset_node_emb, dim_emb - dim_pe)
        if config.GT_gt.with_edges:
            if self.embedding == "linear":
                self.embedding_e = nn.Linear(config.GT_gt.num_linear_emb_edge, config.GT_gnn.dim_inner)
            else:
                self.embedding_e = nn.Embedding(config.dataset_edge_emb, config.GT_gnn.dim_inner)

        else:
            self.embedding_e = None

        #Norm for linear layers
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(rw_steps)
        else:
            self.raw_norm = None
        activation = nn.ReLU
        #Linear layers for node and edge features
        self.fc_node = nn.Linear(rw_steps, dim_pe, bias=True)
        self.fc_edge = nn.Linear(rw_steps, config.GT_gnn.dim_inner, bias=True)
        self.overwrite_old_attr = False
        self.pad_to_full_graph = False
        self.batchnorm = True
        self.layernorm = False
        #Batchnorm definition
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(config.GT_gnn.dim_inner)

        if self.layernorm:
            self.ln = nn.LayerNorm(config.GT_gnn.dim_inner)

        #Padding for non existing edges
        self.fill_value = 0.

        padding = torch.ones(1, dim_pe, dtype=torch.float) * self.fill_value
        self.register_buffer("padding", padding)

        #Definition of MLP layers for node encoding
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(dim_pe, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(dim_pe, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
            #Definition of MLP layers for edge encoding
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(config.GT_gnn.dim_inner, config.GT_gnn.dim_inner))
                layers.append(activation())
            else:
                layers.append(nn.Linear(config.GT_gnn.dim_inner, 2 * config.GT_gnn.dim_inner))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * config.GT_gnn.dim_inner, 2 * config.GT_gnn.dim_inner))
                    layers.append(activation())
                layers.append(nn.Linear(2 * config.GT_gnn.dim_inner, config.GT_gnn.dim_inner))
                layers.append(activation())
            self.ee_encoder = nn.Sequential(*layers)

    def forward(self, data):
        """
            Computes the RRWP encoding with node and edge features
        """
        #Gather information from data object
        rrwp_idx = data.rrwp_index
        rrwp_val = data.rrwp_val
        edge_index = data.edge_index
        edge_attr = self.embedding_e(data.edge_attr.unsqueeze(1).float())
        rrwp_val = self.fc_edge(rrwp_val)
        rrwp_val = self.ee_encoder(rrwp_val)
        #Node encoding
        pos_enc = self.fc_node(data.rrwp)
        pos_enc = self.pe_encoder(pos_enc)

        h = self.embedding_h(data.x.float())
        data.x = torch.cat((h, pos_enc), 1)
        #Edge encoding

        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), rrwp_val.size(1))
            # zero padding for non-existing edges

        if self.overwrite_old_attr: #replace old attributes with rrwp values
            out_idx, out_val = rrwp_idx, rrwp_val
        else:
            # edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)
            edge_index, edge_attr = torch_geometric.utils.add_self_loops(edge_index, edge_attr, num_nodes=data.num_nodes, fill_value=0.)
            out_idx, out_val = torch_sparse.coalesce(
                torch.cat([edge_index, rrwp_idx], dim=1),
                torch.cat([edge_attr.squeeze(1), rrwp_val], dim=0),
                data.num_nodes, data.num_nodes,
                op="add"
            )


        if self.pad_to_full_graph: #Padding to a full graph
            edge_index_full = self.full_edge_index(out_idx, batch=data.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = torch_sparse.coalesce(
               out_idx, out_val, data.num_nodes, data.num_nodes,
               op="add"
            )
        #Apply batch and layer norm
        if self.batchnorm:
            out_val = self.bn(out_val)

        if self.layernorm:
            out_val = self.ln(out_val)

        #replace edge_index and edge_attr of the data object
        data.edge_index, data.edge_attr = out_idx, out_val
        return data

    def reset_parameters(self):
        """
            Resets the embedding parameters
        """
        self.embedding_h.reset_parameters()
        if self.embedding_e is not None:
            self.embedding_e.reset_parameters()
    def full_edge_index(self,edge_index, batch=None):
        """
            Generates the edge index for a full graph from an existing edge index with batch information
            Taken from: https://github.com/LiamMa/GRIT/tree/main
        """
        if batch is None:
            batch = edge_index.new_zeros(edge_index.max().item() + 1)

        batch_size = batch.max().item() + 1
        one = batch.new_ones(batch.size(0))
        num_nodes = torch_scatter.scatter(one, batch,
                            dim=0, dim_size=batch_size, reduce='add') #Compute number of nodes
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

        negative_index_list = []
        for i in range(batch_size):
            n = num_nodes[i].item()
            size = [n, n]
            adj = torch.ones(size, dtype=torch.short,
                             device=edge_index.device) #Generate adjacency matrix

            adj = adj.view(size)
            _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            # _edge_index, _ = remove_self_loops(_edge_index)
            negative_index_list.append(_edge_index + cum_nodes[i])

        edge_index_full = torch.cat(negative_index_list, dim=1).contiguous() #compute complete edge index
        return edge_index_full