import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric


from layers.BasisNet_utils import IGN2to1
from layers.BasisNet_layers import GIN, create_mlp

activation_lst = {'relu': nn.ReLU()}


class SignPlus(nn.Module):
    # negate v, do not negate x
    """ Returns a function of x and -x added together
        """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, v, *args, x=None):
        if x == None:
            return self.model(v) + self.model(-v)
        else:
            return self.model(torch.cat((v, x), dim=-1)) + self.model(torch.cat((-v, x), dim=-1))


class IGNBasisInv(nn.Module):
    """ IGN based basis invariant neural network
    Code from:https://github.com/Graph-COM/SPE/blob/master/src/sign_inv_pe.py
    """
    def __init__(self, mult_lst, in_channels, hidden_channels=16, num_layers=2, device='cuda', **kwargs):
        super(IGNBasisInv, self).__init__()
        self.encs = nn.ModuleList()
        self.mult_to_idx = {}
        curr_idx = 0
        for mult in mult_lst:
            # get a phi for each choice of multiplicity
            self.encs.append(IGN2to1(1, hidden_channels, mult, num_layers=num_layers, device=device))
            self.mult_to_idx[mult] = curr_idx
            curr_idx += 1

    def forward(self, proj, mult):
        enc_idx = self.mult_to_idx[mult]
        x = self.encs[enc_idx](proj)
        return x


class IGNShared(nn.Module):
    """ IGN BasisNet with parameter sharing in phi
    Code from: https://github.com/cptq/SignNet-BasisNet/blob/main/LearningFilters/signbasisnet.py
    """

    def __init__(self, mult_lst, in_channels, hidden_channels=16, num_layers=2):
        super(IGNShared, self).__init__()
        self.enc = IGN2to1(1, hidden_channels, 1, num_layers=num_layers)
        self.fcs = nn.ModuleList()
        self.mult_to_idx = {}
        curr_idx = 0
        for mult in mult_lst:
            # get a fc for each choice of multiplicity
            self.fcs.append(nn.Linear(1, mult))
            self.mult_to_idx[mult] = curr_idx
            curr_idx += 1

    def forward(self, proj, mult):
        enc_idx = self.mult_to_idx[mult]
        x = self.enc(proj)
        x = x.transpose(2, 1)  # b x n x d
        x = self.fcs[enc_idx](x)
        x = x.transpose(2, 1)  # b x d x n
        return x


class BasisNetEncoder(nn.Module):
    """BasisNet encoder for a graphGPS architecture using IGNs and a GIN network.
    Code from: https://github.com/Graph-COM/SPE/blob/master/src/sign_inv_pe.py"""

    def __init__(self, cfg, dim_emb):
        super().__init__()


        #alternative mult_lst given by [1,2,3,4,5,6,7,8,9,10] for 32 eigenvectors from the SPE encoding
        #define phi and rho to be used in the BasisNet encoding
        self.phi = IGNBasisInv([1,2,3,4,5], in_channels=1, hidden_channels=cfg.BasisNet.IGN_hidden_channel) #multlist works for 8 eigenvectors used from the SPE encoding
        self.rho = GIN(cfg.BasisNet.n_layers,2*dim_emb, 2*dim_emb, dim_emb - cfg.BasisNet.node_out, create_mlp)
        self.embedding_h = nn.Linear(cfg.BasisNet.node_feat,cfg.BasisNet.node_out)
        self.embedding_e = nn.Linear(cfg.BasisNet.edge_feat,cfg.BasisNet.edge_out)
    def forward(self, data):
        """forward pass of the BasisNet encoder"""
        eig_feats_list = []
        data.edge_attr = self.embedding_e(data.edge_attr.unsqueeze(1).float())
        Lambda = data.Lambda
        P = data.P
        h= self.embedding_h(data.x.float())
        pe_dim = Lambda.size(-1)
        for i, same_size_projs in enumerate(P):
            N = same_size_projs[list(same_size_projs.keys())[0]].size(-1)

            phi_outs = [self.phi(projs, mult) for mult, projs in same_size_projs.items()] #apply phi to the projectors
            eig_feats = torch.cat([phi_out.reshape(N, -1) for phi_out in phi_outs], dim=-1)  # [N, min(N, pe_dim)]
            eig_feats = torch.cat([eig_feats, torch.zeros([N, pe_dim - torch.min(torch.tensor([N, pe_dim])).item()]).
                                  to(eig_feats.device)], dim=-1)  # [N, pe_dim]
            eig_feats = torch.cat((eig_feats, Lambda[i].unsqueeze(0).repeat(N, 1)), dim=-1)
            eig_feats_list.append(eig_feats)
        eig_feats = torch.cat(eig_feats_list, dim=0)
        data.x = torch.concat([h,self.rho(eig_feats, data.edge_index)], dim=1) #apply rho to the concatenated phis
        return data

def around(x, decimals=5):
     """ round to a number of decimal places """
     return torch.round(x * 10**decimals) / (10**decimals)

