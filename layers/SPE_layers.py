import torch
import torch_geometric
import numpy as np

import layers.SPE_utils

    
class GINPhi(torch.nn.Module):
    
    def __init__(self,config, n_layers, in_dim, hidden_dim, out_dim, bn):
        super().__init__()
        self.gin = layers.SPE_utils.GIN(n_layers=n_layers, in_dims=in_dim, hidden_dims=hidden_dim, out_dims=out_dim, bn=bn, config=config)
        #self.mlp = torch_geometric.nn.MLP(in_channels=1, hidden_channels=64, out_channels=out_dim, num_layers=3, batch_norm=False)
    def forward(self, W_list, edge_index):
        n_max = max(W.size(0) for W in W_list)
        W_pad_list = []
        mask = []
        #if single is False:
        #    for W in W_list:

         #       zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
         #       W_pad = torch.cat([W, zeros], dim=1)  # [N_i, N_max, M]
         #       W_pad_list.append(W_pad)
         #       mask.append((torch.arange(n_max, device=W.device) < W.size(0)).tile((W.size(0), 1)))  # [N_i, N_max]
        for W in W_list:
            zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
            W_pad = torch.cat([W, zeros], dim=1)  # [N_i, N_max, M]
            W_pad_list.append(W_pad)
            mask.append(
                (torch.arange(n_max, device=W.device) < W.size(0)).tile((W.size(0), 1)))  # [N_i, N_max]

        W = torch.cat(W_pad_list, dim=0)  # [N_sum, N_max, M]
        mask = torch.cat(mask, dim=0)  # [N_sum, N_max]
        PE = self.gin(W, edge_index)  # [N_sum, N_max, D_pe]
        #PE = self.mlp(W)
        #if single is False:
        PE = (PE * mask.unsqueeze(-1)).sum(dim=1)
            #PE = (PE * mask.unsqueeze(-1))  # .sum(dim=1)
        #else:
            #PE = (PE * mask.unsqueeze(-1))  # .sum(dim=1)
        return PE  # [N_sum, D_pe]


    def reset_parameters(self):
        self.gin.reset_parameters()
        #self.mlp.reset_parameters()
        return