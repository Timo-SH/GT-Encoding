import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from layers.transformer_layers import Identity

class MaskedBN(nn.Module):
    """Masked Batch Normalization"""
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
    def reset_parameters(self):
        self.bn.reset_parameters()
    def forward(self, x, mask=None):
        if mask is None:
            return self.bn(x.transpose(1,2)).transpose(1,2) #transposing to apply mask to last element
        x[mask] = self.bn(x[mask]) #apply mask to batch norm
        return x

class MaskedLN(nn.Module):
    """Masked Layer Normalization"""
    def __init__(self, num_features):
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=1e-6)
    def reset_parameters(self):
        self.ln.reset_parameters()
    def forward(self, x, mask=None):
        if mask is None:
            return self.ln(x)
        x[mask] = self.ln(x[mask]) #apply mask to layer normn
        return x

class MaskedMLP(nn.Module):
    """Masked MLP with additional masked batch norms"""
    def __init__(self, nin, nout, nhid=None, nlayer=2, final_activation=True, with_norm=True):
        super().__init__()
        n_hid = nin if nhid is None else nhid #hidden channel
        self.layers = nn.ModuleList([nn.Linear(nin if i==0 else n_hid,
                                               n_hid if i<nlayer-1 else nout,
                                               bias=True)
                                     for i in range(nlayer)]) #concatenate linear layers
        self.norms = nn.ModuleList([MaskedBN(n_hid if i<nlayer-1 else nout) if with_norm else Identity()
                                     for i in range(nlayer)]) #concatenate batch norms
        self.nlayer = nlayer
        self.final_activation = final_activation #activation in the final layer

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x, mask=None):
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if mask is not None:
                x[~mask] = 0
            if i < self.nlayer-1 or self.final_activation:
                x = norm(x, mask)
                x = F.relu(x)
        return x

