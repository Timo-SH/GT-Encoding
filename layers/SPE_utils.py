
import torch
import torch.nn as nn
import torch_geometric

"""
DeepSets and GIN layer for the use in the SPE encoder, provided by: https://github.com/Graph-COM/SPE/blob/master/src/deepsets.py
"""
class DeepSets(nn.Module):
    """
            DeepSets layer to be used with the SPE encoding. Provides an element wise DeepSet to be used in the computation
            of psi
            """
    layers: nn.ModuleList

    def __init__(self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, activation: str) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = DeepSetsLayer(in_dims, hidden_dims, activation) #execution of the deepsets layer
            self.layers.append(layer)
            in_dims = hidden_dims

        # layer = DeepSetsLayer(hidden_dims, out_dims, activation='id') # drop last activation
        layer = DeepSetsLayer(hidden_dims, out_dims, activation)
        self.layers.append(layer)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [B, N, D_in]
        :return: Output feature matrix. [B, N, D_out]
        """
        for layer in self.layers:
            X = layer(X)   # [B, N, D_hid] or [B, N, D_out]
        return X           # [B, N, D_out]

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


class DeepSetsLayer(nn.Module):
    """
        A single DeepSets layer, used in the SPE encoder for psi
                """
    fc_one: nn.Linear
    fc_all: nn.Linear
    activation: nn.Module

    def __init__(self, in_dims: int, out_dims: int, activation: str) -> None:
        super().__init__()
        self.fc_curr = nn.Linear(in_dims, out_dims, bias=True)
        self.fc_all = nn.Linear(in_dims, out_dims, bias=False)

        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'id':
            self.activation = nn.Identity()
        else:
            raise ValueError("Invalid activation!")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [B, N, D_in]
        :return: Output feature matrix. [B, N, D_out]
        """
        Z_curr = self.fc_curr(X)                          # [B, N, D_out]
        Z_all = self.fc_all(X.sum(dim=1, keepdim=True))   # [B, 1, D_out]
        # Z_all = self.fc_all(X.max(dim=1, keepdim=True)[0])   # [B, 1, D_out]
        X = Z_curr + Z_all
        #print(X)# [B, N, D_out]
        return self.activation(X)                         # [B, N, D_out]

    def reset_parameters(self):
        self.fc_curr.reset_parameters()
        self.fc_all.reset_parameters()

class GIN(nn.Module):
    """GIN network for the SPE encoder using torch geometric MLPs for the generation of the GIN layer"""
    layers: nn.ModuleList

    def __init__(
        self, n_layers, in_dims, hidden_dims, out_dims,
            bn= False, residual = False, config = None
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.bn = bn
        self.residual = residual
        if bn:
            self.batch_norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = GINLayer(torch_geometric.nn.MLP(in_channels=in_dims, hidden_channels=config.GT_SPE.MLP_hidden_dim, out_channels=hidden_dims, num_layers=config.GT_SPE.MLP_n_layer, norm=None))
            self.layers.append(layer)
            in_dims = hidden_dims
            if bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))

        layer = GINLayer(torch_geometric.nn.MLP(in_channels=hidden_dims, hidden_channels=config.GT_SPE.MLP_hidden_dim, out_channels=out_dims, num_layers=config.GT_SPE.MLP_n_layer, norm=None))
        self.layers.append(layer)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        for i, layer in enumerate(self.layers):
            #print("test")
            X0 = X
            X = layer(X, edge_index)   # [N_sum, ***, D_hid] or [N_sum, ***, D_out]
            # batch normalization
            if self.bn and i < len(self.layers) - 1:
                if X.ndim == 3:
                    X = self.batch_norms[i](X.transpose(2, 1)).transpose(2, 1)
                else:
                    X = self.batch_norms[i](X)
            if self.residual:
                X = X + X0
        return X                       # [N_sum, ***, D_out]

    @property
    def out_dims(self) -> int:
        return self.layers[-1].out_dims

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for norm in self.batch_norms:
            norm.reset_parameters()

        return

class GINLayer(torch_geometric.nn.MessagePassing):
    """GIN layer for the SPE encoder"""


    def __init__(self, mlp) -> None:
        # Use node_dim=0 because message() output has shape [E_sum, ***, D_in] - https://stackoverflow.com/a/68931962
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)

        self.eps = torch.nn.Parameter(data=torch.randn(1), requires_grad=True)
        self.mlp = mlp

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        # Contains sum(j in N(i)) {message(j -> i)} for each node i.
        S = self.propagate(edge_index, X=X)   # [N_sum, *** D_in]

        Z = (1 + self.eps) * X   # [N_sum, ***, D_in]
        Z = Z + S                # [N_sum, ***, D_in]
        return self.mlp(Z)       # [N_sum, ***, D_out]

    def message(self, X_j: torch.Tensor) -> torch.Tensor:
        """
        :param X_j: Features of the edge sources. [E_sum, ***, D_in]
        :return: The messages X_j for each edge (j -> i). [E_sum, ***, D_in]
        """
        return X_j   # [E_sum, ***, D_in]

    @property
    def out_dims(self) -> int:
        return self.mlp.out_dims

    def reset_parameters(self) -> None:
        self.mlp.reset_parameters()
