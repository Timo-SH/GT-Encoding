
import torch
import torch.nn as nn
import torch_geometric

"""
Additional GIN layer for basisNet supporting create_mlp. Code used from: https://github.com/Graph-COM/SPE/blob/master/src/gin.py
"""

class GIN(nn.Module):
    layers: nn.ModuleList
    """
    Creates a GIN layer which accepts a MLP as underlying network. 
    """
    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp,
            bn: bool = False, residual: bool = False
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.bn = bn
        self.residual = residual
        if bn:
            self.batch_norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = GINLayer(create_mlp(in_dims, hidden_dims)) #create the GIN layer using the create_mlp functionality, calling the MLP function
            self.layers.append(layer)
            in_dims = hidden_dims
            if bn: #optional Batch normalization
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))

        layer = GINLayer(create_mlp(hidden_dims, out_dims))
        self.layers.append(layer)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the GIN network applying the GIN layers
        """
        for i, layer in enumerate(self.layers):
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


class GINLayer(torch_geometric.nn.MessagePassing):
    eps: nn.Parameter
    """
        Creates a GINLayer, providing the necessary functionality for the GIN network. 
        """

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


class MLP(nn.Module):
    """
        Creates a MLP using fully connected layers with dropout probability.
        """
    layers: nn.ModuleList
    fc: nn.Linear
    dropout: nn.Dropout

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, use_bn: bool, activation: str,
        dropout_prob: float, norm_type: str = "batch"
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = MLPLayer(in_dims, hidden_dims, use_bn, activation, dropout_prob, norm_type)
            self.layers.append(layer)
            in_dims = hidden_dims

        self.fc = nn.Linear(hidden_dims, out_dims, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [***, D_in]
        :return: Output feature matrix. [***, D_out]
        """
        for layer in self.layers:
            X = layer(X)      # [***, D_hid]
        X = self.fc(X)        # [***, D_out]
        X = self.dropout(X)   # [***, D_out]
        return X

    @property
    def out_dims(self) -> int:
        return self.fc.out_features


class MLPLayer(nn.Module):
    """
    Based on https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP
    """


    def __init__(self, in_dims: int, out_dims: int, use_bn: bool, activation: str,
                 dropout_prob: float, norm_type: str = "batch") -> None:
        super().__init__()
        # self.fc = nn.Linear(in_dims, out_dims, bias=not use_bn)
        self.fc = nn.Linear(in_dims, out_dims, bias=True)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_dims) if norm_type == "batch" else nn.LayerNorm(out_dims)
        else:
            self.bn = None
        # self.bn = nn.BatchNorm1d(out_dims) if use_bn else None
        # self.ln = nn.LayerNorm(out_dims) if use_bn else None

        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError("Invalid activation!")
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [***, D_in]
        :return: Output feature matrix. [***, D_out]
        """
        X = self.fc(X)                     # [***, D_out]
        if self.bn is not None:

            shape = X.size()
            X = X.reshape(-1, shape[-1])   # [prod(***), D_out]
            X = self.bn(X)                 # [prod(***), D_out]
            X = X.reshape(shape)           # [***, D_out]
        X = self.activation(X)             # [***, D_out]
        X = self.dropout(X)                # [***, D_out]
        return X


def create_mlp(in_dim, out_dim):
    """
        Returns an MLP to be passed on for the GIN layer.
        """
    return MLP(n_layers=3,in_dims=in_dim,hidden_dims=32,out_dims=out_dim,use_bn=True,activation="relu", dropout_prob=0)