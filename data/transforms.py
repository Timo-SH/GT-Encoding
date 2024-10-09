from typing import Any

import torch
import torch_geometric.transforms
from torch_sparse import SparseTensor
from torch_geometric.utils import get_laplacian
import torch.nn.functional as F
import torch_scatter
import numpy as np
from torch_geometric.data.datapipes import functional_transform
from functools import partial


@functional_transform('laplacian_pe')
class LaplacianPE(torch_geometric.transforms.BaseTransform):
    """Class to compute the SignNet node level encoding information. Given the graph, the encoding is computed by using
        the eigenvectors of the graph Laplacian.

        Args:
            max_freq (int): The number of eigenvalues and parts of the eigenvectors to be used (hyperparameter)
            attr_name(str): Name for the eigenvalue information collected in the Data object
            attr_name_eigvec(str): Name for the eigenvector information collected in the Data object
        """
    def __init__(self, max_freq, attr_name="laplacian_pe"):
        super().__init__()
        self.max_freq = max_freq
        self.attr_name = attr_name


    def add_node_attr(self,data,val,attr_name):
        """Adds additional keys and values to a Data object. Used for the addition of the eigenvectors"""
        data[attr_name] = val
        return data

    def forward(self, data):
        """Function to calculate the SignNet encoding information and pass it to the data object"""

        #Assert that the graph exists
        assert data.edge_index is not None
        assert data.num_nodes is not None
        num_nodes = data.num_nodes

        edge_index, edge_weight = get_laplacian(data.edge_index,data.edge_weight,normalization="sym", num_nodes=num_nodes)

        L = torch_geometric.utils.to_scipy_sparse_matrix(edge_index,edge_weight)
        #Compute graph Laplacian
        eig_fn =  np.linalg.eigh
        #Compute eigenvalues
        eig_vals, eig_vect = eig_fn(L.todense())
        #Compute eigenvectors
        eig_vect = np.real(eig_vect[:, eig_vals.argsort()])
        pe = torch.from_numpy(eig_vect[:, :self.max_freq ])
        #Pad eigenvectors to ensure same sized embeddings
        if num_nodes <= self.max_freq:
            pe = torch.nn.functional.pad(pe, (0, self.max_freq - num_nodes +1), value=0)
        #add data attributes to
        data = self.add_node_attr(data, pe, attr_name=self.attr_name)

        return data


@functional_transform("SAN_node_LPE")
class SAN_node_LPE(torch_geometric.transforms.BaseTransform):
    """Class to compute the SAN node level encoding information. Given the graph, the encoding is computed by using the eigenvalues
    and eigenvectors of the graph Laplacian.

    Args:
        max_freq (int): The number of eigenvalues and parts of the eigenvectors to be used (hyperparameter)
        attr_name(str): Name for the eigenvalue information collected in the Data object
        attr_name_eigvec(str): Name for the eigenvector information collected in the Data object
    """
    def __init__(self, max_freq, attr_name="san_node_eigval", attr_name_eigvec= "san_node_eigvec"):
        super().__init__()
        self.max_freq = max_freq
        self.attr_name = attr_name
        self.attr_name_eigvec = attr_name_eigvec
    def add_node_attr(self,data,val,attr_name):
        """Adds additional keys and values to a Data object. Used for the addition of eigenvalues and eigenvectors"""
        data[attr_name] = val
        return data

    def forward(self, data):
        """Function to calculate the SAN encoding and pass it to the data object"""

        #Asserting that the graph exists
        assert data.edge_index is not None
        assert data.num_nodes is not None
        num_nodes = data.num_nodes
        edge_index, edge_weight = get_laplacian(data.edge_index, data.edge_weight, normalization="sym",
                                                num_nodes=num_nodes)

        L = torch_geometric.utils.to_scipy_sparse_matrix(edge_index, edge_weight)

        eig_fn = np.linalg.eigh
        # Compute eigenvalues
        eig_vals, eig_vect = eig_fn(L.todense())

        #Gather eigenvalues and eigenvectors from the eigendecomposition up to the maximum amount
        eig_vals, eig_vect = eig_vals[: self.max_freq], eig_vect[:, :self.max_freq] #[:max_freq], [num_nodes, :max_freq]
        eig_vect = torch.from_numpy(eig_vect).float()
        #Normalization of Eigenvectors according to SAN paper
        eig_vect = F.normalize(eig_vect, p=2, dim=1, eps=1e-12, out=None)

        #Padding of eigenvalues to generate same size embeddings
        if num_nodes < self.max_freq:
            eig_vect = F.pad(eig_vect, (0, self.max_freq-num_nodes ), value=float("nan"))

        #Sort eigenvalues according to SAN paper
        eig_vals = torch.from_numpy(np.sort(np.abs(np.real(eig_vals))))

        # Padding of eigenvalues to generate same size embeddings
        if num_nodes < self.max_freq:
            eig_vals = F.pad(eig_vals, (0, self.max_freq - num_nodes ), value=float('nan')).unsqueeze(0) #[1, :max_freq]
        else:
            eig_vals = eig_vals.unsqueeze(0)
        #Repeat eigenvalues for each node
        eig_vals = eig_vals.repeat(num_nodes, 1).unsqueeze(2) #[num_nodes, :max_freq, 1]

        #Add eigenvalues and eigenvectors to the Data object
        data = self.add_node_attr(data, eig_vals, self.attr_name)
        data = self.add_node_attr(data, eig_vect, self.attr_name_eigvec)


        return data



@functional_transform("RRWP")
class RRWP_transform(torch_geometric.transforms.BaseTransform):
    """Class to compute the RRWP encoding, by computing the random walk matrix. The tensor consisting of the random walk matrices
    is passed onto a Data object.

        Args:
            walk_length (int): The number of random walk steps to be used in the encoding (hyperparameter)
        """
    def __init__(self, walk_length):
        super().__init__()
        self.walk_length = walk_length


    def forward(self, data):
        """Computes the necessary transform for the RRWP encoding"""
        #Generate the transformation for the RRWP encoding
        transform = partial(self.add_full_rrwp, walk_length=self.walk_length, attr_name_abs="rrwp", attr_name_rel="rrwp", add_identity=True, spd=False)
        data = transform(data)
        return data
    def add_node_attr(self,data, value: Any,
                      attr_name= None):
        """Function to pass the computed embedding to the Data object."""
        if attr_name is None:
            if 'x' in data:
                x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
                data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
            else:
                data.x = value
        else:
            data[attr_name] = value

        return data

    def add_full_rrwp(self,data,
                      walk_length=8,
                      attr_name_abs="rrwp",  # name: 'rrwp'
                      attr_name_rel="rrwp",  # name: ('rrwp_idx', 'rrwp_val')
                      add_identity=True,
                      spd=False,
                      **kwargs
                      ):
        """Computes the RRWP encoding. Provides both node and edge level encoding data.
        Taken from: https://github.com/LiamMa/GRIT in their implementation of the GRIT architecture"""
        device = data.edge_index.device
        ind_vec = torch.eye(walk_length, dtype=torch.float, device=device)
        num_nodes = data.num_nodes
        edge_index, edge_weight = data.edge_index, data.edge_weight
        #Compute adjacency matrix
        adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                           sparse_sizes=(num_nodes, num_nodes),
                                           )

        # Compute D^{-1} A:
        deg = adj.sum(dim=1)
        deg_inv = 1.0 / adj.sum(dim=1)
        deg_inv[deg_inv == float('inf')] = 0
        adj = adj * deg_inv.view(-1, 1)
        adj = adj.to_dense()

        pe_list = []
        i = 0
        #Add identity matrix as additional information for R^0
        if add_identity:
            pe_list.append(torch.eye(num_nodes, dtype=torch.float))
            i = i + 1

        out = adj
        pe_list.append(adj)
        #Compute the adjacency matrix to the power of up to walk_length
        if walk_length > 2:
            for j in range(i + 1, walk_length):
                out = out @ adj
                pe_list.append(out)
        #Stack random walk matrices into a tensor
        pe = torch.stack(pe_list, dim=-1)  # n x n x k
        #Diagonal of the random walk matrices
        abs_pe = pe.diagonal().transpose(0, 1)  # n x k
        #Compute edge level encoding information
        rel_pe = SparseTensor.from_dense(pe, has_value=True)
        rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
        rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)

        #Additional spd information
        if spd:
            spd_idx = walk_length - torch.arange(walk_length)
            val = (rel_pe_val > 0).type(torch.float) * spd_idx.unsqueeze(0)
            val = torch.argmax(val, dim=-1)
            rel_pe_val = F.one_hot(val, walk_length).type(torch.float)
            abs_pe = torch.zeros_like(abs_pe)
        #Add RRWP information to the data object
        data = self.add_node_attr(data, abs_pe, attr_name=attr_name_abs)
        data = self.add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
        data = self.add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
        data.log_deg = torch.log(deg + 1)
        data.deg = deg.type(torch.long)

        return data



@functional_transform("SPE")
class SPE_transform(torch_geometric.transforms.BaseTransform):
    """Class to compute the SPE encoding, computing the eigenvalues and eigenvectors used in the encoding. The information is
    passed to the data object used in the transform.

            Args:
                d (int): Number of eigenvalues and eigenvectors used in the SPE encoding (hyperparameter)
                attr_name(str): Name for the eigenvalue information collected in the Data object
                attr_name_eigvec(str): Name for the eigenvector information collected in the Data object
            """
    def __init__(self, d, attr_name="Lambda", attr_name_eigvec="V"):
        super().__init__()
        self.d = d
        self.attr_name = attr_name
        self.attr_name_eigvec = attr_name_eigvec

    def forward(self, data):
        """Computes the eigenvalues and eigenvectors for the SPE encoding. """
        num_nodes = data.num_nodes
        self.d_temp = min(self.d, num_nodes)
        L_edge_index, L_values = get_laplacian(data.edge_index, normalization="sym", num_nodes=num_nodes)
        #Compute the graph Laplacian
        L = torch_geometric.utils.to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=num_nodes).squeeze(dim=0)
        #Generate matrices Lambda and V for eigenvalues and eigenvectors
        Lambda = torch.zeros(1, self.d_temp)
        V = torch.zeros(num_nodes, self.d_temp)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        #restrict eigenvalues and eigenvectors to d
        Lambda[0, :self.d_temp] = eigenvalues[0:self.d_temp]
        V[:, :self.d_temp] = eigenvectors[:, 0:self.d_temp]

        if num_nodes < self.d_temp:
            V = F.pad(V, (0, self.d - num_nodes), value=0)
            Lambda = F.pad(Lambda, (0, self.d - num_nodes), value=0)

        #Add embedding to the data object
        data = self.add_node_attr(data, Lambda, self.attr_name)
        data = self.add_node_attr(data, V, self.attr_name_eigvec)
        return data


    def add_node_attr(self, data, val, attr_name):
        """Function to pass the computed embedding to the Data object."""
        data[attr_name] = val
        return data

class BasisNet_transform(torch_geometric.transforms.BaseTransform):
    """Class to compute the BasisNet encoding, computing the projectors VV^T and multiplicities used in the encoding. The information is
        passed to the data object used in the transform.

                Args:
                    d (int): Number of eigenvalues and eigenvectors used in the initial SPE encoding (hyperparameter)
                """
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.SPE_transform = SPE_transform(d=self.d)

    def get_proj(self, instance: torch_geometric.data.Data):
        """Function to pass the computed projectors VV^T and multiplicities to the Data object."""
        instance = self.SPE_transform(instance)
        projs, mults = self.get_projections(eigvals=instance.Lambda, eigvecs=instance.V) #compute the projection matrices and multiplicities
        instance.update({"P": projs, "mults": mults}) #update the Data object
        return instance


    def get_projections(self, eigvals, eigvecs):
        """Function to compute the projectors VV^T and unique multiplicities.
        Adapted from: https://github.com/Graph-COM/SPE/blob/master/src/utils.py"""
        N = eigvecs.size(0)
        pe_dim = np.min([N, eigvals.size(-1)])
        # eigvals, eigvecs = eigvals[:, :N], eigvecs[:, :N]
        rounded_vals = self.around(eigvals, decimals=5)
        # get rid of the padding zeros
        rounded_vals = rounded_vals[0, :pe_dim]
        uniq_vals, inv_inds, counts = rounded_vals.unique(return_inverse=True, return_counts=True)
        uniq_mults = counts.unique()

        sections = torch.cumsum(counts, 0)
        eigenspaces = torch.tensor_split(eigvecs, sections.cpu(), dim=1)[:-1] #Compute eigenspaces
        #Compute the projectors from the eigenvectors
        projectors = [V @ V.T for V in eigenspaces]
        projectors = [P.reshape(1,1,N,N) for P in projectors]
        #Compute same size projectors and unique multiplicities
        same_size_projs = {mult.item(): [] for mult in uniq_mults}
        for i in range(len(projectors)):
            mult = counts[i].item()
            same_size_projs[mult].append(projectors[i])
        for mult, projs in same_size_projs.items():
            same_size_projs[mult] = torch.cat(projs, dim=0)


        return same_size_projs, uniq_mults


    def around(self,x, decimals=5):
        """ round to a number of decimal places"""
        return torch.round(x * 10 ** decimals) / (10 ** decimals)