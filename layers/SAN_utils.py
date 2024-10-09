
import torch
import torch_geometric
import torch_sparse
import torch_scatter

def get_new_edge_feat(batch, e):
    """Generate new edge features for existing edge data,
        used from https://github.com/LiamMa/GRIT/tree/main"""
    device = "cuda"
    fill_value = 0
    edge_index = batch.edge_index
    edge_attr = e
    padding = torch.ones(1, edge_attr.size(1), dtype=torch.float) * fill_value

    edge_index, edge_attr = torch_geometric.utils.add_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes,
                                                                 fill_value=0.) #add self loops
    edge_index_full = get_full_edge_idx(edge_index, batch.batch).to(device) #generate full edge index
    edge_attr_pad = padding.repeat(edge_index_full.size(1), 1).to(device) #pad the new edge features
    edge_index_complete, edge_attr_complete = torch_sparse.coalesce(torch.cat([edge_index, edge_index_full], dim=1),
                                                                    torch.cat([edge_attr, edge_attr_pad], dim=0),
                                                                    batch.num_nodes, batch.num_nodes, op="add") #combine the edge features
    return edge_index_complete, edge_attr_complete


def get_full_edge_idx(edge_index, batch=None):
    """Generate the full edge index of a graph. Same function as for RRWP,
    used from https://github.com/LiamMa/GRIT/tree/main"""
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = torch_scatter.scatter(one, batch,
                                      dim=0, dim_size=batch_size, reduce='add') #calculate the number of nodes
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous() #generate new edge index
        # _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_full = torch.cat(negative_index_list, dim=1).contiguous() #concatenate edge indices
    return edge_index_full
