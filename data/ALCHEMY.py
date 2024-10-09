from torch_geometric.datasets import TUDataset
from data.transforms import LaplacianPE, SAN_node_LPE, SPE_transform, RRWP_transform, BasisNet_transform
import torch_geometric

def load_dataset(root, transform):
    """Loads the ALCHEMY dataset from the specified file and applies the transformation"""
    infile = open("data/alchemy/train_al_10.index", "r")
    for line in infile:
        indices_train = line.split(",")
        indices_train = [int(i) for i in indices_train]

    infile = open("data/alchemy/val_al_10.index", "r")
    for line in infile:
        indices_val = line.split(",")
        indices_val = [int(i) for i in indices_val]

    infile = open("data/alchemy/test_al_10.index", "r")
    for line in infile:
        indices_test = line.split(",")
        indices_test = [int(i) for i in indices_test]

    indices = indices_train
    indices.extend(indices_val)
    indices.extend(indices_test)

    # Use the provided TUDataset loader to apply the transform
    return TUDataset(f"{root}/alchemy", name="alchemy_full", transform=transform)[indices]


class Alchemy_loader():
    """Class to load the ALCHEMY dataset in the main files. Applies load_dataset with the transform selected
    from the configuration file"""
    def __init__(self, config):
        self.config = config
        pass

    def load_data(self, model):

        if model == "SignNet" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "SignNet_PE"):
            transform = LaplacianPE(max_freq=self.config.GT_SignNetPE.max_freq)

        elif model == "SAN_node" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "SAN_PE"):
            transform = SAN_node_LPE(max_freq=self.config.GT_encLAP.max_freq)

        elif model == "BasisNet" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "BasisNet_PE"): \
                # using the SPE transform to derive the necessary Eigenvectors for the BasisNet encoding
            transform = BasisNet_transform(d=self.config.BasisNet.n_phis)

        elif model == "RWPE" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "RWPE_PE"):
            # configuration dependent on the choice of architecture
            if model == "GT":
                transform  = torch_geometric.transforms.AddRandomWalkPE(
                    walk_length=self.config.GT_RWPE.steps)
            else:
                transform =  torch_geometric.transforms.AddRandomWalkPE(
                    walk_length=self.config.model_RWPE_SAN.pos_enc_dim)
        elif model == "GRIT" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "RRWP_PE"):
            transform = RRWP_transform(walk_length=self.config.GT_RWPE.steps)

        elif model == "SPE" or (model == "GT" and self.config.GT_dataset.node_encoder_name == "SPE_PE"):
            transform = SPE_transform(d=self.config.GT_SPE.n_psis)
        else:
            raise ValueError
        dataset = load_dataset("data", transform).copy()

        mean = dataset.y.mean(dim=0, keepdim=True)
        std = dataset.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.y - mean) / std
        mean, std = mean.to(device="cuda"), std.to(device="cuda")
        train_dataset = dataset[:10000]
        val_dataset = dataset[10000:11000]
        test_dataset = dataset[11000:]
        return train_dataset, val_dataset, test_dataset, mean, std
